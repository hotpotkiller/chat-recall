"""Unit tests for passage-level evaluation benchmark and constraints."""

import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from chat_recall.providers import ProviderError
from scripts.evaluate_passages import (
    FixtureValidationError,
    evaluate,
    main,
    score_results,
    validate_hit_integrity,
    validate_search_response_schema,
    validate_suite,
)
from scripts.calibrate_passages import (
    CacheCapacityExceededError,
    CachingProviderWrapper,
    CalibrationError,
    InvalidVectorError,
    calibrate,
    main as calibrate_main,
)


class PassageEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "passage-evaluation.json"

    def test_fixture_declarations_and_constraints(self):
        """Verify fixture declarations, conversations, queries, splits, multilingual count, and bounds."""
        self.assertTrue(self.fixture_path.exists(), "fixtures/passage-evaluation.json must exist")
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))

        # Pre-validation passes on canonical fixture
        validate_suite(suite)

        self.assertTrue(suite.get("synthetic"))
        self.assertEqual(suite.get("version"), "passage-evaluation-v3")

        conversations = suite["conversations"]
        self.assertEqual(len(conversations), 12, "Fixture must contain exactly 12 conversations")

        total_messages = sum(len(c["messages"]) for c in conversations)
        self.assertGreaterEqual(total_messages, 48)
        self.assertLessEqual(total_messages, 60)

        queries = suite["queries"]
        self.assertLessEqual(len(queries), 100)
        positive_queries = [q for q in queries if q["type"] == "positive"]
        unrelated_queries = [q for q in queries if q["type"] == "unrelated"]
        self.assertEqual(len(positive_queries), 48)
        self.assertEqual(len(unrelated_queries), 16)

        # Chinese / multilingual positives >= 12
        chinese_positives = [
            q for q in positive_queries
            if any("\u4e00" <= ch <= "\u9fff" for ch in q["query"])
        ]
        self.assertGreaterEqual(len(chinese_positives), 12)

        # Split counts
        calib_pos = [q for q in positive_queries if q.get("split") == "calibration"]
        holdout_pos = [q for q in positive_queries if q.get("split") == "holdout"]
        self.assertEqual(len(calib_pos), 32)
        self.assertEqual(len(holdout_pos), 16)

        calib_unrel = [q for q in unrelated_queries if q.get("split") == "calibration"]
        holdout_unrel = [q for q in unrelated_queries if q.get("split") == "holdout"]
        self.assertEqual(len(calib_unrel), 8)
        self.assertEqual(len(holdout_unrel), 8)

        # Disjoint conversation IDs across splits
        calib_convs = {
            g["conversation_id"]
            for q in calib_pos
            for g in q["expected_passages"]
        }
        holdout_convs = {
            g["conversation_id"]
            for q in holdout_pos
            for g in q["expected_passages"]
        }
        self.assertEqual(len(calib_convs), 8)
        self.assertEqual(len(holdout_convs), 4)
        self.assertTrue(calib_convs.isdisjoint(holdout_convs))

    def test_fixture_role_balance_user_and_assistant(self):
        """Verify role balance: fixture must contain exactly 8 user and 40 assistant gold targets."""
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        positive_queries = [q for q in suite["queries"] if q["type"] == "positive"]
        self.assertEqual(len(positive_queries), 48)

        msg_roles = {}
        for conv in suite["conversations"]:
            src = conv.get("source_id", "chatgpt")
            acc = conv.get("account_id", "default")
            cid = conv["id"]
            for m in conv["messages"]:
                msg_roles[(src, acc, cid, m["id"])] = m["role"]

        user_golds = 0
        assistant_golds = 0
        for q in positive_queries:
            for g in q["expected_passages"]:
                src = g.get("source_id", "chatgpt")
                acc = g.get("account_id", "default")
                cid = g["conversation_id"]
                mid = g["message_id"]
                role = msg_roles.get((src, acc, cid, mid))
                if role == "user":
                    user_golds += 1
                elif role == "assistant":
                    assistant_golds += 1

        self.assertEqual(user_golds, 8, "Must contain exactly 8 user gold targets")
        self.assertEqual(assistant_golds, 40, "Must contain exactly 40 assistant gold targets")

    def test_scoring_right_chat_wrong_message_miss(self):
        """Right conversation but wrong message must yield passage_rank=None and message_rank=None."""
        golds = [{
            "source_id": "chatgpt",
            "account_id": "default",
            "conversation_id": "conv-1",
            "message_id": "msg-target",
            "start_offset": 10,
            "end_offset": 50,
        }]
        results = [{
            "source_id": "chatgpt",
            "account_id": "default",
            "conversation_id": "conv-1",
            "message_id": "msg-other",
            "focus_offset": 20,
        }]
        ranks = score_results(results, golds)
        self.assertIsNone(ranks["passage_rank"])
        self.assertIsNone(ranks["message_rank"])
        self.assertEqual(ranks["conversation_rank"], 1)

    def test_scoring_right_message_wrong_focus_miss(self):
        """Right conversation and message but focus_offset outside gold span must miss passage_rank."""
        golds = [{
            "source_id": "chatgpt",
            "account_id": "default",
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "start_offset": 100,
            "end_offset": 200,
        }]
        results = [{
            "source_id": "chatgpt",
            "account_id": "default",
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "focus_offset": 50,  # Outside [100, 200)
        }]
        ranks = score_results(results, golds)
        self.assertIsNone(ranks["passage_rank"])
        self.assertEqual(ranks["message_rank"], 1)
        self.assertEqual(ranks["conversation_rank"], 1)

    def test_scoring_missing_focus_offset_misses_passage(self):
        """Missing focus_offset must not default to 0 and must yield passage_rank=None."""
        golds = [{
            "source_id": "chatgpt",
            "account_id": "default",
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "start_offset": 0,
            "end_offset": 50,
        }]
        results = [{
            "source_id": "chatgpt",
            "account_id": "default",
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            # focus_offset is absent
        }]
        ranks = score_results(results, golds)
        self.assertIsNone(ranks["passage_rank"])
        self.assertEqual(ranks["message_rank"], 1)
        self.assertEqual(ranks["conversation_rank"], 1)

    def test_scoring_bool_or_float_focus_offset_rejected(self):
        """Bool and float focus_offsets must be strictly rejected (type(focus_offset) is int)."""
        golds = [{
            "source_id": "chatgpt",
            "account_id": "default",
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "start_offset": 0,
            "end_offset": 50,
        }]
        for invalid_focus in (True, False, 10.5, "10", None):
            results = [{
                "source_id": "chatgpt",
                "account_id": "default",
                "conversation_id": "conv-1",
                "message_id": "msg-1",
                "focus_offset": invalid_focus,
            }]
            ranks = score_results(results, golds)
            self.assertIsNone(ranks["passage_rank"], f"Should reject invalid focus_offset={invalid_focus!r}")
            self.assertEqual(ranks["message_rank"], 1)

    def test_scoring_exclusive_end_boundary(self):
        """Exclusive end boundary: start <= focus < end. focus == end must miss; focus == start must hit."""
        golds = [{
            "source_id": "chatgpt",
            "account_id": "default",
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "start_offset": 10,
            "end_offset": 50,
        }]
        # Hit at start_offset (inclusive)
        res_start = [{
            "source_id": "chatgpt", "account_id": "default",
            "conversation_id": "conv-1", "message_id": "msg-1",
            "focus_offset": 10,
        }]
        self.assertEqual(score_results(res_start, golds)["passage_rank"], 1)

        # Hit inside span
        res_mid = [{
            "source_id": "chatgpt", "account_id": "default",
            "conversation_id": "conv-1", "message_id": "msg-1",
            "focus_offset": 49,
        }]
        self.assertEqual(score_results(res_mid, golds)["passage_rank"], 1)

        # Miss at end_offset (exclusive boundary)
        res_end = [{
            "source_id": "chatgpt", "account_id": "default",
            "conversation_id": "conv-1", "message_id": "msg-1",
            "focus_offset": 50,
        }]
        self.assertIsNone(score_results(res_end, golds)["passage_rank"])

    def test_scoring_gold_rank4_yields_r3_miss_and_mrr_25(self):
        """Gold passage appearing at rank 4 yields R@3 miss and MRR contribution 0.25."""
        golds = [{
            "source_id": "chatgpt",
            "account_id": "default",
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "start_offset": 10,
            "end_offset": 50,
        }]
        results = [
            {"source_id": "chatgpt", "account_id": "default", "conversation_id": "conv-x", "message_id": "m-x", "focus_offset": 0},
            {"source_id": "chatgpt", "account_id": "default", "conversation_id": "conv-y", "message_id": "m-y", "focus_offset": 0},
            {"source_id": "chatgpt", "account_id": "default", "conversation_id": "conv-z", "message_id": "m-z", "focus_offset": 0},
            {"source_id": "chatgpt", "account_id": "default", "conversation_id": "conv-1", "message_id": "msg-1", "focus_offset": 30},
        ]
        ranks = score_results(results, golds)
        self.assertEqual(ranks["passage_rank"], 4)
        # Rank 4 is a miss for Recall@3 (rank <= 3)
        self.assertFalse(ranks["passage_rank"] <= 3)
        # MRR contribution at rank 4 is 1/4 = 0.25
        self.assertEqual(1.0 / ranks["passage_rank"], 0.25)

    def test_scoring_deep_astral_codepoint_focus_pass(self):
        """Focus offset in a deep message after astral Unicode codepoints is correctly scored in codepoints."""
        text_with_astral = "🚀🌟𝒢𝑒𝓂𝒾𝓃𝒾 𝒬𝓊𝒶𝓃𝓉𝓊𝓂 🔬" + ("A" * 500) + "TARGET_SPAN"
        target_start = text_with_astral.index("TARGET_SPAN")
        target_end = target_start + len("TARGET_SPAN")

        golds = [{
            "source_id": "chatgpt",
            "account_id": "default",
            "conversation_id": "conv-astral",
            "message_id": "msg-deep",
            "start_offset": target_start,
            "end_offset": target_end,
        }]
        results = [{
            "source_id": "chatgpt",
            "account_id": "default",
            "conversation_id": "conv-astral",
            "message_id": "msg-deep",
            "focus_offset": target_start + 2,
        }]
        ranks = score_results(results, golds)
        self.assertEqual(ranks["passage_rank"], 1)

    def test_scoring_wrong_account_or_source_same_ids_miss(self):
        """Cross-account collision with matching conversation_id and message_id must not score as hit."""
        golds = [{
            "source_id": "chatgpt",
            "account_id": "account-A",
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "start_offset": 0,
            "end_offset": 50,
        }]
        results = [{
            "source_id": "chatgpt",
            "account_id": "account-B",  # Different account!
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "focus_offset": 20,
        }]
        ranks = score_results(results, golds)
        self.assertIsNone(ranks["passage_rank"])
        self.assertIsNone(ranks["message_rank"])
        self.assertIsNone(ranks["conversation_rank"])

    def test_validation_rejects_wrong_sibling_ancestry(self):
        """Pre-validation must reject gold expected_ancestor_ids that belong to an unchosen sibling branch."""
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        # In conv-calib-6: c6-m2-bike and c6-m2-bus are siblings under c6-m1.
        # c6-m4-bus descends from c6-m3-bus -> c6-m2-bus -> c6-m1.
        # Falsely putting c6-m2-bike in expected_ancestor_ids must fail validation.
        mutated = copy.deepcopy(suite)
        target_q = next(q for q in mutated["queries"] if q.get("id") == "q-calib-pos-23")
        target_q["expected_passages"][0]["expected_ancestor_ids"].append("c6-m2-bike")

        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated)
        self.assertIn("not a true ancestor", str(cm.exception))

    def test_validation_rejects_overlapping_split_conversations(self):
        """Pre-validation must reject a suite if any conversation appears in both calibration and holdout."""
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(suite)
        # Duplicate a valid calibration query/gold, give it split="holdout" and a unique qid
        calib_q = next(q for q in mutated["queries"] if q.get("split") == "calibration" and q.get("type") == "positive")
        dup_q = copy.deepcopy(calib_q)
        dup_q["id"] = "q-test-overlapping-split"
        dup_q["split"] = "holdout"
        mutated["queries"].append(dup_q)

        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated)
        self.assertIn("Disjoint split violation", str(cm.exception))

    def test_validation_rejects_malformed_gold_offsets(self):
        """Pre-validation must reject gold spans that exceed message bounds or have invalid start/end."""
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(suite)
        mutated["queries"][0]["expected_passages"][0]["end_offset"] = 999999

        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated)
        self.assertIn("exceeds message length", str(cm.exception))

    def test_validation_checks_expected_text_matches_original_slice(self):
        """Pre-validation must verify expected_text exactly matches msg_text[start:end]."""
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(suite)
        mutated["queries"][0]["expected_passages"][0]["expected_text"] = "Mismatched text not in slice."

        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated)
        self.assertIn("does not match slice", str(cm.exception))

    def test_validation_rejects_unknown_or_malformed_query_filters(self):
        """Pre-validation must reject unknown filter keys (e.g. source_id) and malformed values."""
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))

        # Unknown filter key (source_id is not an engine filter)
        mutated1 = copy.deepcopy(suite)
        mutated1["queries"][0]["filters"] = {"source_id": "chatgpt"}
        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated1)
        self.assertIn("unknown or disallowed filter key", str(cm.exception))

        # Malformed role
        mutated2 = copy.deepcopy(suite)
        mutated2["queries"][0]["filters"] = {"role": "moderator"}
        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated2)
        self.assertIn("must be 'user' or 'assistant'", str(cm.exception))

        # Malformed account_id
        mutated3 = copy.deepcopy(suite)
        mutated3["queries"][0]["filters"] = {"account_id": "   "}
        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated3)
        self.assertIn("must be a non-empty string", str(cm.exception))

    def test_validation_rejects_gold_query_filter_mismatch(self):
        """Pre-validation must reject gold passage whose account_id or role contradicts query filters."""
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))

        # Account mismatch
        mutated1 = copy.deepcopy(suite)
        target1 = next(q for q in mutated1["queries"] if q["type"] == "positive")
        target1["filters"] = {"account_id": "nonexistent-account"}
        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated1)
        self.assertIn("does not match query filter account_id", str(cm.exception))

        # Role mismatch
        mutated2 = copy.deepcopy(suite)
        target2 = next(q for q in mutated2["queries"] if q["type"] == "positive")
        # Target gold message is an assistant message; filtering for user must fail validation
        target2["filters"] = {"role": "user"}
        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated2)
        self.assertIn("does not match query filter role", str(cm.exception))

    def test_denominator_integrity_on_failed_or_dropped_query(self):
        """Failed or dropped query cannot shrink denominator: denominator remains expected query count."""
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        expected_pos = len([q for q in suite["queries"] if q["type"] == "positive"])

        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None

        from chat_recall.search import SearchIndex
        original_search = SearchIndex.search

        # Simulate throwing on the very first positive query
        first_q = suite["queries"][0]["query"]

        def stub_search(self_idx, query, *s_args, **s_kwargs):
            if query == first_q:
                raise RuntimeError("Simulated query failure")
            return original_search(self_idx, query, *s_args, **s_kwargs)

        with patch.object(SearchIndex, "search", stub_search):
            report = evaluate(args)

            self.assertEqual(len(report["query_errors"]), 1)
            # Denominator must remain full expected count (48)
            self.assertEqual(report["metrics"]["positive_queries"], expected_pos)
            self.assertEqual(report["metrics"]["completed_queries_total"], len(suite["queries"]) - 1)
            # Structure passed must be False
            self.assertFalse(report["gates"]["structure_passed"])
            self.assertFalse(report["gates"]["quality_passed"])

    def test_lexical_structural_baseline_passes_structure_and_fails_quality(self):
        """Offline lexical baseline completes with structure_passed=True and quality_passed=False."""
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None

        report = evaluate(args)

        self.assertTrue(report["synthetic"])
        self.assertEqual(report["mode"], "lexical-baseline-only")
        self.assertFalse(report["semantic_evaluated"])
        self.assertIsNone(report["model_id"])
        self.assertIsNone(report["enrichment"])

        self.assertEqual(report["metrics"]["semantic_used_count"], 0)
        self.assertEqual(report["metrics"]["context_identity_errors"], 0)
        self.assertEqual(len(report["query_errors"]), 0)
        self.assertEqual(len(report["semantic_failures"]), 0)

        self.assertTrue(report["gates"]["structure_passed"])
        self.assertFalse(report["gates"]["quality_passed"])
        self.assertFalse(report["gates"]["holdout_passed"])

    def test_context_injected_wrong_namespace_fails_gates(self):
        """Injected wrong namespace in context must increment context_identity_errors and fail structure gate."""
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None

        from chat_recall.search import SearchIndex
        original_context = SearchIndex.context

        def stub_context(self_idx, conversation_id, message_id, source_id="chatgpt", account_id="default"):
            res = original_context(self_idx, conversation_id, message_id, source_id, account_id)
            res["account_id"] = "wrong-injected-account"
            return res

        with patch.object(SearchIndex, "context", stub_context):
            report = evaluate(args)
            self.assertGreater(report["metrics"]["context_identity_errors"], 0)
            self.assertFalse(report["gates"]["structure_passed"])
            self.assertFalse(report["gates"]["quality_passed"])

    def test_context_injected_modified_text_fails_gates(self):
        """Injected modified text in context messages must fail structural and quality gates."""
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None

        from chat_recall.search import SearchIndex
        original_context = SearchIndex.context

        def stub_context(self_idx, conversation_id, message_id, source_id="chatgpt", account_id="default"):
            res = original_context(self_idx, conversation_id, message_id, source_id, account_id)
            if res.get("messages"):
                res["messages"][0]["text"] = "Injected modified text not matching original fixture."
            return res

        with patch.object(SearchIndex, "context", stub_context):
            report = evaluate(args)
            self.assertGreater(report["metrics"]["context_identity_errors"], 0)
            self.assertFalse(report["gates"]["structure_passed"])

    def test_context_injected_wrong_parent_fails_gates(self):
        """Injected wrong parent_id in context messages must fail structural and quality gates."""
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None

        from chat_recall.search import SearchIndex
        original_context = SearchIndex.context

        def stub_context(self_idx, conversation_id, message_id, source_id="chatgpt", account_id="default"):
            res = original_context(self_idx, conversation_id, message_id, source_id, account_id)
            if len(res.get("messages", [])) > 1:
                res["messages"][1]["parent_id"] = "wrong-parent-id-999"
            return res

        with patch.object(SearchIndex, "context", stub_context):
            report = evaluate(args)
            self.assertGreater(report["metrics"]["context_identity_errors"], 0)
            self.assertFalse(report["gates"]["structure_passed"])

    def test_context_injected_sibling_prefix_fails_gates(self):
        """Injected unchosen sibling node in prefix before target must fail structural and quality gates."""
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None

        from chat_recall.search import SearchIndex
        original_context = SearchIndex.context

        def stub_context(self_idx, conversation_id, message_id, source_id="chatgpt", account_id="default"):
            res = original_context(self_idx, conversation_id, message_id, source_id, account_id)
            if res.get("messages"):
                sibling = {
                    "id": "injected-sibling-node",
                    "role": "assistant",
                    "text": "Sibling branch text.",
                    "parent_id": None,
                }
                res["messages"].insert(0, sibling)
            return res

        with patch.object(SearchIndex, "context", stub_context):
            report = evaluate(args)
            self.assertGreater(report["metrics"]["context_identity_errors"], 0)
            self.assertFalse(report["gates"]["structure_passed"])

    def test_synthetic_account_collision_evaluator(self):
        """Real synthetic account-collision evaluator test: two accounts sharing conversation and message IDs."""
        msg_alpha_2 = "Alpha response regarding five gallon balcony tomato potting containers and daily watering schedule."
        msg_beta_2 = "Beta response regarding wide puncture resistant bicycle tires for rainy street commuting."

        temp_suite = {
            "synthetic": True,
            "version": "passage-evaluation-v3",
            "conversations": [
                {
                    "source_id": "chatgpt",
                    "account_id": "account-alpha",
                    "id": "conv-shared",
                    "title": "Shared ID Alpha Conversation",
                    "messages": [
                        {
                            "id": "msg-shared-1",
                            "parent_id": None,
                            "role": "user",
                            "text": "Alpha account question about growing heirloom balcony tomatoes in pots."
                        },
                        {
                            "id": "msg-shared-2",
                            "parent_id": "msg-shared-1",
                            "role": "assistant",
                            "text": msg_alpha_2
                        }
                    ]
                },
                {
                    "source_id": "chatgpt",
                    "account_id": "account-beta",
                    "id": "conv-shared",
                    "title": "Shared ID Beta Conversation",
                    "messages": [
                        {
                            "id": "msg-shared-1",
                            "parent_id": None,
                            "role": "user",
                            "text": "Beta account question about commuter bicycle gear ratios and rain tire selection."
                        },
                        {
                            "id": "msg-shared-2",
                            "parent_id": "msg-shared-1",
                            "role": "assistant",
                            "text": msg_beta_2
                        }
                    ]
                }
            ],
            "queries": [
                {
                    "id": "q-alpha-1",
                    "split": "calibration",
                    "type": "positive",
                    "query": "balcony tomato potting containers",
                    "filters": {"account_id": "account-alpha"},
                    "expected_passages": [
                        {
                            "source_id": "chatgpt",
                            "account_id": "account-alpha",
                            "conversation_id": "conv-shared",
                            "message_id": "msg-shared-2",
                            "start_offset": 0,
                            "end_offset": len(msg_alpha_2),
                            "expected_text": msg_alpha_2,
                            "expected_ancestor_ids": ["msg-shared-1"]
                        }
                    ]
                },
                {
                    "id": "q-beta-1",
                    "split": "holdout",
                    "type": "positive",
                    "query": "puncture resistant bicycle tires",
                    "filters": {"account_id": "account-beta"},
                    "expected_passages": [
                        {
                            "source_id": "chatgpt",
                            "account_id": "account-beta",
                            "conversation_id": "conv-shared",
                            "message_id": "msg-shared-2",
                            "start_offset": 0,
                            "end_offset": len(msg_beta_2),
                            "expected_text": msg_beta_2,
                            "expected_ancestor_ids": ["msg-shared-1"]
                        }
                    ]
                },
                {
                    "id": "q-alpha-unrel",
                    "split": "calibration",
                    "type": "unrelated",
                    "query": "quantum cryogenic superconductor resonator frequency calibration"
                },
                {
                    "id": "q-beta-unrel",
                    "split": "holdout",
                    "type": "unrelated",
                    "query": "deep seafloor hydrothermal vent geological tectonic plate boundary"
                }
            ]
        }

        validate_suite(temp_suite)

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tf:
            json.dump(temp_suite, tf, indent=2, ensure_ascii=False)
            tf_path = Path(tf.name)

        try:
            args = MagicMock()
            args.fixture = tf_path
            args.ollama = False
            args.provider = None

            report = evaluate(args)
            self.assertTrue(report["gates"]["structure_passed"])
            self.assertEqual(report["metrics"]["context_identity_errors"], 0)
            self.assertEqual(len(report["positive_results"]), 2)
            self.assertEqual(report["positive_results"][0]["ranks"]["conversation_rank"], 1)
            self.assertEqual(report["positive_results"][0]["ranks"]["message_rank"], 1)
            self.assertEqual(report["positive_results"][0]["ranks"]["passage_rank"], 1)
            self.assertEqual(report["positive_results"][1]["ranks"]["conversation_rank"], 1)
            self.assertEqual(report["positive_results"][1]["ranks"]["message_rank"], 1)
            self.assertEqual(report["positive_results"][1]["ranks"]["passage_rank"], 1)
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_filter_scope_off_account_or_wrong_role_result_fails_gates(self):
        """Injected off-account result alongside correct gold hit must increment context_identity_errors and fail gates."""
        msg_alpha_2 = "Alpha response regarding five gallon balcony tomato potting containers and daily watering schedule."
        msg_beta_2 = "Beta response regarding wide puncture resistant bicycle tires for rainy street commuting."

        temp_suite = {
            "synthetic": True,
            "version": "passage-evaluation-v3",
            "conversations": [
                {
                    "source_id": "chatgpt",
                    "account_id": "account-alpha",
                    "id": "conv-shared",
                    "title": "Shared ID Alpha Conversation",
                    "messages": [
                        {"id": "msg-shared-1", "parent_id": None, "role": "user", "text": "Alpha account question about growing heirloom balcony tomatoes in pots."},
                        {"id": "msg-shared-2", "parent_id": "msg-shared-1", "role": "assistant", "text": msg_alpha_2}
                    ]
                },
                {
                    "source_id": "chatgpt",
                    "account_id": "account-beta",
                    "id": "conv-shared",
                    "title": "Shared ID Beta Conversation",
                    "messages": [
                        {"id": "msg-shared-1", "parent_id": None, "role": "user", "text": "Beta account question about commuter bicycle gear ratios and rain tire selection."},
                        {"id": "msg-shared-2", "parent_id": "msg-shared-1", "role": "assistant", "text": msg_beta_2}
                    ]
                }
            ],
            "queries": [
                {
                    "id": "q-alpha-1",
                    "split": "calibration",
                    "type": "positive",
                    "query": "balcony tomato potting containers",
                    "filters": {"account_id": "account-alpha", "role": "assistant"},
                    "expected_passages": [
                        {
                            "source_id": "chatgpt",
                            "account_id": "account-alpha",
                            "conversation_id": "conv-shared",
                            "message_id": "msg-shared-2",
                            "start_offset": 0,
                            "end_offset": len(msg_alpha_2),
                            "expected_text": msg_alpha_2,
                            "expected_ancestor_ids": ["msg-shared-1"]
                        }
                    ]
                },
                {
                    "id": "q-alpha-unrel",
                    "split": "calibration",
                    "type": "unrelated",
                    "query": "quantum cryogenic superconductor resonator frequency calibration"
                }
            ]
        }

        validate_suite(temp_suite)

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tf:
            json.dump(temp_suite, tf, indent=2, ensure_ascii=False)
            tf_path = Path(tf.name)

        try:
            from chat_recall.search import SearchIndex
            original_search = SearchIndex.search

            # Simulate search returning a valid account-beta hit alongside the correct account-alpha gold hit
            def leaky_search(self_idx, query, *s_args, **s_kwargs):
                resp = original_search(self_idx, query, *s_args, **s_kwargs)
                if query == "balcony tomato potting containers":
                    off_account_hit = {
                        "source_id": "chatgpt",
                        "account_id": "account-beta",
                        "conversation_id": "conv-shared",
                        "message_id": "msg-shared-2",
                        "title": "Shared ID Beta Conversation",
                        "role": "assistant",
                        "text": msg_beta_2,
                        "focus_offset": 0,
                        "score": 0.5,
                        "match_type": "lexical"
                    }
                    resp["results"].append(off_account_hit)
                return resp

            with patch.object(SearchIndex, "search", leaky_search):
                args = MagicMock()
                args.fixture = tf_path
                args.ollama = False
                args.provider = None

                report = evaluate(args)
                self.assertGreater(report["metrics"]["context_identity_errors"], 0)
                self.assertFalse(report["gates"]["structure_passed"])
                self.assertFalse(report["gates"]["quality_passed"])
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_provider_failure_regressions_lexical_cannot_pass_quality(self):
        """When query provider fails at runtime after enrichment, quality gate must strictly fail."""
        class MockFailingProvider:
            model_id = "mock:failing-provider"
            minimum_similarity = 0.3

            def __init__(self):
                self.enrich_done = False

            def embed(self, texts):
                if not self.enrich_done:
                    return [[1.0, 0.0] for _ in texts]
                raise ProviderError("Local provider daemon connection refused")

        mock_prov = MockFailingProvider()

        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = mock_prov

        from chat_recall.search import SearchIndex
        original_enrich = SearchIndex.enrich

        def wrapped_enrich(self_index, *e_args, **e_kwargs):
            res = original_enrich(self_index, *e_args, **e_kwargs)
            mock_prov.enrich_done = True
            return res

        with patch.object(SearchIndex, "enrich", wrapped_enrich):
            report = evaluate(args)

            self.assertEqual(report["metrics"]["semantic_used_count"], 0)
            self.assertGreater(len(report["semantic_failures"]), 0)
            self.assertEqual(report["semantic_failures"][0]["status"], "provider_error")

            # Structure passed is True because DB and contexts are intact
            self.assertTrue(report["gates"]["structure_passed"])
            # Quality passed must strictly FAIL
            self.assertFalse(report["gates"]["quality_passed"])

    def test_cli_exit_codes(self):
        """Test CLI arguments and exit codes: --require-quality requires --ollama; baseline exits 0."""
        with patch("sys.argv", ["scripts/evaluate_passages.py", "--require-quality"]), \
             patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertNotEqual(cm.exception.code, 0)

        # Baseline execution exits 0
        with patch("sys.argv", ["scripts/evaluate_passages.py"]), \
             patch("sys.stdout", new_callable=io.StringIO):
            code = main()
            self.assertEqual(code, 0)

    def test_validation_rejects_zero_span_gold_with_expected_text_before_indexing(self):
        """Pre-validation must reject gold spans with start_offset == end_offset == 0 even if expected_text is present.
        Must fail before SearchIndex or Ollama factory calls.
        """
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(suite)
        target_gold = mutated["queries"][0]["expected_passages"][0]
        target_gold["start_offset"] = 0
        target_gold["end_offset"] = 0
        target_gold["expected_text"] = "Invented placeholder text"

        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated)
        self.assertIn("Invalid end_offset", str(cm.exception))

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tf:
            json.dump(mutated, tf)
            tf_path = Path(tf.name)

        try:
            args = MagicMock()
            args.fixture = tf_path
            args.ollama = False
            args.provider = None

            with patch("chat_recall.search.SearchIndex.__init__") as mock_init:
                with self.assertRaises(FixtureValidationError):
                    evaluate(args)
                mock_init.assert_not_called()
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_validation_rejects_parent_graph_cycle_and_invalid_role(self):
        """Pre-validation must reject parent graph cycles and invalid message roles before indexing."""
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))

        # Test cycle in parent links
        mutated_cycle = copy.deepcopy(suite)
        c0 = mutated_cycle["conversations"][0]
        c0["messages"][0]["parent_id"] = c0["messages"][1]["id"]
        c0["messages"][1]["parent_id"] = c0["messages"][0]["id"]

        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated_cycle)
        self.assertIn("Cycle detected", str(cm.exception))

        # Test invalid role
        mutated_role = copy.deepcopy(suite)
        mutated_role["conversations"][0]["messages"][0]["role"] = "system"
        with self.assertRaises(FixtureValidationError) as cm:
            validate_suite(mutated_role)
        self.assertIn("role must be 'user' or 'assistant'", str(cm.exception))

    def test_unrelated_nonempty_results_with_zero_total_fails_gates(self):
        """When unrelated query returns nonempty genuine results but total=0:
        has_any_result must be True, any-result FP rate must be > 0,
        schema error must be recorded, and structure_passed/quality_passed must be False.
        """
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None

        from chat_recall.search import SearchIndex
        original_search = SearchIndex.search

        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        unrel_q_text = next(q["query"] for q in suite["queries"] if q["type"] == "unrelated")

        def zero_total_leaky_search(self_idx, query, *s_args, **s_kwargs):
            resp = original_search(self_idx, query, *s_args, **s_kwargs)
            if query == unrel_q_text:
                # Genuine source-backed hit from conversation 0 message 0
                c0 = suite["conversations"][0]
                m0 = c0["messages"][0]
                genuine_hit = {
                    "source_id": c0.get("source_id", "chatgpt"),
                    "account_id": c0.get("account_id", "default"),
                    "conversation_id": c0["id"],
                    "message_id": m0["id"],
                    "title": c0.get("title", ""),
                    "role": m0["role"],
                    "text": m0["text"],
                    "focus_offset": 0,
                    "score": 0.8,
                    "match_type": "semantic",
                }
                resp["results"] = [genuine_hit]
                resp["total"] = 0  # reviewer patch returning genuine results but claiming total=0
            return resp

        with patch.object(SearchIndex, "search", zero_total_leaky_search):
            report = evaluate(args)

            # has_any_result must detect the nonempty results list despite total=0
            self.assertGreater(report["metrics"]["unrelated_any_result_false_positive_rate"], 0.0)
            self.assertGreater(report["metrics"]["unrelated_any_result_false_positive_count"], 0)

            # Schema error for len(results) > total must be recorded in query_errors
            schema_errors = [e for e in report["query_errors"] if "Search response schema error" in e["error"]]
            self.assertGreater(len(schema_errors), 0)

            # Both structure and quality gates must strictly FAIL
            self.assertFalse(report["gates"]["structure_passed"])
            self.assertFalse(report["gates"]["quality_passed"])

    def test_search_response_schema_validation_and_invalid_totals(self):
        """Search response schema must validate dict, exact int non-negative total, results list, and len(results) <= total."""
        # Valid response
        self.assertEqual(validate_search_response_schema({"total": 1, "results": [{}]}), [])
        self.assertEqual(validate_search_response_schema({"total": 0, "results": []}), [])

        # Non-dict
        self.assertIn("must be a dict", validate_search_response_schema("not-a-dict")[0])

        # Missing total / results
        self.assertIn("missing 'total'", validate_search_response_schema({"results": []})[0])
        self.assertIn("missing 'results'", validate_search_response_schema({"total": 0})[0])

        # Negative total
        errs = validate_search_response_schema({"total": -1, "results": []})
        self.assertTrue(any("non-negative" in e for e in errs))

        # String total
        errs = validate_search_response_schema({"total": "5", "results": []})
        self.assertTrue(any("exact int" in e for e in errs))

        # Bool total (True/False are bool, which subclasses int)
        errs_bool = validate_search_response_schema({"total": True, "results": []})
        self.assertTrue(any("exact int" in e for e in errs_bool))

        # Results not a list
        errs_not_list = validate_search_response_schema({"total": 5, "results": "not-list"})
        self.assertTrue(any("must be a list" in e for e in errs_not_list))

        # len(results) > total
        errs_overflow = validate_search_response_schema({"total": 0, "results": [{"id": 1}]})
        self.assertTrue(any("exceeds total" in e for e in errs_overflow))

    def test_hit_integrity_invented_text_fails_gates(self):
        """Invented hit.text alongside valid source context must increment context_identity_errors and fail gates."""
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None

        from chat_recall.search import SearchIndex
        original_search = SearchIndex.search

        # Mutate displayed text on first returned hit while leaving context/DB untouched
        def invented_hit_text_search(self_idx, query, *s_args, **s_kwargs):
            resp = original_search(self_idx, query, *s_args, **s_kwargs)
            if resp.get("results"):
                # Copy first hit and alter displayed text
                hit = dict(resp["results"][0])
                hit["text"] = "Invented synthetic text not in fixture DB"
                resp["results"][0] = hit
            return resp

        with patch.object(SearchIndex, "search", invented_hit_text_search):
            report = evaluate(args)
            self.assertGreater(report["metrics"]["context_identity_errors"], 0)
            self.assertFalse(report["gates"]["structure_passed"])
            self.assertFalse(report["gates"]["quality_passed"])

    def test_hit_integrity_validation_checks(self):
        """validate_hit_integrity checks displayed text, role, and focus_offset against fixture."""
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        conv_map = {}
        for conv in suite["conversations"]:
            src = conv.get("source_id", "chatgpt")
            acc = conv.get("account_id", "default")
            cid = conv["id"]
            conv_map[(src, acc, cid)] = {m["id"]: m for m in conv.get("messages", [])}

        c0 = suite["conversations"][0]
        m0 = c0["messages"][0]

        valid_hit = {
            "source_id": c0.get("source_id", "chatgpt"),
            "account_id": c0.get("account_id", "default"),
            "conversation_id": c0["id"],
            "message_id": m0["id"],
            "role": m0["role"],
            "text": m0["text"],
            "focus_offset": 0,
        }
        self.assertEqual(validate_hit_integrity(valid_hit, conv_map), [])

        # Text mismatch
        bad_text = dict(valid_hit, text="Wrong text")
        self.assertTrue(any("Hit text mismatch" in e for e in validate_hit_integrity(bad_text, conv_map)))

        # Role mismatch
        bad_role = dict(valid_hit, role="other_role")
        self.assertTrue(any("Hit role mismatch" in e for e in validate_hit_integrity(bad_role, conv_map)))

        # Invalid focus_offset (bool, string, out-of-bounds)
        bad_focus_bool = dict(valid_hit, focus_offset=True)
        self.assertTrue(any("Hit focus_offset must be an exact int" in e for e in validate_hit_integrity(bad_focus_bool, conv_map)))

        bad_focus_str = dict(valid_hit, focus_offset="0")
        self.assertTrue(any("Hit focus_offset must be an exact int" in e for e in validate_hit_integrity(bad_focus_str, conv_map)))

        bad_focus_oob = dict(valid_hit, focus_offset=len(m0["text"]) + 100)
        self.assertTrue(any("out of bounds" in e for e in validate_hit_integrity(bad_focus_oob, conv_map)))

    def test_recording_provider_wrapper_forwards_encoding_identity_and_methods(self):
        from scripts.evaluate_passages import RecordingProviderWrapper

        class FullSpecializedProvider:
            model_id = "test:specialized"
            expected_output_dimension = 1024
            cached_identity = "test:specialized:fingerprint456"
            query_template_version = "qwen-instruct-v1"
            calibration_version = "provisional-v1"

            def __init__(self):
                import threading
                self._lock = threading.RLock()

            @property
            def lock(self):
                return self._lock

            @property
            def operation_lock(self):
                return self._lock

            def resolve_identity(self, doc_profile):
                return self.cached_identity

            def embed_query(self, query):
                return [3.0, 4.0]

            def embed_documents(self, texts):
                return [[3.0, 4.0] for _ in texts]

            def embed(self, texts):
                return [[3.0, 4.0] for _ in texts]

        prov = FullSpecializedProvider()
        wrapper = RecordingProviderWrapper(prov)

        # Forward identity and attributes
        self.assertEqual(wrapper.resolve_identity({"version": "v1"}), "test:specialized:fingerprint456")
        self.assertEqual(wrapper.index_id, "test:specialized:fingerprint456")
        self.assertEqual(wrapper.cached_identity, "test:specialized:fingerprint456")
        self.assertEqual(wrapper.expected_output_dimension, 1024)
        self.assertEqual(wrapper.query_template_version, "qwen-instruct-v1")
        self.assertEqual(wrapper.calibration_version, "provisional-v1")
        self.assertIsNotNone(wrapper.lock)
        self.assertIsNotNone(wrapper.operation_lock)

        # Unresolved versioned provider: index_id must NOT fall back to display alias
        class UnresolvedVersionedProvider:
            model_id = "test:unresolved"
            cached_identity = None
            def resolve_identity(self, doc_profile):
                return "test:unresolved:resolved"
        w_unresolved = RecordingProviderWrapper(UnresolvedVersionedProvider())
        self.assertTrue(callable(getattr(w_unresolved, "resolve_identity", None)))
        self.assertIsNone(w_unresolved.cached_identity)
        self.assertIsNone(w_unresolved.index_id)

        # Legacy provider: capability and status transparency
        class LegacyUnderlyingProvider:
            model_id = "test:legacy-trans"
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]
        w_legacy_trans = RecordingProviderWrapper(LegacyUnderlyingProvider())
        self.assertFalse(callable(getattr(w_legacy_trans, "resolve_identity", None)))
        self.assertEqual(w_legacy_trans.cached_identity, "test:legacy-trans")
        self.assertEqual(w_legacy_trans.index_id, "test:legacy-trans")

        # Legacy fallback must validate provider response is list with exactly one vector
        class EmptyLegacyProvider:
            model_id = "test:empty-legacy"
            def embed(self, texts):
                return []
        w_empty = RecordingProviderWrapper(EmptyLegacyProvider())
        with self.assertRaises(ProviderError):
            w_empty.embed_query("query")

        class MultiLegacyProvider:
            model_id = "test:multi-legacy"
            def embed(self, texts):
                return [[1.0, 0.0], [0.0, 1.0]]
        w_multi = RecordingProviderWrapper(MultiLegacyProvider())
        with self.assertRaises(ProviderError):
            w_multi.embed_query("query")

        # Bound provider raising internal TypeError must call provider exactly once without retry
        class BoundProviderInternalTypeError:
            model_id = "test:bound-err"
            def __init__(self):
                self.calls = 0
            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                self.calls += 1
                raise TypeError("Internal execution error")
        b_prov = BoundProviderInternalTypeError()
        w_b_prov = RecordingProviderWrapper(b_prov)
        with self.assertRaises(TypeError):
            w_b_prov.embed_query("query", index_id="some-id")
        self.assertEqual(b_prov.calls, 1)
        self.assertEqual(w_b_prov.embed_call_count, 1)

        # Legacy provider without index_id parameter is supported and invoked cleanly
        class LegacyNoIndexIdMock:
            model_id = "test:legacy-no-id"
            def __init__(self):
                self.calls = 0
            def embed_query(self, query: str) -> list[float]:
                self.calls += 1
                return [0.6, 0.8]
        leg_prov = LegacyNoIndexIdMock()
        w_leg = RecordingProviderWrapper(leg_prov)
        leg_vec = w_leg.embed_query("legacy query")
        self.assertEqual(leg_vec, [0.6, 0.8])
        self.assertEqual(leg_prov.calls, 1)
        self.assertEqual(w_leg.embed_call_count, 1)

        # embed_query records normalized query vector
        self.assertIsNone(wrapper.last_query_vector)
        q_vec = wrapper.embed_query("passage query")
        self.assertEqual(q_vec, [0.6, 0.8])
        self.assertEqual(wrapper.last_query_vector, [0.6, 0.8])
        self.assertEqual(wrapper.embed_call_count, 1)

        # embed_documents forwards correctly without altering query vector
        wrapper.reset_query_vector()
        self.assertIsNone(wrapper.last_query_vector)
        doc_vecs = wrapper.embed_documents(["doc 1", "doc 2"])
        self.assertEqual(len(doc_vecs), 2)
        self.assertEqual(doc_vecs[0], [3.0, 4.0])
        self.assertIsNone(wrapper.last_query_vector)

    def test_calibration_only_scope_excludes_holdout_and_reports_scope(self):
        """Calibration-only scope imports only calibration conversations, runs only calibration queries,
        and excludes all holdout queries and conversations from execution and report.
        """
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        holdout_conv_ids = {
            g["conversation_id"]
            for q in suite["queries"]
            if q.get("split") == "holdout" and q.get("type") == "positive"
            for g in q.get("expected_passages", [])
        }
        holdout_q_ids = {q["id"] for q in suite["queries"] if q.get("split") == "holdout"}

        # Tracking mock provider to prove holdout queries and documents are never encoded
        class TrackingProvider:
            model_id = "test:tracking-calibration"
            minimum_similarity = 0.50

            def __init__(self):
                self.encoded_queries: list[str] = []
                self.encoded_documents: list[str] = []

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                self.encoded_queries.append(query)
                return [1.0, 0.0]

            def embed_documents(self, texts: list[str], index_id: str | None = None) -> list[list[float]]:
                self.encoded_documents.extend(texts)
                return [[1.0, 0.0] for _ in texts]

        tracking_prov = TrackingProvider()
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = tracking_prov
        args.calibration_only = True
        args.require_quality = False

        report = evaluate(args, calibration_only=True)

        self.assertTrue(report["synthetic"])
        self.assertTrue(report["calibration_only"])
        self.assertEqual(report["run_scope"], "calibration-only")
        expected_fixture_hash = hashlib.sha256(self.fixture_path.read_bytes()).hexdigest()
        self.assertEqual(report["fixture_hash"], expected_fixture_hash)

        # Only 8 calibration conversations imported (4 holdout conversations excluded)
        self.assertEqual(report["import"]["conversations"], 8)
        self.assertEqual(report["metrics"]["positive_queries"], 32)
        self.assertEqual(report["metrics"]["unrelated_queries"], 8)
        self.assertEqual(report["metrics"]["expected_queries_total"], 40)

        # All positive results and unrelated results are calibration split only
        self.assertEqual(len(report["positive_results"]), 32)
        for r in report["positive_results"]:
            self.assertEqual(r["split"], "calibration")
            self.assertNotIn(r["id"], holdout_q_ids)

        self.assertEqual(len(report["unrelated_results"]), 8)
        for r in report["unrelated_results"]:
            self.assertEqual(r["split"], "calibration")
            self.assertNotIn(r["id"], holdout_q_ids)

        # Split results must contain only calibration, omitting holdout entirely
        self.assertIn("calibration", report["metrics"]["split_results"])
        self.assertNotIn("holdout", report["metrics"]["split_results"])

        # Gates: structure passes, but quality/holdout gates MUST NOT pass on partial calibration scope
        self.assertTrue(report["gates"]["structure_passed"])
        self.assertFalse(report["gates"]["quality_passed"])
        self.assertFalse(report["gates"]["holdout_passed"])

        # Verify no holdout queries or documents were ever encoded by the provider
        holdout_queries_text = {q["query"] for q in suite["queries"] if q.get("split") == "holdout"}
        for encoded_q in tracking_prov.encoded_queries:
            self.assertNotIn(encoded_q, holdout_queries_text)

        holdout_messages_text = {
            m["text"]
            for conv in suite["conversations"]
            if conv["id"] in holdout_conv_ids
            for m in conv["messages"]
        }
        for encoded_doc in tracking_prov.encoded_documents:
            for h_text in holdout_messages_text:
                self.assertNotIn(h_text, encoded_doc)

    def test_calibration_only_denominator_integrity_on_failed_query(self):
        """Failed or dropped queries under calibration-only retain full calibration denominator (32)."""
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None
        args.calibration_only = True
        args.require_quality = False

        from chat_recall.search import SearchIndex
        original_search = SearchIndex.search

        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        calib_first_q = next(q["query"] for q in suite["queries"] if q.get("split") == "calibration" and q["type"] == "positive")

        def failing_search(self_idx, query, *s_args, **s_kwargs):
            if query == calib_first_q:
                raise RuntimeError("Simulated calibration query search failure")
            return original_search(self_idx, query, *s_args, **s_kwargs)

        with patch.object(SearchIndex, "search", failing_search):
            report = evaluate(args, calibration_only=True)

            self.assertEqual(len(report["query_errors"]), 1)
            # Denominator remains full calibration positive total (32)
            self.assertEqual(report["metrics"]["positive_queries"], 32)
            self.assertEqual(report["metrics"]["completed_queries_total"], 39)
            self.assertEqual(len(report["positive_results"]), 31)

            # Structure gate must fail
            self.assertFalse(report["gates"]["structure_passed"])
            self.assertFalse(report["gates"]["quality_passed"])

    def test_calibration_only_rejects_require_quality(self):
        """--require-quality cannot be combined with --calibration-only (callable and CLI reject)."""
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = True
        args.provider = None
        args.calibration_only = True
        args.require_quality = True

        with self.assertRaises(ValueError) as cm:
            evaluate(args, calibration_only=True)
        self.assertIn("--require-quality cannot be combined with --calibration-only", str(cm.exception))

        # CLI rejects combination with non-zero exit
        with patch("sys.argv", ["scripts/evaluate_passages.py", "--ollama", "--require-quality", "--calibration-only"]), \
             patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as cm_cli:
                main()
            self.assertNotEqual(cm_cli.exception.code, 0)

    def test_full_gate_default_scope_unchanged(self):
        """Default evaluation scope is full: 12 conversations, 48 positives, 16 unrelated, both splits reported."""
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None
        args.calibration_only = False
        args.require_quality = False

        report = evaluate(args)

        self.assertEqual(report["run_scope"], "full")
        self.assertFalse(report["calibration_only"])
        self.assertEqual(report["import"]["conversations"], 12)
        self.assertEqual(report["metrics"]["positive_queries"], 48)
        self.assertEqual(report["metrics"]["unrelated_queries"], 16)
        self.assertIn("calibration", report["metrics"]["split_results"])
        self.assertIn("holdout", report["metrics"]["split_results"])
        self.assertTrue(report["gates"]["structure_passed"])
        self.assertFalse(report["gates"]["quality_passed"])
        self.assertFalse(report["gates"]["holdout_passed"])

    def test_caching_provider_across_two_thresholds_encodes_each_input_once(self):
        """Across 2 thresholds, each unique query and document is encoded exactly once by underlying provider."""
        class MockTrackingProvider:
            model_id = "test:tracking-caching-provider"
            minimum_similarity = 0.50

            def __init__(self):
                self.query_inputs: list[str] = []
                self.doc_inputs: list[str] = []

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                self.query_inputs.append(query)
                return [1.0, 0.0]

            def embed_documents(self, texts: list[str], index_id: str | None = None) -> list[list[float]]:
                self.doc_inputs.extend(texts)
                return [[1.0, 0.0] for _ in texts]

        mock_prov = MockTrackingProvider()
        caching_wrapper = CachingProviderWrapper(mock_prov)

        report = calibrate(
            thresholds=[0.50, 0.60],
            provider=caching_wrapper,
            fixture_path=self.fixture_path,
        )

        self.assertEqual(report["benchmark"], "passage-calibration-v3")
        self.assertTrue(report["calibration_only"])
        self.assertEqual(report["candidate_thresholds"], [0.50, 0.60])
        self.assertEqual(len(report["threshold_results"]), 2)

        t1_res = report["threshold_results"][0]
        t2_res = report["threshold_results"][1]

        # First threshold does actual inference; second threshold is cached-sweep
        self.assertEqual(t1_res["latency"]["latency_type"], "actual-inference")
        self.assertGreater(t1_res["provider_calls"]["real_provider_calls"], 0)

        self.assertEqual(t2_res["latency"]["latency_type"], "cached-sweep")
        self.assertEqual(t2_res["provider_calls"]["real_provider_calls"], 0)
        self.assertGreater(t2_res["provider_calls"]["cache_hits"], 0)

        # Unique queries across both thresholds: each encoded exactly once
        unique_queries = set(mock_prov.query_inputs)
        self.assertEqual(len(mock_prov.query_inputs), len(unique_queries))
        self.assertEqual(len(unique_queries), 40)  # 32 positive + 8 unrelated calibration queries

        # Unique documents across both thresholds: each encoded exactly once
        unique_docs = set(mock_prov.doc_inputs)
        self.assertEqual(len(mock_prov.doc_inputs), len(unique_docs))

        # Overall gates: partial calibration exploration MUST NOT claim full quality/holdout pass
        self.assertTrue(report["gates"]["structure_passed"])
        self.assertFalse(report["gates"]["quality_passed"])
        self.assertFalse(report["gates"]["holdout_passed"])

    def test_caching_provider_query_and_document_namespaces_never_mix(self):
        """Query and document caches are strictly separate; identical input text never reuses opposite cache."""
        class MockCounterProvider:
            model_id = "test:counter-prov"
            query_calls = 0
            doc_calls = 0

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                self.query_calls += 1
                return [0.6, 0.8]

            def embed_documents(self, texts: list[str], index_id: str | None = None) -> list[list[float]]:
                self.doc_calls += 1
                return [[0.6, 0.8] for _ in texts]

        prov = MockCounterProvider()
        wrapper = CachingProviderWrapper(prov)

        shared_text = "Exact identical content text for query and document"

        # 1. Embed as document
        doc_res = wrapper.embed_documents([shared_text])
        self.assertEqual(doc_res, [[0.6, 0.8]])
        self.assertEqual(prov.doc_calls, 1)
        self.assertEqual(prov.query_calls, 0)
        self.assertEqual(wrapper.doc_cache_hits, 0)

        # 2. Embed same text as query: must NOT pull from document cache
        q_res = wrapper.embed_query(shared_text)
        self.assertEqual(q_res, [0.6, 0.8])
        self.assertEqual(prov.query_calls, 1)  # Provider called!
        self.assertEqual(wrapper.query_cache_hits, 0)  # Cache miss!

        # 3. Embed again as query: must hit query cache
        q_res_2 = wrapper.embed_query(shared_text)
        self.assertEqual(q_res_2, [0.6, 0.8])
        self.assertEqual(prov.query_calls, 1)
        self.assertEqual(wrapper.query_cache_hits, 1)

        # 4. Embed again as document: must hit document cache
        doc_res_2 = wrapper.embed_documents([shared_text])
        self.assertEqual(doc_res_2, [[0.6, 0.8]])
        self.assertEqual(prov.doc_calls, 1)
        self.assertEqual(wrapper.doc_cache_hits, 1)

    def test_caching_provider_type_error_no_retry(self):
        """Provider TypeError during execution propagates immediately without unbound retry."""
        class TypeErrorProvider:
            model_id = "test:type-error-prov"
            calls = 0

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                self.calls += 1
                raise TypeError("Internal execution TypeError")

        prov = TypeErrorProvider()
        wrapper = CachingProviderWrapper(prov)

        with self.assertRaises(TypeError) as cm:
            wrapper.embed_query("failing query")
        self.assertIn("Internal execution TypeError", str(cm.exception))
        # Called exactly once; no retry
        self.assertEqual(prov.calls, 1)

    def test_caching_provider_changed_identity_cannot_reuse_vectors(self):
        """When provider identity changes, cached vectors for previous identity are invalidated and cannot be reused."""
        class DynamicIdentityProvider:
            model_id = "model-v1"
            calls = 0

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                self.calls += 1
                return [1.0, 0.0]

        prov = DynamicIdentityProvider()
        wrapper = CachingProviderWrapper(prov)

        # 1. Embed under model-v1
        v1 = wrapper.embed_query("target query")
        self.assertEqual(v1, [1.0, 0.0])
        self.assertEqual(prov.calls, 1)

        # Re-embedding same query under same identity hits cache
        v1_cached = wrapper.embed_query("target query")
        self.assertEqual(v1_cached, [1.0, 0.0])
        self.assertEqual(prov.calls, 1)
        self.assertEqual(wrapper.query_cache_hits, 1)

        # 2. Change identity to model-v2: cannot reuse model-v1 vectors
        prov.model_id = "model-v2"
        v2 = wrapper.embed_query("target query", index_id="model-v2")
        self.assertEqual(v2, [1.0, 0.0])
        # Provider must be called again because identity changed!
        self.assertEqual(prov.calls, 2)

    def test_caching_provider_cache_cap_and_invalid_vectors_fail(self):
        """Finite memory cap and invalid numeric vectors fail loudly with explicit bounded errors."""
        class DummyProvider:
            model_id = "test:dummy"

            def __init__(self, vector):
                self.vector = vector

            def embed_query(self, query: str, index_id: str | None = None):
                return self.vector

        # 1. Finite memory cap exceeded
        prov_cap = DummyProvider([1.0, 0.0])
        wrapper_cap = CachingProviderWrapper(prov_cap, max_capacity=2)
        wrapper_cap.embed_query("query 1")
        wrapper_cap.embed_query("query 2")
        with self.assertRaises(CacheCapacityExceededError) as cm_cap:
            wrapper_cap.embed_query("query 3")
        self.assertIn("Vector cache cap (2) exceeded", str(cm_cap.exception))

        # 2. Oversized dimension (> 1024)
        oversized_vec = [0.1] * 1025
        prov_oversized = DummyProvider(oversized_vec)
        wrapper_oversized = CachingProviderWrapper(prov_oversized)
        with self.assertRaises(InvalidVectorError) as cm_dim:
            wrapper_oversized.embed_query("query oversized")
        self.assertIn("exceeds cap 1024", str(cm_dim.exception))

        # 3. Non-numeric elements (bool, nan, string, None)
        bad_bool_vec = [1.0, True, 0.5]
        wrapper_bool = CachingProviderWrapper(DummyProvider(bad_bool_vec))
        with self.assertRaises(InvalidVectorError) as cm_bool:
            wrapper_bool.embed_query("query bool")
        self.assertIn("must be numbers (not bool)", str(cm_bool.exception))

        bad_nan_vec = [1.0, float("nan"), 0.5]
        wrapper_nan = CachingProviderWrapper(DummyProvider(bad_nan_vec))
        with self.assertRaises(InvalidVectorError) as cm_nan:
            wrapper_nan.embed_query("query nan")
        self.assertIn("non-finite", str(cm_nan.exception))

        # 4. Empty vector
        wrapper_empty = CachingProviderWrapper(DummyProvider([]))
        with self.assertRaises(InvalidVectorError) as cm_empty:
            wrapper_empty.embed_query("query empty")
        self.assertIn("cannot be empty", str(cm_empty.exception))

        # 5. Non-vector structure
        wrapper_str = CachingProviderWrapper(DummyProvider("not-a-list"))
        with self.assertRaises(InvalidVectorError):
            wrapper_str.embed_query("query non-vector")

    def test_caching_provider_returns_copies_preventing_corruption(self):
        """Cached vectors are immutable tuples; return values are copies so callers cannot corrupt cached originals."""
        class CleanProvider:
            model_id = "test:clean"

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                return [10.0, 20.0, 30.0]

        prov = CleanProvider()
        wrapper = CachingProviderWrapper(prov)

        vec1 = wrapper.embed_query("test query")
        self.assertEqual(vec1, [10.0, 20.0, 30.0])

        # Mutate returned list in-place
        vec1[0] = 999999.0
        vec1[1] = -88888.0

        # Subsequent retrieval must return uncorrupted original values
        vec2 = wrapper.embed_query("test query")
        self.assertEqual(vec2, [10.0, 20.0, 30.0])

    def test_calibrate_offline_lexical_baseline_and_cli(self):
        """Offline lexical baseline explores candidate thresholds without network, model, or writes."""
        report = calibrate(fixture_path=self.fixture_path)

        self.assertEqual(report["benchmark"], "passage-calibration-v3")
        self.assertEqual(report["mode"], "lexical-baseline-only")
        self.assertTrue(report["calibration_only"])
        self.assertEqual(report["candidate_thresholds"], [0.45, 0.50, 0.55, 0.60, 0.65])
        self.assertEqual(len(report["threshold_results"]), 5)

        for res in report["threshold_results"]:
            self.assertTrue(res["structure_passed"])
            self.assertFalse(res["semantic"]["used"])
            self.assertEqual(res["metrics"]["expected_positive_count"], 32)
            self.assertEqual(res["metrics"]["expected_unrelated_count"], 8)

        self.assertTrue(report["gates"]["structure_passed"])
        self.assertFalse(report["gates"]["quality_passed"])
        self.assertFalse(report["gates"]["holdout_passed"])

        # CLI execution for calibrate_passages
        with patch("sys.argv", ["scripts/calibrate_passages.py", "--thresholds", "0.50,0.60"]), \
             patch("sys.stdout", new_callable=io.StringIO):
            code = calibrate_main()
            self.assertEqual(code, 0)

    def test_caching_provider_lazy_binding_constructor_does_not_contact_provider(self):
        """Constructor must not resolve/contact provider; lazily binds on evaluation boundary."""
        class MockVersionedProvider:
            model_id = "test:versioned"
            resolve_calls = 0

            def resolve_identity(self, document_profile=None):
                self.resolve_calls += 1
                return "test:versioned:fingerprint-abc"

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                return [1.0, 0.0]

        prov = MockVersionedProvider()
        wrapper = CachingProviderWrapper(prov)

        # Constructor did NOT resolve identity or contact provider
        self.assertEqual(prov.resolve_calls, 0)
        self.assertIsNone(wrapper._bound_identity)

        # Lazily binds on first call
        res = wrapper.embed_query("first query")
        self.assertEqual(res, [1.0, 0.0])
        self.assertEqual(prov.resolve_calls, 1)
        self.assertEqual(wrapper._bound_identity, "test:versioned:fingerprint-abc")

    def test_caching_provider_versioned_resolve_identity_fails_closed(self):
        """_get_current_identity fails closed and propagates resolve error; never treats model_id as resolved."""
        class FailingVersionedProvider:
            model_id = "test:failing"

            def resolve_identity(self, document_profile=None):
                raise ProviderError("Connection refused to Ollama endpoint")

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                return [1.0, 0.0]

        prov = FailingVersionedProvider()
        wrapper = CachingProviderWrapper(prov)

        with self.assertRaises(ProviderError) as cm:
            wrapper.embed_query("failing query")
        self.assertIn("Connection refused to Ollama endpoint", str(cm.exception))
        # Cache must not be populated
        self.assertEqual(len(wrapper._query_cache), 0)
        self.assertIsNone(wrapper._bound_identity)

    def test_caching_provider_supplied_index_id_must_match_resolved_identity(self):
        """Supplied index_id must match current resolved identity; cache hit does not bypass validation."""
        class IdentityTrackingProvider:
            model_id = "test:tracking"

            def resolve_identity(self, document_profile=None):
                return "test:tracking:resolved-id"

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                return [1.0, 0.0]

        prov = IdentityTrackingProvider()
        wrapper = CachingProviderWrapper(prov)

        # 1. First embed with matching resolved_id succeeds and caches
        v1 = wrapper.embed_query("cached query", index_id="test:tracking:resolved-id")
        self.assertEqual(v1, [1.0, 0.0])
        self.assertEqual(len(wrapper._query_cache), 1)

        # 2. Supplied nonmatching index_id must reject with ProviderError
        with self.assertRaises(ProviderError) as cm:
            wrapper.embed_query("cached query", index_id="unknown-or-bogus-id")
        self.assertIn("does not match current known resolved identity", str(cm.exception))

        # 3. Cache hit with matching index_id succeeds
        v2 = wrapper.embed_query("cached query", index_id="test:tracking:resolved-id")
        self.assertEqual(v2, [1.0, 0.0])
        self.assertEqual(wrapper.query_cache_hits, 1)

    def test_caching_provider_legacy_fake_provider_supported_without_inventing_version(self):
        """Legacy fake providers without resolve_identity remain supported without inventing version identity."""
        class LegacyFakeProvider:
            model_id = "legacy:model"
            index_id = "legacy:model"

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                return [0.6, 0.8]

        prov = LegacyFakeProvider()
        wrapper = CachingProviderWrapper(prov)
        self.assertFalse(callable(getattr(wrapper, "resolve_identity", None)))

        res = wrapper.embed_query("legacy test")
        self.assertEqual(res, [0.6, 0.8])
        self.assertEqual(wrapper._bound_identity, "legacy:model")

    def test_caching_provider_max_capacity_strict_int_and_bounds(self):
        """Enforces max_capacity strict int 1..MAX_CACHE_CAPACITY(1024), not bool/inf/unbounded/None."""
        class DummyProvider:
            model_id = "dummy"

        prov = DummyProvider()
        for invalid_cap in (True, False, 0, -1, -100, 1025, 2000, float("inf"), float("-inf"), None, "1024"):
            with self.assertRaises(ValueError, msg=f"Should reject max_capacity={invalid_cap!r}"):
                CachingProviderWrapper(prov, max_capacity=invalid_cap)

        # Valid capacities
        self.assertEqual(CachingProviderWrapper(prov, max_capacity=1).max_capacity, 1)
        self.assertEqual(CachingProviderWrapper(prov, max_capacity=1024).max_capacity, 1024)

    def test_caching_provider_default_threshold_finite_and_bounds(self):
        """Enforces default_threshold finite float in [0.0, 1.0], rejecting bool/nan/inf/out-of-bounds."""
        class DummyProvider:
            model_id = "dummy"

        prov = DummyProvider()
        for invalid_thresh in (True, False, float("nan"), float("inf"), float("-inf"), -0.01, 1.01, "0.5", None):
            with self.assertRaises(ValueError, msg=f"Should reject default_threshold={invalid_thresh!r}"):
                CachingProviderWrapper(prov, default_threshold=invalid_thresh)

        # Valid default thresholds
        w0 = CachingProviderWrapper(prov, default_threshold=0.0)
        self.assertEqual(w0.minimum_similarity, 0.0)
        w1 = CachingProviderWrapper(prov, default_threshold=1.0)
        self.assertEqual(w1.minimum_similarity, 1.0)

    def test_caching_provider_document_pipeline_fingerprint_in_keys(self):
        """Document cache key includes document pipeline fingerprint (SearchIndex.DOCUMENT_PROFILE, dim, max_bytes)."""
        class FingerprintTestProvider:
            model_id = "test:fp"
            expected_output_dimension = 768
            max_input_bytes = 960

            def embed_documents(self, texts: list[str], index_id: str | None = None) -> list[list[float]]:
                return [[1.0] + [0.0] * 767 for _ in texts]

        prov = FingerprintTestProvider()
        wrapper = CachingProviderWrapper(prov)

        wrapper.embed_documents(["document content"])
        self.assertEqual(len(wrapper._document_cache), 1)

        # Inspect cache key structure: (active_id, doc_fp, text_hash)
        cache_key = next(iter(wrapper._document_cache.keys()))
        active_id, doc_fp, text_hash = cache_key
        self.assertEqual(active_id, "test:fp")
        # Fingerprint is a 16-hex hash of DOCUMENT_PROFILE, dimension, max_input_bytes, not hard-coded 'v2'
        self.assertNotEqual(doc_fp, "v2")
        self.assertEqual(len(doc_fp), 16)

    def test_caching_provider_intra_batch_document_deduplication(self):
        """Deduplicate repeated document texts WITHIN one batch before calling provider; maps result to each input."""
        class DedupCheckingProvider:
            model_id = "test:dedup"
            calls = 0
            embedded_inputs: list[str] = []

            def embed_documents(self, texts: list[str], index_id: str | None = None) -> list[list[float]]:
                self.calls += 1
                self.embedded_inputs.extend(texts)
                return [[1.0, 0.0] for _ in texts]

        prov = DedupCheckingProvider()
        # Capacity is 1: batch has ['same', 'same']. Must succeed with 1 unique stored vector and 1 provider input.
        wrapper = CachingProviderWrapper(prov, max_capacity=1)

        results = wrapper.embed_documents(["same", "same"])
        self.assertEqual(results, [[1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(prov.calls, 1)
        self.assertEqual(prov.embedded_inputs, ["same"])
        self.assertEqual(wrapper.real_provider_embedded_documents, 1)
        self.assertEqual(len(wrapper._document_cache), 1)

        # Subsequent call with same text hits cache
        res2 = wrapper.embed_documents(["same"])
        self.assertEqual(res2, [[1.0, 0.0]])
        self.assertEqual(prov.calls, 1)
        self.assertEqual(wrapper.doc_cache_hits, 1)

    def test_caching_provider_zero_norm_query_vector_rejected_without_cache_pollution(self):
        """Zero norm vector is rejected before inserting into cache; does not consume capacity or leave entry."""
        class ZeroNormProvider:
            model_id = "test:zero-norm"

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                return [0.0, 0.0]

        prov = ZeroNormProvider()
        wrapper = CachingProviderWrapper(prov, max_capacity=5)

        with self.assertRaises(InvalidVectorError) as cm:
            wrapper.embed_query("query zero")
        self.assertIn("magnitude", str(cm.exception))

        # Failed vector must NEVER pollute cache or consume capacity
        self.assertEqual(len(wrapper._query_cache), 0)
        self.assertEqual(len(wrapper._document_cache), 0)

    def test_caching_provider_invalid_batch_does_not_leave_falsely_successful_cache_entries(self):
        """Incomplete or invalid batch does not leave partially successful cache entries."""
        class PartialFailureProvider:
            model_id = "test:partial-failure"

            def embed_documents(self, texts: list[str], index_id: str | None = None) -> list[list[float]]:
                # First vector is valid, second vector is zero-magnitude
                return [[1.0, 0.0], [0.0, 0.0]]

        prov = PartialFailureProvider()
        wrapper = CachingProviderWrapper(prov, max_capacity=5)

        with self.assertRaises(InvalidVectorError):
            wrapper.embed_documents(["valid text", "zero magnitude text"])

        # Neither vector must be in cache!
        self.assertEqual(len(wrapper._document_cache), 0)

    def test_calibrate_thresholds_strict_validation(self):
        """Threshold sweep validates strict finite thresholds 0..1, non-empty bounded list (max 21), rejects bool."""
        # 1. Empty list rejected
        with self.assertRaises(CalibrationError) as cm_empty:
            calibrate(thresholds=[], fixture_path=self.fixture_path)
        self.assertIn("cannot be empty", str(cm_empty.exception))

        # 2. Bool rejected
        for bad_bool in (True, False):
            with self.assertRaises(CalibrationError) as cm_bool:
                calibrate(thresholds=[bad_bool, 0.5], fixture_path=self.fixture_path)
            self.assertIn("must be a number (not bool)", str(cm_bool.exception))

        # 3. Non-finite rejected
        for bad_val in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(CalibrationError) as cm_finite:
                calibrate(thresholds=[bad_val], fixture_path=self.fixture_path)
            self.assertIn("must be finite", str(cm_finite.exception))

        # 4. Out-of-bounds rejected
        for bad_range in (-0.01, 1.01, 2.0):
            with self.assertRaises(CalibrationError) as cm_range:
                calibrate(thresholds=[bad_range], fixture_path=self.fixture_path)
            self.assertIn("within [0.0, 1.0]", str(cm_range.exception))

        # 5. Over max 21 rejected
        with self.assertRaises(CalibrationError) as cm_max:
            calibrate(thresholds=[0.5] * 22, fixture_path=self.fixture_path)
        self.assertIn("exceeds maximum allowed (21)", str(cm_max.exception))

    def test_evaluator_flag_magicmock_defaults_to_safe_full_mode(self):
        """MagicMock args without explicit calibration_only/require_quality default safely to full mode."""
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None
        # Note: calibration_only and require_quality are NOT set on args; accessing them returns a MagicMock

        report = evaluate(args)
        self.assertFalse(report["calibration_only"])
        self.assertEqual(report["run_scope"], "full")
        self.assertEqual(report["import"]["conversations"], 12)
        self.assertTrue(report["gates"]["structure_passed"])

    def test_evaluator_flag_explicit_callable_wrong_types_rejected(self):
        """Explicit callable flag calibration_only rejects non-bool wrong types (str, int, list)."""
        args = MagicMock()
        args.fixture = self.fixture_path
        args.ollama = False
        args.provider = None

        for invalid_flag in ("True", "False", 1, 0, [], {}):
            with self.assertRaises(TypeError, msg=f"Should reject calibration_only={invalid_flag!r}"):
                evaluate(args, calibration_only=invalid_flag)



if __name__ == "__main__":
    unittest.main()
