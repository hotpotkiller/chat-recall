"""Unit tests for synthetic Exact-versus-Smart relevance and memory evaluation."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from chat_recall.providers import ProviderError
from scripts.evaluate_smart import (
    DeterministicSynthetic16DProvider,
    evaluate_memory,
    evaluate_relevance,
    main,
)


class SmartEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "smart-evaluation.json"

    def test_fixture_structure_and_constraints(self):
        """Verify fixture declarations, bounds, topic counts, query counts, and Chinese cases."""
        self.assertTrue(self.fixture_path.exists(), "smart-evaluation.json fixture must exist")
        data = json.loads(self.fixture_path.read_text(encoding="utf-8"))

        self.assertTrue(data.get("synthetic"), "Fixture must explicitly declare synthetic")
        self.assertEqual(data.get("version"), "smart-evaluation-v2")

        conversations = data["conversations"]
        # 16-24 short synthetic conversations
        self.assertGreaterEqual(len(conversations), 16)
        self.assertLessEqual(len(conversations), 24)

        total_messages = 0
        conv_ids = set()
        for conv in conversations:
            conv_id = conv["id"]
            self.assertNotIn(conv_id, conv_ids, f"Duplicate conv id: {conv_id}")
            conv_ids.add(conv_id)
            messages = conv["messages"]
            self.assertGreater(len(messages), 0)
            total_messages += len(messages)
            # Question / answer dialogue with explicit parent_id
            for msg in messages:
                self.assertIn("id", msg)
                self.assertIn("role", msg)
                self.assertIn("text", msg)
                self.assertIn("parent_id", msg)

        # Bound dataset below 100 messages
        self.assertLess(total_messages, 100, "Dataset messages must be bounded below 100")

        queries = data["queries"]
        # Bound dataset below 100 queries
        self.assertLess(len(queries), 100, "Dataset queries must be bounded below 100")

        positive_queries = [q for q in queries if q["type"] == "positive"]
        unrelated_queries = [q for q in queries if q["type"] == "unrelated"]

        # At least 32 positive paraphrase queries + 8 unrelated queries
        self.assertGreaterEqual(len(positive_queries), 32)
        self.assertGreaterEqual(len(unrelated_queries), 8)

        # Positive queries must point to valid conversations
        for q in positive_queries:
            self.assertIn("expected_conversation_id", q)
            self.assertIn(q["expected_conversation_id"], conv_ids)

        # Include at least 8 Chinese / multilingual cases
        chinese_queries = [
            q for q in queries
            if any("\u4e00" <= char <= "\u9fff" for char in q["query"])
        ]
        self.assertGreaterEqual(len(chinese_queries), 8, "Must include at least 8 Chinese/multilingual queries")

        # Verify splits: calibration and holdout tags present
        calibration_queries = [q for q in queries if q.get("split") == "calibration"]
        holdout_queries = [q for q in queries if q.get("split") == "holdout"]
        self.assertEqual(len(calibration_queries), 48)
        self.assertEqual(len(holdout_queries), 16)
        self.assertEqual(len([q for q in holdout_queries if q["type"] == "positive"]), 8)
        self.assertEqual(len([q for q in holdout_queries if q["type"] == "unrelated"]), 8)

        # Verify NVMe SSD query does not claim mechanical noises
        disk_queries = [q for q in queries if "disk" in q.get("id", "")]
        for dq in disk_queries:
            self.assertNotIn("mechanical noises", dq["query"].lower())

    def test_default_lexical_baseline_no_network(self):
        """CLI default tests lexical baseline ONLY, no provider, no network, exit 0 on structure pass."""
        args = MagicMock()
        args.ollama = False
        args.model = "qwen3-embedding:0.6b"
        args.timeout = 30.0
        args.require_quality = False
        args.memory_only = False
        args.output = None

        report = evaluate_relevance(args)

        self.assertTrue(report["synthetic"])
        self.assertEqual(report["mode"], "lexical-baseline-only")
        self.assertFalse(report["semantic_evaluated"])
        self.assertIsNone(report["model_id"])
        self.assertIsNone(report["enrichment"])

        # Semantic used count should be 0 because no provider was configured
        self.assertEqual(report["metrics"]["semantic_used_count"], 0)
        self.assertEqual(report["metrics"]["positive_semantic_used_count"], 0)
        self.assertEqual(report["metrics"]["unrelated_semantic_used_count"], 0)
        self.assertEqual(report["metrics"]["context_identity_errors"], 0)
        self.assertEqual(len(report["query_errors"]), 0)
        self.assertEqual(len(report["semantic_failures"]), 0)

        # Structure passed gate must be True
        self.assertTrue(report["gates"]["structure_passed"])
        # Quality passed gate must be False when no semantic provider is executed
        self.assertFalse(report["gates"]["quality_passed"])

    def test_mock_provider_failure_lexical_good_ranks_cannot_pass_quality(self):
        """Regression test: when a provider fails at query time after genuine stored vectors,

        even if lexical ranking produces results, quality gate must fail with provider_error.
        """
        args = MagicMock()
        args.ollama = True
        args.model = "qwen3-embedding:0.6b"
        args.timeout = 30.0
        args.require_quality = True
        args.memory_only = False
        args.output = None

        class MockFailingProvider:
            model_id = "mock:failing-provider"

            def __init__(self):
                self.enrich_done = False

            def embed(self, texts):
                if not self.enrich_done:
                    # Provide genuine 2D vectors during index.enrich()
                    return [[1.0, 0.0] for _ in texts]
                # Fail whenever called for query embeddings during search
                raise ProviderError("Connection refused to local mock model daemon")

        mock_provider = MockFailingProvider()

        with patch("scripts.evaluate_smart.OllamaProvider", return_value=mock_provider):
            # Intercept enrich completion to flip provider state
            from chat_recall.search import SearchIndex
            original_enrich = SearchIndex.enrich

            def wrapped_enrich(self_index, *e_args, **e_kwargs):
                result = original_enrich(self_index, *e_args, **e_kwargs)
                mock_provider.enrich_done = True
                return result

            with patch.object(SearchIndex, "enrich", wrapped_enrich):
                report = evaluate_relevance(args)

                # Lexical results may still have been produced
                # But semantic_used must be 0, semantic_failures must record provider_error
                self.assertEqual(report["metrics"]["semantic_used_count"], 0)
                self.assertGreater(len(report["semantic_failures"]), 0)
                self.assertEqual(report["semantic_failures"][0]["status"], "provider_error")

                # Structure passed can still be True (no SQLite/context corruption)
                self.assertTrue(report["gates"]["structure_passed"])
                # Quality gate must strictly FAIL because semantic was not used
                self.assertFalse(report["gates"]["quality_passed"])

    def test_deterministic_16d_provider(self):
        """Verify the deterministic 16D provider math and normalization."""
        provider = DeterministicSynthetic16DProvider()
        self.assertEqual(provider.model_id, "synthetic:16d-deterministic-v1")
        vecs = provider.embed(["test one", "test two", "test one"])
        self.assertEqual(len(vecs), 3)
        self.assertEqual(len(vecs[0]), 16)
        # Deterministic check
        self.assertEqual(vecs[0], vecs[2])
        # Unit norm check
        norm = sum(x * x for x in vecs[0])
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_memory_only_evaluation(self):
        """Verify --memory-only mode runs tracemalloc on query without error."""
        report = evaluate_memory(query_count=1)
        self.assertEqual(report["mode"], "memory-only")
        self.assertEqual(report["benchmark"], "synthetic-memory-v1")
        self.assertTrue(report["synthetic"])
        self.assertEqual(report["corpus"]["messages"], 2000)
        self.assertGreater(report["query_benchmark"]["python_peak_allocated_bytes"], 0)
        self.assertIn("result_counts", report["query_benchmark"])

    def test_recording_provider_wrapper_forwards_max_input_bytes(self):
        from scripts.evaluate_smart import RecordingProviderWrapper
        class DummyProviderWithBudget:
            model_id = "test:budget"
            max_input_bytes = 960
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        class DummyProviderNoBudget:
            model_id = "test:no-budget"
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        w_budget = RecordingProviderWrapper(DummyProviderWithBudget())
        self.assertEqual(w_budget.max_input_bytes, 960)

        w_no_budget = RecordingProviderWrapper(DummyProviderNoBudget())
        self.assertIsNone(w_no_budget.max_input_bytes)

    def test_recording_provider_wrapper_forwards_minimum_similarity(self):
        from scripts.evaluate_smart import RecordingProviderWrapper
        class CustomProvider:
            model_id = "test:custom"
            minimum_similarity = 0.55
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        class DefaultProvider:
            model_id = "test:default"
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        w_custom = RecordingProviderWrapper(CustomProvider())
        self.assertEqual(w_custom.minimum_similarity, 0.55)

        w_default = RecordingProviderWrapper(DefaultProvider())
        self.assertEqual(w_default.minimum_similarity, 0.3)

    def test_recording_provider_wrapper_forwards_encoding_identity_and_methods(self):
        from scripts.evaluate_smart import RecordingProviderWrapper

        class FullSpecializedProvider:
            model_id = "test:specialized"
            expected_output_dimension = 1024
            cached_identity = "test:specialized:fingerprint123"
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
        self.assertEqual(wrapper.resolve_identity({"version": "v1"}), "test:specialized:fingerprint123")
        self.assertEqual(wrapper.index_id, "test:specialized:fingerprint123")
        self.assertEqual(wrapper.cached_identity, "test:specialized:fingerprint123")
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
        q_vec = wrapper.embed_query("search query")
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

    def test_default_lexical_split_results_and_no_fake_quality(self):
        args = MagicMock()
        args.ollama = False
        args.model = "qwen3-embedding:0.6b"
        args.timeout = 30.0
        args.require_quality = False
        args.memory_only = False
        args.output = None

        report = evaluate_relevance(args)
        self.assertIn("split_results", report["metrics"])
        self.assertIn("calibration", report["metrics"]["split_results"])
        self.assertIn("holdout", report["metrics"]["split_results"])

        holdout_metrics = report["metrics"]["split_results"]["holdout"]
        self.assertEqual(holdout_metrics["positive_count"], 8)
        self.assertEqual(holdout_metrics["unrelated_count"], 8)
        self.assertEqual(holdout_metrics["positive_semantic_used_count"], 0)
        self.assertEqual(holdout_metrics["unrelated_semantic_used_count"], 0)

        # Lexical baseline must never fake semantic quality or pass quality gates
        self.assertFalse(report["semantic_evaluated"])
        self.assertFalse(report["gates"]["quality_passed"])
        self.assertFalse(report["gates"]["holdout_passed"])

    def test_holdout_gate_integrity_on_failed_holdout_query(self):
        """Regression test: when one holdout query fails/throws, holdout_passed and quality_passed must be FALSE.

        Simulates configured provider where positive queries return their expected conversation with
        semantic.used=True, unrelated return empty with semantic.used=True, BUT exactly one holdout query
        throws. The completed holdout positive count < expected, so holdout_passed and quality_passed must fail.
        """
        args = MagicMock()
        args.ollama = True
        args.model = "qwen3-embedding:0.6b"
        args.timeout = 30.0
        args.require_quality = False
        args.memory_only = False
        args.output = None

        class MockConfiguredProvider:
            model_id = "mock:configured-provider"
            minimum_similarity = 0.3

            def embed(self, texts):
                # Return normalized 2D vectors
                return [[1.0, 0.0] for _ in texts]

        mock_provider = MockConfiguredProvider()

        # Load fixture to know expected conversation mappings and message IDs
        suite = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        first_msg_map = {conv["id"]: conv["messages"][0]["id"] for conv in suite["conversations"]}
        positive_target_map = {
            q["query"]: q["expected_conversation_id"]
            for q in suite["queries"]
            if q["type"] == "positive"
        }
        holdout_queries = [q for q in suite["queries"] if q.get("split") == "holdout"]
        failing_holdout_q = [q for q in holdout_queries if q["type"] == "positive"][0]["query"]

        with patch("scripts.evaluate_smart.OllamaProvider", return_value=mock_provider):
            from chat_recall.search import SearchIndex
            original_search = SearchIndex.search

            def stub_search(self_index, query, mode="exact", limit=50, **kwargs):
                if mode == "exact":
                    return original_search(self_index, query, mode=mode, limit=limit, **kwargs)
                if mode == "smart":
                    if query == failing_holdout_q:
                        raise RuntimeError("Simulated transient query failure on holdout query")
                    if query in positive_target_map:
                        target_cid = positive_target_map[query]
                        first_mid = first_msg_map[target_cid]
                        return {
                            "query": query,
                            "mode": "smart",
                            "total": 1,
                            "results": [{
                                "conversation_id": target_cid,
                                "message_id": first_mid,
                                "match_type": "semantic",
                                "score": 1.0,
                            }],
                            "semantic": {
                                "used": True,
                                "status": "ready",
                                "minimum_similarity": 0.3,
                            },
                        }
                    # Unrelated query: no results, semantic used
                    return {
                        "query": query,
                        "mode": "smart",
                        "total": 0,
                        "results": [],
                        "semantic": {
                            "used": True,
                            "status": "ready",
                            "minimum_similarity": 0.3,
                        },
                    }
                return original_search(self_index, query, mode=mode, limit=limit, **kwargs)

            with patch.object(SearchIndex, "search", stub_search):
                report = evaluate_relevance(args)

                # Exactly one query failed
                self.assertEqual(len(report["query_errors"]), 1)
                self.assertEqual(report["query_errors"][0]["query"], failing_holdout_q)

                # Structure gate must fail due to query error
                self.assertFalse(report["gates"]["structure_passed"])

                # Holdout positive count must be 7 instead of 8 (less than expected)
                holdout_metrics = report["metrics"]["split_results"]["holdout"]
                self.assertEqual(holdout_metrics["positive_count"], 7)

                # Gates must be FALSE
                self.assertFalse(report["gates"]["holdout_passed"])
                self.assertFalse(report["gates"]["quality_passed"])

    def test_cli_argument_parsing_and_exit_codes(self):
        """Test argument validation: --require-quality requires --ollama."""
        with patch("sys.argv", ["scripts/evaluate_smart.py", "--require-quality"]), \
             patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertNotEqual(cm.exception.code, 0)

    def test_cli_default_run_succeeds(self):
        """Test default invocation completes with exit code 0 and silenced stdout."""
        with patch("sys.argv", ["scripts/evaluate_smart.py"]), \
             patch("sys.stdout", new_callable=io.StringIO):
            code = main()
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
