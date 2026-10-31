import json
from pathlib import Path
import random
import re
import sqlite3
import tempfile
import threading
import time
import tracemalloc
import unittest
from unittest.mock import patch, MagicMock

from chat_recall.providers import ProviderError, normalized_vector
from chat_recall.search import SearchIndex, literal_matches, cjk_lexical_terms, cjk_index_tokens


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "demo.json"


class ToyProvider:
    """Deterministic wiring fixture, not evidence of real semantic quality."""
    model_id = "synthetic:semantic-fixture-v1"

    def embed(self, texts):
        return [[1.0, 0.0] if any(w in text.lower() for w in ("soup", "tomato", "acidity", "sour")) else [0.0, 1.0] for text in texts]

    def summarize(self, text):
        return "Reducing tomato soup acidity" if "tomato" in text.lower() else "A saved dialogue"


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.index = SearchIndex(Path(self.temp.name) / "index.sqlite")
        self.payload = json.loads(FIXTURE.read_text())
        self.index.import_payload(self.payload)

    def tearDown(self):
        self.index.close()
        self.temp.cleanup()

    def test_casefold_original_offsets_and_overlap(self):
        text = "😀 Straße STRASSE straße"
        spans = literal_matches(text, "strasse")
        self.assertEqual([text[a:b] for a, b in spans], ["Straße", "STRASSE", "straße"])
        self.assertEqual(literal_matches("banana", "ana"), [[1, 4], [3, 6]])
        self.assertEqual(literal_matches("中文 中文", "中"), [[0, 1], [3, 4]])
        self.assertEqual(literal_matches("ß", "s"), [[0, 1]])

    def test_punctuation_short_unicode_and_original_html(self):
        for query in ("config['api_key']", "ERR_CFG_204", "😀", "中", "<script>"):
            hits = self.index.search(query)
            self.assertGreater(hits["total"], 0, query)
            for row in hits["results"]:
                self.assertTrue(all(query.casefold() in row["text"][a:b].casefold() for a, b in row["matches"]))
        self.assertEqual(self.index.search("internal setup")["total"], 0)

    def test_exact_pagination_and_case_sensitive_filter(self):
        all_results = self.index.search("the", limit=200)
        first = self.index.search("the", limit=2)
        second = self.index.search("the", limit=2, offset=2)
        self.assertEqual(first["results"] + second["results"], all_results["results"][:4])
        self.assertEqual(first["total"], all_results["total"])
        self.assertEqual(self.index.search("STRASSE", case_sensitive=True)["results"][0]["match_count"], 1)
        self.assertEqual(self.index.search("strasse")["results"][0]["match_count"], 3)
        self.assertTrue(all(r["role"] == "user" for r in self.index.search("the", role="user")["results"]))

    def test_idempotent_partial_import_and_account_separation(self):
        before = self.index.status()["messages"]
        report = self.index.import_payload(self.payload)
        self.assertFalse(report["errors"])
        self.assertEqual(self.index.status()["messages"], before)
        self.index.import_payload({"id": "synthetic-code", "title": "Renamed", "messages": [
            {"id": "new", "parent_id": "code-a", "role": "user", "text": "An additional reply"}]})
        self.assertEqual(self.index.search("ERR_CFG_204")["total"], 1)
        self.index.import_payload(self.payload, account_id="another")
        self.assertEqual(self.index.search("ERR_CFG_204")["total"], 2)
        self.assertEqual(self.index.search("ERR_CFG_204", account_id="default")["total"], 1)
        with self.assertRaises(KeyError):
            self.index.context("synthetic-code", "new", account_id="another")

    def test_branch_context_never_includes_sibling_answer(self):
        chosen = self.index.context("synthetic-cipher", "cipher-a")
        alternate = self.index.context("synthetic-cipher", "cipher-alt")
        self.assertEqual([m["id"] for m in chosen["messages"]], ["cipher-q", "cipher-a"])
        self.assertEqual([m["id"] for m in alternate["messages"]], ["cipher-q", "cipher-alt"])
        parent = self.index.context("synthetic-cipher", "cipher-q")
        self.assertEqual([m["id"] for m in parent["messages"]], ["cipher-q", "cipher-a"])

    def test_mapping_node_and_message_ids_can_differ(self):
        first = {"id": "different", "mapping": {
            "n1": {"parent": None, "message": {"id": "m1", "role": "user", "text": "Question"}},
            "hidden": {"parent": "n1", "message": None}}}
        second = {"id": "different", "mapping": {
            "n2": {"parent": "hidden", "message": {"id": "m2", "role": "assistant", "text": "Answer"}}}}
        self.index.import_payload(first)
        self.index.import_payload(second)
        result = self.index.context("different", "m2")
        self.assertEqual([m["id"] for m in result["messages"]], ["m1", "m2"])

    def test_partial_child_before_parent_reconciles_native_message_ids(self):
        child = {"id": "out-of-order", "mapping": {
            "n2": {"parent": "n1", "message": {"id": "m2", "role": "assistant", "text": "Child received first"}}}}
        parent = {"id": "out-of-order", "mapping": {
            "n1": {"parent": None, "message": {"id": "m1", "role": "user", "text": "Parent received later"}}}}
        self.index.import_payload(child)
        self.index.enrich(ToyProvider())
        self.index.import_payload(parent)
        context = self.index.context("out-of-order", "m2")
        self.assertEqual([m["id"] for m in context["messages"]], ["m1", "m2"])
        self.assertEqual(context["messages"][1]["parent_id"], "m1")
        count = self.index.connection.execute("SELECT COUNT(*) FROM enrichment e JOIN messages m ON m.id=e.message_row WHERE m.conversation_id='out-of-order'").fetchone()[0]
        self.assertEqual(count, 0)

    def test_explicit_hidden_update_revokes_every_search_channel(self):
        visible = {"id": "visibility", "mapping": {
            "n1": {"parent": None, "message": {"id": "m1", "role": "assistant", "text": "formerly visible secretphrase"}},
            "n2": {"parent": "n1", "message": {"id": "m2", "role": "user", "text": "A remaining visible child"}}}}
        hidden = {"id": "visibility", "mapping": {
            "n1": {"parent": None, "message": {"id": "m1", "role": "assistant", "metadata": {"is_visually_hidden_from_conversation": True}, "content": {"content_type": "text", "parts": ["formerly visible secretphrase"]}}}}}
        self.index.import_payload(visible)
        self.index.embedding_provider = ToyProvider()
        self.index.enrich()
        self.assertEqual(self.index.search("secretphrase")["total"], 1)
        self.index.import_payload(hidden)
        self.assertEqual(self.index.search("secretphrase")["total"], 0)
        self.assertFalse(any(r["message_id"] == "m1" and r["conversation_id"] == "visibility"
                             for r in self.index.search("secretphrase", mode="smart")["results"]))
        for table in ("message_fts", "message_tri"):
            self.assertEqual(self.index.connection.execute(f"SELECT COUNT(*) FROM {table} WHERE text LIKE '%secretphrase%'").fetchone()[0], 0)
        with self.assertRaises(KeyError):
            self.index.context("visibility", "m1")
        remaining = self.index.context("visibility", "m2")
        self.assertEqual([m["id"] for m in remaining["messages"]], ["m2"])
        # An absent visible message in a later partial import is retained.
        self.index.import_payload({"id": "visibility", "mapping": {"placeholder": {"parent": None, "message": None}}})
        self.assertEqual(self.index.search("remaining visible child")["total"], 1)

    def test_null_placeholder_preserves_existing_visible_node_and_ancestry(self):
        initial = {"id": "sparse-placeholder", "mapping": {
            "n0": {"parent": None, "message": {"id": "m0", "role": "user", "text": "Earlier context"}},
            "n1": {"parent": "n0", "message": {"id": "m1", "role": "assistant", "text": "Parent answer"}},
            "n2": {"parent": "n1", "message": {"id": "m2", "role": "user", "text": "Child follow-up"}}}}
        self.index.import_payload(initial)
        expected = self.index.context("sparse-placeholder", "m2")
        self.index.import_payload({"id": "sparse-placeholder", "mapping": {
            "n1": {"parent": None, "message": None}}})
        self.assertEqual(self.index.context("sparse-placeholder", "m2")["messages"], expected["messages"])
        self.assertEqual(self.index.search("Parent answer")["total"], 1)

    def test_malformed_conversation_rolls_back_without_losing_good_imports(self):
        payload = [
            {"id": "bad", "title": "Invalid", "messages": [
                {"id": "a", "role": "user", "text": "A", "parent_id": "b"},
                {"id": "b", "role": "assistant", "text": "B", "parent_id": "a"}]},
            {"id": "good", "title": "Valid", "messages": [{"id": "a", "role": "user", "text": "unique valid phrase"}]},
            None,
        ]
        report = self.index.import_payload(payload)
        self.assertEqual(report["conversations"], 1)
        self.assertEqual(len(report["errors"]), 2)
        self.assertEqual(self.index.search("unique valid phrase")["total"], 1)

    def test_smart_lexical_fallback_is_explicit(self):
        response = self.index.search("tomatoes acidity", mode="smart")
        self.assertEqual(response["results"][0]["conversation_id"], "synthetic-cooking")
        self.assertFalse(response["semantic"]["used"])
        self.assertEqual(response["semantic"]["relevance_policy"], "meaning-evidence-v1")
        self.assertEqual(response["semantic"]["rejected_keyword_only_count"], 0)
        self.assertFalse(any(r["match_type"] in {"semantic", "hybrid"} for r in response["results"]))
        self.assertEqual(self.index.search('" OR DROP TABLE messages; --', mode="smart")["semantic"]["status"], "not_configured")

    def test_embeddings_topics_are_derived_and_invalidated_by_ancestor_edits(self):
        provider = ToyProvider()
        self.index.embedding_provider = provider
        report = self.index.enrich(limit=100, generate_topics=True)
        self.assertFalse(report["errors"])
        self.assertEqual(report["processed"], self.index.status()["messages"])
        self.assertEqual(self.index.enrich()["processed"], 0)
        result = self.index.search("reduce sourness", mode="smart")
        self.assertTrue(result["semantic"]["used"])
        self.assertEqual(result["results"][0]["conversation_id"], "synthetic-cooking")
        self.assertEqual(self.index.search("Reducing tomato soup acidity")["total"], 0)
        self.index.import_payload({"id": "synthetic-cooking", "title": "Lunch experiments", "messages": [
            {"id": "cook-q", "role": "user", "text": "Updated prompt"}]})
        pending = self.index.enrich(limit=100)
        self.assertEqual(pending["processed"], 2)

    def test_long_messages_have_tail_semantic_chunks(self):
        self.index.import_payload({"id": "long", "messages": [{"id": "long", "role": "user", "text": "x" * 6500 + " tomato soup"}]})
        self.index.embedding_provider = ToyProvider()
        self.index.enrich(limit=100)
        result = self.index.search("reduce sourness", mode="smart")
        self.assertIn("long", [r["conversation_id"] for r in result["results"]])
        semantic_hit = next(r for r in result["results"] if r["conversation_id"] == "long")
        self.assertEqual(semantic_hit["focus_offset"], 5200)
        self.assertIn("tomato soup", semantic_hit["text"][semantic_hit["focus_offset"]:])
        literal_hit = self.index.search("tomato soup")["results"]
        long_literal = next(r for r in literal_hit if r["conversation_id"] == "long")
        self.assertEqual(long_literal["focus_offset"], long_literal["matches"][0][0])
        count = self.index.connection.execute("SELECT COUNT(*) FROM vectors v JOIN messages m ON m.id=v.message_row WHERE m.conversation_id='long'").fetchone()[0]
        self.assertEqual(count, 3)

    def test_partial_index_and_source_change_during_inference_are_honest(self):
        self.index.embedding_provider = ToyProvider()
        self.index.enrich(limit=1)
        partial = self.index.search("routing", mode="smart")
        self.assertEqual(partial["semantic"]["status"], "partial")
        self.assertGreater(partial["semantic"]["pending_messages"], 0)
        self.index.import_payload({"id": "race", "messages": [
            {"id": "r1", "role": "user", "text": "old immutable input"}]}, account_id="race")
        provider = ToyProvider()
        original = provider.embed

        def update_during_inference(texts):
            with SearchIndex(Path(self.temp.name) / "index.sqlite") as writer:
                writer.import_payload({"id": "race", "messages": [
                    {"id": "r1", "role": "user", "text": "newer input from concurrent import"}]}, account_id="race")
            return original(texts)

        provider.embed = update_during_inference
        report = self.index.enrich(provider, account_id="race")
        self.assertEqual(report["processed"], 0)
        self.assertEqual(report["remaining"], 1)
        self.assertEqual(self.index.connection.execute("SELECT COUNT(*) FROM vectors v JOIN messages m ON m.id=v.message_row WHERE m.account_id='race'").fetchone()[0], 0)

    def test_provider_failure_preserves_literal_and_lexical_search(self):
        provider = ToyProvider()
        self.index.enrich(provider)
        provider.embed = lambda _: (_ for _ in ()).throw(ProviderError("fixture offline"))
        self.index.embedding_provider = provider
        response = self.index.search("tomatoes", mode="smart")
        self.assertEqual(response["semantic"]["status"], "provider_error")
        self.assertFalse(response["semantic"]["used"])
        self.assertGreater(response["total"], 0)
        self.assertGreater(self.index.search("tomatoes")["total"], 0)

    def test_literal_recall_against_independent_original_text_oracle(self):
        rng = random.Random(8021)
        alphabet = "abc中😀.-_"
        messages = [{"id": str(i), "role": "user", "text": "".join(rng.choice(alphabet) for _ in range(150))} for i in range(70)]
        self.index.import_payload({"id": "oracle", "messages": messages}, account_id="oracle")
        queries = ["a", "中", "😀", "ab", ".-", "b中", "___", "not-present", "abc", "中😀a", "b.-", "a中."]
        for query in queries:
            expected = {m["id"]: [[i, i + len(query)] for i in range(len(m["text"])) if m["text"].startswith(query, i)] for m in messages}
            expected = {key: value for key, value in expected.items() if value}
            actual = self.index.search(query, case_sensitive=True, limit=200, account_id="oracle")
            self.assertEqual({r["message_id"]: r["matches"] for r in actual["results"]}, expected)

    def test_trigram_literal_quoting_whitespace_nul_and_idempotent_updates(self):
        text = 'First "AND" then\nthree spaces   here. NUL:\x00after-the-nul'
        self.index.import_payload({"id": "tri", "messages": [{"id": "tri", "role": "user", "text": text}]})
        for query in ('"AND"', 'then\nthree', 'spaces   here', 'after-the-nul', '\x00after'):
            self.assertEqual(self.index.search(query)["total"], 1, query)
        self.index.import_payload({"id": "tri", "messages": [{"id": "tri", "role": "user", "text": "entirely replaced"}]})
        self.assertEqual(self.index.search("after-the-nul")["total"], 0)
        self.assertEqual(self.index.search("entirely replaced")["total"], 1)

    def test_nul_literal_portability_and_simulated_fts_omission(self):
        # Deterministic regression simulating an FTS/trigram index which omits NUL rows
        # or truncates indexing at embedded NUL.
        text_with_nul = "prefix-start\x00suffix-target-value and extra text"
        payload_1 = {
            "id": "nul-conv-1",
            "messages": [
                {"id": "msg-nul-user", "role": "user", "text": text_with_nul},
                {"id": "msg-nul-asst", "role": "assistant", "text": "assistant reply\x00suffix-target-value"}
            ]
        }
        payload_other_acc = {
            "id": "nul-conv-2",
            "messages": [
                {"id": "msg-other", "role": "user", "text": "other-acc\x00suffix-target-value"}
            ]
        }
        self.index.import_payload(payload_1, account_id="acc1")
        self.index.import_payload(payload_other_acc, account_id="acc2")

        # Simulate FTS truncating/omitting NUL rows by removing them from message_tri
        self.index.connection.execute(
            "DELETE FROM message_tri WHERE rowid IN (SELECT id FROM messages WHERE instr(CAST(text AS BLOB), X'00') > 0)"
        )
        self.index.connection.commit()

        # Verify post-NUL exact recall and exact offsets
        hits = self.index.search("suffix-target-value", account_id="acc1")
        self.assertEqual(hits["total"], 2)
        user_hit = next(r for r in hits["results"] if r["message_id"] == "msg-nul-user")
        self.assertEqual(user_hit["matches"], [[13, 32]])
        self.assertEqual(user_hit["text"][13:32], "suffix-target-value")
        self.assertEqual(user_hit["focus_offset"], 13)

        # Verify role isolation
        user_only = self.index.search("suffix-target-value", account_id="acc1", role="user")
        self.assertEqual(user_only["total"], 1)
        self.assertEqual(user_only["results"][0]["message_id"], "msg-nul-user")

        asst_only = self.index.search("suffix-target-value", account_id="acc1", role="assistant")
        self.assertEqual(asst_only["total"], 1)
        self.assertEqual(asst_only["results"][0]["message_id"], "msg-nul-asst")

        # Verify namespace / account_id isolation
        acc2_hits = self.index.search("suffix-target-value", account_id="acc2")
        self.assertEqual(acc2_hits["total"], 1)
        self.assertEqual(acc2_hits["results"][0]["message_id"], "msg-other")

        # Verify updates: updating the text removes old NUL text from recall
        self.index.import_payload({
            "id": "nul-conv-1",
            "messages": [
                {"id": "msg-nul-user", "role": "user", "text": "updated clean text without keyword"}
            ]
        }, account_id="acc1")
        hits_after_update = self.index.search("suffix-target-value", account_id="acc1", role="user")
        self.assertEqual(hits_after_update["total"], 0)

    def test_input_validation(self):
        for kwargs in ({"mode": "imaginary"}, {"limit": 0}, {"offset": -1}, {"role": "system"}):
            with self.assertRaises(ValueError):
                self.index.search("a", **kwargs)
        self.assertEqual(self.index.search(" ")["total"], 0)

    def test_late_page_high_frequency_unicode_casefold_overlap_and_filtering(self):
        messages = []
        for i in range(20):
            messages.append({"id": f"m-{i:02d}", "role": "user", "text": f"Entry {i:02d} | ßß ss SS | tail"})
        for i in range(20, 25):
            messages.append({"id": f"m-{i:02d}", "role": "user", "text": f"Entry {i:02d} | ßß SS only | tail"})
        for i in range(25, 30):
            messages.append({"id": f"m-{i:02d}", "role": "user", "text": f"Entry {i:02d} | distractor no hits | tail"})
        self.index.import_payload({"id": "late-page", "messages": messages}, account_id="late-page-acc")

        # Case-insensitive late-page pagination across 25 total hits (offset 20, limit 5)
        late_page = self.index.search("ss", limit=5, offset=20, account_id="late-page-acc")
        self.assertEqual(late_page["total"], 25)
        self.assertEqual([r["message_id"] for r in late_page["results"]], [f"m-{i:02d}" for i in range(20, 25)])
        for r in late_page["results"]:
            self.assertEqual(r["match_count"], 4)
            self.assertEqual(r["matches"], [[11, 12], [11, 13], [12, 13], [14, 16]])
            self.assertEqual(r["focus_offset"], 11)
            self.assertEqual(r["score"], 1.0)
            self.assertEqual(r["match_type"], "exact")

        # Case-insensitive first page (offset 0, limit 2) with fixed spans on high-frequency overlap
        first_page = self.index.search("ss", limit=2, offset=0, account_id="late-page-acc")
        self.assertEqual(first_page["total"], 25)
        self.assertEqual([r["message_id"] for r in first_page["results"]], ["m-00", "m-01"])
        for r in first_page["results"]:
            self.assertEqual(r["match_count"], 5)
            self.assertEqual(r["matches"], [[11, 12], [11, 13], [12, 13], [14, 16], [17, 19]])
            self.assertEqual(r["focus_offset"], 11)

        # Case-sensitive filtering: m-20..m-24 lack lowercase 'ss' so total is 20
        cs_late = self.index.search("ss", case_sensitive=True, limit=5, offset=15, account_id="late-page-acc")
        self.assertEqual(cs_late["total"], 20)
        self.assertEqual([r["message_id"] for r in cs_late["results"]], [f"m-{i:02d}" for i in range(15, 20)])
        for r in cs_late["results"]:
            self.assertEqual(r["match_count"], 1)
            self.assertEqual(r["matches"], [[14, 16]])
            self.assertEqual(r["focus_offset"], 14)

        # Case-sensitive pagination beyond available hits returns empty results with correct total
        cs_past = self.index.search("ss", case_sensitive=True, limit=5, offset=20, account_id="late-page-acc")
        self.assertEqual(cs_past["total"], 20)
        self.assertEqual(cs_past["results"], [])

        # Case-sensitive uppercase query matches all 25 candidates having 'SS'
        cs_upper = self.index.search("SS", case_sensitive=True, limit=5, offset=20, account_id="late-page-acc")
        self.assertEqual(cs_upper["total"], 25)
        self.assertEqual([r["message_id"] for r in cs_upper["results"]], [f"m-{i:02d}" for i in range(20, 25)])
        for r in cs_upper["results"]:
            self.assertEqual(r["match_count"], 1)
            self.assertEqual(r["matches"], [[14, 16]])
            self.assertEqual(r["focus_offset"], 14)

        # Case-sensitive query for German sharp S 'ßß'
        cs_sharp = self.index.search("ßß", case_sensitive=True, limit=5, offset=20, account_id="late-page-acc")
        self.assertEqual(cs_sharp["total"], 25)
        self.assertEqual([r["message_id"] for r in cs_sharp["results"]], [f"m-{i:02d}" for i in range(20, 25)])
        for r in cs_sharp["results"]:
            self.assertEqual(r["match_count"], 1)
            self.assertEqual(r["matches"], [[11, 13]])
            self.assertEqual(r["focus_offset"], 11)

    def _seed_vector(self, message_id: str, channel: str, chunk: int,
                      model_id: str, start_offset: int, end_offset: int,
                      vector: list[float], derived_text: str = "derived",
                      account_id: str = "default"):
        row = self.index.connection.execute(
            "SELECT id, content_hash FROM messages WHERE message_id=? AND account_id=?",
            (message_id, account_id)
        ).fetchone()
        if row is None:
            raise ValueError(f"Message {message_id} not found in account {account_id}")
        norm = normalized_vector(vector)
        self.index.connection.execute(
            """INSERT INTO vectors (message_row, channel, chunk, model_id, content_hash,
                                   start_offset, end_offset, derived_text, vector_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(message_row, channel, chunk, model_id) DO UPDATE SET
               vector_json=excluded.vector_json, start_offset=excluded.start_offset,
               end_offset=excluded.end_offset, content_hash=excluded.content_hash""",
            (row["id"], channel, chunk, model_id, row["content_hash"],
             start_offset, end_offset, derived_text, json.dumps(norm))
        )
        self.index.connection.execute(
            """INSERT INTO enrichment (message_row, model_id, content_hash, topics)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(message_row, model_id) DO UPDATE SET
               content_hash=excluded.content_hash,
               topics=MAX(enrichment.topics, excluded.topics)""",
            (row["id"], model_id, row["content_hash"], 1 if channel == "topic" else 0)
        )
        self.index.connection.commit()

    def test_smart_case_sensitive_uppercase_and_unicode(self):
        payload = {
            "id": "case-smart-conv",
            "messages": [
                {"id": "msg-upper", "role": "user", "text": "Testing UPPERCASE token here."},
                {"id": "msg-lower", "role": "user", "text": "Testing uppercase token here."},
                {"id": "msg-german", "role": "user", "text": "Grüße und Straße in München."},
                {"id": "msg-german-lower", "role": "user", "text": "Grüße und straße in München."}
            ]
        }
        self.index.import_payload(payload, account_id="case-acc")
        # Case sensitive Smart query on uppercase:
        # msg-upper has literal matches and 'literal' channel
        # msg-lower may match lexically (keyword channel) but must NOT have literal channel or literal matches
        cs_res = self.index.search("UPPERCASE", mode="smart", case_sensitive=True, account_id="case-acc")
        by_id = {r["message_id"]: r for r in cs_res["results"]}
        self.assertIn("msg-upper", by_id)
        self.assertIn("literal", by_id["msg-upper"]["retrieval_channels"])
        self.assertEqual(by_id["msg-upper"]["matches"], [[8, 17]])
        if "msg-lower" in by_id:
            self.assertNotIn("literal", by_id["msg-lower"]["retrieval_channels"])
            self.assertEqual(by_id["msg-lower"]["matches"], [])

        # Case sensitive Smart query on German sharp S / umlaut 'Straße' vs 'straße'
        de_res = self.index.search("Straße", mode="smart", case_sensitive=True, account_id="case-acc")
        de_by_id = {r["message_id"]: r for r in de_res["results"]}
        self.assertIn("msg-german", de_by_id)
        self.assertIn("literal", de_by_id["msg-german"]["retrieval_channels"])
        self.assertEqual(de_by_id["msg-german"]["matches"], [[10, 16]])
        if "msg-german-lower" in de_by_id:
            self.assertNotIn("literal", de_by_id["msg-german-lower"]["retrieval_channels"])
            self.assertEqual(de_by_id["msg-german-lower"]["matches"], [])

    def test_smart_paraphrase_without_shared_words(self):
        class StaticQueryProvider:
            model_id = "synthetic:static-provider"
            def __init__(self, q_vec):
                self.q_vec = q_vec
            def embed(self, texts):
                return [self.q_vec for _ in texts]

        payload = {
            "id": "paraphrase-conv",
            "messages": [
                {"id": "msg-target", "role": "assistant", "text": "A friendly hound was barking outdoors."},
                {"id": "msg-other", "role": "assistant", "text": "Completely unrelated subject matter here."}
            ]
        }
        self.index.import_payload(payload, account_id="para-acc")
        # Exact search for 'canine' yields 0 hits because 'canine' is not in text
        exact_res = self.index.search("canine", mode="exact", account_id="para-acc")
        self.assertEqual(exact_res["total"], 0)

        # Seed target with [1.0, 0.0] and other with orthogonal [0.0, 1.0]
        self._seed_vector("msg-target", "dialogue", 0, "synthetic:static-provider", 0, 42, [1.0, 0.0], account_id="para-acc")
        self._seed_vector("msg-other", "dialogue", 0, "synthetic:static-provider", 0, 42, [0.0, 1.0], account_id="para-acc")

        provider = StaticQueryProvider([1.0, 0.0])
        self.index.embedding_provider = provider
        smart_res = self.index.search("canine", mode="smart", account_id="para-acc")
        self.assertEqual(smart_res["semantic"]["minimum_similarity"], 0.3)
        self.assertTrue(smart_res["semantic"]["used"])
        self.assertEqual(smart_res["total"], 1)
        hit = smart_res["results"][0]
        self.assertEqual(hit["message_id"], "msg-target")
        self.assertEqual(hit["match_type"], "semantic")
        self.assertEqual(hit["retrieval_channels"], ["dialogue"])
        self.assertAlmostEqual(hit["semantic_similarity"], 1.0, places=4)
        self.assertEqual(hit["semantic_evidence"], {"channel": "dialogue", "start_offset": 0, "end_offset": 42})

    def test_smart_semantic_gating_excludes_zero_and_negative(self):
        class StaticQueryProvider:
            model_id = "synthetic:gate-provider"
            def __init__(self, q_vec):
                self.q_vec = q_vec
            def embed(self, texts):
                return [self.q_vec for _ in texts]

        payload = {
            "id": "gate-conv",
            "messages": [
                {"id": "msg-pos", "role": "user", "text": "Positive candidate message text."},
                {"id": "msg-low", "role": "user", "text": "Low similarity candidate message text."},
                {"id": "msg-ortho", "role": "user", "text": "Orthogonal candidate message text."},
                {"id": "msg-neg", "role": "user", "text": "Negative candidate message text."}
            ]
        }
        self.index.import_payload(payload, account_id="gate-acc")
        # Query vector is [1.0, 0.0]
        # pos has [1.0, 0.0] -> sim 1.0 >= 0.3 (included)
        # low has [0.2, 0.9797958971] -> sim 0.2 < 0.3 (excluded)
        # ortho has [0.0, 1.0] -> sim 0.0 < 0.3 (excluded)
        # neg has [-1.0, 0.0] -> sim -1.0 < 0.3 (excluded)
        self._seed_vector("msg-pos", "dialogue", 0, "synthetic:gate-provider", 0, 31, [1.0, 0.0], account_id="gate-acc")
        self._seed_vector("msg-low", "dialogue", 0, "synthetic:gate-provider", 0, 37, [0.2, 0.9797958971], account_id="gate-acc")
        self._seed_vector("msg-ortho", "dialogue", 0, "synthetic:gate-provider", 0, 36, [0.0, 1.0], account_id="gate-acc")
        self._seed_vector("msg-neg", "dialogue", 0, "synthetic:gate-provider", 0, 34, [-1.0, 0.0], account_id="gate-acc")

        self.index.embedding_provider = StaticQueryProvider([1.0, 0.0])
        res = self.index.search("unmatchedquery", mode="smart", account_id="gate-acc")
        self.assertTrue(res["semantic"]["used"])
        self.assertEqual(res["semantic"]["minimum_similarity"], 0.3)
        self.assertEqual(res["total"], 1)
        self.assertEqual(res["results"][0]["message_id"], "msg-pos")

        # When all candidates are below threshold, empty result with semantic.used=True
        self.index.embedding_provider = StaticQueryProvider([0.0, 1.0])
        # [0.0, 1.0] dot pos [1.0, 0] = 0.0; dot low = 0.9798; but if we query with [0.0, -1.0]:
        self.index.embedding_provider = StaticQueryProvider([0.0, -1.0])
        # dot pos = 0.0, dot low = -0.9798, dot ortho = -1.0, dot neg = 0.0 -> all <= 0.0 < 0.3
        res_empty = self.index.search("unmatchedquery", mode="smart", account_id="gate-acc")
        self.assertTrue(res_empty["semantic"]["used"])
        self.assertEqual(res_empty["total"], 0)
        self.assertEqual(res_empty["results"], [])

    def test_smart_late_chunk_higher_similarity_and_channel_tie(self):
        class StaticQueryProvider:
            model_id = "synthetic:chunk-pref-provider"
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        # Use long text with a nonzero offset chunk
        long_text = ("Initial paragraph that is quite lengthy and spans across multiple introductory lines. " * 15)
        tail_text = "Tail paragraph containing the crucial information at the very end of the message."
        full_text = long_text + tail_text
        tail_start = len(long_text)
        tail_end = len(full_text)
        self.assertGreater(tail_start, 500)

        payload = {
            "id": "chunk-pref-conv",
            "messages": [
                {"id": "msg-multi", "role": "assistant", "text": full_text}
            ]
        }
        self.index.import_payload(payload, account_id="pref-acc")

        # Chunk 0 has dialogue sim 0.5 (sim [0.5, 0.866025])
        # Chunk 1 (tail, nonzero offset) has topic sim 0.95 (sim [0.95, 0.31225])
        self._seed_vector("msg-multi", "dialogue", 0, "synthetic:chunk-pref-provider",
                           0, 300, [0.5, 0.866025], account_id="pref-acc")
        self._seed_vector("msg-multi", "topic", 1, "synthetic:chunk-pref-provider",
                           tail_start, tail_end, [0.95, 0.31225], account_id="pref-acc")

        self.index.embedding_provider = StaticQueryProvider()
        res = self.index.search("somequery", mode="smart", account_id="pref-acc")
        self.assertEqual(res["total"], 1)
        hit = res["results"][0]
        self.assertEqual(hit["semantic_evidence"]["channel"], "topic")
        self.assertEqual(hit["semantic_evidence"]["start_offset"], tail_start)
        self.assertEqual(hit["semantic_evidence"]["end_offset"], tail_end)
        self.assertEqual(hit["focus_offset"], tail_start)
        self.assertAlmostEqual(hit["semantic_similarity"], 0.95, places=4)

        # Exact tie between dialogue and topic prefers dialogue
        self._seed_vector("msg-multi", "dialogue", 0, "synthetic:chunk-pref-provider",
                           0, 300, [0.95, 0.31225], account_id="pref-acc")
        res_tie = self.index.search("somequery", mode="smart", account_id="pref-acc")
        self.assertEqual(res_tie["results"][0]["semantic_evidence"]["channel"], "dialogue")
        self.assertEqual(res_tie["results"][0]["semantic_evidence"]["start_offset"], 0)

    def test_smart_corrupt_vector_falls_back_without_semantic_leakage(self):
        class StaticQueryProvider:
            model_id = "synthetic:corrupt-test-provider"
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        payload = {
            "id": "corrupt-conv",
            "messages": [
                {"id": "msg-good", "role": "user", "text": "alpha token good message"},
                {"id": "msg-bad", "role": "user", "text": "alpha token bad message"}
            ]
        }
        self.index.import_payload(payload, account_id="corrupt-acc")
        # Seed msg-good with valid vector
        self._seed_vector("msg-good", "dialogue", 0, "synthetic:corrupt-test-provider",
                           0, 10, [1.0, 0.0], account_id="corrupt-acc")
        # Seed msg-bad with valid vector then corrupt its json
        self._seed_vector("msg-bad", "dialogue", 0, "synthetic:corrupt-test-provider",
                           0, 10, [1.0, 0.0], account_id="corrupt-acc")
        bad_row = self.index.connection.execute(
            "SELECT id FROM messages WHERE message_id='msg-bad' AND account_id='corrupt-acc'"
        ).fetchone()
        self.index.connection.execute(
            "UPDATE vectors SET vector_json='[1.0, null]' WHERE message_row=?", (bad_row["id"],)
        )
        self.index.connection.commit()

        self.index.embedding_provider = StaticQueryProvider()
        res = self.index.search("alpha", mode="smart", account_id="corrupt-acc")
        # Corruption after valid vector must discard ALL semantic candidates
        self.assertEqual(res["semantic"]["status"], "provider_error")
        self.assertNotIn("minimum_similarity", res["semantic"])
        for r in res["results"]:
            self.assertNotIn("semantic_similarity", r)
            self.assertNotIn("semantic_evidence", r)
            self.assertNotIn("dialogue", r.get("retrieval_channels", []))
            self.assertNotIn("topic", r.get("retrieval_channels", []))
        # Keyword / literal fallback remains truthful
        self.assertGreater(res["total"], 0)

    def test_smart_filtered_other_account_invalid_vector_isolated(self):
        class StaticQueryProvider:
            model_id = "synthetic:acc-iso-provider"
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        # Valid in acc-clean, corrupted in acc-dirty
        self.index.import_payload({"id": "clean-conv", "messages": [{"id": "m-clean", "role": "user", "text": "keyword in clean"}]}, account_id="acc-clean")
        self.index.import_payload({"id": "dirty-conv", "messages": [{"id": "m-dirty", "role": "user", "text": "keyword in dirty"}]}, account_id="acc-dirty")
        self._seed_vector("m-clean", "dialogue", 0, "synthetic:acc-iso-provider", 0, 16, [1.0, 0.0], account_id="acc-clean")
        self._seed_vector("m-dirty", "dialogue", 0, "synthetic:acc-iso-provider", 0, 16, [1.0, 0.0], account_id="acc-dirty")

        dirty_row = self.index.connection.execute("SELECT id FROM messages WHERE message_id='m-dirty'").fetchone()
        self.index.connection.execute("UPDATE vectors SET vector_json='not-valid-json' WHERE message_row=?", (dirty_row["id"],))
        self.index.connection.commit()

        self.index.embedding_provider = StaticQueryProvider()
        # Searching acc-clean should NOT be affected by dirty-acc's corrupt vector
        clean_res = self.index.search("keyword", mode="smart", account_id="acc-clean")
        self.assertEqual(clean_res["semantic"]["status"], "ready")
        self.assertTrue(clean_res["semantic"]["used"])
        self.assertEqual(clean_res["total"], 1)
        self.assertEqual(clean_res["results"][0]["message_id"], "m-clean")
        self.assertAlmostEqual(clean_res["results"][0]["semantic_similarity"], 1.0, places=4)

    def test_smart_more_than_100_semantic_candidates_bounded_retention(self):
        class StaticQueryProvider:
            model_id = "synthetic:150-provider"
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        # Create 150 messages with no keyword overlap with query "semanticquery"
        messages = [{"id": f"s-msg-{i:03d}", "role": "user", "text": f"unique text content block {i:03d}"} for i in range(150)]
        self.index.import_payload({"id": "many-sem-conv", "messages": messages}, account_id="sem-150-acc")

        # Seed 150 vectors with descending similarities so top 100 are well defined
        for i in range(150):
            # cosine sim = (150 - i) / 200.0, e.g. for i=0: 150/200 = 0.75 >= 0.3; for i=149: 1/200 = 0.005 < 0.3
            sim = 0.3 + (150 - i) * 0.004  # i=0 -> 0.90, i=149 -> 0.304 >= 0.3
            y = (1.0 - sim * sim) ** 0.5
            self._seed_vector(f"s-msg-{i:03d}", "dialogue", 0, "synthetic:150-provider",
                               0, 20, [sim, y], account_id="sem-150-acc")

        self.index.embedding_provider = StaticQueryProvider()
        p1 = self.index.search("semanticquery", mode="smart", account_id="sem-150-acc", limit=50, offset=0)
        p2 = self.index.search("semanticquery", mode="smart", account_id="sem-150-acc", limit=50, offset=50)
        p3 = self.index.search("semanticquery", mode="smart", account_id="sem-150-acc", limit=50, offset=100)

        # Semantic channel retains at most 100 unique messages
        self.assertEqual(p1["total"], 100)
        self.assertEqual(len(p1["results"]), 50)
        self.assertEqual(len(p2["results"]), 50)
        self.assertEqual(len(p3["results"]), 0)
        # Verify first 50 are s-msg-000 .. s-msg-049
        self.assertEqual([r["message_id"] for r in p1["results"]], [f"s-msg-{i:03d}" for i in range(50)])
        self.assertEqual([r["message_id"] for r in p2["results"]], [f"s-msg-{i:03d}" for i in range(50, 100)])

    def test_smart_adversarial_ascending_stream_replacement(self):
        class StaticQueryProvider:
            model_id = "synthetic:adv-provider"
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        # 101 messages: first 100 have ascending scores .40 + .001 * i (i=0..99 -> .400 .. .499).
        # Message 100 arrives later with high score .60.
        messages = [{"id": f"adv-msg-{i:03d}", "role": "user", "text": f"adv content block {i:03d}"} for i in range(101)]
        self.index.import_payload({"id": "adv-conv", "messages": messages}, account_id="adv-acc")

        for i in range(100):
            sim = 0.40 + 0.001 * i  # .400 to .499
            y = (1.0 - sim * sim) ** 0.5
            self._seed_vector(f"adv-msg-{i:03d}", "dialogue", 0, "synthetic:adv-provider",
                               0, 15, [sim, y], account_id="adv-acc")

        # 101st message with high similarity 0.60
        sim_100 = 0.60
        y_100 = (1.0 - sim_100 * sim_100) ** 0.5
        self._seed_vector("adv-msg-100", "dialogue", 0, "synthetic:adv-provider",
                           0, 15, [sim_100, y_100], account_id="adv-acc")

        self.index.embedding_provider = StaticQueryProvider()
        res = self.index.search("nonexistentquery", mode="smart", account_id="adv-acc", limit=100)
        self.assertEqual(res["total"], 100)
        retained_ids = [r["message_id"] for r in res["results"]]
        # adv-msg-100 has highest similarity (0.60), must be first
        self.assertEqual(retained_ids[0], "adv-msg-100")
        self.assertIn("adv-msg-100", retained_ids)
        # Earliest lowest-score candidate adv-msg-000 (.400) must be evicted
        self.assertNotIn("adv-msg-000", retained_ids)
        # adv-msg-001 (.401) through adv-msg-099 (.499) must all be retained
        for i in range(1, 100):
            self.assertIn(f"adv-msg-{i:03d}", retained_ids)

    def test_provider_aware_windows_byte_budget_coverage_and_offsets(self):
        # Test long astral/Chinese/NUL original text
        raw_text = "Hello\x00World! " + "中文测试内容" * 50 + " 😀🚀✨🎉" * 30 + " EndOfMessage"
        payload = {
            "id": "provider-window-conv",
            "messages": [
                {"id": "pw-p1", "role": "user", "text": "Prior turn 1 text " + "A" * 300},
                {"id": "pw-p2", "role": "assistant", "text": "Prior turn 2 text " + "B" * 300},
                {"id": "pw-target", "parent_id": "pw-p2", "role": "user", "text": raw_text},
            ]
        }
        self.index.import_payload(payload, account_id="pw-acc")
        target_row = self.index.connection.execute(
            "SELECT * FROM messages WHERE message_id='pw-target' AND account_id='pw-acc'"
        ).fetchone()

        # 1. Generic provider / default (max_input_bytes=None)
        gen_windows = self.index._windows(target_row)
        self.assertGreater(len(gen_windows), 0)
        self.assertEqual(gen_windows[0][0], 0)
        self.assertEqual(gen_windows[-1][1], len(raw_text))

        # 2. Budget 960 provider
        budget_windows = self.index._windows(target_row, max_input_bytes=960)
        self.assertGreater(len(budget_windows), 1)

        # Union coverage check: every character index [0, len(raw_text)) must be covered
        covered_indices = set()
        prev_start = -1
        for start, end, derived in budget_windows:
            self.assertGreater(start, prev_start)  # strictly positive progress
            prev_start = start
            self.assertLessEqual(start, end)
            self.assertGreater(end, start)
            # Codepoint slices must match exactly untouched raw_text
            self.assertTrue(derived.endswith(target_row["role"] + ": " + raw_text[start:end]))
            # Encoded derived UTF-8 byte length <= 960
            derived_bytes = derived.encode("utf-8")
            self.assertLessEqual(len(derived_bytes), 960)
            for idx in range(start, end):
                covered_indices.add(idx)

        # Check no holes even for emoji or NUL
        self.assertEqual(covered_indices, set(range(len(raw_text))))
        self.assertEqual(budget_windows[0][0], 0)
        self.assertEqual(budget_windows[-1][1], len(raw_text))

        # 3. Unsupported small / different budget raises clear bounded ValueError
        with self.assertRaises(ValueError):
            self.index._windows(target_row, max_input_bytes=100)

    def test_enrich_with_provider_max_input_bytes(self):
        class Mock960Provider:
            model_id = "synthetic:mock-960-provider"
            max_input_bytes = 960
            def __init__(self):
                self.embedded_texts = []
            def embed(self, texts):
                self.embedded_texts.extend(texts)
                for t in texts:
                    if len(t.encode("utf-8")) > 960:
                        raise ProviderError("Payload exceeds 960 bytes")
                return [[1.0, 0.0] for _ in texts]

        class MockGenericProvider:
            model_id = "synthetic:mock-gen-provider"
            def __init__(self):
                self.embedded_texts = []
            def embed(self, texts):
                self.embedded_texts.extend(texts)
                return [[1.0, 0.0] for _ in texts]

        long_text = "Enrichment test with astral emoji 🌟 and 中文字符 " * 40
        payload = {
            "id": "enrich-window-conv",
            "messages": [
                {"id": "enr-target", "role": "user", "text": long_text},
            ]
        }
        self.index.import_payload(payload, account_id="enr-acc")

        # Enrich with Mock960Provider
        p960 = Mock960Provider()
        res960 = self.index.enrich(provider=p960, account_id="enr-acc")
        self.assertFalse(res960["errors"])
        self.assertGreater(res960["vectors"], 1)
        for text in p960.embedded_texts:
            self.assertLessEqual(len(text.encode("utf-8")), 960)

        # Enrich with generic provider
        pgen = MockGenericProvider()
        resgen = self.index.enrich(provider=pgen, account_id="enr-acc")
        self.assertFalse(resgen["errors"])
        self.assertEqual(resgen["vectors"], 1)  # long_text is ~2100 chars, fits in default 3000-char window

    def test_bm25_stopword_filtering_and_literal_preservation(self):
        # Multiword query consisting purely of English function words matching messages
        # without provider returns no results in Smart mode
        stopword_query = "and or the in on at"
        hits = self.index.search(stopword_query, mode="smart")
        self.assertEqual(hits["total"], 0)
        self.assertEqual(len(hits["results"]), 0)

        # But exact literal search preserves exact phrases containing function words
        exact_query = "and"
        exact_hits = self.index.search(exact_query, mode="exact")
        self.assertGreater(exact_hits["total"], 0)

        # Multiword query with function words and content word still retrieves proper result
        mixed_query = "how to temper soup"
        mixed_hits = self.index.search(mixed_query, mode="smart")
        self.assertGreater(mixed_hits["total"], 0)
        top_hit = mixed_hits["results"][0]
        self.assertIn("soup", top_hit["text"].casefold())

    def test_search_minimum_similarity_threshold_and_malformed_fallback(self):
        class CalibratedToyProvider:
            model_id = "synthetic:calibrated-toy-v1"
            minimum_similarity = 0.55

            def embed(self, texts):
                # Returns 0.5 similarity when compared to [1.0, 0.0]
                return [[1.0, 0.0] if "\nuser:" in text or "\nassistant:" in text
                        else [0.5, 0.8660254] for text in texts]

        class MalformedProvider:
            model_id = "synthetic:malformed-toy-v1"
            minimum_similarity = "not-a-float"

            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        # 1. Custom valid minimum_similarity is exposed in response semantic info and applied
        temp_dir = tempfile.TemporaryDirectory()
        try:
            idx = SearchIndex(Path(temp_dir.name) / "test_thresh.sqlite", CalibratedToyProvider())
            idx.import_payload(self.payload)
            idx.enrich()
            resp = idx.search("test query", mode="smart")
            self.assertEqual(resp["semantic"]["minimum_similarity"], 0.55)
            # Since similarity is 0.5 < 0.55, no semantic hits should be retained
            for r in resp["results"]:
                self.assertNotIn("dialogue", r.get("retrieval_channels", []))
                self.assertNotIn("topic", r.get("retrieval_channels", []))
            idx.close()
        finally:
            temp_dir.cleanup()

        # 2. Malformed minimum_similarity triggers safe fallback to keyword/lexical without crashing
        temp_dir2 = tempfile.TemporaryDirectory()
        try:
            idx2 = SearchIndex(Path(temp_dir2.name) / "test_malformed.sqlite", MalformedProvider())
            idx2.import_payload(self.payload)
            idx2.enrich()
            resp2 = idx2.search("tomato", mode="smart")
            self.assertEqual(resp2["semantic"]["status"], "provider_error")
            self.assertIn("Invalid minimum_similarity", resp2["semantic"]["errors"][0])
            self.assertNotIn("minimum_similarity", resp2["semantic"])
            # Fallback to lexical results worked truthfully
            self.assertGreater(resp2["total"], 0)
            idx2.close()
        finally:
            temp_dir2.cleanup()

    def test_no_network_constructor_status_exact(self):
        from unittest.mock import patch
        from chat_recall.providers import OllamaProvider

        provider = OllamaProvider()
        with patch.object(provider, "_request") as mock_req:
            # 1. SearchIndex constructor with OllamaProvider
            idx = SearchIndex(Path(self.temp.name) / "no_net.sqlite", provider)
            mock_req.assert_not_called()

            # 2. status() reads cached identity only, zero network
            stat = idx.status()
            self.assertTrue(stat["semantic"]["configured"])
            self.assertEqual(stat["semantic"]["display_model_id"], "ollama:qwen3-embedding:0.6b")
            self.assertIsNone(stat["semantic"]["model_id"])
            mock_req.assert_not_called()

            # 3. Exact search never contacts Ollama
            idx.import_payload(self.payload)
            res = idx.search("tomato", mode="exact")
            self.assertGreater(res["total"], 0)
            mock_req.assert_not_called()
            idx.close()

    def test_legacy_fake_fallback(self):
        # Generic fake provider with only model_id and raw embed()
        class MinimalLegacyProvider:
            model_id = "minimal:legacy-v1"
            def embed(self, texts):
                return [[1.0, 0.0] if "soup" in t.lower() else [0.0, 1.0] for t in texts]

        prov = MinimalLegacyProvider()
        idx = SearchIndex(Path(self.temp.name) / "legacy_test.sqlite", prov)
        idx.import_payload(self.payload)

        # Enrich uses embed() fallback and records model_id
        rep = idx.enrich()
        self.assertEqual(rep["model_id"], "minimal:legacy-v1")
        self.assertGreater(rep["vectors"], 0)

        # Smart search uses embed() fallback
        res = idx.search("soup", mode="smart")
        self.assertTrue(res["semantic"]["used"])
        self.assertEqual(res["semantic"]["model_id"], "minimal:legacy-v1")
        self.assertEqual(res["semantic"]["display_model_id"], "minimal:legacy-v1")
        self.assertGreater(res["total"], 0)
        idx.close()

    def test_revision_exclusion_and_honest_coverage(self):
        class VersionedMockProvider:
            model_id = "ollama:versioned-test"
            minimum_similarity = 0.3
            expected_output_dimension = 2

            def __init__(self, revision="rev-1"):
                self.revision = revision
                self.cached_identity = f"ollama:versioned-test:{revision}"

            def resolve_identity(self, doc_profile):
                return self.cached_identity

            def embed_query(self, query):
                return [1.0, 0.0]

            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

        # 1. Enrich with revision 1
        prov1 = VersionedMockProvider("rev-1")
        idx = SearchIndex(Path(self.temp.name) / "rev_test.sqlite", prov1)
        idx.import_payload(self.payload)
        rep1 = idx.enrich()
        self.assertEqual(rep1["model_id"], "ollama:versioned-test:rev-1")
        self.assertGreater(rep1["vectors"], 0)

        # 2. Switch provider to revision 2 without re-indexing
        prov2 = VersionedMockProvider("rev-2")
        idx.embedding_provider = prov2

        # Smart search resolves to rev-2, rev-1 vectors are excluded!
        res = idx.search("test query", mode="smart")
        self.assertEqual(res["semantic"]["model_id"], "ollama:versioned-test:rev-2")
        self.assertEqual(res["semantic"]["display_model_id"], "ollama:versioned-test")
        self.assertEqual(res["semantic"]["indexed_messages"], 0)
        self.assertEqual(res["semantic"]["status"], "index_pending")
        self.assertFalse(res["semantic"]["available"])
        self.assertFalse(res["semantic"]["used"])

        # Legacy rev-1 vectors remain in the database intact
        stat = idx.status()
        model_ids_in_db = [m["model_id"] for m in stat["semantic"]["models"]]
        self.assertIn("ollama:versioned-test:rev-1", model_ids_in_db)

        # 3. Enriching with revision 2 indexes coverage for rev-2 without overwriting rev-1
        rep2 = idx.enrich()
        self.assertEqual(rep2["model_id"], "ollama:versioned-test:rev-2")

        stat2 = idx.status()
        model_ids_in_db2 = [m["model_id"] for m in stat2["semantic"]["models"]]
        self.assertIn("ollama:versioned-test:rev-1", model_ids_in_db2)
        self.assertIn("ollama:versioned-test:rev-2", model_ids_in_db2)

        res2 = idx.search("test query", mode="smart")
        self.assertEqual(res2["semantic"]["status"], "ready")
        self.assertTrue(res2["semantic"]["used"])
        idx.close()

    def test_truthful_keyword_fallback_on_resolve_identity_error(self):
        class FailingTagsProvider:
            model_id = "ollama:uninstalled-model"

            def resolve_identity(self, doc_profile):
                raise ProviderError("Configured model uninstalled-model not found in Ollama /api/tags")

            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        idx = SearchIndex(Path(self.temp.name) / "fail_test.sqlite", FailingTagsProvider())
        idx.import_payload(self.payload)

        res = idx.search("tomato", mode="smart")
        self.assertEqual(res["semantic"]["status"], "provider_error")
        self.assertIn("not found in Ollama /api/tags", res["semantic"]["errors"][0])
        self.assertFalse(res["semantic"]["used"])
        self.assertIsNone(res["semantic"]["model_id"])
        self.assertEqual(res["semantic"]["display_model_id"], "ollama:uninstalled-model")
        # Truthful keyword fallback
        self.assertGreater(res["total"], 0)
        idx.close()

    def test_concurrent_two_thread_identity_binding_and_blocking(self):
        import threading
        import time

        class TwoThreadMockProvider:
            model_id = "ollama:two-thread"
            minimum_similarity = 0.3
            expected_output_dimension = 2

            def __init__(self):
                self._lock = threading.RLock()
                self._digest_by_index_id = {}
                self._cached_identity = None
                self._resolved_digest = None
                self.thread_a_in_embed = threading.Event()
                self.thread_b_can_proceed = threading.Event()
                self.thread_b_done = threading.Event()

            def resolve_identity(self, doc_profile):
                with self._lock:
                    rev = doc_profile.get("rev", "rev-default")
                    digest = f"sha256:{rev}"
                    index_id = f"ollama:two-thread:{rev}"
                    self._resolved_digest = digest
                    self._cached_identity = index_id
                    self._digest_by_index_id[index_id] = digest
                    return index_id

            def embed_documents(self, texts, index_id=None):
                with self._lock:
                    expected = self._digest_by_index_id.get(index_id, self._resolved_digest)
                    if index_id == "ollama:two-thread:rev-A":
                        # Signal that Thread A is inside embed_documents
                        self.thread_a_in_embed.set()
                        # Wait for Thread B to attempt its operation
                        self.thread_b_can_proceed.wait(timeout=2.0)
                    if expected != self._digest_by_index_id.get(index_id):
                        raise ProviderError("Digest mismatch during concurrent operation")
                    return [[1.0, 0.0] for _ in texts]

            def embed_query(self, query, index_id=None):
                with self._lock:
                    expected = self._digest_by_index_id.get(index_id, self._resolved_digest)
                    if expected != self._digest_by_index_id.get(index_id):
                        raise ProviderError("Digest mismatch during concurrent operation")
                    return [1.0, 0.0]

        prov = TwoThreadMockProvider()
        idx_a = SearchIndex(Path(self.temp.name) / "thread_test.sqlite", prov)
        idx_a.import_payload(self.payload)

        # Thread A enriches under profile with rev-A
        errors_a = []
        errors_b = []

        def worker_a():
            try:
                prof_a = dict(SearchIndex.DOCUMENT_PROFILE, rev="rev-A")
                id_a = prov.resolve_identity(prof_a)
                # Ensure vectors are computed and published strictly under id_a
                prov.embed_documents(["doc a"], index_id=id_a)
            except Exception as e:
                errors_a.append(e)

        def worker_b():
            # Wait until thread A has resolved its identity
            prov.thread_a_in_embed.wait(timeout=2.0)
            try:
                # Thread B mutates provider state with rev-B
                prof_b = dict(SearchIndex.DOCUMENT_PROFILE, rev="rev-B")
                id_b = prov.resolve_identity(prof_b)
                # Thread B completes its embedding
                prov.embed_documents(["doc b"], index_id=id_b)
                prov.thread_b_done.set()
            except Exception as e:
                errors_b.append(e)
            finally:
                prov.thread_b_can_proceed.set()

        t_a = threading.Thread(target=worker_a)
        t_b = threading.Thread(target=worker_b)

        t_a.start()
        t_b.start()
        t_a.join(timeout=3.0)
        t_b.join(timeout=3.0)

        self.assertEqual(errors_a, [])
        self.assertEqual(errors_b, [])
        # Both identities were preserved in the provider mapping without collision
        self.assertEqual(prov._digest_by_index_id["ollama:two-thread:rev-A"], "sha256:rev-A")
        self.assertEqual(prov._digest_by_index_id["ollama:two-thread:rev-B"], "sha256:rev-B")
        idx_a.close()

    def test_status_with_unresolved_versioned_provider_never_falls_back_to_alias(self):
        class VersionedUnresolvedProvider:
            model_id = "ollama:test-alias"
            cached_identity = None

            def resolve_identity(self, doc_profile):
                return "ollama:test-alias:fingerprint999"

            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        prov = VersionedUnresolvedProvider()
        idx = SearchIndex(Path(self.temp.name) / "unresolved_test.sqlite", prov)
        idx.import_payload(self.payload)

        # Seed legacy alias row into vectors table
        first_row = idx.connection.execute("SELECT id, content_hash FROM messages LIMIT 1").fetchone()
        idx.connection.execute(
            "INSERT INTO vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (first_row["id"], "dialogue", 0, "ollama:test-alias", first_row["content_hash"], 0, 10, "derived", json.dumps([1.0, 0.0]))
        )

        stat = idx.status()
        # For a versioned provider with cached_identity None, status must not fall back to display alias
        self.assertIsNone(stat["semantic"]["model_id"])
        self.assertEqual(stat["semantic"]["display_model_id"], "ollama:test-alias")
        idx.close()

    def test_blocking_embed_with_tags_mutation_during_inference_rejected(self):
        from unittest.mock import patch
        from chat_recall.providers import OllamaProvider

        prov = OllamaProvider()
        idx = SearchIndex(Path(self.temp.name) / "tags_mutation_test.sqlite", prov)
        idx.import_payload(self.payload)

        # Resolve initial identity
        initial_tags = {"models": [{"name": "qwen3-embedding:0.6b", "digest": "sha256:digest111"}]}
        with patch.object(prov, "_request", return_value=initial_tags):
            init_id = prov.resolve_identity(SearchIndex.DOCUMENT_PROFILE)

        # Seed vector row under init_id so smart search proceeds to query embedding
        first_row = idx.connection.execute("SELECT id, content_hash FROM messages LIMIT 1").fetchone()
        idx.connection.execute(
            "INSERT INTO vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (first_row["id"], "dialogue", 0, init_id, first_row["content_hash"], 0, 10, "derived", json.dumps([0.6, 0.8] * 512))
        )
        idx.connection.execute(
            "INSERT INTO enrichment (message_row, model_id, content_hash, topics) VALUES (?, ?, ?, 0)",
            (first_row["id"], init_id, first_row["content_hash"])
        )
        idx.connection.commit()

        # Now simulate concurrency: during /api/embed, server tags mutate to digest222
        mutated_tags = {"models": [{"name": "qwen3-embedding:0.6b", "digest": "sha256:digest222"}]}
        current_tags = initial_tags

        def fake_request(path, *args, **kwargs):
            nonlocal current_tags
            if path == "/api/tags":
                return current_tags
            if path == "/api/embed":
                # Concurrency simulation: /api/tags changes while /api/embed is in flight
                current_tags = mutated_tags
                return {"embeddings": [[0.6, 0.8] * 512]}
            return {}

        with patch.object(prov, "_request", side_effect=fake_request):
            res = idx.search("tomato", mode="smart")
            # Old output must be rejected!
            self.assertEqual(res["semantic"]["status"], "provider_error")
            self.assertFalse(res["semantic"]["used"])
            self.assertTrue(any("Model digest changed" in e for e in res["semantic"]["errors"]))
            # Truthful keyword fallback
            self.assertGreater(res["total"], 0)
        idx.close()

    def test_search_select_to_inference_operation_lock_protects_concurrent_operation(self):
        import threading
        import time
        from chat_recall.providers import OllamaProvider

        db_path = Path(self.temp.name) / "op_lock_test.sqlite"
        prov = OllamaProvider()
        setup_idx = SearchIndex(db_path, prov)
        setup_idx.import_payload(self.payload)

        initial_tags = {"models": [{"name": "qwen3-embedding:0.6b", "digest": "sha256:stable_op_lock"}]}
        with patch.object(prov, "_request", return_value=initial_tags):
            init_id = prov.resolve_identity(SearchIndex.DOCUMENT_PROFILE)

        first_row = setup_idx.connection.execute("SELECT id, content_hash FROM messages LIMIT 1").fetchone()
        setup_idx.connection.execute(
            "INSERT INTO vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (first_row["id"], "dialogue", 0, init_id, first_row["content_hash"], 0, 10, "derived", json.dumps([0.6, 0.8] * 512))
        )
        setup_idx.connection.execute(
            "INSERT INTO enrichment (message_row, model_id, content_hash, topics) VALUES (?, ?, ?, 0)",
            (first_row["id"], init_id, first_row["content_hash"])
        )
        setup_idx.connection.commit()
        setup_idx.close()

        thread_a_in_embed = threading.Event()
        thread_b_attempted = threading.Event()
        thread_b_acquired = threading.Event()

        def blocking_embed_query(query, index_id=None):
            thread_a_in_embed.set()
            thread_b_attempted.wait(timeout=2.0)
            time.sleep(0.05)
            return [0.6, 0.8] * 512

        errors_a = []
        errors_b = []

        def worker_a():
            try:
                # Use a distinct SearchIndex/SQLite connection created inside worker A
                with SearchIndex(db_path, prov) as worker_idx:
                    with patch.object(prov, "_request", return_value=initial_tags):
                        with patch.object(prov, "embed_query", side_effect=blocking_embed_query):
                            worker_idx.search("tomato", mode="smart")
            except Exception as e:
                errors_a.append(e)

        def worker_b():
            try:
                # Wait until Thread A reaches the embed callback
                if not thread_a_in_embed.wait(timeout=2.0):
                    errors_b.append(RuntimeError("Thread A never reached embed callback"))
                    return
                thread_b_attempted.set()
                # Thread B attempts to acquire op_lock while Thread A is inside search
                acquired = prov.operation_lock.acquire(timeout=0.01)
                if acquired:
                    thread_b_acquired.set()
                    prov.operation_lock.release()
            except Exception as e:
                errors_b.append(e)

        t_a = threading.Thread(target=worker_a)
        t_b = threading.Thread(target=worker_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=3.0)
        t_b.join(timeout=3.0)

        self.assertEqual(errors_a, [])
        self.assertEqual(errors_b, [])
        # Explicitly assert thread A reached the embed callback and thread B attempted lock
        self.assertTrue(thread_a_in_embed.is_set(), "Thread A must have reached embed callback")
        self.assertTrue(thread_b_attempted.is_set(), "Thread B must have attempted lock")
        # Thread B must have been blocked while Thread A held op_lock across SELECT-to-inference
        self.assertFalse(thread_b_acquired.is_set(), "Thread B must not acquire lock during SELECT-to-inference")

    def test_internal_type_error_in_bound_provider_never_retries_unbound(self):
        class BoundProviderWithInternalTypeError:
            model_id = "test:bound-type-error"
            minimum_similarity = 0.3

            def __init__(self):
                self.embed_query_calls = 0
                self.embed_documents_calls = 0

            def resolve_identity(self, doc_profile):
                return "test:bound-type-error:rev1"

            def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
                self.embed_query_calls += 1
                # Internal execution error (not signature mismatch!)
                raise TypeError("Internal TypeError in model execution logic")

            def embed_documents(self, texts: list[str], index_id: str | None = None) -> list[list[float]]:
                self.embed_documents_calls += 1
                raise TypeError("Internal TypeError in batch document logic")

        prov = BoundProviderWithInternalTypeError()
        idx = SearchIndex(Path(self.temp.name) / "type_error_test.sqlite", prov)
        idx.import_payload(self.payload)

        # 1. Seed vector under resolved identity so smart search reaches query embedding
        first_row = idx.connection.execute("SELECT id, content_hash FROM messages LIMIT 1").fetchone()
        idx.connection.execute(
            "INSERT INTO vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (first_row["id"], "dialogue", 0, "test:bound-type-error:rev1", first_row["content_hash"], 0, 10, "derived", json.dumps([1.0, 0.0]))
        )
        idx.connection.execute(
            "INSERT INTO enrichment (message_row, model_id, content_hash, topics) VALUES (?, ?, ?, 0)",
            (first_row["id"], "test:bound-type-error:rev1", first_row["content_hash"])
        )
        idx.connection.commit()

        # Smart search must call embed_query exactly once without unbound retry
        res = idx.search("tomato", mode="smart")
        self.assertEqual(prov.embed_query_calls, 1)
        self.assertEqual(res["semantic"]["status"], "provider_error")
        self.assertFalse(res["semantic"]["used"])
        self.assertTrue(any("Internal TypeError" in e for e in res["semantic"]["errors"]))
        # Truthful lexical fallback
        self.assertGreater(res["total"], 0)

        # 2. Enrich must call embed_documents exactly once without unbound retry
        rep = idx.enrich(limit=1)
        self.assertEqual(prov.embed_documents_calls, 1)
        self.assertEqual(rep["processed"], 0)
        self.assertTrue(any(e["error"] == "type_error" for e in rep["errors"]))
        idx.close()

    def test_legacy_no_index_id_provider_supported(self):
        class LegacyNoIndexIdProvider:
            model_id = "test:legacy-no-index-id"
            minimum_similarity = 0.3

            def __init__(self):
                self.query_calls = 0
                self.doc_calls = 0

            # Legacy provider does NOT accept index_id parameter
            def embed_query(self, query: str) -> list[float]:
                self.query_calls += 1
                return [1.0, 0.0]

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                self.doc_calls += 1
                return [[1.0, 0.0] for _ in texts]

        prov = LegacyNoIndexIdProvider()
        idx = SearchIndex(Path(self.temp.name) / "legacy_no_index_id.sqlite", prov)
        idx.import_payload(self.payload)

        # Enrich successfully calls embed_documents(texts)
        rep = idx.enrich(limit=1)
        self.assertGreater(prov.doc_calls, 0)
        self.assertEqual(rep["processed"], 1)

        # Search successfully calls embed_query(query)
        res = idx.search("tomato", mode="smart")
        self.assertEqual(prov.query_calls, 1)
        self.assertTrue(res["semantic"]["used"])
        # With limit=1 against multi-message fixture, 1 message is indexed and pending_messages > 0, so status is partial
        self.assertEqual(res["semantic"]["status"], "partial")
        self.assertEqual(res["semantic"]["indexed_messages"], 1)
        self.assertGreater(res["semantic"]["pending_messages"], 0)
        self.assertTrue(res["semantic"]["available"])
        self.assertEqual(res["semantic"]["model_id"], "test:legacy-no-index-id")
        self.assertEqual(res["semantic"]["display_model_id"], "test:legacy-no-index-id")
        idx.close()

    def test_recording_provider_wrapper_status_transparency_for_legacy_and_versioned(self):
        from scripts.evaluate_smart import RecordingProviderWrapper

        # 1. Wrapper around legacy model_id/embed-only provider
        class SimpleLegacyProvider:
            model_id = "legacy:fixture-model"
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        w_legacy = RecordingProviderWrapper(SimpleLegacyProvider())
        # Capability transparency: wrapper does NOT advertise resolve_identity when underlying provider lacks it
        self.assertFalse(callable(getattr(w_legacy, "resolve_identity", None)))
        self.assertEqual(w_legacy.cached_identity, "legacy:fixture-model")
        self.assertEqual(w_legacy.index_id, "legacy:fixture-model")

        idx_leg = SearchIndex(Path(self.temp.name) / "wrap_leg.sqlite", w_legacy)
        idx_leg.import_payload(self.payload)

        # Seed vector under legacy alias
        first_row = idx_leg.connection.execute("SELECT id, content_hash FROM messages LIMIT 1").fetchone()
        idx_leg.connection.execute(
            "INSERT INTO vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (first_row["id"], "dialogue", 0, "legacy:fixture-model", first_row["content_hash"], 0, 10, "derived", json.dumps([1.0, 0.0]))
        )

        stat_leg = idx_leg.status()
        # Status transparency: reports compatible legacy model_id
        self.assertEqual(stat_leg["semantic"]["model_id"], "legacy:fixture-model")
        self.assertEqual(stat_leg["semantic"]["display_model_id"], "legacy:fixture-model")
        self.assertEqual(len(stat_leg["semantic"]["models"]), 1)
        self.assertEqual(stat_leg["semantic"]["models"][0]["model_id"], "legacy:fixture-model")
        idx_leg.close()

        # 2. Wrapper around truly versioned unresolved provider
        class UnresolvedVersionedProvider:
            model_id = "versioned:ollama-alias"
            cached_identity = None
            def resolve_identity(self, doc_profile):
                return "versioned:ollama-alias:fingerprint999"
            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        w_ver = RecordingProviderWrapper(UnresolvedVersionedProvider())
        # Capability is advertised for truly versioned provider
        self.assertTrue(callable(getattr(w_ver, "resolve_identity", None)))
        self.assertIsNone(w_ver.cached_identity)
        self.assertIsNone(w_ver.index_id)

        idx_ver = SearchIndex(Path(self.temp.name) / "wrap_ver.sqlite", w_ver)
        stat_ver = idx_ver.status()
        # Preserves protection: unresolved versioned provider reports None, never falling back to alias
        self.assertIsNone(stat_ver["semantic"]["model_id"])
        self.assertEqual(stat_ver["semantic"]["display_model_id"], "versioned:ollama-alias")
        idx_ver.close()

    def test_cjk_pure_helper_functions(self):
        # 1. Natural Chinese query extracts >=2-character CJK concepts (bigrams & trigrams)
        terms = cjk_lexical_terms("内存占用太高")
        self.assertIn("内存", terms)
        self.assertIn("占用", terms)
        self.assertIn("太高", terms)
        self.assertTrue(all(len(t) >= 2 for t in terms))
        self.assertLessEqual(len(terms), 40)

        # 2. Singleton Chinese characters produce no terms
        self.assertEqual(cjk_lexical_terms("的"), [])
        self.assertEqual(cjk_lexical_terms("中"), [])
        self.assertEqual(cjk_lexical_terms("高"), [])

        # 3. Noise-only stop grams produce no terms
        self.assertEqual(cjk_lexical_terms("这个 那个 如何 可以"), [])
        self.assertEqual(cjk_lexical_terms("，。！？；"), [])

        # 4. Mixed language: retains English words and mixed identifiers alongside CJK concepts
        mixed_terms = cjk_lexical_terms("qwen3-embedding 内存压力 ERR_CFG_204")
        self.assertIn("qwen3-embedding", mixed_terms)
        self.assertIn("qwen3", mixed_terms)
        self.assertIn("embedding", mixed_terms)
        self.assertIn("内存", mixed_terms)
        self.assertIn("压力", mixed_terms)
        self.assertIn("ERR_CFG_204", mixed_terms)
        self.assertIn("ERR", mixed_terms)
        self.assertIn("CFG", mixed_terms)
        self.assertIn("204", mixed_terms)

        # 5. Index tokens extractor
        tokens = cjk_index_tokens("本地模型 qwen3 内存压力")
        self.assertIn("qwen3", tokens)
        self.assertIn("本地", tokens)
        self.assertIn("模型", tokens)
        self.assertIn("内存", tokens)
        self.assertIn("压力", tokens)

        # 6. Real words containing 和, 与, 及 must NOT lose their concepts
        heping_terms = cjk_lexical_terms("和平方面的安排")
        self.assertIn("和平", heping_terms)
        self.assertIn("方面", heping_terms)
        self.assertIn("安排", heping_terms)

        gonghe_terms = cjk_lexical_terms("共和国历史")
        self.assertIn("共和", gonghe_terms)
        self.assertIn("历史", gonghe_terms)

        canyu_terms = cjk_lexical_terms("参与活动")
        self.assertIn("参与", canyu_terms)
        self.assertIn("活动", canyu_terms)

        # 7. Semantically meaningful negation 没有 / 不是 preserved
        meiyou_terms = cjk_lexical_terms("系统目前没有配置相关参数")
        self.assertIn("没有", meiyou_terms)
        self.assertIn("配置", meiyou_terms)

        bushi_terms = cjk_lexical_terms("不是这个原因导致的异常")
        self.assertIn("不是", bushi_terms)
        self.assertIn("原因", bushi_terms)

    def test_cjk_subword_and_mixed_language_retrieval_with_no_provider(self):
        payload = {
            "id": "cjk-test-conv",
            "messages": [
                {
                    "id": "cjk-msg-1",
                    "role": "assistant",
                    "text": "本地模型内存压力讨论，如何在个人电脑上优化显存与内存分配。",
                },
                {
                    "id": "cjk-msg-2",
                    "role": "assistant",
                    "text": "家庭自制番茄浓汤去酸烹饪技巧与烘焙发酵建议。",
                },
                {
                    "id": "cjk-msg-3",
                    "role": "assistant",
                    "text": "qwen3-embedding:0.6b 部署排障，解决 ERR_CFG_204 配置报错。",
                },
                {
                    "id": "cjk-msg-4",
                    "role": "assistant",
                    "text": "关于和平公园建设规划与绿化植被种植方案讨论。",
                },
                {
                    "id": "cjk-msg-5",
                    "role": "assistant",
                    "text": "系统目前没有配置相关参数，请检查环境变量设置。",
                },
            ],
        }
        self.index.import_payload(payload, account_id="cjk-acc")

        # Subword retrieval with no provider: natural query 内存占用太高 must retrieve 本地模型内存压力
        subword_res = self.index.search("内存占用太高", mode="smart", account_id="cjk-acc")
        self.assertFalse(subword_res["semantic"]["used"])
        self.assertGreater(subword_res["total"], 0)
        hit1 = subword_res["results"][0]
        self.assertEqual(hit1["message_id"], "cjk-msg-1")
        self.assertEqual(hit1["match_type"], "lexical")
        self.assertIn("keyword", hit1["retrieval_channels"])
        self.assertNotIn("dialogue", hit1["retrieval_channels"])
        self.assertNotIn("topic", hit1["retrieval_channels"])

        # Mixed language query: English identifier + Chinese keyword
        mixed_res = self.index.search("qwen3 部署排障", mode="smart", account_id="cjk-acc")
        self.assertFalse(mixed_res["semantic"]["used"])
        self.assertGreater(mixed_res["total"], 0)
        hit3 = mixed_res["results"][0]
        self.assertEqual(hit3["message_id"], "cjk-msg-3")
        self.assertEqual(hit3["match_type"], "lexical")
        self.assertIn("keyword", hit3["retrieval_channels"])

        # Error code identifier query
        err_res = self.index.search("ERR_CFG_204", mode="smart", account_id="cjk-acc")
        self.assertGreater(err_res["total"], 0)
        self.assertEqual(err_res["results"][0]["message_id"], "cjk-msg-3")

        # Real concept with 和 (e.g. 和平) must match 和平公园 message
        heping_res = self.index.search("和平方面的安排", mode="smart", account_id="cjk-acc")
        self.assertGreater(heping_res["total"], 0)
        self.assertEqual(heping_res["results"][0]["message_id"], "cjk-msg-4")
        self.assertIn("keyword", heping_res["results"][0]["retrieval_channels"])

        # Negation concept (没有) preserved rather than treated as stop filler
        negation_res = self.index.search("没有配置", mode="smart", account_id="cjk-acc")
        self.assertGreater(negation_res["total"], 0)
        self.assertEqual(negation_res["results"][0]["message_id"], "cjk-msg-5")

    def test_cjk_noise_only_grams_not_unbounded_result_flood(self):
        # Noise helper emits no singleton or stop-only grams
        for noise_q in ("的", "中", "了", "是", "和", "与", "及", "这个 那个 如何 可以", "，。！？；", "的 了 是 在", "为什么这个可以", "为什么这个和那个可以", "这个 和 那个"):
            self.assertEqual(cjk_lexical_terms(noise_q), [], f"cjk_lexical_terms({noise_q!r}) must be empty")

        # Smart search candidate expansion stays bounded; stop-only queries produce no hits
        for noise_q in ("这个 那个 如何 可以", "为什么这个可以", "为什么这个和那个可以"):
            res = self.index.search(noise_q, mode="smart")
            self.assertEqual(res["total"], 0)
            self.assertEqual(res["results"], [])

        # For single character literals such as '的', '中', or '和', Smart search's literal channel
        # legitimately returns literal hits without keyword candidate expansion flood
        for lit_q in ("的", "中", "和"):
            res = self.index.search(lit_q, mode="smart")
            for r in res["results"]:
                self.assertNotIn("keyword", r.get("retrieval_channels", []))
                self.assertNotEqual(r.get("match_type"), "lexical")

    def test_cjk_account_and_role_isolation(self):
        payload_1 = {
            "id": "cjk-iso-1",
            "messages": [
                {"id": "iso-u1", "role": "user", "text": "分布式数据库存储设计问答"},
                {"id": "iso-a1", "role": "assistant", "text": "分布式数据库读写分离与分库分表实施细节"},
            ],
        }
        payload_2 = {
            "id": "cjk-iso-2",
            "messages": [
                {"id": "iso-u2", "role": "user", "text": "分布式缓存集群运维指南"},
            ],
        }
        self.index.import_payload(payload_1, account_id="acc-cjk-1")
        self.index.import_payload(payload_2, account_id="acc-cjk-2")

        # Account isolation
        acc1_hits = self.index.search("数据库", mode="smart", account_id="acc-cjk-1")
        self.assertEqual(acc1_hits["total"], 2)
        self.assertTrue(all(r["account_id"] == "acc-cjk-1" for r in acc1_hits["results"]))

        acc2_hits = self.index.search("数据库", mode="smart", account_id="acc-cjk-2")
        self.assertEqual(acc2_hits["total"], 0)

        # Role isolation
        user_only = self.index.search("数据库", mode="smart", account_id="acc-cjk-1", role="user")
        self.assertEqual(user_only["total"], 1)
        self.assertEqual(user_only["results"][0]["message_id"], "iso-u1")

        asst_only = self.index.search("数据库", mode="smart", account_id="acc-cjk-1", role="assistant")
        self.assertEqual(asst_only["total"], 1)
        self.assertEqual(asst_only["results"][0]["message_id"], "iso-a1")

        # Controlled corpus test: 120 messages in flood account, 2 messages in target account
        flood_payload = {
            "id": "cjk-flood-conv",
            "messages": [
                {"id": f"flood-msg-{i}", "role": "assistant", "text": f"分布式系统架构性能调优指南第{i}章"}
                for i in range(120)
            ],
        }
        target_payload = {
            "id": "cjk-target-conv",
            "messages": [
                {"id": "target-msg-u", "role": "user", "text": "分布式系统架构性能调优用户问答"},
                {"id": "target-msg-a", "role": "assistant", "text": "分布式系统架构性能调优助手实施方案"},
            ],
        }
        self.index.import_payload(flood_payload, account_id="acc-flood")
        self.index.import_payload(target_payload, account_id="acc-target")

        # Must retrieve both target messages without being displaced by 120 flood messages
        target_res = self.index.search("性能调优", mode="smart", account_id="acc-target")
        self.assertEqual(target_res["total"], 2)
        target_ids = {r["message_id"] for r in target_res["results"]}
        self.assertEqual(target_ids, {"target-msg-u", "target-msg-a"})

        # Role filter within target account
        target_asst = self.index.search("性能调优", mode="smart", account_id="acc-target", role="assistant")
        self.assertEqual(target_asst["total"], 1)
        self.assertEqual(target_asst["results"][0]["message_id"], "target-msg-a")

        # Trustworthy SQL scope check: trace executed queries during search
        executed_sqls = []
        self.index.connection.set_trace_callback(executed_sqls.append)
        try:
            self.index.search("性能调优", mode="smart", account_id="acc-target", role="assistant")
        finally:
            self.index.connection.set_trace_callback(None)

        cte_sqls = [s for s in executed_sqls if "ranked_segments" in s]
        self.assertGreater(len(cte_sqls), 0)
        emitted_cte_sql = cte_sqls[0]
        # Verify segment-to-message join and predicates are inside the CTE definition
        cte_parts = emitted_cte_sql.split(") SELECT", 1)
        self.assertEqual(len(cte_parts), 2)
        cte_body = cte_parts[0]
        outer_query = cte_parts[1]
        self.assertIn("m.account_id='acc-target'", cte_body)
        self.assertIn("m.role='assistant'", cte_body)
        self.assertIn("JOIN message_cjk_segments", cte_body)
        self.assertIn("JOIN messages m", cte_body)
        self.assertIn("bm25(message_cjk)", cte_body)
        self.assertIn("LIMIT -1 OFFSET 0", cte_body)
        self.assertIn("GROUP BY message_row", outer_query)
        self.assertIn("ORDER BY MIN(score)", outer_query)

        # Verify EXPLAIN QUERY PLAN executes cleanly on the compiled CTE query
        explain_sql = f"EXPLAIN QUERY PLAN {emitted_cte_sql}"
        plan_rows = self.index.connection.execute(explain_sql).fetchall()
        self.assertGreater(len(plan_rows), 0)

    def test_cjk_fallback_execution_when_materialized_unsupported(self):
        payload = {
            "id": "cjk-fallback-conv",
            "messages": [
                {"id": "fb-msg-1", "role": "assistant", "text": "本地模型分布式推理显存与内存优化方案"},
                {"id": "fb-msg-2", "role": "assistant", "text": "烘焙面包酵母发酵技巧与火候控制"},
            ],
        }
        self.index.import_payload(payload, account_id="fb-acc")

        real_conn = self.index.connection
        rejected_queries = []

        class MaterializedRejectingConnection:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *args):
                if "MATERIALIZED" in sql:
                    rejected_queries.append(sql)
                    raise sqlite3.OperationalError("near 'MATERIALIZED': syntax error")
                return self._real.execute(sql, *args)

            def executemany(self, sql, *args):
                return self._real.executemany(sql, *args)

            def __getattr__(self, name):
                return getattr(self._real, name)

        proxy = MaterializedRejectingConnection(real_conn)
        self.index.connection = proxy
        try:
            res = self.index.search("内存优化", mode="smart", account_id="fb-acc")
            # Verify MATERIALIZED query was attempted and rejected
            self.assertGreater(len(rejected_queries), 0)
            self.assertIn("MATERIALIZED", rejected_queries[0])
            # Verify fallback query with LIMIT -1 OFFSET 0 barrier really executed on SQLite and returned real hits
            self.assertEqual(res["total"], 1)
            self.assertEqual(res["results"][0]["message_id"], "fb-msg-1")
            self.assertIn("keyword", res["results"][0]["retrieval_channels"])
        finally:
            self.index.connection = real_conn

    def test_cjk_index_maintenance_on_update_and_hidden_revocation(self):
        payload = {
            "id": "cjk-maint-conv",
            "messages": [
                {"id": "cjk-maint-msg", "role": "user", "text": "初期旧文本内容高山乌龙茶冲泡方法"},
            ],
        }
        self.index.import_payload(payload, account_id="maint-acc")
        self.assertEqual(self.index.search("高山乌龙茶", mode="smart", account_id="maint-acc")["total"], 1)

        # Update message
        updated_payload = {
            "id": "cjk-maint-conv",
            "messages": [
                {"id": "cjk-maint-msg", "role": "user", "text": "完全替换的新文本手冲咖啡过萃排查流程"},
            ],
        }
        self.index.import_payload(updated_payload, account_id="maint-acc")
        self.assertEqual(self.index.search("高山乌龙茶", mode="smart", account_id="maint-acc")["total"], 0)
        self.assertEqual(self.index.search("手冲咖啡过萃", mode="smart", account_id="maint-acc")["total"], 1)

        # Hidden revocation
        hidden_payload = {
            "id": "cjk-maint-conv",
            "messages": [
                {
                    "id": "cjk-maint-msg",
                    "role": "user",
                    "text": "完全替换的新文本手冲咖啡过萃排查流程",
                    "metadata": {"is_visually_hidden_from_conversation": True},
                },
            ],
        }
        self.index.import_payload(hidden_payload, account_id="maint-acc")
        self.assertEqual(self.index.search("手冲咖啡过萃", mode="smart", account_id="maint-acc")["total"], 0)

        # Verify message_cjk table has 0 rows for this message
        row = self.index.connection.execute(
            "SELECT COUNT(*) FROM message_cjk WHERE text LIKE '%手冲%'"
        ).fetchone()[0]
        self.assertEqual(row, 0)

    def test_paragraph_sentence_and_codeblock_aware_window_boundaries(self):
        p1 = "First paragraph about tomato soup balance and roasting fresh tomatoes."
        p2 = "Second paragraph discussing dough fermentation in cold winter kitchens."
        combined = f"{p1}\n\n{p2}"

        payload = {
            "id": "para-aware-conv",
            "messages": [
                {"id": "msg-two-para", "role": "assistant", "text": combined},
            ],
        }
        self.index.import_payload(payload, account_id="para-acc")
        row = self.index.connection.execute(
            "SELECT * FROM messages WHERE message_id='msg-two-para' AND account_id='para-acc'"
        ).fetchone()

        windows = self.index._windows(row, max_input_bytes=960)
        self.assertEqual(len(windows), 2)

        # Window 1 covers Paragraph 1
        w1_start, w1_end, w1_derived = windows[0]
        self.assertEqual(w1_start, 0)
        self.assertEqual(w1_end, len(p1) + 2)  # includes trailing \n\n
        self.assertIn(p1, row["text"][w1_start:w1_end])

        # Window 2 covers Paragraph 2 and starts EXACTLY at the beginning of Paragraph 2!
        w2_start, w2_end, w2_derived = windows[1]
        self.assertEqual(w2_start, len(p1) + 2)
        self.assertEqual(w2_end, len(combined))
        self.assertEqual(row["text"][w2_start:w2_end], p2)
        # Window 2 does NOT point to Paragraph 1!
        self.assertNotIn(p1, row["text"][w2_start:w2_end])

        # Large paragraph (> 512 bytes) split at sentence boundaries
        s1 = "This is the first long sentence about algorithmic complexity and data structures. " * 5
        s2 = "This is the second distinct sentence discussing operating system page replacement policies. " * 5
        long_para = s1 + s2
        self.assertGreater(len(long_para.encode("utf-8")), 512)

        payload_long = {
            "id": "long-para-conv",
            "messages": [
                {"id": "msg-long-para", "role": "assistant", "text": long_para},
            ],
        }
        self.index.import_payload(payload_long, account_id="para-acc")
        row_long = self.index.connection.execute(
            "SELECT * FROM messages WHERE message_id='msg-long-para' AND account_id='para-acc'"
        ).fetchone()
        long_windows = self.index._windows(row_long, max_input_bytes=960)
        self.assertGreater(len(long_windows), 1)
        for ws, we, wd in long_windows:
            self.assertLessEqual(len(row_long["text"][ws:we].encode("utf-8")), 512)
            self.assertLessEqual(len(wd.encode("utf-8")), 960)

        # Code block (> 512 bytes) with newlines
        code_lines = "\n".join(f"    line_{i:02d} = compute_value({i}) + offset_constant;" for i in range(25))
        code_text = f"```python\n{code_lines}\n```"
        self.assertGreater(len(code_text.encode("utf-8")), 512)

        payload_code = {
            "id": "code-conv",
            "messages": [
                {"id": "msg-code", "role": "assistant", "text": code_text},
            ],
        }
        self.index.import_payload(payload_code, account_id="para-acc")
        row_code = self.index.connection.execute(
            "SELECT * FROM messages WHERE message_id='msg-code' AND account_id='para-acc'"
        ).fetchone()
        code_windows = self.index._windows(row_code, max_input_bytes=960)
        self.assertGreater(len(code_windows), 1)
        for ws, we, wd in code_windows:
            self.assertLessEqual(len(row_code["text"][ws:we].encode("utf-8")), 512)
            self.assertLessEqual(len(wd.encode("utf-8")), 960)

    def test_dialogue_prefix_coherent_question_and_sibling_exclusion(self):
        long_question = "How can I reduce the sharp sourness and acidity in my homemade tomato soup without dumping in spoons of white sugar? " + ("Additional context details about soup ingredients and stove simmering. " * 10)
        self.assertGreater(len(long_question.encode("utf-8")), 500)

        payload = {
            "id": "prefix-coherent-conv",
            "messages": [
                {"id": "q-root", "role": "user", "text": long_question},
                {"id": "a-target", "parent_id": "q-root", "role": "assistant", "text": "Roasting the tomatoes in the oven softens raw acidity."},
            ],
        }
        self.index.import_payload(payload, account_id="prefix-acc")
        row = self.index.connection.execute(
            "SELECT * FROM messages WHERE message_id='a-target' AND account_id='prefix-acc'"
        ).fetchone()

        windows = self.index._windows(row, max_input_bytes=960)
        self.assertGreater(len(windows), 0)
        derived = windows[0][2]

        # Derived text begins with coherent user question excerpt
        self.assertTrue(derived.startswith("user: How can I reduce the sharp sourness and acidity in my homemade tomato soup"))
        # Entire document is <= 960 bytes
        self.assertLessEqual(len(derived.encode("utf-8")), 960)

    def test_crash_safe_cjk_backfill_reopening_after_interruption(self):
        acc = "crash-acc"
        payload = {
            "id": "crash-conv",
            "messages": [
                {"id": "msg-cjk-1", "role": "user", "text": "高并发分布式数据库事务一致性探讨"},
                {"id": "msg-cjk-2", "role": "assistant", "text": "两阶段提交与分布式共识协议Paxos详解"},
            ]
        }
        self.index.import_payload(payload, account_id=acc)
        res_before = self.index.search("事务一致性", mode="smart", account_id=acc)
        self.assertEqual(res_before["total"], 1)

        orig_messages = list(self.index.connection.execute("SELECT id, content_hash, text FROM messages WHERE account_id=?", (acc,)))
        self.assertEqual(len(orig_messages), 2)

        # Simulate interruption: delete completion marker and truncate message_cjk
        self.index.connection.execute("DELETE FROM index_meta WHERE key='message_cjk_ready'")
        self.index.connection.execute("DELETE FROM message_cjk")
        self.index.connection.execute("DELETE FROM message_cjk_segments")
        self.index.connection.commit()

        db_path = self.index.db_path
        self.index.close()
        reopened_index = SearchIndex(db_path)

        try:
            meta_val = reopened_index.connection.execute(
                "SELECT value FROM index_meta WHERE key='message_cjk_ready'"
            ).fetchone()
            self.assertIsNotNone(meta_val)
            self.assertEqual(meta_val[0], "ready")

            res_after = reopened_index.search("事务一致性", mode="smart", account_id=acc)
            self.assertEqual(res_after["total"], 1)
            self.assertEqual(res_after["results"][0]["message_id"], "msg-cjk-1")

            res_after2 = reopened_index.search("分布式共识协议", mode="smart", account_id=acc)
            self.assertGreaterEqual(res_after2["total"], 1)
            self.assertEqual(res_after2["results"][0]["message_id"], "msg-cjk-2")

            reopened_messages = list(reopened_index.connection.execute("SELECT id, content_hash, text FROM messages WHERE account_id=?", (acc,)))
            self.assertEqual([dict(r) for r in orig_messages], [dict(r) for r in reopened_messages])
        finally:
            reopened_index.close()
            self.index = SearchIndex(db_path)

    def test_cjk_stop_phrases_do_not_leak_cross_boundary_ngrams(self):
        terms = cjk_lexical_terms("为什么这个可以")
        self.assertEqual(terms, [])

        payload = {
            "id": "leak-conv",
            "messages": [
                {"id": "doc-1", "role": "assistant", "text": "关于这个主题的深入讨论，解释为什么会产生这样的现象"},
                {"id": "doc-2", "role": "assistant", "text": "你可以通过配置参数优化性能，不仅这个模块可以，那个模块也可以"},
                {"id": "doc-target", "role": "assistant", "text": "系统性能调优：排查解决本地模型内存压力和过载问题"},
            ]
        }
        self.index.import_payload(payload, account_id="leak-acc")

        res = self.index.search("为什么这个可以", mode="smart", account_id="leak-acc")
        self.assertEqual(res["total"], 0)
        self.assertEqual(res["results"], [])

        terms_and = cjk_lexical_terms("为什么这个和那个可以")
        self.assertEqual(terms_and, [])

        mixed_terms = cjk_lexical_terms("如何配置 ERR_CFG_204 和 内存占用")
        self.assertIn("ERR_CFG_204", mixed_terms)
        self.assertIn("配置", mixed_terms)
        self.assertIn("内存", mixed_terms)
        self.assertIn("占用", mixed_terms)
        self.assertNotIn("如何", mixed_terms)
        for forbidden in ("如何配", "何配", "和内", "和内存"):
            self.assertNotIn(forbidden, mixed_terms)

    def test_large_message_bounded_memory_and_chunk_boundary_preservation(self):
        chunk_pattern = "数据库读写分离架构与分布式缓存一致性保证机制" * 30
        prefix_block = chunk_pattern * 2
        boundary_concept = "跨边界微服务治理框架"
        filler = chunk_pattern * 20
        full_text = prefix_block + boundary_concept + filler

        payload = {
            "id": "large-msg-conv",
            "messages": [
                {"id": "large-msg", "role": "assistant", "text": full_text},
            ]
        }
        self.index.import_payload(payload, account_id="large-acc")

        row = self.index.connection.execute(
            "SELECT id FROM messages WHERE message_id='large-msg' AND account_id='large-acc'"
        ).fetchone()
        self.assertIsNotNone(row)
        msg_row_id = row[0]

        segment_count = self.index.connection.execute(
            "SELECT COUNT(*) FROM message_cjk_segments WHERE message_row=?", (msg_row_id,)
        ).fetchone()[0]
        self.assertGreater(segment_count, 1)

        res_boundary = self.index.search("微服务治理", mode="smart", account_id="large-acc")
        self.assertEqual(res_boundary["total"], 1)
        self.assertEqual(res_boundary["results"][0]["message_id"], "large-msg")

        res_boundary2 = self.index.search("跨边界", mode="smart", account_id="large-acc")
        self.assertEqual(res_boundary2["total"], 1)
        self.assertEqual(res_boundary2["results"][0]["message_id"], "large-msg")

        res_multi = self.index.search("读写分离", mode="smart", account_id="large-acc")
        self.assertEqual(res_multi["total"], 1)
        self.assertEqual(len(res_multi["results"]), 1)
        self.assertEqual(res_multi["results"][0]["message_id"], "large-msg")

    def test_paragraph_windows_avoid_whitespace_only_embeddings(self):
        p1 = "A" * 512
        p2 = "B" * 10
        text = f"{p1}\n\n{p2}"

        spans = SearchIndex._paragraph_aware_spans(text, max_chunk_bytes=512)
        for s, e in spans:
            chunk = text[s:e]
            self.assertTrue(chunk.strip(), f"Span ({s}, {e}) text {chunk!r} must not be whitespace-only")
            self.assertLessEqual(len(chunk.encode("utf-8")), 512)

        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))
        for i in range(len(spans) - 1):
            self.assertEqual(spans[i][1], spans[i + 1][0])
            self.assertLess(spans[i][0], spans[i][1])

        span_lens = [e - s for s, e in spans]
        self.assertEqual(span_lens, [512, 12])

        payload = {
            "id": "whitespace-avoid-conv",
            "messages": [
                {"id": "msg-ws", "role": "assistant", "text": text},
            ]
        }
        self.index.import_payload(payload, account_id="ws-acc")
        row = self.index.connection.execute(
            "SELECT * FROM messages WHERE message_id='msg-ws' AND account_id='ws-acc'"
        ).fetchone()

        windows = self.index._windows(row, max_input_bytes=960)
        self.assertEqual(len(windows), 2)
        for ws, we, derived in windows:
            target_chunk = row["text"][ws:we]
            self.assertTrue(target_chunk.strip())
            self.assertLessEqual(len(derived.encode("utf-8")), 960)

        trailing_text = "A" * 512 + "\n\n"
        trailing_spans = SearchIndex._paragraph_aware_spans(trailing_text, max_chunk_bytes=512)
        for s, e in trailing_spans:
            chunk = trailing_text[s:e]
            self.assertTrue(chunk.strip(), f"Span ({s}, {e}) text {chunk!r} must not be whitespace-only")
            self.assertLessEqual(len(chunk.encode("utf-8")), 512)
        self.assertEqual(trailing_spans[0][0], 0)
        self.assertEqual(trailing_spans[-1][1], len(trailing_text))

        # Long ASCII whitespace gap (600 spaces): contiguous coverage, bounded spans <= 512, windows skip ws
        long_ascii_ws = ("A" * 512) + "\n\n" + (" " * 600) + "\n\n" + ("B" * 50)
        ascii_spans = SearchIndex._paragraph_aware_spans(long_ascii_ws, max_chunk_bytes=512)
        self.assertEqual(ascii_spans[0][0], 0)
        self.assertEqual(ascii_spans[-1][1], len(long_ascii_ws))
        for i in range(len(ascii_spans) - 1):
            self.assertEqual(ascii_spans[i][1], ascii_spans[i + 1][0])
            self.assertLess(ascii_spans[i][0], ascii_spans[i + 1][1])
        for s, e in ascii_spans:
            self.assertLessEqual(len(long_ascii_ws[s:e].encode("utf-8")), 512)

        payload_ascii_ws = {
            "id": "conv-ascii-ws",
            "messages": [{"id": "msg-ascii-ws", "role": "assistant", "text": long_ascii_ws}],
        }
        self.index.import_payload(payload_ascii_ws, account_id="ws-acc")
        row_ascii = self.index.connection.execute(
            "SELECT * FROM messages WHERE message_id='msg-ascii-ws' AND account_id='ws-acc'"
        ).fetchone()
        ascii_windows = self.index._windows(row_ascii, max_input_bytes=960)
        self.assertGreater(len(ascii_windows), 0)
        for ws, we, derived in ascii_windows:
            self.assertTrue(row_ascii["text"][ws:we].strip())
            self.assertLessEqual(len(derived.encode("utf-8")), 960)

        # Long multibyte whitespace gap (ideographic space \u3000): contiguous coverage, bounded spans <= 512
        multibyte_ws = ("A" * 500) + "\n\n" + ("\u3000" * 300) + "\n\n" + ("B" * 50)
        mb_spans = SearchIndex._paragraph_aware_spans(multibyte_ws, max_chunk_bytes=512)
        self.assertEqual(mb_spans[0][0], 0)
        self.assertEqual(mb_spans[-1][1], len(multibyte_ws))
        for i in range(len(mb_spans) - 1):
            self.assertEqual(mb_spans[i][1], mb_spans[i + 1][0])
            self.assertLess(mb_spans[i][0], mb_spans[i + 1][1])
        for s, e in mb_spans:
            self.assertLessEqual(len(multibyte_ws[s:e].encode("utf-8")), 512)

        payload_mb_ws = {
            "id": "conv-mb-ws",
            "messages": [{"id": "msg-mb-ws", "role": "assistant", "text": multibyte_ws}],
        }
        self.index.import_payload(payload_mb_ws, account_id="ws-acc")
        row_mb = self.index.connection.execute(
            "SELECT * FROM messages WHERE message_id='msg-mb-ws' AND account_id='ws-acc'"
        ).fetchone()
        mb_windows = self.index._windows(row_mb, max_input_bytes=960)
        self.assertGreater(len(mb_windows), 0)
        for ws, we, derived in mb_windows:
            self.assertTrue(row_mb["text"][ws:we].strip())
            self.assertLessEqual(len(derived.encode("utf-8")), 960)

        # Leading whitespace (600 spaces)
        leading_ws = (" " * 600) + "Content after leading whitespace"
        lead_spans = SearchIndex._paragraph_aware_spans(leading_ws, max_chunk_bytes=512)
        self.assertEqual(lead_spans[0][0], 0)
        self.assertEqual(lead_spans[-1][1], len(leading_ws))
        for i in range(len(lead_spans) - 1):
            self.assertEqual(lead_spans[i][1], lead_spans[i + 1][0])
            self.assertLess(lead_spans[i][0], lead_spans[i + 1][1])
        for s, e in lead_spans:
            self.assertLessEqual(len(leading_ws[s:e].encode("utf-8")), 512)

        payload_lead = {
            "id": "conv-lead-ws",
            "messages": [{"id": "msg-lead-ws", "role": "assistant", "text": leading_ws}],
        }
        self.index.import_payload(payload_lead, account_id="ws-acc")
        row_lead = self.index.connection.execute(
            "SELECT * FROM messages WHERE message_id='msg-lead-ws' AND account_id='ws-acc'"
        ).fetchone()
        lead_windows = self.index._windows(row_lead, max_input_bytes=960)
        self.assertEqual(len(lead_windows), 1)
        self.assertEqual(row_lead["text"][lead_windows[0][0]:lead_windows[0][1]].strip(), "Content after leading whitespace")
        self.assertLessEqual(len(lead_windows[0][2].encode("utf-8")), 960)

    def test_cjk_term_after_embedded_nul_and_across_chunk_boundary(self):
        # 1. Embedded NUL: source text contains '\0' before CJK terms
        nul_text = "before\x00本地模型内存压力讨论，显存分配优化方案。"
        payload_nul = {
            "id": "nul-cjk-conv",
            "messages": [
                {"id": "msg-nul-1", "role": "assistant", "text": nul_text},
            ],
        }
        self.index.import_payload(payload_nul, account_id="nul-acc")

        # Smart mode: CJK terms after embedded NUL must be retrieved via message_cjk
        res_cjk = self.index.search("内存压力", mode="smart", account_id="nul-acc")
        self.assertGreater(res_cjk["total"], 0)
        self.assertEqual(res_cjk["results"][0]["message_id"], "msg-nul-1")
        self.assertIn("keyword", res_cjk["results"][0]["retrieval_channels"])

        # Exact mode: literal term before NUL is matched
        res_exact = self.index.search("before", mode="exact", account_id="nul-acc")
        self.assertEqual(res_exact["total"], 1)
        self.assertEqual(res_exact["results"][0]["message_id"], "msg-nul-1")

        # 2. CJK term across chunk boundary (4096 bytes)
        pad = "数据库读写分离与高可用集群架构设计实践。" * 120
        boundary_concept = "跨边界微服务治理"
        chunk_edge_text = pad[:4090] + boundary_concept + pad[4090:]

        payload_edge = {
            "id": "edge-cjk-conv",
            "messages": [
                {"id": "msg-edge-1", "role": "assistant", "text": chunk_edge_text},
            ],
        }
        self.index.import_payload(payload_edge, account_id="nul-acc")

        res_edge1 = self.index.search("跨边界", mode="smart", account_id="nul-acc")
        self.assertGreater(res_edge1["total"], 0)
        self.assertEqual(res_edge1["results"][0]["message_id"], "msg-edge-1")

        res_edge2 = self.index.search("微服务治理", mode="smart", account_id="nul-acc")
        self.assertGreater(res_edge2["total"], 0)
        self.assertEqual(res_edge2["results"][0]["message_id"], "msg-edge-1")

        # Role and account isolation
        res_iso_acc = self.index.search("内存压力", mode="smart", account_id="other-acc")
        self.assertEqual(res_iso_acc["total"], 0)
        res_iso_role = self.index.search("内存压力", mode="smart", account_id="nul-acc", role="user")
        self.assertEqual(res_iso_role["total"], 0)

    def test_mixed_language_identifier_cross_chunk_seam_retrieval(self):
        # Long synthetic English identifier placed right across the 4096-byte chunk boundary
        long_identifier = "SYNTHETIC_CROSS_BOUNDARY_UNIQUEIDENTIFIER_XYZ999"
        pad = "分布式数据库系统事务一致性与共识算法详解。" * 120
        prefix_len = 4096 - len(long_identifier) // 2
        seam_text = pad[:prefix_len] + long_identifier + " 讨论本地模型分布式部署方案。" + pad[prefix_len:]

        payload = {
            "id": "seam-conv",
            "messages": [
                {"id": "msg-seam-target", "role": "assistant", "text": seam_text},
                {"id": "msg-seam-other", "role": "assistant", "text": "普通不相关内容：西红柿炒鸡蛋烹饪食谱。"},
            ],
        }
        self.index.import_payload(payload, account_id="seam-acc")

        # Mixed query: long identifier + CJK keyword
        mixed_query = f"{long_identifier} 分布式部署"
        res = self.index.search(mixed_query, mode="smart", account_id="seam-acc")
        self.assertGreater(res["total"], 0)
        self.assertEqual(res["results"][0]["message_id"], "msg-seam-target")
        self.assertIn("keyword", res["results"][0]["retrieval_channels"])

        # Identifier-only query
        id_res = self.index.search(long_identifier, mode="smart", account_id="seam-acc")
        self.assertGreater(id_res["total"], 0)
        self.assertEqual(id_res["results"][0]["message_id"], "msg-seam-target")

        # Role and account filters preserved
        user_res = self.index.search(mixed_query, mode="smart", account_id="seam-acc", role="user")
        self.assertEqual(user_res["total"], 0)
        other_acc_res = self.index.search(mixed_query, mode="smart", account_id="different-acc")
        self.assertEqual(other_acc_res["total"], 0)

    def test_document_profile_version_v2_invalidation(self):
        # DOCUMENT_PROFILE version is v2
        self.assertEqual(SearchIndex.DOCUMENT_PROFILE["version"], "v2")
        self.assertEqual(SearchIndex.DOCUMENT_PROFILE["max_input_bytes"], 960)
        self.assertEqual(SearchIndex.DOCUMENT_PROFILE["prefix_bytes"], 384)
        self.assertEqual(SearchIndex.DOCUMENT_PROFILE["max_chunk_bytes"], 512)

    def test_enrich_anti_join_does_not_use_rows(self):
        with patch.object(self.index, "_rows", side_effect=AssertionError("_rows must not be called in enrich")):
            report = self.index.enrich(ToyProvider(), limit=2)
            self.assertEqual(report["attempted"], 2)
            self.assertEqual(report["processed"], 2)

    def test_disjoint_account_batches_and_coverage(self):
        p1 = {
            "id": "c-acc1",
            "messages": [
                {"id": "m-acc1-1", "role": "user", "text": "Message 1 for account 1"},
                {"id": "m-acc1-2", "role": "assistant", "text": "Message 2 for account 1"},
            ]
        }
        p2 = {
            "id": "c-acc2",
            "messages": [
                {"id": "m-acc2-1", "role": "user", "text": "Message 1 for account 2"},
                {"id": "m-acc2-2", "role": "assistant", "text": "Message 2 for account 2"},
                {"id": "m-acc2-3", "role": "user", "text": "Message 3 for account 2"},
            ]
        }
        self.index.import_payload(p1, account_id="acc1")
        self.index.import_payload(p2, account_id="acc2")
        provider = ToyProvider()
        self.index.embedding_provider = provider

        # Batch enrich acc1
        rep1 = self.index.enrich(provider, account_id="acc1", limit=10)
        self.assertEqual(rep1["processed"], 2)

        # Status for acc1
        st1 = self.index.status(account_id="acc1")
        cov1 = st1["semantic"]["coverage"]
        self.assertEqual(cov1["account_id"], "acc1")
        self.assertEqual(cov1["eligible"], 2)
        self.assertEqual(cov1["current"], 2)
        self.assertEqual(cov1["pending"], 0)
        self.assertEqual(cov1["legacy"], 0)

        # Status for acc2
        st2 = self.index.status(account_id="acc2")
        cov2 = st2["semantic"]["coverage"]
        self.assertEqual(cov2["account_id"], "acc2")
        self.assertEqual(cov2["eligible"], 3)
        self.assertEqual(cov2["current"], 0)
        self.assertEqual(cov2["pending"], 3)
        self.assertEqual(cov2["legacy"], 0)

        # Add a legacy vector with a different model_id for an acc1 message
        row1 = self.index.connection.execute("SELECT id, content_hash FROM messages WHERE account_id='acc1' LIMIT 1").fetchone()
        self.index.connection.execute(
            "INSERT INTO vectors VALUES (?, 'dialogue', 0, 'legacy-model', ?, 0, 10, 'der', '\"[0, 1]\"')",
            (row1["id"], row1["content_hash"])
        )
        self.index.connection.commit()

        # Stale/legacy counted correctly
        st1_updated = self.index.status(account_id="acc1")
        self.assertEqual(st1_updated["semantic"]["coverage"]["legacy"], 1)

    def test_pre_pause_zero_model_calls(self):
        provider = MagicMock()
        provider.model_id = "mock-model"
        report = self.index.enrich(provider, should_stop=lambda: True)
        self.assertTrue(report["paused"])
        self.assertEqual(report["stopped_reason"], "paused")
        self.assertEqual(report["attempted"], 0)
        self.assertEqual(report["processed"], 0)
        provider.embed.assert_not_called()
        provider.embed_documents.assert_not_called()
        provider.summarize.assert_not_called()
        provider.resolve_identity.assert_not_called()

    def test_mid_multiwindow_pause_no_next_request_no_partial_commit(self):
        p = {
            "id": "c-multiwin",
            "messages": [
                {
                    "id": "m-multi",
                    "role": "user",
                    "text": ("Paragraph one with tomato soup details.\n\n" * 10) + ("Paragraph two with soup acidity.\n\n" * 10),
                }
            ]
        }
        self.index.import_payload(p, account_id="mw-acc")
        msg_row = self.index.connection.execute("SELECT id FROM messages WHERE message_id='m-multi'").fetchone()["id"]

        call_count = 0
        stop_state = {"stop": False}

        class PausingProvider:
            model_id = "test-pausing-model"
            def embed_documents(self, texts, index_id=None):
                nonlocal call_count
                call_count += 1
                stop_state["stop"] = True  # Pause requested after first batch
                return [[1.0, 0.0] for _ in texts]

        report = self.index.enrich(PausingProvider(), account_id="mw-acc", limit=1, should_stop=lambda: stop_state["stop"])
        self.assertTrue(report["paused"])
        self.assertEqual(report["stopped_reason"], "paused")
        self.assertEqual(call_count, 1)

        # No partial vectors committed for m-multi
        vec_count = self.index.connection.execute("SELECT COUNT(*) FROM vectors WHERE message_row=?", (msg_row,)).fetchone()[0]
        self.assertEqual(vec_count, 0)
        enrich_count = self.index.connection.execute("SELECT COUNT(*) FROM enrichment WHERE message_row=?", (msg_row,)).fetchone()[0]
        self.assertEqual(enrich_count, 0)

    def test_fake_provider_can_acquire_sqlite_write_lock_during_inference(self):
        db_path = self.index.db_path

        class LockCheckingProvider:
            model_id = "test-lock-model"
            def embed_documents(self, texts, index_id=None):
                conn2 = sqlite3.connect(db_path, timeout=5)
                try:
                    conn2.execute("UPDATE conversations SET title='Modified by provider' WHERE conversation_id='synthetic-cipher'")
                    conn2.commit()
                finally:
                    conn2.close()
                return [[1.0, 0.0] for _ in texts]

        report = self.index.enrich(LockCheckingProvider(), limit=1)
        self.assertGreaterEqual(report["processed"], 1)
        title = self.index.connection.execute("SELECT title FROM conversations WHERE conversation_id='synthetic-cipher'").fetchone()[0]
        self.assertEqual(title, "Modified by provider")

    def test_concurrent_source_edit_prevents_stale_commit(self):
        db_path = self.index.db_path
        target_id = self.index.connection.execute("SELECT id FROM messages ORDER BY id LIMIT 1").fetchone()["id"]

        class MutatingProvider:
            model_id = "test-mutating-model"
            def embed_documents(self, texts, index_id=None):
                conn2 = sqlite3.connect(db_path, timeout=5)
                try:
                    conn2.execute("UPDATE messages SET text='Concurrently changed text', content_hash='corrupted_hash' WHERE id=?", (target_id,))
                    conn2.commit()
                finally:
                    conn2.close()
                return [[1.0, 0.0] for _ in texts]

        report = self.index.enrich(MutatingProvider(), limit=1)
        vecs = self.index.connection.execute("SELECT COUNT(*) FROM vectors WHERE message_row=?", (target_id,)).fetchone()[0]
        self.assertEqual(vecs, 0)
        self.assertEqual(report["processed"], 0)

    def test_strict_validations_and_cursors(self):
        provider = ToyProvider()
        # Invalid limits
        with self.assertRaises(ValueError):
            self.index.enrich(provider, limit=True)
        with self.assertRaises(ValueError):
            self.index.enrich(provider, limit="5")
        with self.assertRaises(ValueError):
            self.index.enrich(provider, limit=0)

        # Invalid generate_topics
        with self.assertRaises(ValueError):
            self.index.enrich(provider, generate_topics="false")
        with self.assertRaises(ValueError):
            self.index.enrich(provider, generate_topics=1)

        # Invalid account_id
        with self.assertRaises(ValueError):
            self.index.enrich(provider, account_id=True)
        with self.assertRaises(ValueError):
            self.index.status(account_id=True)

        # Malformed after_cursor
        with self.assertRaises(ValueError):
            self.index.enrich(provider, after_cursor="invalid")
        with self.assertRaises(ValueError):
            self.index.enrich(provider, after_cursor={"after_id": -1, "model_id": provider.model_id, "account_id": None, "generate_topics": False})
        with self.assertRaises(ValueError):
            self.index.enrich(provider, after_cursor={"after_id": True, "model_id": provider.model_id, "account_id": None, "generate_topics": False})

        # Mismatched model_id
        with self.assertRaises(ValueError):
            self.index.enrich(provider, after_cursor={"after_id": 0, "model_id": "wrong-model", "account_id": None, "generate_topics": False})

        # Mismatched account_id
        with self.assertRaises(ValueError):
            self.index.enrich(provider, account_id="acc-x", after_cursor={"after_id": 0, "model_id": provider.model_id, "account_id": "acc-y", "generate_topics": False})

        # Mismatched generate_topics
        with self.assertRaises(ValueError):
            self.index.enrich(provider, generate_topics=False, after_cursor={"after_id": 0, "model_id": provider.model_id, "account_id": None, "generate_topics": True})

    def test_oversized_first_row_does_not_starve_next_batch(self):
        p = {
            "id": "c-oversize",
            "messages": [
                {"id": "m-huge", "role": "user", "text": "H" * 300000},
                {"id": "m-normal", "role": "assistant", "text": "Normal size reply"},
            ]
        }
        self.index.import_payload(p, account_id="over-acc")
        provider = ToyProvider()

        rep1 = self.index.enrich(provider, limit=1, account_id="over-acc")
        self.assertEqual(rep1["attempted"], 1)
        self.assertEqual(rep1["processed"], 0)
        self.assertEqual(rep1["skipped_budget"], 1)
        self.assertIsNotNone(rep1["next_cursor"])
        huge_id = self.index.connection.execute("SELECT id FROM messages WHERE message_id='m-huge'").fetchone()["id"]
        self.assertEqual(rep1["next_cursor"]["after_id"], huge_id)

        rep2 = self.index.enrich(provider, limit=1, account_id="over-acc", after_cursor=rep1["next_cursor"])
        self.assertEqual(rep2["attempted"], 1)
        self.assertEqual(rep2["processed"], 1)
        self.assertEqual(rep2["skipped_budget"], 0)

    def test_progress_callbacks_no_model_work(self):
        provider = ToyProvider()
        calls = []
        def on_progress(p):
            calls.append(dict(p))

        report = self.index.enrich(provider, limit=2, progress=on_progress)
        self.assertGreater(len(calls), 0)
        for c in calls:
            self.assertIn("attempted", c)
            self.assertIn("processed", c)
            self.assertIn("skipped_budget", c)

    def test_parent_context_huge_sibling_isolated_context_raises(self):
        p = {
            "id": "c-tree",
            "messages": [
                {"id": "m-q", "role": "user", "text": "Which deployment region?"},
                {"id": "m-a1", "parent_id": "m-q", "role": "assistant", "text": "Region us-east-1 recommended."},
                {"id": "m-a2", "parent_id": "m-q", "role": "assistant", "text": "Alternative region eu-west-1 selected."},
            ]
        }
        self.index.import_payload(p, account_id="tree-acc")
        provider = ToyProvider()

        with patch.object(self.index, "context", side_effect=AssertionError("indexing must never call context")):
            rep = self.index.enrich(provider, limit=10, account_id="tree-acc")
            self.assertEqual(rep["processed"], 3)

        a1_row = self.index.connection.execute("SELECT id FROM messages WHERE message_id='m-a1'").fetchone()["id"]
        a1_vec = self.index.connection.execute("SELECT derived_text FROM vectors WHERE message_row=?", (a1_row,)).fetchone()
        self.assertIn("Which deployment region?", a1_vec["derived_text"])
        self.assertIn("Region us-east-1", a1_vec["derived_text"])
        self.assertNotIn("Alternative region eu-west-1", a1_vec["derived_text"])

    def test_cancellation_after_operation_lock_and_blocked_lock(self):
        class LockableProvider:
            model_id = "test-oplock-model"
            def __init__(self):
                self.operation_lock = threading.RLock()
                self.call_count = 0
            def embed_documents(self, texts, index_id=None):
                self.call_count += 1
                return [[1.0, 0.0] for _ in texts]

        provider = LockableProvider()
        stop_state = {"stop": False}
        thread_started = threading.Event()
        enrich_result = {}
        worker_exc = []

        provider.operation_lock.acquire()

        def worker():
            thread_started.set()
            try:
                with SearchIndex(self.index.db_path, embedding_provider=provider) as worker_idx:
                    enrich_result["report"] = worker_idx.enrich(
                        provider, limit=1, should_stop=lambda: stop_state["stop"]
                    )
            except Exception as e:
                worker_exc.append(e)

        t = threading.Thread(target=worker)
        try:
            t.start()
            self.assertTrue(thread_started.wait(timeout=2))
            time.sleep(0.05)
            stop_state["stop"] = True
        finally:
            if provider.operation_lock._is_owned():
                provider.operation_lock.release()
            t.join(timeout=3)

        if worker_exc:
            raise worker_exc[0]

        self.assertFalse(t.is_alive())
        report = enrich_result.get("report")
        self.assertIsNotNone(report)
        self.assertTrue(report["paused"])
        self.assertEqual(report["stopped_reason"], "paused")
        self.assertEqual(provider.call_count, 0)

    def test_bounded_time_and_requests(self):
        provider = ToyProvider()

        with patch("time.monotonic", side_effect=[0.0, 0.0, 200.0, 200.0, 200.0]):
            rep_time = self.index.enrich(provider, limit=10, max_time_seconds=1.0)
            self.assertEqual(rep_time["stopped_reason"], "timeout")

        call_count = 0
        class CountingProvider:
            model_id = "test-count-model"
            def embed_documents(self, texts, index_id=None):
                nonlocal call_count
                call_count += 1
                return [[1.0, 0.0] for _ in texts]

        rep_req = self.index.enrich(CountingProvider(), limit=10, max_requests=2)
        self.assertEqual(rep_req["stopped_reason"], "request_budget")
        self.assertEqual(call_count, 2)

    def test_systemic_provider_error_stops_batch_and_cursor_retries(self):
        p = {
            "id": "c-syserr",
            "messages": [
                {"id": f"m-err-{i}", "role": "user", "text": f"Error message {i}"}
                for i in range(5)
            ]
        }
        self.index.import_payload(p, account_id="syserr-acc")

        calls = 0
        class FailingProvider:
            model_id = "test-fail-model"
            def embed_documents(self, texts, index_id=None):
                nonlocal calls
                calls += 1
                raise ProviderError("Systemic network failure")

        rep = self.index.enrich(FailingProvider(), limit=5, account_id="syserr-acc")
        self.assertEqual(rep["stopped_reason"], "provider_error")
        self.assertEqual(calls, 1)
        self.assertEqual(len(rep["errors"]), 1)
        m0_id = self.index.connection.execute("SELECT id FROM messages WHERE message_id='m-err-0'").fetchone()["id"]
        if rep["next_cursor"] is not None:
            self.assertLess(rep["next_cursor"]["after_id"], m0_id)

    def test_large_corpus_memory_scaling(self):
        provider = ToyProvider()

        self.index.import_payload({
            "id": "c-small",
            "messages": [{"id": f"m-sm-{i}", "role": "user", "text": "Small corpus message body " * 5} for i in range(5)]
        }, account_id="mem-sm")

        tracemalloc.start()
        snapshot_start1 = tracemalloc.take_snapshot()
        self.index.enrich(provider, limit=2, account_id="mem-sm")
        snapshot_end1 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats1 = snapshot_end1.compare_to(snapshot_start1, 'lineno')
        mem1 = sum(stat.size_diff for stat in stats1 if stat.size_diff > 0)

        self.index.import_payload({
            "id": "c-large",
            "messages": [{"id": f"m-lg-{i}", "role": "user", "text": "Large corpus message body " * 5} for i in range(50)]
        }, account_id="mem-lg")

        tracemalloc.start()
        snapshot_start2 = tracemalloc.take_snapshot()
        self.index.enrich(provider, limit=2, account_id="mem-lg")
        snapshot_end2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats2 = snapshot_end2.compare_to(snapshot_start2, 'lineno')
        mem2 = sum(stat.size_diff for stat in stats2 if stat.size_diff > 0)

        self.assertLess(abs(mem2 - mem1), 200000)

    def test_pause_during_final_inference_no_commit(self):
        p = {
            "id": "c-pause-final",
            "messages": [
                {"id": "m-single", "role": "user", "text": "Single window short text."}
            ]
        }
        self.index.import_payload(p, account_id="pause-final-acc")
        msg_id = self.index.connection.execute("SELECT id FROM messages WHERE message_id='m-single'").fetchone()["id"]

        stop_state = {"stop": False}
        class PauseOnFinalProvider:
            model_id = "test-pause-final-model"
            def embed_documents(self, texts, index_id=None):
                stop_state["stop"] = True
                return [[1.0, 0.0] for _ in texts]

        rep = self.index.enrich(PauseOnFinalProvider(), limit=1, account_id="pause-final-acc", should_stop=lambda: stop_state["stop"])
        self.assertTrue(rep["paused"])
        self.assertEqual(rep["stopped_reason"], "paused")
        self.assertEqual(rep["processed"], 0)
        vec_count = self.index.connection.execute("SELECT COUNT(*) FROM vectors WHERE message_row=?", (msg_id,)).fetchone()[0]
        self.assertEqual(vec_count, 0)
        enrich_count = self.index.connection.execute("SELECT COUNT(*) FROM enrichment WHERE message_row=?", (msg_id,)).fetchone()[0]
        self.assertEqual(enrich_count, 0)

    def test_search_index_exposes_enrich_status_windows_and_enriches_tiny_message(self):
        self.assertTrue(callable(getattr(SearchIndex, "enrich", None)))
        self.assertTrue(callable(getattr(SearchIndex, "status", None)))
        self.assertTrue(callable(getattr(SearchIndex, "_windows", None)))
        tiny_payload = {
            "id": "conv-tiny",
            "messages": [
                {"id": "msg-tiny-1", "role": "user", "text": "Tiny tomato soup recipe."}
            ]
        }
        self.index.import_payload(tiny_payload, account_id="tiny-acc")
        provider = ToyProvider()
        self.index.embedding_provider = provider
        windows = self.index._windows({"id": 1, "text": "Tiny tomato soup recipe.", "role": "user", "message_id": "msg-tiny-1", "parent_id": None, "source_id": "chatgpt", "account_id": "tiny-acc", "conversation_id": "conv-tiny"})
        self.assertGreaterEqual(len(windows), 1)
        rep = self.index.enrich(provider, limit=1, account_id="tiny-acc")
        self.assertEqual(rep["processed"], 1)
        self.assertEqual(rep["remaining"], 0)
        st = self.index.status(account_id="tiny-acc")
        self.assertEqual(st["semantic"]["coverage"]["current"], 1)
        self.assertEqual(st["semantic"]["coverage"]["pending"], 0)

    def test_pause_while_publication_lock_waits(self):
        p = {
            "id": "c-pub-wait",
            "messages": [
                {"id": "m-pub-1", "role": "user", "text": "Testing publication lock wait"}
            ]
        }
        self.index.import_payload(p, account_id="pub-acc")
        msg_id = self.index.connection.execute("SELECT id FROM messages WHERE message_id='m-pub-1'").fetchone()["id"]

        stop_state = {"stop": False}
        lock_held_event = threading.Event()
        release_lock_event = threading.Event()
        worker_exc = []

        def lock_holder():
            # Create/close SQLite connection IN worker thread for blocked-lock tests and propagate thread exceptions
            try:
                conn = sqlite3.connect(self.index.db_path, timeout=5)
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    lock_held_event.set()
                    release_lock_event.wait(timeout=5)
                    conn.execute("ROLLBACK")
                finally:
                    conn.close()
            except Exception as e:
                worker_exc.append(e)

        t_holder = threading.Thread(target=lock_holder)

        enrich_res = {}
        enrich_exc = []
        started_enrich = threading.Event()
        begin_called = threading.Event()

        class SyncProvider:
            model_id = "test-pub-wait-model"
            def embed_documents(self, texts, index_id=None):
                started_enrich.set()
                return [[1.0, 0.0] for _ in texts]

        provider = SyncProvider()

        def trace_sql(sql):
            if "BEGIN" in sql:
                begin_called.set()

        def enrich_worker():
            try:
                with SearchIndex(self.index.db_path, embedding_provider=provider) as worker_idx:
                    worker_idx.connection.set_trace_callback(trace_sql)
                    enrich_res["report"] = worker_idx.enrich(
                        provider, limit=1, account_id="pub-acc",
                        should_stop=lambda: stop_state["stop"]
                    )
            except Exception as e:
                enrich_exc.append(e)

        t_enrich = threading.Thread(target=enrich_worker)

        try:
            t_holder.start()
            self.assertTrue(lock_held_event.wait(timeout=2))

            t_enrich.start()
            self.assertTrue(started_enrich.wait(timeout=2))
            self.assertTrue(begin_called.wait(timeout=2))

            # Signal stop while publication lock is waiting
            stop_state["stop"] = True
        finally:
            release_lock_event.set()
            t_holder.join(timeout=3)
            t_enrich.join(timeout=3)

        if worker_exc:
            raise worker_exc[0]
        if enrich_exc:
            raise enrich_exc[0]

        report = enrich_res.get("report")
        self.assertIsNotNone(report)
        self.assertTrue(report["paused"])
        self.assertEqual(report["stopped_reason"], "paused")
        self.assertEqual(report["processed"], 0)

        # No vectors or enrichment committed for m-pub-1
        vec_count = self.index.connection.execute("SELECT COUNT(*) FROM vectors WHERE message_row=?", (msg_id,)).fetchone()[0]
        self.assertEqual(vec_count, 0)
        enr_count = self.index.connection.execute("SELECT COUNT(*) FROM enrichment WHERE message_row=?", (msg_id,)).fetchone()[0]
        self.assertEqual(enr_count, 0)

    def test_pause_while_resolver_lock_waits(self):
        class LockableResolverProvider:
            model_id = "test-resolver-lock-model"
            def __init__(self):
                self.operation_lock = threading.RLock()
                self.resolve_called = False
                self.embed_called = False
            def resolve_identity(self, profile):
                self.resolve_called = True
                return "resolved-id"
            def embed_documents(self, texts, index_id=None):
                self.embed_called = True
                return [[1.0, 0.0] for _ in texts]

        provider = LockableResolverProvider()
        stop_state = {"stop": False}
        thread_started = threading.Event()
        worker_exc = []
        enrich_result = {}

        provider.operation_lock.acquire()

        def worker():
            thread_started.set()
            try:
                with SearchIndex(self.index.db_path, embedding_provider=provider) as worker_idx:
                    enrich_result["report"] = worker_idx.enrich(
                        provider, limit=1, should_stop=lambda: stop_state["stop"]
                    )
            except Exception as e:
                worker_exc.append(e)

        t = threading.Thread(target=worker)
        try:
            t.start()
            self.assertTrue(thread_started.wait(timeout=2))
            time.sleep(0.05)
            # Stop while resolver lock is being waited on
            stop_state["stop"] = True
        finally:
            if provider.operation_lock._is_owned():
                provider.operation_lock.release()
            t.join(timeout=3)

        if worker_exc:
            raise worker_exc[0]

        self.assertFalse(t.is_alive())
        report = enrich_result.get("report")
        self.assertIsNotNone(report)
        self.assertTrue(report["paused"])
        self.assertEqual(report["stopped_reason"], "paused")
        self.assertFalse(provider.resolve_called)
        self.assertFalse(provider.embed_called)

    def test_deterministic_source_growth_after_metadata_guard(self):
        p = {
            "id": "c-growth",
            "messages": [
                {"id": "m-grow-1", "role": "user", "text": "Small text that will grow concurrently"},
                {"id": "m-grow-2", "role": "assistant", "text": "Second message that should be processed later"},
            ]
        }
        self.index.import_payload(p, account_id="growth-acc")
        m1_id = self.index.connection.execute("SELECT id FROM messages WHERE message_id='m-grow-1'").fetchone()["id"]

        class GrowthSimulatingProvider:
            model_id = "test-growth-model"
            def __init__(self, db_path):
                self.db_path = db_path
                self.call_count = 0
            def embed_documents(self, texts, index_id=None):
                self.call_count += 1
                return [[1.0, 0.0] for _ in texts]

        provider = GrowthSimulatingProvider(self.index.db_path)

        real_conn = self.index.connection
        grown = False

        class GrowthInterceptingConnection:
            def __init__(self, real, db_path):
                self._real = real
                self._db_path = db_path

            def execute(self, sql, *args):
                nonlocal grown
                if "cur_bytes FROM messages WHERE id=?" in sql and not grown:
                    grown = True
                    conn2 = sqlite3.connect(self._db_path, timeout=5)
                    try:
                        conn2.execute(
                            "UPDATE messages SET text=?, content_hash='grown_hash' WHERE id=?",
                            ("X" * 300000, m1_id)
                        )
                        conn2.commit()
                    finally:
                        conn2.close()
                return self._real.execute(sql, *args)

            def executemany(self, sql, *args):
                return self._real.executemany(sql, *args)

            def __getattr__(self, name):
                return getattr(self._real, name)

        proxy = GrowthInterceptingConnection(real_conn, self.index.db_path)
        self.index.connection = proxy
        try:
            rep = self.index.enrich(provider, limit=1, account_id="growth-acc")
        finally:
            self.index.connection = real_conn

        self.assertTrue(grown, "Injection must have occurred")
        self.assertEqual(rep["attempted"], 1)
        self.assertEqual(rep["processed"], 0)
        self.assertEqual(rep["skipped_budget"], 1)
        self.assertEqual(len(rep["skipped_details"]), 1)
        self.assertEqual(rep["skipped_details"][0]["reason"], "source_bytes_exceeded")
        self.assertEqual(provider.call_count, 0)
        self.assertIsNotNone(rep["next_cursor"])
        self.assertEqual(rep["next_cursor"]["after_id"], m1_id)
        vec_count = self.index.connection.execute("SELECT COUNT(*) FROM vectors WHERE message_row=?", (m1_id,)).fetchone()[0]
        self.assertEqual(vec_count, 0)

    def test_type_error_enrichment_stable_category_and_no_commit(self):
        p = {
            "id": "c-type-err",
            "messages": [
                {"id": "m-terr-1", "role": "user", "text": "Testing type error in provider"},
                {"id": "m-terr-2", "role": "assistant", "text": "Second message"},
            ]
        }
        self.index.import_payload(p, account_id="type-acc")
        m1_id = self.index.connection.execute("SELECT id FROM messages WHERE message_id='m-terr-1'").fetchone()["id"]

        calls = 0
        fallback_calls = 0
        class TypeErrorProvider:
            model_id = "test-type-error-model"
            def embed_documents(self, texts, index_id=None):
                nonlocal calls
                calls += 1
                raise TypeError("Simulated TypeError in provider batch embedding")
            def embed(self, texts):
                nonlocal fallback_calls
                fallback_calls += 1
                return [[1.0, 0.0] for _ in texts]

        rep = self.index.enrich(TypeErrorProvider(), limit=2, account_id="type-acc")
        self.assertEqual(rep["stopped_reason"], "provider_error")
        self.assertEqual(calls, 1)
        self.assertEqual(fallback_calls, 0)
        self.assertEqual(len(rep["errors"]), 1)
        self.assertEqual(rep["errors"][0]["error"], "type_error")

        vec_count = self.index.connection.execute("SELECT COUNT(*) FROM vectors WHERE message_row=?", (m1_id,)).fetchone()[0]
        self.assertEqual(vec_count, 0)
        enr_count = self.index.connection.execute("SELECT COUNT(*) FROM enrichment WHERE message_row=?", (m1_id,)).fetchone()[0]
        self.assertEqual(enr_count, 0)

        if rep["next_cursor"] is not None:
            self.assertLess(rep["next_cursor"]["after_id"], m1_id)

    def test_invalid_budgets_and_cursor_before_zero_calls(self):
        provider = MagicMock()
        provider.model_id = "mock-model"
        provider.cached_identity = "mock-model"

        # Invalid max_time_seconds
        for bad_time in (0, -1.0, True, 1201, float("nan"), float("inf"), "100"):
            with self.assertRaises(ValueError):
                self.index.enrich(provider, max_time_seconds=bad_time, should_stop=lambda: True)

        # Invalid max_requests
        for bad_req in (0, 1025, True, 1.5, "10"):
            with self.assertRaises(ValueError):
                self.index.enrich(provider, max_requests=bad_req, should_stop=lambda: True)

        # Invalid cursor structures / extra keys / out of bound after_id
        for bad_cursor in (
            "not-a-dict",
            {"after_id": 0},
            {"after_id": 0, "model_id": "mock-model", "account_id": None, "generate_topics": False, "extra_key": 1},
            {"after_id": -1, "model_id": "mock-model", "account_id": None, "generate_topics": False},
            {"after_id": 1 << 63, "model_id": "mock-model", "account_id": None, "generate_topics": False},
            {"after_id": True, "model_id": "mock-model", "account_id": None, "generate_topics": False},
            {"after_id": 0, "model_id": "", "account_id": None, "generate_topics": False},
            {"after_id": 0, "model_id": "mock-model", "account_id": None, "generate_topics": "not-bool"},
        ):
            with self.assertRaises(ValueError):
                self.index.enrich(provider, after_cursor=bad_cursor, should_stop=lambda: True)

        provider.embed_documents.assert_not_called()
        provider.embed.assert_not_called()
        provider.summarize.assert_not_called()
        provider.resolve_identity.assert_not_called()

    def test_stray_vector_and_prepause_counts(self):
        p = {
            "id": "c-stray",
            "messages": [
                {"id": "m-stray-1", "role": "user", "text": "Message with stray vector"},
            ]
        }
        self.index.import_payload(p, account_id="stray-acc")
        msg_id = self.index.connection.execute("SELECT id, content_hash FROM messages WHERE message_id='m-stray-1'").fetchone()

        # Insert a stray vector directly into vectors table WITHOUT an enrichment marker
        self.index.connection.execute(
            "INSERT INTO vectors VALUES (?, 'dialogue', 0, 'test-stray-model', ?, 0, 10, 'derived', '\"[1.0, 0.0]\"')",
            (msg_id["id"], msg_id["content_hash"])
        )
        self.index.connection.commit()

        class StrayTestProvider:
            model_id = "test-stray-model"
            cached_identity = "test-stray-model"

        self.index.embedding_provider = StrayTestProvider()

        # Offline status: stray vector without marker must stay pending!
        st = self.index.status(account_id="stray-acc")
        cov = st["semantic"]["coverage"]
        self.assertEqual(cov["current"], 0)
        self.assertEqual(cov["pending"], 1)
        self.assertEqual(self.index._count_pending("test-stray-model", account_id="stray-acc"), 1)

        # Now properly enrich the message so it has both vectors and enrichment marker
        rep_enrich = self.index.enrich(ToyProvider(), limit=1, account_id="stray-acc")
        self.assertEqual(rep_enrich["processed"], 1)

        # Now test pre-pause with versioned cached identity on already-current row:
        class CachedVersionProvider:
            model_id = "synthetic:semantic-fixture-v1"
            cached_identity = "synthetic:semantic-fixture-v1"
            def resolve_identity(self, profile):
                return "synthetic:semantic-fixture-v1"

        rep_pause = self.index.enrich(CachedVersionProvider(), account_id="stray-acc", should_stop=lambda: True)
        self.assertTrue(rep_pause["paused"])
        self.assertEqual(rep_pause["stopped_reason"], "paused")
        # Prepaused cached-version already-current row should report remaining 0 with actual cached index_id!
        self.assertEqual(rep_pause["remaining"], 0)
        self.assertEqual(rep_pause["model_id"], "synthetic:semantic-fixture-v1")
        self.assertEqual(rep_pause["display_model_id"], "synthetic:semantic-fixture-v1")

    def test_nul_and_legacy_tail_context(self):
        text_with_nul = "Root question before NUL \x00 and after NUL with special parameters"
        child_with_nul = "Child reply before NUL \x00 and after NUL explanation" + (" Tail padding line " * 20)
        p = {
            "id": "c-nul-ctx",
            "messages": [
                {"id": "m-nul-root", "role": "user", "text": text_with_nul},
                {"id": "m-nul-child", "parent_id": "m-nul-root", "role": "assistant", "text": child_with_nul},
                {"id": "m-nul-leaf", "parent_id": "m-nul-child", "role": "user", "text": "Final follow-up question"},
            ]
        }
        self.index.import_payload(p, account_id="nul-acc")

        leaf_row = self.index.connection.execute(
            "SELECT * FROM messages WHERE message_id='m-nul-leaf' AND account_id='nul-acc'"
        ).fetchone()

        # Legacy context: bounded prior messages with is_legacy=True preserving NUL in tail suffix
        legacy_priors = self.index._bounded_prior_messages(leaf_row, is_legacy=True)
        self.assertEqual(len(legacy_priors), 2)
        self.assertIn("\x00", legacy_priors[0]["text"])
        self.assertIn("\x00", legacy_priors[1]["text"])

        # Modern context: bounded prior messages with max_bytes=960 preserving NUL in prefix
        modern_priors = self.index._bounded_prior_messages(leaf_row, max_bytes=960)
        self.assertEqual(len(modern_priors), 2)
        self.assertIn("\x00", modern_priors[0]["text"])
        self.assertIn("\x00", modern_priors[1]["text"])

        # context() returns full branch preserving embedded NULs
        ctx = self.index.context("c-nul-ctx", "m-nul-leaf", account_id="nul-acc")
        self.assertEqual(len(ctx["messages"]), 3)
        self.assertIn("\x00", ctx["messages"][0]["text"])
        self.assertIn("\x00", ctx["messages"][1]["text"])

    def test_meaning_evidence_successful_query_rejects_unrelated_shared_keyword(self):
        class EvidenceFakeProvider:
            model_id = "synthetic:evidence-provider-v1"
            minimum_similarity = 0.50

            def embed_query(self, query: str) -> list[float]:
                return [1.0, 0.0]

            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        payload = {
            "id": "c-meaning-reject",
            "messages": [
                {
                    "id": "m-target",
                    "role": "assistant",
                    "text": "Database replication and failover architecture design.",
                },
                {
                    "id": "m-unrelated",
                    "role": "assistant",
                    "text": "Database recipe for homemade tomato soup with oregano.",
                },
            ],
        }
        self.index.import_payload(payload, account_id="ev-acc")

        # m-target has cosine similarity 1.0 (>= 0.50) with query [1.0, 0.0]
        self._seed_vector("m-target", "dialogue", 0, "synthetic:evidence-provider-v1",
                           0, 30, [1.0, 0.0], account_id="ev-acc")
        # m-unrelated has orthogonal vector [0.0, 1.0], cosine similarity 0.0 (< 0.50)
        self._seed_vector("m-unrelated", "dialogue", 0, "synthetic:evidence-provider-v1",
                           0, 30, [0.0, 1.0], account_id="ev-acc")

        self.index.embedding_provider = EvidenceFakeProvider()

        # Query "Database replication" matches "Database" in both messages via BM25,
        # but m-unrelated is below threshold and not a full-query literal match.
        res = self.index.search("Database replication", mode="smart", account_id="ev-acc")
        self.assertTrue(res["semantic"]["used"])
        self.assertEqual(res["semantic"]["relevance_policy"], "meaning-evidence-v1")
        self.assertEqual(res["semantic"]["rejected_keyword_only_count"], 1)
        self.assertEqual(res["total"], 1)
        self.assertEqual(len(res["results"]), 1)
        self.assertEqual(res["results"][0]["message_id"], "m-target")
        self.assertNotIn("m-unrelated", [r["message_id"] for r in res["results"]])

    def test_meaning_evidence_verified_literal_survives_low_cosine(self):
        class LowCosineProvider:
            model_id = "synthetic:low-cosine-provider-v1"
            minimum_similarity = 0.50

            def embed_query(self, query: str) -> list[float]:
                return [1.0, 0.0]

            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        payload = {
            "id": "c-literal-survive",
            "messages": [
                {
                    "id": "m-literal",
                    "role": "user",
                    "text": "Exact literal needle in haystack with important data.",
                },
                {
                    "id": "m-keyword-only",
                    "role": "user",
                    "text": "Haystack without the full phrase needle.",
                },
            ],
        }
        self.index.import_payload(payload, account_id="lit-acc")

        # Both have orthogonal vectors to query [1.0, 0.0], so cosine is 0.0 (< 0.50 threshold)
        self._seed_vector("m-literal", "dialogue", 0, "synthetic:low-cosine-provider-v1",
                           0, 20, [0.0, 1.0], account_id="lit-acc")
        self._seed_vector("m-keyword-only", "dialogue", 0, "synthetic:low-cosine-provider-v1",
                           0, 20, [0.0, 1.0], account_id="lit-acc")

        self.index.embedding_provider = LowCosineProvider()

        # Query "Exact literal needle" literally matches m-literal.
        # It also matches m-keyword-only lexically on "needle", but not literally.
        res = self.index.search("Exact literal needle", mode="smart", account_id="lit-acc")
        self.assertTrue(res["semantic"]["used"])
        self.assertEqual(res["semantic"]["relevance_policy"], "meaning-evidence-v1")
        self.assertEqual(res["semantic"]["rejected_keyword_only_count"], 1)
        self.assertEqual(res["total"], 1)
        self.assertEqual(len(res["results"]), 1)
        hit = res["results"][0]
        self.assertEqual(hit["message_id"], "m-literal")
        self.assertEqual(hit["match_type"], "exact")
        self.assertIn("literal", hit["retrieval_channels"])
        self.assertNotIn("dialogue", hit["retrieval_channels"])

    def test_meaning_evidence_supported_hybrid_remains_and_ranks_deterministically(self):
        class HybridProvider:
            model_id = "synthetic:hybrid-provider-v1"
            minimum_similarity = 0.50

            def embed_query(self, query: str) -> list[float]:
                return [1.0, 0.0]

            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        payload = {
            "id": "c-hybrid-rank",
            "messages": [
                {
                    "id": "m-hybrid",
                    "role": "assistant",
                    "text": "Distributed cache cluster invalidation protocol.",
                },
                {
                    "id": "m-semantic",
                    "role": "assistant",
                    "text": "In-memory key-value data replication synchronization.",
                },
                {
                    "id": "m-unsupported-kw",
                    "role": "assistant",
                    "text": "Distributed filesystem storage without cache.",
                },
            ],
        }
        self.index.import_payload(payload, account_id="hyb-acc")

        # m-hybrid: sim 0.95 (>= 0.50)
        self._seed_vector("m-hybrid", "dialogue", 0, "synthetic:hybrid-provider-v1",
                           0, 30, [0.95, (1 - 0.95**2)**0.5], account_id="hyb-acc")
        # m-semantic: sim 0.90 (>= 0.50)
        self._seed_vector("m-semantic", "dialogue", 0, "synthetic:hybrid-provider-v1",
                           0, 30, [0.90, (1 - 0.90**2)**0.5], account_id="hyb-acc")
        # m-unsupported-kw: sim 0.10 (< 0.50)
        self._seed_vector("m-unsupported-kw", "dialogue", 0, "synthetic:hybrid-provider-v1",
                           0, 30, [0.10, (1 - 0.10**2)**0.5], account_id="hyb-acc")

        self.index.embedding_provider = HybridProvider()

        # Query "Distributed cache"
        # m-hybrid matches BM25 ("Distributed", "cache") and semantic (sim 0.95) -> hybrid!
        # m-semantic matches semantic (sim 0.90) but has no words from query -> semantic!
        # m-unsupported-kw matches BM25 ("Distributed") but sim 0.10 < 0.50 -> rejected!
        res = self.index.search("Distributed cache", mode="smart", account_id="hyb-acc")
        self.assertTrue(res["semantic"]["used"])
        self.assertEqual(res["semantic"]["relevance_policy"], "meaning-evidence-v1")
        self.assertEqual(res["semantic"]["rejected_keyword_only_count"], 1)
        self.assertEqual(res["total"], 2)

        # m-hybrid has keyword + dialogue channels, ranks first
        top_hit = res["results"][0]
        self.assertEqual(top_hit["message_id"], "m-hybrid")
        self.assertEqual(top_hit["match_type"], "hybrid")
        self.assertIn("keyword", top_hit["retrieval_channels"])
        self.assertIn("dialogue", top_hit["retrieval_channels"])

        # m-semantic ranks second as semantic
        second_hit = res["results"][1]
        self.assertEqual(second_hit["message_id"], "m-semantic")
        self.assertEqual(second_hit["match_type"], "semantic")
        self.assertNotIn("keyword", second_hit["retrieval_channels"])
        self.assertIn("dialogue", second_hit["retrieval_channels"])

    def test_meaning_evidence_used_false_error_and_no_vectors_preserves_fallback(self):
        payload = {
            "id": "c-fallback-test",
            "messages": [
                {
                    "id": "m-fb-1",
                    "role": "user",
                    "text": "PostgreSQL database indexing and query optimization tips.",
                },
                {
                    "id": "m-fb-2",
                    "role": "assistant",
                    "text": "Database indexing guidelines for B-tree and GiST indexes.",
                },
            ],
        }
        self.index.import_payload(payload, account_id="fb-acc")

        # Case 1: Provider is None (used is False)
        self.index.embedding_provider = None
        res_none = self.index.search("database indexing", mode="smart", account_id="fb-acc")
        self.assertFalse(res_none["semantic"]["used"])
        self.assertEqual(res_none["semantic"]["status"], "not_configured")
        self.assertEqual(res_none["semantic"]["relevance_policy"], "meaning-evidence-v1")
        self.assertEqual(res_none["semantic"]["rejected_keyword_only_count"], 0)
        self.assertEqual(res_none["total"], 2)
        self.assertTrue(all(r["match_type"] in ("lexical", "exact") for r in res_none["results"]))

        # Case 2: Provider configured but zero vectors indexed (indexed_messages == 0)
        class NoCallProvider:
            model_id = "synthetic:no-call-provider-v1"
            def __init__(self):
                self.calls = 0
            def embed_query(self, query: str) -> list[float]:
                self.calls += 1
                return [1.0, 0.0]

        no_call_prov = NoCallProvider()
        self.index.embedding_provider = no_call_prov
        res_zero = self.index.search("database indexing", mode="smart", account_id="fb-acc")
        self.assertEqual(no_call_prov.calls, 0, "embed_query must not be called when no current vectors exist")
        self.assertFalse(res_zero["semantic"]["used"])
        self.assertEqual(res_zero["semantic"]["status"], "index_pending")
        self.assertEqual(res_zero["semantic"]["relevance_policy"], "meaning-evidence-v1")
        self.assertEqual(res_zero["semantic"]["rejected_keyword_only_count"], 0)
        self.assertEqual(res_zero["total"], 2)

        # Now seed vectors for error and overbudget test
        self._seed_vector("m-fb-1", "dialogue", 0, "synthetic:error-prov-v1",
                           0, 30, [1.0, 0.0], account_id="fb-acc")

        # Case 3: Provider raises ProviderError during embed_query
        class ErrorProvider:
            model_id = "synthetic:error-prov-v1"
            minimum_similarity = 0.50
            def embed_query(self, query: str) -> list[float]:
                raise ProviderError("Ollama daemon unavailable")

        self.index.embedding_provider = ErrorProvider()
        res_err = self.index.search("database indexing", mode="smart", account_id="fb-acc")
        self.assertFalse(res_err["semantic"]["used"])
        self.assertEqual(res_err["semantic"]["status"], "provider_error")
        self.assertEqual(res_err["semantic"]["relevance_policy"], "meaning-evidence-v1")
        self.assertEqual(res_err["semantic"]["rejected_keyword_only_count"], 0)
        self.assertIn("Ollama daemon unavailable", res_err["semantic"]["errors"][0])
        self.assertEqual(res_err["total"], 2)

        # Case 4: Query overbudget (raises ProviderError)
        class OverbudgetProvider:
            model_id = "synthetic:error-prov-v1"
            minimum_similarity = 0.50
            def embed_query(self, query: str) -> list[float]:
                raise ProviderError("Ollama input exceeds 960 UTF-8 bytes; ask for a shorter Smart description")

        self.index.embedding_provider = OverbudgetProvider()
        res_over = self.index.search("database indexing", mode="smart", account_id="fb-acc")
        self.assertFalse(res_over["semantic"]["used"])
        self.assertEqual(res_over["semantic"]["status"], "provider_error")
        self.assertEqual(res_over["semantic"]["relevance_policy"], "meaning-evidence-v1")
        self.assertEqual(res_over["semantic"]["rejected_keyword_only_count"], 0)
        self.assertTrue(any("exceeds 960 UTF-8 bytes" in e for e in res_over["semantic"]["errors"]))
        self.assertEqual(res_over["total"], 2)

    def test_meaning_evidence_account_role_and_profile_filtering(self):
        class ScopeProvider:
            model_id = "synthetic:scope-provider-v1"
            minimum_similarity = 0.50
            def embed_query(self, query: str) -> list[float]:
                return [1.0, 0.0]

        payload_1 = {
            "id": "c-scope-1",
            "messages": [
                {"id": "m-s1-user", "role": "user", "text": "Kubernetes cluster ingress controller routing setup."},
                {"id": "m-s1-asst", "role": "assistant", "text": "Kubernetes cluster service mesh proxy routing implementation."},
            ],
        }
        payload_2 = {
            "id": "c-scope-2",
            "messages": [
                {"id": "m-s2-user", "role": "user", "text": "Kubernetes cluster monitoring with Prometheus and Grafana."},
            ],
        }
        self.index.import_payload(payload_1, account_id="scope-acc-1")
        self.index.import_payload(payload_2, account_id="scope-acc-2")

        # Seed vectors for acc-1
        self._seed_vector("m-s1-user", "dialogue", 0, "synthetic:scope-provider-v1",
                           0, 20, [1.0, 0.0], account_id="scope-acc-1")
        self._seed_vector("m-s1-asst", "dialogue", 0, "synthetic:scope-provider-v1",
                           0, 20, [1.0, 0.0], account_id="scope-acc-1")
        # Seed vector for acc-2
        self._seed_vector("m-s2-user", "dialogue", 0, "synthetic:scope-provider-v1",
                           0, 20, [1.0, 0.0], account_id="scope-acc-2")

        self.index.embedding_provider = ScopeProvider()

        # 1. Account isolation: only scope-acc-1 messages returned
        res_acc1 = self.index.search("Kubernetes cluster", mode="smart", account_id="scope-acc-1")
        self.assertTrue(res_acc1["semantic"]["used"])
        self.assertEqual(res_acc1["total"], 2)
        self.assertTrue(all(r["account_id"] == "scope-acc-1" for r in res_acc1["results"]))

        # 2. Role isolation: filter for user only within scope-acc-1
        res_user = self.index.search("Kubernetes cluster", mode="smart", account_id="scope-acc-1", role="user")
        self.assertEqual(res_user["total"], 1)
        self.assertEqual(res_user["results"][0]["message_id"], "m-s1-user")
        self.assertEqual(res_user["results"][0]["role"], "user")

        # 3. Role isolation: filter for assistant only within scope-acc-1
        res_asst = self.index.search("Kubernetes cluster", mode="smart", account_id="scope-acc-1", role="assistant")
        self.assertEqual(res_asst["total"], 1)
        self.assertEqual(res_asst["results"][0]["message_id"], "m-s1-asst")
        self.assertEqual(res_asst["results"][0]["role"], "assistant")

        # 4. Outdated / different model profile exclusion
        class MismatchedProfileProvider:
            model_id = "synthetic:different-profile-v2"
            minimum_similarity = 0.50
            def embed_query(self, query: str) -> list[float]:
                return [1.0, 0.0]

        self.index.embedding_provider = MismatchedProfileProvider()
        res_mismatch = self.index.search("Kubernetes cluster", mode="smart", account_id="scope-acc-1")
        # Incompatible vectors are excluded -> indexed_messages == 0 -> used == False
        self.assertFalse(res_mismatch["semantic"]["used"])
        self.assertEqual(res_mismatch["semantic"]["indexed_messages"], 0)
        self.assertEqual(res_mismatch["semantic"]["rejected_keyword_only_count"], 0)
        # Truthful keyword fallback returns both
        self.assertEqual(res_mismatch["total"], 2)

    def test_meaning_evidence_pagination_counts_accepted_unique_messages(self):
        class PagingFakeProvider:
            model_id = "synthetic:paging-provider-v1"
            minimum_similarity = 0.50
            def embed_query(self, query: str) -> list[float]:
                return [1.0, 0.0]

        # 10 messages all sharing keyword "container"
        # 3 messages have vector matching query (sim 1.0 >= 0.50)
        # 7 messages have orthogonal vector (sim 0.0 < 0.50) and no literal match
        messages = [
            {"id": f"m-sem-{i}", "role": "user", "text": f"Container virtualization technique {i}."}
            for i in range(3)
        ] + [
            {"id": f"m-kw-{i}", "role": "user", "text": f"Unrelated container gardening idea {i}."}
            for i in range(7)
        ]
        self.index.import_payload({"id": "c-paging", "messages": messages}, account_id="page-acc")

        # Seed 3 supported semantic vectors
        for i in range(3):
            self._seed_vector(f"m-sem-{i}", "dialogue", 0, "synthetic:paging-provider-v1",
                               0, 20, [1.0, 0.0], account_id="page-acc")
        # Seed 7 unsupported vectors
        for i in range(7):
            self._seed_vector(f"m-kw-{i}", "dialogue", 0, "synthetic:paging-provider-v1",
                               0, 20, [0.0, 1.0], account_id="page-acc")

        self.index.embedding_provider = PagingFakeProvider()

        # Query "Container virtualization"
        # 7 unsupported keyword messages are rejected
        # Total accepted unique messages = 3
        p1 = self.index.search("Container virtualization", mode="smart", account_id="page-acc", limit=2, offset=0)
        self.assertTrue(p1["semantic"]["used"])
        self.assertEqual(p1["semantic"]["relevance_policy"], "meaning-evidence-v1")
        self.assertEqual(p1["semantic"]["rejected_keyword_only_count"], 7)
        self.assertEqual(p1["total"], 3)
        self.assertEqual(len(p1["results"]), 2)

        p2 = self.index.search("Container virtualization", mode="smart", account_id="page-acc", limit=2, offset=2)
        self.assertEqual(p2["total"], 3)
        self.assertEqual(len(p2["results"]), 1)

        p3 = self.index.search("Container virtualization", mode="smart", account_id="page-acc", limit=2, offset=4)
        self.assertEqual(p3["total"], 3)
        self.assertEqual(len(p3["results"]), 0)

        # Union of page 1 and page 2 contains all 3 semantic messages
        all_paged = {r["message_id"] for r in p1["results"] + p2["results"]}
        self.assertEqual(all_paged, {"m-sem-0", "m-sem-1", "m-sem-2"})

        # Verified literal match is also accepted and counted in pagination
        # Import an exact literal message that has low cosine
        self.index.import_payload({
            "id": "c-paging-lit",
            "messages": [
                {"id": "m-lit-exact", "role": "user", "text": "Container virtualization exact matching text."}
            ]
        }, account_id="page-acc")
        self._seed_vector("m-lit-exact", "dialogue", 0, "synthetic:paging-provider-v1",
                           0, 20, [0.0, 1.0], account_id="page-acc")

        p_lit = self.index.search("Container virtualization", mode="smart", account_id="page-acc", limit=10, offset=0)
        # 3 semantic + 1 literal = 4 accepted unique messages; 7 rejected keyword-only
        self.assertEqual(p_lit["total"], 4)
        self.assertEqual(p_lit["semantic"]["rejected_keyword_only_count"], 7)
        accepted_mids = [r["message_id"] for r in p_lit["results"]]
        self.assertIn("m-lit-exact", accepted_mids)
        for i in range(3):
            self.assertIn(f"m-sem-{i}", accepted_mids)


if __name__ == "__main__":
    unittest.main()
