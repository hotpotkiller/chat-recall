import json
from pathlib import Path
import random
import tempfile
import unittest

from chat_recall.search import SearchIndex, literal_matches


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


from chat_recall.providers import ProviderError

if __name__ == "__main__":
    unittest.main()
