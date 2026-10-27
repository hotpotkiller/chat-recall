import json
from pathlib import Path
import random
import tempfile
import unittest

from chat_recall.providers import ProviderError, normalized_vector
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


if __name__ == "__main__":
    unittest.main()
