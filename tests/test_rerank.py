import unittest
from chat_recall.rerank import CohereReranker, RerankError, rerank_results


class RerankTests(unittest.TestCase):
    def provider(self, result):
        provider = CohereReranker("synthetic-key-not-real")
        provider._request = lambda payload: result
        return provider

    def test_preserves_candidate_identifiers_and_order(self):
        provider = self.provider({"results": [{"index": 1, "relevance_score": .9}, {"index": 0, "relevance_score": .2}]})
        self.assertEqual(provider.rerank("topic", ["first", "second"]), [(1, .9), (0, .2)])

    def test_rejects_duplicates_missing_nonfinite_and_unknown(self):
        for rows in ([{"index": 0, "relevance_score": 1}] * 2,
                     [{"index": 0, "relevance_score": 1}],
                     [{"index": 0, "relevance_score": float("nan")}, {"index": 1, "relevance_score": .2}],
                     [{"index": 9, "relevance_score": .9}, {"index": 0, "relevance_score": .2}]):
            with self.subTest(rows=rows), self.assertRaises(RerankError):
                self.provider({"results": rows}).rerank("topic", ["first", "second"])

    def test_limits_are_checked_before_network(self):
        with self.assertRaises(ValueError):
            self.provider({}).rerank("topic", ["x"] * 31)
        with self.assertRaises(ValueError):
            self.provider({}).rerank("topic", ["x" * 9001])

    def source_response(self, *, focus=0, matches=None):
        text = "😀" * 7500 + "Keep the recovery key offline."
        hit = {"conversation_id": "c", "message_id": "m", "source_id": "chatgpt",
               "account_id": "fixture", "title": "Long source", "role": "assistant",
               "text": text, "matches": matches or [], "focus_offset": focus}

        class Index:
            def context(self, *args):
                return {"messages": [{"id": "m", "role": "assistant", "text": text}]}

        return Index(), {"query": "backup recovery advice", "total": 1, "results": [hit]}

    def capture_documents(self, response, index):
        class RecordingRanker:
            documents = None

            def rerank(self, query, documents):
                self.documents = documents
                return [(0, .9)]

        ranker = RecordingRanker()
        result = rerank_results(index, response, ranker)
        self.assertEqual(result["results"][0]["message_id"], "m")
        return ranker.documents

    def test_semantic_tail_is_retained_in_cloud_source_window(self):
        index, response = self.source_response(focus=7500)
        documents = self.capture_documents(response, index)
        self.assertIn("Keep the recovery key offline.", documents[0])
        self.assertLess(len(documents[0]), 9000)

    def test_literal_evidence_takes_precedence_over_semantic_hint(self):
        index, response = self.source_response(focus=7500, matches=[[100, 101]])
        documents = self.capture_documents(response, index)
        self.assertNotIn("recovery key", documents[0])

    def test_focus_hint_is_clamped_and_legacy_results_are_supported(self):
        for hint, includes_tail in ((999999, True), (-50, False), ("7500", False), (True, False)):
            with self.subTest(hint=hint):
                index, response = self.source_response(focus=hint)
                documents = self.capture_documents(response, index)
                self.assertEqual("recovery key" in documents[0], includes_tail)
        index, response = self.source_response()
        response["results"][0].pop("focus_offset")
        self.assertTrue(self.capture_documents(response, index))
