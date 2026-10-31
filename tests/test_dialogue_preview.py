import json
from pathlib import Path
import tempfile
import unittest

from chat_recall.providers import normalized_vector
from chat_recall.search import SearchIndex


class StaticQueryProvider:
    model_id = "synthetic:dialogue-preview-fixture"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class DialoguePreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.index = SearchIndex(Path(self.temp.name) / "index.sqlite")

    def tearDown(self):
        self.index.close()
        self.temp.cleanup()

    def _seed_vector(self, message_id: str, channel: str = "dialogue", chunk: int = 0,
                     vector: list[float] | None = None, account_id: str = "default",
                     source_id: str = "chatgpt"):
        if vector is None:
            vector = [1.0, 0.0]
        row = self.index.connection.execute(
            "SELECT id, text, content_hash FROM messages WHERE source_id=? AND account_id=? AND message_id=?",
            (source_id, account_id, message_id)
        ).fetchone()
        if row is None:
            raise ValueError(f"Message {message_id} not found")
        norm = normalized_vector(vector)
        self.index.connection.execute(
            """INSERT INTO vectors (message_row, channel, chunk, model_id, content_hash,
                                   start_offset, end_offset, derived_text, vector_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(message_row, channel, chunk, model_id) DO UPDATE SET
               vector_json=excluded.vector_json, start_offset=excluded.start_offset,
               end_offset=excluded.end_offset, content_hash=excluded.content_hash""",
            (row["id"], channel, chunk, StaticQueryProvider.model_id, row["content_hash"],
             0, len(row["text"]), "derived", json.dumps(norm))
        )
        self.index.connection.execute(
            """INSERT INTO enrichment (message_row, model_id, content_hash, topics)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(message_row, model_id) DO UPDATE SET
               content_hash=excluded.content_hash,
               topics=MAX(enrichment.topics, excluded.topics)""",
            (row["id"], StaticQueryProvider.model_id, row["content_hash"], 1 if channel == "topic" else 0)
        )
        self.index.connection.commit()

    def test_generic_reply_shows_bounded_preceding_ancestors_oldest_first(self):
        payload = {
            "id": "conv-dialogue-chain",
            "messages": [
                {"id": "msg-0", "role": "user", "text": "What is the capital of France?"},
                {"id": "msg-1", "parent_id": "msg-0", "role": "assistant", "text": "The capital of France is Paris."},
                {"id": "msg-2", "parent_id": "msg-1", "role": "user", "text": "Are you certain?"},
                {"id": "msg-3", "parent_id": "msg-2", "role": "assistant", "text": "Yes."}
            ]
        }
        self.index.import_payload(payload)
        self._seed_vector("msg-3")
        self.index.embedding_provider = StaticQueryProvider()

        # Exact mode must never include dialogue_preview
        exact_res = self.index.search("Yes.", mode="exact")
        self.assertGreater(exact_res["total"], 0)
        self.assertNotIn("dialogue_preview", exact_res["results"][0])

        # Smart mode for semantic/hybrid result shows bounded preceding 2 ancestors oldest first
        smart_res = self.index.search("capital inquiry", mode="smart")
        hits_by_id = {r["message_id"]: r for r in smart_res["results"]}
        self.assertIn("msg-3", hits_by_id)
        hit = hits_by_id["msg-3"]
        self.assertEqual(hit["match_type"], "semantic")
        self.assertIn("dialogue_preview", hit)
        preview = hit["dialogue_preview"]
        self.assertEqual(len(preview), 2)
        self.assertEqual([p["message_id"] for p in preview], ["msg-1", "msg-2"])
        self.assertEqual(preview[0], {
            "message_id": "msg-1",
            "role": "assistant",
            "text": "The capital of France is Paris.",
            "start_offset": 0,
            "end_offset": len("The capital of France is Paris.")
        })
        self.assertEqual(preview[1], {
            "message_id": "msg-2",
            "role": "user",
            "text": "Are you certain?",
            "start_offset": 0,
            "end_offset": len("Are you certain?")
        })

    def test_regenerated_sibling_branch_excluded_from_preview(self):
        payload = {
            "id": "conv-branches",
            "mapping": {
                "n-root": {
                    "parent": None,
                    "message": {"id": "m-root", "role": "user", "text": "Original root question"}
                },
                "n-branch-a": {
                    "parent": "n-root",
                    "message": {"id": "m-branch-a", "role": "assistant", "text": "Branch A alternative response"}
                },
                "n-branch-b": {
                    "parent": "n-root",
                    "message": {"id": "m-branch-b", "role": "assistant", "text": "Branch B selected response"}
                },
                "n-followup": {
                    "parent": "n-branch-b",
                    "message": {"id": "m-followup", "role": "user", "text": "Follow-up question on branch B"}
                }
            }
        }
        self.index.import_payload(payload)
        self._seed_vector("m-followup")
        self.index.embedding_provider = StaticQueryProvider()

        res = self.index.search("unmatched semantic query", mode="smart")
        self.assertEqual(res["total"], 1)
        hit = res["results"][0]
        self.assertEqual(hit["message_id"], "m-followup")
        preview = hit["dialogue_preview"]
        self.assertEqual([p["message_id"] for p in preview], ["m-root", "m-branch-b"])
        preview_ids = {p["message_id"] for p in preview}
        self.assertNotIn("m-branch-a", preview_ids)

    def test_same_native_ids_in_different_accounts_remain_isolated(self):
        payload_acc1 = {
            "id": "shared-conv-id",
            "messages": [
                {"id": "msg-parent", "role": "user", "text": "Account 1 parent context"},
                {"id": "msg-target", "parent_id": "msg-parent", "role": "assistant", "text": "Account 1 target answer"}
            ]
        }
        payload_acc2 = {
            "id": "shared-conv-id",
            "messages": [
                {"id": "msg-parent", "role": "user", "text": "Account 2 completely different parent context"},
                {"id": "msg-target", "parent_id": "msg-parent", "role": "assistant", "text": "Account 2 target answer"}
            ]
        }
        self.index.import_payload(payload_acc1, account_id="acc-1")
        self.index.import_payload(payload_acc2, account_id="acc-2")
        self._seed_vector("msg-target", account_id="acc-1")
        self._seed_vector("msg-target", account_id="acc-2")
        self.index.embedding_provider = StaticQueryProvider()

        res1 = self.index.search("query", mode="smart", account_id="acc-1")
        self.assertEqual(res1["total"], 1)
        p1 = res1["results"][0]["dialogue_preview"]
        self.assertEqual(len(p1), 1)
        self.assertEqual(p1[0]["text"], "Account 1 parent context")
        self.assertEqual(res1["results"][0]["account_id"], "acc-1")

        res2 = self.index.search("query", mode="smart", account_id="acc-2")
        self.assertEqual(res2["total"], 1)
        p2 = res2["results"][0]["dialogue_preview"]
        self.assertEqual(len(p2), 1)
        self.assertEqual(p2[0]["text"], "Account 2 completely different parent context")
        self.assertEqual(res2["results"][0]["account_id"], "acc-2")

    def test_unicode_astral_and_control_nul_sliced_to_240_codepoints_and_empty_ancestor(self):
        astral_prefix = "🌟" * 200 + "A\x00B"
        long_parent_text = astral_prefix + ("Z" * 100)
        self.assertEqual(len(long_parent_text), 303)

        payload = {
            "id": "unicode-conv",
            "messages": [
                {"id": "root-msg", "role": "user", "text": long_parent_text},
                {"id": "child-msg", "parent_id": "root-msg", "role": "assistant", "text": "Child reply."}
            ]
        }
        self.index.import_payload(payload)
        self._seed_vector("root-msg")
        self._seed_vector("child-msg")
        self.index.embedding_provider = StaticQueryProvider()

        # Child has root-msg as ancestor: text must equal exact original [:240] codepoints
        res_child = self.index.search("unmatched", mode="smart")
        hits_by_id = {r["message_id"]: r for r in res_child["results"]}
        child_hit = hits_by_id["child-msg"]
        preview = child_hit["dialogue_preview"]
        self.assertEqual(len(preview), 1)
        item = preview[0]
        expected_text = long_parent_text[:240]
        self.assertEqual(len(expected_text), 240)
        self.assertEqual(item["text"], expected_text)
        self.assertEqual(item["start_offset"], 0)
        self.assertEqual(item["end_offset"], 240)
        self.assertIn("\x00", item["text"])
        self.assertTrue(item["text"].startswith("🌟" * 200))

        # Root message has no ancestor: dialogue_preview must safely be []
        root_hit = hits_by_id["root-msg"]
        self.assertEqual(root_hit["dialogue_preview"], [])

    def test_cjk_dialogue_preview_codepoints_and_role(self):
        cjk_parent = "换季时家里小户型衣柜和储物间堆放混乱，衣服和杂物越来越多，应该如何科学规划空间和分类整理？" * 10
        self.assertGreater(len(cjk_parent), 240)
        payload = {
            "id": "cjk-preview-conv",
            "messages": [
                {"id": "cjk-root", "role": "user", "text": cjk_parent},
                {"id": "cjk-child", "parent_id": "cjk-root", "role": "assistant", "text": "分类整理方案"},
            ],
        }
        self.index.import_payload(payload, account_id="cjk-preview-acc")
        self._seed_vector("cjk-child", account_id="cjk-preview-acc")
        self.index.embedding_provider = StaticQueryProvider()

        res = self.index.search("整理方案", mode="smart", account_id="cjk-preview-acc")
        self.assertGreater(res["total"], 0)
        hit = res["results"][0]
        self.assertIn("dialogue_preview", hit)
        preview = hit["dialogue_preview"]
        self.assertEqual(len(preview), 1)
        self.assertEqual(preview[0]["message_id"], "cjk-root")
        self.assertEqual(preview[0]["role"], "user")
        self.assertEqual(preview[0]["text"], cjk_parent[:240])
        self.assertEqual(preview[0]["start_offset"], 0)
        self.assertEqual(preview[0]["end_offset"], 240)


if __name__ == "__main__":
    unittest.main()
