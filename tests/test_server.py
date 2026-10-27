"""Exercise real loopback requests, origin boundaries, imports and dialogue links."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from chat_recall.search import SearchIndex
from chat_recall.server import RecallServer, load_token


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.token = load_token(self.path)
        db = self.path / "recall.sqlite3"
        index = SearchIndex(db)
        index.close()
        self.origin = "chrome-extension://" + "a" * 32
        self.server = RecallServer(("127.0.0.1", 0), db, token=self.token, extension_origins=[self.origin])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, path, *, method="GET", body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        encoded = json.dumps(body).encode() if body is not None else None
        combined = dict(headers or {})
        if body is not None:
            combined.setdefault("Content-Type", "application/json")
        connection.request(method, path, body=encoded, headers=combined)
        response = connection.getresponse()
        raw = response.read()
        content_type = response.getheader("Content-Type", "")
        result = json.loads(raw) if raw and "application/json" in content_type else raw
        status, response_headers = response.status, dict(response.getheaders())
        connection.close()
        return status, result, response_headers

    def fixture(self):
        return {"payload": [{"id": "c1", "title": "An actual conversation", "messages": [
            {"id": "m1", "role": "user", "text": "Where is the rareword?"},
            {"id": "m2", "role": "assistant", "text": "rareword rareword 🧠", "parent_id": "m1"}
        ]}]}

    def test_end_to_end_import_search_context(self):
        code, result, _ = self.request("/api/import", method="POST", body=self.fixture(),
                                       headers={"Authorization": "Bearer " + self.token})
        self.assertEqual(code, 200)
        self.assertEqual(result["messages"], 2)
        code, result, headers = self.request("/api/search?q=rareword&mode=exact")
        self.assertEqual(code, 200)
        self.assertEqual(result["total"], 2)
        self.assertEqual(sum(x["match_count"] for x in result["results"]), 3)
        self.assertEqual(headers["Cache-Control"], "no-store")
        code, context, _ = self.request("/api/context?conversation_id=c1&message_id=m2")
        self.assertEqual(code, 200)
        self.assertEqual([x["id"] for x in context["messages"]], ["m1", "m2"])

    def test_import_requires_token(self):
        code, _, _ = self.request("/api/import", method="POST", body=self.fixture())
        self.assertEqual(code, 401)

    def test_untrusted_origin_and_rebinding_host(self):
        for headers in ({"Origin": "https://attacker.invalid"}, {"Host": "attacker.invalid"},
                        {"Sec-Fetch-Site": "cross-site"}, {"Origin": "null"}):
            code, _, _ = self.request("/api/config", headers=headers)
            self.assertEqual(code, 403)

    def test_extension_pairs_but_cannot_read_ui_token(self):
        code, _, _ = self.request("/api/config", headers={"Origin": self.origin})
        self.assertEqual(code, 403)
        code, _, _ = self.request("/api/status", headers={"Origin": self.origin})
        self.assertEqual(code, 401)
        code, _, headers = self.request("/api/status", headers={"Origin": self.origin, "Authorization": "Bearer " + self.token})
        self.assertEqual(code, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), self.origin)

    def test_ui_config_and_token_file(self):
        code, result, headers = self.request("/api/config")
        self.assertEqual(code, 200)
        self.assertEqual(result["token"], self.token)
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(load_token(self.path), self.token)
        self.assertEqual((self.path / "bridge-token").stat().st_mode & 0o777, 0o600)

    def test_unknown_target_and_bad_pagination(self):
        self.assertEqual(self.request("/api/context?conversation_id=nope&message_id=nope")[0], 404)
        self.assertEqual(self.request("/api/search?q=a&limit=0")[0], 400)
        self.assertEqual(self.request("/api/search?q=a&offset=-1")[0], 400)
        self.assertEqual(self.request("/api/search?q=a&limit=nan")[0], 400)

    def test_private_files_never_served(self):
        for path in ("/../.git/config", "/.runtime/bridge-token", "/server.py", "/%2e%2e/LICENSE"):
            self.assertEqual(self.request(path)[0], 404)

    def test_import_coverage_receipt_is_explicit(self):
        body = self.fixture()
        body["coverage"] = {"complete": False, "archived": "unknown", "native_app_verified": False}
        self.assertEqual(self.request("/api/import", method="POST", body=body,
                                     headers={"Authorization": "Bearer " + self.token})[0], 200)
        result = self.request("/api/status")[1]
        self.assertFalse(result["sync_receipts"]["chatgpt:default"]["complete"])

    def test_absent_semantic_provider_is_explicit(self):
        code, result, _ = self.request("/api/enrich", method="POST", body={},
                                       headers={"Authorization": "Bearer " + self.token})
        self.assertEqual(code, 400)
        self.assertIn("provider", result["error"])

    def test_remote_bind_is_not_supported(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            RecallServer(("0.0.0.0", 0), self.path / "other.db", token=self.token)

    def wait_for_job(self, expected):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.server.semantic_job["state"] == expected:
                return
            time.sleep(.01)
        self.fail(f"Expected semantic job {expected}, got {self.server.semantic_job}")

    def test_index_construction_failure_is_reported_and_can_retry(self):
        self.server.provider = object()  # Synthetic provider; no inference calls.
        with patch.object(self.server, "index", side_effect=OSError("synthetic database unavailable")):
            self.server.start_enrichment()
            self.wait_for_job("failed")
        self.assertIn("database unavailable", self.server.semantic_job["error"])
        index = MagicMock()
        index.enrich.return_value = {"processed": 1, "remaining": 0, "errors": []}
        with patch.object(self.server, "index", return_value=index):
            self.server.start_enrichment()
            self.wait_for_job("complete")
        self.assertEqual(self.server.semantic_job["report"]["processed"], 1)
        index.close.assert_called_once()

    def test_index_close_failure_does_not_leave_a_running_job(self):
        self.server.provider = object()
        index = MagicMock()
        index.enrich.return_value = {"processed": 1, "remaining": 0, "errors": []}
        index.close.side_effect = OSError("synthetic close failure")
        with patch.object(self.server, "index", return_value=index):
            self.server.start_enrichment()
            self.wait_for_job("failed")
        self.assertIn("close failure", self.server.semantic_job["error"])

    def test_enrichment_zero_success_error_marks_failed_and_retains_report(self):
        self.server.provider = object()
        index = MagicMock()
        error_report = {
            "processed": 0, "remaining": 2, "vectors": 0, "topics": 0,
            "errors": [{"conversation_id": "c1", "message_id": "m1", "error": "synthetic rate limit"}],
            "model_id": "synthetic",
        }
        index.enrich.return_value = error_report
        with patch.object(self.server, "index", return_value=index):
            self.server.start_enrichment()
            self.wait_for_job("failed")
        job = self.server.semantic_job
        self.assertEqual(job["state"], "failed")
        self.assertIsNone(job["error"])
        self.assertEqual(job["report"], error_report)
        self.assertEqual(job["report"]["processed"], 0)
        self.assertEqual(len(job["report"]["errors"]), 1)

    def test_enrichment_partial_success_marks_partial_and_retains_report(self):
        self.server.provider = object()
        index = MagicMock()
        partial_report = {
            "processed": 1, "remaining": 1, "vectors": 2, "topics": 0,
            "errors": [{"conversation_id": "c1", "message_id": "m2", "error": "synthetic message timeout"}],
            "model_id": "synthetic",
        }
        index.enrich.return_value = partial_report
        with patch.object(self.server, "index", return_value=index):
            self.server.start_enrichment()
            self.wait_for_job("partial")
        job = self.server.semantic_job
        self.assertEqual(job["state"], "partial")
        self.assertIsNone(job["error"])
        self.assertEqual(job["report"], partial_report)
        self.assertEqual(job["report"]["processed"], 1)
        self.assertEqual(job["report"]["remaining"], 1)

    def test_enrichment_manual_retry_succeeds_after_partial_or_failure(self):
        self.server.provider = object()
        index = MagicMock()
        partial_report = {
            "processed": 1, "remaining": 1, "vectors": 2, "topics": 0,
            "errors": [{"conversation_id": "c1", "message_id": "m2", "error": "synthetic transient failure"}],
            "model_id": "synthetic",
        }
        index.enrich.return_value = partial_report
        with patch.object(self.server, "index", return_value=index):
            self.server.start_enrichment()
            self.wait_for_job("partial")
        self.assertEqual(self.server.semantic_job["state"], "partial")

        # Retry after partial failure runs to clean completion.
        retry_report = {
            "processed": 1, "remaining": 0, "vectors": 2, "topics": 0,
            "errors": [], "model_id": "synthetic",
        }
        retry_index = MagicMock()
        retry_index.enrich.return_value = retry_report
        with patch.object(self.server, "index", return_value=retry_index):
            self.server.start_enrichment()
            self.wait_for_job("complete")
        job = self.server.semantic_job
        self.assertEqual(job["state"], "complete")
        self.assertIsNone(job["error"])
        self.assertEqual(job["report"]["processed"], 1)
        self.assertEqual(job["report"]["remaining"], 0)
        self.assertEqual(job["report"]["errors"], [])

    def test_enrichment_clean_success_marks_complete(self):
        self.server.provider = object()
        index = MagicMock()
        clean_report = {
            "processed": 2, "remaining": 0, "vectors": 4, "topics": 0,
            "errors": [], "model_id": "synthetic",
        }
        index.enrich.return_value = clean_report
        with patch.object(self.server, "index", return_value=index):
            self.server.start_enrichment()
            self.wait_for_job("complete")
        job = self.server.semantic_job
        self.assertEqual(job["state"], "complete")
        self.assertIsNone(job["error"])
        self.assertEqual(job["report"], clean_report)
        self.assertEqual(job["report"]["processed"], 2)
        self.assertEqual(job["report"]["errors"], [])


if __name__ == "__main__":
    unittest.main()
