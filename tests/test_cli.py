"""Explicit local-provider CLI wiring, using synthetic vectors and no network."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chat_recall.__main__ import default_data_dir, main
from chat_recall.search import SearchIndex


class SyntheticProvider:
    model_id = "synthetic:cli-model"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "archive.sqlite"
        self.provider = SyntheticProvider()
        with SearchIndex(self.db, self.provider) as index:
            index.import_payload({"id": "cli-fixture", "messages": [
                {"id": "m1", "role": "assistant", "text": "Store a printed emergency key offline."}]})
            index.enrich()
        self.provider.calls = 0

    def run_search(self, *arguments):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = main(["search", "backup recovery", "--mode", "smart", "--db", str(self.db), *arguments])
        self.assertEqual(result, 0)
        return json.loads(stdout.getvalue())

    def test_smart_search_uses_explicit_matching_embedding_provider(self):
        with patch("chat_recall.providers.OllamaProvider", return_value=self.provider) as constructor:
            result = self.run_search("--embedding-model", "test-model", "--ollama-url", "http://127.0.0.1:11999")
        constructor.assert_called_once_with(base_url="http://127.0.0.1:11999", embedding_model="test-model")
        self.assertTrue(result["semantic"]["used"])
        self.assertEqual(result["results"][0]["conversation_id"], "cli-fixture")
        self.assertEqual(self.provider.calls, 1)

    def test_absent_cli_provider_stays_explicit_lexical_fallback(self):
        with patch("chat_recall.providers.OllamaProvider") as constructor:
            result = self.run_search()
        constructor.assert_not_called()
        self.assertFalse(result["semantic"]["used"])
        self.assertEqual(result["semantic"]["status"], "not_configured")

    def test_exact_search_does_not_invoke_configured_provider(self):
        stdout = io.StringIO()
        with patch("chat_recall.providers.OllamaProvider", return_value=self.provider), contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["search", "emergency key", "--db", str(self.db), "--embedding-model", "test-model"]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["total"], 1)
        self.assertEqual(self.provider.calls, 0)


class DefaultDataDirTests(unittest.TestCase):
    def test_env_override_explicit_relative_path(self):
        with patch.dict(os.environ, {"CHAT_RECALL_DATA_DIR": "local/recall_data"}, clear=False):
            data_dir = default_data_dir()
            self.assertEqual(data_dir, Path("local/recall_data"))
            self.assertFalse(data_dir.is_absolute())

    def test_env_override_expanduser(self):
        fake_home = Path("/synthetic/home")
        with patch.dict(os.environ, {"CHAT_RECALL_DATA_DIR": "~/custom_recall", "HOME": str(fake_home)}, clear=False), \
             patch("os.path.expanduser", side_effect=lambda p: str(fake_home) + p[1:] if p.startswith("~") else p):
            data_dir = default_data_dir()
            self.assertEqual(data_dir, fake_home / "custom_recall")

    def test_env_override_whitespace_or_empty_falls_back(self):
        fake_home = Path("/mock/home")
        for empty_val in ("", "   ", "\t\n"):
            with patch.dict(os.environ, {"CHAT_RECALL_DATA_DIR": empty_val}, clear=False), \
                 patch("sys.platform", "darwin"), \
                 patch.object(Path, "home", return_value=fake_home):
                self.assertEqual(
                    default_data_dir(),
                    fake_home / "Library" / "Application Support" / "Chat Recall",
                )

    def test_platform_default_darwin(self):
        fake_home = Path("/mock/darwin/home")
        clean_env = {k: v for k, v in os.environ.items() if k != "CHAT_RECALL_DATA_DIR"}
        with patch.dict(os.environ, clean_env, clear=True), \
             patch("sys.platform", "darwin"), \
             patch.object(Path, "home", return_value=fake_home):
            self.assertEqual(
                default_data_dir(),
                fake_home / "Library" / "Application Support" / "Chat Recall",
            )

    def test_platform_default_win32_localappdata(self):
        fake_local_app = Path("/mock/win/appdata")
        clean_env = {"LOCALAPPDATA": str(fake_local_app)}
        with patch.dict(os.environ, clean_env, clear=True), \
             patch("sys.platform", "win32"):
            self.assertEqual(
                default_data_dir(),
                fake_local_app / "Chat Recall",
            )

    def test_platform_default_win32_fallback_home(self):
        fake_home = Path("/mock/win/home")
        with patch.dict(os.environ, {}, clear=True), \
             patch("sys.platform", "win32"), \
             patch.object(Path, "home", return_value=fake_home):
            self.assertEqual(
                default_data_dir(),
                fake_home / "AppData" / "Local" / "Chat Recall",
            )

    def test_platform_default_win32_whitespace_localappdata_falls_back(self):
        fake_home = Path("/mock/win/home")
        with patch.dict(os.environ, {"LOCALAPPDATA": "   "}, clear=True), \
             patch("sys.platform", "win32"), \
             patch.object(Path, "home", return_value=fake_home):
            self.assertEqual(
                default_data_dir(),
                fake_home / "AppData" / "Local" / "Chat Recall",
            )

    def test_platform_default_linux_xdg_data_home(self):
        fake_xdg = Path("/mock/xdg/data")
        clean_env = {"XDG_DATA_HOME": str(fake_xdg)}
        with patch.dict(os.environ, clean_env, clear=True), \
             patch("sys.platform", "linux"):
            self.assertEqual(
                default_data_dir(),
                fake_xdg / "chat-recall",
            )

    def test_platform_default_linux_fallback_home(self):
        fake_home = Path("/mock/linux/home")
        with patch.dict(os.environ, {}, clear=True), \
             patch("sys.platform", "linux"), \
             patch.object(Path, "home", return_value=fake_home):
            self.assertEqual(
                default_data_dir(),
                fake_home / ".local" / "share" / "chat-recall",
            )

    def test_platform_default_linux_whitespace_xdg_falls_back(self):
        fake_home = Path("/mock/linux/home")
        with patch.dict(os.environ, {"XDG_DATA_HOME": "   "}, clear=True), \
             patch("sys.platform", "linux"), \
             patch.object(Path, "home", return_value=fake_home):
            self.assertEqual(
                default_data_dir(),
                fake_home / ".local" / "share" / "chat-recall",
            )

    def test_cli_commands_share_default_db_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            synth_dir = Path(tmp) / "app_data"
            db_path = synth_dir / "recall.sqlite3"
            with patch("chat_recall.__main__.default_data_dir", return_value=synth_dir):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(main(["status"]), 0)
                self.assertTrue(db_path.exists())
                status = json.loads(stdout.getvalue())
                self.assertEqual(status["conversations"], 0)

                with SearchIndex(db_path) as idx:
                    idx.import_payload({"id": "synth-shared", "messages": [{"id": "m1", "role": "user", "text": "findable token"}]})
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(main(["search", "findable"]), 0)
                search_res = json.loads(stdout.getvalue())
                self.assertEqual(search_res["total"], 1)

            explicit_db = Path(tmp) / "override" / "other.sqlite3"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["status", "--db", str(explicit_db)]), 0)
            self.assertTrue(explicit_db.exists())

    def test_serve_cli_default_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            synth_dir = Path(tmp) / "serve_data"
            with patch("chat_recall.__main__.default_data_dir", return_value=synth_dir), \
                 patch("chat_recall.server.RecallServer") as mock_server, \
                 patch("chat_recall.server.load_token", return_value="synthetic_token"), \
                 patch("chat_recall.search.SearchIndex"):
                mock_server.return_value.server_port = 8765
                mock_server.return_value.serve_forever.return_value = None
                self.assertEqual(main(["serve"]), 0)
                self.assertTrue(synth_dir.exists())
                mock_server.assert_called_once()
                self.assertEqual(mock_server.call_args[0][1], synth_dir / "recall.sqlite3")

            explicit_dir = Path(tmp) / "explicit_serve"
            with patch("chat_recall.server.RecallServer") as mock_server, \
                 patch("chat_recall.server.load_token", return_value="synthetic_token"), \
                 patch("chat_recall.search.SearchIndex"):
                mock_server.return_value.server_port = 8765
                mock_server.return_value.serve_forever.return_value = None
                self.assertEqual(main(["serve", "--data-dir", str(explicit_dir)]), 0)
                self.assertTrue(explicit_dir.exists())
                self.assertEqual(mock_server.call_args[0][1], explicit_dir / "recall.sqlite3")


if __name__ == "__main__":
    unittest.main()
