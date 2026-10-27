"""Fake terminal runtimes only: never download or load an inference model."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import gallery_worker


@unittest.skipUnless(os.name == "posix", "Gallery terminal runtime uses macOS/Linux process groups")
class GalleryWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name).resolve() / "project"
        self.project.mkdir()
        self.model = self.project / "existing.litertlm"
        self.model.write_bytes(b"synthetic model fixture; never loaded")
        self.runtime = self.project / "fake-runtime"
        self.root_patch = patch.object(gallery_worker, "PROJECT_ROOT", self.project)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def fake(self, code):
        self.runtime.write_text(f"#!{sys.executable}\n" + code + "\n", encoding="utf-8")
        self.runtime.chmod(0o700)
        return [str(self.runtime)]

    def arguments(self, *extra):
        return ["--model-path", str(self.model), "--runtime", str(self.runtime),
                "--config", str(self.project / ".runtime/config.json"), *extra]

    def invoke(self, *extra, prompt="Implement only the assigned synthetic task."):
        output = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO(prompt)), contextlib.redirect_stdout(output):
            result = gallery_worker.main(self.arguments(*extra))
        return result, output.getvalue()

    def assert_dead(self, pid):
        # A killed orphan may briefly remain a zombie before the host reaps it.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            status = Path(f"/proc/{pid}/status")
            if status.is_file() and "State:\tZ" in status.read_text():
                return
            time.sleep(0.02)
        self.fail(f"Child {pid} survived bounded runtime cleanup")

    @staticmethod
    def cleanup_child(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def test_dry_run_preserves_interface_without_launching_runtime(self):
        marker = self.project / "runtime-launched"
        self.fake(f"from pathlib import Path\nPath({str(marker)!r}).touch()")
        result, text = self.invoke("--dry-run")
        configuration = json.loads(text)
        self.assertEqual(result, 0)
        self.assertEqual(configuration["role"], "bounded-local-worker")
        self.assertEqual(configuration["context_tokens"], 2048)
        self.assertFalse(configuration["authority_roles"])
        self.assertFalse(configuration["downloads_model"])
        self.assertEqual(configuration["model_file"], "existing.litertlm")
        self.assertFalse(marker.exists())
        self.assertFalse((self.project / ".runtime").exists())

    def test_success_prints_and_exclusively_saves_bounded_proposal(self):
        self.fake("print('A bounded proposal with 中文 and emoji 🙂')")
        destination = self.project / ".runtime/proposals/task-one.txt"
        result, text = self.invoke("--output", str(destination))
        self.assertEqual(result, 0)
        self.assertEqual(destination.read_text(), text)
        self.assertIn("bounded proposal", text)
        self.assertEqual(destination.stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.invoke("--output", str(destination))
        self.assertEqual(destination.read_text(), text)

    def test_stdout_noise_is_rejected_without_saving_partial_output(self):
        self.fake("import os\nwhile True: os.write(1, b'x' * 8192)")
        destination = self.project / ".runtime/proposals/noise.txt"
        with self.assertRaisesRegex(ValueError, "stdout byte limit"):
            self.invoke("--output", str(destination), "--timeout", "2")
        self.assertFalse(destination.exists())

    def test_stderr_noise_is_rejected_without_saving_partial_output(self):
        self.fake("import os\nwhile True: os.write(2, b'x' * 8192)")
        destination = self.project / ".runtime/proposals/noise.txt"
        with self.assertRaisesRegex(ValueError, "stderr byte limit"):
            self.invoke("--output", str(destination), "--timeout", "2")
        self.assertFalse(destination.exists())

    def test_exact_byte_caps_are_accepted_and_both_streams_drained(self):
        self.fake(f"import os\nos.write(1, b'p' * {gallery_worker.MAX_PROPOSAL_BYTES})\n"
                  f"os.write(2, b'd' * {gallery_worker.MAX_DIAGNOSTIC_BYTES})")
        result = gallery_worker.run_bounded([str(self.runtime)], 2)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(result.stdout), gallery_worker.MAX_PROPOSAL_BYTES)
        self.assertEqual(len(result.stderr), gallery_worker.MAX_DIAGNOSTIC_BYTES)

    def test_nonzero_runtime_prints_only_bounded_diagnostics(self):
        self.fake("import sys\nsys.stdout.write('not a valid proposal')\n"
                  "sys.stderr.write('diagnostic ' * 1000)\nraise SystemExit(3)")
        destination = self.project / ".runtime/proposals/failed.txt"
        diagnostic = io.StringIO()
        with contextlib.redirect_stderr(diagnostic):
            with self.assertRaisesRegex(ValueError, "LiteRT-LM failed"):
                self.invoke("--output", str(destination))
        self.assertLessEqual(len(diagnostic.getvalue()), 4001)
        self.assertFalse(destination.exists())

    def test_timeout_kills_own_process_group_including_child(self):
        child_pid = self.project / "child-pid"
        self.fake("import subprocess, sys, time\nfrom pathlib import Path\n"
                  "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                  f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
                  "time.sleep(30)")
        started = time.monotonic()
        with self.assertRaisesRegex(ValueError, "timed out"):
            gallery_worker.run_bounded([str(self.runtime)], 1)
        self.assertLess(time.monotonic() - started, 3)
        self.assertTrue(child_pid.exists())
        pid = int(child_pid.read_text())
        self.addCleanup(self.cleanup_child, pid)
        self.assert_dead(pid)

    def test_exited_parent_cannot_leave_child_holding_output_pipes(self):
        child_pid = self.project / "orphan-pid"
        self.fake("import subprocess, sys\nfrom pathlib import Path\n"
                  "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                  f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
                  "print('parent exits, child still owns pipe', flush=True)")
        with self.assertRaisesRegex(ValueError, "timed out"):
            gallery_worker.run_bounded([str(self.runtime)], 1)
        pid = int(child_pid.read_text())
        self.addCleanup(self.cleanup_child, pid)
        self.assert_dead(pid)

    def test_success_retires_background_children_with_closed_pipes(self):
        child_pid = self.project / "background-pid"
        self.fake("import subprocess, sys\nfrom pathlib import Path\n"
                  "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
                  "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                  f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
                  "print('proposal', flush=True)")
        result = gallery_worker.run_bounded([str(self.runtime)], 2)
        self.assertEqual(result.returncode, 0)
        pid = int(child_pid.read_text())
        self.addCleanup(self.cleanup_child, pid)
        self.assert_dead(pid)

    def test_source_or_arbitrary_output_rejected_before_inference(self):
        marker = self.project / "runtime-launched"
        self.fake(f"from pathlib import Path\nPath({str(marker)!r}).touch()")
        source = self.project / "scripts/source.py"
        source.parent.mkdir()
        source.write_text("original source\n")
        destinations = (source, self.project / "new-source.py",
                        self.project / ".runtime/proposals/../../governance/policy.json")
        for destination in destinations:
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError, "below this checkout"):
                    self.invoke("--output", str(destination))
        self.assertEqual(source.read_text(), "original source\n")
        self.assertFalse(marker.exists())

    def test_symlink_output_and_directory_escape_are_rejected(self):
        self.fake("print('proposal')")
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        proposals = self.project / ".runtime/proposals"
        proposals.mkdir(parents=True)
        (proposals / "escape").symlink_to(outside, target_is_directory=True)
        (proposals / "link.txt").symlink_to(outside / "new.txt")
        for destination in (proposals / "escape/new.txt", proposals / "link.txt"):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    self.invoke("--output", str(destination))
        self.assertEqual(list(outside.iterdir()), [])

    def test_symlink_runtime_root_is_rejected(self):
        self.fake("print('proposal')")
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (self.project / ".runtime").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.invoke("--output", str(self.project / ".runtime/proposals/new.txt"))
        self.assertEqual(list(outside.iterdir()), [])

    def test_exclusive_open_cannot_overwrite_file_created_after_validation(self):
        proposals = self.project / ".runtime/proposals"
        proposals.mkdir(parents=True)
        target = proposals / "new.txt"
        target.write_text("created after preflight")
        with patch.object(gallery_worker, "proposal_parts", return_value=(".runtime", "proposals", "new.txt")):
            with self.assertRaises(FileExistsError):
                gallery_worker.write_proposal(target, b"must not overwrite")
        self.assertEqual(target.read_text(), "created after preflight")

    def test_descriptor_walk_cannot_follow_symlink_created_after_validation(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (self.project / ".runtime").mkdir()
        (self.project / ".runtime/proposals").symlink_to(outside, target_is_directory=True)
        target = self.project / ".runtime/proposals/new.txt"
        with patch.object(gallery_worker, "proposal_parts", return_value=(".runtime", "proposals", "new.txt")):
            with self.assertRaises(OSError):
                gallery_worker.write_proposal(target, b"must not escape")
        self.assertEqual(list(outside.iterdir()), [])

    def test_prompt_and_context_bounds_do_not_launch_inference(self):
        self.fake("raise AssertionError('runtime must not start')")
        for prompt in ("", "x" * 24001):
            with self.subTest(prompt_length=len(prompt)):
                with self.assertRaisesRegex(ValueError, "bounded task"):
                    self.invoke(prompt=prompt)
        with self.assertRaisesRegex(ValueError, "context"):
            self.invoke("--context", "4097")

    def test_helper_rejects_nonfinite_timeout_before_launch(self):
        self.fake("raise AssertionError('must not launch')")
        for timeout in (float("nan"), float("inf"), float("-inf"), 0, -1):
            with self.subTest(timeout=timeout), patch.object(subprocess, "Popen") as launch:
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    gallery_worker.run_bounded([str(self.runtime)], timeout)
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
