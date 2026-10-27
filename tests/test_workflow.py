"""Real Git/worktree and subprocess evidence tests; model identities are synthetic."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from chat_recall.workflow import Workflow, WorkflowError, main


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Synthetic test coordinator")
        self.git("config", "user.email", "test@example.invalid")
        (self.repo / "seed.txt").write_text("seed\n")
        (self.repo / ".gitignore").write_text("__pycache__/\n.runtime/\n")
        self.git("add", ".")
        self.git("commit", "-m", "test base")
        self.workflow = Workflow(self.repo)
        self.workflow.initialize()
        self.workflow.register_identity("codex-online", "local-worker", "Ollama",
                                        "synthetic-model", "local", "terminal",
                                        "synthetic local account", ["worker"])
        self.workflow.register_identity("codex-online", "agy-worker", "synthetic-cloud",
                                        "synthetic-online-model", "online", "agy",
                                        "synthetic account B", ["worker"])
        self.ok_command = shlex.quote(sys.executable) + " -c 'print(\"acceptance passed\")'"

    def git(self, *args, cwd=None):
        result = subprocess.run(["git", *args], cwd=cwd or self.repo, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr)
        return result.stdout.strip()

    def create(self, task_id="task-one", scopes=None, worker="local-worker", acceptance=None):
        checkout = self.base / ("worktree-" + task_id)
        task = self.workflow.create_task(
            "codex-online", task_id, "Implement a bounded synthetic feature",
            "worker/" + task_id, "HEAD", scopes or ["feature.txt"],
            acceptance or [self.ok_command], worker, worktree=str(checkout),
            constraints=["No network or credentials"],
        )
        return task, checkout

    def commit(self, checkout, path="feature.txt", text="implemented\n"):
        destination = checkout / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        self.git("add", path, cwd=checkout)
        self.git("commit", "-m", "Implement bounded feature", cwd=checkout)
        return self.git("rev-parse", "HEAD", cwd=checkout)

    def prepared(self, task_id="task-one", **kwargs):
        task, checkout = self.create(task_id, **kwargs)
        candidate = self.commit(checkout, path=task["scopes"][0])
        handoff = self.workflow.handoff(task["worker"], task_id, candidate, str(checkout), findings="Synthetic implementation")
        return task, checkout, candidate, handoff

    def single_owner_policy(self):
        path = self.repo / "governance/policy.json"
        policy = json.loads(path.read_text())
        policy["authority_owner"] = "codex-online"
        path.write_text(json.dumps(policy))

    def test_single_owner_blocks_legacy_online_authority_and_takeover(self):
        self.workflow.register_identity("codex-online", "legacy-online", "synthetic cloud",
                                        "synthetic model", "online", "agy", "synthetic",
                                        ["orchestrator", "architect", "reviewer", "integrator", "worker"])
        self.single_owner_policy()
        for role in ("architect", "orchestrator", "reviewer", "integrator"):
            with self.subTest(role=role), self.assertRaisesRegex(WorkflowError, "Single orchestrator"):
                self.workflow.authority("legacy-online", role, available=False)
        with self.assertRaisesRegex(WorkflowError, "forbids coordinator takeover"):
            self.workflow.select_coordinator("codex-online", "legacy-online", "Do not delegate leadership")
        self.workflow.record_decision("codex-online", "planning", "Keep one online authority")
        self.assertTrue(self.workflow.validate()["ok"])

    def test_single_owner_outage_preserves_worker_handoff_but_no_takeover(self):
        task, tree = self.create(worker="agy-worker")
        candidate = self.commit(tree)
        self.single_owner_policy()
        self.workflow.set_availability("codex-online", False, "Primary quota unavailable")
        self.assertEqual(self.workflow.handoff("agy-worker", task["id"], candidate, str(tree))["checks"][0]["returncode"], 0)
        with self.assertRaisesRegex(WorkflowError, "unavailable"):
            self.workflow.record_decision("codex-online", "planning", "Must wait")
        with self.assertRaises(WorkflowError):
            self.workflow.select_coordinator("agy-worker", "agy-worker", "No promotion")
        self.assertEqual(self.workflow.policy()["active_coordinator"], "codex-online")

    def test_single_owner_validation_rejects_changed_coordinator(self):
        self.single_owner_policy()
        path = self.repo / "governance/policy.json"
        policy = json.loads(path.read_text())
        policy["active_coordinator"] = "agy-worker"
        path.write_text(json.dumps(policy))
        with self.assertRaisesRegex(WorkflowError, "single authority owner"):
            self.workflow.validate()

    def test_local_models_cannot_take_authority_roles(self):
        with self.assertRaisesRegex(WorkflowError, "online"):
            self.workflow.record_decision("local-worker", "architecture", "Change all interfaces")
        with self.assertRaisesRegex(WorkflowError, "online"):
            self.workflow.create_task("local-worker", "denied", "assign", "worker/denied",
                                      "HEAD", ["feature.txt"], [self.ok_command], "local-worker")
        with self.assertRaisesRegex(WorkflowError, "Local model"):
            self.workflow.register_identity("codex-online", "bad-local", "local", "model",
                                            "local", "terminal", "account", ["reviewer"])

    def transfer_to_agy(self):
        self.workflow.register_identity("codex-online", "agy-lead", "synthetic-cloud",
                                        "synthetic-lead-model", "online", "Antigravity CLI",
                                        "synthetic", ["architect", "orchestrator", "reviewer", "integrator"])
        self.single_owner_policy()
        path = self.repo / "governance/policy.json"
        policy = json.loads(path.read_text())
        policy.update(authority_owner="agy-lead", active_coordinator="agy-lead")
        path.write_text(json.dumps(policy))

    def test_user_transferred_agy_can_review_integrate_without_codex(self):
        task, tree, candidate, handoff = self.prepared(worker="agy-worker")
        self.transfer_to_agy()
        for role in ("architect", "orchestrator", "reviewer", "integrator"):
            self.assertEqual(self.workflow.authority("agy-lead", role)["mode"], "online")
            with self.assertRaisesRegex(WorkflowError, "Single orchestrator"):
                self.workflow.authority("codex-online", role)
        self.workflow.record_decision("agy-lead", "planning", "Synthetic user-authorized continuation")
        self.workflow.review("agy-lead", task["id"], candidate, "Independent online review")
        result = self.workflow.integrate("agy-lead", task["id"])
        self.assertEqual(result["integrated_by"]["id"], "agy-lead")
        self.assertTrue(self.workflow.validate()["ok"])

    def test_agy_lead_launcher_uses_online_policy_model(self):
        from scripts.launch_agy_worker import launch_spec
        self.transfer_to_agy()
        spec = launch_spec(self.workflow)
        self.assertEqual(spec["role"], "lead")
        self.assertEqual(spec["actor"], "agy-lead")
        self.assertEqual(spec["model"], "synthetic-lead-model")
        self.assertEqual(spec["checkout"], self.repo.resolve())
        self.assertIn("Keep Smart disabled and importing stopped", spec["prompt"])
        self.assertIn("Codex is normal primary coordinator", spec["prompt"])
        self.assertIn("same identity", spec["prompt"])
        self.workflow.set_availability("agy-lead", False, "Synthetic outage")
        with self.assertRaisesRegex(WorkflowError, "unavailable"):
            launch_spec(self.workflow)

    def test_agy_worker_launcher_respects_assignment_model_and_branch(self):
        from scripts.launch_agy_worker import launch_spec
        self.workflow.register_identity("codex-online", "agy-fast", "synthetic-cloud",
                                        "synthetic-worker-model", "online", "Antigravity CLI",
                                        "synthetic", ["worker"])
        task, tree = self.create(worker="agy-fast")
        self.transfer_to_agy()
        spec = launch_spec(self.workflow, task["id"])
        self.assertEqual(spec["role"], "worker")
        self.assertEqual(spec["model"], "synthetic-worker-model")
        self.assertEqual(spec["checkout"], tree.resolve())
        self.assertEqual(spec["branch"], task["branch"])
        self.assertIn("Do not assign, approve, integrate", spec["prompt"])
        self.git("switch", "-c", "unexpected-branch", cwd=tree)
        with self.assertRaisesRegex(ValueError, "branch differs"):
            launch_spec(self.workflow, task["id"])

    def test_agy_launcher_rejects_untransferred_codex_coordinator(self):
        from scripts.launch_agy_worker import launch_spec
        self.single_owner_policy()
        with self.assertRaisesRegex(ValueError, "Active coordinator is Codex"):
            launch_spec(self.workflow)

    def test_optional_handover_flow_codex_agy_codex(self):
        from scripts.launch_agy_worker import launch_spec, return_to_codex
        # Register agy-lead-high-v1
        self.workflow.register_identity("codex-online", "agy-lead-high-v1", "Google",
                                        "gemini-3.8-flash-high", "online", "Antigravity CLI",
                                        "synthetic-high-account",
                                        ["architect", "orchestrator", "reviewer", "integrator"])
        # Configure policy with authorized_coordinators allowlist
        path = self.repo / "governance/policy.json"
        policy = json.loads(path.read_text())
        policy.update(
            authority_owner="codex-online",
            active_coordinator="codex-online",
            primary_coordinator="codex-online",
            fallback_coordinator="agy-lead-high-v1",
            authorized_coordinators=["codex-online", "agy-lead-high-v1"],
        )
        path.write_text(json.dumps(policy))

        # 1. Dry run under Codex with --take-over does not mutate policy, reports planned selection/model and handover_required
        dry = launch_spec(self.workflow, take_over=True, dry_run=True)
        self.assertTrue(dry["handover_required"])
        self.assertEqual(dry["planned_coordinator"], "agy-lead-high-v1")
        self.assertEqual(dry["planned_model"], "gemini-3.8-flash-high")
        self.assertEqual(self.workflow.policy()["active_coordinator"], "codex-online")
        self.assertEqual(self.workflow.policy()["authority_owner"], "codex-online")

        # 2. Dry run without take-over under Codex raises clear message
        with self.assertRaisesRegex(ValueError, "Active coordinator is Codex. Pass --take-over"):
            launch_spec(self.workflow, take_over=False, dry_run=True)

        # 3. Take-over explicitly transfers authority_owner + active_coordinator to agy-lead-high-v1
        from unittest.mock import patch
        with patch("shutil.which", return_value="/synthetic/bin/agy"):
            lead_spec = launch_spec(self.workflow, take_over=True, dry_run=False)
        self.assertEqual(lead_spec["role"], "lead")
        self.assertEqual(lead_spec["actor"], "agy-lead-high-v1")
        self.assertEqual(self.workflow.policy()["active_coordinator"], "agy-lead-high-v1")
        self.assertEqual(self.workflow.policy()["authority_owner"], "agy-lead-high-v1")
        self.assertEqual(self.workflow.policy()["requested_orchestrator_model"], "gemini-3.8-flash-high")

        # 4. Once AGY is active, take_over=False runs AGY lead normally
        lead_spec2 = launch_spec(self.workflow, take_over=False, dry_run=False)
        self.assertEqual(lead_spec2["actor"], "agy-lead-high-v1")

        # 5. Worker task mode under Codex or AGY: assigned online AGY worker can launch under current coordinator
        self.workflow.register_identity("agy-lead-high-v1", "agy-worker-fast", "Google",
                                        "gemini-3.8-flash-low", "online", "Antigravity CLI",
                                        "synthetic-worker", ["worker"])
        checkout = self.base / "worktree-task-under-lead"
        task = self.workflow.create_task(
            "agy-lead-high-v1", "task-under-lead", "Implement a bounded synthetic feature under lead",
            "worker/task-under-lead", "HEAD", ["feature.txt"],
            [self.ok_command], "agy-worker-fast", worktree=str(checkout),
            constraints=["No network or credentials"],
        )
        worker_spec = launch_spec(self.workflow, task_id=task["id"])
        self.assertEqual(worker_spec["role"], "worker")
        self.assertEqual(worker_spec["orchestrator"], "agy-lead-high-v1")

        # 6. Return to Codex dry-run is strictly read-only and does not mutate policy
        dry_return = return_to_codex(self.workflow, dry_run=True)
        self.assertTrue(dry_return["handover_required"])
        self.assertEqual(dry_return["planned_coordinator"], "codex-online")
        self.assertEqual(self.workflow.policy()["active_coordinator"], "agy-lead-high-v1")
        self.assertEqual(self.workflow.policy()["authority_owner"], "agy-lead-high-v1")

        # Return to Codex selects primary and restores authority
        return_to_codex(self.workflow)
        self.assertEqual(self.workflow.policy()["active_coordinator"], "codex-online")
        self.assertEqual(self.workflow.policy()["authority_owner"], "codex-online")
        self.assertEqual(self.workflow.policy()["requested_orchestrator_model"], "GPT-6 family; exact runtime variant not exposed")

        # 6b. Return to Codex when already at Codex has handover_required=False
        dry_return_already = return_to_codex(self.workflow, dry_run=True)
        self.assertFalse(dry_return_already["handover_required"])
        self.assertEqual(dry_return_already["planned_coordinator"], "codex-online")

        # Worker can also launch when Codex is active coordinator
        worker_spec_codex = launch_spec(self.workflow, task_id=task["id"])
        self.assertEqual(worker_spec_codex["role"], "worker")
        self.assertEqual(worker_spec_codex["orchestrator"], "codex-online")

        # 7. Unauthorized / local target rejection
        with self.assertRaisesRegex(WorkflowError, "not in authorized_coordinators"):
            self.workflow.select_coordinator("codex-online", "local-worker", "Attempt local target")

        self.workflow.register_identity("codex-online", "rogue-online", "OpenAI", "gpt-4",
                                        "online", "Codex", "rogue", ["orchestrator"])
        with self.assertRaisesRegex(WorkflowError, "not in authorized_coordinators"):
            self.workflow.select_coordinator("codex-online", "rogue-online", "Attempt unauthorized online")

        # Non-owner cannot transfer
        with self.assertRaisesRegex(WorkflowError, "Single orchestrator"):
            self.workflow.select_coordinator("rogue-online", "agy-lead-high-v1", "Non-owner takeover")

    def test_agy_missing_executable_leaves_owner_unchanged_before_selection(self):
        from unittest.mock import patch
        from scripts.launch_agy_worker import launch_spec
        self.workflow.register_identity("codex-online", "agy-lead-high-v1", "Google",
                                        "gemini-3.8-flash-high", "online", "Antigravity CLI",
                                        "synthetic-high-account",
                                        ["architect", "orchestrator", "reviewer", "integrator"])
        path = self.repo / "governance/policy.json"
        policy = json.loads(path.read_text())
        policy.update(
            authority_owner="codex-online",
            active_coordinator="codex-online",
            fallback_coordinator="agy-lead-high-v1",
            authorized_coordinators=["codex-online", "agy-lead-high-v1"],
        )
        path.write_text(json.dumps(policy))

        with patch("shutil.which", return_value=None):
            # Dry-run succeeds even if agy CLI is absent
            dry = launch_spec(self.workflow, take_over=True, dry_run=True)
            self.assertTrue(dry["handover_required"])

            # Actual take-over raises ValueError because agy is missing, but leaves Codex owner/active unchanged
            with self.assertRaisesRegex(ValueError, "Antigravity CLI is unavailable"):
                launch_spec(self.workflow, take_over=True, dry_run=False)

            self.assertEqual(self.workflow.policy()["active_coordinator"], "codex-online")
            self.assertEqual(self.workflow.policy()["authority_owner"], "codex-online")

    def test_malformed_target_validation_and_read_only_checks(self):
        from scripts.launch_agy_worker import launch_spec
        path = self.repo / "governance/policy.json"
        policy = json.loads(path.read_text())
        policy.update(
            authority_owner="codex-online",
            active_coordinator="codex-online",
            fallback_coordinator="agy-lead-high-v1",
            authorized_coordinators=["codex-online", "agy-lead-high-v1"],
        )
        path.write_text(json.dumps(policy))

        # Target not in identities
        with self.assertRaisesRegex(WorkflowError, "Unknown actor"):
            launch_spec(self.workflow, take_over=True, dry_run=True)

        # Target with non-AGY tool
        self.workflow.register_identity("codex-online", "agy-lead-high-v1", "Google",
                                        "gemini-3.8-flash-high", "online", "Terminal",
                                        "synthetic-high-account",
                                        ["architect", "orchestrator", "reviewer", "integrator"])
        with self.assertRaisesRegex(ValueError, "not an Antigravity tool"):
            launch_spec(self.workflow, take_over=True, dry_run=True)

        # Target with local mode
        identities = self.workflow.identities()
        identities["agy-lead-high-v1"]["tool"] = "Antigravity CLI"
        identities["agy-lead-high-v1"]["mode"] = "local"
        (self.repo / "governance/identities.json").write_text(json.dumps(identities))
        with self.assertRaisesRegex(ValueError, "must be an online model"):
            launch_spec(self.workflow, take_over=True, dry_run=True)

        # Target missing required authority roles (e.g. only orchestrator)
        identities["agy-lead-high-v1"]["mode"] = "online"
        identities["agy-lead-high-v1"]["roles"] = ["orchestrator"]
        (self.repo / "governance/identities.json").write_text(json.dumps(identities))
        with self.assertRaisesRegex(ValueError, "lacks required authority roles"):
            launch_spec(self.workflow, take_over=True, dry_run=True)

        # Policy left completely unmutated throughout
        self.assertEqual(self.workflow.policy()["active_coordinator"], "codex-online")
        self.assertEqual(self.workflow.policy()["authority_owner"], "codex-online")

    def test_two_account_tools_and_local_worker_complete_isolated_handoffs(self):
        first, first_tree = self.create("first", scopes=["first.txt"], worker="local-worker")
        second, second_tree = self.create("second", scopes=["second.txt"], worker="agy-worker")
        first_commit = self.commit(first_tree, "first.txt")
        second_commit = self.commit(second_tree, "second.txt")
        self.assertFalse((first_tree / "second.txt").exists())
        self.assertFalse((second_tree / "first.txt").exists())
        for task, tree, candidate in [(first, first_tree, first_commit), (second, second_tree, second_commit)]:
            handoff = self.workflow.handoff(task["worker"], task["id"], candidate, str(tree))
            self.assertEqual(handoff["checks"][0]["returncode"], 0)
            self.assertIn("acceptance passed", handoff["checks"][0]["stdout"])
            self.workflow.review("codex-online", task["id"], candidate, "Inspected synthetic diff and actual checks")
            result = self.workflow.integrate("codex-online", task["id"])
            self.assertEqual(result["status"], "integrated")
        self.assertTrue((self.repo / "first.txt").exists())
        self.assertTrue((self.repo / "second.txt").exists())
        self.assertTrue(self.workflow.validate()["ok"])
        audit = [json.loads(line) for line in (self.repo / "governance/audit.jsonl").read_text().splitlines()]
        workers = [record["actor"] for record in audit if record["event"] == "handoff"]
        self.assertEqual({record["tool"] for record in workers}, {"terminal", "agy"})
        self.assertEqual({record["mode"] for record in workers}, {"local", "online"})

    def test_active_task_branch_cannot_be_reused(self):
        task, checkout = self.create("first")
        second_checkout = self.base / "second-checkout"
        with self.assertRaisesRegex(WorkflowError, "Branch is already owned by active task first"):
            self.workflow.create_task("codex-online", "second", "Different task",
                                      task["branch"], "HEAD", ["other.txt"],
                                      [self.ok_command], "agy-worker", worktree=str(second_checkout))
        self.assertFalse(second_checkout.exists())
        self.assertFalse((self.repo / "governance/tasks/second.json").exists())

    def test_active_task_worktree_cannot_be_reused_or_aliased(self):
        task, checkout = self.create("first")
        alias = self.base / "worktree-alias"
        alias.symlink_to(checkout, target_is_directory=True)
        for destination in (checkout, alias):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(WorkflowError, "Worktree is already owned by active task first"):
                    self.workflow.create_task("codex-online", "second", "Different task",
                                              "worker/second", "HEAD", ["other.txt"],
                                              [self.ok_command], "agy-worker", worktree=str(destination))
        self.assertNotEqual(self.git("branch", "--show-current", cwd=checkout), "worker/second")
        self.assertFalse((self.repo / "governance/tasks/second.json").exists())

    def test_integrated_branch_worktree_can_be_reassigned_to_a_new_task(self):
        task, checkout, candidate, _ = self.prepared()
        self.workflow.review("codex-online", task["id"], candidate, "Reviewed before reuse")
        self.workflow.integrate("codex-online", task["id"])
        replacement = self.workflow.create_task("codex-online", "next-task", "Next bounded task",
                                                task["branch"], candidate, ["feature.txt"],
                                                [self.ok_command], "agy-worker", worktree=str(checkout))
        self.assertEqual(replacement["status"], "assigned")

    def test_offline_controller_allows_existing_handoff_but_no_authority_actions(self):
        task, checkout = self.create()
        candidate = self.commit(checkout)
        self.workflow.set_availability("codex-online", False, "Synthetic outage")
        handoff = self.workflow.handoff("local-worker", task["id"], candidate, str(checkout))
        self.assertEqual(handoff["candidate_commit"], candidate)
        for call in [
            lambda: self.workflow.record_decision("codex-online", "planning", "new plan"),
            lambda: self.workflow.assign("codex-online", task["id"], "agy-worker"),
            lambda: self.workflow.review("codex-online", task["id"], candidate, "review"),
            lambda: self.workflow.integrate("codex-online", task["id"]),
        ]:
            with self.assertRaisesRegex(WorkflowError, "unavailable"):
                call()
        self.workflow.set_availability("codex-online", True, "Synthetic recovery")
        self.workflow.review("codex-online", task["id"], candidate, "Reviewed after recovery")
        self.assertEqual(self.workflow.integrate("codex-online", task["id"])["status"], "integrated")

    def test_quota_outage_can_switch_only_to_registered_online_coordinator(self):
        self.workflow.register_identity("codex-online", "backup-online", "synthetic second provider",
                                        "synthetic backup model", "online", "agy",
                                        "synthetic backup account", ["orchestrator", "architect", "reviewer", "integrator"])
        self.workflow.set_availability("codex-online", False, "Primary model quota exhausted")
        with self.assertRaisesRegex(WorkflowError, "online orchestrator"):
            self.workflow.select_coordinator("codex-online", "local-worker", "Must not promote a 2B local model")
        restored = self.workflow.select_coordinator("backup-online", "backup-online", "Registered online model available after primary quota failure")
        self.assertTrue(restored["coordinator_available"])
        self.assertEqual(restored["active_coordinator"], "backup-online")
        self.workflow.record_decision("backup-online", "planning", "Online fallback resumes planning")
        with self.assertRaisesRegex(WorkflowError, "selected online coordinator"):
            self.workflow.assign("codex-online", "unused", "local-worker")
        event = json.loads((self.repo / "governance/audit.jsonl").read_text().splitlines()[-2])
        self.assertEqual(event["event"], "coordinator-selected")
        self.assertEqual(event["selected"]["provider"], "synthetic second provider")
        self.assertEqual(event["selected"]["model"], "synthetic backup model")

    def test_integration_requires_review_and_online_integrator(self):
        task, checkout, candidate, _ = self.prepared()
        with self.assertRaisesRegex(WorkflowError, "online approval"):
            self.workflow.integrate("codex-online", task["id"])
        with self.assertRaisesRegex(WorkflowError, "online"):
            self.workflow.review("local-worker", task["id"], candidate, "self-approval")
        self.workflow.review("codex-online", task["id"], candidate, "Inspected source and evidence")
        with self.assertRaisesRegex(WorkflowError, "online"):
            self.workflow.integrate("local-worker", task["id"])

    def test_online_worker_still_needs_independent_review(self):
        self.workflow.register_identity("codex-online", "online-author", "synthetic-cloud", "model",
                                        "online", "ChatGPT", "synthetic C", ["worker", "reviewer"])
        task, checkout, candidate, _ = self.prepared(worker="online-author")
        with self.assertRaisesRegex(WorkflowError, "independent"):
            self.workflow.review("online-author", task["id"], candidate, "Approve my own change")

    def test_changed_candidate_invalidates_prior_review(self):
        task, checkout, candidate, _ = self.prepared()
        self.workflow.review("codex-online", task["id"], candidate, "Reviewed first candidate")
        new_candidate = self.commit(checkout, text="new implementation\n")
        with self.assertRaisesRegex(WorkflowError, "stale"):
            self.workflow.integrate("codex-online", task["id"])
        self.workflow.handoff("local-worker", task["id"], new_candidate, str(checkout))
        with self.assertRaisesRegex(WorkflowError, "online approval"):
            self.workflow.integrate("codex-online", task["id"])
        self.workflow.review("codex-online", task["id"], new_candidate, "Reviewed updated candidate")
        self.workflow.integrate("codex-online", task["id"])
        self.assertEqual((self.repo / "feature.txt").read_text(), "new implementation\n")

    def test_failed_actual_acceptance_prevents_approval(self):
        fail = shlex.quote(sys.executable) + " -c 'raise SystemExit(9)'"
        task, checkout, candidate, handoff = self.prepared(acceptance=[fail])
        self.assertEqual(handoff["checks"][0]["returncode"], 9)
        with self.assertRaisesRegex(WorkflowError, "passed"):
            self.workflow.review("codex-online", task["id"], candidate, "Cannot approve failed checks")

    def test_acceptance_timeout_kills_shell_children(self):
        import time
        sleep = shlex.quote(sys.executable) + " -c 'import time; time.sleep(10)'"
        task, checkout = self.create(acceptance=[sleep])
        candidate = self.commit(checkout)
        started = time.monotonic()
        handoff = self.workflow.handoff("local-worker", task["id"], candidate, str(checkout), timeout=0.05)
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(handoff["checks"][0]["returncode"], 124)
        self.assertIn("timed out", handoff["checks"][0]["stderr"])

    def test_acceptance_timeout_requires_finite_positive_value(self):
        from unittest.mock import patch
        task, checkout = self.create()
        candidate = self.commit(checkout)
        for timeout in (float("nan"), float("inf"), float("-inf"), 0, -1):
            with self.subTest(timeout=timeout), patch.object(self.workflow, "_run_check") as run:
                with self.assertRaisesRegex(WorkflowError, "finite positive"):
                    self.workflow.handoff("local-worker", task["id"], candidate, str(checkout), timeout=timeout)
                run.assert_not_called()
        self.assertIsNone(self.workflow.task(task["id"])["handoff"])

    def test_sensitive_local_change_requires_online_authorization(self):
        task, checkout, candidate, handoff = self.prepared(scopes=["pyproject.toml"])
        self.assertIn("online architecture", handoff["escalations"][0])
        with self.assertRaisesRegex(WorkflowError, "escalation"):
            self.workflow.review("codex-online", task["id"], candidate, "Needs architecture approval")
        with self.assertRaisesRegex(WorkflowError, "online"):
            self.workflow.authorize_changes("local-worker", task["id"], ["pyproject.toml"], "Self escalation")
        self.workflow.authorize_changes("codex-online", task["id"], ["pyproject.toml"],
                                        "Online architect explicitly approves this dependency metadata change")
        handoff = self.workflow.handoff("local-worker", task["id"], candidate, str(checkout))
        self.assertEqual(handoff["escalations"], [])
        self.workflow.review("codex-online", task["id"], candidate, "Reviewed approved architecture change")

    def test_out_of_scope_change_is_reported_even_for_online_worker(self):
        task, checkout = self.create(worker="agy-worker")
        candidate = self.commit(checkout, "outside.txt")
        handoff = self.workflow.handoff("agy-worker", task["id"], candidate, str(checkout))
        self.assertEqual(handoff["escalations"], ["outside scope: outside.txt"])
        with self.assertRaisesRegex(WorkflowError, "escalation"):
            self.workflow.review("codex-online", task["id"], candidate, "Scope conflict")

    def test_uncommitted_worktree_cannot_claim_commit_test_evidence(self):
        task, checkout = self.create()
        candidate = self.commit(checkout)
        (checkout / "feature.txt").write_text("untested worktree change")
        with self.assertRaisesRegex(WorkflowError, "Commit or remove"):
            self.workflow.handoff("local-worker", task["id"], candidate, str(checkout))

    def test_acceptance_may_not_rewrite_candidate(self):
        rewrite = shlex.quote(sys.executable) + " -c 'open(\"feature.txt\", \"w\").write(\"changed\")'"
        task, checkout = self.create(acceptance=[rewrite])
        candidate = self.commit(checkout)
        with self.assertRaisesRegex(WorkflowError, "changed the candidate"):
            self.workflow.handoff("local-worker", task["id"], candidate, str(checkout))

    def test_target_code_changes_after_review_require_new_review(self):
        task, checkout, candidate, _ = self.prepared()
        self.workflow.review("codex-online", task["id"], candidate, "Review before target change")
        self.commit(self.repo, "other.txt")
        with self.assertRaisesRegex(WorkflowError, "target code changed"):
            self.workflow.integrate("codex-online", task["id"])
        self.workflow.review("codex-online", task["id"], candidate, "Reviewed against new target")
        self.workflow.integrate("codex-online", task["id"])

    def test_audit_only_target_commit_does_not_stale_review(self):
        task, checkout, candidate, _ = self.prepared()
        self.workflow.review("codex-online", task["id"], candidate, "Actual online review")
        self.git("add", "governance")
        self.git("commit", "-m", "Record authority and review evidence")
        self.workflow.integrate("codex-online", task["id"])
        self.assertTrue(self.workflow.validate()["ok"])

    def test_merge_conflict_does_not_modify_checkout(self):
        task, checkout, candidate, _ = self.prepared()
        self.commit(self.repo, text="conflicting target version\n")
        self.workflow.review("codex-online", task["id"], candidate, "Inspected conflict; needs online resolution")
        before = self.git("rev-parse", "HEAD")
        with self.assertRaisesRegex(WorkflowError, "Merge conflicts"):
            self.workflow.integrate("codex-online", task["id"])
        self.assertEqual(self.git("rev-parse", "HEAD"), before)
        self.assertEqual((self.repo / "feature.txt").read_text(), "conflicting target version\n")

    def test_unresolved_issues_block_approval(self):
        task, checkout = self.create()
        candidate = self.commit(checkout)
        self.workflow.handoff("local-worker", task["id"], candidate, str(checkout), issues=["Known test gap"])
        with self.assertRaisesRegex(WorkflowError, "issues"):
            self.workflow.review("codex-online", task["id"], candidate, "Must resolve gap")

    def test_tampered_evidence_fails_validation(self):
        task, checkout, candidate, _ = self.prepared()
        self.workflow.review("codex-online", task["id"], candidate, "Inspected evidence")
        altered = self.workflow.task(task["id"])
        altered["handoff"]["checks"][0]["stdout"] = "rewritten evidence"
        self.workflow.save_task(altered)
        with self.assertRaisesRegex(WorkflowError, "evidence was changed"):
            self.workflow.validate()
        with self.assertRaisesRegex(WorkflowError, "no longer matches"):
            self.workflow.integrate("codex-online", task["id"])

    def test_reassignment_invalidates_prior_handoff(self):
        task, checkout, candidate, _ = self.prepared()
        self.workflow.assign("codex-online", task["id"], "agy-worker")
        with self.assertRaisesRegex(WorkflowError, "assigned worker"):
            self.workflow.handoff("local-worker", task["id"], candidate, str(checkout))
        with self.assertRaisesRegex(WorkflowError, "current handoff"):
            self.workflow.review("codex-online", task["id"], candidate, "Old worker evidence invalid")

    def test_validation_works_after_worker_branches_are_removed(self):
        task, checkout, candidate, _ = self.prepared()
        self.workflow.review("codex-online", task["id"], candidate, "Inspected source")
        self.workflow.integrate("codex-online", task["id"])
        self.git("worktree", "remove", str(checkout))
        self.git("branch", "-d", task["branch"])
        self.assertTrue(self.workflow.validate()["ok"])

    def test_invalid_task_and_path_inputs_are_rejected(self):
        with self.assertRaisesRegex(WorkflowError, "IDs"):
            self.workflow.task("../../outside")
        with self.assertRaisesRegex(WorkflowError, "repository-relative"):
            self.create(scopes=["../outside.txt"])
        with self.assertRaisesRegex(WorkflowError, "Invalid Git revision"):
            self.workflow.resolve("--help")

    def test_initialize_preserves_existing_availability(self):
        self.workflow.set_availability("codex-online", False, "Synthetic outage")
        self.assertFalse(self.workflow.initialize()["policy"]["coordinator_available"])

    def test_cli_status_and_denial_are_machine_readable(self):
        import contextlib
        import io
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--repo", str(self.repo), "status"]), 0)
        self.assertIn("policy", json.loads(stdout.getvalue()))
        with contextlib.redirect_stderr(stderr):
            result = main(["--repo", str(self.repo), "--actor", "local-worker", "decision",
                           "--kind", "planning", "--summary", "Denied local plan"])
        self.assertEqual(result, 2)
        self.assertIn("online", json.loads(stderr.getvalue())["error"])

    def test_cli_mutations_require_explicit_identity(self):
        import contextlib
        import io
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = main(["--repo", str(self.repo), "decision", "--kind", "planning",
                           "--summary", "No silent default authority"])
        self.assertEqual(result, 2)
        self.assertIn("explicit --actor", json.loads(stderr.getvalue())["error"])


if __name__ == "__main__":
    unittest.main()
