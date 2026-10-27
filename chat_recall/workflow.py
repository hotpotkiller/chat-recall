"""Portable, auditable online-led task coordination (no model API required).

Configured model identities are human/operator attestations. This program gates
the workflow against those attestations; it cannot prove where inference ran.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


AUTHORITY_ROLES = {"architect", "orchestrator", "reviewer", "integrator"}
ALL_ROLES = AUTHORITY_ROLES | {"worker"}
DECISION_KINDS = {"requirements", "planning", "architecture", "evaluation"}
SENSITIVE_NAMES = {
    "AGENTS.md", "pyproject.toml", "package.json", "package-lock.json",
    "pnpm-lock.yaml", "Cargo.toml", "Cargo.lock", "requirements.txt",
    "uv.lock", "rust-toolchain.toml",
}
DEFAULT_POLICY = {
    "version": 1,
    "coordinator_available": True,
    "active_coordinator": "codex-online",
    "target_branch": "main",
    "independent_review": True,
    "authority_roles": sorted(AUTHORITY_ROLES),
    "identity_verification": "operator-attestation; not cryptographic remote-inference proof",
}
DEFAULT_IDENTITIES = {
    "codex-online": {
        "provider": "OpenAI",
        "model": "GPT-6 family; exact runtime variant not exposed",
        "mode": "online",
        "tool": "Codex",
        "account": "current online Codex session",
        "roles": sorted(ALL_ROLES),
    }
}


class WorkflowError(ValueError):
    """A denied or invalid workflow transition."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise WorkflowError("IDs must be 1–64 lowercase letters, numbers, underscores or hyphens")
    return value


def _json_read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Cannot read {path.name}: {exc}") from exc


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _sensitive(path: str) -> bool:
    parts = Path(path).parts
    return (
        Path(path).name in SENSITIVE_NAMES
        or any(part in {"contracts", "migrations", "schemas", "governance"} for part in parts)
        or path == "docs/architecture.md"
        or path.startswith(".github/")
    )


class Workflow:
    """State is portable JSON under governance/; Git is the code source of truth."""

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.state = self.root / "governance"

    def git(self, *arguments: str, check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *arguments], cwd=cwd or self.root, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if check and result.returncode:
            raise WorkflowError((result.stderr or result.stdout).strip() or "Git operation failed")
        return result

    @contextlib.contextmanager
    def lock(self):
        """Serialize controllers sharing a Git repository, including worktrees."""
        import fcntl  # Supported execution hosts: macOS and Linux.

        common = self.git("rev-parse", "--git-common-dir").stdout.strip()
        lock_path = (self.root / common).resolve() / "chat-recall-workflow.lock"
        with lock_path.open("a", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WorkflowError("Another controller is updating this repository") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def initialize(self) -> dict[str, Any]:
        self.git("rev-parse", "--show-toplevel")
        if not (self.state / "policy.json").exists():
            _json_write(self.state / "policy.json", DEFAULT_POLICY)
        if not (self.state / "identities.json").exists():
            _json_write(self.state / "identities.json", DEFAULT_IDENTITIES)
        return self.status()

    def policy(self) -> dict[str, Any]:
        return _json_read(self.state / "policy.json")

    def identities(self) -> dict[str, Any]:
        return _json_read(self.state / "identities.json")

    def identity(self, actor: str) -> dict[str, Any]:
        identities = self.identities()
        if actor not in identities:
            raise WorkflowError(f"Unknown actor {actor!r}; an online orchestrator must register it")
        record = identities[actor]
        if record.get("mode") not in {"online", "local"}:
            raise WorkflowError(f"Invalid mode for {actor}")
        return {"id": actor, **record}

    def authority(self, actor: str, role: str, *, available: bool = True) -> dict[str, Any]:
        identity = self.identity(actor)
        if identity["mode"] != "online" or role not in identity.get("roles", []):
            raise WorkflowError(f"{role} authority requires a configured online model")
        owner = self.policy().get("authority_owner")
        if owner is not None and actor != owner:
            raise WorkflowError(f"Single orchestrator policy reserves authority for {owner}; other models are workers only")
        if available and not self.policy().get("coordinator_available", False):
            raise WorkflowError("Online coordinator unavailable: authority actions wait; existing workers may hand off")
        if available and role == "orchestrator" and actor != self.policy().get("active_coordinator", "codex-online"):
            raise WorkflowError("Only the selected online coordinator may orchestrate; select a registered fallback first")
        return identity

    def register_identity(self, actor: str, identity_id: str, provider: str, model: str,
                          mode: str, tool: str, account: str, roles: list[str]) -> dict[str, Any]:
        self.authority(actor, "orchestrator")
        _identifier(identity_id)
        if mode not in {"online", "local"} or not set(roles) <= ALL_ROLES or not roles:
            raise WorkflowError("Invalid identity mode or roles")
        if mode == "local" and set(roles) & AUTHORITY_ROLES:
            raise WorkflowError("Local model identities may only have the worker role")
        if not all(isinstance(value, str) and value.strip() for value in (provider, model, tool, account)):
            raise WorkflowError("Provider, model, tool and account labels are required")
        identities = self.identities()
        if identity_id in identities:
            raise WorkflowError("Identity already exists; add a new versioned identity instead of rewriting provenance")
        record = {"provider": provider, "model": model, "mode": mode,
                  "tool": tool, "account": account, "roles": sorted(set(roles))}
        identities[identity_id] = record
        _json_write(self.state / "identities.json", identities)
        self.audit("identity-registered", actor, {"identity_id": identity_id, "identity": record})
        return record

    def audit(self, event: str, actor: str, detail: dict[str, Any]) -> None:
        identity = self.identity(actor)
        record = {"event": event, "at": now(), "actor": identity, **detail}
        # The ledger is inspectable/versioned; repository controls provide integrity.
        ledger = self.state / "audit.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def set_availability(self, actor: str, available: bool, reason: str) -> dict[str, Any]:
        self.authority(actor, "orchestrator", available=False)
        if not reason.strip():
            raise WorkflowError("Availability changes require a reason")
        policy = self.policy()
        policy["coordinator_available"] = available
        _json_write(self.state / "policy.json", policy)
        self.audit("availability", actor, {"available": available, "reason": reason})
        return policy

    def select_coordinator(self, actor: str, identity_id: str, reason: str) -> dict[str, Any]:
        """An existing online authority may select another registered online model."""
        self.authority(actor, "orchestrator", available=False)
        target = self.identity(identity_id)
        policy = self.policy()
        owner = policy.get("authority_owner")
        authorized = policy.get("authorized_coordinators")
        if authorized is not None:
            if actor != owner:
                raise WorkflowError("Only the current authority owner may transfer coordinator authority")
            if identity_id not in authorized:
                raise WorkflowError(f"Target coordinator {identity_id!r} is not in authorized_coordinators")
            if target.get("mode") != "online" or "orchestrator" not in target.get("roles", []):
                raise WorkflowError("Fallback coordinator must be an already registered online orchestrator")
            owner = identity_id
        else:
            if owner is not None and identity_id != owner:
                raise WorkflowError("Single orchestrator policy forbids coordinator takeover")
            if target["mode"] != "online" or "orchestrator" not in target.get("roles", []):
                raise WorkflowError("Fallback coordinator must be an already registered online orchestrator")
        if not reason.strip():
            raise WorkflowError("Coordinator selection requires a reason, including any quota/outage")
        previous = policy.get("active_coordinator", "codex-online")
        policy.update(active_coordinator=identity_id, authority_owner=owner, coordinator_available=True)
        if target.get("model"):
            policy["requested_orchestrator_model"] = target["model"]
        _json_write(self.state / "policy.json", policy)
        self.audit("coordinator-selected", actor, {"previous": previous, "selected": target,
                                                   "reason": reason})
        return policy

    def record_decision(self, actor: str, kind: str, summary: str) -> dict[str, Any]:
        self.authority(actor, "architect")
        if kind not in DECISION_KINDS or not summary.strip():
            raise WorkflowError("A supported decision kind and nonempty summary are required")
        record = {"at": now(), "actor": self.identity(actor), "kind": kind, "summary": summary}
        decisions = self.state / "decisions.jsonl"
        with decisions.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.audit("decision", actor, {"kind": kind, "summary": summary})
        return record

    def resolve(self, revision: str) -> str:
        if not revision or revision.startswith("-"):
            raise WorkflowError("Invalid Git revision")
        return self.git("rev-parse", "--verify", "--end-of-options", revision + "^{commit}").stdout.strip()

    def task(self, task_id: str) -> dict[str, Any]:
        return _json_read(self.state / "tasks" / (_identifier(task_id) + ".json"))

    def save_task(self, task: dict[str, Any]) -> None:
        _json_write(self.state / "tasks" / (task["id"] + ".json"), task)

    def create_task(self, actor: str, task_id: str, objective: str, branch: str,
                    base: str, scopes: list[str], acceptance: list[str], worker: str,
                    dependencies: list[str] | None = None, worktree: str | None = None,
                    constraints: list[str] | None = None) -> dict[str, Any]:
        authority = self.authority(actor, "orchestrator")
        _identifier(task_id)
        if (self.state / "tasks" / (task_id + ".json")).exists():
            raise WorkflowError("Task already exists")
        self.git("check-ref-format", "--branch", branch)
        if branch == self.policy()["target_branch"]:
            raise WorkflowError("Worker tasks require their own branch")
        if "worker" not in self.identity(worker).get("roles", []):
            raise WorkflowError("Assigned actor is not a worker")
        if not objective.strip() or not scopes or not acceptance or any(not item.strip() for item in acceptance):
            raise WorkflowError("A bounded objective, scopes and acceptance commands are required")
        if any(Path(pattern).is_absolute() or ".." in Path(pattern).parts for pattern in scopes):
            raise WorkflowError("Scope patterns must be repository-relative and may not traverse upwards")
        destination = (self.root / worktree).resolve() if worktree else None
        for record in sorted((self.state / "tasks").glob("*.json")):
            active = _json_read(record)
            if active.get("status") == "integrated":
                continue
            if active.get("branch") == branch:
                raise WorkflowError(f"Branch is already owned by active task {active['id']}")
            existing_hint = active.get("worktree_hint")
            if destination is not None and existing_hint and (self.root / existing_hint).resolve() == destination:
                raise WorkflowError(f"Worktree is already owned by active task {active['id']}")
        dependencies = dependencies or []
        for dependency in dependencies:
            self.task(dependency)
        base_commit = self.resolve(base)
        existing = self.git("show-ref", "--verify", "refs/heads/" + branch, check=False)
        if worktree:
            if destination.exists():
                actual_branch = self.git("branch", "--show-current", cwd=destination).stdout.strip()
                if actual_branch != branch:
                    raise WorkflowError("Existing worktree belongs to a different branch")
            elif existing.returncode == 0:
                self.git("worktree", "add", str(destination), branch)
            else:
                self.git("worktree", "add", "-b", branch, str(destination), base_commit)
            worktree_hint = os.path.relpath(destination, self.root)
        else:
            if existing.returncode:
                self.git("branch", branch, base_commit)
            worktree_hint = None
        record = {
            "id": task_id, "objective": objective, "base_commit": base_commit,
            "branch": branch, "scopes": scopes, "acceptance": acceptance,
            "worker": worker, "dependencies": dependencies,
            "constraints": constraints or [],
            "escalation_conditions": ["outside assigned scope", "new dependencies",
                                      "schema/contracts/governance changes", "requirements change"],
            "authorized_changes": [], "created_by": authority, "created_at": now(),
            "worktree_hint": worktree_hint, "revision": 1, "status": "assigned",
            "handoff": None, "reviews": [],
        }
        self.validate_candidate(record, self.resolve(branch), require_tip=False)
        self.save_task(record)
        self.audit("task-created", actor, {"task_id": task_id, "worker": worker,
                                           "branch": branch, "base_commit": base_commit})
        return record

    def assign(self, actor: str, task_id: str, worker: str) -> dict[str, Any]:
        self.authority(actor, "orchestrator")
        task = self.task(task_id)
        if task["status"] == "integrated":
            raise WorkflowError("Integrated tasks cannot be reassigned")
        if "worker" not in self.identity(worker).get("roles", []):
            raise WorkflowError("Assigned actor is not a worker")
        task.update(worker=worker, status="assigned", handoff=None, revision=task["revision"] + 1)
        self.save_task(task)
        self.audit("task-assigned", actor, {"task_id": task_id, "worker": worker})
        return task

    def authorize_changes(self, actor: str, task_id: str, paths: list[str], reason: str) -> dict[str, Any]:
        self.authority(actor, "architect")
        task = self.task(task_id)
        if task["status"] == "integrated" or not paths or not reason.strip():
            raise WorkflowError("Active task, explicit paths and reason required")
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in paths):
            raise WorkflowError("Authorized paths must be repository-relative")
        task["authorized_changes"].extend({"path": path, "reason": reason,
                                           "actor": self.identity(actor), "at": now()} for path in paths)
        task.update(revision=task["revision"] + 1, status="assigned", handoff=None)
        self.save_task(task)
        self.audit("changes-authorized", actor, {"task_id": task_id, "paths": paths, "reason": reason})
        return task

    def changed_paths(self, base: str, candidate: str) -> list[str]:
        return [path for path in self.git("diff", "--no-renames", "--name-only", "-z", base, candidate).stdout.split("\0") if path]

    def validate_candidate(self, task: dict[str, Any], candidate: str, *, require_tip: bool = True) -> list[str]:
        candidate = self.resolve(candidate)
        if self.git("merge-base", "--is-ancestor", task["base_commit"], candidate, check=False).returncode:
            raise WorkflowError("Candidate is not descended from the task base commit")
        if require_tip and self.resolve(task["branch"]) != candidate:
            raise WorkflowError("Candidate is stale: worker branch has changed")
        return self.changed_paths(task["base_commit"], candidate)

    def escalations(self, task: dict[str, Any], paths: list[str]) -> list[str]:
        authorized = [record["path"] for record in task["authorized_changes"]]
        local_worker = self.identity(task["worker"])["mode"] == "local"
        result = []
        for path in paths:
            if _matches(path, authorized):
                continue
            if not _matches(path, task["scopes"]):
                result.append(f"outside scope: {path}")
            elif local_worker and _sensitive(path):
                result.append(f"online architecture authorization required: {path}")
        return result

    def _dirty_paths(self, checkout: Path) -> list[str]:
        tracked = self.git("diff", "HEAD", "--name-only", "-z", cwd=checkout).stdout
        untracked = self.git("ls-files", "--others", "--exclude-standard", "-z", cwd=checkout).stdout
        return sorted(set(path for path in (tracked + untracked).split("\0") if path))

    def _run_check(self, command: str, checkout: Path, timeout: float) -> dict[str, Any]:
        """Bound subprocess time/output, including children started by a shell."""
        started = time.monotonic()
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(command, shell=True, cwd=checkout,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            timed_out = False
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                returncode = 124

            def tail(stream):
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - 8000))
                return stream.read().decode("utf-8", errors="replace")

            return {"command": command, "returncode": returncode,
                    "stdout": tail(stdout),
                    "stderr": tail(stderr) + ("\nAcceptance command timed out" if timed_out else ""),
                    "seconds": round(time.monotonic() - started, 3)}

    def handoff(self, actor: str, task_id: str, candidate: str | None = None,
                worktree: str | None = None, findings: str = "", issues: list[str] | None = None,
                timeout: float = 300) -> dict[str, Any]:
        # Explicitly permitted while coordinator_available is false.
        task = self.task(task_id)
        if actor != task["worker"] or "worker" not in self.identity(actor).get("roles", []):
            raise WorkflowError("Only the assigned worker may submit this handoff")
        if task["status"] == "integrated":
            raise WorkflowError("Integrated tasks cannot receive another handoff")
        candidate_commit = self.resolve(candidate or task["branch"])
        changed = self.validate_candidate(task, candidate_commit)
        hint = worktree or task["worktree_hint"]
        if not hint:
            raise WorkflowError("Supply --worktree so acceptance commands run in the candidate checkout")
        checkout = (self.root / hint).resolve()
        if self.git("rev-parse", "HEAD", cwd=checkout).stdout.strip() != candidate_commit:
            raise WorkflowError("Acceptance worktree does not contain the exact candidate")
        if self.git("branch", "--show-current", cwd=checkout).stdout.strip() != task["branch"]:
            raise WorkflowError("Acceptance worktree is not on the task branch")
        if self._dirty_paths(checkout):
            raise WorkflowError("Commit or remove worktree changes before running acceptance checks")
        if not math.isfinite(timeout) or timeout <= 0:
            raise WorkflowError("Acceptance timeout must be a finite positive number")
        checks = [self._run_check(command, checkout, timeout) for command in task["acceptance"]]
        # Commands are trusted task inputs, but must not quietly rewrite tested code.
        if self._dirty_paths(checkout) or self.git("rev-parse", "HEAD", cwd=checkout).stdout.strip() != candidate_commit:
            raise WorkflowError("Acceptance checks changed the candidate checkout; commit and rerun")
        escalation = self.escalations(task, changed)
        handoff = {
            "at": now(), "worker": self.identity(actor), "candidate_commit": candidate_commit,
            "base_commit": task["base_commit"], "task_revision": task["revision"],
            "changed_paths": changed, "checks": checks, "findings": findings,
            "unresolved_issues": issues or [], "escalations": escalation,
        }
        task.update(handoff=handoff, status="needs-escalation" if escalation else "handed-off")
        self.save_task(task)
        self.audit("handoff", actor, {"task_id": task_id, "candidate_commit": candidate_commit,
                                      "checks_passed": all(check["returncode"] == 0 for check in checks),
                                      "escalations": escalation})
        return handoff

    def _reviewable(self, task: dict[str, Any], candidate: str, *, require_tip: bool = True) -> dict[str, Any]:
        handoff = task.get("handoff")
        if not handoff or handoff["candidate_commit"] != candidate or handoff["task_revision"] != task["revision"]:
            raise WorkflowError("A current handoff for this exact candidate and task revision is required")
        changed = self.validate_candidate(task, candidate, require_tip=require_tip)
        if self.escalations(task, changed) or handoff.get("escalations"):
            raise WorkflowError("Unresolved scope or architecture escalation")
        if not handoff.get("checks") or any(check["returncode"] != 0 for check in handoff["checks"]):
            raise WorkflowError("All acceptance checks must have actually passed")
        if [check["command"] for check in handoff["checks"]] != task["acceptance"]:
            raise WorkflowError("Handoff acceptance evidence differs from the task brief")
        if handoff.get("unresolved_issues"):
            raise WorkflowError("Resolve handoff issues before approval")
        return handoff

    def review(self, actor: str, task_id: str, candidate: str, summary: str,
               approve: bool = True) -> dict[str, Any]:
        identity = self.authority(actor, "reviewer")
        task = self.task(task_id)
        candidate_commit = self.resolve(candidate)
        if task["status"] == "integrated":
            raise WorkflowError("Integrated task reviews are immutable")
        if not summary.strip():
            raise WorkflowError("An actual review summary is required")
        if self.policy().get("independent_review", True) and actor == task["worker"]:
            raise WorkflowError("An independent online reviewer must review the worker's change")
        if approve:
            self._reviewable(task, candidate_commit)
        elif not task.get("handoff") or task["handoff"]["candidate_commit"] != candidate_commit:
            raise WorkflowError("Review must refer to the handed-off candidate")
        handoff_digest = hashlib.sha256(json.dumps(task["handoff"], sort_keys=True).encode()).hexdigest()
        record = {
            "at": now(), "reviewer": identity, "candidate_commit": candidate_commit,
            "base_commit": task["base_commit"], "target_commit": self.resolve("HEAD"),
            "task_revision": task["revision"], "handoff_sha256": handoff_digest,
            "decision": "approve" if approve else "request-changes", "summary": summary,
        }
        task["reviews"].append(record)
        task["status"] = "approved" if approve else "changes-requested"
        self.save_task(task)
        self.audit("review", actor, {"task_id": task_id, "candidate_commit": candidate_commit,
                                     "decision": record["decision"]})
        return record

    def integrate(self, actor: str, task_id: str) -> dict[str, Any]:
        self.authority(actor, "integrator")
        task = self.task(task_id)
        if task["status"] != "approved" or not task["reviews"]:
            raise WorkflowError("The exact candidate requires online approval before integration")
        review = task["reviews"][-1]
        reviewer = self.identity(review["reviewer"]["id"])
        if reviewer["mode"] != "online" or "reviewer" not in reviewer["roles"]:
            raise WorkflowError("Review authority is no longer valid")
        candidate = review["candidate_commit"]
        handoff = self._reviewable(task, candidate)
        digest = hashlib.sha256(json.dumps(handoff, sort_keys=True).encode()).hexdigest()
        if review["decision"] != "approve" or review["task_revision"] != task["revision"] or digest != review["handoff_sha256"]:
            raise WorkflowError("Review no longer matches the task/handoff")
        if self.git("branch", "--show-current").stdout.strip() != self.policy()["target_branch"]:
            raise WorkflowError("Integration must run from the configured target branch")
        dirty = [path for path in self._dirty_paths(self.root) if not path.startswith("governance/")]
        if dirty:
            raise WorkflowError("Integration checkout has uncommitted code changes: " + ", ".join(dirty))
        target_changes = self.changed_paths(review["target_commit"], self.resolve("HEAD"))
        if any(not path.startswith("governance/") for path in target_changes):
            raise WorkflowError("Integration target code changed since review; obtain a new online review")
        for dependency in task["dependencies"]:
            if self.task(dependency)["status"] != "integrated":
                raise WorkflowError(f"Dependency {dependency} is not integrated")
        merge_preview = self.git("merge-tree", "--write-tree", "HEAD", candidate, check=False)
        if merge_preview.returncode:
            raise WorkflowError("Merge conflicts require an online integrator; checkout was not changed")
        self.git("-c", "user.name=Chat Recall online integrator", "-c",
                 "user.email=online-integrator@chat-recall.invalid", "merge", "--no-ff",
                 "--no-edit", "--no-gpg-sign", candidate)
        task.update(status="integrated", integration_commit=self.resolve("HEAD"),
                    integrated_by=self.identity(actor), integrated_at=now())
        self.save_task(task)
        self.audit("integrated", actor, {"task_id": task_id,
                                         "candidate_commit": candidate,
                                         "integration_commit": task["integration_commit"]})
        return task

    def status(self) -> dict[str, Any]:
        tasks = [_json_read(path) for path in sorted((self.state / "tasks").glob("*.json"))]
        return {"policy": self.policy(), "identities": self.identities(), "tasks": tasks}

    def validate(self) -> dict[str, Any]:
        """Offline CI validation; does not require worker branches/worktrees to exist."""
        identities = self.identities()
        errors = []
        policy = self.policy()
        owner = policy.get("authority_owner")
        if owner is not None and (not isinstance(owner, str) or owner != policy.get("active_coordinator")):
            errors.append("single authority owner must equal the active coordinator")
        coordinator = identities.get(policy.get("active_coordinator", "codex-online"), {})
        if coordinator.get("mode") != "online" or "orchestrator" not in coordinator.get("roles", []):
            errors.append("selected coordinator must be a configured online orchestrator")
        for name, identity in identities.items():
            if identity.get("mode") not in {"online", "local"}:
                errors.append(f"{name}: invalid identity mode")
            if identity.get("mode") == "local" and set(identity.get("roles", [])) & AUTHORITY_ROLES:
                errors.append(f"{name}: local identity has authority roles")
        tasks = self.status()["tasks"]
        for task in tasks:
            try:
                if task["created_by"]["mode"] != "online":
                    raise WorkflowError("task was not created by an online orchestrator")
                if task["status"] in {"approved", "integrated"}:
                    review = task["reviews"][-1]
                    self._reviewable(task, review["candidate_commit"], require_tip=False)
                    if review["reviewer"]["mode"] != "online" or review["decision"] != "approve":
                        raise WorkflowError("approval must come from an online reviewer")
                    if self.policy().get("independent_review", True) and review["reviewer"]["id"] == task["worker"]:
                        raise WorkflowError("reviewer and worker must differ")
                    if review["task_revision"] != task["revision"]:
                        raise WorkflowError("stale task revision in review")
                    digest = hashlib.sha256(json.dumps(task["handoff"], sort_keys=True).encode()).hexdigest()
                    if digest != review["handoff_sha256"]:
                        raise WorkflowError("reviewed evidence was changed")
                    if task["status"] == "integrated":
                        if task["integrated_by"]["mode"] != "online":
                            raise WorkflowError("integration authority must be online")
                        if self.git("merge-base", "--is-ancestor", review["candidate_commit"],
                                    task["integration_commit"], check=False).returncode:
                            raise WorkflowError("integration does not contain reviewed candidate")
            except (WorkflowError, KeyError, IndexError, TypeError) as exc:
                errors.append(f"{task.get('id', '?')}: {exc}")
        if errors:
            raise WorkflowError("Governance validation failed: " + "; ".join(errors))
        return {"ok": True, "identities": len(identities), "tasks": len(tasks),
                "identity_verification": self.policy()["identity_verification"]}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--repo", default=".", help="Canonical repository checkout containing governance records")
    root.add_argument("--actor", help="Explicit configured model identity attesting to a mutating action")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create default versioned policy and online identity if missing")
    commands.add_parser("status", help="Read policy, identities and portable task records")
    commands.add_parser("validate", help="Validate governance records offline (CI)")
    identity = commands.add_parser("identity-add", help="Online orchestrator registers a model/tool/account identity")
    identity.add_argument("id")
    for field in ("provider", "model", "tool", "account"):
        identity.add_argument("--" + field, required=True)
    identity.add_argument("--mode", choices=["online", "local"], required=True)
    identity.add_argument("--role", choices=sorted(ALL_ROLES), action="append", required=True)
    availability = commands.add_parser("availability", help="Attest coordinator availability; existing handoffs remain allowed")
    availability.add_argument("state", choices=["online", "offline"])
    availability.add_argument("--reason", required=True)
    coordinator = commands.add_parser("coordinator-select", help="Select an already registered online fallback after quota/outage")
    coordinator.add_argument("id")
    coordinator.add_argument("--reason", required=True)
    decision = commands.add_parser("decision", help="Online architect records planning/architecture/evaluation decisions")
    decision.add_argument("--kind", choices=sorted(DECISION_KINDS), required=True)
    decision.add_argument("--summary", required=True)
    task = commands.add_parser("task-create", help="Create or adopt a bounded task branch and optional worktree")
    task.add_argument("id")
    for field in ("objective", "branch", "worker"):
        task.add_argument("--" + field, required=True)
    task.add_argument("--base", default="HEAD")
    task.add_argument("--scope", action="append", required=True)
    task.add_argument("--acceptance", action="append", required=True)
    task.add_argument("--dependency", action="append", default=[])
    task.add_argument("--constraint", action="append", default=[])
    task.add_argument("--worktree")
    assign = commands.add_parser("assign", help="Online orchestrator reassigns a task; invalidates previous handoff")
    assign.add_argument("id")
    assign.add_argument("--worker", required=True)
    authorize = commands.add_parser("authorize", help="Online architect approves explicit escalated change paths")
    authorize.add_argument("id")
    authorize.add_argument("--path", action="append", required=True)
    authorize.add_argument("--reason", required=True)
    handoff = commands.add_parser("handoff", help="Assigned worker runs acceptance commands and records exact commit evidence")
    handoff.add_argument("id")
    handoff.add_argument("--candidate")
    handoff.add_argument("--worktree")
    handoff.add_argument("--findings", default="")
    handoff.add_argument("--issue", action="append", default=[])
    handoff.add_argument("--timeout", type=float, default=300)
    review = commands.add_parser("review", help="Independent online model records its review of an exact handoff")
    review.add_argument("id")
    review.add_argument("--candidate", required=True)
    review.add_argument("--summary", required=True)
    review.add_argument("--request-changes", action="store_true")
    integrate = commands.add_parser("integrate", help="Gate and merge an online-approved candidate into target branch")
    integrate.add_argument("id")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    workflow = Workflow(args.repo)
    try:
        if args.command in {"status", "validate"}:
            result = getattr(workflow, args.command)()
        else:
            if not args.actor:
                raise WorkflowError("Mutating actions require explicit --actor identity attestation")
            with workflow.lock():
                if args.command == "init":
                    if args.actor != "codex-online":
                        raise WorkflowError("Bootstrap requires the recorded codex-online authority")
                    result = workflow.initialize()
                elif args.command == "identity-add":
                    result = workflow.register_identity(args.actor, args.id, args.provider, args.model,
                                                        args.mode, args.tool, args.account, args.role)
                elif args.command == "availability":
                    result = workflow.set_availability(args.actor, args.state == "online", args.reason)
                elif args.command == "coordinator-select":
                    result = workflow.select_coordinator(args.actor, args.id, args.reason)
                elif args.command == "decision":
                    result = workflow.record_decision(args.actor, args.kind, args.summary)
                elif args.command == "task-create":
                    result = workflow.create_task(args.actor, args.id, args.objective, args.branch,
                                                  args.base, args.scope, args.acceptance, args.worker,
                                                  args.dependency, args.worktree, args.constraint)
                elif args.command == "assign":
                    result = workflow.assign(args.actor, args.id, args.worker)
                elif args.command == "authorize":
                    result = workflow.authorize_changes(args.actor, args.id, args.path, args.reason)
                elif args.command == "handoff":
                    result = workflow.handoff(args.actor, args.id, args.candidate, args.worktree,
                                              args.findings, args.issue, args.timeout)
                elif args.command == "review":
                    result = workflow.review(args.actor, args.id, args.candidate, args.summary,
                                             not args.request_changes)
                else:
                    result = workflow.integrate(args.actor, args.id)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (WorkflowError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
