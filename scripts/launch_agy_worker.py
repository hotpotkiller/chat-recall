"""Open the configured online AGY lead, or an explicitly assigned AGY worker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chat_recall.workflow import Workflow, WorkflowError


def launch_spec(workflow, task_id=None, take_over=False, dry_run=False):
    policy = workflow.policy()
    coordinator = policy.get("active_coordinator")
    owner = policy.get("authority_owner")
    if coordinator != owner:
        raise ValueError("The single authority owner must match the active coordinator")

    canonical = workflow.root
    common = (
        f"Canonical repository: {canonical}. Read its AGENTS.md and current governance policy; "
        "Codex is normal primary coordinator, with explicit optional AGY High handover when requested. "
        "Keep Smart disabled and importing stopped. Use synthetic fixtures only; no private histories, "
        "credentials, real embeddings or model downloads. Focus functions and reproduced bugs, "
        "with targeted checks rather than repeated imports. "
    )

    if task_id is None:
        # Lead mode
        fallback = policy.get("fallback_coordinator", "agy-lead-high-v1")
        handover_required = False
        target_coordinator = coordinator

        lead = workflow.authority(coordinator, "orchestrator", available=False)
        is_agy_lead = "antigravity" in lead.get("tool", "").lower()

        if take_over:
            if coordinator != fallback:
                handover_required = True
                target_coordinator = fallback

                # Validate planned target identity and policy constraints BEFORE mutating owner or checking executable
                target_identity = workflow.identity(target_coordinator)
                authorized = policy.get("authorized_coordinators")
                if authorized is not None:
                    if target_coordinator not in authorized:
                        raise ValueError(f"Target coordinator {target_coordinator!r} is not in authorized_coordinators")
                else:
                    if owner is not None and target_coordinator != owner:
                        raise ValueError("Single orchestrator policy forbids coordinator takeover")
                if "antigravity" not in target_identity.get("tool", "").lower():
                    raise ValueError(f"Planned coordinator {target_coordinator!r} is not an Antigravity tool")
                if target_identity.get("mode") != "online":
                    raise ValueError(f"Planned coordinator {target_coordinator!r} must be an online model")
                required_roles = {"architect", "orchestrator", "reviewer", "integrator"}
                if not required_roles.issubset(set(target_identity.get("roles", []))):
                    raise ValueError(f"Planned coordinator {target_coordinator!r} lacks required authority roles")

                planned_model = target_identity["model"]

                if dry_run:
                    return {
                        "checkout": canonical,
                        "actor": target_coordinator,
                        "role": "lead",
                        "model": planned_model,
                        "planned_coordinator": target_coordinator,
                        "planned_model": planned_model,
                        "handover_required": handover_required,
                        "orchestrator": target_coordinator,
                    }
                else:
                    # Check AGY executable exists BEFORE actual authority selection so missing CLI leaves Codex owner unchanged
                    executable = shutil.which("agy")
                    if not executable:
                        raise ValueError("Antigravity CLI is unavailable; report the blocker without a local-model fallback")

                    workflow.select_coordinator(
                        actor=coordinator,
                        identity_id=fallback,
                        reason="operator-selected launch: take-over fallback coordinator",
                    )
                    policy = workflow.policy()
                    coordinator = policy.get("active_coordinator")
                    lead = workflow.authority(coordinator, "orchestrator", available=False)
        else:
            if not is_agy_lead:
                raise ValueError("Active coordinator is Codex. Pass --take-over to switch to AGY fallback lead, or run with a task ID for worker mode.")

        if dry_run:
            target_identity = workflow.identity(target_coordinator)
            planned_model = target_identity["model"]
            return {
                "checkout": canonical,
                "actor": target_coordinator,
                "role": "lead",
                "model": planned_model,
                "planned_coordinator": target_coordinator,
                "planned_model": planned_model,
                "handover_required": handover_required,
                "orchestrator": target_coordinator,
            }

        active_lead = workflow.authority(coordinator, "orchestrator")
        for role in ("architect", "reviewer", "integrator"):
            workflow.authority(coordinator, role)
        prompt = (
            f"You are {coordinator}, using configured cloud model {active_lead['model']}. "
            "Codex is normal primary coordinator, with explicit optional AGY High handover. " + common +
            "Read AGENTS.md, docs/AGY-HANDOFF.md and current policy/task records. Start with "
            "an assigned task, or identify one useful function or reproducible bug; "
            "preserve any existing work. Use a separate AGY worker context and its recorded worker "
            "model for implementation; do not implement and approve under the same identity. Use "
            f"chat_recall.workflow with --repo {canonical} and --actor {coordinator} for authority "
            "actions. Inspect the diff, actual acceptance results and exact candidate before "
            "review/integration. Commit review records before integration. Then continue useful "
            "bounded function/bug work, one task at a time. Report concrete changes, checks and "
            "blockers concisely. Normal CLI permissions remain enabled. Do not publish to GitHub "
            "without the user's publication instruction. Do not manufacture work or run an endless model loop."
        )
        return {
            "checkout": canonical,
            "actor": coordinator,
            "role": "lead",
            "model": active_lead["model"],
            "prompt": prompt,
            "orchestrator": coordinator,
            "handover_required": handover_required,
        }

    # Worker mode (task_id provided)
    # Assigned online AGY workers may run under either current allowed coordinator.
    task = workflow.task(task_id)
    worker_id = task["worker"]
    identity = workflow.identity(worker_id)
    if task["status"] != "assigned" or "worker" not in identity.get("roles", []):
        raise ValueError("Choose an assigned worker task")
    if "antigravity" not in identity.get("tool", "").lower() or identity["mode"] != "online":
        raise ValueError("This launcher requires a configured online AGY worker")
    hint = task.get("worktree_hint")
    if not hint:
        raise ValueError("Assigned task needs an isolated worktree")
    checkout = (canonical / hint).resolve()
    if not checkout.is_dir() or checkout == canonical:
        raise ValueError("Assigned isolated worktree is unavailable")
    branch = workflow.git("branch", "--show-current", cwd=checkout).stdout.strip()
    if branch != task["branch"]:
        raise ValueError("Worktree branch differs from the task brief; report to the online AGY lead")

    if dry_run:
        return {
            "checkout": checkout,
            "actor": worker_id,
            "role": "worker",
            "model": identity["model"],
            "planned_coordinator": coordinator,
            "planned_model": identity["model"],
            "handover_required": False,
            "orchestrator": coordinator,
            "task": task_id,
            "branch": branch,
        }

    prompt = (
        f"You are {worker_id}, implementation worker using cloud model {identity['model']}, "
        f"under online lead {coordinator}. " + common +
        "Work only in the assigned scope/worktree. Escalate interface/dependency/requirements "
        "changes to the AGY lead. Do not assign, approve, integrate or change governance. "
        "Run the task checks, commit and submit actual results; never fabricate tests. Task:\n" +
        json.dumps(task, ensure_ascii=False)
    )
    return {
        "checkout": checkout,
        "actor": worker_id,
        "role": "worker",
        "model": identity["model"],
        "prompt": prompt,
        "orchestrator": coordinator,
        "task": task_id,
        "branch": branch,
        "handover_required": False,
    }


def return_to_codex(workflow, dry_run=False):
    policy = workflow.policy()
    coordinator = policy.get("active_coordinator")
    primary = policy.get("primary_coordinator", "codex-online")
    handover_required = (coordinator != primary)
    primary_identity = workflow.identity(primary)
    planned_model = primary_identity["model"]

    if dry_run:
        return {
            "role": "lead",
            "actor": primary,
            "planned_coordinator": primary,
            "planned_model": planned_model,
            "handover_required": handover_required,
            "orchestrator": primary,
        }

    if handover_required:
        workflow.select_coordinator(
            actor=coordinator,
            identity_id=primary,
            reason="operator-selected launch: return to Codex primary",
        )
    print(f"Returned authority to primary coordinator {primary}. Exiting without AGY inference.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", help="Omit to open the independent online lead")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--take-over", action="store_true", help="Explicitly select configured fallback coordinator")
    parser.add_argument("--return-to-codex", action="store_true", help="Select primary Codex coordinator and exit without AGY inference")
    args = parser.parse_args(argv)

    workflow = Workflow(ROOT)

    if args.return_to_codex:
        result = return_to_codex(workflow, dry_run=args.dry_run)
        if args.dry_run:
            print(json.dumps({key: str(value) if isinstance(value, Path) else value
                              for key, value in result.items()}))
            return 0
        return result

    spec = launch_spec(workflow, args.task, take_over=args.take_over, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value
                          for key, value in spec.items() if key != "prompt"}))
        return 0
    executable = shutil.which("agy")
    if not executable:
        raise ValueError("Antigravity CLI is unavailable; report the blocker without a local-model fallback")
    os.chdir(spec["checkout"])
    os.execv(executable, [executable, "--model", spec["model"], "--mode", "accept-edits",
                         "--sandbox", "--prompt-interactive", spec["prompt"]])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, EOFError, WorkflowError) as exc:
        print(f"Antigravity: {exc}", file=sys.stderr)
        raise SystemExit(2)
