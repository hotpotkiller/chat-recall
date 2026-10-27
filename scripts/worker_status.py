"""Output the deployed orchestrator and worker routing configuration as JSON."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, NoReturn


def fail(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(2)


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        fail(f"missing required file: {path.name}")
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path.name}: {exc}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    policy_path = repo_root / "governance" / "policy.json"
    identities_path = repo_root / "governance" / "identities.json"

    policy = load_json(policy_path)
    if not isinstance(policy, dict):
        fail("policy.json must contain a JSON object")

    identities = load_json(identities_path)
    if not isinstance(identities, dict):
        fail("identities.json must contain a JSON object")

    active_coordinator = policy.get("active_coordinator")
    if not isinstance(active_coordinator, str) or not active_coordinator.strip():
        fail("active_coordinator must be a nonempty string")
    if active_coordinator not in identities:
        fail(f"unknown active_coordinator identity: {active_coordinator}")

    coord_record = identities[active_coordinator]
    if not isinstance(coord_record, dict):
        fail(f"invalid identity record for {active_coordinator}")
    if coord_record.get("mode") != "online":
        fail(f"coordinator {active_coordinator} must be online")
    coord_roles = coord_record.get("roles")
    if not isinstance(coord_roles, list) or "orchestrator" not in coord_roles:
        fail(f"coordinator {active_coordinator} must have orchestrator role")

    authority_owner = policy.get("authority_owner")
    if not isinstance(authority_owner, str) or not authority_owner.strip():
        fail("authority_owner must be a nonempty string")
    if authority_owner not in identities:
        fail(f"unknown authority_owner identity: {authority_owner}")

    requested_model = policy.get("requested_orchestrator_model")
    if not isinstance(requested_model, str) or not requested_model.strip():
        fail("requested_orchestrator_model must be a nonempty string")

    coordinator_available = policy.get("coordinator_available")
    if type(coordinator_available) is not bool:
        fail("coordinator_available must be a boolean")

    coordinator_takeover = policy.get("coordinator_takeover")
    if type(coordinator_takeover) is not bool:
        fail("coordinator_takeover must be a boolean")

    worker_priority = policy.get("worker_priority")
    if not isinstance(worker_priority, list) or not worker_priority:
        fail("worker_priority must be a nonempty list")
    for wid in worker_priority:
        if not isinstance(wid, str) or not wid.strip():
            fail("worker_priority entries must be nonempty strings")
    if len(worker_priority) != len(set(worker_priority)):
        fail("worker_priority IDs must be unique")

    worker_modes: list[str] = []
    for wid in worker_priority:
        if wid not in identities:
            fail(f"missing identity for worker: {wid}")
        worker_record = identities[wid]
        if not isinstance(worker_record, dict):
            fail(f"invalid identity record for worker: {wid}")
        mode = worker_record.get("mode")
        if not isinstance(mode, str) or mode not in {"online", "local"}:
            fail(f"worker {wid} mode must be 'online' or 'local'")
        roles = worker_record.get("roles")
        if not isinstance(roles, list) or "worker" not in roles:
            fail(f"worker {wid} must have 'worker' role")
        worker_modes.append(mode)

    single_orchestrator = (authority_owner == active_coordinator) and (coordinator_takeover is False)

    status = {
        "orchestrator": active_coordinator,
        "requested_model": requested_model,
        "available": coordinator_available,
        "single_orchestrator": single_orchestrator,
        "worker_order": worker_priority,
        "worker_modes": worker_modes,
    }

    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
