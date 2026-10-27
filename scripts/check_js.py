"""Run offline JavaScript syntax and behavior checks with an existing Node binary.

Use ``make check-js NODE=/absolute/path/to/node`` to select a runtime explicitly.
This command does not install packages, download models, or access real accounts.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
TESTS = (
    ROOT / "browser-extension" / "test_adapter.mjs",
    ROOT / "browser-extension" / "test_web.mjs",
    ROOT / "browser-extension" / "test_smart_dialogue.mjs",
)


def main() -> int:
    requested = os.environ.get("NODE", "").strip()
    node = shutil.which(os.path.expanduser(requested)) if requested else shutil.which("node")
    if node is None:
        detail = f"The NODE override {requested!r} is not executable. " if requested else ""
        print(
            detail + "Node.js is required for make check-js. Install Node.js 24 LTS "
            "or set NODE=/absolute/path/to/node. No runtime was downloaded.",
            file=sys.stderr,
        )
        return 2

    scripts = sorted(
        path
        for directory in (ROOT / "chat_recall" / "web", ROOT / "browser-extension")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".js", ".mjs"}
    )
    if not scripts or any(not path.is_file() for path in TESTS):
        print("JavaScript source or required behavior tests are missing.", file=sys.stderr)
        return 1

    try:
        version = subprocess.run(
            [node, "--version"], check=True, capture_output=True, text=True,
            cwd=ROOT, timeout=10,
        ).stdout.strip()
        print(f"Checking {len(scripts)} JavaScript files with Node {version}", flush=True)
        for path in scripts:
            subprocess.run([node, "--check", str(path)], check=True, cwd=ROOT, timeout=30)
        for path in TESTS:
            subprocess.run([node, str(path)], check=True, cwd=ROOT, timeout=30)
    except subprocess.CalledProcessError as exc:
        print(f"JavaScript check failed with exit code {exc.returncode}.", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("JavaScript check exceeded its bounded execution time.", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Could not execute the selected Node runtime: {exc}", file=sys.stderr)
        return 1

    print(f"JavaScript syntax and {len(TESTS)} offline behavior suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
