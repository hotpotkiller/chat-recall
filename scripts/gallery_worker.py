"""One-shot local worker using an existing Gallery model; never an orchestrator.

Reads a bounded prompt from stdin and prints a proposal only. It does not give
the model shell/filesystem tools or apply its output. An online reviewer owns
any resulting changes. LiteRT-LM is an optional, separately installed tool.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_PROPOSAL_BYTES = 128 * 1024
MAX_DIAGNOSTIC_BYTES = 64 * 1024


def proposal_parts(output):
    """Validate a destination without creating anything or following symlinks."""
    project = PROJECT_ROOT.resolve()
    destination = Path(os.path.abspath(output.expanduser()))
    try:
        relative = destination.relative_to(project / ".runtime" / "proposals")
    except ValueError as exc:
        raise ValueError("Proposal output must be a new file below this checkout's .runtime/proposals/") from exc
    if not relative.parts:
        raise ValueError("Select a new proposal file below .runtime/proposals/")
    current = project
    for part in (".runtime", "proposals", *relative.parts):
        current /= part
        if current.is_symlink():
            raise ValueError("Proposal output may not traverse a symlink")
    if destination.exists():
        raise ValueError("Proposal output already exists; choose a new filename")
    return (".runtime", "proposals", *relative.parts)


def write_proposal(output, content):
    """Exclusive descriptor-relative creation prevents overwrite/symlink races."""
    parts = proposal_parts(output)
    if os.name != "posix":
        raise ValueError("Bounded Gallery proposal writes require macOS or Linux")
    directory = os.open(PROJECT_ROOT.resolve(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
    finally:
        os.close(directory)


def run_bounded(command, timeout):
    """Bound both output streams and the lifetime of our own process group."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Local task timeout must be finite and positive")
    if os.name != "posix":
        raise ValueError("Bounded Gallery inference requires macOS or Linux")
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    caps = {"stdout": MAX_PROPOSAL_BYTES, "stderr": MAX_DIAGNOSTIC_BYTES}
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("Local task timed out; no code was applied")
                for key, _ in selector.select(min(remaining, 0.1)):
                    try:
                        chunk = os.read(key.fd, 8192)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    name = key.data
                    if len(captured[name]) + len(chunk) > caps[name]:
                        raise ValueError(f"Local task exceeded the {name} byte limit; no proposal was saved")
                    captured[name].extend(chunk)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("Local task timed out; no code was applied")
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise ValueError("Local task timed out; no code was applied") from exc
        return subprocess.CompletedProcess(command, returncode,
                                           bytes(captured["stdout"]), bytes(captured["stderr"]))
    finally:
        # Always retire this one-shot worker's group, including a wrapper's child
        # that inherited pipes or outlived its parent. Other sessions are untouched.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()
        process.stderr.close()


def discover_model():
    root = Path.home() / "Library/Application Support/com.google.AIEdgeGallery/Documents"
    matches = sorted(root.glob("Gemma_4_E2B_it/*/*.litertlm"))
    if not matches:
        raise ValueError("No downloaded Gemma 4 E2B Gallery file found; pass --model-path")
    return matches[-1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--runtime", default=os.environ.get("CHAT_RECALL_LITERT", "litert-lm"))
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--config", type=Path, default=Path(".runtime/litert-worker.json"))
    parser.add_argument("--output", type=Path,
                        help="Save a new file below this checkout's ignored .runtime/proposals/; never overwrite or apply code")
    parser.add_argument("--dry-run", action="store_true", help="Print non-sensitive configuration without inference")
    args = parser.parse_args(argv)
    if not 256 <= args.context <= 4096 or not 1 <= args.timeout <= 600:
        raise ValueError("Use context 256–4096 and timeout 1–600 seconds")
    model = (args.model_path or discover_model()).expanduser().resolve()
    if not model.is_file() or model.suffix != ".litertlm":
        raise ValueError("Select an existing .litertlm file")
    runtime = shutil.which(args.runtime)
    if not runtime:
        raise ValueError("Install Google's optional litert-lm==0.18.1 CLI, or pass --runtime /path/to/litert-lm")
    if args.dry_run:
        print(json.dumps({"role": "bounded-local-worker", "model_file": model.name,
                          "model_bytes": model.stat().st_size, "backend": args.backend,
                          "context_tokens": args.context, "downloads_model": False,
                          "authority_roles": False}))
        return 0
    if args.output:
        proposal_parts(args.output)
    prompt = sys.stdin.read(24001)
    if not prompt.strip() or len(prompt) > 24000:
        raise ValueError("Supply a bounded task prompt on stdin (1–24000 characters)")
    args.config.parent.mkdir(parents=True, exist_ok=True)
    if not args.config.exists():
        args.config.write_text("{}\n", "utf-8")
    instruction = (
        "You are a bounded local implementation worker. Follow only the supplied task. "
        "Do not change requirements, architecture, dependencies, or interfaces. "
        "Return a concise proposal/code for online review; you have no tool access.\n\n"
    )
    command = [runtime, "--config", str(args.config.resolve()), "run", str(model),
               "--backend", args.backend, "--cache", "no", "--max-num-tokens", str(args.context),
               "--thinking", "false", "--temperature", "0", "--prompt", instruction + prompt]
    result = run_bounded(command, args.timeout)
    if result.returncode:
        # Tool diagnostics are local and may include source paths; keep them out
        # of public governance evidence unless an operator explicitly reviews it.
        print(result.stderr[-4000:].decode("utf-8", errors="replace"), file=sys.stderr)
        raise ValueError("LiteRT-LM failed; no code was applied")
    if args.output:
        write_proposal(args.output, result.stdout)
    print(result.stdout.decode("utf-8", errors="replace"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Gallery worker: {exc}", file=sys.stderr)
        raise SystemExit(2)
