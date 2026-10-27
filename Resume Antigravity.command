#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ -z "${NODE:-}" ]] && ! command -v node >/dev/null 2>&1; then
  recall_node="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
  if [[ -x "$recall_node" ]]; then
    export NODE="$recall_node"
  fi
fi
exec python3 scripts/launch_agy_worker.py --take-over "$@"
