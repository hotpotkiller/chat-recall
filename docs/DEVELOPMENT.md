# Development

Use Python 3.11+ with SQLite FTS5, Git, and an existing Node runtime for JavaScript
tests. The default app has no third-party Python dependencies or frontend build.

```sh
make bootstrap
make check
make check-js
make eval
```

Keep development data isolated with `CHAT_RECALL_DATA_DIR=.runtime/demo` and use
synthetic fixtures. Exact search is the default. Do not enable Smart, download
models or start automatic browser importing as a development check.

For a change, reproduce the issue, create a small branch/worktree, implement and
run relevant checks, then review the diff and results before merging. Preserve
source text and branch identity. CI repeats Python, JavaScript and offline
evaluation checks against the public source.

The optional `chat_recall.workflow` CLI is a custom Python-standard-library
coordination layer, not LangGraph, CrewAI or an imported agent framework. The
online model makes planning decisions; the CLI records bounded tasks, executes
acceptance commands, and gates handoffs, reviews and integration. It does not
itself call a model API or run a persistent autonomous orchestrator. Its model
identity labels are operator attestations, not proof of remote inference.
This release includes clean starter configuration; personal operator history and
machine-specific task records are intentionally absent.

Codex is the normal configured coordinator. `Resume Antigravity.command` explicitly
selects the configured online Gemini 3.8 Flash High lead; `Return to Codex.command`
selects the primary again. Only one coordinator is active. Assigned worker launches
use the worker model recorded in their task and do not transfer authority. There
is no automatic quota detector or permission bypass.

An existing archive is not a test fixture. Never add chats, tokens, auth exports,
local databases, model weights or browser profiles to the repository. New public
assets should be synthetic. Public issues should contain sanitized reproductions.
