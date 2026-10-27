# Contributor instructions

Ordinary ChatGPT conversation search is the product. Prefer focused function
improvements and reproduced bugs. Keep original text, source identity, Unicode
offsets and conversation branches authoritative.

Smart/model processing and automatic importing stay off unless the user explicitly
requests them. Use synthetic fixtures for development. Do not inspect personal
histories or credentials, run real model evaluations or download models as checks.

Use Python 3.11+ standard library, SQLite FTS5 and plain HTML/CSS/JavaScript. No new
dependencies, provider/API contracts or database migrations without online design
review. Imported HTML must remain inert. Keep secrets and runtime files out of Git.

The active configured online coordinator plans, assigns, reviews and integrates.
Codex is the normal primary. Explicit AGY continuation uses online
`gemini-3.8-flash-high`; bounded implementation uses online Flash Low. Keep one
active coordinator, separate worker/reviewer identities and one worktree per
concurrent task. No automatic quota detection or local-model fallback.

Run checks appropriate to the changed code: `make check`, `make check-js`,
`make test-integration`, `make eval`. Do not repeat unrelated full evaluations.
Record actual commands/results and review the exact candidate before integration.
Do not claim live account coverage or semantic relevance from synthetic tests.

Identity/provider labels are configuration attestations, not cryptographic proof
of where inference ran. Normal CLI permission controls stay enabled.
