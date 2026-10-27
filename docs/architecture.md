# Architecture and implementation contract

## ADR 001: first executable version

Decision authority: OpenAI online Codex session, GPT-6 family. The runtime does
not expose the exact variant; do not invent one. Recorded 2026-10-26.

Use a portable Python standard-library service with SQLite FTS5 and a browser UI
for v0.1. This environment has Python 3.14.6/SQLite 3.50.4 but no Node/Rust tools
on the active PATH. The accepted plan's portable engine/provider boundaries are
preserved. A native Tauri wrapper is deferred until the search and ingestion
proofs pass. No code from third-party competitors is incorporated.

## Modules and fixed public interfaces

`chat_recall.search.SearchIndex(db_path)` owns source data and derived search.
It supports `close()`, `import_payload(payload, source_id='chatgpt',
account_id='default')`, `search(query, mode='exact', limit=20, offset=0,
case_sensitive=False, role=None, account_id=None)`,
`context(conversation_id, message_id, source_id='chatgpt', account_id='default')`,
and `status()`. Imports accept an OpenAI conversations.json list, a single
conversation object, or `{'conversations': [...]}`. They also accept the simple
fixture format `{'id','title','messages':[{'id','role','text','parent_id'?}]}`.
An OpenAI mapping node's parent forms the branch; store every visible
user/assistant text node, skip hidden/internal content with reported counts.

Import reports contain `conversations`, `messages`, `skipped`, `errors` (list),
`source_id`, and `account_id`. Idempotent reimports update source messages.
Absence from a partial import never deletes existing records. Namespace all
identifiers by source and account. Keep original text untouched.

Search response: `{'query','mode','total','results': [...], 'semantic': {...}}`.
Each result contains `conversation_id`, `message_id`, `source_id`, `account_id`,
`title`, `role`, `text`, `snippet`, `matches` (list of `[start,end]` Python
Unicode code-point offsets into original text), `match_count`, `score`,
`match_type` ('exact'/'lexical'/'semantic'/'hybrid'), `url`, `created_at`,
and additive `focus_offset` (bounded original Unicode codepoint offset). Literal
evidence takes precedence; otherwise the best semantic chunk supplies the focus
used by excerpts and cloud source windows.
Exact mode has exhaustive message recall with pagination; all occurrences in
each message are counted, including overlaps. Case-insensitive matching uses
Unicode casefold with an offset map. Empty queries return zero results.

Smart mode unions BM25 and optional stored dialogue/summary embeddings with RRF
(k=60). Semantic absence is explicit; lexical fallback is never labelled semantic.
Exact mode never searches generated text. Original dialogue windows preserve
branch ancestry. An optional Ollama provider may enrich changed messages and
topics asynchronously; fake-provider tests do not establish real-model quality.

Context response: `{'conversation_id','message_id','source_id','account_id',
'title','url','messages': [{'id','role','text','parent_id','created_at'}]}`.
Return the selected message's ancestor chain and at most one deterministic
descendant path, never interleave sibling branches. Unknown targets raise
`KeyError`. Status contains `conversations`, `messages`, `sources` and optional
semantic indexing state.

`chat_recall.server` serves the UI and JSON on loopback. Endpoints:
GET `/api/status`; GET `/api/search?q=&mode=exact&limit=20&offset=0&case_sensitive=0&role=&account_id=`;
GET `/api/context?conversation_id=&message_id=&source_id=chatgpt&account_id=default`;
POST `/api/import` with JSON `{'payload': ..., 'source_id':..., 'account_id':...}`.
Use a per-install bearer token for imports; do not accept unauthenticated writes.
GET `/api/config` is same-origin UI only and supplies the UI token. Validate
Host/Origin; restrict CORS to configured extension origins. UI uses textContent
for imported text and converts code-point offsets before highlighting in JS.
Health checks contain no private text. Default port 8765, database path supplied
by CLI/environment; demo data is explicitly synthetic.

Browser helper uses the same import API with a user-paired token. Reads through
the user's signed-in ChatGPT web session; no credentials leave the browser.
Import/export-file support is stable; the account API adapter is experimental,
checkpointed, bounded, and must never imply full/native-app coverage without
live verification. No ChatGPT write/delete endpoints.

## ADR 002: online-led development

The workflow CLI configures online identities and policy, records decisions,
creates task briefs and worktrees, records worker handoffs and online reviews,
and gates integration on an approved review of the exact candidate commit.
The deployed `authority_owner=codex-online` pins all authority to one online
Codex orchestrator, with GPT-6.1 Sol requested by the user (exact current runtime
variant is not exposed). Antigravity implements bounded tasks first; Codex
considers local E2B after Antigravity quota exhaustion/unavailability. Historical
backup identities are blocked from authority by this lock. No takeover occurs.
Policy state is versioned in the repository. Availability loss blocks new
planning, assignment, reviews, and integration; existing workers may finish and
handoff. Tests use explicit synthetic identities. A locally edited JSON record
cannot prove remote inference, so repository permissions and CI are the trust
boundary. No automatic cross-account messaging is assumed.

## Release gates

Default checks are offline, synthetic, deterministic. Exact search is compared
with a brute-force oracle; smart lexical baseline and optional semantic results
are reported separately. Live account sync, native-app refresh, browser-closed
operation, model relevance and A18 Pro throughput require separate evidence.
