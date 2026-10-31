# Smart search design and evaluation

Exact words answers: "Where did I say these words?" It verifies original text,
returns every matching message across pages and counts overlapping occurrences.
It requires no model and remains the default.

Smart answers: "Where did we discuss this idea, even with different wording?"
It ranks original passages using literal, BM25 keyword and optional dialogue/topic
embedding channels, combined with reciprocal rank fusion (k=60). It is bounded
retrieval, not an exhaustive occurrence count or a generated answer.

## Current development boundary

Codex is the online architect/reviewer/integrator; assigned online AGY workers
implement bounded tasks. Smart and importing are off by default. Activation needs
measured quality and memory evidence shown to the user. An isolated,
synthetic evaluation may use an already installed embedding model explicitly;
default checks never load a model, download weights or read personal chats.

## Acceptance contracts

- Qwen 0.6B uses cosine threshold 0.50, retained after calibration-only testing.
  This remains provisional: the final disjoint holdout failed quality targets.
  Other providers retain an unvalidated0.30 heuristic. Scores are ranking signals,
  never confidence. Zero/negative evidence cannot become relevance through RRF.
- Query and document encoding pipelines are separated:
  - Document chunks are embedded directly without instruction prefixes using `DOCUMENT_PROFILE`.
  - Queries for Qwen models are prepended with:
    `Instruct: Given a description of a past conversation, retrieve the relevant original dialogue passages.\nQuery: `
  - The 960 UTF-8 byte limit is enforced AFTER prepending the prefix, rejecting overbudget queries before HTTP transmission.
- Model and document pipeline identity is version-immutable:
  - `resolve_identity(document_profile) -> str` computes a stable `index_id` fingerprint over the installed model digest (from metadata-only `/api/tags`), model alias, expected output dimension (1024 for Qwen 0.6b), `SearchIndex.DOCUMENT_PROFILE`, normalization (`l2`), and context/maxbytes (`num_ctx: 1024`, `max_input_bytes: 960`).
  - `query_template_version` and `calibration_version` are tracked separately; modifying only the query template does not invalidate or change document index identity, preserving indexed document vectors. Both versions are exposed in search and evaluation metadata separately from document index identity without implying calibration has occurred.
  - Mutable tag protection: model digest is verified before and after each real embedding request against the resolved digest; any tag change during inference immediately rejects results rather than mixing incompatible vectors. Operations from SELECT to inference are serialized under a provider operation lock.
  - Offline contract: provider constructor, status, and Exact search never contact Ollama; status reads cached identity only.
  - Legacy alias-only vectors are preserved in `vectors.model_id` but excluded from versioned retrieval; honest index and pending coverage are exposed via separate `display_model_id`.
- SQL applies account/speaker/model/current-content filters before retrieval.
  Stream vector columns without derived text, retain at most 100 unique messages
  per semantic channel, and fetch full message bodies only for the requested page.
  Ranking CPU remains linear in stored vectors; this is not an ANN index.
- Query-provider failures discard all semantic evidence and retain explicitly
  labelled keyword fallback. No configured/current vectors means no embedding
  call. Exact search never calls a provider.
- Meaning evidence candidate filtering policy (`meaning-evidence-v1`):
  - Clearly differentiates Exact exhaustive words (answering "where did I say these exact words?", covering imported original words across all messages), Smart indexed concepts (evaluated against current compatible vectors), and fallback.
  - Candidate filtering applies strictly and only when `semantic.used` is True (query actually embedded and current compatible vectors evaluated successfully).
  - Before reciprocal rank fusion (RRF) and pagination, keyword-only message candidates unsupported by retained dialogue or topic semantic channels are removed, unless they are verified full-query literal candidates.
  - Verified full-query literal candidates always remain; actual semantic candidates retain their channels.
  - The keyword rank list is recomputed over accepted candidates, followed by standard reciprocal rank fusion (`k=60`) ranking.
  - If the provider is absent, unavailable, errored, vectors unavailable or query overbudget (`semantic.used` is False), truthful lexical fallback is preserved without candidate filtering.
  - Partial semantic coverage never claims archive exhaustiveness; unsupported unindexed keyword suggestions are excluded from meaning results while Exact still covers imported original words. Existing coverage metadata (`indexed_messages`, `pending_messages`, `status`) remains truthful.
  - Exposes additive `semantic.relevance_policy` with stable identifier `meaning-evidence-v1` and integer `semantic.rejected_keyword_only_count`.
  - Configured threshold (0.50 for Qwen 0.6b, 0.30 heuristic for others) remains unchanged without automatic calibration version or default mutations. Similarity and fusion are not confidence. No claim of real semantic quality passes until root measures.
- Return best-chunk similarity and original-text offsets separately from verified
  literal matches. The best chunk across channels determines semantic focus.
  Never present cosine or fused rank as confidence or highlight invented wording.
- Smart cards explain their retrieval channel and show bounded nearby dialogue
  from the same source/account/branch. A generic reply such as "yes, do that"
  must not hide the preceding exchange that explains its retrieval.
- Exact results open at the first valid literal mark, including deep messages
  with astral Unicode. Semantic results keep their original passage anchor.
- Ollama embedding requests ask for immediate unloading with keep_alive=0.
  This trades repeat-query latency for lower idle model residency. A model still
  uses memory while executing; Python allocation measurements do not include it.
  See [Ollama lifecycle documentation](https://docs.ollama.com/faq#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately).
- The local profile uses context 1024 and batch 64. Inputs are capped at 960
  UTF-8 bytes; original source windows use at most 512 bytes plus bounded prior
  dialogue. Longer descriptions fall back with an explicit shorter-input error.
  Existing source text and full dialogue viewing are preserved.
- Multilingual and CJK lexical channel:
  - Smart search provides a standard-library CJK lexical channel using an evaluated character-ngram strategy (>=2-character CJK concepts, character bigrams and trigrams, English and mixed identifiers retained).
  - Derived rebuildable FTS5 index table `message_cjk` and segment mapping table `message_cjk_segments` with atomic, crash-safe migration marker `index_meta(message_cjk_ready)`; synchronized on message inserts, updates, and explicit hidden revocations without rewriting original messages or vectors.
  - Bounded per-message streaming via SQLite `substr(CAST(text AS BLOB), ...)` byte chunks (4096 bytes) with incremental UTF-8 boundary trimming and tail carryover, preserving embedded NULs, complete Unicode, and boundary ngrams/identifiers without $O(N^2)$ character rescanning; segment scores are computed in a valid FTS5 context via a ranked CTE with segment-to-message joins and role/account predicates pushed inside the CTE before an optimization barrier (`LIMIT -1 OFFSET 0` / `MATERIALIZED`), deduplicated and bounded to at most 100 distinct filtered messages before ranking.
  - Cross-chunk mixed-language identifier seam support: queries containing non-CJK identifiers/terms search both `message_cjk` and `message_fts` (which indexes intact documents across chunk boundaries) under identical role and account filters, prioritizing double-channel hits and deduplicating up to 100 candidates.
  - Declared multi-character function phrases (>=2 characters) are stripped longest-first with space boundaries before CJK ngram generation, preventing cross-boundary ngram leaks while preserving real words containing 和/与/及 (e.g. 和平, 共和国, 参与) and semantically meaningful negation (没有, 不是).
  - Avoids indiscriminate singleton Chinese-character OR queries that would flood retrieval; filters noise-only stop grams.
  - Scopes account and role before retrieval, caps unique candidates at 100, and labels keyword evidence truthfully under the `keyword` retrieval channel.
- Faithful paragraph/sentence/code-line-aware dialogue windows:
  - Replaces arbitrary byte windows with faithful paragraph/sentence/code-line-aware spans.
  - Prefers whole paragraphs <= 512 UTF-8 bytes; larger paragraphs are split at sentence, newline, or code boundaries without exceeding cap, with safe Unicode fallback.
  - Paragraph separators and whitespace gaps are strictly bounded (spans <= 512 bytes) ensuring 100% contiguous coverage without borrow-induced byte expansion; `_windows()` skips whitespace-only spans when creating model inputs so no whitespace-only windows are emitted.
  - Preserves all original characters/coverage (no gaps), strictly positive progress, and exact original codepoint offsets into untouched source text.
  - Adjacent paragraph target windows never point to an irrelevant preceding paragraph when the relevant paragraph can be a separate window.
  - Branch-correct previous dialogue context <= 384 UTF-8 bytes prefers a coherent original question excerpt rather than arbitrary chopped tail bytes; no generated summary.
  - Entire final document is guaranteed <= 960 UTF-8 bytes.
  - `SearchIndex.DOCUMENT_PROFILE` version is `v2`, ensuring old and new windows never share identity, while preserving legacy generic fake-provider window behavior when unversioned.
- Explicit bounded meaning-index batches and coverage controls (v3):
  - Explicit bounded batch contract: semantic enrichment (`POST /api/enrich`) runs exactly one explicit batch (`limit <= 1000`), never an automatic loop, background daemon, or periodic schedule. Passing an account scope (`account_id`) bounds processing to that account.
  - SQL anti-join selection before body materialization: pending metadata rows (`id`, namespace, `content_hash`, `length(CAST(text AS BLOB))`) are selected via a SQL anti-join on active `model_id` (`resolved_id`), matching `content_hash`, and topic requirement, ordered by `m.id LIMIT ?`. The metadata SELECT cursor completes before inference to avoid long-lived read locks blocking concurrent source writers.
  - Strict resource budgets:
    - `MAX_ENRICH_SOURCE_BYTES = 262144` (256 KiB): messages exceeding this limit are skipped before loading the text body, recorded as `skipped_budget`, and cursor advances past them to avoid starvation while keeping original text and old vectors intact and pending.
    - `MAX_ENRICH_WINDOWS = 512`: messages generating more than 512 derived windows (or topic equivalents) are rejected before the first inference call, recorded as `skipped_budget`, and cursor advances past them.
    - Batch time and request limits: defaults 120-second soft deadline and 128 external model requests per explicit batch, checked before each request and immediately after `provider.operation_lock` acquisition.
  - Ancestry without `context()`: window derivation queries at most two direct namespace-scoped ancestors using byte-bounded prefixes (`substr(text, 1, 960)`), preserving exact document semantics without loading full conversations or crossing sibling branches.
  - Safe pause/resume lifecycle:
    - Server Event drives states `running -> pausing -> paused`. No concurrent or overlapping worker can start while `running` or `pausing`.
    - Authenticated `POST /api/enrich/pause` requires a matching random `job_id`, rejecting stale targets without touching active jobs.
    - In-flight HTTP requests finish cleanly; no subsequent provider requests are initiated. Partial vectors for an interrupted message are discarded, leaving the message pending. Terminal state is published only after database cleanup.
  - Bounded structured cursor:
    - Structured cursor metadata: `{after_id: nonnegative int, model_id: resolved profile, account_id: string|null, generate_topics: bool}`.
    - Mismatched scope, profile, or topic flags reject before inference; malformed cursors reject immediately.
    - `next_cursor` is returned only while later pending rows exist; if a pass finishes with remaining pending rows (e.g. failed or oversized), `pass_exhausted: true` is reported and the UI shows "Some passages need attention" with "Retry remaining".
  - Scoped status coverage:
    - `SearchIndex.status(account_id=None)` and `GET /api/status?account_id=` report scoped `semantic.coverage` (`eligible`, `current`, `pending`, `legacy`) for the requested account using offline cached identity only, without network calls or false alias counting.
    - `/api/config` advertises `enrich_controls_version: 1` before UI renders pause and resume controls.

## Quality and memory gates

Use a synthetic, versioned evaluation with paraphrases, ambiguous near-neighbours,
unrelated queries and multilingual cases. Report Exact and Smart side by side,
actual semantic use, target recall at 3, reciprocal rank, unrelated-query false
positives, source-context correctness, query latency and the model identity.
Initial activation targets: target recall at 3 >= 0.85, MRR at 10 >= 0.8,
unrelated-query false-positive rate <= 0.2, and no source/branch correctness errors.
These small synthetic gates do not establish personal-history relevance.

Measure retrieval allocations on a fixed synthetic corpus separately from model
residency. Use tracemalloc for Python query allocations, explicitly excluding
SQLite native allocations, whole-system pressure and model memory. Record model
loaded size from Ollama while a bounded real-model trial is active and verify it
unloads afterward. Do not equate a disk weight size or API residency sample with
peak process RSS. Report unobserved quantities as unknown.

No activation based only on mocked vectors, renamed UI, or a passing unit suite.
Show measured results and their limits before changing personal runtime settings.

## Current measured result (v3)

**Smart remains experimental and failed its final quality gate. Exact is the
recommended default for remembered words, phrases and identifiers.** Smart is an
optional way to look for ideas when the original wording is forgotten; inspect
the source passage.

The frozen benchmark has 12 synthetic conversations, 49 messages, 48 positive and 16
unrelated queries. Calibration used only 8 conversations (32 positive/8 unrelated),
then threshold 0.50 and policy `meaning-evidence-v1` were frozen before the final
conversation-disjoint holdout. All64 queries used the actual installed
`qwen3-embedding:0.6b`; all source/branch/context integrity checks passed.

| Metric | Before upgrades | Final | Required |
|---|---:|---:|---:|
| All-positive passage Recall@3 |83.33%|87.50%|85%|
| All-positive MRR@10 |0.6986|0.7569|0.80|
| All-unrelated any-result false positives |6/16|2/16|<=20%|
| Holdout passage Recall@3 |12/16|13/16|85%|
| Holdout MRR@10 |0.6375|0.6562|0.80|
| Holdout unrelated any-result false positives |4/8|2/8|<=20%|

The implementation improved recall and rejection, but the holdout target
passage/ranking/rejection requirements remain unmet. No thresholds or gold labels
were changed after the final run. Do not describe these results as production
semantic readiness or personal-history relevance.

Final query median 714 ms, p95 743 ms. Evaluator peak RSS was 73,908,224 bytes
(70.5 MiB, measured with wait4); separately sampled Ollama loaded residency was
789,871,329 bytes (0.74 GiB). These are different measurements and must not be added
as a measured total. The model unloaded after the trial; minimum observed system
free percentage was 32% on the8 GiB host. This is no future memory guarantee.

A separate fixed two-message indexing batch traced164,520 bytes with 21 source
messages and 146,914 bytes with 2,001. Setup, SQLite native memory, RSS and models are
excluded from these Python allocation figures. Eleven independent synthetic
race/bounds checks and an integrated HTTP/SQLite pause/account-switch/resume flow
passed. Desktop1440 px and mobile 390 px browser flows opened Exact and meaning-only
Smart results at the correct source; those UI checks used a fake provider and
are separate from real-model quality evidence.

Raw reports: [before](evaluation-passage-v3-before.json),
[final quality](evaluation-passage-v3-final.json),
[calibration](evaluation-passage-v3-calibration.json),
[memory](evaluation-passage-v3-memory.json).

Reproduce offline structure with `python3 scripts/evaluate_passages.py`.
An explicit installed-model run is
`python3 scripts/evaluate_passages.py --ollama --require-quality`.
The recorded final run correctly exited 2 for unmet quality. Calibration diagnostics
use `python3 scripts/calibrate_passages.py --ollama`; partial calibration runs
never claim a full quality/holdout pass. Default checks never load a model.

## Historical conversation-only result (v2)

The real-model synthetic evaluation used 18 conversations and 64 queries (historical conversation-level with overlapping conversations, not new passage success).
Smart found the target in its top three for 48/48 paraphrases; their complete
phrases had zero exact matches. MRR at 10 was 0.9792. No unrelated query produced
a semantic suggestion, while one of 16 produced a keyword-only suggestion.
The fresh subset achieved 8/8 target recall, MRR 0.9375 and one keyword-only
false positive among eight unrelated queries. All context identity checks passed.

Query p95 was 682.368 ms with immediate unloading. Sampled model loaded size was
789,871,329 bytes (0.74 GiB), down from a stopped large-context trial's
2,370,652,077 bytes (2.21 GiB). The model was unloaded after both trials. These
are sampled Ollama residency figures on an 8 GiB unified-memory host, not peak
process RSS, a future memory guarantee, browser visual check, or personal-history relevance evidence.
See [the relevance report](evaluation-smart-v2.json) and
[the allocation/residency report](evaluation-smart-memory.json).

Reproduce the legacy conversation evaluation with `python3 scripts/evaluate_smart.py`
or query-allocation measurement with `python3 scripts/evaluate_smart.py --memory-only`.

## Passage-level evaluation benchmark (v3)

The previous evaluation suite (`smart-evaluation-v2`, 48 positive / 16 unrelated queries across 18 conversations) evaluated relevance at the whole-conversation level, and its holdout split allowed overlapping conversation identities.

The new passage-level benchmark (`passage-evaluation-v3`, in [evaluate_passages.py](../scripts/evaluate_passages.py) and [passage-evaluation.json](../fixtures/passage-evaluation.json)) supersedes the conversation-only benchmark:
- **Passage-level scoring**: Gold targets specify exact character codepoint spans (`start_offset`, `end_offset`), target message and conversation, and explicit expected ancestor IDs. A hit counts as a passage match only if the retrieved result matches the full namespace identity (`source_id`, `account_id`, `conversation_id`, `message_id`) and its `focus_offset` falls within the gold codepoint span. Conversation and message ranks are reported separately.
- **Frozen adversarial dataset**: 12 conversations with 49 messages total, containing 48 positive queries and 16 unrelated queries (including >= 12 Chinese / mixed-language queries). It includes same-conversation near-neighbour distractors, negation reversals, sibling branching trees, deep messages after astral Unicode codepoints, and dialogue acknowledgements requiring antecedent context.
- **Strict disjoint splits**: 8 calibration conversations (32 positive / 8 unrelated) and 4 completely disjoint holdout conversations (16 positive / 8 unrelated). Conversation identities across splits are validated before any model execution or indexing begins.
- **Denominator integrity**: Missing, failed, or dropped queries do not shrink metric denominators; failures are recorded against total expected suite queries.
- **Context ancestry verification**: Retrieved top hits verify contextual ancestry chains against gold `expected_ancestor_ids` under the correct `source_id` and `account_id` namespace.

Run the default offline lexical baseline:
```bash
python3 scripts/evaluate_passages.py --output /tmp/chat-recall-passage-baseline.json
```
An explicit `--ollama --require-quality` run executes against an installed local model provider and checks quality gates (focused passage Recall@3 >= 0.85, MRR@10 >= 0.80, unrelated any-result FP <= 0.20 across both global and holdout splits). Lexical baselines exit 0 with quality false; quality failures exit 2; structural or validation failures exit 1.
