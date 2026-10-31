# Reproducible validation

All public fixtures and benchmark corpora are synthetic. No personal history or
provider credentials are needed for the default checks.

The verified implementation passed 272 Python tests, seven JavaScript syntax
checks and all three offline JavaScript behavior suites. The default Exact
evaluation retrieved all 60 expected queries. Eleven independent offline
cancellation/source-bound checks and the integrated HTTP pause/account/resume
flow passed. Actual desktop 1440 px/mobile 390 px browser navigation used synthetic
chats and a fake provider; real-model quality was evaluated separately.

```sh
make check
make check-js
make eval
python3 -m chat_recall.workflow validate
```

Python checks cover search/import/provider contracts, local HTTP boundaries and
real temporary Git/worktree review flows. JavaScript checks execute actual
application/background functions in synthetic environments, including UI request
races, Unicode highlighting, branch/source parameters, safe fallback labels,
Smart-off controls and persisted browser retry behavior.

## Recorded exact-search scale sample

| Item | Recorded value |
|---|---:|
| Synthetic conversations | 5,000 |
| Original synthetic messages | 25,000 |
| Oracle failures, baseline and optimized | 0 |
| Baseline first-page p95 | 906.422 ms |
| Optimized first-page p95 | 133.612 ms |

Reports: [baseline](evaluation-scale-baseline.json),
[optimized](evaluation-scale-optimized.json).

Reproduce using `python3 scripts/evaluate_scale.py --conversations 5000`.
Use `--help` for its output option. This creates synthetic data in
a temporary directory and does not load a model. The full all-pages run can take
time; it is not required for every UI change.

These are single-run per-query timing samples on one environment. They are not
production latency promises or a held-out personal-history relevance evaluation.
The small optional semantic experiment is retained in
[its original synthetic report](evaluation-real-semantic.json) as historical
research evidence only. Smart remains disabled by default; reproducing that
optional real-model experiment is not part of the default checks.

CI is defined in `.github/workflows/checks.yml`. Its actual status is visible on
the repository's Actions tab; a definition alone is not a passing run.

## Historical conversation-level Smart evaluation

The legacy v2 real-model synthetic run used qwen3-embedding:0.6b on 18 conversations
and 64 queries. Target recall at 3 was 48/48; MRR at 10 was 0.9792. The 16 fresh
holdout queries achieved 8/8 target recall and MRR 0.9375; one unrelated holdout
query produced a keyword-only suggestion, while none produced a semantic one.
No source/context identity errors occurred. Query p95 was 682.368 ms.

Sampled loaded model size fell from 2.21 GiB in a stopped large-context trial to
0.74 GiB with the small profile; it unloaded afterward. On a separate fixed
2,000-message/16-dimensional synthetic query sample, peak traced Python
allocations fell from 4,562,214 to 150,577 bytes. These exclude native/model RSS.

See [Smart design and reproduction](smart-search.md),
[raw relevance results](evaluation-smart-v2.json) and
[allocation/residency results](evaluation-smart-memory.json). The cutoff was
chosen on the calibration subset and frozen before the fresh holdout run.
This was conversation-level relevance, not passage-level correctness. Its calibration and
holdout queries shared conversation identities, so the holdout was not conversation-disjoint.
These figures do not establish passage quality or personal-history relevance.
Default checks remain offline and Smart/importing remain off by default.

## Passage-level evaluation

The final actual `qwen3-embedding:0.6b` run completed all 64 queries with zero
source/branch/context errors. **The quality gate failed.** Passage Recall@3
was 42/48 overall and 13/16 on the conversation-disjoint holdout. MRR@10 was
0.7569 overall and 0.6562 on holdout. Unrelated-query any-result false positives
were 2/16 overall and 2/8 on holdout. The gate requires Recall@3 >= 0.85,
MRR@10 >= 0.80 and unrelated FP <= 0.20, globally and on holdout.

The earlier v3 baseline was 40/48 overall,12/16 holdout, holdout MRR0.6375 and
holdout unrelated FP4/8. Gains do not imply that Smart is ready as a default.
No gold labels or cutoffs were changed after the final holdout run.

Evaluator peak RSS70.5 MiB excludes the separate Ollama server. Sampled model
residency0.74 GiB is not model process RSS or a total-memory peak; it unloaded
afterward. Query median 714 ms and p95 743 ms are single-run synthetic timings.
See [the measured report and limits](smart-search.md#current-measured-result-v3),
[raw quality](evaluation-passage-v3-final.json) and
[memory observations](evaluation-passage-v3-memory.json).

`python3 scripts/evaluate_passages.py` runs the offline lexical structure check.
The explicit `--ollama --require-quality` run uses an already installed local
model and correctly exits2 when the quality gate fails. Default checks never
load it. `scripts/calibrate_passages.py --ollama` indexes/queries calibration
conversations only; its cached-sweep timings are distinguished from actual
inference and it cannot claim holdout success.
