# Reproducible validation

All public fixtures and benchmark corpora are synthetic. No personal history or
provider credentials are needed for the default checks.

The clean release snapshot passed 146 Python unit/integration tests, seven
JavaScript syntax checks and all three offline JavaScript behavior suites before
publication. Its small default evaluation retrieved all 60 expected exact queries.
These local release checks do not load an embedding model or contact a chat account.

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

## Optional Smart evaluation

The current real-model synthetic run used qwen3-embedding:0.6b on 18 conversations
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
This small synthetic evaluation does not establish personal-history relevance.
Default checks remain offline and Smart/importing remain off by default.
