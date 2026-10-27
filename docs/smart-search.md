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

- Semantic candidates use the provider's validated cosine threshold: 0.5 for
  qwen3-embedding:0.6b, calibrated on the original 48-query subset; other providers
  retain the initial 0.3 heuristic. This is not a probability or universal threshold.
  Fresh holdout queries are reported separately. Zero/negative candidates must
  not become positive results merely because rank fusion assigned them a score.
- SQL applies account/speaker/model/current-content filters before retrieval.
  Stream vector columns without derived text, retain at most 100 unique messages
  per semantic channel, and fetch full message bodies only for the requested page.
  Ranking CPU remains linear in stored vectors; this is not an ANN index.
- Query-provider failures discard all semantic evidence and retain explicitly
  labelled keyword fallback. No configured/current vectors means no embedding
  call. Exact search never calls a provider.
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

## Recorded result

The real-model synthetic evaluation used 18 conversations and 64 queries.
Smart found the target in its top three for 48/48 paraphrases; their complete
phrases had zero exact matches. MRR at 10 was 0.9792. No unrelated query produced
a semantic suggestion, while one of 16 produced a keyword-only suggestion.
The fresh subset achieved 8/8 target recall, MRR 0.9375 and one keyword-only
false positive among eight unrelated queries. All context identity checks passed.

Query p95 was 682.368 ms with immediate unloading. Sampled model loaded size was
789,871,329 bytes (0.74 GiB), down from a stopped large-context trial's
2,370,652,077 bytes (2.21 GiB). The model was unloaded after both trials. These
are sampled Ollama residency figures on an 8 GiB unified-memory host, not peak
process RSS, a future memory guarantee or personal-history relevance evidence.
See [the relevance report](evaluation-smart-v2.json) and
[the allocation/residency report](evaluation-smart-memory.json).

Reproduce the default lexical evaluation with `python3 scripts/evaluate_smart.py`
or query-allocation measurement with `python3 scripts/evaluate_smart.py --memory-only`.
An explicit `--ollama --require-quality` run uses an already installed local model
and checks the synthetic gates. It is excluded from ordinary CI/default checks.
