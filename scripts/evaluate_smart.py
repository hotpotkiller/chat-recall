#!/usr/bin/env python3
"""Reproducible synthetic Exact-versus-Smart relevance and memory evaluation.

Follows AGENTS.md:
- Synthetic evaluation fixture only, no private chats, credentials, or machine paths.
- Default mode tests lexical baseline only without network calls or provider downloads.
- Opt-in --ollama mode evaluates OllamaProvider embedding on the synthetic suite.
- Opt-in --memory-only mode profiles peak memory allocations with tracemalloc using
  a deterministic 16-dimensional synthetic provider.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat_recall.providers import OllamaProvider, normalized_vector
from chat_recall.search import SearchIndex


class DeterministicSynthetic16DProvider:
    """Deterministic 16-dimensional synthetic provider for memory benchmarking.

    No network, no model files, completely reproducible.
    """
    model_id = "synthetic:16d-deterministic-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            # Deterministic pseudo-vector derived from sha256
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [((digest[i] % 100) - 50) / 50.0 for i in range(16)]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            embeddings.append([x / norm for x in raw])
        return embeddings


class RecordingProviderWrapper:
    """Thin wrapper around an EmbeddingProvider that captures the last single-input query vector.

    Eliminates redundant model calls for diagnostic similarity computations.
    """
    def __init__(self, provider):
        self._provider = provider
        self.last_query_vector: list[float] | None = None
        self.embed_call_count: int = 0

    @property
    def model_id(self) -> str:
        return self._provider.model_id

    @property
    def max_input_bytes(self) -> int | None:
        return getattr(self._provider, "max_input_bytes", None)

    @property
    def minimum_similarity(self) -> float:
        return getattr(self._provider, "minimum_similarity", 0.3)

    def summarize(self, text: str) -> str:
        if callable(getattr(self._provider, "summarize", None)):
            return self._provider.summarize(text)
        raise AttributeError("Provider does not support topic summarization")

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_call_count += 1
        vectors = self._provider.embed(texts)
        if len(texts) == 1 and vectors:
            self.last_query_vector = normalized_vector(vectors[0])
        return vectors

    def reset_query_vector(self) -> None:
        self.last_query_vector = None


def evaluate_memory(query_count: int = 1) -> dict:
    """Run memory profiling using a 2000-message synthetic dataset outside tracemalloc."""
    provider = DeterministicSynthetic16DProvider()

    # 1. Generate 2000 short-to-medium synthetic messages across conversations
    # Keep fixture text + derived text well below 16MiB
    synthetic_conversations = []
    total_messages = 2000
    msg_idx = 0
    conv_id = 0
    while msg_idx < total_messages:
        conv_id += 1
        messages_in_conv = min(10, total_messages - msg_idx)
        conv_messages = []
        parent_id = None
        for i in range(messages_in_conv):
            msg_id = f"mem-msg-{msg_idx}"
            role = "user" if i % 2 == 0 else "assistant"
            # Realistic short-to-medium dialogue (~120 chars)
            text = (
                f"Synthetic message {msg_idx} regarding benchmark memory consumption. "
                f"Topic sector {msg_idx % 25} testing allocation limits and search efficiency."
            )
            conv_messages.append({
                "id": msg_id,
                "parent_id": parent_id,
                "role": role,
                "text": text
            })
            parent_id = msg_id
            msg_idx += 1
        synthetic_conversations.append({
            "id": f"mem-conv-{conv_id}",
            "title": f"Synthetic memory benchmark conversation {conv_id}",
            "messages": conv_messages
        })

    payload = {"conversations": synthetic_conversations}

    # 2. Setup index and enrich OUTSIDE tracemalloc
    with tempfile.TemporaryDirectory(prefix="chat-recall-mem-") as directory:
        db_path = Path(directory) / "memory_eval.sqlite"
        with SearchIndex(db_path, provider) as index:
            index.import_payload(payload)
            index.enrich(limit=10000)

            # Fixed test queries
            test_queries = [
                "benchmark memory consumption sector 5",
                "testing allocation limits and search efficiency"
            ][:max(1, query_count)]

            # 3. Measure query Python peak allocated bytes and elapsed time inside tracemalloc
            tracemalloc.reset_peak()
            tracemalloc.start()
            start_time = time.perf_counter()

            result_counts = []
            for q in test_queries:
                resp = index.search(q, mode="smart", limit=20)
                result_counts.append(resp["total"])

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

    report = {
        "mode": "memory-only",
        "benchmark": "synthetic-memory-v1",
        "fixture_corpus_version": "synthetic-2000-msgs-v1",
        "synthetic": True,
        "note": "Query phase traced with tracemalloc; excludes index creation, import, and enrich allocations. Native/model process RSS excluded.",
        "corpus": {
            "conversations": len(synthetic_conversations),
            "messages": total_messages,
            "provider_model_id": provider.model_id
        },
        "query_benchmark": {
            "queries_executed": len(test_queries),
            "result_counts": result_counts,
            "elapsed_ms": round(elapsed_ms, 3),
            "python_peak_allocated_bytes": peak_bytes,
            "python_current_allocated_bytes": current_bytes
        }
    }
    return report


def evaluate_relevance(args) -> dict:
    """Evaluate Exact versus Smart search on the synthetic fixture."""
    fixture_path = ROOT / "fixtures/smart-evaluation.json"
    suite = json.loads(fixture_path.read_text(encoding="utf-8"))

    raw_provider = None
    if getattr(args, "ollama", False):
        raw_provider = OllamaProvider(
            embedding_model=args.model,
            timeout=args.timeout
        )

    recording_wrapper = RecordingProviderWrapper(raw_provider) if raw_provider else None
    provider = recording_wrapper

    with tempfile.TemporaryDirectory(prefix="chat-recall-smart-eval-") as directory:
        db_path = Path(directory) / "smart_eval.sqlite"
        with SearchIndex(db_path, provider) as index:
            # 1. Import synthetic conversations
            import_report = index.import_payload({"conversations": suite["conversations"]})
            import_errors = list(import_report.get("errors", []))

            # Enrich only when provider is opted-in
            enrichment_report = None
            enrichment_errors = []
            if provider:
                enrichment_report = index.enrich(limit=10000)
                enrichment_errors = list(enrichment_report.get("errors", []))

            # Helper to inspect stored vectors for computing semantic similarities without rerunning models
            msg_to_conv = {}
            for r in index.connection.execute("SELECT id, conversation_id FROM messages"):
                msg_to_conv[r["id"]] = r["conversation_id"]

            stored_vectors = []
            if provider:
                for r in index.connection.execute("SELECT message_row, channel, vector_json FROM vectors WHERE model_id=?", (provider.model_id,)):
                    stored_vectors.append({
                        "message_row": r["message_row"],
                        "conversation_id": msg_to_conv.get(r["message_row"]),
                        "channel": r["channel"],
                        "vector": json.loads(r["vector_json"])
                    })

            # Prepare queries
            queries = suite["queries"]
            positive_queries = [q for q in queries if q["type"] == "positive"]
            unrelated_queries = [q for q in queries if q["type"] == "unrelated"]

            exact_target_hits = 0
            smart_target_top3 = 0
            smart_target_top10 = 0
            mrr_at_10_sum = 0.0
            context_identity_errors = 0
            semantic_used_positive = 0
            semantic_used_unrelated = 0
            semantic_failures = []
            query_timings_ms = []
            positive_results = []
            unrelated_results = []
            query_errors = []
            collected_minimum_similarity = None

            # 2. Evaluate positive queries on both exact and smart mode
            for item in positive_queries:
                q_text = item["query"]
                target_conv = item["expected_conversation_id"]

                # Exact search evaluation
                try:
                    exact_resp = index.search(q_text, mode="exact", limit=50)
                    exact_convs = [r["conversation_id"] for r in exact_resp["results"]]
                    if target_conv in exact_convs:
                        exact_target_hits += 1
                except Exception as exc:
                    query_errors.append({"query": q_text, "mode": "exact", "error": str(exc)})

                # Reset recorded query vector before smart search
                if recording_wrapper:
                    recording_wrapper.reset_query_vector()

                # Smart search evaluation
                t0 = time.perf_counter()
                try:
                    smart_resp = index.search(q_text, mode="smart", limit=50)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    query_timings_ms.append(elapsed)

                    sem_info = smart_resp.get("semantic", {})
                    sem_used = bool(sem_info.get("used"))
                    sem_status = sem_info.get("status")
                    if "minimum_similarity" in sem_info and sem_info["minimum_similarity"] is not None:
                        collected_minimum_similarity = sem_info["minimum_similarity"]

                    if sem_used:
                        semantic_used_positive += 1
                    elif provider is not None:
                        # Provider was configured, but semantic was not used
                        semantic_failures.append({
                            "query": q_text,
                            "type": "positive",
                            "status": sem_status,
                            "errors": sem_info.get("errors", [])
                        })

                    # Calculate target semantic similarity using captured query vector (no second model call)
                    target_best_similarity = None
                    if sem_used and stored_vectors and recording_wrapper and recording_wrapper.last_query_vector:
                        q_vec = recording_wrapper.last_query_vector
                        target_sims = [
                            sum(a * b for a, b in zip(q_vec, sv["vector"]))
                            for sv in stored_vectors
                            if sv["conversation_id"] == target_conv
                        ]
                        if target_sims:
                            target_best_similarity = round(max(target_sims), 4)

                    # Get distinct conversation ranking
                    found_convs = list(dict.fromkeys(r["conversation_id"] for r in smart_resp["results"]))
                    rank = found_convs.index(target_conv) + 1 if target_conv in found_convs else None

                    if rank is not None and rank <= 3:
                        smart_target_top3 += 1
                    if rank is not None and rank <= 10:
                        smart_target_top10 += 1
                        mrr_at_10_sum += 1.0 / rank

                    # Verify context target identity/branch correctness for top hit if present
                    if smart_resp["results"]:
                        top_res = smart_resp["results"][0]
                        ctx = index.context(top_res["conversation_id"], top_res["message_id"])
                        # Verify returned context contains the target message_id and conversation_id
                        if ctx["conversation_id"] != top_res["conversation_id"] or ctx["message_id"] != top_res["message_id"]:
                            context_identity_errors += 1
                        # Verify message chain integrity
                        ctx_msg_ids = [m["id"] for m in ctx["messages"]]
                        if top_res["message_id"] not in ctx_msg_ids:
                            context_identity_errors += 1

                    positive_results.append({
                        "id": item.get("id"),
                        "split": item.get("split", "calibration"),
                        "query": q_text,
                        "expected_conversation_id": target_conv,
                        "rank": rank,
                        "elapsed_ms": round(elapsed, 2),
                        "semantic_used": sem_used,
                        "semantic_status": sem_status,
                        "target_best_semantic_similarity": target_best_similarity
                    })
                except Exception as exc:
                    query_errors.append({"query": q_text, "mode": "smart", "error": str(exc)})

            # 3. Evaluate unrelated queries
            unrelated_semantic_false_positives = 0
            unrelated_any_result_false_positives = 0
            for item in unrelated_queries:
                q_text = item["query"]

                if recording_wrapper:
                    recording_wrapper.reset_query_vector()

                t0 = time.perf_counter()
                try:
                    smart_resp = index.search(q_text, mode="smart", limit=50)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    query_timings_ms.append(elapsed)

                    sem_info = smart_resp.get("semantic", {})
                    sem_used = bool(sem_info.get("used"))
                    sem_status = sem_info.get("status")
                    if "minimum_similarity" in sem_info and sem_info["minimum_similarity"] is not None:
                        collected_minimum_similarity = sem_info["minimum_similarity"]

                    if sem_used:
                        semantic_used_unrelated += 1
                    elif provider is not None:
                        semantic_failures.append({
                            "query": q_text,
                            "type": "unrelated",
                            "status": sem_status,
                            "errors": sem_info.get("errors", [])
                        })

                    # Calculate max semantic similarity using captured query vector (no second model call)
                    max_similarity = None
                    if sem_used and stored_vectors and recording_wrapper and recording_wrapper.last_query_vector:
                        q_vec = recording_wrapper.last_query_vector
                        all_sims = [
                            sum(a * b for a, b in zip(q_vec, sv["vector"]))
                            for sv in stored_vectors
                        ]
                        if all_sims:
                            max_similarity = round(max(all_sims), 4)

                    # An unrelated query produces a semantic false positive if semantic search was used
                    # and returned any semantic or hybrid results
                    has_semantic_result = any(
                        r.get("match_type") in ("semantic", "hybrid")
                        for r in smart_resp["results"]
                    )
                    if has_semantic_result:
                        unrelated_semantic_false_positives += 1

                    has_any_result = smart_resp["total"] > 0
                    if has_any_result:
                        unrelated_any_result_false_positives += 1

                    unrelated_results.append({
                        "id": item.get("id"),
                        "split": item.get("split", "calibration"),
                        "query": q_text,
                        "semantic_used": sem_used,
                        "semantic_status": sem_status,
                        "total_results": smart_resp["total"],
                        "has_semantic_result": has_semantic_result,
                        "has_any_result": has_any_result,
                        "max_semantic_similarity": max_similarity,
                        "elapsed_ms": round(elapsed, 2)
                    })
                except Exception as exc:
                    query_errors.append({"query": q_text, "mode": "smart", "error": str(exc)})

            # Calculate metrics
            num_pos = len(positive_queries)
            num_unrelated = len(unrelated_queries)
            total_queries = num_pos + num_unrelated
            total_semantic_used = semantic_used_positive + semantic_used_unrelated

            top3_recall = smart_target_top3 / num_pos if num_pos else 0.0
            mrr_10 = mrr_at_10_sum / num_pos if num_pos else 0.0
            exact_recall = exact_target_hits / num_pos if num_pos else 0.0
            unrelated_fp_rate = (unrelated_semantic_false_positives / num_unrelated) if num_unrelated else 0.0
            unrelated_any_result_fp_rate = (unrelated_any_result_false_positives / num_unrelated) if num_unrelated else 0.0

            # Calculate metrics per split
            expected_split_counts = {
                "calibration": {
                    "positive": len([q for q in positive_queries if q.get("split") == "calibration"]),
                    "unrelated": len([q for q in unrelated_queries if q.get("split") == "calibration"]),
                },
                "holdout": {
                    "positive": len([q for q in positive_queries if q.get("split") == "holdout"]),
                    "unrelated": len([q for q in unrelated_queries if q.get("split") == "holdout"]),
                },
            }

            split_results = {}
            for sp in ("calibration", "holdout"):
                sp_pos = [r for r in positive_results if r.get("split") == sp]
                sp_unrelated = [r for r in unrelated_results if r.get("split") == sp]
                sp_num_pos = len(sp_pos)
                sp_num_unrelated = len(sp_unrelated)
                sp_top3 = sum(1 for r in sp_pos if r.get("rank") is not None and r["rank"] <= 3)
                sp_mrr10_sum = sum(1.0 / r["rank"] for r in sp_pos if r.get("rank") is not None and r["rank"] <= 10)
                sp_sem_fp = sum(1 for r in sp_unrelated if r.get("has_semantic_result"))
                sp_any_fp = sum(1 for r in sp_unrelated if r.get("has_any_result"))
                sp_sem_used_pos = sum(1 for r in sp_pos if r.get("semantic_used"))
                sp_sem_used_unrelated = sum(1 for r in sp_unrelated if r.get("semantic_used"))

                split_results[sp] = {
                    "positive_count": sp_num_pos,
                    "top3_target_recall": round(sp_top3 / sp_num_pos, 4) if sp_num_pos else 0.0,
                    "mrr_at_10": round(sp_mrr10_sum / sp_num_pos, 4) if sp_num_pos else 0.0,
                    "positive_semantic_used_count": sp_sem_used_pos,
                    "unrelated_count": sp_num_unrelated,
                    "unrelated_semantic_used_count": sp_sem_used_unrelated,
                    "unrelated_any_semantic_result_false_positive_rate": round(sp_sem_fp / sp_num_unrelated, 4) if sp_num_unrelated else 0.0,
                    "unrelated_any_result_false_positive_rate": round(sp_any_fp / sp_num_unrelated, 4) if sp_num_unrelated else 0.0,
                }

            sorted_timings = sorted(query_timings_ms) if query_timings_ms else [0.0]
            median_ms = round(statistics.median(sorted_timings), 3)
            p95_index = min(len(sorted_timings) - 1, math.ceil(len(sorted_timings) * 0.95) - 1)
            p95_ms = round(sorted_timings[p95_index], 3)

            target_sims = [
                r["target_best_semantic_similarity"]
                for r in positive_results
                if r.get("target_best_semantic_similarity") is not None
            ]
            min_target_sim = min(target_sims) if target_sims else None

            # semantic_evaluated requires actual semantic execution, not mere provider construction
            semantic_evaluated = (total_semantic_used > 0)

            # Structural validity gate:
            # - import errors empty
            # - query errors empty
            # - context identity errors == 0
            structure_passed = (
                len(import_errors) == 0
                and len(query_errors) == 0
                and context_identity_errors == 0
            )

            # Holdout subset quality metrics
            expected_holdout_pos = expected_split_counts["holdout"]["positive"]
            expected_holdout_unrelated = expected_split_counts["holdout"]["unrelated"]
            holdout_metrics = split_results.get("holdout", {})
            holdout_completed_pos = holdout_metrics.get("positive_count", 0)
            holdout_completed_unrelated = holdout_metrics.get("unrelated_count", 0)

            holdout_passed = (
                structure_passed
                and len(enrichment_errors) == 0
                and len(semantic_failures) == 0
                and semantic_evaluated
                and expected_holdout_pos > 0
                and expected_holdout_unrelated > 0
                and holdout_completed_pos == expected_holdout_pos
                and holdout_completed_unrelated == expected_holdout_unrelated
                and holdout_metrics.get("positive_semantic_used_count", 0) == expected_holdout_pos
                and holdout_metrics.get("unrelated_semantic_used_count", 0) == expected_holdout_unrelated
                and holdout_metrics.get("top3_target_recall", 0.0) >= 0.85
                and holdout_metrics.get("mrr_at_10", 0.0) >= 0.80
                and holdout_metrics.get("unrelated_any_semantic_result_false_positive_rate", 1.0) <= 0.20
                and holdout_metrics.get("unrelated_any_result_false_positive_rate", 1.0) <= 0.20
            )

            # Semantic Quality gate:
            # - semantic retrieval actually used for every query (positive and unrelated)
            # - semantic_failures empty (no provider_error or index_pending during queries)
            # - enrichment errors empty
            # - import errors empty
            # - query errors empty
            # - zero context identity errors
            # - top3_recall >= 0.85
            # - mrr_10 >= 0.80
            # - unrelated_fp_rate <= 0.20
            # - holdout subset satisfies same quality gates and any-result FP <= 0.20
            # (Lexical ranking must never satisfy quality gate when semantic retrieval was not used)
            quality_passed = (
                semantic_evaluated
                and total_semantic_used == total_queries
                and len(semantic_failures) == 0
                and len(enrichment_errors) == 0
                and len(import_errors) == 0
                and len(query_errors) == 0
                and context_identity_errors == 0
                and top3_recall >= 0.85
                and mrr_10 >= 0.80
                and unrelated_fp_rate <= 0.20
                and holdout_passed
            )

            report = {
                "benchmark": "smart-evaluation-v2",
                "synthetic": True,
                "fixture_corpus_version": suite.get("version", "smart-evaluation-v2"),
                "mode": "ollama-hybrid" if provider else "lexical-baseline-only",
                "semantic_evaluated": semantic_evaluated,
                "model_id": getattr(provider, "model_id", None),
                "minimum_similarity": collected_minimum_similarity,
                "import": import_report,
                "enrichment": enrichment_report,
                "metrics": {
                    "total_queries": total_queries,
                    "semantic_used_count": total_semantic_used,
                    "positive_queries": num_pos,
                    "positive_semantic_used_count": semantic_used_positive,
                    "top3_target_recall": round(top3_recall, 4),
                    "mrr_at_10": round(mrr_10, 4),
                    "exact_target_recall": round(exact_recall, 4),
                    "unrelated_queries": num_unrelated,
                    "unrelated_semantic_used_count": semantic_used_unrelated,
                    "unrelated_any_semantic_result_false_positive_rate": round(unrelated_fp_rate, 4),
                    "unrelated_any_result_false_positive_rate": round(unrelated_any_result_fp_rate, 4),
                    "context_identity_errors": context_identity_errors,
                    "min_target_observed_semantic_similarity": min_target_sim,
                    "split_results": split_results,
                    "query_timings_ms": {
                        "median": median_ms,
                        "p95": p95_ms
                    }
                },
                "gates": {
                    "structure_passed": structure_passed,
                    "quality_passed": quality_passed,
                    "holdout_passed": holdout_passed,
                    "rules": {
                        "require_all_queries_semantic_used": True,
                        "min_top3_recall": 0.85,
                        "min_mrr_10": 0.80,
                        "max_unrelated_fp_rate": 0.20,
                        "max_holdout_unrelated_any_result_fp_rate": 0.20,
                        "max_context_identity_errors": 0,
                        "max_import_errors": 0,
                        "max_enrichment_errors": 0,
                        "max_query_errors": 0,
                        "max_semantic_failures": 0
                    }
                },
                "semantic_failures": semantic_failures,
                "query_errors": query_errors,
                "positive_results": positive_results,
                "unrelated_results": unrelated_results
            }
            return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", action="store_true", help="Explicitly call local OllamaProvider")
    parser.add_argument("--model", default="qwen3-embedding:0.6b", help="Ollama embedding model name")
    parser.add_argument("--timeout", type=float, default=30.0, help="Ollama timeout in seconds (default 30)")
    parser.add_argument("--require-quality", action="store_true", help="Require semantic quality gates to pass (requires --ollama)")
    parser.add_argument("--memory-only", action="store_true", help="Run deterministic 16D memory profiling mode")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    if args.require_quality and not args.ollama:
        parser.error("--require-quality requires --ollama")

    if args.memory_only:
        report = evaluate_memory()
    else:
        report = evaluate_relevance(args)

    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)

    if args.memory_only:
        return 0

    if not report["gates"]["structure_passed"]:
        return 1

    if args.require_quality and not report["gates"]["quality_passed"]:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
