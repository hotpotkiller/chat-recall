#!/usr/bin/env python3
"""Reproducible synthetic retrieval evaluation; real model calls are opt-in."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat_recall.providers import OllamaProvider
from chat_recall.search import SearchIndex


def run(args):
    payload = json.loads((ROOT / "fixtures/demo.json").read_text())
    suite = json.loads((ROOT / "fixtures/evaluation.json").read_text())
    provider = OllamaProvider(embedding_model=args.model, topic_model=args.topic_model) if args.ollama else None
    with tempfile.TemporaryDirectory(prefix="chat-recall-eval-") as directory:
        with SearchIndex(Path(directory) / "evaluation.sqlite", provider) as index:
            imported = index.import_payload(payload)
            original = {r["id"]: r["text"] for r in index.connection.execute("SELECT id,text FROM messages")}
            exact_failures, timings = [], []
            for query in suite["exact_queries"]:
                # Independent original-text oracle; no calls to literal_matches.
                expected = {key for key, text in original.items() if query.casefold() in text.casefold()}
                started = time.perf_counter()
                response = index.search(query, limit=200)
                timings.append((time.perf_counter() - started) * 1000)
                actual = {(r["conversation_id"], r["message_id"]) for r in response["results"]}
                expected_names = {(r["conversation_id"], r["message_id"]) for r in index.connection.execute("SELECT id,conversation_id,message_id FROM messages") if r["id"] in expected}
                if actual != expected_names:
                    exact_failures.append({"query": query, "expected": len(expected), "actual": len(actual)})
            enrichment = index.enrich(limit=10000, generate_topics=args.topics) if provider else None
            results, ranks, semantic_used = [], [], 0
            for case in suite["smart_queries"]:
                started = time.perf_counter()
                response = index.search(case["query"], mode="smart", limit=100)
                elapsed = (time.perf_counter() - started) * 1000
                conversations = list(dict.fromkeys(r["conversation_id"] for r in response["results"]))
                rank = conversations.index(case["conversation_id"]) + 1 if case["conversation_id"] in conversations else None
                semantic_used += int(response["semantic"]["used"])
                ranks.append(rank)
                results.append({"query": case["query"], "expected_conversation": case["conversation_id"],
                    "rank": rank, "elapsed_ms": round(elapsed, 2), "semantic_used": response["semantic"]["used"],
                    "semantic_status": response["semantic"]["status"]})
            total = len(ranks)
            hit5 = sum(rank is not None and rank <= 5 for rank in ranks) / total
            report = {
                "benchmark": "synthetic-demo-v1", "synthetic": True,
                "warning": "Synthetic integration evidence; not a held-out personal-history quality claim.",
                "import": imported, "exact": {"queries": len(suite["exact_queries"]),
                    "query_recall": 1 - len(exact_failures) / len(suite["exact_queries"]),
                    "failures": exact_failures, "median_ms": round(statistics.median(timings), 3),
                    "p95_ms": round(sorted(timings)[min(len(timings) - 1, math.ceil(len(timings) * .95) - 1)], 3)},
                "smart": {"queries": total, "mode": "real-ollama-hybrid" if provider else "lexical-baseline",
                    "model_id": getattr(provider, "model_id", None), "semantic_queries": semantic_used,
                    "target_in_top_5": hit5,
                    "mrr_at_10": sum(1 / rank if rank and rank <= 10 else 0 for rank in ranks) / total,
                    "ndcg_at_10": sum(1 / math.log2(rank + 1) if rank and rank <= 10 else 0 for rank in ranks) / total,
                    "enrichment": enrichment, "results": results},
                "gates": {"exact_passed": not exact_failures,
                    "synthetic_semantic_passed": bool(provider) and semantic_used == total and hit5 >= .9,
                    "real_history_quality": "not_evaluated", "live_chatgpt_sync": "not_evaluated"},
            }
            return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", action="store_true", help="Explicitly call an already installed local embedding model")
    parser.add_argument("--model", default="qwen3-embedding:0.6b")
    parser.add_argument("--topic-model", default="qwen3:1.7b")
    parser.add_argument("--topics", action="store_true", help="Also generate topic summaries (requires --ollama)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.topics and not args.ollama:
        parser.error("--topics requires --ollama")
    report = run(args)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not report["gates"]["exact_passed"]:
        return 1
    if args.ollama and not report["gates"]["synthetic_semantic_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
