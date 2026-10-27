#!/usr/bin/env python3
"""Scale and pagination retrieval evaluation against an independent original-text oracle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chat_recall.search import SearchIndex

QUERIES = [
    "shared archive",
    "serial:000000/0",
    "group:000",
    "STRASSE",
    "你好",
    "a+b(c)",
    "ana",
    "absent marker",
]


def count_overlapping(haystack: str, needle: str) -> int:
    """Independent oracle: count overlapping casefolded occurrences without engine helpers."""
    h = haystack.casefold()
    n = needle.casefold()
    if not n:
        return 0
    count = 0
    start = 0
    while True:
        pos = h.find(n, start)
        if pos == -1:
            break
        count += 1
        start = pos + 1
    return count


def generate_corpus(num_conversations: int) -> tuple[dict, list[tuple[str, str, str]]]:
    """Generate deterministic conversations and separate original oracle fixtures."""
    conversations = []
    oracle_fixtures = []

    for i in range(num_conversations):
        conv_id = f"conv-{i:06d}"
        conv_title = f"Scale Evaluation Conversation {i}"
        messages = []
        prev_id = None

        for j in range(5):
            msg_id = f"msg-{i:06d}-{j}"
            role = "user" if j % 2 == 0 else "assistant"
            global_idx = i * 5 + j

            if global_idx % 6 == 1:
                occasional = " with Straße 42"
            elif global_idx % 6 == 2:
                occasional = " with 你好世界"
            elif global_idx % 6 == 3:
                occasional = " with token a+b(c) expr"
            elif global_idx % 6 == 4:
                occasional = " with yellow banana fruit"
            else:
                occasional = ""

            text = (
                f"Record {i:06d}-{j}. Synthetic retrieval document verifying lexical "
                f"search integrity across large corpora without model execution. "
                f"The system evaluates exact query substring retrieval and "
                f"deterministic paging behavior. Entry serial:{i:06d}/{j} in "
                f"group:{i % 100:03d} under shared archive synthetic benchmark "
                f"payload{occasional}."
            )

            messages.append({
                "id": msg_id,
                "role": role,
                "text": text,
                "parent_id": prev_id,
            })
            oracle_fixtures.append((conv_id, msg_id, text))
            prev_id = msg_id

        conversations.append({
            "id": conv_id,
            "title": conv_title,
            "messages": messages,
        })

    payload = {"conversations": conversations}
    return payload, oracle_fixtures


def evaluate_scale(num_conversations: int) -> tuple[dict, list[dict]]:
    """Run full scale retrieval and pagination against the independent oracle."""
    payload, oracle_fixtures = generate_corpus(num_conversations)
    total_messages = len(oracle_fixtures)
    total_utf8_bytes = sum(len(text.encode("utf-8")) for _, _, text in oracle_fixtures)

    page_limit = 200
    query_metrics = []
    first_page_latencies = []
    oracle_failures = []

    with tempfile.TemporaryDirectory(prefix="chat-recall-scale-") as directory:
        db_path = Path(directory) / "scale.sqlite"
        with SearchIndex(db_path) as index:
            import_start = time.perf_counter()
            import_report = index.import_payload(payload)
            import_seconds = time.perf_counter() - import_start

            if import_report.get("errors"):
                oracle_failures.append({
                    "stage": "import",
                    "error": f"Import reported errors: {import_report['errors']}",
                })

            for query in QUERIES:
                expected_matches = {}
                for conv_id, msg_id, text in oracle_fixtures:
                    if query.casefold() in text.casefold():
                        expected_matches[(conv_id, msg_id)] = count_overlapping(text, query)

                expected_total = len(expected_matches)
                expected_identities = set(expected_matches.keys())

                results = []
                offset = 0
                first_page_ms = None
                pages_fetched = 0
                query_start = time.perf_counter()

                max_iterations = expected_total + 1
                iterations = 0

                while True:
                    iterations += 1
                    if iterations > max_iterations:
                        oracle_failures.append({
                            "query": query,
                            "error": f"Pagination exceeded maximum allowed loop bound ({max_iterations})",
                        })
                        break

                    page_start = time.perf_counter()
                    response = index.search(query, mode="exact", limit=page_limit, offset=offset)
                    page_elapsed_ms = (time.perf_counter() - page_start) * 1000
                    pages_fetched += 1

                    if first_page_ms is None:
                        first_page_ms = page_elapsed_ms

                    total = response["total"]
                    page_results = response["results"]

                    if total != expected_total:
                        oracle_failures.append({
                            "query": query,
                            "error": f"Page at offset {offset} returned total {total}, expected {expected_total}",
                        })

                    if len(page_results) > page_limit:
                        oracle_failures.append({
                            "query": query,
                            "error": f"Page at offset {offset} returned {len(page_results)} items, exceeding limit {page_limit}",
                        })

                    if total == 0:
                        if page_results:
                            oracle_failures.append({
                                "query": query,
                                "error": f"Total was 0 but received {len(page_results)} results on first page",
                            })
                        break

                    if not page_results:
                        if len(results) < total:
                            oracle_failures.append({
                                "query": query,
                                "error": (
                                    f"Pagination cannot make progress at offset {offset}: "
                                    f"received 0 results but total is {total} (retrieved {len(results)})"
                                ),
                            })
                        break

                    results.extend(page_results)
                    offset += len(page_results)

                    if len(results) >= total:
                        break

                query_elapsed_ms = (time.perf_counter() - query_start) * 1000

                if first_page_ms is not None:
                    first_page_latencies.append(first_page_ms)

                if total != expected_total:
                    oracle_failures.append({
                        "query": query,
                        "error": f"Returned total {total} does not equal oracle {expected_total}",
                    })

                if len(results) != expected_total:
                    oracle_failures.append({
                        "query": query,
                        "error": f"Paging retrieved {len(results)} items, expected {expected_total}",
                    })

                actual_identities = [(r["conversation_id"], r["message_id"]) for r in results]
                if len(actual_identities) != len(set(actual_identities)):
                    oracle_failures.append({
                        "query": query,
                        "error": f"Duplicate result identities returned across pages for '{query}'",
                    })

                if set(actual_identities) != expected_identities:
                    oracle_failures.append({
                        "query": query,
                        "error": f"Result identity set does not match oracle expected set for '{query}'",
                    })

                match_count_sum = 0
                for r in results:
                    ident = (r["conversation_id"], r["message_id"])
                    expected_mc = expected_matches.get(ident)
                    actual_mc = r["match_count"]
                    match_count_sum += actual_mc
                    if actual_mc != expected_mc:
                        oracle_failures.append({
                            "query": query,
                            "error": f"Match count mismatch for {ident}: engine={actual_mc}, oracle={expected_mc}",
                        })
                        break

                query_metrics.append({
                    "query": query,
                    "oracle_expected": expected_total,
                    "retrieved_total": len(results),
                    "pages_fetched": pages_fetched,
                    "first_page_ms": round(first_page_ms if first_page_ms is not None else 0.0, 3),
                    "total_pagination_ms": round(query_elapsed_ms, 3),
                    "match_count_sum": match_count_sum,
                })

        db_bytes = sum(p.stat().st_size for p in Path(directory).glob("scale.sqlite*"))

    if first_page_latencies:
        median_first_page = round(statistics.median(first_page_latencies), 3)
        sorted_latencies = sorted(first_page_latencies)
        p95_index = min(len(sorted_latencies) - 1, math.ceil(len(sorted_latencies) * 0.95) - 1)
        p95_first_page = round(sorted_latencies[p95_index], 3)
    else:
        median_first_page = 0.0
        p95_first_page = 0.0

    report = {
        "benchmark": "scale-exact-eval",
        "timing_sample": True,
        "warning": "Single run per query timing sample; not a production or held-out performance guarantee.",
        "environment": {
            "python_version": sys.version.split()[0],
            "sqlite_version": sqlite3.sqlite_version,
        },
        "corpus": {
            "conversations": num_conversations,
            "messages": total_messages,
            "utf8_bytes": total_utf8_bytes,
            "db_bytes": db_bytes,
            "import_seconds": round(import_seconds, 4),
        },
        "latency_summary": {
            "first_page_median_ms": median_first_page,
            "first_page_p95_ms": p95_first_page,
        },
        "queries": query_metrics,
        "oracle_failures": oracle_failures,
        "gates": {
            "exact_passed": len(oracle_failures) == 0,
        },
    }
    return report, oracle_failures


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=1000,
        help="Number of conversations to generate and evaluate (1..5000, default 1000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write evaluation report JSON",
    )
    args = parser.parse_args(argv)
    if args.conversations < 1 or args.conversations > 5000:
        parser.error("--conversations must be an integer between 1 and 5000")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    report, failures = evaluate_scale(args.conversations)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)

    if failures or not report["gates"]["exact_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
