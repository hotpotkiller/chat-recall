#!/usr/bin/env python3
"""Passage-level similarity threshold calibration benchmark and CLI runner.

Follows AGENTS.md:
- Synthetic evaluation fixture only, no private chats, credentials, or machine paths.
- Default mode tests offline lexical baseline without network calls or provider downloads.
- Opt-in --ollama mode evaluates local OllamaProvider embedding on the synthetic calibration split.
- Caches validated document/query vectors in memory across trials with finite cap.
- Isolated caches bound by immutable index_id, pipeline version, and input hash.
- Threshold exploration is diagnostic only: no auto-writing provider defaults or calibration version.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat_recall.providers import OllamaProvider, ProviderError, callable_accepts_index_id, normalized_vector
from chat_recall.search import SearchIndex
from scripts.evaluate_passages import FixtureValidationError, evaluate, validate_suite


class CalibrationError(Exception):
    """Base error for calibration failures."""
    pass


class CacheCapacityExceededError(CalibrationError):
    """Raised when in-memory vector cache exceeds configured finite capacity cap."""
    pass


class InvalidVectorError(CalibrationError, ValueError):
    """Raised when provider returns invalid, non-numeric, or oversized embedding vectors."""
    pass


class CachingProviderWrapper:
    """In-memory vector caching provider wrapper bounded by identity, version, and finite cap.

    - Separate query vs document caches (namespaces never mix).
    - Immutable index_id + query/document pipeline version + input hash.
    - Standard immutable numeric vectors cached as tuples and returned as fresh list copies
      so engine normalization cannot corrupt cached originals.
    - Model/profile change invalidates and clears caches to prevent stale reuse.
    - Enforces finite memory cap (default <= 1024 vectors, dimensions <= 1024).
    - Provider TypeError and exceptions propagate immediately without retry.
    """

    MAX_CACHE_CAPACITY: int = 1024
    MAX_VECTOR_DIM: int = 1024

    def __init__(self, provider: Any, default_threshold: float = 0.50, max_capacity: int = 1024):
        if type(max_capacity) is not int or max_capacity < 1 or max_capacity > self.MAX_CACHE_CAPACITY:
            raise ValueError(
                f"max_capacity must be a strict int between 1 and {self.MAX_CACHE_CAPACITY}, got {max_capacity!r}"
            )
        if (
            type(default_threshold) is bool
            or not isinstance(default_threshold, (int, float))
            or not math.isfinite(default_threshold)
            or default_threshold < 0.0
            or default_threshold > 1.0
        ):
            raise ValueError(f"Invalid default_threshold: {default_threshold!r}")

        self._provider = provider
        self._lock = threading.RLock()
        self._query_cache: dict[tuple[str, str, str], tuple[float, ...]] = {}
        self._document_cache: dict[tuple[str, str, str], tuple[float, ...]] = {}
        self._bound_identity: str | None = None
        self._minimum_similarity: float = float(default_threshold)
        self.max_capacity: int = max_capacity

        self.real_provider_query_calls: int = 0
        self.real_provider_doc_calls: int = 0
        self.real_provider_embedded_documents: int = 0
        self.query_cache_hits: int = 0
        self.doc_cache_hits: int = 0
        self.embed_call_count: int = 0
        self.last_query_vector: list[float] | None = None

        if callable(getattr(provider, "resolve_identity", None)):
            self.resolve_identity = self._resolve_identity

        # Lazy binding: constructor does not resolve or contact provider
        self._bound_identity = None

    @property
    def model_id(self) -> str:
        return getattr(self._provider, "model_id", "unknown")

    @property
    def max_input_bytes(self) -> int | None:
        return getattr(self._provider, "max_input_bytes", None)

    @property
    def expected_output_dimension(self) -> int | None:
        return getattr(self._provider, "expected_output_dimension", None)

    @property
    def minimum_similarity(self) -> float:
        return self._minimum_similarity

    @minimum_similarity.setter
    def minimum_similarity(self, value: float) -> None:
        if type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"Invalid minimum_similarity: {value!r}")
        self._minimum_similarity = float(value)

    @property
    def lock(self) -> Any:
        return (
            getattr(self._provider, "operation_lock", None)
            or getattr(self._provider, "lock", None)
            or getattr(self._provider, "_lock", None)
            or self._lock
        )

    @property
    def operation_lock(self) -> Any:
        return self.lock

    @property
    def query_template_version(self) -> str | None:
        return getattr(self._provider, "query_template_version", None)

    @property
    def calibration_version(self) -> str | None:
        return getattr(self._provider, "calibration_version", None)

    @property
    def cached_identity(self) -> str | None:
        if callable(getattr(self._provider, "resolve_identity", None)):
            return getattr(self._provider, "cached_identity", None) or self._bound_identity
        return getattr(self._provider, "cached_identity", None) or getattr(self._provider, "index_id", None) or self.model_id

    @property
    def index_id(self) -> str | None:
        if callable(getattr(self._provider, "resolve_identity", None)):
            return getattr(self._provider, "cached_identity", None) or self._bound_identity or getattr(self._provider, "index_id", None)
        return getattr(self._provider, "index_id", None) or self.cached_identity

    def _get_document_pipeline_fingerprint(self) -> str:
        profile = getattr(SearchIndex, "DOCUMENT_PROFILE", {})
        data = {
            "document_profile": profile,
            "dimension": self.expected_output_dimension,
            "max_input_bytes": self.max_input_bytes,
        }
        encoded = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _get_current_identity(self, profile: Any = None) -> str:
        if callable(getattr(self._provider, "resolve_identity", None)):
            target_profile = profile if profile is not None else SearchIndex.DOCUMENT_PROFILE
            if self._bound_identity is not None and profile is None:
                prov_cached = getattr(self._provider, "cached_identity", None)
                if prov_cached is not None and prov_cached != self._bound_identity:
                    return prov_cached
                return self._bound_identity
            res = self._provider.resolve_identity(target_profile)
            if not isinstance(res, str) or not res.strip():
                raise ProviderError("Provider resolve_identity returned empty or invalid identity")
            return res
        return getattr(self._provider, "cached_identity", None) or getattr(self._provider, "index_id", None) or getattr(self._provider, "model_id", "unknown")

    def _check_identity_bound(self, active_identity: str) -> None:
        if self._bound_identity is None:
            self._bound_identity = active_identity
        elif self._bound_identity != active_identity:
            # Model or profile changed: invalidate caches to prevent vector reuse across identities
            self._query_cache.clear()
            self._document_cache.clear()
            self._bound_identity = active_identity

    def _resolve_identity(self, document_profile: Any = None) -> str:
        with self.operation_lock:
            target_profile = document_profile if document_profile is not None else SearchIndex.DOCUMENT_PROFILE
            res = self._provider.resolve_identity(target_profile)
            if not isinstance(res, str) or not res.strip():
                raise ProviderError("Provider resolve_identity returned empty or invalid identity")
            self._check_identity_bound(res)
            return res

    def summarize(self, text: str) -> str:
        if callable(getattr(self._provider, "summarize", None)):
            return self._provider.summarize(text)
        raise AttributeError("Provider does not support topic summarization")

    def _validate_numeric_vector(self, vec: Any) -> tuple[float, ...]:
        if not isinstance(vec, (list, tuple)) or isinstance(vec, (str, bytes)):
            raise InvalidVectorError(f"Vector must be a numeric list or tuple, got {type(vec).__name__}")
        if len(vec) == 0:
            raise InvalidVectorError("Vector cannot be empty")
        if len(vec) > self.MAX_VECTOR_DIM:
            raise InvalidVectorError(f"Vector dimension {len(vec)} exceeds cap {self.MAX_VECTOR_DIM}")
        expected_dim = self.expected_output_dimension
        if expected_dim is not None and len(vec) != expected_dim:
            raise InvalidVectorError(f"Vector dimension {len(vec)} does not match expected {expected_dim}")
        cleaned = []
        for x in vec:
            if type(x) is bool or not isinstance(x, (int, float)):
                raise InvalidVectorError(f"Vector elements must be numbers (not bool), got {type(x).__name__} ({x!r})")
            fx = float(x)
            if not math.isfinite(fx):
                raise InvalidVectorError(f"Vector contains non-finite numbers: {fx!r}")
            cleaned.append(fx)
        norm_sq = sum(c * c for c in cleaned)
        if not math.isfinite(norm_sq) or norm_sq <= 0.0:
            raise InvalidVectorError(f"Vector has zero or invalid magnitude: norm_sq={norm_sq!r}")
        return tuple(cleaned)

    def _check_cache_capacity(self, additional: int = 1) -> None:
        if (len(self._query_cache) + len(self._document_cache) + additional) > self.max_capacity:
            raise CacheCapacityExceededError(
                f"Vector cache cap ({self.max_capacity}) exceeded. Finite memory cap reached."
            )

    def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
        self.embed_call_count += 1
        with self.operation_lock:
            active_id = self._get_current_identity()
            if index_id is not None and index_id != active_id:
                raise ProviderError(
                    f"Supplied index_id {index_id!r} does not match current known resolved identity {active_id!r}"
                )
            self._check_identity_bound(active_id)
            q_ver = self.query_template_version or "default"
            q_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
            cache_key = (active_id, q_ver, q_hash)

            if cache_key in self._query_cache:
                self.query_cache_hits += 1
                cached = self._query_cache[cache_key]
                norm = normalized_vector([float(x) for x in cached])
                self.last_query_vector = norm
                return [float(x) for x in cached]

            self.real_provider_query_calls += 1
            if callable(getattr(self._provider, "embed_query", None)):
                if callable_accepts_index_id(self._provider.embed_query):
                    raw = self._provider.embed_query(query, index_id=active_id)
                else:
                    raw = self._provider.embed_query(query)
            else:
                raw_list = self._provider.embed([query])
                if not isinstance(raw_list, list) or len(raw_list) != 1 or not isinstance(raw_list[0], (list, tuple)):
                    raise ProviderError("Provider did not return one embedding for the query")
                raw = raw_list[0]

            valid_tuple = self._validate_numeric_vector(raw)
            norm = normalized_vector([float(x) for x in valid_tuple])
            self._check_cache_capacity(additional=1)
            self._query_cache[cache_key] = valid_tuple
            self.last_query_vector = norm
            return [float(x) for x in valid_tuple]

    def embed_documents(self, texts: list[str], index_id: str | None = None) -> list[list[float]]:
        self.embed_call_count += 1
        with self.operation_lock:
            active_id = self._get_current_identity()
            if index_id is not None and index_id != active_id:
                raise ProviderError(
                    f"Supplied index_id {index_id!r} does not match current known resolved identity {active_id!r}"
                )
            self._check_identity_bound(active_id)
            doc_fp = self._get_document_pipeline_fingerprint()

            results: list[list[float] | None] = [None] * len(texts)
            unique_missing_texts: list[str] = []
            missing_text_to_indices: dict[str, list[int]] = {}
            missing_text_to_key: dict[str, tuple[str, str, str]] = {}

            for idx, text in enumerate(texts):
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                cache_key = (active_id, doc_fp, text_hash)
                if cache_key in self._document_cache:
                    self.doc_cache_hits += 1
                    results[idx] = [float(x) for x in self._document_cache[cache_key]]
                else:
                    if text not in missing_text_to_indices:
                        unique_missing_texts.append(text)
                        missing_text_to_indices[text] = [idx]
                        missing_text_to_key[text] = cache_key
                    else:
                        missing_text_to_indices[text].append(idx)

            if unique_missing_texts:
                self.real_provider_doc_calls += 1
                self.real_provider_embedded_documents += len(unique_missing_texts)
                if callable(getattr(self._provider, "embed_documents", None)):
                    if callable_accepts_index_id(self._provider.embed_documents):
                        raw_vecs = self._provider.embed_documents(unique_missing_texts, index_id=active_id)
                    else:
                        raw_vecs = self._provider.embed_documents(unique_missing_texts)
                else:
                    raw_vecs = self._provider.embed(unique_missing_texts)

                if not isinstance(raw_vecs, list) or len(raw_vecs) != len(unique_missing_texts):
                    raise ProviderError(f"Provider did not return one embedding per input: expected {len(unique_missing_texts)}")

                validated_tuples: list[tuple[float, ...]] = []
                for raw_vec in raw_vecs:
                    valid_tuple = self._validate_numeric_vector(raw_vec)
                    normalized_vector([float(x) for x in valid_tuple])
                    validated_tuples.append(valid_tuple)

                self._check_cache_capacity(additional=len(validated_tuples))

                for text, valid_tuple in zip(unique_missing_texts, validated_tuples):
                    cache_key = missing_text_to_key[text]
                    self._document_cache[cache_key] = valid_tuple
                    for orig_idx in missing_text_to_indices[text]:
                        results[orig_idx] = [float(x) for x in valid_tuple]

            return [r for r in results if r is not None]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) == 1:
            vec = self.embed_query(texts[0])
            return [vec]
        return self.embed_documents(texts)

    def reset_query_vector(self) -> None:
        self.last_query_vector = None


def _parse_and_validate_thresholds(
    thresholds: list[float] | None = None,
    args: Any = None,
) -> list[float]:
    raw_list: list[Any] = []
    if thresholds is not None:
        if not isinstance(thresholds, (list, tuple)):
            raise CalibrationError(f"Thresholds must be a list or tuple, got {type(thresholds).__name__}")
        raw_list = list(thresholds)
    elif args is not None and getattr(args, "thresholds", None) is not None:
        raw_t = args.thresholds
        if isinstance(raw_t, str):
            parts = [x.strip() for x in raw_t.split(",") if x.strip()]
            raw_list = parts
        elif isinstance(raw_t, (list, tuple)):
            raw_list = list(raw_t)
        else:
            raw_list = [raw_t]
    else:
        raw_list = [0.45, 0.50, 0.55, 0.60, 0.65]

    if len(raw_list) == 0:
        raise CalibrationError("Candidate thresholds list cannot be empty")
    if len(raw_list) > 21:
        raise CalibrationError(f"Candidate thresholds count {len(raw_list)} exceeds maximum allowed (21)")

    validated: list[float] = []
    for idx, item in enumerate(raw_list):
        if type(item) is bool:
            raise CalibrationError(f"Threshold at index {idx} must be a number (not bool), got {item!r}")
        if isinstance(item, str):
            try:
                val = float(item)
            except ValueError:
                raise CalibrationError(f"Invalid threshold string at index {idx}: {item!r}")
        elif isinstance(item, (int, float)):
            val = float(item)
        else:
            raise CalibrationError(f"Invalid threshold type at index {idx}: {type(item).__name__} ({item!r})")

        if not math.isfinite(val):
            raise CalibrationError(f"Threshold at index {idx} must be finite, got {val!r}")
        if val < 0.0 or val > 1.0:
            raise CalibrationError(f"Threshold at index {idx} ({val}) must be within [0.0, 1.0]")
        validated.append(val)

    return validated


def calibrate(
    args: Any = None,
    thresholds: list[float] | None = None,
    provider: Any = None,
    fixture_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run diagnostic passage calibration benchmark over candidate similarity thresholds.

    Calibration-only:
      - Validates original full fixture and disjointness before filtering.
      - Evaluates only calibration split positive and unrelated queries.
      - Caches document and query vectors across thresholds in memory.
      - Diagnostic only: does not auto-write provider defaults or calibration version.
      - Quality gates are partial-scope and cannot pass full quality/holdout gates.
    """
    candidate_thresholds = _parse_and_validate_thresholds(thresholds, args)

    f_path = fixture_path or getattr(args, "fixture", None) or (ROOT / "fixtures/passage-evaluation.json")
    fixture_bytes = Path(f_path).read_bytes()
    fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
    suite = json.loads(fixture_bytes.decode("utf-8"))

    # Strict pre-validation: rejects malformed suite before indexing
    validate_suite(suite)

    calib_pos_queries = [
        q for q in suite["queries"]
        if q.get("split") == "calibration" and q.get("type") == "positive"
    ]
    calib_gold_conv_keys = {
        (g.get("source_id", "chatgpt"), g.get("account_id", "default"), g["conversation_id"])
        for q in calib_pos_queries
        for g in q.get("expected_passages", [])
    }
    imported_conversations = [
        conv for conv in suite["conversations"]
        if (conv.get("source_id", "chatgpt"), conv.get("account_id", "default"), conv["id"]) in calib_gold_conv_keys
    ]

    raw_provider = provider or getattr(args, "provider", None)
    if raw_provider is None and getattr(args, "ollama", False):
        raw_provider = OllamaProvider(
            embedding_model=getattr(args, "model", "qwen3-embedding:0.6b"),
            timeout=getattr(args, "timeout", 30.0),
        )

    if raw_provider:
        caching_wrapper = raw_provider if isinstance(raw_provider, CachingProviderWrapper) else CachingProviderWrapper(raw_provider)
    else:
        caching_wrapper = None

    threshold_results = []
    all_structure_passed = True

    for t in candidate_thresholds:
        t_float = float(t)
        if caching_wrapper:
            caching_wrapper.minimum_similarity = t_float

        prev_calls = (caching_wrapper.real_provider_query_calls + caching_wrapper.real_provider_doc_calls) if caching_wrapper else 0
        prev_hits = (caching_wrapper.query_cache_hits + caching_wrapper.doc_cache_hits) if caching_wrapper else 0

        t0 = time.perf_counter()
        eval_args = argparse.Namespace(
            fixture=f_path,
            provider=caching_wrapper,
            ollama=False,
            model=getattr(args, "model", "qwen3-embedding:0.6b") if args else "qwen3-embedding:0.6b",
            timeout=getattr(args, "timeout", 30.0) if args else 30.0,
            require_quality=False,
            calibration_only=True,
        )

        eval_report = evaluate(eval_args, calibration_only=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        curr_calls = (caching_wrapper.real_provider_query_calls + caching_wrapper.real_provider_doc_calls) if caching_wrapper else 0
        curr_hits = (caching_wrapper.query_cache_hits + caching_wrapper.doc_cache_hits) if caching_wrapper else 0
        delta_calls = curr_calls - prev_calls
        delta_hits = curr_hits - prev_hits

        metrics = eval_report["metrics"]
        gates = eval_report["gates"]
        if not gates["structure_passed"]:
            all_structure_passed = False

        latency_type = "actual-inference" if delta_calls > 0 else "cached-sweep"

        threshold_entry = {
            "threshold": t_float,
            "metrics": {
                "passage_recall_at_3": metrics["passage_recall_at_3"],
                "passage_mrr_at_10": metrics["passage_mrr_at_10"],
                "message_recall_at_3": metrics["message_recall_at_3"],
                "message_mrr_at_10": metrics["message_mrr_at_10"],
                "conversation_recall_at_3": metrics["conversation_recall_at_3"],
                "conversation_mrr_at_10": metrics["conversation_mrr_at_10"],
                "unrelated_any_result_false_positive_rate": metrics["unrelated_any_result_false_positive_rate"],
                "unrelated_any_result_false_positive_count": metrics["unrelated_any_result_false_positive_count"],
                "unrelated_semantic_false_positive_rate": metrics["unrelated_semantic_false_positive_rate"],
                "expected_positive_count": metrics["positive_queries"],
                "completed_positive_count": len(eval_report["positive_results"]),
                "expected_unrelated_count": metrics["unrelated_queries"],
                "completed_unrelated_count": len(eval_report["unrelated_results"]),
                "context_identity_errors": metrics["context_identity_errors"],
            },
            "semantic": {
                "used": eval_report["semantic_evaluated"],
                "positive_semantic_used_count": metrics["positive_semantic_used_count"],
                "unrelated_semantic_used_count": metrics["unrelated_semantic_used_count"],
                "errors": [f["status"] for f in eval_report["semantic_failures"]] if eval_report.get("semantic_failures") else [],
            },
            "provider_calls": {
                "real_provider_calls": delta_calls,
                "cache_hits": delta_hits,
            },
            "latency": {
                "latency_type": latency_type,
                "elapsed_ms": round(elapsed_ms, 2),
                "median_query_ms": metrics["query_timings_ms"]["median"],
                "p95_query_ms": metrics["query_timings_ms"]["p95"],
            },
            "structure_passed": gates["structure_passed"],
        }
        threshold_results.append(threshold_entry)

    report = {
        "benchmark": "passage-calibration-v3",
        "synthetic": True,
        "fixture_hash": fixture_hash,
        "fixture_corpus_version": suite.get("version", "passage-evaluation-v3"),
        "mode": "ollama-hybrid" if caching_wrapper else "lexical-baseline-only",
        "run_scope": "calibration-only",
        "calibration_only": True,
        "candidate_thresholds": candidate_thresholds,
        "model_id": getattr(caching_wrapper, "index_id", None) or getattr(caching_wrapper, "cached_identity", None) or getattr(caching_wrapper, "model_id", None),
        "display_model_id": getattr(caching_wrapper, "model_id", None),
        "query_template_version": getattr(caching_wrapper, "query_template_version", None),
        "calibration_version": getattr(caching_wrapper, "calibration_version", None),
        "source_corpus_scope": {
            "conversations": len(imported_conversations),
            "messages": sum(len(c.get("messages", [])) for c in imported_conversations),
            "namespaces": sorted(list({f"{c.get('source_id', 'chatgpt')}:{c.get('account_id', 'default')}" for c in imported_conversations})),
        },
        "cache_stats": {
            "query_vectors_cached": len(caching_wrapper._query_cache) if caching_wrapper else 0,
            "document_vectors_cached": len(caching_wrapper._document_cache) if caching_wrapper else 0,
            "total_vectors_cached": (len(caching_wrapper._query_cache) + len(caching_wrapper._document_cache)) if caching_wrapper else 0,
            "capacity_limit": caching_wrapper.max_capacity if caching_wrapper else 1024,
            "real_provider_calls": (caching_wrapper.real_provider_query_calls + caching_wrapper.real_provider_doc_calls) if caching_wrapper else 0,
            "cache_hits": (caching_wrapper.query_cache_hits + caching_wrapper.doc_cache_hits) if caching_wrapper else 0,
        },
        "threshold_results": threshold_results,
        "gates": {
            "structure_passed": all_structure_passed,
            "quality_passed": False,
            "holdout_passed": False,
            "rules": {
                "calibration_only": True,
                "quality_gate_forbidden_for_partial_scope": True,
            },
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="Path to fixture JSON (default fixtures/passage-evaluation.json)")
    parser.add_argument("--ollama", action="store_true", help="Explicitly evaluate using local OllamaProvider")
    parser.add_argument("--model", default="qwen3-embedding:0.6b", help="Ollama embedding model name")
    parser.add_argument("--timeout", type=float, default=30.0, help="Ollama timeout in seconds (default 30)")
    parser.add_argument("--thresholds", type=str, default="0.45,0.50,0.55,0.60,0.65", help="Comma-separated candidate similarity thresholds")
    parser.add_argument("--output", type=Path, help="Optional output JSON report path")
    args = parser.parse_args()

    try:
        report = calibrate(args)
    except FixtureValidationError as exc:
        print(f"Fixture validation error: {exc}", file=sys.stderr)
        return 1
    except CalibrationError as exc:
        print(f"Calibration error: {exc}", file=sys.stderr)
        return 1

    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)

    if not report["gates"]["structure_passed"]:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
