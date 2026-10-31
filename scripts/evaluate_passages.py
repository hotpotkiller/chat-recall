#!/usr/bin/env python3
"""Passage-level evaluation benchmark and CLI runner.

Follows AGENTS.md:
- Synthetic evaluation fixture only, no private chats, credentials, or machine paths.
- Default mode tests offline lexical baseline without network calls or provider downloads.
- Opt-in --ollama mode evaluates OllamaProvider embedding on the synthetic suite.
- Rejects malformed suites and disjoint split violations BEFORE indexing or model execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat_recall.providers import OllamaProvider, ProviderError, callable_accepts_index_id, normalized_vector
from chat_recall.search import SearchIndex


class FixtureValidationError(ValueError):
    """Raised when a passage evaluation fixture fails structural or contract constraints."""
    pass


class RecordingProviderWrapper:
    """Thin wrapper around an EmbeddingProvider that captures query vectors and forwards provider attributes."""

    def __init__(self, provider):
        self._provider = provider
        self.last_query_vector: list[float] | None = None
        self.embed_call_count: int = 0
        if callable(getattr(provider, "resolve_identity", None)):
            self.resolve_identity = self._resolve_identity

    @property
    def model_id(self) -> str:
        return self._provider.model_id

    @property
    def max_input_bytes(self) -> int | None:
        return getattr(self._provider, "max_input_bytes", None)

    @property
    def minimum_similarity(self) -> float:
        return getattr(self._provider, "minimum_similarity", 0.3)

    @property
    def expected_output_dimension(self) -> int | None:
        return getattr(self._provider, "expected_output_dimension", None)

    @property
    def lock(self):
        return getattr(self._provider, "operation_lock", None) or getattr(self._provider, "lock", None) or getattr(self._provider, "_lock", None)

    @property
    def operation_lock(self):
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
            return getattr(self._provider, "cached_identity", None)
        return self._provider.model_id

    @property
    def index_id(self) -> str | None:
        if callable(getattr(self._provider, "resolve_identity", None)):
            return getattr(self._provider, "cached_identity", None) or getattr(self._provider, "index_id", None)
        return getattr(self._provider, "index_id", None) or self._provider.model_id

    def _resolve_identity(self, document_profile: Any = None) -> str:
        return self._provider.resolve_identity(document_profile)

    def summarize(self, text: str) -> str:
        if callable(getattr(self._provider, "summarize", None)):
            return self._provider.summarize(text)
        raise AttributeError("Provider does not support topic summarization")

    def embed_query(self, query: str, index_id: str | None = None) -> list[float]:
        self.embed_call_count += 1
        if callable(getattr(self._provider, "embed_query", None)):
            if callable_accepts_index_id(self._provider.embed_query):
                vector = self._provider.embed_query(query, index_id=index_id)
            else:
                vector = self._provider.embed_query(query)
        else:
            vectors = self._provider.embed([query])
            if not isinstance(vectors, list) or len(vectors) != 1 or not isinstance(vectors[0], (list, tuple)):
                raise ProviderError("Provider did not return one embedding for the query")
            vector = vectors[0]
        norm = normalized_vector(vector)
        self.last_query_vector = norm
        return norm

    def embed_documents(self, texts: list[str], index_id: str | None = None) -> list[list[float]]:
        self.embed_call_count += 1
        if callable(getattr(self._provider, "embed_documents", None)):
            if callable_accepts_index_id(self._provider.embed_documents):
                return self._provider.embed_documents(texts, index_id=index_id)
            return self._provider.embed_documents(texts)
        return self._provider.embed(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_call_count += 1
        vectors = self._provider.embed(texts)
        if len(texts) == 1 and vectors:
            self.last_query_vector = normalized_vector(vectors[0])
        return vectors

    def reset_query_vector(self) -> None:
        self.last_query_vector = None


def validate_suite(suite: dict[str, Any]) -> None:
    """Validate suite structural integrity, namespaces, bounds, types, and disjoint splits.

    Runs BEFORE indexing and model execution. Raises FixtureValidationError on failure.
    """
    if not isinstance(suite, dict):
        raise FixtureValidationError("Suite root must be a JSON object")

    if not suite.get("synthetic"):
        raise FixtureValidationError("Fixture must declare synthetic=true")

    conversations = suite.get("conversations")
    if not isinstance(conversations, list) or len(conversations) == 0:
        raise FixtureValidationError("Suite must contain a non-empty conversations list")

    queries = suite.get("queries")
    if not isinstance(queries, list) or len(queries) == 0:
        raise FixtureValidationError("Suite must contain a non-empty queries list")

    # Map conversations: (source_id, account_id, conversation_id) -> message_dict
    conv_map: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    conv_titles: dict[tuple[str, str, str], str] = {}
    all_conv_keys: set[tuple[str, str, str]] = set()

    for conv in conversations:
        if not isinstance(conv, dict):
            raise FixtureValidationError("Conversation entry must be an object")
        source_id = conv.get("source_id", "chatgpt")
        account_id = conv.get("account_id", "default")
        conv_id = conv.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise FixtureValidationError(f"Invalid source_id in conversation {conv}")
        if not isinstance(account_id, str) or not account_id:
            raise FixtureValidationError(f"Invalid account_id in conversation {conv}")
        if not isinstance(conv_id, str) or not conv_id:
            raise FixtureValidationError(f"Invalid conversation id in conversation {conv}")

        key = (source_id, account_id, conv_id)
        if key in all_conv_keys:
            raise FixtureValidationError(f"Duplicate conversation identity {key}")
        all_conv_keys.add(key)

        messages = conv.get("messages")
        if not isinstance(messages, list) or len(messages) == 0:
            raise FixtureValidationError(f"Conversation {conv_id} must have a non-empty messages list")

        msg_map: dict[str, dict[str, Any]] = {}
        for msg in messages:
            if not isinstance(msg, dict):
                raise FixtureValidationError(f"Message in {conv_id} must be an object")
            msg_id = msg.get("id")
            if not isinstance(msg_id, str) or not msg_id:
                raise FixtureValidationError(f"Invalid message id in conversation {conv_id}")
            if msg_id in msg_map:
                raise FixtureValidationError(f"Duplicate message id {msg_id} in conversation {conv_id}")
            text = msg.get("text")
            if not isinstance(text, str):
                raise FixtureValidationError(f"Message {msg_id} must have a text string")
            role = msg.get("role")
            if role not in ("user", "assistant"):
                raise FixtureValidationError(
                    f"Message {msg_id} in {conv_id} role must be 'user' or 'assistant', got {role!r}"
                )
            msg_map[msg_id] = msg

        # Validate parent_id links within the conversation and enforce acyclicity
        for msg_id, msg in msg_map.items():
            pid = msg.get("parent_id")
            if pid is not None:
                if not isinstance(pid, str) or pid not in msg_map:
                    raise FixtureValidationError(f"Message {msg_id} parent_id '{pid}' not found in conversation {conv_id}")
            # Detect cycles
            cur = pid
            seen_ancestors = {msg_id}
            while cur is not None:
                if cur in seen_ancestors:
                    raise FixtureValidationError(
                        f"Cycle detected in parent links of conversation {conv_id} at message {cur}"
                    )
                seen_ancestors.add(cur)
                parent_msg = msg_map.get(cur)
                if parent_msg is None:
                    break
                cur = parent_msg.get("parent_id")

        conv_map[key] = msg_map
        conv_titles[key] = conv.get("title", "")

    # Validate queries, splits, expected_passages, bounds, ancestors, and disjointness
    calibration_convs: set[tuple[str, str, str]] = set()
    holdout_convs: set[tuple[str, str, str]] = set()
    query_ids: set[str] = set()

    for q in queries:
        if not isinstance(q, dict):
            raise FixtureValidationError("Query entry must be an object")
        qid = q.get("id")
        if not isinstance(qid, str) or not qid:
            raise FixtureValidationError(f"Query id must be a non-empty string: {q}")
        if qid in query_ids:
            raise FixtureValidationError(f"Duplicate query id: {qid}")
        query_ids.add(qid)

        split = q.get("split")
        if split not in ("calibration", "holdout"):
            raise FixtureValidationError(f"Query {qid} split must be 'calibration' or 'holdout', got {split}")

        qtype = q.get("type")
        if qtype not in ("positive", "unrelated"):
            raise FixtureValidationError(f"Query {qid} type must be 'positive' or 'unrelated', got {qtype}")

        qtext = q.get("query")
        if not isinstance(qtext, str) or not qtext.strip():
            raise FixtureValidationError(f"Query {qid} must have a non-empty query string")

        # Validate query filters if present: only 'account_id' and 'role' are permitted
        filters = q.get("filters")
        if filters is not None:
            if not isinstance(filters, dict):
                raise FixtureValidationError(f"Query {qid} filters must be an object if present")
            for f_key, f_val in filters.items():
                if f_key not in ("account_id", "role"):
                    raise FixtureValidationError(
                        f"Query {qid} contains unknown or disallowed filter key: '{f_key}'. Only 'account_id' and 'role' are allowed."
                    )
                if f_key == "account_id":
                    if not isinstance(f_val, str) or not f_val.strip():
                        raise FixtureValidationError(f"Query {qid} filter 'account_id' must be a non-empty string: {f_val!r}")
                elif f_key == "role":
                    if f_val not in ("user", "assistant"):
                        raise FixtureValidationError(f"Query {qid} filter 'role' must be 'user' or 'assistant': {f_val!r}")

        if qtype == "unrelated":
            if "expected_passages" in q and q["expected_passages"]:
                raise FixtureValidationError(f"Unrelated query {qid} must not specify expected_passages")
            continue

        # Positive query validation
        golds = q.get("expected_passages")
        if not isinstance(golds, list) or len(golds) == 0:
            raise FixtureValidationError(f"Positive query {qid} must have a non-empty expected_passages list")

        for gold in golds:
            if not isinstance(gold, dict):
                raise FixtureValidationError(f"Gold passage in query {qid} must be an object")
            src = gold.get("source_id", "chatgpt")
            acc = gold.get("account_id", "default")
            cid = gold.get("conversation_id")
            mid = gold.get("message_id")
            if not isinstance(src, str) or not src:
                raise FixtureValidationError(f"Gold passage in {qid} missing source_id")
            if not isinstance(acc, str) or not acc:
                raise FixtureValidationError(f"Gold passage in {qid} missing account_id")
            if not isinstance(cid, str) or not cid:
                raise FixtureValidationError(f"Gold passage in {qid} missing conversation_id")
            if not isinstance(mid, str) or not mid:
                raise FixtureValidationError(f"Gold passage in {qid} missing message_id")

            gold_key = (src, acc, cid)
            if gold_key not in conv_map:
                raise FixtureValidationError(f"Gold conversation {gold_key} in query {qid} does not exist in conversations")
            if mid not in conv_map[gold_key]:
                raise FixtureValidationError(f"Gold message {mid} in query {qid} does not exist in {gold_key}")

            msg_obj = conv_map[gold_key][mid]
            msg_text = msg_obj["text"]

            # Validate gold agrees with query filters before models
            if filters:
                req_acc = filters.get("account_id")
                req_role = filters.get("role")
                if req_acc is not None and acc != req_acc:
                    raise FixtureValidationError(
                        f"Gold passage in query {qid} account_id '{acc}' does not match query filter account_id '{req_acc}'"
                    )
                if req_role is not None and msg_obj.get("role") != req_role:
                    raise FixtureValidationError(
                        f"Gold message {mid} in query {qid} role '{msg_obj.get('role')}' does not match query filter role '{req_role}'"
                    )

            start_off = gold.get("start_offset")
            end_off = gold.get("end_offset")
            exp_text = gold.get("expected_text")
            if not isinstance(start_off, int) or isinstance(start_off, bool) or start_off < 0:
                raise FixtureValidationError(f"Invalid start_offset in gold passage of {qid}: {start_off}")
            if not isinstance(end_off, int) or isinstance(end_off, bool):
                raise FixtureValidationError(f"Invalid end_offset in gold passage of {qid}: {end_off}")
            if end_off <= start_off:
                raise FixtureValidationError(f"Invalid end_offset in gold passage of {qid}: {end_off} <= {start_off}")
            if end_off > len(msg_text):
                raise FixtureValidationError(
                    f"Gold span [{start_off}:{end_off}] exceeds message length {len(msg_text)} for message {mid} in {qid}"
                )

            # Validate expected_text matches slice if supplied (no zero-span exemption)
            if exp_text is not None:
                if not isinstance(exp_text, str) or not exp_text:
                    raise FixtureValidationError(f"Gold expected_text in query {qid} must be a non-empty string: {exp_text!r}")
                actual_slice = msg_text[start_off:end_off]
                if actual_slice != exp_text:
                    raise FixtureValidationError(
                        f"Gold expected_text does not match slice [{start_off}:{end_off}] in message {mid} for query {qid}: {exp_text!r} vs {actual_slice!r}"
                    )

            # Optional expected_ancestor_ids validation
            ancestors = gold.get("expected_ancestor_ids")
            if ancestors is not None:
                if not isinstance(ancestors, list):
                    raise FixtureValidationError(f"expected_ancestor_ids in {qid} must be a list if present")
                # Walk the true message ancestors
                cur_pid = msg_obj.get("parent_id")
                true_ancestors = []
                seen_walk = set()
                while cur_pid is not None and cur_pid not in seen_walk:
                    seen_walk.add(cur_pid)
                    true_ancestors.append(cur_pid)
                    parent_msg = conv_map[gold_key].get(cur_pid)
                    if parent_msg is None:
                        break
                    cur_pid = parent_msg.get("parent_id")

                for anc in ancestors:
                    if not isinstance(anc, str) or anc not in true_ancestors:
                        raise FixtureValidationError(
                            f"Gold ancestor {anc} for message {mid} in {qid} is not a true ancestor ({true_ancestors})"
                        )

            # Record split conversations
            if split == "calibration":
                calibration_convs.add(gold_key)
            else:
                holdout_convs.add(gold_key)

    # Disjoint split conversation identities validation
    split_overlap = calibration_convs.intersection(holdout_convs)
    if split_overlap:
        raise FixtureValidationError(
            f"Disjoint split violation: conversations appear in both calibration and holdout: {split_overlap}"
        )


def score_results(results: list[dict[str, Any]], golds: list[dict[str, Any]]) -> dict[str, int | None]:
    """Score raw search results against positive gold expected_passages.

    Requires explicit identities (no namespace defaults).
    Requires type(focus_offset) is int (rejects missing, None, bool, float).
    Matches passage rank using exclusive end: start_offset <= focus_offset < end_offset.

    Returns dict with keys:
      - passage_rank: 1-based rank of first raw result matching full identity AND focus_offset within gold span, or None
      - message_rank: 1-based rank of first raw result matching full message identity, or None
      - conversation_rank: 1-based rank of first raw result matching full conversation identity, or None
    """
    if not golds:
        return {"passage_rank": None, "message_rank": None, "conversation_rank": None}

    passage_rank = None
    message_rank = None
    conversation_rank = None

    for rank_idx, r in enumerate(results, start=1):
        r_src = r.get("source_id")
        r_acc = r.get("account_id")
        r_cid = r.get("conversation_id")
        r_mid = r.get("message_id")
        r_focus = r.get("focus_offset")

        for gold in golds:
            g_src = gold.get("source_id")
            g_acc = gold.get("account_id")
            g_cid = gold.get("conversation_id")
            g_mid = gold.get("message_id")
            g_start = gold.get("start_offset", 0)
            g_end = gold.get("end_offset", 0)

            # Conversation match: requires explicit matching namespace
            if (
                conversation_rank is None
                and (r_src, r_acc, r_cid) == (g_src, g_acc, g_cid)
                and r_src is not None and r_acc is not None and r_cid is not None
            ):
                conversation_rank = rank_idx

            # Message match: requires explicit matching namespace and message
            if (
                message_rank is None
                and (r_src, r_acc, r_cid, r_mid) == (g_src, g_acc, g_cid, g_mid)
                and r_src is not None and r_acc is not None and r_cid is not None and r_mid is not None
            ):
                message_rank = rank_idx

            # Passage match: requires explicit identity, type(focus_offset) is int, and start <= focus < end
            if (
                passage_rank is None
                and (r_src, r_acc, r_cid, r_mid) == (g_src, g_acc, g_cid, g_mid)
                and r_src is not None and r_acc is not None and r_cid is not None and r_mid is not None
            ):
                if type(r_focus) is int and g_start <= r_focus < g_end:
                    passage_rank = rank_idx

        if passage_rank is not None and message_rank is not None and conversation_rank is not None:
            break

    return {
        "passage_rank": passage_rank,
        "message_rank": message_rank,
        "conversation_rank": conversation_rank,
    }


def validate_context_structure(
    ctx: dict[str, Any],
    expected_src: str | None,
    expected_acc: str | None,
    expected_cid: str | None,
    expected_mid: str | None,
    conv_map: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
) -> list[str]:
    """Validate context namespace fields, original selected message text, parent links, and ancestor chain.

    Ensures:
      - All namespace fields match expected identity.
      - Original selected message text, role, and parent match the synthetic fixture.
      - Exact ancestor-prefix sequence through the selected message matches the fixture, with NO sibling nodes.
      - Actual parent links in prefix form a valid unbroken chain.
      - Any messages following the target are valid deterministic descendants along the chosen path.
    """
    errors: list[str] = []

    if ctx.get("source_id") != expected_src:
        errors.append(f"Context source_id mismatch: {ctx.get('source_id')!r} != {expected_src!r}")
    if ctx.get("account_id") != expected_acc:
        errors.append(f"Context account_id mismatch: {ctx.get('account_id')!r} != {expected_acc!r}")
    if ctx.get("conversation_id") != expected_cid:
        errors.append(f"Context conversation_id mismatch: {ctx.get('conversation_id')!r} != {expected_cid!r}")
    if ctx.get("message_id") != expected_mid:
        errors.append(f"Context message_id mismatch: {ctx.get('message_id')!r} != {expected_mid!r}")

    key = (expected_src or "", expected_acc or "", expected_cid or "")
    if key not in conv_map:
        errors.append(f"Conversation {key} not found in fixture conv_map")
        return errors

    msg_dict = conv_map[key]
    if expected_mid not in msg_dict:
        errors.append(f"Target message {expected_mid} not found in fixture conversation {key}")
        return errors

    # Build true ancestor chain from fixture
    target_fixture_msg = msg_dict[expected_mid]
    ancestors = []
    cur_p = target_fixture_msg.get("parent_id")
    seen_walk = set()
    while cur_p is not None:
        if cur_p in seen_walk or cur_p not in msg_dict:
            break
        seen_walk.add(cur_p)
        ancestors.append(cur_p)
        cur_p = msg_dict[cur_p].get("parent_id")
    ancestors.reverse()
    expected_prefix_ids = ancestors + [expected_mid]

    ctx_messages = ctx.get("messages")
    if not isinstance(ctx_messages, list) or not ctx_messages:
        errors.append("Context messages list is missing or empty")
        return errors

    target_idx = None
    for idx, m in enumerate(ctx_messages):
        if m.get("id") == expected_mid:
            target_idx = idx
            break

    if target_idx is None:
        errors.append(f"Target message {expected_mid} not found in context messages")
        return errors

    # Exact ancestor-prefix sequence validation (rejects sibling nodes in prefix)
    actual_prefix_ids = [m.get("id") for m in ctx_messages[:target_idx + 1]]
    if actual_prefix_ids != expected_prefix_ids:
        errors.append(
            f"Context prefix sequence {actual_prefix_ids} != expected ancestor chain {expected_prefix_ids}"
        )

    # Validate text, role, parent_id, and links for all prefix messages against fixture
    for idx, m in enumerate(ctx_messages[:target_idx + 1]):
        mid = m.get("id")
        if mid not in msg_dict:
            errors.append(f"Prefix message {mid} not in fixture")
            continue
        orig = msg_dict[mid]
        if m.get("text") != orig.get("text"):
            errors.append(f"Prefix message {mid} text mismatch")
        if m.get("role") != orig.get("role"):
            errors.append(f"Prefix message {mid} role mismatch")
        if m.get("parent_id") != orig.get("parent_id"):
            errors.append(f"Prefix message {mid} parent_id mismatch: {m.get('parent_id')!r} vs {orig.get('parent_id')!r}")
        if idx > 0:
            prev_id = ctx_messages[idx - 1].get("id")
            if m.get("parent_id") != prev_id:
                errors.append(f"Prefix message {mid} does not link to previous message {prev_id}")

    # Validate deterministic descendants AFTER target
    for idx in range(target_idx + 1, len(ctx_messages)):
        m = ctx_messages[idx]
        mid = m.get("id")
        prev_id = ctx_messages[idx - 1].get("id")
        if mid not in msg_dict:
            errors.append(f"Descendant message {mid} not in fixture")
            continue
        orig = msg_dict[mid]
        if m.get("text") != orig.get("text"):
            errors.append(f"Descendant message {mid} text mismatch")
        if m.get("role") != orig.get("role"):
            errors.append(f"Descendant message {mid} role mismatch")
        if m.get("parent_id") != orig.get("parent_id"):
            errors.append(f"Descendant message {mid} parent_id mismatch")
        if m.get("parent_id") != prev_id:
            errors.append(f"Descendant message {mid} does not link to prior message {prev_id}")

    return errors


def validate_search_response_schema(resp: Any) -> list[str]:
    """Validate that a search response satisfies basic schema requirements.

    Returns a list of error descriptions (empty if valid).
    Schema rules:
      - resp must be a dict
      - 'total' must be exact int (not bool), and total >= 0
      - 'results' must be a list
      - len(results) <= total
    """
    errors: list[str] = []
    if not isinstance(resp, dict):
        return [f"Search response must be a dict, got {type(resp).__name__}"]

    total = resp.get("total")
    if total is None:
        errors.append("Search response missing 'total'")
    elif type(total) is not int:
        errors.append(f"Search response 'total' must be exact int (not bool), got {type(total).__name__} ({total!r})")
    elif total < 0:
        errors.append(f"Search response 'total' must be non-negative, got {total}")

    results = resp.get("results")
    if results is None:
        errors.append("Search response missing 'results'")
    elif not isinstance(results, list):
        errors.append(f"Search response 'results' must be a list, got {type(results).__name__}")
    else:
        if type(total) is int and total >= 0:
            if len(results) > total:
                errors.append(f"Search response len(results) ({len(results)}) exceeds total ({total})")

    return errors


def validate_hit_integrity(
    hit: dict[str, Any],
    conv_map: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
) -> list[str]:
    """Validate that displayed hit text and role match fixture original,
    and focus_offset is a valid int within that original message.
    """
    errors: list[str] = []
    if not isinstance(hit, dict):
        return ["Hit result must be an object"]

    src = hit.get("source_id", "chatgpt")
    acc = hit.get("account_id", "default")
    cid = hit.get("conversation_id")
    mid = hit.get("message_id")

    key = (src, acc, cid)
    if key not in conv_map:
        errors.append(f"Hit conversation {key} not found in fixture")
        return errors

    msg_dict = conv_map[key]
    if mid not in msg_dict:
        errors.append(f"Hit message {mid} not found in fixture conversation {key}")
        return errors

    orig_msg = msg_dict[mid]
    orig_text = orig_msg.get("text", "")
    orig_role = orig_msg.get("role")

    hit_text = hit.get("text")
    if hit_text != orig_text:
        exp_preview = repr(orig_text)[:50]
        hit_preview = repr(hit_text)[:50]
        errors.append(
            f"Hit text mismatch for message {mid}: expected {exp_preview}..., got {hit_preview}..."
        )

    hit_role = hit.get("role")
    if hit_role != orig_role:
        errors.append(
            f"Hit role mismatch for message {mid}: expected {orig_role!r}, got {hit_role!r}"
        )

    hit_focus = hit.get("focus_offset")
    if type(hit_focus) is not int:
        errors.append(
            f"Hit focus_offset must be an exact int (not bool/missing), got {type(hit_focus).__name__} ({hit_focus!r})"
        )
    else:
        if len(orig_text) == 0:
            if hit_focus != 0:
                errors.append(f"Hit focus_offset {hit_focus} must be 0 for empty message")
        elif not (0 <= hit_focus <= len(orig_text)):
            errors.append(
                f"Hit focus_offset {hit_focus} out of bounds for message length {len(orig_text)}"
            )

    return errors


def evaluate(args: Any = None, calibration_only: bool | None = None) -> dict[str, Any]:
    """Run passage-level evaluation benchmark."""
    if calibration_only is not None:
        if type(calibration_only) is not bool:
            raise TypeError(f"calibration_only must be a bool or None, got {type(calibration_only).__name__}")
        is_calibration_only = (calibration_only is True)
    elif args is not None:
        calib_attr = getattr(args, "calibration_only", False)
        is_calibration_only = (calib_attr is True)
    else:
        is_calibration_only = False

    if args is not None:
        rq_attr = getattr(args, "require_quality", False)
        require_quality = (rq_attr is True)
    else:
        require_quality = False

    if is_calibration_only and require_quality:
        raise ValueError("--require-quality cannot be combined with --calibration-only")

    fixture_path = getattr(args, "fixture", None) or (ROOT / "fixtures/passage-evaluation.json")
    fixture_bytes = Path(fixture_path).read_bytes()
    fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
    suite = json.loads(fixture_bytes.decode("utf-8"))

    # Strict pre-validation: rejects malformed fixture and disjoint split violations BEFORE models/indexing
    validate_suite(suite)

    if is_calibration_only:
        calib_pos_queries = [
            q for q in suite["queries"]
            if q.get("split") == "calibration" and q.get("type") == "positive"
        ]
        calib_gold_conv_keys = {
            (g.get("source_id", "chatgpt"), g.get("account_id", "default"), g["conversation_id"])
            for q in calib_pos_queries
            for g in q.get("expected_passages", [])
        }
        conversations_to_import = [
            conv for conv in suite["conversations"]
            if (conv.get("source_id", "chatgpt"), conv.get("account_id", "default"), conv["id"]) in calib_gold_conv_keys
        ]
        queries_to_run = [
            q for q in suite["queries"]
            if q.get("split") == "calibration"
        ]
    else:
        conversations_to_import = suite["conversations"]
        queries_to_run = suite["queries"]

    # Build reference conversation map for context checking
    conv_map: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for conv in conversations_to_import:
        src = conv.get("source_id", "chatgpt")
        acc = conv.get("account_id", "default")
        cid = conv["id"]
        conv_map[(src, acc, cid)] = {m["id"]: m for m in conv.get("messages", [])}

    # Provider configuration
    raw_provider = getattr(args, "provider", None)
    if raw_provider is None and getattr(args, "ollama", False):
        raw_provider = OllamaProvider(
            embedding_model=getattr(args, "model", "qwen3-embedding:0.6b"),
            timeout=getattr(args, "timeout", 30.0),
        )

    if raw_provider:
        provider = raw_provider if isinstance(raw_provider, RecordingProviderWrapper) else RecordingProviderWrapper(raw_provider)
    else:
        provider = None

    with tempfile.TemporaryDirectory(prefix="chat-recall-passage-eval-") as directory:
        db_path = Path(directory) / "passage_eval.sqlite"
        with SearchIndex(db_path, provider) as index:
            # 1. Group conversations by (source_id, account_id) for import_payload
            convs_by_ns: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for conv in conversations_to_import:
                src = conv.get("source_id", "chatgpt")
                acc = conv.get("account_id", "default")
                convs_by_ns.setdefault((src, acc), []).append(conv)

            combined_import_report = {
                "conversations": 0, "messages": 0, "skipped": 0, "errors": [],
                "namespaces": [],
            }
            import_errors = []
            for (src, acc), conv_list in convs_by_ns.items():
                rep = index.import_payload({"conversations": conv_list}, source_id=src, account_id=acc)
                combined_import_report["conversations"] += rep.get("conversations", 0)
                combined_import_report["messages"] += rep.get("messages", 0)
                combined_import_report["skipped"] += rep.get("skipped", 0)
                combined_import_report["namespaces"].append(f"{src}:{acc}")
                if rep.get("errors"):
                    combined_import_report["errors"].extend(rep["errors"])
                    import_errors.extend(rep["errors"])
            import_report = combined_import_report

            # 2. Enrich when provider is present
            enrichment_report = None
            enrichment_errors = []
            if provider:
                enrichment_report = index.enrich(limit=10000)
                enrichment_errors = list(enrichment_report.get("errors", []))

            queries = queries_to_run
            positive_queries = [q for q in queries if q["type"] == "positive"]
            unrelated_queries = [q for q in queries if q["type"] == "unrelated"]

            positive_results = []
            unrelated_results = []
            query_errors = []
            semantic_failures = []
            query_timings_ms = []
            context_identity_errors = 0
            collected_minimum_similarity = None

            # 3. Evaluate positive queries
            for item in positive_queries:
                q_text = item["query"]
                q_id = item["id"]
                q_split = item["split"]
                golds = item["expected_passages"]
                q_filters = item.get("filters") or {}

                if provider:
                    provider.reset_query_vector()

                t0 = time.perf_counter()
                try:
                    search_kwargs: dict[str, Any] = {}
                    if "account_id" in q_filters:
                        search_kwargs["account_id"] = q_filters["account_id"]
                    if "role" in q_filters:
                        search_kwargs["role"] = q_filters["role"]

                    search_resp = index.search(q_text, mode="smart", limit=50, **search_kwargs)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    query_timings_ms.append(elapsed)

                    schema_errs = validate_search_response_schema(search_resp)
                    if schema_errs:
                        for s_err in schema_errs:
                            query_errors.append({
                                "query_id": q_id,
                                "query": q_text,
                                "mode": "smart",
                                "error": f"Search response schema error: {s_err}",
                            })

                    sem_info = search_resp.get("semantic", {})
                    sem_used = bool(sem_info.get("used"))
                    sem_status = sem_info.get("status")
                    if "minimum_similarity" in sem_info and sem_info["minimum_similarity"] is not None:
                        collected_minimum_similarity = sem_info["minimum_similarity"]

                    if not sem_used and provider is not None:
                        semantic_failures.append({
                            "query_id": q_id,
                            "query": q_text,
                            "type": "positive",
                            "status": sem_status,
                            "errors": sem_info.get("errors", []),
                        })

                    raw_results = (
                        search_resp.get("results", [])
                        if isinstance(search_resp, dict) and isinstance(search_resp.get("results"), list)
                        else []
                    )
                    ranks = score_results(raw_results, golds)

                    # Context and hit integrity verification: validate all top 10 returned hits against fixture,
                    # including non-gold results and query filter scope compliance
                    for hit_result in raw_results[:10]:
                        h_src = hit_result.get("source_id")
                        h_acc = hit_result.get("account_id")
                        h_cid = hit_result.get("conversation_id")
                        h_mid = hit_result.get("message_id")
                        h_role = hit_result.get("role")

                        # Validate hit matches requested query filters
                        if "account_id" in q_filters and h_acc != q_filters["account_id"]:
                            context_identity_errors += 1
                        if "role" in q_filters and h_role != q_filters["role"]:
                            context_identity_errors += 1

                        # Validate displayed text, role, and focus_offset against fixture original
                        hit_errors = validate_hit_integrity(hit_result, conv_map)
                        if hit_errors:
                            context_identity_errors += len(hit_errors)

                        try:
                            ctx = index.context(
                                conversation_id=h_cid,
                                message_id=h_mid,
                                source_id=h_src,
                                account_id=h_acc,
                            )
                            c_errs = validate_context_structure(ctx, h_src, h_acc, h_cid, h_mid, conv_map)
                            if c_errs:
                                context_identity_errors += len(c_errs)
                        except Exception:
                            context_identity_errors += 1

                    # If gold passage hit is beyond top 10, also validate its hit integrity, context, and filter compliance
                    if ranks["passage_rank"] is not None and ranks["passage_rank"] > 10:
                        hit_result = raw_results[ranks["passage_rank"] - 1]
                        h_src = hit_result.get("source_id")
                        h_acc = hit_result.get("account_id")
                        h_cid = hit_result.get("conversation_id")
                        h_mid = hit_result.get("message_id")
                        h_role = hit_result.get("role")

                        if "account_id" in q_filters and h_acc != q_filters["account_id"]:
                            context_identity_errors += 1
                        if "role" in q_filters and h_role != q_filters["role"]:
                            context_identity_errors += 1

                        hit_errors = validate_hit_integrity(hit_result, conv_map)
                        if hit_errors:
                            context_identity_errors += len(hit_errors)

                        try:
                            ctx = index.context(
                                conversation_id=h_cid,
                                message_id=h_mid,
                                source_id=h_src,
                                account_id=h_acc,
                            )
                            c_errs = validate_context_structure(ctx, h_src, h_acc, h_cid, h_mid, conv_map)
                            if c_errs:
                                context_identity_errors += len(c_errs)
                        except Exception:
                            context_identity_errors += 1

                    # For any returned gold passage, also validate explicit expected_ancestor_ids if defined
                    if ranks["passage_rank"] is not None:
                        hit_result = raw_results[ranks["passage_rank"] - 1]
                        for gold in golds:
                            if (
                                gold.get("source_id") == hit_result.get("source_id")
                                and gold.get("account_id") == hit_result.get("account_id")
                                and gold.get("conversation_id") == hit_result.get("conversation_id")
                                and gold.get("message_id") == hit_result.get("message_id")
                            ):
                                expected_anc = gold.get("expected_ancestor_ids")
                                if expected_anc:
                                    try:
                                        ctx = index.context(
                                            conversation_id=hit_result["conversation_id"],
                                            message_id=hit_result["message_id"],
                                            source_id=hit_result.get("source_id"),
                                            account_id=hit_result.get("account_id"),
                                        )
                                        ctx_mids = [m["id"] for m in ctx.get("messages", [])]
                                        for anc_id in expected_anc:
                                            if anc_id not in ctx_mids:
                                                context_identity_errors += 1
                                    except Exception:
                                        context_identity_errors += 1
                                break

                    positive_results.append({
                        "id": q_id,
                        "split": q_split,
                        "query": q_text,
                        "ranks": ranks,
                        "elapsed_ms": round(elapsed, 2),
                        "semantic_used": sem_used,
                        "semantic_status": sem_status,
                    })
                except Exception as exc:
                    query_errors.append({"query_id": q_id, "query": q_text, "mode": "smart", "error": str(exc)})

            # 4. Evaluate unrelated queries
            for item in unrelated_queries:
                q_text = item["query"]
                q_id = item["id"]
                q_split = item["split"]
                q_filters = item.get("filters") or {}

                if provider:
                    provider.reset_query_vector()

                t0 = time.perf_counter()
                try:
                    search_kwargs = {}
                    if "account_id" in q_filters:
                        search_kwargs["account_id"] = q_filters["account_id"]
                    if "role" in q_filters:
                        search_kwargs["role"] = q_filters["role"]

                    search_resp = index.search(q_text, mode="smart", limit=50, **search_kwargs)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    query_timings_ms.append(elapsed)

                    schema_errs = validate_search_response_schema(search_resp)
                    if schema_errs:
                        for s_err in schema_errs:
                            query_errors.append({
                                "query_id": q_id,
                                "query": q_text,
                                "mode": "smart",
                                "error": f"Search response schema error: {s_err}",
                            })

                    sem_info = search_resp.get("semantic", {})
                    sem_used = bool(sem_info.get("used"))
                    sem_status = sem_info.get("status")
                    if "minimum_similarity" in sem_info and sem_info["minimum_similarity"] is not None:
                        collected_minimum_similarity = sem_info["minimum_similarity"]

                    if not sem_used and provider is not None:
                        semantic_failures.append({
                            "query_id": q_id,
                            "query": q_text,
                            "type": "unrelated",
                            "status": sem_status,
                            "errors": sem_info.get("errors", []),
                        })

                    raw_unrel_results = (
                        search_resp.get("results", [])
                        if isinstance(search_resp, dict) and isinstance(search_resp.get("results"), list)
                        else []
                    )

                    # Context and hit integrity verification for any top 10 results returned by unrelated queries
                    for hit_result in raw_unrel_results[:10]:
                        h_src = hit_result.get("source_id")
                        h_acc = hit_result.get("account_id")
                        h_cid = hit_result.get("conversation_id")
                        h_mid = hit_result.get("message_id")
                        h_role = hit_result.get("role")

                        # Validate hit matches requested query filters
                        if "account_id" in q_filters and h_acc != q_filters["account_id"]:
                            context_identity_errors += 1
                        if "role" in q_filters and h_role != q_filters["role"]:
                            context_identity_errors += 1

                        # Validate displayed text, role, and focus_offset against fixture original
                        hit_errors = validate_hit_integrity(hit_result, conv_map)
                        if hit_errors:
                            context_identity_errors += len(hit_errors)

                        try:
                            ctx = index.context(
                                conversation_id=h_cid,
                                message_id=h_mid,
                                source_id=h_src,
                                account_id=h_acc,
                            )
                            c_errs = validate_context_structure(ctx, h_src, h_acc, h_cid, h_mid, conv_map)
                            if c_errs:
                                context_identity_errors += len(c_errs)
                        except Exception:
                            context_identity_errors += 1

                    has_semantic_result = any(
                        r.get("match_type") in ("semantic", "hybrid")
                        for r in raw_unrel_results
                    )
                    total_val = search_resp.get("total") if isinstance(search_resp, dict) else None
                    valid_positive_total = type(total_val) is int and total_val > 0
                    has_any_result = bool(raw_unrel_results) or valid_positive_total

                    unrelated_results.append({
                        "id": q_id,
                        "split": q_split,
                        "query": q_text,
                        "semantic_used": sem_used,
                        "semantic_status": sem_status,
                        "total_results": total_val if type(total_val) is int and total_val >= 0 else 0,
                        "has_semantic_result": has_semantic_result,
                        "has_any_result": has_any_result,
                        "elapsed_ms": round(elapsed, 2),
                    })
                except Exception as exc:
                    query_errors.append({"query_id": q_id, "query": q_text, "mode": "smart", "error": str(exc)})

            # Denominators MUST include failures and dropped queries (full expected suite query count)
            expected_pos_total = len(positive_queries)
            expected_unrel_total = len(unrelated_queries)
            expected_queries_total = expected_pos_total + expected_unrel_total

            completed_pos_count = len(positive_results)
            completed_unrel_count = len(unrelated_results)
            total_semantic_used = (
                sum(1 for r in positive_results if r.get("semantic_used"))
                + sum(1 for r in unrelated_results if r.get("semantic_used"))
            )

            # Aggregate passage metrics across positive queries (denom = expected_pos_total)
            passage_top3_hits = sum(1 for r in positive_results if r["ranks"]["passage_rank"] is not None and r["ranks"]["passage_rank"] <= 3)
            passage_mrr10_sum = sum(1.0 / r["ranks"]["passage_rank"] for r in positive_results if r["ranks"]["passage_rank"] is not None and r["ranks"]["passage_rank"] <= 10)

            message_top3_hits = sum(1 for r in positive_results if r["ranks"]["message_rank"] is not None and r["ranks"]["message_rank"] <= 3)
            message_mrr10_sum = sum(1.0 / r["ranks"]["message_rank"] for r in positive_results if r["ranks"]["message_rank"] is not None and r["ranks"]["message_rank"] <= 10)

            conversation_top3_hits = sum(1 for r in positive_results if r["ranks"]["conversation_rank"] is not None and r["ranks"]["conversation_rank"] <= 3)
            conversation_mrr10_sum = sum(1.0 / r["ranks"]["conversation_rank"] for r in positive_results if r["ranks"]["conversation_rank"] is not None and r["ranks"]["conversation_rank"] <= 10)

            passage_recall_at_3 = passage_top3_hits / expected_pos_total if expected_pos_total else 0.0
            passage_mrr_at_10 = passage_mrr10_sum / expected_pos_total if expected_pos_total else 0.0

            message_recall_at_3 = message_top3_hits / expected_pos_total if expected_pos_total else 0.0
            message_mrr_at_10 = message_mrr10_sum / expected_pos_total if expected_pos_total else 0.0

            conversation_recall_at_3 = conversation_top3_hits / expected_pos_total if expected_pos_total else 0.0
            conversation_mrr_at_10 = conversation_mrr10_sum / expected_pos_total if expected_pos_total else 0.0

            # Unrelated false positive rates (denom = expected_unrel_total)
            unrel_semantic_fps = sum(1 for r in unrelated_results if r["has_semantic_result"])
            unrel_any_fps = sum(1 for r in unrelated_results if r["has_any_result"])
            unrelated_semantic_fp_rate = unrel_semantic_fps / expected_unrel_total if expected_unrel_total else 0.0
            unrelated_any_result_fp_rate = unrel_any_fps / expected_unrel_total if expected_unrel_total else 0.0

            # Split metrics
            split_results = {}
            split_names = ("calibration",) if is_calibration_only else ("calibration", "holdout")
            for sp in split_names:
                sp_expected_pos = len([q for q in positive_queries if q.get("split") == sp])
                sp_expected_unrel = len([q for q in unrelated_queries if q.get("split") == sp])

                sp_pos = [r for r in positive_results if r.get("split") == sp]
                sp_unrel = [r for r in unrelated_results if r.get("split") == sp]

                sp_pass_top3 = sum(1 for r in sp_pos if r["ranks"]["passage_rank"] is not None and r["ranks"]["passage_rank"] <= 3)
                sp_pass_mrr_sum = sum(1.0 / r["ranks"]["passage_rank"] for r in sp_pos if r["ranks"]["passage_rank"] is not None and r["ranks"]["passage_rank"] <= 10)

                sp_msg_top3 = sum(1 for r in sp_pos if r["ranks"]["message_rank"] is not None and r["ranks"]["message_rank"] <= 3)
                sp_msg_mrr_sum = sum(1.0 / r["ranks"]["message_rank"] for r in sp_pos if r["ranks"]["message_rank"] is not None and r["ranks"]["message_rank"] <= 10)

                sp_conv_top3 = sum(1 for r in sp_pos if r["ranks"]["conversation_rank"] is not None and r["ranks"]["conversation_rank"] <= 3)
                sp_conv_mrr_sum = sum(1.0 / r["ranks"]["conversation_rank"] for r in sp_pos if r["ranks"]["conversation_rank"] is not None and r["ranks"]["conversation_rank"] <= 10)

                sp_sem_fps = sum(1 for r in sp_unrel if r["has_semantic_result"])
                sp_any_fps = sum(1 for r in sp_unrel if r["has_any_result"])

                sp_pos_sem_used = sum(1 for r in sp_pos if r["semantic_used"])
                sp_unrel_sem_used = sum(1 for r in sp_unrel if r["semantic_used"])

                split_results[sp] = {
                    "expected_positive_count": sp_expected_pos,
                    "completed_positive_count": len(sp_pos),
                    "positive_semantic_used_count": sp_pos_sem_used,
                    "passage_recall_at_3": round(sp_pass_top3 / sp_expected_pos, 4) if sp_expected_pos else 0.0,
                    "passage_mrr_at_10": round(sp_pass_mrr_sum / sp_expected_pos, 4) if sp_expected_pos else 0.0,
                    "message_recall_at_3": round(sp_msg_top3 / sp_expected_pos, 4) if sp_expected_pos else 0.0,
                    "message_mrr_at_10": round(sp_msg_mrr_sum / sp_expected_pos, 4) if sp_expected_pos else 0.0,
                    "conversation_recall_at_3": round(sp_conv_top3 / sp_expected_pos, 4) if sp_expected_pos else 0.0,
                    "conversation_mrr_at_10": round(sp_conv_mrr_sum / sp_expected_pos, 4) if sp_expected_pos else 0.0,
                    "expected_unrelated_count": sp_expected_unrel,
                    "completed_unrelated_count": len(sp_unrel),
                    "unrelated_semantic_used_count": sp_unrel_sem_used,
                    "unrelated_semantic_false_positive_rate": round(sp_sem_fps / sp_expected_unrel, 4) if sp_expected_unrel else 0.0,
                    "unrelated_any_result_false_positive_rate": round(sp_any_fps / sp_expected_unrel, 4) if sp_expected_unrel else 0.0,
                }

            sorted_timings = sorted(query_timings_ms) if query_timings_ms else [0.0]
            median_ms = round(statistics.median(sorted_timings), 3)
            p95_idx = min(len(sorted_timings) - 1, math.ceil(len(sorted_timings) * 0.95) - 1)
            p95_ms = round(sorted_timings[p95_idx], 3)

            semantic_evaluated = (total_semantic_used > 0)

            # Gates
            structure_passed = (
                len(import_errors) == 0
                and len(query_errors) == 0
                and context_identity_errors == 0
                and completed_pos_count == expected_pos_total
                and completed_unrel_count == expected_unrel_total
            )

            if is_calibration_only:
                holdout_passed = False
                quality_passed = False
            else:
                holdout_metrics = split_results.get("holdout", {})
                sp_holdout_expected_pos = holdout_metrics.get("expected_positive_count", 0)
                sp_holdout_expected_unrel = holdout_metrics.get("expected_unrelated_count", 0)

                holdout_passed = (
                    structure_passed
                    and len(enrichment_errors) == 0
                    and len(semantic_failures) == 0
                    and semantic_evaluated
                    and sp_holdout_expected_pos > 0
                    and sp_holdout_expected_unrel > 0
                    and holdout_metrics.get("completed_positive_count") == sp_holdout_expected_pos
                    and holdout_metrics.get("completed_unrelated_count") == sp_holdout_expected_unrel
                    and holdout_metrics.get("positive_semantic_used_count") == sp_holdout_expected_pos
                    and holdout_metrics.get("unrelated_semantic_used_count") == sp_holdout_expected_unrel
                    and holdout_metrics.get("passage_recall_at_3", 0.0) >= 0.85
                    and holdout_metrics.get("passage_mrr_at_10", 0.0) >= 0.80
                    and holdout_metrics.get("unrelated_any_result_false_positive_rate", 1.0) <= 0.20
                )

                quality_passed = (
                    semantic_evaluated
                    and total_semantic_used == expected_queries_total
                    and len(semantic_failures) == 0
                    and len(enrichment_errors) == 0
                    and len(import_errors) == 0
                    and len(query_errors) == 0
                    and context_identity_errors == 0
                    and passage_recall_at_3 >= 0.85
                    and passage_mrr_at_10 >= 0.80
                    and unrelated_any_result_fp_rate <= 0.20
                    and holdout_passed
                )

            report = {
                "benchmark": "passage-evaluation-v3",
                "synthetic": True,
                "fixture_hash": fixture_hash,
                "fixture_corpus_version": suite.get("version", "passage-evaluation-v3"),
                "mode": "ollama-hybrid" if provider else "lexical-baseline-only",
                "run_scope": "calibration-only" if is_calibration_only else "full",
                "calibration_only": is_calibration_only,
                "semantic_evaluated": semantic_evaluated,
                "model_id": getattr(provider, "index_id", None) or getattr(provider, "cached_identity", None) or getattr(provider, "model_id", None),
                "display_model_id": getattr(provider, "model_id", None),
                "query_template_version": getattr(provider, "query_template_version", None),
                "calibration_version": getattr(provider, "calibration_version", None),
                "minimum_similarity": collected_minimum_similarity,
                "limits_and_disclaimer": "Synthetic-only benchmark with small bounded corpus. Passing metrics do not establish personal user history relevance or quality.",
                "import": import_report,
                "enrichment": enrichment_report,
                "metrics": {
                    "expected_queries_total": expected_queries_total,
                    "completed_queries_total": completed_pos_count + completed_unrel_count,
                    "semantic_used_count": total_semantic_used,
                    "positive_queries": expected_pos_total,
                    "positive_semantic_used_count": sum(1 for r in positive_results if r["semantic_used"]),
                    "passage_recall_at_3": round(passage_recall_at_3, 4),
                    "passage_mrr_at_10": round(passage_mrr_at_10, 4),
                    "message_recall_at_3": round(message_recall_at_3, 4),
                    "message_mrr_at_10": round(message_mrr_at_10, 4),
                    "conversation_recall_at_3": round(conversation_recall_at_3, 4),
                    "conversation_mrr_at_10": round(conversation_mrr_at_10, 4),
                    "unrelated_queries": expected_unrel_total,
                    "unrelated_semantic_used_count": sum(1 for r in unrelated_results if r["semantic_used"]),
                    "unrelated_semantic_false_positive_rate": round(unrelated_semantic_fp_rate, 4),
                    "unrelated_any_result_false_positive_rate": round(unrelated_any_result_fp_rate, 4),
                    "unrelated_any_result_false_positive_count": unrel_any_fps,
                    "context_identity_errors": context_identity_errors,
                    "split_results": split_results,
                    "query_timings_ms": {
                        "median": median_ms,
                        "p95": p95_ms,
                    },
                },
                "gates": {
                    "structure_passed": structure_passed,
                    "quality_passed": quality_passed,
                    "holdout_passed": holdout_passed,
                    "rules": {
                        "require_all_queries_semantic_used": True,
                        "min_passage_recall_at_3": 0.85,
                        "min_passage_mrr_at_10": 0.80,
                        "max_unrelated_any_result_fp_rate": 0.20,
                        "max_holdout_unrelated_any_result_fp_rate": 0.20,
                        "max_context_identity_errors": 0,
                        "max_import_errors": 0,
                        "max_enrichment_errors": 0,
                        "max_query_errors": 0,
                        "max_semantic_failures": 0,
                    },
                },
                "semantic_failures": semantic_failures,
                "query_errors": query_errors,
                "positive_results": positive_results,
                "unrelated_results": unrelated_results,
            }
            return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="Path to fixture JSON (default fixtures/passage-evaluation.json)")
    parser.add_argument("--ollama", action="store_true", help="Explicitly evaluate using local OllamaProvider")
    parser.add_argument("--model", default="qwen3-embedding:0.6b", help="Ollama embedding model name")
    parser.add_argument("--timeout", type=float, default=30.0, help="Ollama timeout in seconds (default 30)")
    parser.add_argument("--require-quality", action="store_true", help="Require semantic quality gates to pass (requires --ollama)")
    parser.add_argument("--calibration-only", action="store_true", help="Run calibration queries only without exposing holdout conversations or queries")
    parser.add_argument("--output", type=Path, help="Optional output JSON report path")
    args = parser.parse_args()

    if args.require_quality and not args.ollama:
        parser.error("--require-quality requires --ollama")
    if args.require_quality and args.calibration_only:
        parser.error("--require-quality cannot be combined with --calibration-only")

    try:
        report = evaluate(args)
    except FixtureValidationError as exc:
        print(f"Fixture validation error: {exc}", file=sys.stderr)
        return 1

    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)

    if not report["gates"]["structure_passed"]:
        return 1

    if args.require_quality and not report["gates"]["quality_passed"]:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
