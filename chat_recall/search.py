"""SQLite source archive, verified literal retrieval and optional hybrid search.

No model, network, or download is needed for importing, exact search, lexical
ranking, or passage viewing. Semantic indexing is explicitly invoked by enrich.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from .providers import EmbeddingProvider, ProviderError, callable_accepts_index_id, normalized_vector

MAX_ENRICH_SOURCE_BYTES = 262144  # 256 KiB
MAX_ENRICH_WINDOWS = 512
DEFAULT_MAX_TIME_SECONDS = 120.0
DEFAULT_MAX_REQUESTS = 128

_BM25_FUNCTION_WORDS = frozenset({
    "and", "or", "the", "a", "an", "of", "to", "in", "on", "at", "for",
    "from", "with", "by", "is", "are", "was", "were", "be", "been", "it",
    "its", "this", "that", "these", "those", "i", "we", "you", "my", "your",
    "our", "how", "what", "when", "where", "why", "do", "did", "does", "can",
    "could", "would", "should", "please", "me", "about",
})

_CJK_FUNCTION_PHRASES = tuple(sorted({
    "为什么", "怎么样", "怎么回事", "怎样才能",
    "怎么办", "有些什么", "是什么",
    "什么", "怎么", "怎样", "如何", "哪个", "哪些", "这个", "那个",
    "这些", "那些", "可以", "应该", "如果", "但是", "而且", "或者", "以及",
    "因为", "所以", "虽然", "尽管", "关于", "对于", "一下", "一个", "有些",
    "这么", "那么", "这样", "那样", "是否", "能否", "可否",
}, key=len, reverse=True))


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2FA1F
        or 0x30000 <= cp <= 0x323AF
        or 0x3040 <= cp <= 0x309F
        or 0x30A0 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
    )


def cjk_lexical_terms(text: str) -> list[str]:
    """Extract bounded >=2-character CJK concepts and non-CJK keywords for query matching.

    Pure, testable helper. Removes longest declared function phrases BEFORE generating
    CJK ngrams with word boundaries that cannot bridge removed phrases.
    Avoids indiscriminate singleton Chinese-character terms that would flood retrieval.
    """
    if not text:
        return []

    terms: list[str] = []
    seen: set[str] = set()

    # 1. Non-CJK alphanumeric tokens and mixed identifiers
    non_cjk_clean = "".join(" " if _is_cjk(c) else c for c in text)
    for word in re.findall(r"[^\W_]+(?:[_\-:.]+[^\W_]+)*", non_cjk_clean, re.UNICODE):
        wl = word.lower()
        if wl not in _BM25_FUNCTION_WORDS and word not in seen:
            seen.add(word)
            terms.append(word)
        if any(sep in word for sep in "_-:."):
            for sub in re.findall(r"[^\W_]+", word, re.UNICODE):
                sub_l = sub.lower()
                if sub_l not in _BM25_FUNCTION_WORDS and sub not in seen:
                    seen.add(sub)
                    terms.append(sub)

    # 2. Remove longest declared function phrases BEFORE generating CJK ngrams
    # Replace with space boundaries so CJK ngrams cannot bridge across removed phrases
    cleaned_cjk = text
    for phrase in _CJK_FUNCTION_PHRASES:
        if phrase in cleaned_cjk:
            cleaned_cjk = cleaned_cjk.replace(phrase, " ")

    # Contiguous CJK character runs -> bigrams & trigrams (>=2-character concepts)
    cjk_runs: list[str] = []
    current_run: list[str] = []
    for ch in cleaned_cjk:
        if _is_cjk(ch):
            current_run.append(ch)
        else:
            if current_run:
                cjk_runs.append("".join(current_run))
                current_run = []
    if current_run:
        cjk_runs.append("".join(current_run))

    for run in cjk_runs:
        run_len = len(run)
        if run_len >= 2:
            for i in range(run_len - 1):
                bg = run[i:i + 2]
                if bg not in seen:
                    seen.add(bg)
                    terms.append(bg)
        if run_len >= 3:
            for i in range(run_len - 2):
                tg = run[i:i + 3]
                if tg not in seen:
                    seen.add(tg)
                    terms.append(tg)

    return terms[:40]


def cjk_index_tokens(text: str) -> list[str]:
    """Extract index tokens for derived FTS5 message_cjk table.

    Includes non-CJK alphanumeric words and >=2-character CJK bigrams and trigrams.
    """
    if not text:
        return []

    tokens: list[str] = []

    # Non-CJK words and mixed identifiers
    non_cjk_clean = "".join(" " if _is_cjk(c) else c for c in text)
    for word in re.findall(r"[^\W_]+(?:[_\-:.]+[^\W_]+)*", non_cjk_clean, re.UNICODE):
        tokens.append(word)
        if any(sep in word for sep in "_-:."):
            for sub in re.findall(r"[^\W_]+", word, re.UNICODE):
                tokens.append(sub)

    # Contiguous CJK character runs -> bigrams & trigrams
    cjk_runs: list[str] = []
    current_run: list[str] = []
    for ch in text:
        if _is_cjk(ch):
            current_run.append(ch)
        else:
            if current_run:
                cjk_runs.append("".join(current_run))
                current_run = []
    if current_run:
        cjk_runs.append("".join(current_run))

    for run in cjk_runs:
        run_len = len(run)
        if run_len >= 2:
            for i in range(run_len - 1):
                tokens.append(run[i:i + 2])
        if run_len >= 3:
            for i in range(run_len - 2):
                tokens.append(run[i:i + 3])

    return tokens


def literal_matches(text: str, query: str, case_sensitive: bool = False) -> list[list[int]]:
    """Find overlapping occurrences, returning offsets into untouched text.

    A folded character may expand (ß -> ss). Mapping both ends back to source
    code points avoids broken highlights. Multiple folded hits mapping to one
    original occurrence are collapsed.
    """
    if not query:
        return []
    if case_sensitive:
        haystack, needle, offsets = text, query, None
    else:
        pieces, offsets = [], []
        for index, char in enumerate(text):
            folded = char.casefold()
            pieces.append(folded)
            offsets.extend([index] * len(folded))
        haystack, needle = "".join(pieces), query.casefold()
    if not needle:
        return []
    matches, seen, start = [], set(), 0
    while (found := haystack.find(needle, start)) >= 0:
        end = found + len(needle)
        span = (found, end) if offsets is None else (offsets[found], offsets[end - 1] + 1)
        if span not in seen:
            matches.append(list(span))
            seen.add(span)
        start = found + 1
    return matches


def _stamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            return None
    return value if isinstance(value, str) else None


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError(f"{label} must be a non-empty string of at most 1024 characters")
    return value


def _hash(*values: str) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def _validate_graph(lookup: dict[str, tuple]) -> None:
    verified = set()
    for node in lookup:
        path, current = set(), node
        while current in lookup and current not in verified:
            if current in path:
                raise ValueError("Conversation parent graph contains a cycle")
            path.add(current)
            current = lookup[current][0]
        verified.update(path)


def _visible_parent(node_id: str, lookup: dict[str, tuple]) -> str | None:
    """Resolve source node ancestry to the nearest visible native message ID."""
    parent = lookup[node_id][0]
    seen = {node_id}
    while parent in lookup:
        if parent in seen:
            raise ValueError("Conversation parent graph contains a cycle")
        seen.add(parent)
        next_parent, parent_message = lookup[parent]
        if parent_message is not None:
            return parent_message
        parent = next_parent
    return parent


def _explicitly_hidden(message: Any) -> bool:
    """An explicit internal message supersedes an older visible node.

    A missing/null placeholder is not a deletion instruction in a partial
    import. Actual source metadata declaring content internal is authoritative.
    """
    if not isinstance(message, dict):
        return False
    author = message.get("author") or {}
    role = message.get("role") or (author.get("role") if isinstance(author, dict) else None)
    metadata = message.get("metadata") or {}
    metadata = metadata if isinstance(metadata, dict) else {}
    content = message.get("content") or {}
    content = content if isinstance(content, dict) else {}
    return (
        (role is not None and role not in {"user", "assistant"})
        or bool(metadata.get("is_visually_hidden_from_conversation"))
        or bool(metadata.get("is_thinking_preamble_message"))
        or message.get("channel") in {"analysis", "justify", "confidence"}
        or content.get("content_type") in {"thoughts", "reasoning_recap", "user_editable_context", "model_editable_context"}
    )


def _visible_text(message: dict) -> tuple[str | None, int]:
    metadata = message.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    if (
        metadata.get("is_visually_hidden_from_conversation")
        or metadata.get("is_thinking_preamble_message")
        or message.get("channel") in {"analysis", "justify", "confidence"}
    ):
        return None, 1
    if "text" in message and isinstance(message["text"], str):
        return message["text"], 0
    content = message.get("content")
    if not isinstance(content, dict):
        return None, 1
    kind = content.get("content_type", "text")
    if kind in {"code", "audio_transcription"} and isinstance(content.get("text"), str):
        return content["text"], 0
    if kind not in {"text", "multimodal_text"}:
        return None, 1
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None, 1
    text_parts, skipped = [], 0
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict) and part.get("content_type") in {"text", "audio_transcription"} and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        else:
            skipped += 1
    return ("\n".join(text_parts) if text_parts else None), skipped


def _safe_utf8_prefix(blob: bytes, max_bytes: int) -> str:
    """Decode up to max_bytes UTF-8 preserving embedded NULs without trailing partial sequence."""
    if not blob:
        return ""
    sub = blob[:max_bytes] if len(blob) > max_bytes else blob
    for trim in range(min(4, len(sub)) + 1):
        cand = sub[:len(sub) - trim] if trim else sub
        try:
            return cand.decode("utf-8")
        except UnicodeDecodeError:
            continue
    return sub.decode("utf-8", errors="replace")


def _safe_utf8_suffix(blob: bytes, max_bytes: int = 8000) -> str:
    """Decode up to max_bytes trailing suffix preserving embedded NULs without leading partial sequence."""
    if not blob:
        return ""
    sub = blob[-max_bytes:] if len(blob) > max_bytes else blob
    for trim in range(min(4, len(sub)) + 1):
        cand = sub[trim:] if trim else sub
        try:
            return cand.decode("utf-8")
        except UnicodeDecodeError:
            continue
    return sub.decode("utf-8", errors="replace")


class SearchIndex:
    _safe_utf8_prefix = staticmethod(_safe_utf8_prefix)
    _safe_utf8_suffix = staticmethod(_safe_utf8_suffix)
    MAX_ENRICH_SOURCE_BYTES = MAX_ENRICH_SOURCE_BYTES
    MAX_ENRICH_WINDOWS = MAX_ENRICH_WINDOWS

    DOCUMENT_PROFILE = {
        "version": "v2",
        "max_input_bytes": 960,
        "prefix_bytes": 384,
        "max_chunk_bytes": 512,
        "overlap_max": 64,
        "overlap_ratio": 4,
    }

    def __init__(self, db_path: str | Path, embedding_provider: EmbeddingProvider | None = None):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.embedding_provider = embedding_provider
        self._schema()

    def _schema(self) -> None:
        had_trigrams = self.connection.execute("SELECT 1 FROM sqlite_master WHERE name='message_tri'").fetchone() is not None
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                source_id TEXT NOT NULL, account_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL, title TEXT NOT NULL,
                created_at TEXT, updated_at TEXT, current_node TEXT, raw_json TEXT,
                PRIMARY KEY(source_id,account_id,conversation_id)
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, source_id TEXT NOT NULL,
                account_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                message_id TEXT NOT NULL, node_id TEXT NOT NULL, parent_id TEXT,
                role TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT,
                content_hash TEXT NOT NULL,
                UNIQUE(source_id,account_id,conversation_id,message_id),
                FOREIGN KEY(source_id,account_id,conversation_id)
                  REFERENCES conversations(source_id,account_id,conversation_id)
            );
            CREATE INDEX IF NOT EXISTS messages_namespace ON messages(source_id,account_id,conversation_id);
            CREATE INDEX IF NOT EXISTS messages_with_nul ON messages(id) WHERE instr(CAST(text AS BLOB), X'00') > 0;
            CREATE TABLE IF NOT EXISTS nodes (
                source_id TEXT NOT NULL, account_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL, node_id TEXT NOT NULL,
                parent_node TEXT, message_id TEXT,
                PRIMARY KEY(source_id,account_id,conversation_id,node_id)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
                text, tokenize='unicode61 remove_diacritics 0'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS message_tri USING fts5(
                text, tokenize='trigram case_sensitive 1'
            );
            CREATE TABLE IF NOT EXISTS message_cjk_segments (
                segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_row INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_cjk_segments_msg ON message_cjk_segments(message_row);
            CREATE VIRTUAL TABLE IF NOT EXISTS message_cjk USING fts5(
                text, tokenize='unicode61 remove_diacritics 0'
            );
            CREATE TABLE IF NOT EXISTS vectors (
                message_row INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                channel TEXT NOT NULL, chunk INTEGER NOT NULL,
                model_id TEXT NOT NULL, content_hash TEXT NOT NULL,
                start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL,
                derived_text TEXT NOT NULL, vector_json TEXT NOT NULL,
                PRIMARY KEY(message_row,channel,chunk,model_id)
            );
            CREATE TABLE IF NOT EXISTS enrichment (
                message_row INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                model_id TEXT NOT NULL, content_hash TEXT NOT NULL,
                topics INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(message_row,model_id)
            );
        """)
        if not had_trigrams:
            self.connection.executemany("INSERT INTO message_tri(rowid,text) VALUES (?,?)", (
                (r["id"], r["text"].casefold()) for r in self.connection.execute("SELECT id,text FROM messages")))
        cjk_ready = self.connection.execute("SELECT value FROM index_meta WHERE key='message_cjk_ready'").fetchone()
        if cjk_ready is None or cjk_ready[0] != "ready":
            # Rebuild / populate message_cjk atomically without touching source messages or vectors
            self.connection.execute("DELETE FROM message_cjk")
            self.connection.execute("DELETE FROM message_cjk_segments")
            msg_ids = [r[0] for r in self.connection.execute("SELECT id FROM messages ORDER BY id")]
            for mid in msg_ids:
                self._index_cjk_message(mid)
            self.connection.execute(
                "INSERT INTO index_meta(key, value) VALUES ('message_cjk_ready', 'ready') "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
            )
        self.connection.commit()

    def _index_cjk_message(self, message_row: int) -> None:
        """Stream chunks of a message via SQLite blob substr to index into derived message_cjk.

        NUL-safe, bounded memory streaming without O(N^2) character offset scans.
        Preserves complete Unicode (including astral and NUL), boundary bigrams/trigrams,
        and mixed-language identifiers.
        """
        self.connection.execute(
            "DELETE FROM message_cjk WHERE rowid IN (SELECT segment_id FROM message_cjk_segments WHERE message_row=?)",
            (message_row,)
        )
        self.connection.execute("DELETE FROM message_cjk_segments WHERE message_row=?", (message_row,))

        len_row = self.connection.execute(
            "SELECT length(CAST(text AS BLOB)) FROM messages WHERE id=?",
            (message_row,)
        ).fetchone()
        if len_row is None or not len_row[0]:
            return
        total_bytes = len_row[0]

        chunk_bytes = 4096
        byte_offset = 1  # SQLite 1-based indexing for substr
        tail_prefix = ""

        while byte_offset <= total_bytes:
            raw_blob = self.connection.execute(
                "SELECT substr(CAST(text AS BLOB), ?, ?) FROM messages WHERE id=?",
                (byte_offset, chunk_bytes, message_row)
            ).fetchone()[0]
            if not raw_blob:
                break

            actual_len = len(raw_blob)
            is_eof = (byte_offset + actual_len > total_bytes)

            if is_eof:
                valid_bytes = raw_blob
                consumed_bytes = actual_len
            else:
                valid_len = actual_len
                while valid_len > actual_len - 4 and valid_len > 0:
                    try:
                        raw_blob[:valid_len].decode("utf-8", errors="surrogatepass")
                        break
                    except UnicodeDecodeError:
                        valid_len -= 1
                else:
                    valid_len = actual_len
                valid_bytes = raw_blob[:valid_len]
                consumed_bytes = valid_len

            chunk_text = tail_prefix + valid_bytes.decode("utf-8", errors="replace")

            tokens = cjk_index_tokens(chunk_text)
            if tokens:
                cur = self.connection.execute(
                    "INSERT INTO message_cjk_segments(message_row) VALUES (?)",
                    (message_row,)
                )
                seg_id = cur.lastrowid
                self.connection.execute(
                    "INSERT INTO message_cjk(rowid, text) VALUES (?, ?)",
                    (seg_id, " ".join(tokens))
                )

            if is_eof:
                break

            tail_len = min(128, len(chunk_text))
            tail_prefix = chunk_text[-tail_len:] if tail_len > 0 else ""
            byte_offset += consumed_bytes

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def import_payload(self, payload: Any, source_id: str = "chatgpt", account_id: str = "default") -> dict:
        _identifier(source_id, "source_id")
        _identifier(account_id, "account_id")
        if isinstance(payload, dict) and "conversations" in payload:
            items = payload["conversations"]
        elif isinstance(payload, dict):
            items = [payload]
        else:
            items = payload
        if not isinstance(items, list):
            raise ValueError("Import must contain a conversation object or list")
        report = {"conversations": 0, "messages": 0, "skipped": 0,
                  "errors": [], "source_id": source_id, "account_id": account_id}
        for position, item in enumerate(items):
            self.connection.execute("SAVEPOINT import_one")
            try:
                if not isinstance(item, dict):
                    raise ValueError("Conversation must be an object")
                conv_id = _identifier(item.get("id") or item.get("conversation_id"), "conversation id")
                title = item.get("title", "Untitled conversation")
                if not isinstance(title, str):
                    raise ValueError("Conversation title must be a string")
                parsed, graph, skipped = self._parse(item)
                explicit_hidden_nodes = set()
                placeholder_nodes = set()
                if isinstance(item.get("mapping"), dict):
                    explicit_hidden_nodes = {node_id for node_id, node in item["mapping"].items()
                        if isinstance(node, dict) and _explicitly_hidden(node.get("message"))}
                    placeholder_nodes = {node_id for node_id, node in item["mapping"].items()
                        if isinstance(node, dict) and node.get("message") is None}
                elif isinstance(item.get("messages"), list):
                    explicit_hidden_nodes = {message.get("id", f"message-{i}") for i, message in enumerate(item["messages"])
                        if _explicitly_hidden(message)}
                namespace = (source_id, account_id, conv_id)
                prior_conv = self.connection.execute(
                    "SELECT title FROM conversations WHERE source_id=? AND account_id=? AND conversation_id=?", namespace
                ).fetchone()
                self.connection.execute("""INSERT INTO conversations VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(source_id,account_id,conversation_id) DO UPDATE SET
                    title=excluded.title, updated_at=excluded.updated_at,
                    current_node=COALESCE(excluded.current_node,conversations.current_node),
                    raw_json=excluded.raw_json""", (*namespace, title,
                    _stamp(item.get("create_time") or item.get("created_at")),
                    _stamp(item.get("update_time") or item.get("updated_at")),
                    item.get("current_node"), json.dumps(item, ensure_ascii=False)))
                changed = prior_conv is None or prior_conv["title"] != title
                for node_id, parent_node, message_id in graph:
                    prior = self.connection.execute("""SELECT parent_node,message_id FROM nodes
                        WHERE source_id=? AND account_id=? AND conversation_id=? AND node_id=?""", (*namespace, node_id)).fetchone()
                    if prior is not None and node_id in placeholder_nodes:
                        # A sparse graph can mention an old node without its
                        # message. Preserve both its identity and ancestry.
                        parent_node, message_id = prior["parent_node"], prior["message_id"]
                    if prior is None or tuple(prior) != (parent_node, message_id):
                        changed = True
                    self.connection.execute("""INSERT INTO nodes VALUES (?,?,?,?,?,?)
                        ON CONFLICT(source_id,account_id,conversation_id,node_id) DO UPDATE SET
                        parent_node=excluded.parent_node,message_id=excluded.message_id""",
                        (*namespace, node_id, parent_node, message_id))
                merged_graph = {r["node_id"]: (r["parent_node"], r["message_id"])
                    for r in self.connection.execute("SELECT * FROM nodes WHERE source_id=? AND account_id=? AND conversation_id=?", namespace)}
                _validate_graph(merged_graph)
                # Absence in a partial import never removes records. Explicit
                # internal/hidden updates, however, must revoke old searchable
                # text and its semantic derivatives in every search channel.
                for previous_message in list(self.connection.execute("SELECT id,node_id,message_id FROM messages WHERE source_id=? AND account_id=? AND conversation_id=?", namespace)):
                    replacement = merged_graph.get(previous_message["node_id"], (None, None))[1]
                    if previous_message["node_id"] in explicit_hidden_nodes or (replacement is not None and replacement != previous_message["message_id"]):
                        changed = True
                        for table in ("message_fts", "message_tri"):
                            self.connection.execute(f"DELETE FROM {table} WHERE rowid=?", (previous_message["id"],))
                        self.connection.execute(
                            "DELETE FROM message_cjk WHERE rowid IN (SELECT segment_id FROM message_cjk_segments WHERE message_row=?)",
                            (previous_message["id"],)
                        )
                        self.connection.execute("DELETE FROM message_cjk_segments WHERE message_row=?", (previous_message["id"],))
                        self.connection.execute("DELETE FROM messages WHERE id=?", (previous_message["id"],))
                for message in parsed:
                    message["parent_id"] = _visible_parent(message["node_id"], merged_graph)
                for message in parsed:
                    digest = _hash(message["role"], message["text"], message.get("parent_id") or "")
                    existing = self.connection.execute("""SELECT id,content_hash FROM messages
                        WHERE source_id=? AND account_id=? AND conversation_id=? AND message_id=?""",
                        (*namespace, message["id"])).fetchone()
                    self.connection.execute("""INSERT INTO messages
                        (source_id,account_id,conversation_id,message_id,node_id,parent_id,role,text,created_at,content_hash)
                        VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(source_id,account_id,conversation_id,message_id)
                        DO UPDATE SET node_id=excluded.node_id,parent_id=excluded.parent_id,role=excluded.role,
                        text=excluded.text,created_at=excluded.created_at,content_hash=excluded.content_hash""",
                        (*namespace, message["id"], message["node_id"], message.get("parent_id"),
                         message["role"], message["text"], message.get("created_at"), digest))
                    row = self.connection.execute("""SELECT id FROM messages WHERE source_id=? AND
                        account_id=? AND conversation_id=? AND message_id=?""", (*namespace, message["id"])).fetchone()
                    if existing is None or existing["content_hash"] != digest:
                        changed = True
                        self.connection.execute("DELETE FROM message_fts WHERE rowid=?", (row["id"],))
                        self.connection.execute("INSERT INTO message_fts(rowid,text) VALUES (?,?)", (row["id"], message["text"]))
                        self.connection.execute("DELETE FROM message_tri WHERE rowid=?", (row["id"],))
                        self.connection.execute("INSERT INTO message_tri(rowid,text) VALUES (?,?)", (row["id"], message["text"].casefold()))
                        self._index_cjk_message(row["id"])
                # A child can arrive before its parent in separate partial
                # imports. Reconcile every surviving message after graph union,
                # including messages absent from this particular payload.
                for existing_message in list(self.connection.execute("SELECT id,node_id,parent_id,role,text FROM messages WHERE source_id=? AND account_id=? AND conversation_id=?", namespace)):
                    if existing_message["node_id"] not in merged_graph:
                        continue
                    parent = _visible_parent(existing_message["node_id"], merged_graph)
                    if parent != existing_message["parent_id"]:
                        changed = True
                        digest = _hash(existing_message["role"], existing_message["text"], parent or "")
                        self.connection.execute("UPDATE messages SET parent_id=?,content_hash=? WHERE id=?", (parent, digest, existing_message["id"]))
                if changed:
                    # Ancestor edits also change descendant windows/topics. Rebuild
                    # all semantic derivatives in that conversation, never stale them.
                    for table in ("vectors", "enrichment"):
                        self.connection.execute(f"DELETE FROM {table} WHERE message_row IN (SELECT id FROM messages WHERE source_id=? AND account_id=? AND conversation_id=?)", namespace)
                self.connection.execute("RELEASE import_one")
                report["conversations"] += 1
                report["messages"] += len(parsed)
                report["skipped"] += skipped
            except (ValueError, TypeError, KeyError, sqlite3.Error) as exc:
                self.connection.execute("ROLLBACK TO import_one")
                self.connection.execute("RELEASE import_one")
                report["errors"].append({"index": position, "error": str(exc)})
        self.connection.commit()
        return report

    def _parse(self, item: dict) -> tuple[list[dict], list[tuple], int]:
        mapping = item.get("mapping")
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError("Conversation mapping must be an object")
        records, graph, skipped = [], [], 0
        if mapping is not None:
            raw = []
            for key, node in mapping.items():
                _identifier(key, "node id")
                if not isinstance(node, dict):
                    raise ValueError("Mapping node must be an object")
                parent = node.get("parent")
                if parent is not None:
                    _identifier(parent, "parent node")
                message = node.get("message")
                raw.append((key, parent, message))
        else:
            messages = item.get("messages")
            if not isinstance(messages, list):
                raise ValueError("Conversation requires mapping or messages")
            raw, previous = [], None
            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValueError("Message must be an object")
                key = _identifier(message.get("id", f"message-{index}"), "message id")
                parent = message.get("parent_id", previous)
                if parent is not None:
                    _identifier(parent, "parent message")
                raw.append((key, parent, message))
                previous = key
        for node_id, parent_node, message in raw:
            if message is None:
                graph.append((node_id, parent_node, None))
                continue
            if not isinstance(message, dict):
                raise ValueError("Node message must be an object or null")
            author = message.get("author") or {}
            role = message.get("role") or (author.get("role") if isinstance(author, dict) else None)
            text, omitted = _visible_text(message)
            skipped += omitted
            if role not in {"user", "assistant"} or text is None or not text.strip():
                skipped += int(not omitted)
                graph.append((node_id, parent_node, None))
                continue
            message_id = _identifier(message.get("id") or node_id, "message id")
            graph.append((node_id, parent_node, message_id))
            records.append({"id": message_id, "node_id": node_id, "parent_node": parent_node,
                            "role": role, "text": text,
                            "created_at": _stamp(message.get("create_time") or message.get("created_at"))})
        lookup = {node: (parent, msg) for node, parent, msg in graph}
        if len(lookup) != len(graph):
            raise ValueError("Duplicate node identifiers in conversation")
        if len({m["id"] for m in records}) != len(records):
            raise ValueError("Duplicate message identifiers in conversation")
        _validate_graph(lookup)
        for record in records:
            parent, seen = record.pop("parent_node"), {record["node_id"]}
            while parent in lookup:
                if parent in seen:
                    raise ValueError("Conversation parent graph contains a cycle")
                seen.add(parent)
                next_parent, parent_message = lookup[parent]
                if parent_message is not None:
                    record["parent_id"] = parent_message
                    break
                parent = next_parent
            else:
                record["parent_id"] = parent  # may be an ancestor from an earlier partial import
        return records, graph, skipped

    def _rows(self, role: str | None = None, account_id: str | None = None,
              literal_query: str | None = None) -> list[sqlite3.Row]:
        clauses, values = [], []
        if role:
            if role not in {"user", "assistant"}:
                raise ValueError("role must be user or assistant")
            clauses.append("m.role=?")
            values.append(role)
        if account_id is not None:
            clauses.append("m.account_id=?")
            values.append(account_id)
        if literal_query is not None and len(literal_query.casefold()) >= 3 and "\x00" not in literal_query:
            clauses.append("m.id IN (SELECT rowid FROM message_tri WHERE message_tri MATCH ? UNION SELECT id FROM messages WHERE instr(CAST(text AS BLOB), X'00') > 0)")
            values.append('"' + literal_query.casefold().replace('"', '""') + '"')
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return list(self.connection.execute("""SELECT m.*,c.title FROM messages m JOIN conversations c
            USING(source_id,account_id,conversation_id)""" + where + " ORDER BY m.id", values))

    def _result(self, row: sqlite3.Row, query: str, score: float, kind: str,
                case_sensitive: bool, start_hint: int = 0,
                retrieval_channels: list[str] | None = None,
                semantic_similarity: float | None = None,
                semantic_evidence: dict | None = None,
                dialogue_preview: list[dict] | None = None) -> dict:
        matches = literal_matches(row["text"], query, case_sensitive)
        hint = start_hint if type(start_hint) is int else 0
        focus = matches[0][0] if matches else max(0, min(len(row["text"]), hint))
        start, end = max(0, focus - 85), min(len(row["text"]), focus + 230)
        snippet = ("…" if start else "") + row["text"][start:end] + ("…" if end < len(row["text"]) else "")
        res = {"conversation_id": row["conversation_id"], "message_id": row["message_id"],
               "source_id": row["source_id"], "account_id": row["account_id"],
               "title": row["title"], "role": row["role"], "text": row["text"],
               "snippet": snippet, "matches": matches, "match_count": len(matches),
               "focus_offset": focus,
               "score": score, "match_type": kind, "created_at": row["created_at"],
               "url": self._url(row["source_id"], row["conversation_id"])}
        if retrieval_channels is not None:
            res["retrieval_channels"] = retrieval_channels
        if semantic_similarity is not None:
            res["semantic_similarity"] = semantic_similarity
        if semantic_evidence is not None:
            res["semantic_evidence"] = semantic_evidence
        if dialogue_preview is not None:
            res["dialogue_preview"] = dialogue_preview
        return res

    @staticmethod
    def _url(source_id: str, conversation_id: str) -> str | None:
        from urllib.parse import quote
        return "https://chatgpt.com/c/" + quote(conversation_id, safe="") if source_id == "chatgpt" else None

    def search(self, query: str, mode: str = "exact", limit: int = 20, offset: int = 0,
               case_sensitive: bool = False, role: str | None = None,
               account_id: str | None = None) -> dict:
        if not isinstance(query, str) or len(query) > 12000:
            raise ValueError("query must be a string of at most 12000 characters")
        if mode not in {"exact", "smart"}:
            raise ValueError("mode must be exact or smart")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be nonnegative")
        if role is not None and role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        semantic = {"available": False, "used": False, "status": "not_configured",
                    "model_id": None, "display_model_id": None, "errors": [],
                    "relevance_policy": "meaning-evidence-v1",
                    "rejected_keyword_only_count": 0}
        response = {"query": query, "mode": mode, "total": 0, "results": [], "semantic": semantic}
        if not query or not query.strip():
            return response
        if mode == "exact":
            rows = self._rows(role, account_id, query)
            by_id = {row["id"]: row for row in rows}
            if case_sensitive:
                literal = [row["id"] for row in rows if query and query in row["text"]]
            else:
                folded_query = query.casefold()
                literal = [row["id"] for row in rows if folded_query and folded_query in row["text"].casefold()]
            response["total"] = len(literal)
            response["results"] = [self._result(by_id[i], query, 1.0, "exact", case_sensitive)
                                   for i in literal[offset:offset + limit]]
            return response

        # Smart mode: bounded Python memory; short literal queries and vector CPU may still scan.
        # Filter messages by role and account_id.
        where_clauses = []
        where_params: list[Any] = []
        if role:
            where_clauses.append("role=?")
            where_params.append(role)
        if account_id is not None:
            where_clauses.append("account_id=?")
            where_params.append(account_id)
        msg_where = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Literal candidates via existing trigram superset + Python verification, retain only 100 IDs
        literal_candidates: list[int] = []
        tri_query = query.casefold()
        if len(tri_query) >= 3 and "\x00" not in query:
            tri_clauses = [
                "m.id IN (SELECT rowid FROM message_tri WHERE message_tri MATCH ? UNION SELECT id FROM messages WHERE instr(CAST(text AS BLOB), X'00') > 0)"
            ]
            tri_params: list[Any] = ['"' + tri_query.replace('"', '""') + '"']
            if role:
                tri_clauses.append("m.role=?")
                tri_params.append(role)
            if account_id is not None:
                tri_clauses.append("m.account_id=?")
                tri_params.append(account_id)
            tri_sql = "SELECT m.id, m.text FROM messages m WHERE " + " AND ".join(tri_clauses) + " ORDER BY m.id"
            cursor = self.connection.execute(tri_sql, tri_params)
        else:
            # Fallback for short query (< 3 chars) or contains NUL: stream (id, text) with role/account filter
            cursor = self.connection.execute("SELECT id, text FROM messages" + msg_where + " ORDER BY id", where_params)

        for mid, text in cursor:
            if case_sensitive:
                if query in text:
                    literal_candidates.append(mid)
            else:
                if tri_query in text.casefold():
                    literal_candidates.append(mid)
            if len(literal_candidates) >= 100:
                break

        # Lexical SQL BM25 <= 100 with role/account filter
        has_cjk = any(_is_cjk(ch) for ch in query)
        lexical_candidates: list[int] = []
        if has_cjk:
            cjk_terms = cjk_lexical_terms(query)
            if cjk_terms:
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in cjk_terms[:40])
                fts_params: list[Any] = [expression]
                cjk_clauses = ["message_cjk MATCH ?"]
                if role:
                    cjk_clauses.append("m.role=?")
                    fts_params.append(role)
                if account_id is not None:
                    cjk_clauses.append("m.account_id=?")
                    fts_params.append(account_id)
                where_sql = "WHERE " + " AND ".join(cjk_clauses)

                sql_mat = (
                    "WITH ranked_segments AS MATERIALIZED ("
                    "SELECT s.message_row, bm25(message_cjk) AS score "
                    "FROM message_cjk "
                    "JOIN message_cjk_segments s ON s.segment_id = message_cjk.rowid "
                    "JOIN messages m ON m.id = s.message_row "
                    f"{where_sql} "
                    "LIMIT -1 OFFSET 0 "
                    ") "
                    "SELECT message_row FROM ranked_segments "
                    "GROUP BY message_row ORDER BY MIN(score), message_row LIMIT 100"
                )
                try:
                    lexical_candidates = [r[0] for r in self.connection.execute(sql_mat, fts_params)]
                except sqlite3.OperationalError:
                    sql_no_mat = (
                        "WITH ranked_segments AS ("
                        "SELECT s.message_row, bm25(message_cjk) AS score "
                        "FROM message_cjk "
                        "JOIN message_cjk_segments s ON s.segment_id = message_cjk.rowid "
                        "JOIN messages m ON m.id = s.message_row "
                        f"{where_sql} "
                        "LIMIT -1 OFFSET 0 "
                        ") "
                        "SELECT message_row FROM ranked_segments "
                        "GROUP BY message_row ORDER BY MIN(score), message_row LIMIT 100"
                    )
                    lexical_candidates = [r[0] for r in self.connection.execute(sql_no_mat, fts_params)]

            # Cross-chunk mixed-language identifier seam support:
            # Query message_fts for non-CJK terms/identifiers (which indexes complete messages
            # without segment boundaries) with the same filters, merging and deduplicating up to 100.
            non_cjk_clean = "".join(" " if _is_cjk(c) else c for c in query)
            non_cjk_terms = [
                t for t in re.findall(r"[^\W_]+(?:[_\-:.]+[^\W_]+)*", non_cjk_clean, re.UNICODE)
                if t.lower() not in _BM25_FUNCTION_WORDS
            ]
            if non_cjk_terms:
                fts_expr = " OR ".join('"' + term.replace('"', '""') + '"' for term in non_cjk_terms[:40])
                fts_sql = (
                    "SELECT f.rowid FROM message_fts f JOIN messages m ON m.id = f.rowid "
                    "WHERE message_fts MATCH ?"
                )
                fts_params = [fts_expr]
                if role:
                    fts_sql += " AND m.role=?"
                    fts_params.append(role)
                if account_id is not None:
                    fts_sql += " AND m.account_id=?"
                    fts_params.append(account_id)
                fts_sql += " ORDER BY bm25(message_fts), f.rowid LIMIT 100"
                fts_candidates = [r[0] for r in self.connection.execute(fts_sql, fts_params)]

                if fts_candidates:
                    seen = set()
                    merged = []
                    cjk_set = set(lexical_candidates)
                    fts_set = set(fts_candidates)
                    for mid in lexical_candidates:
                        if mid in fts_set and mid not in seen:
                            seen.add(mid)
                            merged.append(mid)
                    for mid in fts_candidates:
                        if mid in cjk_set and mid not in seen:
                            seen.add(mid)
                            merged.append(mid)
                    for mid in lexical_candidates:
                        if mid not in seen:
                            seen.add(mid)
                            merged.append(mid)
                    for mid in fts_candidates:
                        if mid not in seen:
                            seen.add(mid)
                            merged.append(mid)
                    lexical_candidates = merged[:100]
        else:
            terms = [t for t in re.findall(r"[^\W_]+(?:[_\-:.]+[^\W_]+)*", query, re.UNICODE) if t.lower() not in _BM25_FUNCTION_WORDS]
            if terms:
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms[:40])
                fts_sql = (
                    "SELECT f.rowid FROM message_fts f JOIN messages m ON m.id = f.rowid "
                    "WHERE message_fts MATCH ?"
                )
                fts_params = [expression]
                if role:
                    fts_sql += " AND m.role=?"
                    fts_params.append(role)
                if account_id is not None:
                    fts_sql += " AND m.account_id=?"
                    fts_params.append(account_id)
                fts_sql += " ORDER BY bm25(message_fts), f.rowid LIMIT 100"
                lexical_candidates = [r[0] for r in self.connection.execute(fts_sql, fts_params)]

        # Semantic retrieval
        provider = self.embedding_provider
        semantic_channel_lists: dict[str, list[int]] = {}
        best_semantic_per_msg: dict[int, tuple[float, str, int, int]] = {}  # msg_id -> (sim, channel, start_offset, end_offset)

        if provider is not None:
            op_lock = getattr(provider, "operation_lock", None) or getattr(provider, "lock", None)
            with op_lock if op_lock is not None else contextlib.nullcontext():
                display_id = getattr(provider, "model_id", None)
                semantic["display_model_id"] = display_id
                q_ver = getattr(provider, "query_template_version", None)
                c_ver = getattr(provider, "calibration_version", None)
                if q_ver is not None:
                    semantic["query_template_version"] = q_ver
                if c_ver is not None:
                    semantic["calibration_version"] = c_ver

                try:
                    if callable(getattr(provider, "resolve_identity", None)):
                        resolved_id = provider.resolve_identity(self.DOCUMENT_PROFILE)
                    else:
                        resolved_id = display_id
                except Exception as exc:
                    semantic["status"] = "provider_error"
                    semantic["errors"].append(str(exc))
                    resolved_id = None

                if resolved_id is not None:
                    semantic["model_id"] = resolved_id
                    # Stream only needed vector columns joined to filtered messages and matching content_hash/model, count coverage with SQL
                    # First count filtered messages and indexed messages for semantic status
                    count_total_sql = "SELECT COUNT(*) FROM messages" + msg_where
                    total_filtered_messages = self.connection.execute(count_total_sql, where_params).fetchone()[0]

                    count_indexed_sql = (
                        "SELECT COUNT(DISTINCT m.id) FROM messages m "
                        "JOIN vectors v ON v.message_row = m.id AND v.content_hash = m.content_hash "
                        "JOIN enrichment e ON e.message_row = m.id AND e.model_id = v.model_id AND e.content_hash = m.content_hash "
                        "WHERE v.model_id = ?"
                    )
                    count_indexed_params: list[Any] = [resolved_id]
                    if role:
                        count_indexed_sql += " AND m.role = ?"
                        count_indexed_params.append(role)
                    if account_id is not None:
                        count_indexed_sql += " AND m.account_id = ?"
                        count_indexed_params.append(account_id)

                    indexed_messages = self.connection.execute(count_indexed_sql, count_indexed_params).fetchone()[0]
                    semantic["available"] = indexed_messages > 0
                    semantic["indexed_messages"] = indexed_messages
                    semantic["pending_messages"] = total_filtered_messages - indexed_messages
                    semantic["status"] = "index_pending" if not indexed_messages else "partial" if indexed_messages < total_filtered_messages else "ready"

                    # Query embedding never runs if no filtered current vectors
                    if indexed_messages > 0:
                        try:
                            if callable(getattr(provider, "embed_query", None)):
                                if callable_accepts_index_id(provider.embed_query):
                                    query_vector = normalized_vector(provider.embed_query(query, index_id=resolved_id))
                                else:
                                    query_vector = normalized_vector(provider.embed_query(query))
                            else:
                                result = provider.embed([query])
                                if not isinstance(result, list) or len(result) != 1:
                                    raise ProviderError("Query embedding response count mismatch")
                                query_vector = normalized_vector(result[0])
                            threshold = 0.3
                            if hasattr(provider, "minimum_similarity"):
                                raw_thresh = provider.minimum_similarity
                                if isinstance(raw_thresh, bool) or not isinstance(raw_thresh, (int, float)) or not math.isfinite(raw_thresh) or raw_thresh < 0 or raw_thresh > 1:
                                    raise ProviderError("Invalid minimum_similarity configured on provider")
                                threshold = float(raw_thresh)
                            semantic["minimum_similarity"] = threshold

                            # Stream only needed vector columns: message_row, channel, start_offset, end_offset, vector_json
                            vec_sql = (
                                "SELECT v.message_row, v.channel, v.start_offset, v.end_offset, v.vector_json "
                                "FROM vectors v JOIN messages m ON m.id = v.message_row AND m.content_hash = v.content_hash "
                                "JOIN enrichment e ON e.message_row = m.id AND e.model_id = v.model_id AND e.content_hash = m.content_hash "
                                "WHERE v.model_id = ?"
                            )
                            vec_params: list[Any] = [resolved_id]
                            if role:
                                vec_sql += " AND m.role = ?"
                                vec_params.append(role)
                            if account_id is not None:
                                vec_sql += " AND m.account_id = ?"
                                vec_params.append(account_id)
                            vec_sql += " ORDER BY v.message_row, v.channel, v.chunk"

                            # Stream ORDER BY message_row and lazily group per message using itertools.groupby.
                            # Compute at most two local channel maxima + one global best chunk for that message.
                            # Retain at most 100 unique messages per channel during traversal using deterministic
                            # score-desc/id-asc cutoff (ranking key: (-sim, mid), pruning worst (+sim, -mid)).
                            retained_channels: dict[str, dict[int, float]] = {"dialogue": {}, "topic": {}}
                            retained_best: dict[int, tuple[float, str, int, int]] = {}
                            expected_dim = len(query_vector)

                            def record_channel(ch: str, mid: int, sim: float) -> None:
                                d = retained_channels[ch]
                                if len(d) < 100:
                                    d[mid] = sim
                                else:
                                    worst_mid = max(d, key=lambda m: (-d[m], m))
                                    if (-sim, mid) < (-d[worst_mid], worst_mid):
                                        del d[worst_mid]
                                        d[mid] = sim

                            cursor = self.connection.execute(vec_sql, vec_params)
                            for mid, group in itertools.groupby(cursor, key=lambda r: r["message_row"]):
                                local_channel_max: dict[str, float] = {}
                                local_best_chunk: tuple[float, str, int, int] | None = None
                                for row in group:
                                    ch = row["channel"]
                                    if ch not in ("dialogue", "topic"):
                                        continue
                                    try:
                                        vec = json.loads(row["vector_json"])
                                        if not isinstance(vec, (list, tuple)):
                                            raise ProviderError("Corrupt vector format")
                                        norm_vec = normalized_vector(vec)
                                    except Exception as e:
                                        raise ProviderError("Corrupt or invalid stored vector") from e

                                    if len(norm_vec) != expected_dim:
                                        raise ProviderError("Stored vector dimensions differ from the query model")

                                    sim = sum(a * b for a, b in zip(query_vector, norm_vec))
                                    if sim >= threshold:
                                        if ch not in local_channel_max or sim > local_channel_max[ch]:
                                            local_channel_max[ch] = sim
                                        s_off, e_off = row["start_offset"], row["end_offset"]
                                        if local_best_chunk is None:
                                            local_best_chunk = (sim, ch, s_off, e_off)
                                        else:
                                            cur_sim, cur_ch, cur_s, cur_e = local_best_chunk
                                            if sim > cur_sim or (sim == cur_sim and ch == "dialogue" and cur_ch != "dialogue"):
                                                local_best_chunk = (sim, ch, s_off, e_off)

                                if local_best_chunk is not None:
                                    retained_best[mid] = local_best_chunk
                                    for ch, sim in local_channel_max.items():
                                        record_channel(ch, mid, sim)

                                    # Prune retained_best to retained union to prevent unbounded memory growth
                                    surviving = set(retained_channels["dialogue"]).union(retained_channels["topic"])
                                    for dead_mid in [m for m in retained_best if m not in surviving]:
                                        del retained_best[dead_mid]

                            # Final sorted lists per channel
                            for ch in ("dialogue", "topic"):
                                d = retained_channels[ch]
                                if d:
                                    semantic_channel_lists[ch] = sorted(d.keys(), key=lambda m: (-d[m], m))

                            best_semantic_per_msg = retained_best
                            semantic["used"] = True
                        except (ProviderError, ValueError, OSError, TypeError) as exc:
                            semantic["status"] = "provider_error"
                            semantic["errors"].append(str(exc))
                            # On any query provider/vector error discard ALL semantic candidates and keep truthful keyword fallback.
                            semantic_channel_lists.clear()
                            best_semantic_per_msg.clear()
                            semantic.pop("minimum_similarity", None)

        # Apply meaning-evidence-v1 candidate filtering policy when semantic was successfully used
        rejected_keyword_only_count = 0
        if semantic.get("used"):
            supported_semantic_ids = (
                set(semantic_channel_lists.get("dialogue", []))
                | set(semantic_channel_lists.get("topic", []))
            )
            literal_ids = set(literal_candidates)
            accepted_lexical: list[int] = []
            for mid in lexical_candidates:
                if mid in supported_semantic_ids or mid in literal_ids:
                    accepted_lexical.append(mid)
                else:
                    rejected_keyword_only_count += 1
            lexical_candidates = accepted_lexical

        semantic["relevance_policy"] = "meaning-evidence-v1"
        semantic["rejected_keyword_only_count"] = rejected_keyword_only_count

        # RRF combines only actual literal/keyword/semantic channels
        active_channels: list[tuple[str, list[int]]] = []
        if literal_candidates:
            active_channels.append(("literal", literal_candidates))
        if lexical_candidates:
            active_channels.append(("keyword", lexical_candidates))
        for ch in ("dialogue", "topic"):
            if ch in semantic_channel_lists and semantic_channel_lists[ch]:
                active_channels.append((ch, semantic_channel_lists[ch]))

        scores: dict[int, float] = {}
        msg_channels: dict[int, list[str]] = {}
        for ch_name, candidates in active_channels:
            for rank, mid in enumerate(candidates, start=1):
                scores[mid] = scores.get(mid, 0.0) + 1.0 / (60 + rank)
                if mid not in msg_channels:
                    msg_channels[mid] = []
                if ch_name not in msg_channels[mid]:
                    msg_channels[mid].append(ch_name)

        ordered_all = sorted(scores.keys(), key=lambda i: (-scores[i], i))
        response["total"] = len(ordered_all)
        page_ids = ordered_all[offset:offset + limit]

        # Fetch complete message/title only for final requested page, preserve deterministic pagination
        if page_ids:
            placeholders = ",".join("?" for _ in page_ids)
            page_rows = self.connection.execute(
                f"""SELECT m.*, c.title FROM messages m JOIN conversations c
                USING(source_id,account_id,conversation_id) WHERE m.id IN ({placeholders})""",
                page_ids
            ).fetchall()
            page_by_id = {r["id"]: r for r in page_rows}
        else:
            page_by_id = {}

        results = []
        literal_set = set(literal_candidates)
        for mid in page_ids:
            row = page_by_id[mid]
            ch_list = [c for c in ("literal", "keyword", "dialogue", "topic") if c in msg_channels.get(mid, [])]
            has_sem = mid in best_semantic_per_msg
            has_lex = "literal" in ch_list or "keyword" in ch_list
            if has_sem and has_lex:
                kind = "hybrid"
            elif has_sem:
                kind = "semantic"
            elif mid in literal_set:
                kind = "exact"
            else:
                kind = "lexical"

            sem_sim = None
            sem_ev = None
            start_hint = 0
            dialogue_preview = None
            if has_sem:
                sim, best_ch, s_off, e_off = best_semantic_per_msg[mid]
                sem_sim = round(sim, 6)
                sem_ev = {"channel": best_ch, "start_offset": s_off, "end_offset": e_off}
                start_hint = s_off

                ancestor_snippets = []
                current_parent_id = row["parent_id"]
                seen_ancestor_ids = {row["message_id"]}
                for _ in range(2):
                    if not current_parent_id or current_parent_id in seen_ancestor_ids:
                        break
                    seen_ancestor_ids.add(current_parent_id)
                    ancestor_row = self.connection.execute(
                        "SELECT message_id, parent_id, role, text FROM messages "
                        "WHERE source_id=? AND account_id=? AND conversation_id=? AND message_id=?",
                        (row["source_id"], row["account_id"], row["conversation_id"], current_parent_id)
                    ).fetchone()
                    if ancestor_row is None:
                        break
                    raw_text = ancestor_row["text"]
                    snippet_text = raw_text[:240]
                    ancestor_snippets.append({
                        "message_id": ancestor_row["message_id"],
                        "role": ancestor_row["role"],
                        "text": snippet_text,
                        "start_offset": 0,
                        "end_offset": len(snippet_text),
                    })
                    current_parent_id = ancestor_row["parent_id"]
                ancestor_snippets.reverse()
                dialogue_preview = ancestor_snippets

            results.append(
                self._result(
                    row,
                    query,
                    scores[mid],
                    kind,
                    case_sensitive,
                    start_hint=start_hint,
                    retrieval_channels=ch_list,
                    semantic_similarity=sem_sim,
                    semantic_evidence=sem_ev,
                    dialogue_preview=dialogue_preview,
                )
            )

        response["results"] = results
        return response

    def context(self, conversation_id: str, message_id: str, source_id: str = "chatgpt",
                account_id: str = "default") -> dict:
        namespace = (source_id, account_id, conversation_id)
        conversation = self.connection.execute("SELECT * FROM conversations WHERE source_id=? AND account_id=? AND conversation_id=?", namespace).fetchone()
        rows = list(self.connection.execute("SELECT * FROM messages WHERE source_id=? AND account_id=? AND conversation_id=? ORDER BY id", namespace))
        lookup = {r["message_id"]: r for r in rows}
        if conversation is None or message_id not in lookup:
            raise KeyError("Unknown conversation or message in selected source/account")
        path, seen, current = [], set(), message_id
        while current in lookup and current not in seen:
            seen.add(current)
            path.append(lookup[current])
            current = lookup[current]["parent_id"]
        path.reverse()
        # Prefer a child on the conversation's selected current-node ancestry.
        selected_nodes, node = set(), conversation["current_node"]
        node_lookup = {r["node_id"]: r for r in self.connection.execute("SELECT * FROM nodes WHERE source_id=? AND account_id=? AND conversation_id=?", namespace)}
        while node in node_lookup and node not in selected_nodes:
            selected_nodes.add(node)
            node = node_lookup[node]["parent_node"]
        current = message_id
        while True:
            children = [r for r in rows if r["parent_id"] == current and r["message_id"] not in seen]
            if not children:
                break
            children.sort(key=lambda r: (r["node_id"] not in selected_nodes, r["created_at"] or "", r["message_id"]))
            chosen = children[0]
            seen.add(chosen["message_id"])
            path.append(chosen)
            current = chosen["message_id"]
        return {"conversation_id": conversation_id, "message_id": message_id,
                "source_id": source_id, "account_id": account_id,
                "title": conversation["title"], "url": self._url(source_id, conversation_id),
                "messages": [{"id": r["message_id"], "role": r["role"], "text": r["text"],
                              "parent_id": r["parent_id"], "created_at": r["created_at"]} for r in path]}

    def _bounded_prior_messages(self, row: sqlite3.Row | dict, max_bytes: int = 960, is_legacy: bool = False) -> list[dict]:
        """Lookup at most 2 namespace-scoped ancestors with SQL byte-bounded prefixes.

        Does not call context() or load entire conversation. Isolated from sibling branches.
        """
        prior: list[dict] = []
        seen = {row["message_id"]}
        curr_parent_id = row.get("parent_id") if isinstance(row, dict) else row["parent_id"]
        source_id = row.get("source_id") if isinstance(row, dict) else row["source_id"]
        account_id = row.get("account_id") if isinstance(row, dict) else row["account_id"]
        conversation_id = row.get("conversation_id") if isinstance(row, dict) else row["conversation_id"]

        for _ in range(2):
            if not curr_parent_id or curr_parent_id in seen:
                break
            seen.add(curr_parent_id)
            if is_legacy:
                parent_row = self.connection.execute(
                    "SELECT message_id, parent_id, role, substr(CAST(text AS BLOB), -8000) AS text_blob FROM messages "
                    "WHERE source_id=? AND account_id=? AND conversation_id=? AND message_id=?",
                    (source_id, account_id, conversation_id, curr_parent_id),
                ).fetchone()
                if parent_row is None:
                    break
                text_val = _safe_utf8_suffix(parent_row["text_blob"], 8000)
            else:
                parent_row = self.connection.execute(
                    "SELECT message_id, parent_id, role, substr(CAST(text AS BLOB), 1, ?) AS text_blob FROM messages "
                    "WHERE source_id=? AND account_id=? AND conversation_id=? AND message_id=?",
                    (max_bytes, source_id, account_id, conversation_id, curr_parent_id),
                ).fetchone()
                if parent_row is None:
                    break
                text_val = _safe_utf8_prefix(parent_row["text_blob"], max_bytes)

            prior.append({
                "message_id": parent_row["message_id"],
                "parent_id": parent_row["parent_id"],
                "role": parent_row["role"],
                "text": text_val,
            })
            curr_parent_id = parent_row["parent_id"]

        prior.reverse()
        return prior

    def _windows(self, row: sqlite3.Row | dict, max_input_bytes: int | None = None) -> list[tuple[int, int, str]]:
        if max_input_bytes is None:
            prior_messages = self._bounded_prior_messages(row, is_legacy=True)
            prefix = "\n".join(m["role"] + ": " + m["text"] for m in prior_messages)[-2000:]
            windows, text, start = [], row["text"], 0
            while start < len(text):
                end = min(len(text), start + 3000)
                windows.append((start, end, prefix + "\n" + row["role"] + ": " + text[start:end]))
                if end == len(text):
                    break
                start = end - 400
            return windows

        if max_input_bytes != 960:
            raise ValueError(f"Unsupported enrichment byte budget: {max_input_bytes}")

        prior_messages = self._bounded_prior_messages(row, max_bytes=960)
        prefix_str = self._format_dialogue_prefix(prior_messages, max_bytes=384)
        prefix_bytes = prefix_str.encode("utf-8")

        role_overhead = ("\n" + row["role"] + ": ").encode("utf-8")
        budget_for_target = 960 - len(prefix_bytes) - len(role_overhead)
        max_chunk_bytes = min(512, budget_for_target)
        if max_chunk_bytes <= 0:
            raise ValueError(f"Cannot fit target chunk within enrichment byte budget {max_input_bytes}")

        spans = self._paragraph_aware_spans(row["text"], max_chunk_bytes=max_chunk_bytes)
        windows = []
        for start, end in spans:
            chunk_text = row["text"][start:end]
            if not chunk_text.strip():
                continue
            derived = prefix_str + "\n" + row["role"] + ": " + chunk_text
            windows.append((start, end, derived))

        return windows

    @staticmethod
    def _format_dialogue_prefix(prior_messages: list[dict], max_bytes: int = 384) -> str:
        """Format branch-correct context prefix <= max_bytes UTF-8 bytes.

        Prefers coherent original question excerpt rather than arbitrary chopped bytes.
        No generated summary.
        """
        if not prior_messages:
            return ""

        full = "\n".join(m["role"] + ": " + m["text"] for m in prior_messages)
        if len(full.encode("utf-8")) <= max_bytes:
            return full

        # Find the question message (prefer user role, else the last prior turn)
        target_turn = None
        for m in reversed(prior_messages):
            if m.get("role") == "user":
                target_turn = m
                break
        if target_turn is None:
            target_turn = prior_messages[-1]

        role = target_turn.get("role", "user")
        text = target_turn.get("text", "")
        header = f"{role}: "
        header_bytes_len = len(header.encode("utf-8"))
        avail = max_bytes - header_bytes_len
        if avail <= 0:
            return ""

        # Step back from avail to find codepoint end that fits in UTF-8
        tentative_end = min(len(text), avail)
        while tentative_end > 0 and len(text[:tentative_end].encode("utf-8")) > avail:
            excess = len(text[:tentative_end].encode("utf-8")) - avail
            tentative_end -= max(1, excess // 4)

        if tentative_end <= 0:
            return ""

        slice_text = text[:tentative_end]
        # Look for sentence boundary in slice
        sent_matches = list(re.finditer(r'(?:[.?!](?:\s+|$)|[。！？；][”’）]?(?:\s*|$))', slice_text))
        if sent_matches:
            rightmost = sent_matches[-1].end()
            if rightmost >= len(slice_text) // 2:
                slice_text = slice_text[:rightmost].rstrip()
        else:
            sp = slice_text.rfind(" ")
            if sp >= len(slice_text) // 2:
                slice_text = slice_text[:sp].rstrip()

        return header + slice_text

    @staticmethod
    def _paragraph_aware_spans(text: str, max_chunk_bytes: int = 512) -> list[tuple[int, int]]:
        """Split text into faithful paragraph/sentence/code-line-aware spans.

        Prefer whole paragraphs <= 512 UTF-8 bytes.
        Split larger paragraphs at sentence/newline/code boundaries without exceeding cap.
        If single huge token, safe Unicode fallback.
        Preserves all original characters/coverage, positive progress, and original codepoint offsets.
        Adjacent paragraph target windows do not point to irrelevant paragraphs.
        Avoids whitespace-only spans when possible; long whitespace gaps are kept as bounded spans
        and skipped during embedding window creation without exceeding cap.
        """
        if not text:
            return []

        text_len = len(text)
        if not text.strip():
            spans = []
            cur = 0
            while cur < text_len:
                nxt = min(text_len, cur + max_chunk_bytes)
                while nxt > cur and len(text[cur:nxt].encode("utf-8")) > max_chunk_bytes:
                    excess = len(text[cur:nxt].encode("utf-8")) - max_chunk_bytes
                    nxt -= max(1, excess // 4)
                if nxt <= cur:
                    nxt = cur + 1
                spans.append((cur, nxt))
                cur = nxt
            return spans

        # 1. Segment by paragraph breaks (two or more newlines)
        break_matches = list(re.finditer(r"(?:\r?\n\s*){2,}", text))
        para_spans: list[tuple[int, int]] = []
        cur_p = 0
        for m in break_matches:
            b_start = m.start()
            b_end = m.end()
            sep_bytes = len(text[cur_p:b_end].encode("utf-8"))
            has_subsequent_content = any(c.strip() for c in text[b_end:])
            if sep_bytes <= max_chunk_bytes or not has_subsequent_content:
                if b_end > cur_p:
                    para_spans.append((cur_p, b_end))
                    cur_p = b_end
            else:
                if b_start > cur_p:
                    para_spans.append((cur_p, b_start))
                    cur_p = b_start
                elif b_end > cur_p:
                    cur_p = b_end

        if cur_p < text_len:
            para_spans.append((cur_p, text_len))
        if not para_spans:
            para_spans = [(0, text_len)]

        # 2. Split any paragraphs that exceed max_chunk_bytes
        final_spans: list[tuple[int, int]] = []
        for p_start, p_end in para_spans:
            p_bytes = len(text[p_start:p_end].encode("utf-8"))
            if p_bytes <= max_chunk_bytes:
                final_spans.append((p_start, p_end))
            else:
                cur = p_start
                while cur < p_end:
                    rem_bytes = len(text[cur:p_end].encode("utf-8"))
                    if rem_bytes <= max_chunk_bytes:
                        final_spans.append((cur, p_end))
                        break

                    max_end = min(p_end, cur + max_chunk_bytes)
                    while max_end > cur and len(text[cur:max_end].encode("utf-8")) > max_chunk_bytes:
                        excess = len(text[cur:max_end].encode("utf-8")) - max_chunk_bytes
                        max_end -= max(1, excess // 4)

                    if max_end <= cur:
                        max_end = cur + 1

                    sub_text = text[cur:max_end]
                    split_pt = None

                    # 1. Code fence or newline boundary (\n)
                    nl_matches = [m.end() for m in re.finditer(r"\r?\n", sub_text)]
                    if nl_matches:
                        for n_idx in reversed(nl_matches):
                            if n_idx >= len(sub_text) // 3:
                                cand = cur + n_idx
                                if text[cur:cand].strip():
                                    split_pt = cand
                                    break

                    # 2. Sentence boundary
                    if split_pt is None:
                        s_matches = [m.end() for m in re.finditer(r'(?:[.?!](?:\s+|$)|[。！？；][”’）]?(?:\s*|$))', sub_text)]
                        if s_matches:
                            for s_idx in reversed(s_matches):
                                if s_idx >= len(sub_text) // 4:
                                    cand = cur + s_idx
                                    if text[cur:cand].strip():
                                        split_pt = cand
                                        break

                    # 3. Word boundary
                    if split_pt is None:
                        sp = sub_text.rfind(" ")
                        if sp >= len(sub_text) // 4:
                            cand = cur + sp + 1
                            if text[cur:cand].strip():
                                split_pt = cand

                    # 4. Safe Unicode fallback
                    if split_pt is None or split_pt <= cur or split_pt > max_end:
                        split_pt = max_end

                    final_spans.append((cur, split_pt))
                    cur = split_pt

        # 3. Post-validation: merge short whitespace with neighbors if <= max_chunk_bytes;
        # if cannot merge safely without exceeding cap, retain bounded whitespace spans.
        cleaned_spans: list[tuple[int, int]] = []
        for s, e in final_spans:
            is_ws = not text[s:e].strip()
            if not is_ws:
                if cleaned_spans and not text[cleaned_spans[-1][0]:cleaned_spans[-1][1]].strip():
                    prev_ws_s, prev_ws_e = cleaned_spans[-1]
                    comb_bytes = len(text[prev_ws_s:e].encode("utf-8"))
                    if comb_bytes <= max_chunk_bytes:
                        cleaned_spans.pop()
                        cleaned_spans.append((prev_ws_s, e))
                        continue
                cleaned_spans.append((s, e))
            else:
                if cleaned_spans:
                    prev_s, prev_e = cleaned_spans[-1]
                    comb_bytes = len(text[prev_s:e].encode("utf-8"))
                    if comb_bytes <= max_chunk_bytes:
                        cleaned_spans.pop()
                        cleaned_spans.append((prev_s, e))
                        continue
                    if text[prev_s:prev_e].strip():
                        borrow = min(32, max(1, (prev_e - prev_s) // 2))
                        new_split = prev_e - borrow
                        part1_bytes = len(text[prev_s:new_split].encode("utf-8"))
                        part2_bytes = len(text[new_split:e].encode("utf-8"))
                        if (
                            part1_bytes <= max_chunk_bytes
                            and part2_bytes <= max_chunk_bytes
                            and text[prev_s:new_split].strip()
                            and text[new_split:e].strip()
                        ):
                            cleaned_spans.pop()
                            cleaned_spans.append((prev_s, new_split))
                            cleaned_spans.append((new_split, e))
                            continue
                cleaned_spans.append((s, e))

        return cleaned_spans

    def _count_pending_unknown(self, account_id: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM messages"
        params: list[Any] = []
        if account_id is not None:
            sql += " WHERE account_id = ?"
            params.append(account_id)
        return self.connection.execute(sql, params).fetchone()[0]

    def _count_pending(self, model_id: str, generate_topics: bool = False, account_id: str | None = None) -> int:
        sql = (
            "SELECT COUNT(*) FROM messages m WHERE NOT ("
            "EXISTS (SELECT 1 FROM enrichment e WHERE e.message_row = m.id AND e.model_id = ? AND e.content_hash = m.content_hash"
        )
        params: list[Any] = [model_id]
        if generate_topics:
            sql += " AND e.topics = 1"
        sql += ") AND EXISTS (SELECT 1 FROM vectors v WHERE v.message_row = m.id AND v.model_id = ? AND v.content_hash = m.content_hash))"
        params.append(model_id)
        if account_id is not None:
            sql += " AND m.account_id = ?"
            params.append(account_id)
        return self.connection.execute(sql, params).fetchone()[0]

    def _has_later_pending(self, model_id: str, after_id: int, generate_topics: bool = False, account_id: str | None = None) -> bool:
        sql = (
            "SELECT 1 FROM messages m WHERE m.id > ? AND NOT ("
            "EXISTS (SELECT 1 FROM enrichment e WHERE e.message_row = m.id AND e.model_id = ? AND e.content_hash = m.content_hash"
        )
        params: list[Any] = [after_id, model_id]
        if generate_topics:
            sql += " AND e.topics = 1"
        sql += ") AND EXISTS (SELECT 1 FROM vectors v WHERE v.message_row = m.id AND v.model_id = ? AND v.content_hash = m.content_hash))"
        params.append(model_id)
        if account_id is not None:
            sql += " AND m.account_id = ?"
            params.append(account_id)
        sql += " LIMIT 1"
        return self.connection.execute(sql, params).fetchone() is not None

    def enrich(self, provider: EmbeddingProvider | None = None, limit: int = 100,
               generate_topics: bool = False, account_id: str | None = None,
               after_cursor: dict | None = None, should_stop: Any = None,
               progress: Any = None, max_time_seconds: float = DEFAULT_MAX_TIME_SECONDS,
               max_requests: int = DEFAULT_MAX_REQUESTS) -> dict:
        """Explicit bounded resumable enrichment. Caller may invoke this in a worker.

        Commits after each message. Errors retain lexical data and leave the
        failed message pending. Topics are discovery aids, never literal evidence.
        """
        provider = provider or self.embedding_provider
        if provider is None:
            raise ValueError("Configure a provider before semantic enrichment")
        if type(limit) is not int or isinstance(limit, bool) or limit < 1 or limit > 10000:
            raise ValueError("limit must be between 1 and 10000")
        if type(generate_topics) is not bool:
            raise ValueError("generate_topics must be a boolean")
        if account_id is not None:
            _identifier(account_id, "account_id")
        if should_stop is not None and not callable(should_stop):
            raise ValueError("should_stop must be callable")
        if progress is not None and not callable(progress):
            raise ValueError("progress must be callable")
        if (
            isinstance(max_time_seconds, bool)
            or not isinstance(max_time_seconds, (int, float))
            or not math.isfinite(max_time_seconds)
            or max_time_seconds <= 0
            or max_time_seconds > 1200
        ):
            raise ValueError("max_time_seconds must be a positive finite number of at most 1200 seconds")
        if type(max_requests) is not int or isinstance(max_requests, bool) or max_requests < 1 or max_requests > 1024:
            raise ValueError("max_requests must be an integer between 1 and 1024")

        after_id = 0
        if after_cursor is not None:
            if not isinstance(after_cursor, dict):
                raise ValueError("after_cursor must be an object")
            expected_keys = {"after_id", "model_id", "account_id", "generate_topics"}
            if set(after_cursor.keys()) != expected_keys:
                raise ValueError("after_cursor must contain exactly after_id, model_id, account_id, and generate_topics")
            raw_after_id = after_cursor["after_id"]
            if type(raw_after_id) is not int or isinstance(raw_after_id, bool) or raw_after_id < 0 or raw_after_id > 9223372036854775807:
                raise ValueError("after_cursor.after_id must be a non-negative signed 64-bit integer")
            after_id = raw_after_id

            cursor_model = after_cursor["model_id"]
            if not isinstance(cursor_model, str) or not cursor_model or len(cursor_model) > 1024:
                raise ValueError("after_cursor.model_id must be a non-empty string of at most 1024 characters")

            cursor_acc = after_cursor["account_id"]
            if cursor_acc != account_id:
                raise ValueError("Cursor account_id does not match batch scope")

            cursor_topics = after_cursor["generate_topics"]
            if type(cursor_topics) is not bool or cursor_topics != generate_topics:
                raise ValueError("Cursor generate_topics does not match batch scope")

        display_id = getattr(provider, "model_id", None) or "unknown"
        if callable(getattr(provider, "resolve_identity", None)):
            cached_id = getattr(provider, "cached_identity", None)
            cached_available = isinstance(cached_id, str) and bool(cached_id)
            cached_or_display_id = cached_id if cached_available else "unknown"
        else:
            cached_available = True
            cached_or_display_id = display_id

        if after_cursor is not None and cached_available:
            if cursor_model != cached_or_display_id:
                raise ValueError(f"Cursor model_id '{cursor_model}' does not match resolved profile '{cached_or_display_id}'")

        if should_stop is not None and should_stop():
            pending_count = (
                self._count_pending(cached_or_display_id, generate_topics=generate_topics, account_id=account_id)
                if cached_or_display_id != "unknown"
                else self._count_pending_unknown(account_id=account_id)
            )
            return {
                "attempted": 0,
                "processed": 0,
                "remaining": pending_count,
                "skipped_budget": 0,
                "skipped_details": [],
                "vectors": 0,
                "topics": 0,
                "errors": [],
                "paused": True,
                "stopped_reason": "paused",
                "next_cursor": after_cursor,
                "pass_exhausted": False,
                "model_id": cached_or_display_id,
                "display_model_id": display_id,
                "scope": {"account_id": account_id, "model_id": cached_or_display_id, "generate_topics": generate_topics},
            }

        start_time = time.monotonic()
        deadline = start_time + max_time_seconds
        op_lock = getattr(provider, "operation_lock", None) or getattr(provider, "lock", None)

        resolved_id = None
        paused_before_resolver = False
        timeout_before_resolver = False

        with op_lock if op_lock is not None else contextlib.nullcontext():
            if should_stop is not None and should_stop():
                paused_before_resolver = True
            elif time.monotonic() >= deadline:
                timeout_before_resolver = True
            else:
                if callable(getattr(provider, "resolve_identity", None)):
                    resolved_id = provider.resolve_identity(self.DOCUMENT_PROFILE)
                else:
                    resolved_id = display_id

        if paused_before_resolver:
            pending_count = (
                self._count_pending(cached_or_display_id, generate_topics=generate_topics, account_id=account_id)
                if cached_or_display_id != "unknown"
                else self._count_pending_unknown(account_id=account_id)
            )
            return {
                "attempted": 0,
                "processed": 0,
                "remaining": pending_count,
                "skipped_budget": 0,
                "skipped_details": [],
                "vectors": 0,
                "topics": 0,
                "errors": [],
                "paused": True,
                "stopped_reason": "paused",
                "next_cursor": after_cursor,
                "pass_exhausted": False,
                "model_id": cached_or_display_id,
                "display_model_id": display_id,
                "scope": {"account_id": account_id, "model_id": cached_or_display_id, "generate_topics": generate_topics},
            }

        if timeout_before_resolver:
            pending_count = (
                self._count_pending(cached_or_display_id, generate_topics=generate_topics, account_id=account_id)
                if cached_or_display_id != "unknown"
                else self._count_pending_unknown(account_id=account_id)
            )
            return {
                "attempted": 0,
                "processed": 0,
                "remaining": pending_count,
                "skipped_budget": 0,
                "skipped_details": [],
                "vectors": 0,
                "topics": 0,
                "errors": [],
                "paused": False,
                "stopped_reason": "timeout",
                "next_cursor": after_cursor,
                "pass_exhausted": False,
                "model_id": cached_or_display_id,
                "display_model_id": display_id,
                "scope": {"account_id": account_id, "model_id": cached_or_display_id, "generate_topics": generate_topics},
            }

        if after_cursor is not None:
            if cursor_model != resolved_id:
                raise ValueError(f"Cursor model_id '{cursor_model}' does not match resolved profile '{resolved_id}'")

        if generate_topics and not callable(getattr(provider, "summarize", None)):
            raise ValueError("This provider does not support topic generation")

        meta_sql = (
            "SELECT m.id, m.source_id, m.account_id, m.conversation_id, m.message_id, "
            "m.node_id, m.parent_id, m.role, m.created_at, m.content_hash, "
            "length(CAST(m.text AS BLOB)) AS source_bytes "
            "FROM messages m WHERE NOT ("
            "EXISTS (SELECT 1 FROM enrichment e WHERE e.message_row = m.id AND e.model_id = ? AND e.content_hash = m.content_hash"
        )
        meta_params: list[Any] = [resolved_id]
        if generate_topics:
            meta_sql += " AND e.topics = 1"
        meta_sql += ") AND EXISTS (SELECT 1 FROM vectors v WHERE v.message_row = m.id AND v.model_id = ? AND v.content_hash = m.content_hash))"
        meta_params.append(resolved_id)
        if account_id is not None:
            meta_sql += " AND m.account_id = ?"
            meta_params.append(account_id)
        if after_id > 0:
            meta_sql += " AND m.id > ?"
            meta_params.append(after_id)
        meta_sql += " ORDER BY m.id LIMIT ?"
        meta_params.append(limit)

        # Finish metadata SELECT cursor before inference so a long-lived read transaction cannot block concurrent source writers
        meta_rows = [dict(r) for r in self.connection.execute(meta_sql, meta_params).fetchall()]

        report = {
            "attempted": 0,
            "processed": 0,
            "remaining": 0,
            "skipped_budget": 0,
            "skipped_details": [],
            "vectors": 0,
            "topics": 0,
            "errors": [],
            "paused": False,
            "stopped_reason": None,
            "next_cursor": None,
            "pass_exhausted": False,
            "model_id": resolved_id,
            "display_model_id": display_id,
            "scope": {"account_id": account_id, "model_id": resolved_id, "generate_topics": generate_topics},
        }

        request_count = 0
        last_attempted_id = after_id
        raw_budget = getattr(provider, "max_input_bytes", None)
        max_input_bytes = raw_budget if isinstance(raw_budget, int) and not isinstance(raw_budget, bool) else None

        def check_request_guard() -> str | None:
            if should_stop is not None and should_stop():
                return "paused"
            if time.monotonic() >= deadline:
                return "timeout"
            if request_count >= max_requests:
                return "request_budget"
            return None

        for meta_row in meta_rows:
            stop_reason = check_request_guard()
            if stop_reason is not None:
                if stop_reason == "paused":
                    report["paused"] = True
                report["stopped_reason"] = stop_reason
                break

            if progress is not None:
                progress({
                    "attempted": report["attempted"],
                    "processed": report["processed"],
                    "remaining": report.get("remaining"),
                    "skipped_budget": report["skipped_budget"],
                })

            row = dict(meta_row)
            source_bytes = row["source_bytes"] or 0
            if source_bytes > MAX_ENRICH_SOURCE_BYTES:
                report["attempted"] += 1
                report["skipped_budget"] += 1
                report["skipped_details"].append({
                    "conversation_id": row["conversation_id"],
                    "message_id": row["message_id"],
                    "reason": "source_bytes_exceeded",
                    "source_bytes": source_bytes,
                    "max_bytes": MAX_ENRICH_SOURCE_BYTES,
                })
                last_attempted_id = row["id"]
                continue

            # Detect changed/deleted/oversized source using metadata only and leave pending
            cur_meta = self.connection.execute(
                "SELECT content_hash, length(CAST(text AS BLOB)) AS cur_bytes FROM messages WHERE id=?",
                (row["id"],)
            ).fetchone()
            if cur_meta is None:
                last_attempted_id = row["id"]
                continue
            if cur_meta["cur_bytes"] > MAX_ENRICH_SOURCE_BYTES:
                report["attempted"] += 1
                report["skipped_budget"] += 1
                report["skipped_details"].append({
                    "conversation_id": row["conversation_id"],
                    "message_id": row["message_id"],
                    "reason": "source_bytes_exceeded",
                    "source_bytes": cur_meta["cur_bytes"],
                    "max_bytes": MAX_ENRICH_SOURCE_BYTES,
                })
                last_attempted_id = row["id"]
                continue
            if cur_meta["content_hash"] != row["content_hash"]:
                last_attempted_id = row["id"]
                continue

            # Actual body SELECT includes captured hash AND SQL byte cap 262144
            body_row = self.connection.execute(
                "SELECT text FROM messages WHERE id=? AND content_hash=? AND length(CAST(text AS BLOB)) <= ?",
                (row["id"], row["content_hash"], MAX_ENRICH_SOURCE_BYTES)
            ).fetchone()
            if body_row is None:
                last_attempted_id = row["id"]
                continue
            row["text"] = body_row[0]

            try:
                windows = self._windows(row, max_input_bytes=max_input_bytes)
            except Exception as exc:
                report["attempted"] += 1
                err_code = "type_error" if isinstance(exc, TypeError) else "window_error"
                report["errors"].append({"conversation_id": row["conversation_id"], "message_id": row["message_id"], "error": err_code})
                last_attempted_id = row["id"]
                continue

            if len(windows) > MAX_ENRICH_WINDOWS:
                report["attempted"] += 1
                report["skipped_budget"] += 1
                report["skipped_details"].append({
                    "conversation_id": row["conversation_id"],
                    "message_id": row["message_id"],
                    "reason": "window_budget_exceeded",
                    "windows": len(windows),
                    "max_windows": MAX_ENRICH_WINDOWS,
                })
                last_attempted_id = row["id"]
                continue

            records = [("dialogue", i, start, end, text) for i, (start, end, text) in enumerate(windows)]

            interrupted = False

            if generate_topics:
                for i, (start, end, text) in enumerate(windows):
                    with op_lock if op_lock is not None else contextlib.nullcontext():
                        guard = check_request_guard()
                        if guard is not None:
                            if guard == "paused":
                                report["paused"] = True
                            report["stopped_reason"] = guard
                            interrupted = True
                            break
                        request_count += 1
                        try:
                            topic = provider.summarize(text)
                        except Exception as exc:
                            err_code = "type_error" if isinstance(exc, TypeError) else "provider_error"
                            report["errors"].append({"conversation_id": row["conversation_id"], "message_id": row["message_id"], "error": err_code})
                            report["stopped_reason"] = "provider_error"
                            interrupted = True
                            break

                    if interrupted:
                        break

                    if not isinstance(topic, str) or not topic.strip():
                        report["errors"].append({"conversation_id": row["conversation_id"], "message_id": row["message_id"], "error": "Topic provider returned empty text"})
                        report["stopped_reason"] = "provider_error"
                        interrupted = True
                        break
                    records.append(("topic", i, start, end, topic))

            if interrupted:
                break

            vectors = []
            for b_start in range(0, len(records), 8):
                batch = records[b_start:b_start + 8]
                texts_to_embed = [r[4] for r in batch]

                with op_lock if op_lock is not None else contextlib.nullcontext():
                    guard = check_request_guard()
                    if guard is not None:
                        if guard == "paused":
                            report["paused"] = True
                        report["stopped_reason"] = guard
                        interrupted = True
                        break
                    request_count += 1
                    try:
                        if callable(getattr(provider, "embed_documents", None)):
                            if callable_accepts_index_id(provider.embed_documents):
                                result = provider.embed_documents(texts_to_embed, index_id=resolved_id)
                            else:
                                result = provider.embed_documents(texts_to_embed)
                        else:
                            result = provider.embed(texts_to_embed)
                    except Exception as exc:
                        err_code = "type_error" if isinstance(exc, TypeError) else "provider_error"
                        report["errors"].append({"conversation_id": row["conversation_id"], "message_id": row["message_id"], "error": err_code})
                        report["stopped_reason"] = "provider_error"
                        interrupted = True
                        break

                if interrupted:
                    break

                if not isinstance(result, list) or len(result) != len(batch):
                    report["errors"].append({"conversation_id": row["conversation_id"], "message_id": row["message_id"], "error": "Provider did not return one embedding per input"})
                    report["stopped_reason"] = "provider_error"
                    interrupted = True
                    break

                try:
                    for v in result:
                        vectors.append(normalized_vector(v))
                except Exception as exc:
                    err_code = "type_error" if isinstance(exc, TypeError) else "provider_error"
                    report["errors"].append({"conversation_id": row["conversation_id"], "message_id": row["message_id"], "error": err_code})
                    report["stopped_reason"] = "provider_error"
                    interrupted = True
                    break

            if interrupted:
                break

            expected_dim = getattr(provider, "expected_output_dimension", None)
            if expected_dim is not None and any(len(v) != expected_dim for v in vectors):
                report["errors"].append({"conversation_id": row["conversation_id"], "message_id": row["message_id"], "error": f"Embedding dimension does not match profile expected {expected_dim}"})
                report["stopped_reason"] = "provider_error"
                break
            if len({len(v) for v in vectors}) != 1:
                report["errors"].append({"conversation_id": row["conversation_id"], "message_id": row["message_id"], "error": "Embedding dimensions differ within a message"})
                report["stopped_reason"] = "provider_error"
                break

            # Cancellation check immediately before publication
            if should_stop is not None and should_stop():
                report["paused"] = True
                report["stopped_reason"] = "paused"
                break

            # Lock source writers only for publication, never for inference.
            # If a concurrent import changed text or ancestry, retry later.
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                # Recheck immediately AFTER BEGIN IMMEDIATE returns
                if should_stop is not None and should_stop():
                    self.connection.execute("ROLLBACK")
                    report["paused"] = True
                    report["stopped_reason"] = "paused"
                    break

                # Publication lookup guards bytes/hash before reading body
                cur_meta = self.connection.execute(
                    "SELECT content_hash, length(CAST(text AS BLOB)) AS cur_bytes FROM messages WHERE id=?",
                    (row["id"],)
                ).fetchone()
                if (
                    cur_meta is None
                    or cur_meta["cur_bytes"] > MAX_ENRICH_SOURCE_BYTES
                    or cur_meta["content_hash"] != row["content_hash"]
                ):
                    self.connection.execute("ROLLBACK")
                    last_attempted_id = row["id"]
                    continue

                current = self.connection.execute(
                    "SELECT * FROM messages WHERE id=? AND content_hash=? AND length(CAST(text AS BLOB)) <= ?",
                    (row["id"], row["content_hash"], MAX_ENRICH_SOURCE_BYTES)
                ).fetchone()
                if current is None or self._windows(current, max_input_bytes=max_input_bytes) != windows:
                    self.connection.execute("ROLLBACK")
                    last_attempted_id = row["id"]
                    continue

                self.connection.execute("DELETE FROM vectors WHERE message_row=? AND model_id=?", (row["id"], resolved_id))
                for record, vector in zip(records, vectors):
                    channel, chunk, start, end, derived = record
                    self.connection.execute("INSERT INTO vectors VALUES (?,?,?,?,?,?,?,?,?)", (
                        row["id"], channel, chunk, resolved_id, row["content_hash"], start, end, derived, json.dumps(vector)))
                self.connection.execute(
                    "INSERT INTO enrichment VALUES (?,?,?,?) ON CONFLICT(message_row,model_id) DO UPDATE SET content_hash=excluded.content_hash,topics=excluded.topics",
                    (row["id"], resolved_id, row["content_hash"], int(generate_topics))
                )

                # Recheck immediately before commit
                if should_stop is not None and should_stop():
                    self.connection.execute("ROLLBACK")
                    report["paused"] = True
                    report["stopped_reason"] = "paused"
                    break

                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise

            report["attempted"] += 1
            report["processed"] += 1
            report["vectors"] += len(records)
            report["topics"] += sum(r[0] == "topic" for r in records)
            last_attempted_id = row["id"]

        report["remaining"] = self._count_pending(resolved_id, generate_topics=generate_topics, account_id=account_id)
        has_later = self._has_later_pending(resolved_id, last_attempted_id, generate_topics=generate_topics, account_id=account_id)
        if has_later:
            report["next_cursor"] = {
                "after_id": last_attempted_id,
                "model_id": resolved_id,
                "account_id": account_id,
                "generate_topics": generate_topics,
            }
            report["pass_exhausted"] = False
        else:
            report["next_cursor"] = None
            report["pass_exhausted"] = True

        return report

    def status(self, account_id: str | None = None) -> dict:
        if account_id is not None:
            _identifier(account_id, "account_id")
        scalar = lambda sql, params=(): self.connection.execute(sql, params).fetchone()[0]
        sources = [dict(r) for r in self.connection.execute("""SELECT c.source_id,c.account_id,
            COUNT(DISTINCT c.conversation_id) AS conversations,COUNT(m.id) AS messages
            FROM conversations c LEFT JOIN messages m USING(source_id,account_id,conversation_id)
            GROUP BY c.source_id,c.account_id ORDER BY c.source_id,c.account_id""")]
        models = [dict(r) for r in self.connection.execute("SELECT model_id,COUNT(DISTINCT message_row) AS indexed_messages,COUNT(*) AS vectors FROM vectors GROUP BY model_id")]
        display_id = getattr(self.embedding_provider, "model_id", None)
        if self.embedding_provider is None:
            effective_model_id = None
        elif callable(getattr(self.embedding_provider, "resolve_identity", None)):
            effective_model_id = getattr(self.embedding_provider, "cached_identity", None)
        else:
            effective_model_id = display_id

        if account_id is not None:
            eligible = scalar("SELECT COUNT(*) FROM messages WHERE account_id=?", (account_id,))
        else:
            eligible = scalar("SELECT COUNT(*) FROM messages")

        if effective_model_id is not None:
            current_sql = (
                "SELECT COUNT(*) FROM messages m WHERE "
                "EXISTS (SELECT 1 FROM enrichment e WHERE e.message_row = m.id AND e.model_id = ? AND e.content_hash = m.content_hash) "
                "AND EXISTS (SELECT 1 FROM vectors v WHERE v.message_row = m.id AND v.model_id = ? AND v.content_hash = m.content_hash)"
            )
            current_params: list[Any] = [effective_model_id, effective_model_id]
            if account_id is not None:
                current_sql += " AND m.account_id = ?"
                current_params.append(account_id)
            current = scalar(current_sql, current_params)

            legacy_sql = (
                "SELECT COUNT(DISTINCT v.message_row) FROM vectors v "
                "JOIN messages m ON m.id = v.message_row "
                "WHERE (v.model_id != ? OR v.content_hash != m.content_hash)"
            )
            legacy_params: list[Any] = [effective_model_id]
            if account_id is not None:
                legacy_sql += " AND m.account_id = ?"
                legacy_params.append(account_id)
            legacy = scalar(legacy_sql, legacy_params)

            pending = max(0, eligible - current)
        else:
            current = 0
            pending = eligible
            if account_id is not None:
                legacy = scalar(
                    "SELECT COUNT(DISTINCT v.message_row) FROM vectors v "
                    "JOIN messages m ON m.id = v.message_row WHERE m.account_id=?", (account_id,)
                )
            else:
                legacy = scalar("SELECT COUNT(DISTINCT v.message_row) FROM vectors v")

        coverage = {
            "account_id": account_id,
            "eligible": eligible,
            "current": current,
            "pending": pending,
            "legacy": legacy,
        }

        sem_dict = {
            "configured": self.embedding_provider is not None,
            "model_id": effective_model_id,
            "display_model_id": display_id,
            "models": models,
            "coverage": coverage,
        }
        if self.embedding_provider is not None:
            q_ver = getattr(self.embedding_provider, "query_template_version", None)
            c_ver = getattr(self.embedding_provider, "calibration_version", None)
            if q_ver is not None:
                sem_dict["query_template_version"] = q_ver
            if c_ver is not None:
                sem_dict["calibration_version"] = c_ver
        return {"conversations": scalar("SELECT COUNT(*) FROM conversations"),
                "messages": scalar("SELECT COUNT(*) FROM messages"), "sources": sources,
                "semantic": sem_dict}
