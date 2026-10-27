"""SQLite source archive, verified literal retrieval and optional hybrid search.

No model, network, or download is needed for importing, exact search, lexical
ranking, or passage viewing. Semantic indexing is explicitly invoked by enrich.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .providers import EmbeddingProvider, ProviderError, normalized_vector


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


class SearchIndex:
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
        self.connection.commit()

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
            clauses.append("m.id IN (SELECT rowid FROM message_tri WHERE message_tri MATCH ?)")
            values.append('"' + literal_query.casefold().replace('"', '""') + '"')
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return list(self.connection.execute("""SELECT m.*,c.title FROM messages m JOIN conversations c
            USING(source_id,account_id,conversation_id)""" + where + " ORDER BY m.id", values))

    def _result(self, row: sqlite3.Row, query: str, score: float, kind: str,
                case_sensitive: bool, start_hint: int = 0) -> dict:
        matches = literal_matches(row["text"], query, case_sensitive)
        hint = start_hint if type(start_hint) is int else 0
        focus = matches[0][0] if matches else max(0, min(len(row["text"]), hint))
        start, end = max(0, focus - 85), min(len(row["text"]), focus + 230)
        snippet = ("…" if start else "") + row["text"][start:end] + ("…" if end < len(row["text"]) else "")
        return {"conversation_id": row["conversation_id"], "message_id": row["message_id"],
                "source_id": row["source_id"], "account_id": row["account_id"],
                "title": row["title"], "role": row["role"], "text": row["text"],
                "snippet": snippet, "matches": matches, "match_count": len(matches),
                "focus_offset": focus,
                "score": score, "match_type": kind, "created_at": row["created_at"],
                "url": self._url(row["source_id"], row["conversation_id"])}

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
        semantic = {"available": False, "used": False, "status": "not_configured",
                    "model_id": None, "errors": []}
        response = {"query": query, "mode": mode, "total": 0, "results": [], "semantic": semantic}
        if not query or not query.strip():
            return response
        rows = self._rows(role, account_id, query if mode == "exact" else None)
        by_id = {row["id"]: row for row in rows}
        if case_sensitive:
            literal = [row["id"] for row in rows if query and query in row["text"]]
        else:
            folded_query = query.casefold()
            literal = [row["id"] for row in rows if folded_query and folded_query in row["text"].casefold()]
        if mode == "exact":
            response["total"] = len(literal)
            response["results"] = [self._result(by_id[i], query, 1.0, "exact", case_sensitive)
                                   for i in literal[offset:offset + limit]]
            return response
        channels: list[list[int]] = []
        if literal:
            channels.append(literal[:100])
        terms = re.findall(r"[^\W_]+", query, re.UNICODE)
        if terms:
            expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms[:40])
            lexical = [r[0] for r in self.connection.execute(
                "SELECT rowid FROM message_fts WHERE message_fts MATCH ? ORDER BY bm25(message_fts),rowid",
                (expression,)) if r[0] in by_id][:100]
            if lexical:
                channels.append(lexical)
        vector_hits, start_hints = set(), {}
        lexical_hits = {i for channel in channels for i in channel}
        provider = self.embedding_provider
        if provider is not None:
            semantic["model_id"] = provider.model_id
            stored = list(self.connection.execute("SELECT * FROM vectors WHERE model_id=?", (provider.model_id,)))
            stored = [r for r in stored if r["message_row"] in by_id]
            semantic["available"] = bool(stored)
            indexed_messages = len({r["message_row"] for r in stored})
            semantic["indexed_messages"] = indexed_messages
            semantic["pending_messages"] = len(rows) - indexed_messages
            semantic["status"] = "index_pending" if not stored else "partial" if indexed_messages < len(rows) else "ready"
            if stored:
                try:
                    result = provider.embed([query])
                    if not isinstance(result, list) or len(result) != 1:
                        raise ProviderError("Query embedding response count mismatch")
                    query_vector = normalized_vector(result[0])
                    ranked: dict[str, dict[int, tuple[float, int]]] = {}
                    for record in stored:
                        vector = json.loads(record["vector_json"])
                        if len(vector) != len(query_vector):
                            raise ProviderError("Stored vector dimensions differ from the query model")
                        similarity = sum(a * b for a, b in zip(query_vector, vector))
                        channel = ranked.setdefault(record["channel"], {})
                        key = record["message_row"]
                        if key not in channel or similarity > channel[key][0]:
                            channel[key] = (similarity, record["start_offset"])
                    for channel in ranked.values():
                        ordered = sorted(channel, key=lambda i: (-channel[i][0], i))[:100]
                        channels.append(ordered)
                        vector_hits.update(ordered)
                        for i in ordered:
                            start_hints.setdefault(i, channel[i][1])
                    semantic["used"] = True
                except (ProviderError, ValueError, OSError, TypeError) as exc:
                    semantic["status"] = "provider_error"
                    semantic["errors"].append(str(exc))
        scores: dict[int, float] = {}
        for channel in channels:
            for rank, key in enumerate(channel, start=1):
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
        ordered = sorted(scores, key=lambda i: (-scores[i], i))
        response["total"] = len(ordered)
        literal_set = set(literal)
        response["results"] = [self._result(by_id[i], query, scores[i],
            "hybrid" if i in vector_hits and i in lexical_hits else
            "semantic" if i in vector_hits else "exact" if i in literal_set else "lexical",
            case_sensitive, start_hints.get(i, 0)) for i in ordered[offset:offset + limit]]
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

    def _windows(self, row: sqlite3.Row) -> list[tuple[int, int, str]]:
        history = self.context(row["conversation_id"], row["message_id"], row["source_id"], row["account_id"])["messages"]
        target = next(i for i, m in enumerate(history) if m["id"] == row["message_id"])
        prefix = "\n".join(m["role"] + ": " + m["text"] for m in history[max(0, target - 2):target])[-2000:]
        windows, text, start = [], row["text"], 0
        while start < len(text):
            end = min(len(text), start + 3000)
            windows.append((start, end, prefix + "\n" + row["role"] + ": " + text[start:end]))
            if end == len(text):
                break
            start = end - 400
        return windows

    def enrich(self, provider: EmbeddingProvider | None = None, limit: int = 100,
               generate_topics: bool = False, account_id: str | None = None) -> dict:
        """Explicit resumable enrichment. Caller may invoke this in a worker.

        Commits after each message. Errors retain lexical data and leave the
        failed message pending. Topics are discovery aids, never literal evidence.
        """
        provider = provider or self.embedding_provider
        if provider is None:
            raise ValueError("Configure a provider before semantic enrichment")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 10000:
            raise ValueError("limit must be between 1 and 10000")
        if generate_topics and not callable(getattr(provider, "summarize", None)):
            raise ValueError("This provider does not support topic generation")
        report = {"processed": 0, "vectors": 0, "topics": 0, "errors": [], "model_id": provider.model_id}
        pending = []
        for row in self._rows(account_id=account_id):
            ready = self.connection.execute("SELECT * FROM enrichment WHERE message_row=? AND model_id=?", (row["id"], provider.model_id)).fetchone()
            if ready is None or (generate_topics and not ready["topics"]):
                pending.append(row)
        for row in pending[:limit]:
            try:
                windows = self._windows(row)
                records = [("dialogue", i, start, end, text) for i, (start, end, text) in enumerate(windows)]
                if generate_topics:
                    for i, (start, end, text) in enumerate(windows):
                        topic = provider.summarize(text)
                        if not isinstance(topic, str) or not topic.strip():
                            raise ProviderError("Topic provider returned empty text")
                        records.append(("topic", i, start, end, topic))
                vectors = []
                for start in range(0, len(records), 8):
                    batch = records[start:start + 8]
                    result = provider.embed([r[4] for r in batch])
                    if not isinstance(result, list) or len(result) != len(batch):
                        raise ProviderError("Provider did not return one embedding per input")
                    vectors.extend(normalized_vector(v) for v in result)
                if len({len(v) for v in vectors}) != 1:
                    raise ProviderError("Embedding dimensions differ within a message")
                # Lock source writers only for publication, never for inference.
                # If a concurrent import changed text or ancestry, retry later.
                self.connection.execute("BEGIN IMMEDIATE")
                with self.connection:
                    current = self.connection.execute("SELECT * FROM messages WHERE id=?", (row["id"],)).fetchone()
                    if current is None or current["content_hash"] != row["content_hash"] or self._windows(current) != windows:
                        continue
                    self.connection.execute("DELETE FROM vectors WHERE message_row=? AND model_id=?", (row["id"], provider.model_id))
                    for record, vector in zip(records, vectors):
                        channel, chunk, start, end, derived = record
                        self.connection.execute("INSERT INTO vectors VALUES (?,?,?,?,?,?,?,?,?)", (
                            row["id"], channel, chunk, provider.model_id, row["content_hash"], start, end, derived, json.dumps(vector)))
                    self.connection.execute("INSERT INTO enrichment VALUES (?,?,?,?) ON CONFLICT(message_row,model_id) DO UPDATE SET content_hash=excluded.content_hash,topics=excluded.topics", (
                        row["id"], provider.model_id, row["content_hash"], int(generate_topics)))
                report["processed"] += 1
                report["vectors"] += len(records)
                report["topics"] += sum(r[0] == "topic" for r in records)
            except (ProviderError, ValueError, TypeError, OSError) as exc:
                report["errors"].append({"conversation_id": row["conversation_id"], "message_id": row["message_id"], "error": str(exc)})
        report["remaining"] = len(pending) - report["processed"]
        return report

    def status(self) -> dict:
        scalar = lambda sql: self.connection.execute(sql).fetchone()[0]
        sources = [dict(r) for r in self.connection.execute("""SELECT c.source_id,c.account_id,
            COUNT(DISTINCT c.conversation_id) AS conversations,COUNT(m.id) AS messages
            FROM conversations c LEFT JOIN messages m USING(source_id,account_id,conversation_id)
            GROUP BY c.source_id,c.account_id ORDER BY c.source_id,c.account_id""")]
        models = [dict(r) for r in self.connection.execute("SELECT model_id,COUNT(DISTINCT message_row) AS indexed_messages,COUNT(*) AS vectors FROM vectors GROUP BY model_id")]
        return {"conversations": scalar("SELECT COUNT(*) FROM conversations"),
                "messages": scalar("SELECT COUNT(*) FROM messages"), "sources": sources,
                "semantic": {"configured": self.embedding_provider is not None,
                             "model_id": getattr(self.embedding_provider, "model_id", None),
                             "models": models}}
