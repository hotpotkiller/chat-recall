"""Loopback-only passage-search server; private data never becomes static content."""
from __future__ import annotations

import hmac
import json
import mimetypes
import secrets
import sqlite3
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .providers import ProviderError
from .search import SearchIndex

MAX_IMPORT_BYTES = 64 * 1024 * 1024
WEB_ROOT = Path(__file__).with_name("web")
FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "demo.json"


class RecallServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, db_path, *, token, extension_origins=(), provider=None, reranker=None):
        if address[0] not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("This release listens on loopback only")
        self.db_path = str(db_path)
        self.token = token
        self.extension_origins = frozenset(extension_origins)
        if any(not x.startswith("chrome-extension://") or "/" in x[19:]
               for x in self.extension_origins):
            raise ValueError("Allowlist exact chrome-extension://<extension-id> origins")
        self.provider = provider
        self.reranker = reranker
        self.job_lock = threading.Lock()
        self.semantic_job = {"state": "idle", "job_id": None, "error": None}
        self._current_stop_event = None
        self.receipt_path = Path(db_path).with_suffix(".sync.json")
        super().__init__(address, RecallHandler)

    def index(self):
        return SearchIndex(self.db_path, embedding_provider=self.provider)

    def read_receipts(self):
        try:
            return json.loads(self.receipt_path.read_text("utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    def save_receipt(self, namespace, coverage):
        # The receipt describes source claims, never a verified completeness claim.
        with self.job_lock:
            receipts = self.read_receipts()
            receipts[namespace] = coverage
            temp = self.receipt_path.with_suffix(".tmp")
            temp.write_text(json.dumps(receipts, ensure_ascii=False), "utf-8")
            temp.chmod(0o600)
            temp.replace(self.receipt_path)

    def start_enrichment(self, *, limit=100, generate_topics=False, account_id=None, after_cursor=None):
        if self.provider is None:
            raise ValueError("Configure an embedding provider before semantic indexing")
        if type(limit) is not int or isinstance(limit, bool) or not (1 <= limit <= 1000):
            raise ValueError("Use enrichment limit 1–1000")
        if type(generate_topics) is not bool:
            raise ValueError("generate_topics must be a boolean")
        if account_id is not None:
            if isinstance(account_id, bool) or not isinstance(account_id, str) or not account_id or len(account_id) > 1024:
                raise ValueError("Invalid account_id")
        if after_cursor is not None and not isinstance(after_cursor, dict):
            raise ValueError("after_cursor must be an object")

        with self.job_lock:
            if self.semantic_job.get("state") in ("running", "pausing"):
                return dict(self.semantic_job)
            job_id = secrets.token_hex(16)
            stop_event = threading.Event()
            self._current_stop_event = stop_event
            self.semantic_job = {
                "job_id": job_id,
                "state": "running",
                "error": None,
                "scope": {"account_id": account_id, "limit": limit, "generate_topics": generate_topics},
                "progress": {"attempted": 0, "processed": 0, "remaining": None, "skipped_budget": 0},
            }

        def run():
            index = None
            report = None
            error = None
            try:
                index = self.index()
                def on_progress(p):
                    with self.job_lock:
                        if self.semantic_job.get("job_id") == job_id:
                            self.semantic_job["progress"] = {
                                "attempted": p.get("attempted", 0),
                                "processed": p.get("processed", 0),
                                "remaining": p.get("remaining"),
                                "skipped_budget": p.get("skipped_budget", 0),
                            }
                report = index.enrich(
                    limit=limit,
                    generate_topics=generate_topics,
                    account_id=account_id,
                    after_cursor=after_cursor,
                    should_stop=stop_event.is_set,
                    progress=on_progress,
                )
            except ProviderError:
                error = "provider_error"
            except sqlite3.Error:
                error = "sqlite_error"
            except OSError as exc:
                error = str(exc)
            except Exception:
                error = "unexpected_error"
            finally:
                cleanup_error = None
                if index is not None:
                    try:
                        index.close()
                    except OSError as exc:
                        cleanup_error = str(exc)
                    except Exception:
                        cleanup_error = "cleanup_error"
                # Publish terminal state only after cleanup, so an immediately
                # retried batch cannot be overwritten by the old worker.
                with self.job_lock:
                    if self.semantic_job.get("job_id") == job_id:
                        scope = {"account_id": account_id, "limit": limit, "generate_topics": generate_topics}
                        # Cleanup failure remains failed (existing contract), preserving prior report counts
                        if cleanup_error is not None:
                            self.semantic_job = {
                                "job_id": job_id,
                                "state": "failed",
                                "error": cleanup_error,
                                "scope": scope,
                            }
                            if report is not None:
                                self.semantic_job["report"] = report
                        elif report is not None and report.get("paused") and error is None:
                            self.semantic_job = {
                                "job_id": job_id,
                                "state": "paused",
                                "error": None,
                                "report": report,
                                "scope": scope,
                            }
                        elif error is not None:
                            state = "failed"
                            job = {"job_id": job_id, "state": state, "error": error, "scope": scope}
                            if report is not None:
                                job["report"] = report
                            self.semantic_job = job
                        elif report is None:
                            self.semantic_job = {
                                "job_id": job_id,
                                "state": "failed",
                                "error": "missing_report",
                                "scope": scope,
                            }
                        elif report.get("errors"):
                            state = "partial" if report.get("processed", 0) > 0 else "failed"
                            self.semantic_job = {
                                "job_id": job_id,
                                "state": state,
                                "error": None,
                                "report": report,
                                "scope": scope,
                            }
                        else:
                            self.semantic_job = {
                                "job_id": job_id,
                                "state": "complete",
                                "error": None,
                                "report": report,
                                "scope": scope,
                            }

        threading.Thread(target=run, daemon=True, name="recall-enrichment").start()
        return dict(self.semantic_job)

    def pause_enrichment(self, *, job_id: str):
        with self.job_lock:
            current_job_id = self.semantic_job.get("job_id")
            current_state = self.semantic_job.get("state")
            if not current_job_id or current_job_id != job_id:
                raise ValueError("Stale or invalid job_id")
            if current_state == "running":
                self.semantic_job["state"] = "pausing"
                if self._current_stop_event is not None:
                    self._current_stop_event.set()
            return dict(self.semantic_job)


class RecallHandler(BaseHTTPRequestHandler):
    server: RecallServer

    def log_message(self, _format, *args):
        # URLs can contain private search phrases. Do not log them.
        pass

    def _origin_allowed(self, *, extension=True):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        if self.headers.get("Host", "") not in hosts:
            return False
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            same = parsed.scheme == "http" and parsed.netloc in hosts and not parsed.path
            return same or (extension and origin in self.server.extension_origins)
        return self.headers.get("Sec-Fetch-Site", "none") in {"none", "same-origin"}

    def _authorized(self):
        expected = "Bearer " + self.server.token
        actual = self.headers.get("Authorization", "")
        return hmac.compare_digest(actual, expected)

    def _send(self, status, body, content_type="application/json; charset=utf-8", *, cors=True):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        origin = self.headers.get("Origin")
        if cors and origin in self.server.extension_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._send(status, {"error": message})

    def do_OPTIONS(self):
        if not self._origin_allowed() or self.headers.get("Origin") not in self.server.extension_origins:
            return self._error(403, "Origin is not paired with this server")
        if self.headers.get("Access-Control-Request-Method") not in {"GET", "POST"}:
            return self._error(405, "Unsupported method")
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if not self._origin_allowed():
            return self._error(403, "Untrusted Host or Origin")
        parsed = urlsplit(self.path)
        params = parse_qs(parsed.query, keep_blank_values=True)
        get = lambda key, default=None: params.get(key, [default])[0]
        if parsed.path == "/api/config":
            if not self._origin_allowed(extension=False):
                return self._error(403, "Open pairing settings in the local application")
            return self._send(200, {"token": self.server.token, "version": "0.1.0", "max_import_bytes": MAX_IMPORT_BYTES,
                                    "extension_origins": sorted(self.server.extension_origins),
                                    "semantic_available": self.server.provider is not None,
                                    "cloud_rerank_available": self.server.reranker is not None,
                                    "cloud_rerank_model": getattr(self.server.reranker, "model_id", None),
                                    "enrich_controls_version": 1})
        if parsed.path.startswith("/api/"):
            if self.headers.get("Origin") in self.server.extension_origins and not self._authorized():
                return self._error(401, "A paired bearer token is required")
            index = self.server.index()
            try:
                if parsed.path == "/api/status":
                    account_id = get("account_id") or None
                    result = index.status(account_id=account_id)
                    result["sync_receipts"] = self.server.read_receipts()
                    with self.server.job_lock:
                        result["semantic_job"] = dict(self.server.semantic_job)
                elif parsed.path == "/api/search":
                    limit, offset = int(get("limit", "20")), int(get("offset", "0"))
                    if not 1 <= limit <= 100 or offset < 0:
                        raise ValueError("Use limit 1–100 and a non-negative offset")
                    cloud = get("cloud") == "1" and get("mode", "exact") == "smart" and self.server.reranker is not None
                    result = index.search(get("q", ""), mode=get("mode", "exact"), limit=30 if cloud else limit, offset=0 if cloud else offset,
                                          case_sensitive=get("case_sensitive", "0") in {"1", "true"},
                                          role=get("role") or None, account_id=get("account_id") or None)
                    if cloud:
                        from .rerank import RerankError, rerank_results
                        ranking = {"used": False, "model": self.server.reranker.model_id, "candidate_limit": 30}
                        try:
                            result = rerank_results(index, result, self.server.reranker)
                            ranking["used"] = True
                        except (RerankError, ValueError, KeyError):
                            ranking["error"] = "Provider unavailable or invalid response; local ranking retained"
                        result["retrieval_total"] = result["total"]
                        result["total"] = len(result["results"])
                        result["results"] = result["results"][offset:offset + limit]
                        result["semantic"]["reranking"] = ranking
                elif parsed.path == "/api/context":
                    result = index.context(get("conversation_id", ""), get("message_id", ""),
                                           source_id=get("source_id", "chatgpt"), account_id=get("account_id", "default"))
                else:
                    return self._error(404, "Unknown API endpoint")
                return self._send(200, result)
            except (ValueError, TypeError) as exc:
                return self._error(400, str(exc))
            except KeyError:
                return self._error(404, "Conversation or message not found")
            finally:
                index.close()
        # Only bundled assets are served. Database, source files and tokens cannot
        # be reached through static paths, even if the working directory changes.
        name = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
        target = (WEB_ROOT / name).resolve()
        if WEB_ROOT.resolve() not in target.parents or not target.is_file():
            return self._error(404, "Page not found")
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript"}:
            content_type += "; charset=utf-8"
        return self._send(200, target.read_bytes(), content_type, cors=False)

    def do_POST(self):
        if not self._origin_allowed():
            return self._error(403, "Untrusted Host or Origin")
        if not self._authorized():
            return self._error(401, "A paired bearer token is required")
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._error(415, "Send application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_IMPORT_BYTES:
                return self._error(413, "Send a JSON body no larger than 64 MiB")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Request body must be an object")
            path = urlsplit(self.path).path
            if path == "/api/enrich/pause":
                job_id = data.get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    return self._error(400, "job_id is required")
                try:
                    res = self.server.pause_enrichment(job_id=job_id)
                    return self._send(200, res)
                except ValueError as exc:
                    return self._error(400, str(exc))
            if path == "/api/enrich":
                raw_limit = data.get("limit", 100)
                if type(raw_limit) is not int or isinstance(raw_limit, bool) or not (1 <= raw_limit <= 1000):
                    return self._error(400, "Use enrichment limit 1–1000")
                if "generate_topics" in data and type(data["generate_topics"]) is not bool:
                    return self._error(400, "generate_topics must be a boolean")
                account_id = data.get("account_id")
                if account_id is not None and (isinstance(account_id, bool) or not isinstance(account_id, str) or not account_id or len(account_id) > 1024):
                    return self._error(400, "Invalid account_id")
                after_cursor = data.get("after_cursor")
                if after_cursor is not None and not isinstance(after_cursor, dict):
                    return self._error(400, "Invalid after_cursor")
                try:
                    job = self.server.start_enrichment(
                        limit=raw_limit,
                        generate_topics=bool(data.get("generate_topics", False)),
                        account_id=account_id,
                        after_cursor=after_cursor,
                    )
                    return self._send(202, job)
                except ValueError as exc:
                    return self._error(400, str(exc))
            if path == "/api/demo":
                payload = json.loads(FIXTURE.read_text("utf-8"))
                source, account = "chatgpt", "demo"
            elif path == "/api/import":
                if "payload" not in data:
                    raise ValueError("Missing payload")
                payload = data["payload"]
                source, account = data.get("source_id", "chatgpt"), data.get("account_id", "default")
                if not isinstance(source, str) or not isinstance(account, str) or not source or not account:
                    raise ValueError("source_id and account_id must be non-empty strings")
                if len(source) > 200 or len(account) > 300:
                    raise ValueError("Source or account identifier is too long")
            else:
                return self._error(404, "Unknown API endpoint")
            index = self.server.index()
            try:
                report = index.import_payload(payload, source_id=source, account_id=account)
            finally:
                index.close()
            if isinstance(data.get("coverage"), dict):
                self.server.save_receipt(source + ":" + account, data["coverage"])
            return self._send(200, report)
        except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            return self._error(400, str(exc))
        except Exception:
            # Internal exceptions may contain private data or paths.
            return self._error(500, "Operation failed; source data was not reported as fully imported")


def load_token(directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / "bridge-token"
    if path.exists():
        if path.is_symlink():
            raise ValueError("Token path must not be a symlink")
        token = path.read_text("utf-8").strip()
        if len(token) < 32:
            raise ValueError("Invalid saved bridge token")
        return token
    token = secrets.token_urlsafe(32)
    # Exclusive creation prevents overwriting a concurrent process's token.
    import os
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text("utf-8").strip()
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    return token
