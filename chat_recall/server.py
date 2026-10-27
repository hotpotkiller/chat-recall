"""Loopback-only passage-search server; private data never becomes static content."""
from __future__ import annotations

import hmac
import json
import mimetypes
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

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
        self.semantic_job = {"state": "idle", "error": None}
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

    def start_enrichment(self, *, limit=100, generate_topics=False):
        if self.provider is None:
            raise ValueError("Configure an embedding provider before semantic indexing")
        with self.job_lock:
            if self.semantic_job["state"] == "running":
                return dict(self.semantic_job)
            self.semantic_job = {"state": "running", "error": None}

        def run():
            index = None
            report = None
            error = None
            try:
                index = self.index()
                report = index.enrich(limit=limit, generate_topics=generate_topics)
            except Exception as exc:
                error = str(exc)
            finally:
                if index is not None:
                    try:
                        index.close()
                    except Exception as exc:
                        error = str(exc)
                # Publish terminal state only after cleanup, so an immediately
                # retried batch cannot be overwritten by the old worker.
                with self.job_lock:
                    if error is not None:
                        job = {"state": "failed", "error": error}
                        if report is not None:
                            job["report"] = report
                        self.semantic_job = job
                    elif report is not None and report.get("errors"):
                        state = "partial" if report.get("processed", 0) > 0 else "failed"
                        self.semantic_job = {"state": state, "error": None, "report": report}
                    else:
                        self.semantic_job = {"state": "complete", "error": None, "report": report}

        threading.Thread(target=run, daemon=True, name="recall-enrichment").start()
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
                                    "cloud_rerank_model": getattr(self.server.reranker, "model_id", None)})
        if parsed.path.startswith("/api/"):
            if self.headers.get("Origin") in self.server.extension_origins and not self._authorized():
                return self._error(401, "A paired bearer token is required")
            index = self.server.index()
            try:
                if parsed.path == "/api/status":
                    result = index.status()
                    result["sync_receipts"] = self.server.read_receipts()
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
            if path == "/api/enrich":
                limit = int(data.get("limit", 100))
                if not 1 <= limit <= 1000:
                    raise ValueError("Use enrichment limit 1–1000")
                return self._send(202, self.server.start_enrichment(limit=limit, generate_topics=bool(data.get("generate_topics", False))))
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
