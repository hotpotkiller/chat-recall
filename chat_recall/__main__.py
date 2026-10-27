"""Terminal entrypoint shared by humans and all coding agents."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import webbrowser


def default_data_dir() -> Path:
    env_dir = os.environ.get("CHAT_RECALL_DATA_DIR")
    if env_dir is not None and env_dir.strip():
        return Path(env_dir.strip()).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Chat Recall"
    if sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app is not None and local_app.strip():
            return Path(local_app.strip()).expanduser() / "Chat Recall"
        return Path.home() / "AppData" / "Local" / "Chat Recall"
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data is not None and xdg_data.strip():
        return Path(xdg_data.strip()).expanduser() / "chat-recall"
    return Path.home() / ".local" / "share" / "chat-recall"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chat Recall — search actual ChatGPT passages")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Start the loopback application")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--data-dir", type=Path, default=default_data_dir())
    serve.add_argument("--demo", action="store_true", help="Import synthetic example conversations")
    serve.add_argument("--open", action="store_true", help="Open the local app in your browser")
    serve.add_argument("--extension-origin", action="append", default=[])
    serve.add_argument("--embedding-model", help="Opt in to local Ollama embeddings; no automatic download")
    serve.add_argument("--summary-model", default="qwen3:1.7b")
    serve.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    serve.add_argument("--cloud-rerank", action="store_true", help="Offer opt-in cloud reranking; reads COHERE_API_KEY")
    serve.add_argument("--rerank-model", default="rerank-v5.0-pro")
    for command in ("import", "search", "status", "enrich"):
        p = sub.add_parser(command)
        p.add_argument("--db", type=Path, default=default_data_dir() / "recall.sqlite3")
        if command == "import":
            p.add_argument("file", type=Path)
            p.add_argument("--account", default="default")
            p.add_argument("--source", default="chatgpt")
        elif command == "search":
            p.add_argument("query")
            p.add_argument("--mode", choices=["exact", "smart"], default="exact")
            p.add_argument("--case-sensitive", action="store_true")
            p.add_argument("--limit", type=int, default=20)
            p.add_argument("--account")
            p.add_argument("--embedding-model", help="Explicit local Ollama model for Smart query embeddings; must match indexed vectors")
            p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
        elif command == "enrich":
            p.add_argument("--model", default="qwen3-embedding:0.6b")
            p.add_argument("--summary-model", default="qwen3:1.7b")
            p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
            p.add_argument("--limit", type=int, default=100)
            p.add_argument("--topics", action="store_true")
    args = parser.parse_args(argv)
    from .search import SearchIndex
    if args.command == "serve":
        from .server import FIXTURE, RecallServer, load_token
        provider = None
        if args.embedding_model:
            from .providers import OllamaProvider
            provider = OllamaProvider(base_url=args.ollama_url, embedding_model=args.embedding_model,
                                      topic_model=args.summary_model)
        reranker = None
        if args.cloud_rerank:
            from .rerank import CohereReranker
            reranker = CohereReranker(os.environ.get("COHERE_API_KEY", ""), model=args.rerank_model)
        args.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        db = args.data_dir / "recall.sqlite3"
        index = SearchIndex(db, embedding_provider=provider)
        try:
            if args.demo:
                index.import_payload(json.loads(FIXTURE.read_text("utf-8")), account_id="demo")
        finally:
            index.close()
        server = RecallServer(("127.0.0.1", args.port), db, token=load_token(args.data_dir),
                              extension_origins=args.extension_origin, provider=provider, reranker=reranker)
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Chat Recall is running at {url}", flush=True)
        print("Private search phrases and bridge tokens are not logged. Ctrl+C stops the app.", flush=True)
        if args.open:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    args.db.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    provider = None
    if args.command == "enrich":
        from .providers import OllamaProvider
        provider = OllamaProvider(base_url=args.ollama_url, embedding_model=args.model,
                                  topic_model=args.summary_model)
    elif args.command == "search" and args.embedding_model:
        from .providers import OllamaProvider
        provider = OllamaProvider(base_url=args.ollama_url, embedding_model=args.embedding_model)
    index = SearchIndex(args.db, embedding_provider=provider)
    try:
        if args.command == "import":
            result = index.import_payload(json.loads(args.file.read_text("utf-8")), source_id=args.source, account_id=args.account)
        elif args.command == "search":
            result = index.search(args.query, mode=args.mode, case_sensitive=args.case_sensitive,
                                  limit=args.limit, account_id=args.account)
        elif args.command == "enrich":
            result = index.enrich(limit=args.limit, generate_topics=args.topics)
        else:
            result = index.status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        index.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Chat Recall: {exc}", file=sys.stderr)
        raise SystemExit(1)
