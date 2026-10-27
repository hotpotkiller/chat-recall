"""Check the complete zero-dependency toolchain without changing the machine."""
import shutil
import sqlite3
import sys

if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ required; obtain it from python.org or your package manager")
connection = sqlite3.connect(":memory:")
connection.execute("CREATE VIRTUAL TABLE capability USING fts5(text)")
connection.close()
if not shutil.which("git"):
    raise SystemExit("Git is required for development workflow commands")
print(f"Ready: Python {sys.version.split()[0]}, SQLite {sqlite3.sqlite_version} with FTS5, Git")
print("No dependencies or models were installed. Ollama is optional for semantic enrichment.")
