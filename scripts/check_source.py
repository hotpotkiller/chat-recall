"""Offline syntax checks with no dependency installation or network calls."""
import ast
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
bad = []
count = 0
for directory in ("chat_recall", "scripts", "tests"):
    for path in (root / directory).rglob("*.py"):
        count += 1
        try:
            text = path.read_text("utf-8")
            ast.parse(text, filename=str(path.relative_to(root)))
            if any(line.rstrip() != line for line in text.splitlines()):
                bad.append(f"{path.relative_to(root)}: trailing whitespace")
        except (SyntaxError, UnicodeError) as exc:
            bad.append(str(exc))
if bad:
    print("\n".join(bad), file=sys.stderr)
    raise SystemExit(1)
print(f"Syntax and whitespace checks passed for {count} Python files")
