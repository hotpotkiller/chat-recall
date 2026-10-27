"""Package an exact committed snapshot; never include runtime or untracked data."""
import hashlib
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
output = root.parent / "Chat-Recall-0.1.0.zip"
if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root).returncode:
    raise SystemExit("Commit tracked changes before packaging an auditable snapshot.")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
subprocess.run(["git", "archive", "--format=zip", "--prefix=chat-recall/",
                "--output=" + str(output), commit], cwd=root, check=True)
digest = hashlib.sha256(output.read_bytes()).hexdigest()
output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", "utf-8")
output.with_suffix(".source.txt").write_text(f"Source commit: {commit}\nLicense: MIT\n", "utf-8")
print(f"Packaged commit {commit}: {output.name}; SHA-256 {digest}")
