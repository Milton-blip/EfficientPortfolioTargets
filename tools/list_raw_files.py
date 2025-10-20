#!/usr/bin/env python3
import subprocess, urllib.parse, os, pathlib, sys

# --- Hardcoded absolute project root ---
PROJECT_ROOT = pathlib.Path("/Users/tkeller/PycharmProjects/EfficientPortfolioTargets").resolve()
os.chdir(PROJECT_ROOT)

# --- Output file ---
OUT_PATH = PROJECT_ROOT / "file_list.txt"

print(f"📂 Project root: {PROJECT_ROOT}")
print(f"📄 Output file:  {OUT_PATH}")

# --- Git metadata ---
try:
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
except subprocess.CalledProcessError as e:
    sys.exit(f"❌ Git error: {e}")

owner_repo = origin.split("github.com/")[-1].replace(".git", "")
owner, repo = owner_repo.split("/", 1)

# --- Exclusion rules ---
EXCLUDE_CONTAINS = [
    "/.idea/",
    "/.venv/",
    "/.venv312/",
    "/.venv312ds/",
    "/subdirectory/",
]
EXCLUDE_NAMES = (".DS_Store",)

# --- Gather tracked files ---
try:
    files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
except subprocess.CalledProcessError:
    sys.exit("❌ Unable to list files (is this a Git repo?).")

urls = []
skipped = 0
for f in files:
    # Normalize path for matching
    path_str = f"/{f}/"
    if any(excl in path_str for excl in EXCLUDE_CONTAINS) or any(f.endswith(n) for n in EXCLUDE_NAMES):
        skipped += 1
        continue

    enc = urllib.parse.quote(f, safe="/-._~")
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{enc}"
    urls.append(url)

# --- Write output ---
OUT_PATH.write_text("\n".join(urls) + "\n", encoding="utf-8")

print(f"✅ Wrote {len(urls)} raw URLs to {OUT_PATH}")
print(f"🚫 Skipped {skipped} excluded files or directories")