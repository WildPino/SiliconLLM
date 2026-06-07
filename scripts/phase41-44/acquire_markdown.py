"""
Phase 41a.2.4: Acquire Technical Markdown corpus (64MB sample).

Walks 3 cloned technical-documentation repositories, collects all .md
files in deterministic order (sorted path per repo, repos concatenated
in declared order), concatenates with double-newline separator.

Excludes node_modules and .git directories.

Pre-requisite: each repo shallow-cloned into:
  data/external/markdown_corpus/kubernetes_website
  data/external/markdown_corpus/rust_book
  data/external/markdown_corpus/typescript_website

Output: experiments/phase41a/corpora/markdown_64mb.txt
"""
import os
import sys
import hashlib

REPO_DIRS = [
    "data/external/markdown_corpus/kubernetes_website",
    "data/external/markdown_corpus/rust_book",
    "data/external/markdown_corpus/typescript_website",
    "data/external/markdown_corpus/mdn",
]
SAMPLE_DIR = "experiments/phase41a/corpora"
SAMPLE_FILE = os.path.join(SAMPLE_DIR, "markdown_64mb.txt")
SAMPLE_SIZE = 64 * 1024 * 1024

EXCLUDE_DIRS = {".git", "node_modules", ".vscode", ".idea", "dist", "build", "_book"}

os.makedirs(SAMPLE_DIR, exist_ok=True)

for repo in REPO_DIRS:
    if not os.path.isdir(repo):
        print(f"ERROR: {repo} not found.")
        print("Run step 1 (git clone --depth 1 ...) first.")
        sys.exit(1)

if os.path.exists(SAMPLE_FILE) and os.path.getsize(SAMPLE_FILE) == SAMPLE_SIZE:
    print(f"Sample already exists at full size: {SAMPLE_FILE}")
    print("Delete first if you want to rebuild.")
    sys.exit(0)

# Collect .md files in deterministic order
print("Walking markdown corpus trees...")
all_files = []
for repo in REPO_DIRS:
    repo_files = []
    for root, dirs, files in os.walk(repo):
        # Filter excluded dirs in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".md"):
                repo_files.append(os.path.join(root, f))
    repo_files.sort()
    print(f"  {repo}: {len(repo_files)} .md files")
    all_files.extend(repo_files)

print(f"Total: {len(all_files)} .md files")

# Concatenate until SAMPLE_SIZE
print(f"Building {SAMPLE_SIZE} byte sample at {SAMPLE_FILE}...")
total = 0
n_files = 0
skipped = 0
with open(SAMPLE_FILE, "wb") as out:
    for path in all_files:
        try:
            with open(path, "rb") as f:
                content = f.read()
        except Exception:
            skipped += 1
            continue
        out.write(content)
        out.write(b"\n\n")
        total += len(content) + 2
        n_files += 1
        if total >= SAMPLE_SIZE:
            break

if total > SAMPLE_SIZE:
    with open(SAMPLE_FILE, "rb") as f:
        data = f.read(SAMPLE_SIZE)
    with open(SAMPLE_FILE, "wb") as f:
        f.write(data)

actual = os.path.getsize(SAMPLE_FILE)
with open(SAMPLE_FILE, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()

shortfall = SAMPLE_SIZE - actual
status = "FULL" if shortfall == 0 else f"SHORT by {shortfall/1024/1024:.2f} MB"

print(f"\n=== Markdown 64MB sample ===")
print(f"  File:     {SAMPLE_FILE}")
print(f"  Size:     {actual} bytes ({actual/1024/1024:.2f} MB) [{status}]")
print(f"  Files:    {n_files} .md concatenated, {skipped} skipped")
print(f"  SHA-256:  {sha[:32]}...")
print("Done.")
