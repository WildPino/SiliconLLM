"""
Phase 41a.2.2: Acquire Python code corpus (64MB sample).

Walks multiple cloned Python repositories (stdlib + application code),
collects all .py files in deterministic order (sorted path per repo,
repos concatenated in declared order), concatenates with double-newline
separator. No artificial tokens introduced.

Pre-requisite: each repo cloned with --depth 1 into:
  data/external/the_stack_python/cpython
  data/external/the_stack_python/django

Output: experiments/phase41a/corpora/python_code_64mb.txt
"""
import os
import sys
import hashlib

REPO_DIRS = [
    "data/external/the_stack_python/cpython",
    "data/external/the_stack_python/django",
]
SAMPLE_DIR = "experiments/phase41a/corpora"
SAMPLE_FILE = os.path.join(SAMPLE_DIR, "python_code_64mb.txt")
SAMPLE_SIZE = 64 * 1024 * 1024

os.makedirs(SAMPLE_DIR, exist_ok=True)

# Verify all repos present
for repo in REPO_DIRS:
    if not os.path.isdir(repo):
        print(f"ERROR: {repo} not found.")
        print("Clone the missing repos first (see script header).")
        sys.exit(1)

if os.path.exists(SAMPLE_FILE) and os.path.getsize(SAMPLE_FILE) == SAMPLE_SIZE:
    print(f"Sample already exists at full size: {SAMPLE_FILE}")
    print("Delete it first if you want to rebuild.")
    sys.exit(0)

# Collect all .py files in deterministic order, per-repo
print("Walking Python source trees...")
all_files = []
for repo in REPO_DIRS:
    repo_files = []
    for root, dirs, files in os.walk(repo):
        if ".git" in dirs:
            dirs.remove(".git")
        for f in files:
            if f.endswith(".py"):
                repo_files.append(os.path.join(root, f))
    repo_files.sort()
    print(f"  {repo}: {len(repo_files)} .py files")
    all_files.extend(repo_files)

print(f"Total: {len(all_files)} .py files")

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

print(f"\n=== Python code 64MB sample ===")
print(f"  File:     {SAMPLE_FILE}")
print(f"  Size:     {actual} bytes ({actual/1024/1024:.2f} MB)")
print(f"  Files:    {n_files} concatenated, {skipped} skipped")
print(f"  SHA-256:  {sha[:32]}...")
print("Done.")
