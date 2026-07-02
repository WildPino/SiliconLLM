"""
Phase 41a.2.1: Acquire TinyStories corpus (64MB sample).

Uses HuggingFace streaming mode to avoid downloading the full ~2GB
dataset. Concatenates stories with "\\n\\n" separator (no special tokens
introduced — keeps the byte stream silicon-neutral).

Output: experiments/phase41a/corpora/tinystories_64mb.txt
"""
import os
import sys
import hashlib
from datasets import load_dataset
from tqdm import tqdm

SAMPLE_DIR = "experiments/phase41a/corpora"
SAMPLE_FILE = os.path.join(SAMPLE_DIR, "tinystories_64mb.txt")
SAMPLE_SIZE = 64 * 1024 * 1024  # 64 MB exactly

os.makedirs(SAMPLE_DIR, exist_ok=True)

# Skip if already done
if os.path.exists(SAMPLE_FILE) and os.path.getsize(SAMPLE_FILE) == SAMPLE_SIZE:
    print(f"Sample already exists: {SAMPLE_FILE} ({SAMPLE_SIZE} bytes). Skipping.")
    sys.exit(0)

print("Loading TinyStories from HuggingFace (streaming mode)...")
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

print(f"Building {SAMPLE_SIZE} byte sample to {SAMPLE_FILE}...")
total = 0
n_stories = 0
with open(SAMPLE_FILE, "wb") as f:
    pbar = tqdm(total=SAMPLE_SIZE, unit="B", unit_scale=True)
    for example in ds:
        data = (example["text"] + "\n\n").encode("utf-8")
        f.write(data)
        total += len(data)
        n_stories += 1
        pbar.update(len(data))
        if total >= SAMPLE_SIZE:
            break
    pbar.close()

# Trim to exactly SAMPLE_SIZE bytes
if total > SAMPLE_SIZE:
    with open(SAMPLE_FILE, "rb") as f:
        data = f.read(SAMPLE_SIZE)
    with open(SAMPLE_FILE, "wb") as f:
        f.write(data)

# Verify and report
actual = os.path.getsize(SAMPLE_FILE)
with open(SAMPLE_FILE, "rb") as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()

print(f"\n=== TinyStories 64MB sample ===")
print(f"  File:     {SAMPLE_FILE}")
print(f"  Size:     {actual} bytes ({actual/1024/1024:.2f} MB)")
print(f"  Stories:  {n_stories} (truncated at last story boundary)")
print(f"  SHA-256:  {sha256[:32]}...")
print("Done.")
