#!/usr/bin/env python3
"""64.4/T2 — tokenizer-boundary coverage: student vocab V x teacher tokenizer, on the pinned P62 code-val.
Instrumentation, no gate. Feeds the D3 mapping-vs-fallback choice (decided at the sealed MVE gate).

For each student vocab V in {1024 (the existing P62 code BPE), 2048, 4096 (trained here on a sample of the code
TRAIN split, same 300 KB recipe corpus.py used for the 1024)} and each teacher tokenizer:
  (a) boundary coincidence  = % of student token boundaries that are also a teacher token boundary
  (b) byte coverage         = % of corpus BYTES sitting inside student tokens that are FULLY aligned
                              (both endpoints land on teacher boundaries) -> these are the tokens that can carry
                              mapped teacher top-K span probabilities; the rest falls back to plain CE.
Everything is done in BYTE space (code-val is strict utf-8; teacher char offsets are converted exactly).
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase62"))
from cartography import Bpe, train_bpe

VAL_TXT = "results/phase62/code_val.txt"
BPE1024 = "results/phase62/bpe1024_code.bin"
TRAIN_IDS = "results/phase62/code_train.u16"
BPE_TRAIN_BYTES = 300_000          # same sample size corpus.py used to train the existing V=1024
TEACHERS = [
    ("Qwen2.5-Coder (0.5/1.5/3/7B)", "Qwen/Qwen2.5-Coder-1.5B"),
    ("starcoder2-3b",                "bigcode/starcoder2-3b"),
    ("deepseek-coder-1.3b",          "deepseek-ai/deepseek-coder-1.3b-base"),
]

raw = open(VAL_TXT, "rb").read()
text = raw.decode("utf-8")                      # strict: byte<->char mapping is exact
N = len(raw)

# char index -> byte offset (exact; the val has a few multi-byte chars)
cb = np.zeros(len(text) + 1, dtype=np.int64)
acc = 0
for i, ch in enumerate(text):
    acc += len(ch.encode("utf-8")); cb[i + 1] = acc
assert cb[-1] == N

# ---- student side: byte-level BPE boundaries -------------------------------------------------
base = Bpe.load(BPE1024)
train_raw = b"".join(base.exp_bytes[i] for i in np.fromfile(TRAIN_IDS, dtype="<u2"))
print(f"train split: {len(train_raw)} raw bytes; BPE sample = {BPE_TRAIN_BYTES}", file=sys.stderr)

students = {}
students[1024] = base                            # the existing, pinned P62 code BPE
for V in (2048, 4096):
    t0 = time.time()
    students[V] = train_bpe(train_raw[:BPE_TRAIN_BYTES], V, verbose=False)
    print(f"trained student BPE V={V} in {time.time()-t0:.0f}s", file=sys.stderr)

def student_bounds(bpe):
    """END byte-offset of every student token (byte space)."""
    ids = bpe.encode(raw)
    lens = np.fromiter((bpe.exp_len[i] for i in ids), dtype=np.int64, count=len(ids))
    ends = np.cumsum(lens)
    assert ends[-1] == N, (ends[-1], N)
    return ends

# ---- teacher side: HF fast-tokenizer boundaries, converted to byte space ---------------------
from transformers import AutoTokenizer
def teacher_bounds(hf_id):
    tok = AutoTokenizer.from_pretrained(hf_id)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)   # chat-template OFF, raw text
    offs = enc["offset_mapping"]
    ends = {int(cb[e]) for (s, e) in offs if e > s}     # char->byte, skip empty spans
    return ends, len(offs)

print(f"\n==== 64.4/T2 tokenizer-boundary coverage (code-val, {N} bytes) ====")
print(f"{'student V':>10} {'stu tok':>9} {'B/tok':>6} | " +
      " | ".join(f"{n:^30}" for n, _ in TEACHERS))
print(f"{'':>10} {'':>9} {'':>6} | " + " | ".join(f"{'(a)bnd (b)byte  seg=B/stu/tea':^30}" for _ in TEACHERS))

tb_cache = {}
for name, hf in TEACHERS:
    tb_cache[hf] = teacher_bounds(hf)

for V in (1024, 2048, 4096):
    ends = student_bounds(students[V])
    starts = np.concatenate(([0], ends[:-1]))
    ntok = len(ends)
    cells = []
    for name, hf in TEACHERS:
        tset, ttok = tb_cache[hf]
        tset0 = tset | {0}
        coinc = sum(1 for e in ends if int(e) in tset)                       # (a)
        aligned_bytes = sum(int(e) - int(s) for s, e in zip(starts, ends)
                            if int(s) in tset0 and int(e) in tset)           # (b)
        # (c) granularity of the segments the coinciding boundaries cut the corpus into: span-KD attaches HERE,
        #     and these segments tile 100% of the bytes (unlike (b), which is the strict 1:1-token case).
        nseg = max(coinc, 1)
        seg_bytes, seg_stu, seg_tea = N / nseg, ntok / nseg, ttok / nseg
        cells.append(f"{100.0*coinc/ntok:6.2f}% {100.0*aligned_bytes/N:6.2f}% {seg_bytes:5.1f}B/{seg_stu:.1f}s/{seg_tea:.1f}t")
    print(f"{V:>10} {ntok:>9} {N/ntok:6.2f} | " + " | ".join(f"{c:^30}" for c in cells))

print("\nteacher token counts on the same 1.5 MB: " +
      ", ".join(f"{n}={tb_cache[hf][1]}" for n, hf in TEACHERS))
print("(a) = student boundaries that are also teacher boundaries; (b) = bytes in FULLY-aligned student tokens")
print("     (both endpoints on teacher boundaries) -> the KD-mappable fraction; the rest falls back to plain CE.")
