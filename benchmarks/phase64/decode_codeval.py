#!/usr/bin/env python3
"""64.4/T1 helper: reconstruct the EXACT pinned P62 code-val raw bytes from code_val.u16 (BPE1024 tokens),
so cross-tokenizer teacher BPB is computed on the identical byte slice all teachers see.
Output: results/phase62/code_val.txt (raw bytes) + prints the byte count (the BPB denominator)."""
import sys, os, struct, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase62"))
from cartography import Bpe

BPE = "results/phase62/bpe1024_code.bin"
IDS = "results/phase62/code_val.u16"
OUT = "results/phase62/code_val.txt"

bpe = Bpe.load(BPE)
ids = np.fromfile(IDS, dtype="<u2")
raw = b"".join(bpe.exp_bytes[i] for i in ids)
with open(OUT, "wb") as f:
    f.write(raw)
# sanity: byte length via exp_len must match the concatenation
declared = sum(bpe.exp_len[i] for i in ids)
print(f"decoded {len(ids)} tokens -> {len(raw)} raw bytes (exp_len sum {declared}, match={len(raw)==declared})")
print(f"wrote {OUT}  |  BPB denominator = {len(raw)} bytes")
