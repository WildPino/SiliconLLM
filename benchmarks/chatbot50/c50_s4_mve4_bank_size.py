"""C50-S4 MVE-4: the memory bank's size on S4's training split (S3 block 1, shards 0-469), from n-gram counts.

A row exists for every trigram and every bigram that occurs at least twice in the training split (S4 design, decision
5); a value is 2048 fp16 numbers.  Prints rows, bytes and m = bytes / T0's stored bytes.

    python -I benchmarks/chatbot50/c50_s4_mve4_bank_size.py --json benchmarks/chatbot50/results/c50_s4_mve4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

C = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
TRAIN_SHARDS, SHARD_DOCS, WIDTH, T0_BYTES, MIN_COUNT = 470, 48, 2048, 9_912_067_296, 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path)
    a = ap.parse_args()
    man = [json.loads(x) for x in (C / "s3/teacher_v1/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    last_doc = max(r["docs"][1] for r in man if r["shard"] < TRAIN_SHARDS)
    assert last_doc == TRAIN_SHARDS * SHARD_DOCS - 1, last_doc
    bi, tri, n_tok = [], [], 0
    with open(C / "s3/corpus_v1/teacher.tsv", encoding="ascii") as f:
        for line in f:
            d, s = line.split("\t", 1)
            if int(d) > last_doc:
                break
            x = np.array(s.split(","), dtype=np.int64)
            n_tok += len(x)
            bi.append((x[:-1] << 18) | x[1:])
            tri.append((x[:-2] << 36) | (x[1:-1] << 18) | x[2:])
    out = {"cell": "C50-S4 MVE-4", "train_docs": last_doc + 1, "train_tokens": n_tok, "min_count": MIN_COUNT}
    rows = 0
    for name, arr in (("bigram", bi), ("trigram", tri)):
        _, c = np.unique(np.concatenate(arr), return_counts=True)
        kept = int((c >= MIN_COUNT).sum())
        out[name] = {"distinct": int(len(c)), "rows_count_ge_2": kept,
                     "occurrences_covered": float(c[c >= MIN_COUNT].sum() / c.sum())}
        rows += kept
    out["rows"], out["bytes_fp16"] = rows, rows * WIDTH * 2
    out["m_vs_t0_stored"] = out["bytes_fp16"] / T0_BYTES
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
