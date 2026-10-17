"""C50-A4 diagnostic 1 (post hoc, after the verdict; it decides nothing): where the document dBPB comes from.

On the recorded A4 document outputs (no new run), per candidate arm against BF16:
  - the net bits, all positions and outside the positions where BF16's target log-probability is below TAIL;
  - top-1 agreement with BF16's top-1, and top-1 accuracy against the target;
  - the mean absolute per-position difference of the target log-probability (nats);
  - the share of the tail positions per category.
It answers one question about the instrument: does the planted Q2_K's G1 pass come from the tail?

Usage:
    python -I benchmarks/chatbot50/c50_a4_diag_tail.py --work E:/.../a4 --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_compare import byte_lengths  # noqa: E402
import c50_a4_checks as A4  # noqa: E402

REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])
TAIL = -15.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    w = Path(args.work)
    r = {a: np.fromfile(w / a / "docs.pos.bin", dtype=REC) for a in ("bf16", "p0", "pc")}
    b = r["bf16"]
    nbytes = float(byte_lengths(Path(A4.TOKENIZER))[b["target"]].astype(np.float64).sum())
    meta = [json.loads(ln) for ln in (w / "inputs" / "docs_meta.jsonl").read_text(encoding="utf-8").splitlines()]
    cat = {m["index"]: m["category"] for m in meta}
    tail = b["logp_target"] < TAIL
    out = {"cell": "C50-A4 diagnostic 1 (post hoc)", "tail_logp_below": TAIL, "positions": int(len(b)),
           "tail_positions": int(tail.sum()), "bf16_top1_accuracy": float((b["top1"] == b["target"]).mean()),
           "tail_by_category": {c: int((tail & np.isin(b["doc"], [k for k, v in cat.items() if v == c])).sum())
                                for c in sorted(set(cat.values()))},
           "bf16_tail_median_logp_target": float(np.median(b["logp_target"][tail])),
           "bf16_tail_median_logp_top1": float(np.median(b["logp_top1"][tail])), "arms": {}}
    for a in ("p0", "pc"):
        x = r[a]
        if not ((x["doc"] == b["doc"]).all() and (x["pos"] == b["pos"]).all() and (x["target"] == b["target"]).all()):
            raise SystemExit(f"{a}: positions differ from BF16")
        dbits = (b["logp_target"].astype(np.float64) - x["logp_target"].astype(np.float64)) / np.log(2)
        out["arms"][a] = {"net_dbits": float(dbits.sum()), "net_dbits_outside_tail": float(dbits[~tail].sum()),
                          "net_dbits_tail": float(dbits[tail].sum()), "dbpb_pooled": float(dbits.sum() / nbytes),
                          "top1_agree_bf16": float((x["top1"] == b["top1"]).mean()),
                          "top1_accuracy": float((x["top1"] == x["target"]).mean()),
                          "mean_abs_dlogp_nats": float(np.abs(dbits * np.log(2)).mean()),
                          "tail_median_logp_target": float(np.median(x["logp_target"][tail]))}
    print(json.dumps(out, indent=1))
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
