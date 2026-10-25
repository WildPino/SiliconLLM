"""C50-T1 part 2b review, addendum 1: null LoRAs with A1's per-tensor size and no learned map.

From A1's `final.pt` (keys lora.<i>_<m>.A [r, n_in] and .B [n_out, r], dW = B A), writes a file of the same keys and
shapes, so that `c50_t1_merge.py` merges it unchanged:
- `rand`: A' and B' Gaussian, B' scaled so that ||B' A'||_F = ||B A||_F per tensor;
- `rot`:  A' = A, B' = c B Q with Q a random r x r orthogonal matrix, c so that ||B' A'||_F = ||B A||_F: A1's input and
          output subspaces, with the learned pairing between them scrambled;
- `self`: A' = A, B' = B (the planted control: the merge's deltas must equal A1's bit for bit).

    python -I benchmarks/chatbot50/c50_t1_null.py --lora final.pt --kind rand|rot|self --seed S --out null.pt --json REC
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_t1_merge import deltas, sha256  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--kind", required=True, choices=["rand", "rot", "self"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    sd = torch.load(a.lora, map_location="cpu", weights_only=True)
    g = torch.Generator().manual_seed(a.seed)
    rec = {"cell": "C50-T1 part 2b review, null LoRA", "kind": a.kind, "seed": a.seed, "lora": str(a.lora),
           "lora_sha256": sha256(a.lora), "tensors": {}}
    print("CONFIG", json.dumps({k: rec[k] for k in ("kind", "seed", "lora", "lora_sha256")}), flush=True)
    out = {}
    for k in sorted(sd):
        if not k.endswith(".A"):
            continue
        A, B = sd[k].double(), sd[k[:-2] + ".B"].double()
        r = A.shape[0]
        if a.kind == "self":
            A2, B2 = A, B
        elif a.kind == "rand":
            A2 = torch.randn(A.shape, generator=g, dtype=torch.float64)
            B2 = torch.randn(B.shape, generator=g, dtype=torch.float64)
        else:
            q, rr = torch.linalg.qr(torch.randn((r, r), generator=g, dtype=torch.float64))
            A2, B2 = A, B @ (q * torch.sign(torch.diagonal(rr)))           # Haar: fix the signs of QR's diagonal
        if a.kind != "self":
            B2 = B2 * (torch.linalg.norm(B @ A) / torch.linalg.norm(B2 @ A2))
        out[k], out[k[:-2] + ".B"] = (A2.float(), B2.float()) if a.kind != "self" else (sd[k], sd[k[:-2] + ".B"])
    torch.save(out, a.out)
    d1, d2 = deltas(a.lora), deltas(a.out)
    assert set(d1) == set(d2) and len(d1) == 100
    for n in sorted(d1):
        x, y = d1[n].double(), d2[n].double()
        rec["tensors"][n] = {"norm_a1": float(x.norm()), "norm_null": float(y.norm()),
                             "cos_with_a1": float((x * y).sum() / (x.norm() * y.norm()))}
    rel = [abs(v["norm_null"] / v["norm_a1"] - 1) for v in rec["tensors"].values()]
    cos = [abs(v["cos_with_a1"]) for v in rec["tensors"].values()]
    rec["norm_rel_err_max"], rec["abs_cos_median"] = float(max(rel)), float(np.median(cos))
    rec["out_sha256"] = sha256(a.out)
    if a.kind == "self":
        rec["self_bit_exact"] = all(torch.equal(d1[n], d2[n]) for n in d1)
        rec["ok"] = rec["self_bit_exact"]
    else:
        rec["ok"] = rec["norm_rel_err_max"] < 1e-4 and rec["abs_cos_median"] < 0.2
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("NULL", a.kind, "ok" if rec["ok"] else "FAIL", {k: rec.get(k) for k in
          ("norm_rel_err_max", "abs_cos_median", "self_bit_exact", "out_sha256")}, flush=True)
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
