"""S2N: mini's expert load, for cell F.  ks_stk4's router on the documents panel at k = 8: per MoE layer, each expert's share of the slots, sorted
from the busiest.  F samples its routing from these shares (flash's layer i takes mini's layer round(i x 19 / 31)), so its chunks carry a real
load imbalance.  GPU, a few minutes.

Usage: python -I benchmarks/stage2/s2n_route_hist.py --json benchmarks/stage2/results/s2n_route_hist.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2n_core as C  # noqa: E402
import s2n_train as T  # noqa: E402

DOCS_USED = 24


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    layers, embed, head, out_norm, _ = T.load(dev, "none")
    counts = np.zeros((T.SH.layers, T.SH.experts), dtype=np.int64)
    real_route = C.route
    cur = {"i": 0}

    def counting(x, gate, bias, sh, form):
        ids, wts = real_route(x, gate, bias, sh, form)
        counts[cur["i"]] += np.bincount(ids.reshape(-1).cpu().numpy(), minlength=sh.experts)
        return ids, wts
    C.route = counting
    lines = T.DOCS.read_text(encoding="utf-8").splitlines()[:DOCS_USED]
    tokens = 0
    for line in lines:
        toks = torch.tensor([int(x) for x in line.split("\t", 1)[1].split(",")][:2048])[None]
        h = embed[toks.reshape(-1)].view(1, -1, T.SH.d).to(dev)
        cs = C.context(tuple(toks.shape), T.SH, dev, h.dtype)
        with torch.no_grad():
            for L in layers:
                cur["i"] = L.i
                w = C.to_gpu(L.host, dev)
                h = C.layer_forward(L, w, h, cs, T.SH, T.FORMS["k8"])
                del w
        tokens += toks.shape[1]
    shares = []
    for i in range(1, T.SH.layers):
        s = np.sort(counts[i])[::-1] / counts[i].sum()
        shares.append(s.tolist())
    peak = [float(s[0] * T.SH.experts) for s in map(np.array, shares)]
    out = {"tokens": tokens, "docs": len(lines), "form": "k = 8", "shares_sorted": shares, "busiest_over_mean": peak,
           "busiest_over_mean_median": float(np.median(peak))}
    a.json.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "shares_sorted"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
