"""Stage 2 test 2b, after the verdict: where the cell's dBPB comes from.  A diagnosis, not a cell: it judges nothing and moves no line.

Float, on the states already dumped, every composed distribution against the exact head on the documents' real next tokens.  For each
preview and K: dBPB as the panel counts it (mean over documents, bootstrap CI), the share from targets inside and outside the candidates.

Tails:
    offset    the registered tail: preview + mean(exact - preview) over the candidates
    none      the preview alone
    const     offset + c, with c fitted on dev so the tail's mass is right on average
    affine    a * preview + b, a and b fitted on dev by mass-weighted least squares over the tail

c, a and b come from dev only; the documents do not choose them.  The row `offset` of tern_tr at K = 1,024 must give back the panel's
+0.0114: that is the parity check, and without it the rest says nothing.

Usage: python -I benchmarks/stage2/s2t_tail_probe.py --json benchmarks/stage2/results/s2t_tail_probe.json
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
sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))
import s2t_head as T  # noqa: E402

CELLS = [("tern_tr", 1024), ("tern_tr", 4096), ("tern_tr", 16384), ("tern_im", 4096), ("tern_im", 16384), ("int4", 1024), ("int4", 4096)]
T.CHUNK = 256                                  # beside the trainer on the same card
FIT_ROWS = 20000
DRAWS = 10000


def tail_terms(ze: torch.Tensor, zp: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    off = (ze.gather(1, c) - zp.gather(1, c)).mean(1, keepdim=True)
    mask = torch.ones_like(ze, dtype=torch.bool)
    mask.scatter_(1, c, False)
    return off, mask


def fit(p: dict, k: int, W: torch.Tensor, H: torch.Tensor) -> dict:
    """On dev: the constant that fixes the tail's mass, and the mass-weighted affine map from preview to exact."""
    logr, sw, sx, sy, sxx, sxy = [], 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(0, H.shape[0], T.CHUNK):
        h = H[i:i + T.CHUNK].to(T.DEV)
        ze, zp = h @ W.T, T.logits(p, h)
        c = zp.topk(k, 1).indices
        off, mask = tail_terms(ze, zp, c)
        lse_e = ze.masked_fill(~mask, -torch.inf).logsumexp(1)
        lse_p = (zp + off).masked_fill(~mask, -torch.inf).logsumexp(1)
        logr.append((lse_e - lse_p).double().cpu())
        w = (ze - lse_e[:, None]).exp() * mask
        x, y = zp.double(), ze.double()
        w = w.double()
        sw += float(w.sum()); sx += float((w * x).sum()); sy += float((w * y).sum())
        sxx += float((w * x * x).sum()); sxy += float((w * x * y).sum())
    a = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
    return {"const": float(torch.cat(logr).mean()), "a": a, "b": (sy - a * sx) / sw}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    from c50_compare import byte_lengths
    import c50_a4_checks as A4
    W = torch.load(T.WORK / "previews" / "exact.pt", weights_only=True)["w"].to(T.DEV).float()
    pos = np.fromfile(T.WORK / "h" / "docs.pos.bin", T.REC)
    H = T.hidden(T.WORK / "h" / "docs.h.f32")
    Hd = T.hidden(T.WORK / "h" / "dev.h.f32", FIT_ROWS)
    if H.shape[0] != len(pos):
        raise SystemExit("docs: states and records differ")
    tgt = torch.from_numpy(pos["target"].astype(np.int64))
    blen = byte_lengths(Path(A4.TOKENIZER))[pos["target"]].astype(np.float64)
    docs = np.unique(pos["doc"])
    print("CONFIG", json.dumps({"cells": CELLS, "fit_rows": int(Hd.shape[0]), "positions": len(pos), "dev": T.DEV}), flush=True)
    out = []
    for name in sorted({c[0] for c in CELLS}):
        p = {k: v.to(T.DEV).float() for k, v in torch.load(T.WORK / "previews" / f"{name}.pt", weights_only=True).items()}
        for _, k in [c for c in CELLS if c[0] == name]:
            f = fit(p, k, W, Hd)
            d = {t: np.zeros(len(pos)) for t in ("offset", "none", "const", "affine")}
            inc = np.zeros(len(pos), bool)
            for i in range(0, H.shape[0], T.CHUNK):
                h = H[i:i + T.CHUNK].to(T.DEV)
                t = tgt[i:i + T.CHUNK].to(T.DEV)[:, None]
                ze, zp = h @ W.T, T.logits(p, h)
                c = zp.topk(k, 1).indices
                off, _ = tail_terms(ze, zp, c)
                ex = ze.gather(1, c)
                lp = ze.log_softmax(1).gather(1, t).squeeze(1)
                inc[i:i + T.CHUNK] = (c == t).any(1).cpu().numpy()
                for tail, z in (("offset", zp + off), ("none", zp.clone()), ("const", zp + off + f["const"]), ("affine", f["a"] * zp + f["b"])):
                    z.scatter_(1, c, ex)
                    d[tail][i:i + T.CHUNK] = (lp - z.log_softmax(1).gather(1, t).squeeze(1)).double().cpu().numpy()
            row = {"preview": name, "k": k, "fit": f, "target_in_candidates": float(inc.mean()), "tails": {}}
            for tail, v in d.items():
                dd = np.array([v[pos["doc"] == x].sum() / np.log(2) / blen[pos["doc"] == x].sum() for x in docs])
                rng = np.random.default_rng(T.SEED)
                bb = dd[rng.integers(0, len(dd), (DRAWS, len(dd)))].mean(1)
                row["tails"][tail] = {"dbpb": float(dd.mean()), "ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
                                      "nats_from_candidates": float(v[inc].sum() / len(v)), "nats_from_tail": float(v[~inc].sum() / len(v))}
            out.append(row)
            print(json.dumps({x: row[x] for x in ("preview", "k", "target_in_candidates")}),
                  {t: (round(r["dbpb"], 4), [round(x, 4) for x in r["ci95"]]) for t, r in row["tails"].items()}, {x: round(y, 3) for x, y in f.items()}, flush=True)
            a.json.write_text(json.dumps({"note": "diagnosis after the verdict; judges nothing", "cells": out}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
