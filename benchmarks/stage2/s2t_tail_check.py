"""Stage 2 test 2b, after the verdict: does the affine tail of s2t_tail_probe.py hold where nothing chose it?  A diagnosis, it judges nothing.

For tern_tr at K = 1,024 and tern_im at K = 4,096, a and b as the probe fitted them on dev's first 20,000 rows:
    dev's other rows   dBPB on real next tokens, a part of the corpus neither the fit nor the choice among tails saw;
    the documents      KL over the whole vocabulary and the tail's mass, exact and composed: the other lines of the test.

Usage: python -I benchmarks/stage2/s2t_tail_check.py --probe benchmarks/stage2/results/s2t_tail_probe.json --json benchmarks/stage2/results/s2t_tail_check.json
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

T.CHUNK = 256
CELLS = [("tern_tr", 1024), ("tern_im", 4096)]
FIT_ROWS = 20000


def composed(ze: torch.Tensor, zp: torch.Tensor, c: torch.Tensor, f: dict) -> dict:
    off = (ze.gather(1, c) - zp.gather(1, c)).mean(1, keepdim=True)
    ex = ze.gather(1, c)
    out = {}
    for tail, z in (("offset", zp + off), ("none", zp.clone()), ("affine", f["a"] * zp + f["b"])):
        z.scatter_(1, c, ex)
        out[tail] = z.log_softmax(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    from c50_compare import byte_lengths
    import c50_a4_checks as A4
    fits = {(r["preview"], r["k"]): r["fit"] for r in json.loads(a.probe.read_text(encoding="utf-8"))["cells"]}
    W = torch.load(T.WORK / "previews" / "exact.pt", weights_only=True)["w"].to(T.DEV).float()
    blen_of = byte_lengths(Path(A4.TOKENIZER))
    dev = np.fromfile(T.WORK / "h" / "dev.pos.bin", T.REC)
    Hv = T.hidden(T.WORK / "h" / "dev.h.f32")[FIT_ROWS:]
    tv = torch.from_numpy(dev["target"][FIT_ROWS:].astype(np.int64))
    Hd = T.hidden(T.WORK / "h" / "docs.h.f32")
    print("CONFIG", json.dumps({"cells": CELLS, "dev_rows": int(Hv.shape[0]), "docs_rows": int(Hd.shape[0])}), flush=True)
    bytes_dev = float(blen_of[dev["target"][FIT_ROWS:]].astype(np.float64).sum())
    res = []
    for name, k in CELLS:
        p = {x: v.to(T.DEV).float() for x, v in torch.load(T.WORK / "previews" / f"{name}.pt", weights_only=True).items()}
        f = fits[(name, k)]
        d = {t: 0.0 for t in ("offset", "none", "affine")}
        for i in range(0, Hv.shape[0], T.CHUNK):
            h = Hv[i:i + T.CHUNK].to(T.DEV)
            t = tv[i:i + T.CHUNK].to(T.DEV)[:, None]
            ze, zp = h @ W.T, T.logits(p, h)
            lp = ze.log_softmax(1).gather(1, t)
            for tail, lq in composed(ze, zp, zp.topk(k, 1).indices, f).items():
                d[tail] += float((lp - lq.gather(1, t)).double().sum())
        kl = {t: 0.0 for t in d}
        mass_c = {t: 0.0 for t in d}
        mass_e = 0.0
        for i in range(0, Hd.shape[0], T.CHUNK):
            h = Hd[i:i + T.CHUNK].to(T.DEV)
            ze, zp = h @ W.T, T.logits(p, h)
            c = zp.topk(k, 1).indices
            lp = ze.log_softmax(1)
            mass_e += float((1 - lp.gather(1, c).exp().sum(1)).sum())
            for tail, lq in composed(ze, zp, c, f).items():
                kl[tail] += float((lp.exp() * (lp - lq)).sum())
                mass_c[tail] += float((1 - lq.gather(1, c).exp().sum(1)).sum())
        n = Hd.shape[0]
        row = {"preview": name, "k": k, "fit": f, "dev_dbpb": {t: v / np.log(2) / bytes_dev for t, v in d.items()},
               "docs_kl_full_vocab": {t: v / n for t, v in kl.items()}, "docs_tail_mass_exact": mass_e / n,
               "docs_tail_mass_composed": {t: v / n for t, v in mass_c.items()}}
        res.append(row)
        print(json.dumps(row), flush=True)
        a.json.write_text(json.dumps({"note": "diagnosis after the verdict; judges nothing", "cells": res}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
