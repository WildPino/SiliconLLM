"""C50-A5 part 1 diagnostics after the FAIL (numbers only; the verdict is the selection script's, unchanged).

    known positive  the full-rank preview (r = 2048, float32) through the same rank code as the candidates:
                    at K=1 it must equal the float32 oracle agreement up to float rounding, and recall at K
                    in {2, 8, 64} shows how far the float32-vs-Q8_K flips reach
    spectrum        fraction of the head's squared Frobenius mass in the top-r eigenvalues of W^T W
    strata          recall of the best candidate of each family, split by the engine's p(top1) bin

Usage:
    python -I benchmarks/chatbot50/c50_a5_diag.py --capture E:/.../a5/FIT12 --work E:/.../a5 \
        --out benchmarks/chatbot50/results/c50_a5_diag_fit.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from c50_a5_offline import D, REC, kmeans, q4_sim, ranks_of_top1

BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01)


def strata(hit: np.ndarray, ptop: np.ndarray) -> list[dict]:
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (ptop >= lo) & (ptop < hi)
        out.append({"p_top1": [lo, min(hi, 1.0)], "positions": int(m.sum()),
                    "recall": float(hit[m].mean()) if m.any() else None})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rec = np.fromfile(args.capture + ".pos.bin", dtype=REC)
    h = np.fromfile(args.capture + ".hid.f32", dtype=np.float32).reshape(-1, D)
    top1 = rec["top1"].astype(np.int64)
    ptop = np.exp(rec["logp_top1"].astype(np.float64))
    w = np.asarray(np.load(Path(args.work) / "head_q6k_dequant_f32.npy", mmap_mode="r"))
    rep = {"cell": "C50-A5 part 1 diagnostics (FIT)", "positions": int(len(rec)),
           "p_top1_strata_positions": [s["positions"] for s in strata(np.ones(len(rec), bool), ptop)]}

    gram = w.astype(np.float64).T @ w.astype(np.float64)
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    ev = evals[order]
    rep["spectrum_mass_top_r"] = {str(r): float(ev[:r].sum() / ev.sum()) for r in (128, 256, 512, 1024, 1536)}

    vfull = evecs[:, order].astype(np.float32)
    pfull = (w @ vfull).astype(np.float32)
    rk = ranks_of_top1(lambda x: (x @ vfull) @ pfull.T, h, top1)
    rk_f32 = ranks_of_top1(lambda x: x @ w.T, h, top1)
    rep["known_positive_full_rank"] = {
        "recall_at_K": {str(k): float((rk < k).mean()) for k in (1, 2, 8, 64)},
        "f32_head_recall_at_K": {str(k): float((rk_f32 < k).mean()) for k in (1, 2, 8, 64)},
        "rank_disagreements_vs_f32_head": int((rk != rk_f32).sum())}
    print("known positive", rep["known_positive_full_rank"], f"{time.time() - t0:.0f}s", flush=True)
    rep["f32_head_strata_at_1"] = strata(rk_f32 < 1, ptop)
    del pfull

    vr = evecs[:, order[:512]].astype(np.float32)
    p = q4_sim((w @ vr).astype(np.float32))
    rk = ranks_of_top1(lambda x: (x @ vr) @ p.T, h, top1)
    rep["preview_r512_q4sim_K1024_strata"] = strata(rk < 1024, ptop)
    rep["preview_r512_q4sim_rank_quantiles"] = {str(q): float(np.quantile(rk, q)) for q in (0.5, 0.9, 0.99, 0.999)}

    cent, assign = kmeans(w, 4096)
    sc_rank = np.empty(len(h), dtype=np.int64)
    cl = assign[top1]
    for s in range(0, len(h), 2048):
        sc = h[s:s + 2048] @ cent.T
        t = sc[np.arange(len(sc)), cl[s:s + 2048]]
        sc_rank[s:s + 2048] = (sc > t[:, None]).sum(axis=1)
    rep["ivf_C4096_P128_strata"] = strata(sc_rank < 128, ptop)
    rep["ivf_C4096_probe_rank_quantiles"] = {str(q): float(np.quantile(sc_rank, q)) for q in (0.5, 0.9, 0.99, 0.999)}
    rep["wall_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
