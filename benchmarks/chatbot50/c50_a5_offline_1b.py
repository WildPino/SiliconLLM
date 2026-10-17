"""C50-A5 part 1b: the registered grid extension after the part 1 FAIL (numbers only).

Same capture, ground truth (engine top-1), controls and rank code as c50_a5_offline.py.  Extends the
preview family, which is the only one with a measured path to 0.999:

    rotated     W ~ (W V_r) V_r^T, r in {256, 512, 1024}; the preview matrix stored at 4.5, 3.5 or 2.5
                bits/weight (simulated, groups of 32, float16 scale); K in {1024 .. 65536}
    full rank   W itself stored at 3.5 or 2.5 bits/weight (no rotation); K in {8 .. 65536}

    q4sim 4.5   absmax / 7, levels -8..7
    q3sim 3.5   absmax / 3, levels -4..3
    t2sim 2.5   ternary (TWN): delta = 0.7 mean|g|, scale = mean of |g| above delta; stored in 2 bits

Bytes per token: preview V*r*bits/8 (+ D*r*4 B for V_r when rotated) + K rerank rows x 1680 B (Q6_K).

Usage:
    python -I benchmarks/chatbot50/c50_a5_offline_1b.py --model <q4_k_m.gguf> --capture E:/.../a5/FIT12 \
        --work E:/.../a5 --out benchmarks/chatbot50/results/c50_a5_offline_1b_fit.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from c50_a5_offline import D, REC, ROW_BYTES_Q6K, V, load_head, q4_sim, q8k, ranks_of_top1

RANKS = (256, 512, 1024)
KS_ROT = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
KS_FULL = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


def q3_sim(a: np.ndarray) -> np.ndarray:
    g = a.reshape(a.shape[0], -1, 32)
    s = (np.abs(g).max(axis=2, keepdims=True) / 3.0).astype(np.float16).astype(np.float32)
    s[s == 0] = 1.0
    return (np.clip(np.round(g / s), -4, 3) * s).reshape(a.shape).astype(np.float32)


def t2_sim(a: np.ndarray) -> np.ndarray:
    g = a.reshape(a.shape[0], -1, 32).astype(np.float32)
    ag = np.abs(g)
    delta = 0.7 * ag.mean(axis=2, keepdims=True)
    keep = ag > delta
    cnt = np.maximum(keep.sum(axis=2, keepdims=True), 1)
    s = ((ag * keep).sum(axis=2, keepdims=True) / cnt).astype(np.float16).astype(np.float32)
    return (np.sign(g) * keep * s).reshape(a.shape).astype(np.float32)


STORES = {"q4sim": (q4_sim, 4.5), "q3sim": (q3_sim, 3.5), "t2sim": (t2_sim, 2.5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rec = np.fromfile(args.capture + ".pos.bin", dtype=REC)
    h = np.fromfile(args.capture + ".hid.f32", dtype=np.float32).reshape(-1, D)
    if len(h) != len(rec):
        raise SystemExit("capture: hidden rows and records differ")
    top1 = rec["top1"].astype(np.int64)
    w = np.asarray(load_head(Path(args.model), Path(args.work) / "head_q6k_dequant_f32.npy"))
    rep = {"cell": "C50-A5 part 1b", "capture": args.capture, "positions": int(len(rec)),
           "exact_head_mb_per_token": V * D * 6.5625 / 8 / 1e6}

    sub = np.arange(0, len(h), 10)
    rk = ranks_of_top1(lambda x: q8k(x) @ w.T, h[sub], top1[sub])
    rep["oracle"] = {"positions": int(len(sub)), "q8k_rank_code_recall_at_1": float((rk == 0).mean())}
    if rep["oracle"]["q8k_rank_code_recall_at_1"] < 0.999:
        raise SystemExit(f"oracle control failed: {rep['oracle']}: VOID")
    print("oracle", rep["oracle"], f"{time.time() - t0:.0f}s", flush=True)

    gram = w.astype(np.float64).T @ w.astype(np.float64)
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    rep["preview_1b"] = {}
    for r in RANKS:
        vr = evecs[:, order[:r]].astype(np.float32)
        base = (w @ vr).astype(np.float32)
        for name, (fn, bits) in STORES.items():
            p = fn(base)
            rk = ranks_of_top1(lambda x: (x @ vr) @ p.T, h, top1)
            pre = V * r * bits / 8 / 1e6 + D * r * 4 / 1e6
            rep["preview_1b"][f"rot_r{r}_{name}"] = {
                str(k): {"recall": float((rk < k).mean()), "mb_per_token": pre + k * ROW_BYTES_Q6K / 1e6} for k in KS_ROT}
            print("rot", r, name, {k: round(v["recall"], 5) for k, v in rep["preview_1b"][f"rot_r{r}_{name}"].items()},
                  f"{time.time() - t0:.0f}s", flush=True)
        del base
    for name in ("q3sim", "t2sim"):
        fn, bits = STORES[name]
        p = fn(w)
        rk = ranks_of_top1(lambda x: x @ p.T, h, top1)
        pre = V * D * bits / 8 / 1e6
        rep["preview_1b"][f"full_{name}"] = {
            str(k): {"recall": float((rk < k).mean()), "mb_per_token": pre + k * ROW_BYTES_Q6K / 1e6} for k in KS_FULL}
        print("full", name, {k: round(v["recall"], 5) for k, v in rep["preview_1b"][f"full_{name}"].items()},
              f"{time.time() - t0:.0f}s", flush=True)
        del p
    rep["wall_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
