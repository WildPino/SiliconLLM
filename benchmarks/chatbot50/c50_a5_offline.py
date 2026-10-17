"""C50-A5 offline: approximate-head candidates on captured FIT hidden states (numbers only).

Ground truth is the engine's own top-1 at each captured position (record field top1), i.e. the exact
Q6_K head as deployed.  A configuration's top-1 recall is the fraction of positions whose engine top-1
is inside its candidate set: with an exact rerank of the candidates by the same head kernel, that is
exactly the probability that the approximate head returns the engine's top-1.

    oracle      dequantized head (gguf-py, the pin's own reference) x captured state in float32; its
                argmax against the engine top-1 on every 10th position (a check of the capture, not a
                candidate)
    preview     W ~ (W V_r) V_r^T, V_r = top-r eigenvectors of W^T W; candidates = top-K preview
                logits; r in {128, 256, 512}, K in {8 .. 1024}; the preview matrix in float32 and with
                a simulated 4-bit storage (groups of 32, symmetric, float16 scale: 4.5 bits/weight)
    ivf         k-means of the head rows (L2, seed 20261016, 12 Lloyd iterations, init from random
                rows), C in {256, 1024, 4096}; probe the P clusters with the largest centroid . h;
                candidates = every row of the probed clusters; P in {1 .. 128}

Bytes per token (what a decode step streams for the head):
    exact       157184 rows x 2048 x 6.5625 bits = 264.1 MB
    preview     157184 x r x 4.5 bits (preview) + 2048 x r x 4 B (V_r, float32) + K rerank rows x 1680 B
    ivf         C x 2048 x 8.5 bits (centroids, Q8_0) + mean candidate rows x 1680 B

Usage:
    python -I benchmarks/chatbot50/c50_a5_offline.py --model <q4_k_m.gguf> --capture E:/.../a5/FIT12 \
        --work E:/.../a5 --out benchmarks/chatbot50/results/c50_a5_offline_fit.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LLAMA_PIN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4")
REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])
SEED = 20261016
D, V = 2048, 157184
ROW_BYTES_Q6K = 2048 // 256 * 210  # 1680
RANKS, KS = (128, 256, 512), (8, 16, 32, 64, 128, 256, 512, 1024)
CS, PS = (256, 1024, 4096), (1, 2, 4, 8, 16, 32, 64, 128)


def load_head(model: Path, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    sys.path.insert(0, str(LLAMA_PIN / "gguf-py"))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    t = next(t for t in GGUFReader(str(model)).tensors if t.name == "output.weight")
    if t.tensor_type.name != "Q6_K":
        raise SystemExit(f"head type {t.tensor_type.name}, expected Q6_K")
    w = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(V, D)
    np.save(cache, w)
    return w


def q4_sim(a: np.ndarray) -> np.ndarray:
    g = a.reshape(a.shape[0], -1, 32)
    s = (np.abs(g).max(axis=2, keepdims=True) / 7.0).astype(np.float16).astype(np.float32)
    s[s == 0] = 1.0
    return (np.clip(np.round(g / s), -8, 7) * s).reshape(a.shape).astype(np.float32)


def q8k(x: np.ndarray) -> np.ndarray:
    """ggml quantize_row_q8_K_ref then dequantize: per 256 values, iscale = -127 / (the signed value of
    largest magnitude), q = min(127, round-half-even(iscale * x)), d = 1 / iscale (float32 throughout).
    The engine multiplies the head with this activation (A5 addendum 1)."""
    g = x.reshape(len(x), -1, 256).astype(np.float32)
    mx = np.take_along_axis(g, np.abs(g).argmax(axis=2)[..., None], axis=2)
    mx[mx == 0] = 1.0
    iscale = (np.float32(-127.0) / mx).astype(np.float32)
    q = np.minimum(127, np.rint(iscale * g)).astype(np.float32)
    return (q * (np.float32(1.0) / iscale)).reshape(x.shape).astype(np.float32)


def ranks_of_top1(scores_fn, h: np.ndarray, top1: np.ndarray, chunk: int = 256) -> np.ndarray:
    out = np.empty(len(h), dtype=np.int64)
    for s in range(0, len(h), chunk):
        sc = scores_fn(h[s:s + chunk])
        t = sc[np.arange(len(sc)), top1[s:s + chunk]]
        out[s:s + chunk] = (sc > t[:, None]).sum(axis=1)
    return out


def kmeans(w: np.ndarray, c: int, iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    cent = np.array(w[rng.choice(len(w), size=c, replace=False)], dtype=np.float32)
    wn = (w.astype(np.float32) ** 2).sum(axis=1)
    for _ in range(iters):
        assign = np.empty(len(w), dtype=np.int64)
        cn = (cent ** 2).sum(axis=1)
        for s in range(0, len(w), 16384):
            d = wn[s:s + 16384, None] - 2.0 * (w[s:s + 16384] @ cent.T) + cn[None, :]
            assign[s:s + 16384] = d.argmin(axis=1)
        order = np.argsort(assign, kind="stable")
        cnt = np.bincount(assign, minlength=c)
        nz = np.flatnonzero(cnt)
        starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])[nz]
        sums = np.add.reduceat(np.asarray(w)[order], starts, axis=0)
        cent[nz] = sums / cnt[nz, None].astype(np.float32)
    return cent, assign


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--capture", required=True, help="prefix of the engine capture (.pos.bin, .hid.f32)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    work = Path(args.work)
    rec = np.fromfile(args.capture + ".pos.bin", dtype=REC)
    h = np.fromfile(args.capture + ".hid.f32", dtype=np.float32).reshape(-1, D)
    if len(h) != len(rec):
        raise SystemExit("capture: hidden rows and records differ")
    top1 = rec["top1"].astype(np.int64)
    w = np.asarray(load_head(Path(args.model), work / "head_q6k_dequant_f32.npy"))
    report = {"cell": "C50-A5 offline", "capture": args.capture, "positions": int(len(rec)),
              "exact_head_mb_per_token": V * D * 6.5625 / 8 / 1e6}

    sub = np.arange(0, len(h), 10)
    report["oracle"] = {"positions": int(len(sub))}
    for act, f in (("q8k", q8k), ("f32", lambda x: x)):  # q8k is the control; f32 is reported beside it
        agree, lse_err = 0, []
        for s in range(0, len(sub), 256):
            idx = sub[s:s + 256]
            lg = f(h[idx]) @ w.T
            agree += int((lg.argmax(axis=1) == top1[idx]).sum())
            m = lg.max(axis=1, keepdims=True)
            lse = (m[:, 0] + np.log(np.exp(lg - m).sum(axis=1))).astype(np.float64)
            lse_err.extend(np.abs(lse - rec["lse"][idx]).tolist())
        report["oracle"][act] = {"argmax_agreement": agree / len(sub), "lse_abs_err_median": float(np.median(lse_err)),
                                 "lse_abs_err_max": float(np.max(lse_err))}
    o = report["oracle"]["q8k"]
    if o["argmax_agreement"] < 0.999 or o["lse_abs_err_median"] > 1e-4:
        raise SystemExit(f"oracle control failed: {report['oracle']}: VOID")
    report["oracle"]["argmax_agreement"] = o["argmax_agreement"]
    # control of the rank code: through ranks_of_top1, the full head must reproduce the oracle agreement
    rk_full = ranks_of_top1(lambda x: q8k(x) @ w.T, h[sub], top1[sub])
    report["oracle"]["rank_code_recall_at_1_full_head"] = float((rk_full == 0).mean())
    # rank 0 = no logit strictly above the engine top-1; argmax = first maximum: they differ only on exact ties
    if abs(report["oracle"]["rank_code_recall_at_1_full_head"] - report["oracle"]["argmax_agreement"]) * len(sub) > 2:
        raise SystemExit("rank code disagrees with the oracle argmax beyond exact ties: VOID")
    print("oracle", report["oracle"], f"{time.time() - t0:.0f}s", flush=True)

    gram = (w.astype(np.float64).T @ w.astype(np.float64))
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    report["preview"] = {}
    for r in RANKS:
        vr = evecs[:, order[:r]].astype(np.float32)
        for store in ("f32", "q4sim"):
            p = (w @ vr).astype(np.float32)
            if store == "q4sim":
                p = q4_sim(p)
            rk = ranks_of_top1(lambda x: (x @ vr) @ p.T, h, top1)
            pre_mb = V * r * (4.5 if store == "q4sim" else 32) / 8 / 1e6 + D * r * 4 / 1e6
            report["preview"][f"r{r}_{store}"] = {
                str(k): {"recall": float((rk < k).mean()), "mb_per_token": pre_mb + k * ROW_BYTES_Q6K / 1e6} for k in KS}
            print("preview", r, store, {k: round(v["recall"], 5) for k, v in report["preview"][f"r{r}_{store}"].items()},
                  f"{time.time() - t0:.0f}s", flush=True)

    report["ivf"] = {}
    for c in CS:
        cent, assign = kmeans(w, c)
        sizes = np.bincount(assign, minlength=c)
        cl_top1 = assign[top1]
        res = {}
        crank = np.empty(len(h), dtype=np.int64)
        need = {p: np.zeros(len(h), dtype=np.int64) for p in PS}
        for s in range(0, len(h), 2048):
            sc = h[s:s + 2048] @ cent.T
            t = sc[np.arange(len(sc)), cl_top1[s:s + 2048]]
            crank[s:s + 2048] = (sc > t[:, None]).sum(axis=1)
            srt = np.argsort(-sc, axis=1)
            csz = np.cumsum(sizes[srt], axis=1)
            for p in PS:
                need[p][s:s + 2048] = csz[:, p - 1]
        for p in PS:
            cand = float(need[p].mean())
            res[str(p)] = {"recall": float((crank < p).mean()), "mean_candidates": cand,
                           "mb_per_token": c * D * 8.5 / 8 / 1e6 + cand * ROW_BYTES_Q6K / 1e6}
        report["ivf"][f"C{c}"] = {"cluster_size_min_median_max": [int(sizes.min()), int(np.median(sizes)), int(sizes.max())],
                                  "probes": res}
        print("ivf", c, {p: (round(v["recall"], 5), round(v["mb_per_token"], 2)) for p, v in res.items()},
              f"{time.time() - t0:.0f}s", flush=True)
    report["wall_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
