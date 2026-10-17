"""C50-A5 part 6: rotated preview plus a residual-norm bonus (protocol C50_A5_PART6_RESIDUAL_BONUS_PROTOCOL).

Candidate score s_v = preview_v + c * |res_v| * |h_perp|, preview = (h V_r) P^T with P = q3sim(W V_r) as in part 1b,
|res_v| = sqrt(|w_v|^2 - |w_v V_r|^2) stored as float16, |h_perp|^2 = |h|^2 - |h V_r|^2.  The K highest s_v are
reranked exactly; recall = fraction of positions whose engine top-1 has fewer than K rows with a strictly higher s_v.

--select   FIT12 grid r x c x K; control: c = 0 must reproduce part 1b's recall (1e-9) else VOID; selection = the
           fewest bytes with recall >= 0.9995 (ties to the smaller c); the selected arrays are saved and hashed.
--confirm  DEV-B capture against the frozen arrays (sha checked): the gate line first, then reported extras.

Usage:
    python -I benchmarks/chatbot50/c50_a5_residual.py --select --out benchmarks/chatbot50/results/c50_a5_residual_fit.json
    python -I benchmarks/chatbot50/c50_a5_residual.py --confirm --fit-record benchmarks/chatbot50/results/c50_a5_residual_fit.json \
        --capture E:/.../a5/DEVB --devb-ids benchmarks/chatbot50/corpus/c50_holdout_ext_v1/devb_ids.jsonl \
        --out benchmarks/chatbot50/results/c50_a5_residual_devb.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_a5_offline import D, REC, ROW_BYTES_Q6K, V, load_head  # noqa: E402
from c50_a5_offline_1b import q3_sim  # noqa: E402

A5 = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/a5")
MODEL = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/gguf/ling-mini-2.0-a810f64-q4_k_m.gguf")
RANKS, CS, KS = (512, 1024), (0.0, 0.025, 0.05, 0.1, 0.2, 0.4), (2048, 4096, 8192, 16384, 32768)
SELECT_BAR, GATE = 0.9995, 0.999
PART1B = Path(__file__).resolve().parent / "results" / "c50_a5_offline_1b_fit.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mb(r: int, c: float, k: int) -> float:
    return (V * r * 3.5 / 8 + D * r * 4 + (V * 2 if c > 0 else 0) + k * ROW_BYTES_Q6K) / 1e6


def build(w: np.ndarray, r: int):
    gram = w.astype(np.float64).T @ w.astype(np.float64)
    evals, evecs = np.linalg.eigh(gram)
    vr = evecs[:, np.argsort(evals)[::-1][:r]].astype(np.float32)
    base = (w @ vr).astype(np.float32)
    res2 = (np.asarray(w, dtype=np.float64) ** 2).sum(axis=1) - (base.astype(np.float64) ** 2).sum(axis=1)
    norms = np.sqrt(np.maximum(res2, 0.0)).astype(np.float16)
    return vr, q3_sim(base), norms


def ranks(h, top1, vr, p, norms, cs, chunk: int = 256) -> dict:
    n32 = norms.astype(np.float32)
    out = {c: np.empty(len(h), dtype=np.int64) for c in cs}
    for s in range(0, len(h), chunk):
        x = h[s:s + chunk]
        xr = x @ vr
        hp = np.sqrt(np.maximum((x.astype(np.float64) ** 2).sum(axis=1) - (xr.astype(np.float64) ** 2).sum(axis=1), 0.0)).astype(np.float32)
        sc0 = xr @ p.T
        idx = np.arange(len(x)), top1[s:s + chunk]
        for c in cs:
            sc = sc0 if c == 0 else sc0 + np.float32(c) * hp[:, None] * n32[None, :]
            out[c][s:s + chunk] = (sc > sc[idx][:, None]).sum(axis=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--fit-record")
    ap.add_argument("--capture")
    ap.add_argument("--devb-ids")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    if args.select:
        rec = np.fromfile(A5 / "FIT12.pos.bin", dtype=REC)
        h = np.fromfile(A5 / "FIT12.hid.f32", dtype=np.float32).reshape(-1, D)
        if len(h) != len(rec):
            raise SystemExit("capture: hidden rows and records differ")
        top1 = rec["top1"].astype(np.int64)
        w = np.asarray(load_head(MODEL, A5 / "head_q6k_dequant_f32.npy"))
        p1b = json.loads(PART1B.read_text(encoding="utf-8"))["preview_1b"]
        out = {"cell": "C50-A5 part 6 FIT", "positions": int(len(rec)), "grid": [], "control": {}}
        for r in RANKS:
            vr, p, norms = build(w, r)
            rk = ranks(h, top1, vr, p, norms, CS)
            for k in KS:
                ref = p1b[f"rot_r{r}_q3sim"][str(k)]["recall"]
                got = float((rk[0.0] < k).mean())
                out["control"][f"r{r}_K{k}"] = {"part1b": ref, "c0": got}
                if abs(got - ref) > 1e-9:
                    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
                    raise SystemExit(f"VOID: c = 0 does not reproduce part 1b at r{r} K{k}: {got} vs {ref}")
            for c in CS:
                for k in KS:
                    out["grid"].append({"r": r, "c": c, "K": k, "mb_per_token": mb(r, c, k),
                                        "fit12_recall": float((rk[c] < k).mean())})
                print(f"r{r} c={c:<5} " + " ".join(f"K{k}:{(rk[c] < k).mean():.5f}" for k in KS), f"{time.time() - t0:.0f}s", flush=True)
            np.save(A5 / f"part6_r{r}_vr.npy", vr); np.save(A5 / f"part6_r{r}_preview.npy", p); np.save(A5 / f"part6_r{r}_norms.npy", norms)
            del p
        q = sorted((g for g in out["grid"] if g["fit12_recall"] >= SELECT_BAR), key=lambda g: (g["mb_per_token"], g["c"]))
        out["selected"] = q[0] if q else None
        if q:
            r = q[0]["r"]
            out["frozen"] = {n: {"path": str(A5 / f"part6_r{r}_{n}.npy"), "sha256": sha(A5 / f"part6_r{r}_{n}.npy")}
                             for n in ("vr", "preview", "norms")}
        print("SELECTED", out["selected"] if q else "none (part 6 FAILS on FIT; DEV-B is not read)")
        out["wall_s"] = time.time() - t0
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return 0
    if args.confirm:
        fitrec = json.loads(Path(args.fit_record).read_text(encoding="utf-8"))
        sel = fitrec.get("selected")
        if not sel:
            raise SystemExit("no selection on FIT: DEV-B is not read")
        arr = {}
        for n, f in fitrec["frozen"].items():
            if sha(Path(f["path"])) != f["sha256"]:
                raise SystemExit(f"frozen {n} changed since the FIT record")
            arr[n] = np.load(f["path"])
        rec = np.fromfile(args.capture + ".pos.bin", dtype=REC)
        h = np.fromfile(args.capture + ".hid.f32", dtype=np.float32).reshape(-1, D)
        if len(h) != len(rec):
            raise SystemExit("capture: hidden rows and records differ")
        top1 = rec["top1"].astype(np.int64)
        rk = ranks(h, top1, arr["vr"], arr["preview"], arr["norms"], [sel["c"]])[sel["c"]]
        hit = rk < sel["K"]
        recall = float(hit.mean())
        out = {"cell": "C50-A5 part 6 DEV-B confirmation", "positions": int(len(rec)), "selected": sel,
               "devb_recall": recall, "gate": GATE, "verdict": "PASS" if recall >= GATE else "FAIL"}
        print(f"GATE r{sel['r']} c={sel['c']} K={sel['K']} DEV-B recall {recall:.5f} vs {GATE}: {out['verdict']}", flush=True)
        rk_all = ranks(h, top1, arr["vr"], arr["preview"], arr["norms"], CS)
        out["reported_grid_same_r"] = [{"c": c, "K": k, "devb_recall": float((rk_all[c] < k).mean())} for c in CS for k in KS]
        cats = [json.loads(x)["category"] for x in Path(args.devb_ids).read_text(encoding="utf-8").splitlines()]
        dcat = np.array([cats[d] for d in rec["doc"]])
        out["reported_by_category"] = {c: {"positions": int((dcat == c).sum()), "recall": float(hit[dcat == c].mean())}
                                       for c in sorted(set(cats))}
        p = np.exp(rec["logp_top1"][~hit].astype(np.float64))
        out["reported_misses"] = {"count": int((~hit).sum()), "documents": int(len(set(rec["doc"][~hit].tolist()))),
                                  "p_top1_median": float(np.median(p)) if len(p) else None,
                                  "share_p_ge_0.9": float((p >= 0.9).mean()) if len(p) else None}
        out["wall_s"] = time.time() - t0
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return 0
    raise SystemExit("--select or --confirm")


if __name__ == "__main__":
    raise SystemExit(main())
