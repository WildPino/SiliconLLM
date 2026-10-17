"""C50-A5 part 2: freeze the selected head approximation, then confirm it once on DEV (numbers only).

    --freeze    builds the part 1b selection (rotated r512 preview, 3.5-bit simulated storage, K = 32,768)
                from the gguf-py head, writes V_r and the preview matrix as .npy with their sha256, and
                reproduces the FIT recall from the frozen files (must equal part 1b's 0.99921 within 1e-5)
    --confirm   loads the frozen files (sha256 asserted), runs the Q8_K oracle control on the DEV capture
                (every 10th position; rank-code recall at 1 >= 0.999, else VOID), then computes DEV top-1
                recall at K = 32,768 against the engine top-1.  The gate is applied by
                c50_a5_select.py --grid dev.

Usage:
    python -I benchmarks/chatbot50/c50_a5_dev.py --freeze --model <gguf> --capture E:/.../a5/FIT12 \
        --work E:/.../a5 --out benchmarks/chatbot50/results/c50_a5_freeze.json
    python -I benchmarks/chatbot50/c50_a5_dev.py --confirm --freeze-report benchmarks/chatbot50/results/c50_a5_freeze.json \
        --capture E:/.../a5/DEV12 --work E:/.../a5 --out benchmarks/chatbot50/results/c50_a5_dev.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from c50_a5_offline import D, REC, load_head, q8k, ranks_of_top1
from c50_a5_offline_1b import q3_sim

R, K = 512, 32768
FIT_RECALL_1B = 0.9992096250109774


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_capture(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    rec = np.fromfile(prefix + ".pos.bin", dtype=REC)
    h = np.fromfile(prefix + ".hid.f32", dtype=np.float32).reshape(-1, D)
    if len(h) != len(rec):
        raise SystemExit("capture: hidden rows and records differ")
    return rec, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--freeze-report")
    ap.add_argument("--capture", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.freeze == args.confirm:
        raise SystemExit("exactly one of --freeze / --confirm")
    t0 = time.time()
    work = Path(args.work)
    rec, h = load_capture(args.capture)
    top1 = rec["top1"].astype(np.int64)
    if args.freeze:
        w = np.asarray(load_head(Path(args.model), work / "head_q6k_dequant_f32.npy"))
        gram = w.astype(np.float64).T @ w.astype(np.float64)
        evals, evecs = np.linalg.eigh(gram)
        vr = np.ascontiguousarray(evecs[:, np.argsort(evals)[::-1][:R]].astype(np.float32))
        p = q3_sim((w @ vr).astype(np.float32))
        fv, fp = work / "a5_sel_r512_q3sim_vr.npy", work / "a5_sel_r512_q3sim_preview.npy"
        np.save(fv, vr)
        np.save(fp, p)
        vr, p = np.load(fv), np.load(fp)
        rk = ranks_of_top1(lambda x: (x @ vr) @ p.T, h, top1)
        rep = {"cell": "C50-A5 part 2 freeze", "config": "rot_r512_q3sim", "K": K, "capture": args.capture,
               "files": {str(fv): sha(fv), str(fp): sha(fp)},
               "fit_recall_from_frozen": float((rk < K).mean()), "fit_recall_part1b": FIT_RECALL_1B}
        if abs(rep["fit_recall_from_frozen"] - FIT_RECALL_1B) > 1e-5:
            raise SystemExit(f"frozen files do not reproduce part 1b: {rep}: VOID")
    else:
        fr = json.loads(Path(args.freeze_report).read_text(encoding="utf-8"))
        for f, s in fr["files"].items():
            if sha(Path(f)) != s:
                raise SystemExit(f"frozen file changed: {f}")
        fv, fp = list(fr["files"])
        vr, p = np.load(fv), np.load(fp)
        w = np.asarray(np.load(work / "head_q6k_dequant_f32.npy", mmap_mode="r"))
        sub = np.arange(0, len(h), 10)
        rk0 = ranks_of_top1(lambda x: q8k(x) @ w.T, h[sub], top1[sub])
        rep = {"cell": "C50-A5 part 2 DEV confirmation", "config": fr["config"], "K": fr["K"],
               "capture": args.capture, "positions": int(len(rec)), "docs": int(len(np.unique(rec["doc"]))),
               "oracle": {"positions": int(len(sub)), "q8k_rank_code_recall_at_1": float((rk0 == 0).mean())}}
        if rep["oracle"]["q8k_rank_code_recall_at_1"] < 0.999:
            raise SystemExit(f"oracle control failed: {rep['oracle']}: VOID")
        rk = ranks_of_top1(lambda x: (x @ vr) @ p.T, h, top1)
        hit = rk < fr["K"]
        per_doc = [float(hit[rec["doc"] == d].mean()) for d in np.unique(rec["doc"])]
        rep.update({"dev_recall": float(hit.mean()), "misses": int((~hit).sum()),
                    "per_doc_recall_min_median": [min(per_doc), float(np.median(per_doc))],
                    "recall_at_K": {str(k): float((rk < k).mean()) for k in (1024, 4096, 16384, 32768, 65536)}})
    rep["wall_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
