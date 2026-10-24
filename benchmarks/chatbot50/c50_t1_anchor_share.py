"""C50-T1 part 2b, MVE-A: how many training positions the anchor would take, measured before the protocol fixes it.

On S4's validation shards (470-471 of block 1, never trained on), T0 (P0, llama.cpp, S2's score tool and settings)
scores the same documents the teacher shards hold. Per position: p_T0(y) from the tool, p_F0(y) from the shard's
`lpt`. A position is anchored iff p_T0(y) >= TAU and p_T0(y) >= p_F0(y): T0 is sure of the right token and the
teacher is no better. Reported: the anchored share, and how the T0 -> F0 bit gap on the target splits between
anchored and free positions, for a few TAU.

    python -I benchmarks/chatbot50/c50_t1_anchor_share.py --work DIR --json OUT
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_s2_run import HERE, MAX_PROJECTED_S, MODELS, sha256  # noqa: E402
from c50_s4_student import REC  # noqa: E402

C = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
SHARDS = C / "s3/teacher_v1"
CORPUS = C / "s3/corpus_v1/teacher.tsv"
VAL = (470, 471)
TAUS = (0.3, 0.5, 0.7, 0.9)
SREC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                 ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    rec = np.concatenate([np.fromfile(SHARDS / f"shard_{s:06d}.bin", dtype=REC) for s in VAL])
    docs = sorted(set(np.unique(rec["doc"]).tolist()))
    ids = {}
    with CORPUS.open(encoding="ascii") as f:
        for line in f:
            d, s = line.split("\t", 1)
            if int(d) in docs:
                ids[int(d)] = s.strip().rstrip(",")
    assert set(ids) == set(docs)
    lens = [len(v.split(",")) for v in ids.values()]
    tsv = a.work / "val.tsv"
    tsv.write_text("".join(f"{d}\t{ids[d]}\n" for d in docs), encoding="ascii")
    model, pin = MODELS["t0"]
    exe = HERE / "bin" / "c50_llama_score_avx2.exe"
    cmd = [str(exe), "--model", str(model), "--ids", str(tsv), "--out", str(a.work / "val"), "--threads", "6",
           "--ubatch", "512", "--max-projected-s", str(MAX_PROJECTED_S), "--ctx", str(max(8192, max(lens)))]
    print("CONFIG", json.dumps({"docs": len(docs), "positions": int(len(rec)), "max_len": max(lens), "cmd": cmd}),
          flush=True)
    out = {"cell": "C50-T1 part 2b MVE-A, anchor share on S4's validation shards", "shards": VAL, "docs": len(docs),
           "model_sha256_pin": pin, "exe_sha256": sha256(exe), "cmd": cmd}
    if not (a.work / "val.pos.bin").exists():
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        out.update({"returncode": p.returncode, "wall_s": round(time.time() - t0, 1),
                    "config": [ln for ln in p.stdout.splitlines() if ln.startswith("CONFIG ")],
                    "stderr_tail": p.stderr[-400:]})
        if p.returncode != 0:
            Path(a.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
            raise SystemExit(f"score failed: {p.returncode}")
    s = np.fromfile(a.work / "val.pos.bin", dtype=SREC)
    key_t = {(int(d), int(q)): i for i, (d, q) in enumerate(zip(s["doc"], s["pos"]))}
    # the tool's pos is the position of the predicted token, as in the shards (1..L-1); targets must agree
    j = np.array([key_t[(int(d), int(q))] for d, q in zip(rec["doc"], rec["pos"])])
    assert np.array_equal(s["target"][j], rec["target"]), "positions do not line up"
    lt, lf = s["logp_target"][j].astype(np.float64), rec["lpt"].astype(np.float64)
    gap = (lf - lt) / np.log(2)                    # bits T0 would save on y by matching F0 there
    out["positions"] = int(len(rec))
    out["bits_gap_total"] = float(gap.sum())
    out["mean_p_t0"], out["mean_p_f0"] = float(np.exp(lt).mean()), float(np.exp(lf).mean())
    out["by_tau"] = {}
    for tau in TAUS:
        m = (np.exp(lt) >= tau) & (lt >= lf)
        out["by_tau"][str(tau)] = {"anchored_share": float(m.mean()),
                                   "gap_bits_anchored": float(gap[m].sum()), "gap_bits_free": float(gap[~m].sum()),
                                   "t0_better_share_overall": float((lt >= lf).mean())}
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("positions", "bits_gap_total", "mean_p_t0", "mean_p_f0", "by_tau")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
