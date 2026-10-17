"""C50-A5 part 4 (exploratory, FIT only): head restricted by a fixed script rule (numbers only).

The kept set comes from tokenizer.json alone (rule in C50_A5_PART4_SCRIPT_HEAD_PROTOCOL_20261017.md):
added tokens, non-UTF-8 byte fragments, and valid UTF-8 tokens whose code points all lie in the kept
ranges.  The restricted scan is exact; a position misses iff its engine top-1 is dropped.  The part 1b
preview (frozen files of part 2) restricted to the kept rows is the combined arm.

Usage:
    python -I benchmarks/chatbot50/c50_a5_script_head.py --tokenizer <dir>/tokenizer.json \
        --capture E:/.../a5/FIT12 --work E:/.../a5 --freeze-report benchmarks/chatbot50/results/c50_a5_freeze.json \
        --out benchmarks/chatbot50/results/c50_a5_script_head_fit.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from c50_a5_offline import D, REC, ROW_BYTES_Q6K, V, q8k, ranks_of_top1

KEEP_RANGES = ((0x0000, 0x052F), (0x1E00, 0x1FFF), (0x2000, 0x2BFF), (0xFB00, 0xFB06), (0xFE00, 0xFE0F),
               (0x1F000, 0x1FAFF))
KS = (1024, 2048, 4096, 8192, 16384, 32768)
CATS = ("code", "technical_general", "prose")  # FIT role order, 12 documents each in FIT12


def byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def classify(tok: dict) -> tuple[np.ndarray, dict, list]:
    dec = byte_decoder()
    keep = np.zeros(V, dtype=bool)
    cls = {"added": 0, "fragment": 0, "kept_script": 0, "dropped_script": 0, "padding": 0}
    text = [""] * V
    for s, i in tok["model"]["vocab"].items():
        b = bytes(dec[c] for c in s)
        try:
            u = b.decode("utf-8")
        except UnicodeDecodeError:
            keep[i] = True; cls["fragment"] += 1; text[i] = repr(b); continue
        text[i] = u
        if all(any(lo <= ord(ch) <= hi for lo, hi in KEEP_RANGES) for ch in u):
            keep[i] = True; cls["kept_script"] += 1
        else:
            cls["dropped_script"] += 1
    for a in tok.get("added_tokens", []):
        keep[a["id"]] = True; cls["added"] += 1; text[a["id"]] = a["content"]
    cls["padding"] = V - (len(tok["model"]["vocab"]) + len(tok.get("added_tokens", [])))
    return keep, cls, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--freeze-report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    keep, cls, text = classify(json.loads(Path(args.tokenizer).read_text(encoding="utf-8")))
    rec = np.fromfile(args.capture + ".pos.bin", dtype=REC)
    h = np.fromfile(args.capture + ".hid.f32", dtype=np.float32).reshape(-1, D)
    top1 = rec["top1"].astype(np.int64)
    w = np.asarray(np.load(Path(args.work) / "head_q6k_dequant_f32.npy", mmap_mode="r"))
    sub = np.arange(0, len(h), 10)
    rk0 = ranks_of_top1(lambda x: q8k(x) @ w.T, h[sub], top1[sub])
    rep = {"cell": "C50-A5 part 4 script-restricted head (FIT, exploratory)", "capture": args.capture,
           "positions": int(len(rec)), "oracle": {"positions": int(len(sub)), "q8k_rank_code_recall_at_1": float((rk0 == 0).mean())}}
    if rep["oracle"]["q8k_rank_code_recall_at_1"] < 0.999:
        raise SystemExit(f"oracle control failed: {rep['oracle']}: VOID")
    n_keep = int(keep.sum())
    hit = keep[top1]
    cat = rec["doc"] // 12
    rep["kept_set"] = {"n_keep": n_keep, "classes": cls, "exact_restricted_mb_per_token": n_keep * ROW_BYTES_Q6K / 1e6}
    rep["exact_restricted"] = {"recall": float(hit.mean()), "misses": int((~hit).sum()),
                               "per_category": {CATS[c]: float(hit[cat == c].mean()) for c in range(3)},
                               "missed_tokens": sorted({(int(t), text[t]) for t in top1[~hit]})[:200]}
    print("kept", rep["kept_set"], "recall", rep["exact_restricted"]["recall"], f"{time.time() - t0:.0f}s", flush=True)

    fr = json.loads(Path(args.freeze_report).read_text(encoding="utf-8"))
    vr, p = (np.load(f) for f in fr["files"])
    idx = np.flatnonzero(keep)
    pk = np.ascontiguousarray(p[idx])
    pos_in = np.full(V, -1, dtype=np.int64)
    pos_in[idx] = np.arange(n_keep)
    t_in = pos_in[top1]
    rk = np.full(len(h), np.iinfo(np.int64).max, dtype=np.int64)
    ok = t_in >= 0
    rk[ok] = ranks_of_top1(lambda x: (x @ vr) @ pk.T, h[ok], t_in[ok])
    pre = n_keep * 512 * 3.5 / 8 / 1e6 + D * 512 * 4 / 1e6
    rep["combined_r512_q3sim"] = {str(k): {"recall": float((rk < k).mean()), "mb_per_token": pre + k * ROW_BYTES_Q6K / 1e6}
                                  for k in KS}
    print("combined", {k: (round(v["recall"], 5), round(v["mb_per_token"], 1)) for k, v in rep["combined_r512_q3sim"].items()},
          flush=True)
    rep["wall_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
