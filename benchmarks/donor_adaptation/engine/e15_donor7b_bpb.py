#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E15 -- does the artifact the goal is quoted on actually predict?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E15_DOES_THE_7B_PREDICT.md,
pushed before any arm ran.

qwen25-coder7b_p.bin carries every tok/s this programme has quoted on a real donor.  Its sidecar
says rule = R0, calib_seqs = None -- the rule E12 measured ABOVE the chance line at 0.5B, 1.5B and
3B -- and its BPB has never been measured.  E7's 160/160 is the fp32 arm (G-G); the packed arm is
E7's PLANTED CONTROL (G-C), required to fail, and it scored 0/160 with the first divergence at a
12.436-nat top-2 gap.  Nothing in that contradicts a working model and nothing in it supports one:
nobody looked.

Protocol notes that are load-bearing:
  --seqlen 512 is passed EXPLICITLY.  donor_engine.c:1249 defaults SL = n, the whole ids file as
  one sequence.  E1 passes it; E14's harness does NOT, which is why E14's A0 reads 4.629292 where
  E1 reads 4.531234 on the identical file.  E15 uses E1's protocol.

  The byte convention is E1's: nb = len(tok.decode(ids[1:]).encode("utf-8")) per sequence, summed
  = 51870, over 24 * 511 = 12264 predictions.  Charged bytes, not moved bytes.

Bands, verbatim from brief section 5:
  G-Q0  BPB(B0) < 1.000                      PLANTED CONTROL.  Must fire or E15 is VOID.
  G-Q1  BPB(B1) vs 4.070106 +/- 0.000896     PREDICTS / AT-CHANCE / DOES-NOT-PREDICT
  G-Q2  BPB(B1) - BPB(B0)                    DESCRIPTIVE.  Not damage if B1 is at or above chance.

Env: D_THREADS (default 6), E15_SEQS (default 24; use 2 for the smoke), E15_ARMS (default "B0,B1")
"""
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DENSITY = os.path.join(os.path.dirname(HERE), "density")
sys.path.insert(0, DENSITY)

ENGINE = r"D:\_ktmp\e13\donor_engine.exe"          # the E13 build
E7 = r"D:\_ktmp\e7"
OUT = os.path.join(HERE, "results", "e15_donor7b.json")

MODEL = "Qwen/Qwen2.5-Coder-7B"
VOCAB_PADDED = 152064          # config.vocab_size -- the conservative choice for an ABOVE claim
SEQ_LEN = 512
SLICE_SHA = "a1a48dc9fc5a6dc1"

ARMS = [
    ("B0", os.path.join(E7, "qwen25-coder7b_f32.bin"), "fp32 -- the planted control, must predict"),
    ("B1", os.path.join(E7, "qwen25-coder7b_p.bin"),   "packed R0 head-ternary -- the shipped artifact"),
]

GQ0_MAX = 1.000
THREADS = os.environ.get("D_THREADS", "6")
N_SEQ = int(os.environ.get("E15_SEQS", "24"))
WANT = os.environ.get("E15_ARMS", "B0,B1").split(",")


def build_slice():
    """The pinned density heldout slice, tokenized and byte-counted under the CODER tokenizer.

    common.get_slice caches on (part, n_seq, seq_len, seed) and NOT on the tokenizer, so asking it
    for a Coder slice would hand back the Qwen2.5 one.  The crossing was verified safe before the
    brief was written (24/24 sequences decode identically, re-encode to byte-identical ids, same
    per-sequence byte counts), so the cached ids ARE the right ids -- but the byte counts are
    re-derived here under the Coder tokenizer rather than inherited, so the denominator in this
    probe is one this probe computed.
    """
    import torch
    from transformers import AutoTokenizer
    import common as C

    d = torch.load(os.path.join(C.RESULTS, "slice_heldout_24x512_s1234.pt"))
    ids, meta = d["ids"], d["meta"]
    assert meta["ids_sha256"].startswith(SLICE_SHA), meta["ids_sha256"]
    assert ids.shape[1] == SEQ_LEN, ids.shape

    tk = AutoTokenizer.from_pretrained(MODEL)
    byts = []
    for i in range(ids.shape[0]):
        row = ids[i, 1:].tolist()
        dec = tk.decode(row)
        assert tk(dec, add_special_tokens=False)["input_ids"] == row, \
            "seq %d does not round-trip under the Coder tokenizer" % i
        byts.append(len(dec.encode("utf-8")))
    assert sum(byts) == meta["total_scored_bytes"], (sum(byts), meta["total_scored_bytes"])
    return ids, byts, meta, len(tk)


def run_arm(weights, idsp, n_seq):
    cmd = [ENGINE, "--weights", weights, "--threads", THREADS,
           "--seqlen", str(SEQ_LEN), "--bpb", idsp]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("engine failed (%d):\n%s\n%s" % (r.returncode, r.stdout[-2000:],
                                                          r.stderr[-2000:]))
    m = re.search(r"NATS_TOTAL ([0-9.eE+-]+)\s+N_PREDICTED (\d+)", r.stdout)
    if not m:
        raise SystemExit("no NATS_TOTAL in engine output:\n" + r.stdout[-2000:])
    total, npred = float(m.group(1)), int(m.group(2))
    assert npred == n_seq * (SEQ_LEN - 1), (npred, n_seq)
    return total, npred, time.time() - t0


def main():
    ids, byts, meta, tk_len = build_slice()
    ids = ids[:N_SEQ]
    byts = byts[:N_SEQ]
    scored = sum(byts)
    npred_expect = N_SEQ * (SEQ_LEN - 1)
    bpt = scored / float(npred_expect)
    chance = math.log(VOCAB_PADDED, 2) / bpt
    chance_tk = math.log(tk_len, 2) / bpt
    band = chance - chance_tk

    smoke = N_SEQ != 24
    print("E15  %s  seqs=%d  scored_bytes=%d  bytes/token=%.6f  slice %s%s"
          % (MODEL, N_SEQ, scored, bpt, meta["ids_sha256"][:16], "  [SMOKE]" if smoke else ""),
          flush=True)
    print("chance = %.6f BPB (V=%d)   %.6f (V=%d, tokenizer)   band = %.6f"
          % (chance, VOCAB_PADDED, chance_tk, tk_len, band), flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    idsp = os.path.join(E7, "e15_ids_%dx%d.bin" % (N_SEQ, SEQ_LEN))
    ids.numpy().astype("<i4").tofile(idsp)

    out = {"brief": "briefs/BRIEF_E15_DOES_THE_7B_PREDICT.md", "model": MODEL, "smoke": smoke,
           "slice": {"ids_sha256": meta["ids_sha256"], "n_seq": N_SEQ, "seq_len": SEQ_LEN,
                     "scored_bytes": scored, "n_predicted": npred_expect,
                     "bytes_per_token": bpt},
           "chance_bpb": chance, "chance_bpb_tokenizer_vocab": chance_tk, "band": band,
           "vocab_padded": VOCAB_PADDED, "tokenizer_len": tk_len,
           "threads": int(THREADS), "arms": {}}
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
            if prev.get("slice", {}).get("n_seq") == N_SEQ:
                out["arms"] = prev.get("arms", {})
        except Exception:
            pass

    for tag, w, note in ARMS:
        if tag not in WANT:
            continue
        if tag in out["arms"]:
            print("  %-3s CACHED  BPB = %.9f" % (tag, out["arms"][tag]["bpb"]), flush=True)
            continue
        if not os.path.exists(w):
            raise SystemExit("missing %s" % w)
        print("  %-3s running %s (%.2f GB)  %s"
              % (tag, os.path.basename(w), os.path.getsize(w) / 1e9, note), flush=True)
        total, npred, secs = run_arm(w, idsp, N_SEQ)
        bpb = total / math.log(2) / scored
        out["arms"][tag] = {"weights": w, "nats_total": total, "n_predicted": npred,
                            "nats_per_token": total / npred, "bpb": bpb,
                            "vs_chance": bpb - chance, "seconds": secs, "note": note}
        print("      BPB = %.9f   (%.6f nats/tok, %.0fs)   vs chance %+.6f"
              % (bpb, total / npred, secs, bpb - chance), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    print("\n---- gates ----", flush=True)
    b0 = out["arms"].get("B0")
    b1 = out["arms"].get("B1")
    if b0:
        out["G_Q0"] = b0["bpb"] < GQ0_MAX
        print("G-Q0  planted control: BPB(B0) = %.9f < %.3f -> %s"
              % (b0["bpb"], GQ0_MAX, "FIRES" if out["G_Q0"] else "*** DOES NOT FIRE -- E15 VOID ***"))
    if b1:
        d = b1["bpb"] - chance
        out["G_Q1"] = ("AT-CHANCE" if abs(d) <= band else
                       "PREDICTS" if d < 0 else "DOES-NOT-PREDICT")
        print("G-Q1  BPB(B1) = %.9f   chance %.6f   delta %+.6f (band %.6f) -> %s"
              % (b1["bpb"], chance, d, band, out["G_Q1"]))
    if b0 and b1:
        out["G_Q2_distance"] = b1["bpb"] - b0["bpb"]
        above = b1["bpb"] >= chance - band
        print("G-Q2  BPB(B1) - BPB(B0) = %+.9f   %s"
              % (out["G_Q2_distance"],
                 "DESCRIPTIVE ONLY -- B1 is at or above chance, so this is NOT damage (E12 s2)"
                 if above else "descriptive"))
    if b0 and b1 and not out.get("G_Q0"):
        print("\nE15 IS VOID: the planted control did not fire, so G-Q1 does not count.")
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
