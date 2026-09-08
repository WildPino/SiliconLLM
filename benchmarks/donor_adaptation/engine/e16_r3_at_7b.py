#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E16 -- is there a ternary 7B that predicts?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E16_R3_AT_7B.md, pushed before the
export ran (commit be65920).

E15 found the conversion is what is broken -- the packed Coder-7B reads 5.299200075 against a
chance line of 4.070106 while the same weights in fp32 read 0.674026555 -- but could not say WHICH
PART of the conversion, because qwen25-coder7b_p.bin differs from every other standing artifact on
TWO axes at once: rule R0 (vs R3) and fold layers (vs none).

E16 moves one at a time.  Two fixed reference points, three arms:

  ref  B1  qwen25-coder7b_p.bin      R0, fold layers   5.299200075   (E15)
  ref  --  qwen25-15b_tqh.bin        R3, fold none     3.475706691632780  (E1 s2.2, engine)

  C0   the 1.5B TQH file, re-run here             PLANTED CONTROL / replication.  Runs FIRST.
  B2   Coder-7B, R3, fold LAYERS                  the RULE at 7B      (B2 - B1, fold fixed)
  B3   Coder-7B, R3, fold none                    the SCALE vs C0     (rule and fold fixed)

and B2 - B3 is the fold under R3 at 7B.

Why R3 is worth the hours: it is the rule the pipeline actually SHIPS
(e1_bpb_through_engine.py:395) and it CROSSES the chance line between 0.5B (4.531234, +0.461 above)
and 1.5B (3.475707, -0.594 below).  R0 gets WORSE with scale (E12: +0.518/+1.436/+1.665).  Two
curves in opposite directions, and E15 sits on the wrong one.

Bands, verbatim from brief section 5:
  G-R0  C0 reproduces 3.475706691632780 to <= 0.01 AND is below chance, or E16 is VOID
  G-R1  BPB(B2) vs 4.070106 +/- 0.000896  -> RULE-FIXES-IT / AT-CHANCE / RULE-IS-NOT-THE-PROBLEM
  G-R2  BPB(B2) - 5.299200075             descriptive unless B2 is below the line (E12 s2)
  G-R3  BPB(B3) vs the same band; and BPB(B3) - 3.475706691632780
  G-R4  BPB(B2) - BPB(B3)                 the fold under R3 at 7B, descriptive

Section 3's HARD DESIGN GATE is enforced in check_sidecar() before any BPB is read: a new arm's
sidecar may differ from its reference on the axis under test and on bytes/sha256/zero-fraction,
and on NOTHING else.  If any other field moved, more than one thing moved and the contrast is not
the contrast that was registered.

No timing is taken here and none may be quoted from it: a ternary weight costs the same bandwidth
whatever rule produced it, so 6.79 tok/s exact stands whatever E16 returns.

Env: D_THREADS (default 6), E16_SEQS (default 24; 2 for a smoke), E16_ARMS (default "C0,B2,B3"),
     E16_EXPORT (default "1"; set 0 to refuse to export and only measure what already exists)
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

ENGINE = r"D:\_ktmp\e13\donor_engine.exe"          # the E13 build, as E15 used
E7 = r"D:\_ktmp\e7"
E1 = r"D:\_ktmp\e1"
OUT = os.path.join(HERE, "results", "e16_r3_at_7b.json")
EXPORTER = os.path.join(HERE, "qwen_export.py")

CODER = "Qwen/Qwen2.5-Coder-7B"
Q15 = "Qwen/Qwen2.5-1.5B"
REV15 = "8faed761d45a263340a0528343f099c05c9a4323"     # the revision E1 pinned

SEQ_LEN = 512
SLICE_SHA = "a1a48dc9fc5a6dc1"

B1_BPB = 5.299200075086085          # E15, qwen25-coder7b_p.bin
C0_REFERENCE = 3.475706691632780    # E1 s2.2, engine column, same slice and --seqlen 512
GR0_TOL = 0.01

# arm -> (model, vocab_padded, weights path, export flags or None, note)
ARMS = [
    ("C0", Q15, 151936, os.path.join(E1, "qwen25-15b_tqh.bin"), None,
     "1.5B R3 fold-none -- PLANTED CONTROL, must reproduce E1 and sit below chance"),
    ("B2", CODER, 152064, os.path.join(E7, "qwen25-coder7b_p_r3.bin"),
     ["--rule", "R3", "--calib-seqs", "32", "--fold", "layers"],
     "7B R3 fold-LAYERS -- the rule at 7B, everything else as B1"),
    ("B3", CODER, 152064, os.path.join(E7, "qwen25-coder7b_p_r3_nofold.bin"),
     ["--rule", "R3", "--calib-seqs", "32", "--fold", "none"],
     "7B R3 fold-none -- the scale, rule and fold fixed against C0"),
]

THREADS = os.environ.get("D_THREADS", "6")
N_SEQ = int(os.environ.get("E16_SEQS", "24"))
WANT = os.environ.get("E16_ARMS", "C0,B2,B3").split(",")
MAY_EXPORT = os.environ.get("E16_EXPORT", "1") != "0"

# Section 3's design gate.  B2 is compared against B1's sidecar, B3 against B2's.
# Fields allowed to move, per comparison; everything else must be equal.
FREE_ALWAYS = ("bytes", "sha256", "mean_ternary_zero_fraction")
GATE = {
    "B2": (os.path.join(E7, "qwen25-coder7b_p.bin.json"), ("rule", "calib_seqs",
                                                           "calib_matches_t2_operating_point")),
    "B3": (os.path.join(E7, "qwen25-coder7b_p_r3.bin.json"), ("fold", "n_gains_folded")),
}

# B1's sidecar predates two exporter fields.  Their ABSENCE is a documentation gap, not a
# difference in the artifact, and each is admitted only against ONE stated value -- any other
# value still fails the gate.  load_dtype: the bf16 loader REFUSES --fold (qwen_export.py:240)
# and B1 carries fold=layers with 56 gains folded, so B1 was exported under the float32 default;
# export_packed.log confirms it.  This is the same class of check E7 s13 filed, and it is
# recorded here rather than waved through.
ABSENT_OK = {"load_dtype": "float32"}


def build_slice(model):
    """The pinned heldout slice, with byte counts RE-DERIVED under this arm's own tokenizer.

    common.get_slice caches on (part, n_seq, seq_len, seed) and NOT on the tokenizer.  E15 s4
    verified the Qwen2.5 -> Coder crossing on this slice (24/24 round-trip, byte-identical ids,
    same byte counts) and e16_calib_slice_check.py verified it independently for the CALIB slice
    the R3 export calibrates on.  The counts are still recomputed here rather than inherited.
    """
    import torch
    from transformers import AutoTokenizer
    import common as C

    d = torch.load(os.path.join(C.RESULTS, "slice_heldout_24x512_s1234.pt"))
    ids, meta = d["ids"], d["meta"]
    assert meta["ids_sha256"].startswith(SLICE_SHA), meta["ids_sha256"]
    assert ids.shape[1] == SEQ_LEN, ids.shape

    tk = AutoTokenizer.from_pretrained(model)
    byts = []
    for i in range(ids.shape[0]):
        row = ids[i, 1:].tolist()
        dec = tk.decode(row)
        assert tk(dec, add_special_tokens=False)["input_ids"] == row, \
            "seq %d does not round-trip under %s" % (i, model)
        byts.append(len(dec.encode("utf-8")))
    assert sum(byts) == meta["total_scored_bytes"], (sum(byts), meta["total_scored_bytes"])
    return ids, byts, meta, len(tk)


def check_sidecar(tag, path):
    """Section 3's hard design gate.  Returns (ok, list of offending fields)."""
    if tag not in GATE:
        return True, []
    ref_path, free = GATE[tag]
    if not os.path.exists(ref_path):
        return False, ["reference sidecar missing: %s" % ref_path]
    ref = json.load(open(ref_path, encoding="utf-8"))
    new = json.load(open(path + ".json", encoding="utf-8"))
    allowed = set(free) | set(FREE_ALWAYS)
    bad, noted = [], []
    for k in sorted(set(ref) | set(new)):
        if k in allowed:
            continue
        if ref.get(k) == new.get(k):
            continue
        if k not in ref and k in ABSENT_OK and new.get(k) == ABSENT_OK[k]:
            noted.append("%s absent in the reference sidecar, new = %r (admitted, see ABSENT_OK)"
                         % (k, new.get(k)))
            continue
        bad.append("%s: %r -> %r" % (k, ref.get(k), new.get(k)))
    for n in noted:
        print("      note: " + n, flush=True)
    return (not bad), bad


def export(tag, model, path, flags):
    cmd = [sys.executable, EXPORTER, "--model", model, "--quant", "packed", "--head-ternary",
           "--threads", "6", "--out", path] + flags
    print("  %-3s exporting: %s" % (tag, " ".join(cmd[2:])), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("export failed (%d):\n%s\n%s"
                         % (r.returncode, r.stdout[-3000:], r.stderr[-3000:]))
    for ln in r.stdout.strip().splitlines()[-3:]:
        print("      " + ln, flush=True)
    print("      export took %.0fs (%.2f h)" % (time.time() - t0, (time.time() - t0) / 3600.0),
          flush=True)
    return time.time() - t0


def run_arm(weights, idsp, n_seq):
    cmd = [ENGINE, "--weights", weights, "--threads", THREADS,
           "--seqlen", str(SEQ_LEN), "--bpb", idsp]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("engine failed (%d):\n%s\n%s"
                         % (r.returncode, r.stdout[-2000:], r.stderr[-2000:]))
    m = re.search(r"NATS_TOTAL ([0-9.eE+-]+)\s+N_PREDICTED (\d+)", r.stdout)
    if not m:
        raise SystemExit("no NATS_TOTAL in engine output:\n" + r.stdout[-2000:])
    total, npred = float(m.group(1)), int(m.group(2))
    assert npred == n_seq * (SEQ_LEN - 1), (npred, n_seq)
    return total, npred, time.time() - t0


def label(delta, band):
    if abs(delta) <= band:
        return "AT-CHANCE"
    return "RULE-FIXES-IT" if delta < 0 else "RULE-IS-NOT-THE-PROBLEM"


def main():
    smoke = N_SEQ != 24
    npred_expect = N_SEQ * (SEQ_LEN - 1)
    print("E16  seqs=%d  seqlen=%d  threads=%s%s"
          % (N_SEQ, SEQ_LEN, THREADS, "  [SMOKE]" if smoke else ""), flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = {"brief": "briefs/BRIEF_E16_R3_AT_7B.md", "smoke": smoke,
           "n_seq": N_SEQ, "seq_len": SEQ_LEN, "threads": int(THREADS),
           "reference_B1_bpb": B1_BPB, "reference_C0_e1_bpb": C0_REFERENCE, "arms": {}}
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
            if prev.get("n_seq") == N_SEQ:
                out["arms"] = prev.get("arms", {})
        except Exception:
            pass

    cache = {}
    for tag, model, vocab, w, flags, note in ARMS:
        if tag not in WANT:
            continue
        if tag in out["arms"]:
            print("  %-3s CACHED  BPB = %.9f" % (tag, out["arms"][tag]["bpb"]), flush=True)
            continue

        if model not in cache:
            cache[model] = build_slice(model)
        ids, byts, meta, tk_len = cache[model]
        ids_n, byts_n = ids[:N_SEQ], byts[:N_SEQ]
        scored = sum(byts_n)
        bpt = scored / float(npred_expect)
        chance = math.log(vocab, 2) / bpt
        band = chance - math.log(tk_len, 2) / bpt

        idsp = os.path.join(E7, "e16_ids_%dx%d.bin" % (N_SEQ, SEQ_LEN))
        ids_n.numpy().astype("<i4").tofile(idsp)

        if not os.path.exists(w):
            if flags is None:
                raise SystemExit("missing %s and it is not exportable here" % w)
            if not MAY_EXPORT:
                raise SystemExit("missing %s and E16_EXPORT=0" % w)
            export(tag, model, w, flags)

        ok, bad = check_sidecar(tag, w)
        if not ok:
            print("  %-3s *** DESIGN GATE FAILED -- more than the registered axis moved:" % tag,
                  flush=True)
            for b in bad:
                print("        " + b, flush=True)
            raise SystemExit("section 3's design gate failed for %s; the contrast would not be "
                             "the contrast that was registered." % tag)
        if tag in GATE:
            print("  %-3s design gate PASSES (only %s moved)"
                  % (tag, ", ".join(GATE[tag][1])), flush=True)

        print("  %-3s running %s (%.2f GB)  %s"
              % (tag, os.path.basename(w), os.path.getsize(w) / 1e9, note), flush=True)
        print("      %s  chance %.6f (V=%d)  band %.6f  scored_bytes %d  bytes/tok %.6f"
              % (model, chance, vocab, band, scored, bpt), flush=True)
        total, npred, secs = run_arm(w, idsp, N_SEQ)
        bpb = total / math.log(2) / scored
        out["arms"][tag] = {"weights": w, "model": model, "vocab_padded": vocab,
                            "tokenizer_len": tk_len, "scored_bytes": scored,
                            "bytes_per_token": bpt, "chance_bpb": chance, "band": band,
                            "nats_total": total, "n_predicted": npred,
                            "nats_per_token": total / npred, "bpb": bpb,
                            "vs_chance": bpb - chance, "seconds": secs, "note": note}
        print("      BPB = %.9f   (%.6f nats/tok, %.0fs)   vs chance %+.6f"
              % (bpb, total / npred, secs, bpb - chance), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

        if tag == "C0" and smoke:
            # E12's law: a control that passes on a SMOKE is not thereby a control.  The
            # converse holds too and matters here -- C0_REFERENCE is a 24-sequence number, so
            # comparing a 2-sequence reading against it is not a failed replication, it is the
            # wrong comparison.  G-R0 is registered on the full slice and is only read there.
            print("      [SMOKE] G-R0 NOT APPLICABLE: the reference is a 24-sequence number.",
                  flush=True)
        elif tag == "C0":
            d = abs(bpb - C0_REFERENCE)
            fires = d <= GR0_TOL and bpb < chance - band
            out["G_R0"] = {"fires": bool(fires), "abs_delta_vs_e1": d, "tol": GR0_TOL}
            print("\nG-R0  planted control: |%.9f - %.9f| = %.2e  (tol %.2f), below chance: %s"
                  " -> %s" % (bpb, C0_REFERENCE, d, GR0_TOL, bpb < chance - band,
                              "FIRES" if fires else "*** DOES NOT FIRE -- E16 IS VOID ***"),
                  flush=True)
            json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
            if not fires:
                raise SystemExit("E16 is VOID: the planted control did not fire, so no null "
                                 "below it would count.")

    print("\n---- gates ----", flush=True)
    c0 = out["arms"].get("C0")
    b2 = out["arms"].get("B2")
    b3 = out["arms"].get("B3")

    if c0 and "G_R0" in out:
        print("G-R0  control %s (|delta vs E1| = %.2e)"
              % ("FIRES" if out["G_R0"]["fires"] else "DOES NOT FIRE",
                 out["G_R0"]["abs_delta_vs_e1"]))
    if b2:
        d = b2["bpb"] - b2["chance_bpb"]
        out["G_R1"] = label(d, b2["band"])
        print("G-R1  BPB(B2) = %.9f   chance %.6f   delta %+.6f (band %.6f) -> %s"
              % (b2["bpb"], b2["chance_bpb"], d, b2["band"], out["G_R1"]))
        out["G_R2_vs_B1"] = b2["bpb"] - B1_BPB
        below = b2["bpb"] < b2["chance_bpb"] - b2["band"]
        print("G-R2  BPB(B2) - BPB(B1) = %+.9f   %s"
              % (out["G_R2_vs_B1"],
                 "the RULE's effect at 7B, fold fixed" if below else
                 "DESCRIPTIVE ONLY -- B2 is at or above chance, so this is NOT damage (E12 s2)"))
    if b3:
        d = b3["bpb"] - b3["chance_bpb"]
        out["G_R3"] = label(d, b3["band"])
        out["G_R3_vs_C0_reference"] = b3["bpb"] - C0_REFERENCE
        print("G-R3  BPB(B3) = %.9f   chance %.6f   delta %+.6f (band %.6f) -> %s"
              % (b3["bpb"], b3["chance_bpb"], d, b3["band"], out["G_R3"]))
        print("      BPB(B3) - 1.5B TQH (%.9f) = %+.9f   the 1.5B -> 7B step under R3"
              % (C0_REFERENCE, out["G_R3_vs_C0_reference"]))
    if b2 and b3:
        out["G_R4_fold"] = b2["bpb"] - b3["bpb"]
        print("G-R4  BPB(B2) - BPB(B3) = %+.9f   the fold under R3 at 7B, DESCRIPTIVE"
              % out["G_R4_fold"])

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
