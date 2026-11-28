#!/usr/bin/env python3
# Sparse Capacity, phase 4 (D-M): the verdict, from the registered rules of F4_BEYOND_VRAM_PROTOCOL.md section 4.
# It reads the files and applies the rules; it chooses nothing.
#
#   python benchmarks/sparse_capacity/sc4_verdict.py [--pace-only]
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "sparse_capacity")
GPU_ARMS, BIG = ("e2k", "e8k"), "e32k"
PACE_MIN = 0.5                # G1: tok/s of the beyond-VRAM arm / tok/s of the fastest GPU arm
GAIN_MIN = 0.005              # G2: BPB(e8k) - BPB(e32k), and on half B at the fitted temperature (sigma floor R1)
EXPL_MARGIN, EXPL_SE = 0.10, 4.0   # D-R fallback rule (PLAN, phase 4): the explicit book in the e32k run


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pace-only", action="store_true"); a = ap.parse_args()
    lines = []; say = lambda s: (print(s), lines.append(s))
    # G1, pace (the pace cell, official files only)
    t = {arm: load(f"f4_time_{arm}_s0.json") for arm in GPU_ARMS + (BIG,)}
    if any(v is None or not v.get("official") for v in t.values()):
        say("G1 PACE: MISSING (official pace-cell files for e2k, e8k, e32k)"); g1 = None
    else:
        ref = max(t[x]["tok_s"] for x in GPU_ARMS)
        pace = t[BIG]["tok_s"] / ref
        g1 = pace >= PACE_MIN
        say(f"G1 PACE: e32k {t[BIG]['tok_s']:.0f} tok/s / fastest GPU arm {ref:.0f} = {pace:.3f} (>= {PACE_MIN}) -> {'PASS' if g1 else 'FAIL'}")
    if a.pace_only:
        open(os.path.join(RES, "f4_verdict_pace.txt"), "w").write("\n".join(lines) + "\n"); return
    r = {arm: load(f"f4_{arm}_s0/result.json") for arm in GPU_ARMS + (BIG,)}
    if any(v is None for v in r.values()):
        say("G2 LOG E: MISSING (result.json of every arm)"); g2 = None
    else:
        b = {k: v["val"]["bpb"] for k, v in r.items()}; bB = {k: v["val"]["bpb_B_tfit"] for k, v in r.items()}
        top1 = {k: v["val"]["top1"] for k, v in r.items()}
        mono = b["e2k"] > b["e8k"] > b[BIG]
        gain, gainB = b["e8k"] - b[BIG], bB["e8k"] - bB[BIG]
        g2 = mono and gain >= GAIN_MIN and gainB >= GAIN_MIN
        say("BPB val: " + ", ".join(f"{k} {b[k]:.4f}" for k in b) + "; top-1: " + ", ".join(f"{k} {top1[k]:.4f}" for k in top1))
        say(f"steps: e2k->e8k {b['e2k'] - b['e8k']:+.4f}, e8k->e32k {gain:+.4f}; half B at the fitted temperature e8k->e32k {gainB:+.4f}")
        say(f"G2 LOG E: monotone {mono}, last step {gain:.4f} and {gainB:.4f} (>= {GAIN_MIN}) -> {'PASS' if g2 else 'FAIL'}")
        e = r[BIG]["e"]; ctl = max(e["ctl_copy"], e["ctl_common_rule"], e["ctl_wrong_name"])
        se = math.sqrt(max(ctl * (1 - ctl), 1e-9) / e["n"]); marg = e["acc"] - ctl
        dr = marg >= EXPL_MARGIN and marg >= EXPL_SE * se
        say(f"D-R fallback (reading): explicit book in e32k {e['acc']:.3f}, strongest control {ctl:.3f}, margin {marg:+.3f} "
            f"({marg / se:.1f} SE) -> {'passes: D-R measurable at this core' if dr else 'does not pass: the D-R phase with a larger core opens after phase 4'}")
    if g1 is not None and g2 is not None:
        say(f"PHASE 4: {'PASS' if (g1 and g2) else 'FAIL'}")
    open(os.path.join(RES, "f4_verdict.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
