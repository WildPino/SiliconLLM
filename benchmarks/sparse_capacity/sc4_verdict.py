#!/usr/bin/env python3
# Sparse Capacity, phase 4: the verdict, from the registered rules of F4_BEYOND_VRAM_PROTOCOL.md section 4.
# It reads the files and applies the rules; it chooses nothing. D-M = G0 and G1; the plan's phase-4 gate = G2.
#
#   python benchmarks/sparse_capacity/sc4_verdict.py [--pace-only] [--g0-only]
import argparse, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "sparse_capacity")
STEPS = 29_296
GPU_ARMS, BIG, EQ = ("e2k", "e8k"), "e32k", "e2k-h"
G0_POINTS, G0_TOL = (2000, 4000), 0.010     # G0: |curve BPB e2k-h - e2k| at these steps (phase 3's curve tolerance)
PACE_MIN = 0.5                # G1: tok/s of e32k / tok/s of the fastest GPU arm (same code)
GAIN_MIN = 0.005              # G2: BPB(e8k) - BPB(e32k), on the whole val and on half B at the fitted temperature
GAP_FROM, GAP_GROW = 14_000, 0.002          # G2 FAIL reading: curve gap e8k - e32k at the end minus at GAP_FROM
EXPL_MARGIN, EXPL_SE = 0.10, 4.0            # D-R fallback rule (PLAN, phase 4): the explicit book in the e32k run
NAMED_BOOKS = ("64", "512")                 # D-R reading, read only if the explicit book passes in e32k (protocol section 6):
NAMED_CTL = (0.10, 4.0)                     #   e32k above its strongest control by >= 0.10 and >= 4 binomial SE, and
NAMED_DIFF = (0.05, 4.0)                    #   e32k above e2k by >= 0.05 and >= 4 SE of the difference, in >= 1 book


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def log(arm):
    p = os.path.join(RES, f"f4_{arm}_s0", "train.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def curve(arm):
    return {r["step"]: r for r in log(arm) if r.get("kind") == "curve"}


def provenance(arm, need_steps):
    """A run counts when its config says registered and pushed with no dirty files, and it reached its steps."""
    cfg = load(f"f4_{arm}_s0/config.json")
    if cfg is None:
        return "missing config"
    bad = [k for k, ok in (("not registered", cfg.get("registered")), ("not pushed", cfg.get("pushed")),
                           ("dirty", not cfg.get("dirty_files"))) if not ok]
    stops = [r for r in log(arm) if r.get("kind") == "stop"]
    if stops:
        bad.append(f"stopped: {stops[-1]['why']} at step {stops[-1]['step']}")
    if need_steps is not None:
        r = load(f"f4_{arm}_s0/result.json")
        if r is None or r.get("steps") != need_steps:
            bad.append("incomplete")
    return "; ".join(bad)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pace-only", action="store_true"); ap.add_argument("--g0-only", action="store_true")
    a = ap.parse_args()
    lines = []; say = lambda s: (print(s), lines.append(s))
    out = "f4_verdict_pace.txt" if a.pace_only else "f4_verdict_g0.txt" if a.g0_only else "f4_verdict.txt"

    # G1, pace (pace cell, official files only); the readings beside it
    t = {arm: load(f"f4_time_{arm}_s0.json") for arm in GPU_ARMS + (BIG, "m1")}
    if any(v is None or not v.get("official") for v in t.values()):
        say("G1 PACE: MISSING (official pace-cell files for e2k, e8k, e32k, m1)"); g1 = None
    else:
        ref = max(t[x]["tok_s"] for x in GPU_ARMS)
        pace = t[BIG]["tok_s"] / ref
        g1 = pace >= PACE_MIN
        say(f"G1 PACE: e32k {t[BIG]['tok_s']:.0f} tok/s / fastest GPU arm {ref:.0f} = {pace:.3f} (>= {PACE_MIN}) -> {'PASS' if g1 else 'FAIL'}")
        say(f"  reading: e32k / m1 (phase 2, dense, E = 128) {t[BIG]['tok_s']:.0f} / {t['m1']['tok_s']:.0f} = {t[BIG]['tok_s'] / t['m1']['tok_s']:.3f}; "
            f"e2k / m1 = {t['e2k']['tok_s'] / t['m1']['tok_s']:.3f} (the cost of the sparse computation alone)")
        nh = load(f"f4_time_{BIG}_nohost_s0.json"); parts = load(f"f4_time_parts_{BIG}_s0.json")
        if nh:
            exp_host = t[BIG]["step_s"] - nh["step_s"]
            say(f"  reading: exposed host cost (e32k step - e32k step with the host update skipped) {exp_host:.2f} s of {t[BIG]['step_s']:.2f} s")
            if parts and parts.get("official"):
                m1s, dr, full = t["m1"]["step_s"], parts["delta_route_per_step_s"], t[BIG]["host"]["host_update"]
                proj, proj_full = m1s + dr + exp_host, m1s + dr + full
                say(f"  reading, projected pace if the sparse computation cost as much as m1's dense bank: m1 step {m1s:.2f} s "
                    f"+ sparse routing over m1's router scan {dr:.2f} s + exposed host {exp_host:.2f} s = {proj:.2f} s -> "
                    f"{m1s / proj:.3f} of m1; with the whole host update exposed ({full:.2f} s, the bound: a shorter forward "
                    f"hides less) {proj_full:.2f} s -> {m1s / proj_full:.3f} of m1")
        for x in t:
            say(f"  {x}: step {t[x]['step_s']:.2f} s (min {t[x]['step_min']:.2f}, median {t[x]['step_median']:.2f}, max {t[x]['step_max']:.2f}); "
                f"CPU load {t[x]['cpu_load_before']}/{t[x]['cpu_load_after']}%; other python: {len(t[x]['other_python'] or [])}")
    if a.pace_only:
        open(os.path.join(RES, out), "w").write("\n".join(lines) + "\n"); return

    # G0, long equivalence: e2k-h (host path) against e2k (GPU path), same seed, schedule and batches
    pe, pg = provenance(EQ, G0_POINTS[-1]), provenance("e2k", None)
    ce, cg = curve(EQ), curve("e2k")
    if pe or pg or any(s not in ce or s not in cg for s in G0_POINTS):
        say(f"G0 EQUIVALENCE: MISSING ({pe or 'e2k-h ok'}; {pg or 'e2k ok'}; curve points {G0_POINTS})"); g0 = None
    else:
        d = {s: ce[s]["bpb"] - cg[s]["bpb"] for s in G0_POINTS}
        g0 = all(abs(v) <= G0_TOL for v in d.values())
        say("G0 EQUIVALENCE: curve BPB e2k-h - e2k " + ", ".join(f"step {s} {v:+.4f}" for s, v in d.items())
            + f" (|.| <= {G0_TOL}) -> {'PASS' if g0 else 'FAIL'}")
    if a.g0_only:
        open(os.path.join(RES, out), "w").write("\n".join(lines) + "\n"); return

    # G2, log E, at equal router and tokens
    arms = GPU_ARMS + (BIG,)
    prov = {x: provenance(x, STEPS) for x in arms}
    stopped = {x: p for x, p in prov.items() if "stopped" in p}
    r = {x: load(f"f4_{x}_s0/result.json") for x in arms}
    if stopped:
        say("G2 LOG E: VOID (" + "; ".join(f"{x} {p}" for x, p in stopped.items()) + ")"); g2 = None
    elif any(prov.values()):
        say("G2 LOG E: MISSING (" + "; ".join(f"{x} {p}" for x, p in prov.items() if p) + ")"); g2 = None
    else:
        effs = {x: r[x]["usage"]["eff_min"] for x in arms}
        b = {x: r[x]["val"]["bpb"] for x in arms}; bB = {x: r[x]["val"]["bpb_B_tfit"] for x in arms}
        top1 = {x: r[x]["val"]["top1"] for x in arms}
        mono = b["e2k"] > b["e8k"] > b[BIG]
        gain, gainB = b["e8k"] - b[BIG], bB["e8k"] - bB[BIG]
        g2 = mono and gain >= GAIN_MIN and gainB >= GAIN_MIN
        say("BPB val: " + ", ".join(f"{k} {b[k]:.4f}" for k in b) + "; top-1: " + ", ".join(f"{k} {top1[k]:.4f}" for k in top1))
        say("expert usage eff_min (final): " + ", ".join(f"{k} {effs[k]:.3f}" for k in effs))
        say(f"steps: e2k->e8k {b['e2k'] - b['e8k']:+.4f}, e8k->e32k {gain:+.4f}; half B at the fitted temperature e8k->e32k {gainB:+.4f}")
        say(f"G2 LOG E: monotone {mono}, last step {gain:.4f} and {gainB:.4f} (>= {GAIN_MIN}) -> {'PASS' if g2 else 'FAIL'}")
        c8, c32 = curve("e8k"), curve(BIG)
        if GAP_FROM in c8 and GAP_FROM in c32 and STEPS in c8 and STEPS in c32:
            g_mid, g_end = c8[GAP_FROM]["bpb"] - c32[GAP_FROM]["bpb"], c8[STEPS]["bpb"] - c32[STEPS]["bpb"]
            grow = g_end - g_mid >= GAP_GROW
            say(f"  curve gap e8k - e32k: step {GAP_FROM} {g_mid:+.4f}, end {g_end:+.4f} -> "
                f"{'still growing (>= %.3f): if G2 fails, the recommendation is more tokens' % GAP_GROW if grow else 'flat: if G2 fails, the gain closes at this core'}")
        e = r[BIG]["e"]; ctl = max(e["ctl_copy"], e["ctl_common_rule"], e["ctl_wrong_name"])
        se = math.sqrt(max(ctl * (1 - ctl), 1e-9) / e["n"]); marg = e["acc"] - ctl
        dr = marg >= EXPL_MARGIN and marg >= EXPL_SE * se
        say(f"D-R fallback (reading): explicit book in e32k {e['acc']:.3f}, strongest control {ctl:.3f}, margin {marg:+.3f} "
            f"({marg / se:.1f} SE) -> {'passes: D-R measurable at this core' if dr else 'does not pass: the D-R phase with a larger core opens after phase 4'}")
        if not dr:
            say("D-R named books (reading): not read, the explicit book did not pass in e32k")
        else:
            sig = []
            for bk in NAMED_BOOKS:
                pb, p2 = r[BIG]["p"][bk], r["e2k"]["p"][bk]
                c = max(pb["ctl_copy"], pb["ctl_common_rule"], r[BIG]["p"]["unseen"]); n = pb["n"]
                se_c = math.sqrt(max(c * (1 - c), 1e-9) / n); m_c = pb["acc"] - c
                se_d = math.sqrt(max(pb["acc"] * (1 - pb["acc"]) / n + p2["acc"] * (1 - p2["acc"]) / p2["n"], 1e-9))
                d = pb["acc"] - p2["acc"]
                ok = m_c >= NAMED_CTL[0] and m_c >= NAMED_CTL[1] * se_c and d >= NAMED_DIFF[0] and d >= NAMED_DIFF[1] * se_d
                sig.append(ok)
                say(f"  P{bk}: e32k {pb['acc']:.3f} (n {n}), strongest control {c:.3f}, margin {m_c:+.3f} ({m_c / se_c:.1f} SE); "
                    f"e2k {p2['acc']:.3f}, e32k - e2k {d:+.3f} ({d / se_d:.1f} SE) -> {'signal' if ok else 'no signal'}")
            say(f"D-R named books (registered reading, not a gate of D-M): {'SIGNAL' if any(sig) else 'NO SIGNAL'}")
    if g0 is not None and g1 is not None:
        say(f"D-M (G0 and G1): {'PASS' if (g0 and g1) else 'FAIL'}")
    if g2 is not None:
        say(f"PHASE 4 GATE (G2, log E): {'PASS' if g2 else 'FAIL'}")
    open(os.path.join(RES, out), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
