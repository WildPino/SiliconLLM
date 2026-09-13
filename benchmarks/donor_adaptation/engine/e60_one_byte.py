#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E60 -- what does a weight cost at ONE byte, in speed and in fidelity?

Brief: `docs/research/donor_adaptation/briefs/BRIEF_E60_THE_RUNG_THAT_WAS_NEVER_BUILT.md`
Pre-registered and pushed (see git log) before this file existed.

The engine's precision ladder is 4 B/weight (fp32) and 0.5 B/weight (packed ternary) and nothing
between.  E58 measured the faithful arms AT the memory wall (38.5/38.7 GB/s) and the token at
96-98% weight-organ time, so on a faithful arm the only lever left is bytes per weight -- and the
ladder skips the whole middle.  `matvec_sel`'s quant==1 fallback is already a general int8 x fp32
dot with a per-row scale, so the middle rung is an EXPORTER question, not a kernel one.

WHAT THIS RUNNER IS CAREFUL ABOUT.  E59 nearly published "the kernel costs nothing" because it
scored a treatment against a reference that was already on its floor.  Here the control arm F32
reads 160/160 against HuggingFace, so the fidelity counter has its full range in the direction
that fails -- and G-E60b makes the instrument prove it reproduces E57's known positives (F32 at
160, PACKED at 3 and 10) before any treatment number is read.

Every arm runs on ONE binary (donor_engine_e60.exe), so no comparison crosses builds, and
G-E60a proves that binary is token-identical to the frozen donor_engine_e53.exe on both paths
that already existed.
"""
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e44_interval as E44                                          # noqa: E402
import e51_math_errno as E51                                        # noqa: E402

OUTDIR = os.path.join(HERE, "results")
ENGINE = os.path.join(HERE, "donor_engine_e60.exe")
FROZEN = os.path.join(HERE, "donor_engine_e53.exe")     # E59's binary, for G-E60a
THREADS = 6
BENCH_N = 160
REPS = 9
SEED = 6001

E1 = os.path.join("D:", os.sep, "_ktmp", "e1")
E60 = os.path.join("D:", os.sep, "_ktmp", "e60")

# The standard slice: 24 x 512, 12264 predicted tokens, 51870 scored bytes.  The ids file is
# byte-identical across every E1 arm except the 15b_f32 SMOKE (2048 ids), which is why the tq
# copy is named here rather than the f32 one.
BPB_IDS = os.path.join(E1, "ids_qwen25-05b_tqh.bin")
BPB_SEQLEN = 512
BPB_BYTES = 51870
BPB_NPRED = 12264
CHANCE_BPB = 4.069819

# E1's measured BPB on that slice (density/results/e1_bpb_through_engine_*.json).  15b F32 was
# only ever run at 4 sequences (n=2044), so this runner MEASURES it at 24 -- an owed number.
E1_BPB = {("05b", "F32"): 0.8718104610834241, ("05b", "PACKED"): 4.531233733626962,
          ("15b", "PACKED"): 3.47570669163278}

# E57's greedy counts against HuggingFace -- G-E60b's known positives.
E57_HF = {("05b", "F32"): 160, ("05b", "PACKED"): 3,
          ("15b", "F32"): 160, ("15b", "PACKED"): 10}
HF_KEY = {"05b": "Qwen/Qwen2.5-0.5B", "15b": "Qwen/Qwen2.5-1.5B"}

# arm -> (file, is_new).  T1 exists only at 0.5 B: it holds PACKED's trits at I8's bytes, which
# is what makes G-E60c able to say whether speed on this rung is a byte effect or a value one.
ARMS = {
    "05b": [("F32",    os.path.join(E1, "qwen25-05b_f32.bin")),
            ("PACKED", os.path.join(E1, "qwen25-05b_tqh.bin")),
            ("I8",     os.path.join(E60, "qwen25-05b_i8h.bin")),
            ("T1",     os.path.join(E60, "qwen25-05b_t1h.bin"))],
    "15b": [("F32",    os.path.join(E1, "qwen25-15b_f32.bin")),
            ("PACKED", os.path.join(E1, "qwen25-15b_tqh.bin")),
            ("I8",     os.path.join(E60, "qwen25-15b_i8h.bin"))],
}
BAR = 50.0


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


def quantile(v, q):
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    p = q * (len(s) - 1)
    lo = int(p)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (p - lo)


def boot_ci(v, rng, n=2000):
    b = []
    for _ in range(n):
        b.append(statistics.median([v[rng.randrange(len(v))] for _ in v]))
    b.sort()
    return b[int(0.025 * n)], b[int(0.975 * n) - 1]


def paired_agreement(a, b):
    same = tot = 0
    for x, y in zip(a, b):
        for u, v in zip(x, y):
            tot += 1
            same += (u == v)
    return same, tot


def first_divergence(a, b):
    for i, (u, v) in enumerate(zip(a, b)):
        if u != v:
            return i
    return -1


# ============================================================ the decision functions
def g_e60a(pairs):
    """PLANTED: the quant=5 alias must be inert.  `pairs` = [(arm, identical_bool), ...]."""
    if not pairs:
        return "DEAD -- nothing compared"
    bad = [n for n, ok in pairs if not ok]
    if bad:
        return "DEAD -- e60 differs from e53 on " + ",".join(bad)
    return "FIRES"


def g_e60b(obs):
    """PLANTED: the fidelity instrument must reproduce E57's known positives within +-1.

    `obs` = {(size, arm): matched}.  A tolerance of 1 and not 0 because E57's counts were taken
    on a different binary; a wider tolerance would let a broken instrument through, and a
    tolerance of 0 would make a build difference look like an instrument failure.
    """
    if not obs:
        return "DEAD -- no known-positive was scored"
    bad = ["%s/%s %d!=%d" % (s, a, m, E57_HF[(s, a)])
           for (s, a), m in sorted(obs.items()) if abs(m - E57_HF[(s, a)]) > 1]
    return "FIRES" if not bad else "DEAD -- " + "; ".join(bad)


def g_e60c(ci_i8, ci_t1, med_i8, med_t1):
    """Speed is a byte effect: the two 1 B/weight arms must not separate.

    The bar is the dispersion the axis actually shows on this sweep (the wider of the two
    bootstrap half-widths), never a tolerance picked in advance.
    """
    hw = max(ci_i8[1] - ci_i8[0], ci_t1[1] - ci_t1[0]) / 2.0
    if hw <= 0:
        return "UNRESOLVABLE", hw
    sep = ci_i8[1] < ci_t1[0] or ci_t1[1] < ci_i8[0]
    if sep:
        return "SEPARATED", hw
    return ("WITHIN", hw) if abs(med_i8 - med_t1) <= hw else ("UNRESOLVABLE", hw)


def g_e60d(ci, bar=BAR):
    if ci[0] > bar:
        return "ABOVE"
    if ci[1] < bar:
        return "BELOW"
    return "STRADDLES"


def g_e60e(counts, dbpb, n_scored, floor=14, need=150, dmax=0.020):
    """counts/dbpb keyed by size.  E18 part A's floor is 11-12 plus E17's margin of 2."""
    if not counts:
        return "DEAD -- nothing scored"
    if any(c <= floor for c in counts.values()):
        return "AT-FLOOR"
    ok = all(c >= need for c in counts.values())
    ok = ok and all(d is not None and d <= dmax for d in dbpb.values())
    return "FAITHFUL" if ok else "DEGRADED"


def selftest():
    n = [0]

    def chk(name, cond):
        n[0] += 1
        log("  %-64s %s" % (name, "fires" if cond else "*** DEAD ***"))
        if not cond:
            raise SystemExit("a decision function did not fire on a known-positive.  STOP.")

    chk("A-1 G-E60a FIRES when every pair is identical",
        g_e60a([("F32", True), ("PACKED", True)]) == "FIRES")
    chk("A-2 and is DEAD when one differs",
        g_e60a([("F32", True), ("PACKED", False)]).startswith("DEAD"))
    chk("A-3 and is DEAD when nothing was compared", g_e60a([]).startswith("DEAD"))
    chk("B-1 G-E60b FIRES on E57's own numbers",
        g_e60b({("05b", "F32"): 160, ("05b", "PACKED"): 3}) == "FIRES")
    chk("B-2 and tolerates +-1", g_e60b({("05b", "PACKED"): 4}) == "FIRES")
    chk("B-3 and is DEAD at +-2", g_e60b({("05b", "PACKED"): 5}).startswith("DEAD"))
    chk("B-4 and is DEAD with nothing scored", g_e60b({}).startswith("DEAD"))
    chk("C-1 G-E60c WITHIN when the gap is inside the measured dispersion",
        g_e60c((60.0, 64.0), (59.0, 63.0), 62.0, 61.0)[0] == "WITHIN")
    chk("C-2 and SEPARATED on disjoint intervals",
        g_e60c((60.0, 62.0), (50.0, 52.0), 61.0, 51.0)[0] == "SEPARATED")
    chk("C-3 and UNRESOLVABLE when overlapping but further apart than the dispersion",
        g_e60c((50.0, 62.0), (49.0, 61.0), 61.0, 50.0)[0] == "UNRESOLVABLE")
    chk("D-1 G-E60d ABOVE when the whole interval clears the bar",
        g_e60d((55.0, 60.0)) == "ABOVE")
    chk("D-2 BELOW when it does not", g_e60d((40.0, 48.0)) == "BELOW")
    chk("D-3 STRADDLES when it spans the bar", g_e60d((48.0, 55.0)) == "STRADDLES")
    chk("E-1 G-E60e FAITHFUL on high counts and a small dBPB",
        g_e60e({"05b": 158, "15b": 159}, {"05b": 0.004, "15b": 0.003}, 160) == "FAITHFUL")
    chk("E-2 DEGRADED when the count holds but BPB does not",
        g_e60e({"05b": 158, "15b": 159}, {"05b": 0.9, "15b": 0.003}, 160) == "DEGRADED")
    chk("E-3 DEGRADED when one scale misses the count",
        g_e60e({"05b": 120, "15b": 159}, {"05b": 0.004, "15b": 0.003}, 160) == "DEGRADED")
    chk("E-4 AT-FLOOR at E18's floor, which OUTRANKS the count clause",
        g_e60e({"05b": 12, "15b": 159}, {"05b": 0.004, "15b": 0.003}, 160) == "AT-FLOOR")
    chk("E-5 and DEAD with nothing scored", g_e60e({}, {}, 160).startswith("DEAD"))
    chk("F-1 paired agreement counts a changed token",
        paired_agreement([[1, 2, 3]], [[1, 9, 3]]) == (2, 3))
    chk("F-2 first divergence is located", first_divergence([1, 2, 3], [1, 9, 3]) == 1)
    log("")
    log("  %d of %d fire." % (n[0], n[0]))


# ============================================================ measurement primitives
def generate(engine, weights, n_new, n_prompts, tag):
    runs = []
    for i in range(n_prompts):
        pfx = os.path.join(E51.TMP, "e60_%s_p%d" % (tag, i))
        ids = os.path.join(E51.TMP, "p%d.bin" % i)
        base = [engine, "--weights", weights, "--threads", str(THREADS)]
        r = E51.parse_gen(E51.run(base + ["--generate", ids, str(n_new), pfx], "gen")[0])
        runs.append({"prompt": i, "ids": r["ids"], "decode_toks": r.get("decode_toks")})
        for f in (pfx + ".prefill.bin", pfx + ".ids.bin"):
            if os.path.exists(f):
                os.remove(f)
    return runs


def bpb(engine, weights):
    """(bpb, nats, n_predicted, seconds).  The engine reports NATS; BPB is ours."""
    cmd = [engine, "--weights", weights, "--threads", str(THREADS),
           "--seqlen", str(BPB_SEQLEN), "--bpb", BPB_IDS]
    txt, dt = E51.run(cmd, "bpb")
    m = E51.RE_NATS.search(txt)
    if not m:
        raise SystemExit("no NATS_TOTAL for " + weights)
    tot, npred = float(m.group(1)), int(m.group(2))
    if npred != BPB_NPRED:
        raise SystemExit("slice changed: %d predicted, expected %d" % (npred, BPB_NPRED))
    return tot / (math.log(2.0) * BPB_BYTES), tot, npred, dt


def one_cell(weights):
    sp = E44.Split()
    cmd = [ENGINE, "--weights", weights, "--threads", str(THREADS), "--bench", str(BENCH_N)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sout, serr = p.communicate()
    child = E44._process_times(int(p._handle))
    if p.returncode != 0:
        sys.stderr.write(serr.decode("utf-8", "replace")[-1200:])
        raise SystemExit("engine failed: " + " ".join(cmd))
    occ, foc = sp.close(child)
    txt = sout.decode("utf-8", "replace")
    m = E44.RE_BENCH.search(txt)
    if not m:
        raise SystemExit("no BENCH line")
    # The CONFIG line names the format.  It is captured, not trusted from the filename: an
    # int8-valued and a ternary-valued file differ in nothing else, which is why quant=5 exists.
    cfg = [l for l in txt.splitlines() if l.startswith("CONFIG")]
    cell = {"rate": float(m.group(3)), "occ": occ, "foreign": foc,
            "config": cfg[0] if cfg else ""}
    cell.update(sp.record())
    return cell


def save(out):
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, "e60_one_byte.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return p


# ============================================================
def main():
    log("E60 -- the rung that was never built: one byte per weight")
    log("brief: BRIEF_E60_THE_RUNG_THAT_WAS_NEVER_BUILT.md")
    log("")
    log("== self-test ==")
    selftest()
    if "--selftest" in sys.argv:
        return
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    e6 = json.load(open(os.path.join(E51.E6_RES, "engine.json")))
    ref = json.load(open(os.path.join(E51.E6_RES, "ref.json")))
    n_new = e6["n_new"]
    n_p = len(e6["prompt_ids"])
    E51.write_prompt_ids(e6["prompt_ids"])
    os.makedirs(E51.TMP, exist_ok=True)
    rng = random.Random(SEED)

    out = {"brief": "BRIEF_E60_THE_RUNG_THAT_WAS_NEVER_BUILT.md",
           "engine": os.path.basename(ENGINE), "frozen": os.path.basename(FROZEN),
           "bench_n": BENCH_N, "reps": REPS, "threads": THREADS,
           "n_new": n_new, "n_prompts": n_p, "slice": {"n_seq": 24, "seq_len": BPB_SEQLEN,
           "bytes": BPB_BYTES, "n_predicted": BPB_NPRED, "chance_bpb": CHANCE_BPB},
           "sizes": {}}

    sizes = [s for s in ("05b", "15b") if only is None or s == only]
    gens_all, hf_obs, counts, dbpbs = {}, {}, {}, {}

    # ---------------------------------------------------------------- phase 1: G-E60a
    log("")
    log("== G-E60a -- PLANTED: donor_engine_e60 must be token-identical to the frozen e53 ==")
    log("   on both paths that already existed.  Nothing below is read if this does not fire.")
    pairs = []
    for size in sizes:
        for arm, wp in ARMS[size]:
            if arm not in ("F32", "PACKED") or not os.path.exists(wp):
                continue
            a = generate(ENGINE, wp, n_new, n_p, "par60_%s_%s" % (size, arm))
            b = generate(FROZEN, wp, n_new, n_p, "par53_%s_%s" % (size, arm))
            ia = [r["ids"] for r in a]
            ib = [r["ids"] for r in b]
            same, tot = paired_agreement(ia, ib)
            pairs.append(("%s/%s" % (size, arm), same == tot))
            log("   %-4s %-7s  %d/%d identical" % (size, arm, same, tot))
            gens_all[(size, arm)] = [r["ids"][-n_new:] for r in a]
    va = g_e60a(pairs)
    log("   G-E60a : %s" % va)
    out["G_E60a"] = {"pairs": [[n, ok] for n, ok in pairs], "verdict": va}
    save(out)
    if va != "FIRES":
        raise SystemExit("the planted control did not fire.  STOP.")

    # ---------------------------------------------------------------- phase 2+3: fidelity
    log("")
    log("== G-E60b / G-E60e -- fidelity, %d prompts x %d new tokens, greedy ==" % (n_p, n_new))
    for size in sizes:
        hf = HF_KEY[size]
        rr = {x["prompt"]: x for x in ref[hf]}
        log("")
        log("-- %s --" % size)
        log("   %-7s %10s %12s %10s   %s" % ("arm", "vs HF", "vs F32", "1st div vs F32", "note"))
        sz = {"arms": {}}
        for arm, wp in ARMS[size]:
            if not os.path.exists(wp):
                log("   %-7s SKIP -- not on disk" % arm)
                continue
            if (size, arm) not in gens_all:
                recs = generate(ENGINE, wp, n_new, n_p, "fid_%s_%s" % (size, arm))
                gens_all[(size, arm)] = [r["ids"][-n_new:] for r in recs]
            g = gens_all[(size, arm)]
            match = tot = 0
            for i, row in enumerate(g):
                t = rr[i]["ids"][-n_new:]
                for k in range(n_new):
                    tot += 1
                    match += (row[k] == t[k])
            base = gens_all.get((size, "F32"))
            vf = paired_agreement(base, g) if base else (0, 0)
            dv = [first_divergence(x, base[i]) for i, x in enumerate(g)] if base else []
            note = ""
            if (size, arm) in E57_HF:
                hf_obs[(size, arm)] = match
                note = "E57 says %d -> %s" % (E57_HF[(size, arm)],
                        "REPRODUCED" if abs(match - E57_HF[(size, arm)]) <= 1 else "*** DEVIATED ***")
            log("   %-7s %6d/%-4d %7d/%-5d %14s   %s" % (arm, match, tot, vf[0], vf[1],
                                                         str(dv), note))
            sz["arms"][arm] = {"weights": wp, "vs_hf": [match, tot], "vs_f32": list(vf),
                               "first_div_vs_f32": dv}
        out["sizes"][size] = sz
        save(out)

    vb = g_e60b(hf_obs)
    log("")
    log("   G-E60b : %s" % vb)
    out["G_E60b"] = {"observed": dict(("%s/%s" % k, v) for k, v in hf_obs.items()),
                     "verdict": vb}
    save(out)
    if vb != "FIRES":
        raise SystemExit("the fidelity instrument did not reproduce E57.  No verdict.  STOP.")

    # ---------------------------------------------------------------- BPB
    log("")
    log("== BPB on the standard slice (24 x 512, %d predicted, chance %.6f) ==" % (BPB_NPRED,
                                                                                   CHANCE_BPB))
    for size in sizes:
        f32b = None
        for arm, wp in ARMS[size]:
            if not os.path.exists(wp) or arm == "T1":
                continue
            known = E1_BPB.get((size, arm))
            if known is not None and arm != "F32":
                log("   %-4s %-7s %.9f   (E1, not re-run)" % (size, arm, known))
                out["sizes"][size]["arms"][arm]["bpb"] = known
                out["sizes"][size]["arms"][arm]["bpb_source"] = "E1"
                continue
            b, nats, npred, dt = bpb(ENGINE, wp)
            src = "measured"
            if known is not None:
                d = abs(b - known)
                src = "measured, E1 says %.9f, |d|=%.2e" % (known, d)
            log("   %-4s %-7s %.9f   (%.0f s)  %s" % (size, arm, b, dt, src))
            out["sizes"][size]["arms"][arm]["bpb"] = b
            out["sizes"][size]["arms"][arm]["bpb_source"] = src
            out["sizes"][size]["arms"][arm]["bpb_seconds"] = dt
            save(out)
        a = out["sizes"][size]["arms"]
        f32b = a.get("F32", {}).get("bpb")
        for arm in a:
            if f32b is not None and a[arm].get("bpb") is not None:
                a[arm]["dbpb_vs_f32"] = a[arm]["bpb"] - f32b
        if "I8" in a and a["I8"].get("dbpb_vs_f32") is not None:
            dbpbs[size] = a["I8"]["dbpb_vs_f32"]
            counts[size] = a["I8"]["vs_hf"][0]
        save(out)

    ve = g_e60e(counts, dbpbs, n_p * n_new)
    log("")
    log("   G-E60e : %s   (counts %s, dBPB %s)" % (ve, counts,
        dict((k, round(v, 6)) for k, v in dbpbs.items())))
    log("   G-E60f : the RANK partner for that SCORE is the `vs HF` greedy column above,")
    log("            registered in the brief, not assembled after the fact.")
    out["G_E60e"] = {"counts": counts, "dbpb": dbpbs, "verdict": ve}
    save(out)

    # ---------------------------------------------------------------- speed
    log("")
    log("== G-E60c / G-E60d -- speed, %d reps, --bench %d, arms INTERLEAVED in one sweep ==" %
        (REPS, BENCH_N))
    for size in sizes:
        avail = [(a, w) for a, w in ARMS[size] if os.path.exists(w)]
        log("")
        log("-- %s --" % size)
        for arm, wp in avail:
            c = one_cell(wp)
            log("   warm-up %-7s DISCARDED: %7.2f tok/s   %s" % (arm, c["rate"], c["config"]))
        cells = dict((a, []) for a, _ in avail)
        for r in range(REPS):
            for arm, wp in avail:
                c = one_cell(wp)
                cells[arm].append(c)
                log("   rep %d  %-7s %7.2f tok/s  foreign %5.2f%%" % (r + 1, arm, c["rate"],
                                                                      c["foreign"]))
        log("")
        log("   %-7s %9s %9s %18s %9s %9s" % ("arm", "median", "min..max", "95% CI",
                                              "foreign", "vs 50"))
        med, ci = {}, {}
        for arm, _ in avail:
            v = [c["rate"] for c in cells[arm]]
            fo = [c["foreign"] for c in cells[arm]]
            med[arm] = statistics.median(v)
            ci[arm] = boot_ci(v, rng)
            out["sizes"][size]["arms"][arm]["speed"] = {
                "rates": v, "median": med[arm], "ci": list(ci[arm]),
                "foreign": fo, "foreign_mean": sum(fo) / len(fo),
                "over_occ_bar": sum(1 for x in fo if x > 4.39)}
            log("   %-7s %9.2f %4.1f..%-4.1f %8.2f..%-8.2f %8.2f%% %9s"
                % (arm, med[arm], min(v), max(v), ci[arm][0], ci[arm][1],
                   sum(fo) / len(fo), g_e60d(ci[arm])))
        save(out)

        if "I8" in med and "T1" in med:
            vc, hw = g_e60c(ci["I8"], ci["T1"], med["I8"], med["T1"])
            log("")
            log("   G-E60c : %s   |I8-T1| = %.2f tok/s, measured dispersion half-width %.2f"
                % (vc, abs(med["I8"] - med["T1"]), hw))
            out["sizes"][size]["G_E60c"] = {"verdict": vc, "half_width": hw,
                                            "gap": abs(med["I8"] - med["T1"])}
        if "I8" in ci:
            vd = g_e60d(ci["I8"])
            log("   G-E60d : I8 at %s vs the %.1f tok/s bar -> %s  (CI %.2f..%.2f)"
                % (size, BAR, vd, ci["I8"][0], ci["I8"][1]))
            out["sizes"][size]["G_E60d"] = {"verdict": vd, "ci": list(ci["I8"]),
                                            "median": med["I8"]}
            if "F32" in med:
                log("            and %.2fx the faithful fp32 arm (%.2f tok/s)"
                    % (med["I8"] / med["F32"], med["F32"]))
                out["sizes"][size]["I8_over_F32"] = med["I8"] / med["F32"]
        save(out)

    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = save(out)
    log("")
    log("wrote %s" % p)


if __name__ == "__main__":
    main()
