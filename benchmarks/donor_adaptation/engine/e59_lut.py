#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E59 -- does the fastest kernel this programme has survive contact with a trained model?

Brief: `docs/research/donor_adaptation/briefs/BRIEF_E59_DOES_THE_FAST_KERNEL_SURVIVE_A_TRAINED_MODEL.md`
Pre-registered and pushed before this file existed.

E58 addendum A left one engine-side lever -- `Rem` is 3.7%/2.0% of the token, so only a faster
weight kernel can move a ternary arm -- and the binary ships one: `--lutblk`, measured end-to-end
by E13 at x1.217/x1.358.  It quantises ACTIVATIONS to int8 and its fidelity on a trained artifact
has never been measured (E13 sec.8 item 2).

THE MEASUREMENT THIS RUNNER EXISTS TO GET RIGHT.  The obvious fidelity test -- score each arm
against HuggingFace -- is worthless here: both artifacts already read 3/160 and 10/160 on the
packed path, and a counter on its floor cannot show further damage
(feedback_gate_is_not_a_progress_meter).  The load-bearing quantity is the PAIRED one: greedy
agreement of each LUT arm against PACKED on the SAME artifact, which isolates what the kernel
changes and has a floor nowhere near saturated.  The HF column is reported and is not allowed to
carry the verdict.
"""
import json
import os
import random
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e44_interval as E44                                          # noqa: E402
import e51_math_errno as E51                                        # noqa: E402

OUTDIR = os.path.join(HERE, "results")
ENGINE = os.path.join(HERE, "donor_engine_e53.exe")
THREADS = 6
BENCH_N = 160
REPS = 9
SEED = 5901

ARTIFACTS = [("05b_tqh", os.path.join("D:", os.sep, "_ktmp", "e1", "qwen25-05b_tqh.bin")),
             ("15b_tqh", os.path.join("D:", os.sep, "_ktmp", "e1", "qwen25-15b_tqh.bin"))]

# E6's stored HuggingFace reference key per artifact, and E57's packed greedy count, so the
# HF column carries its own control instead of being a bare number.
HF_KEY = {"05b_tqh": "Qwen/Qwen2.5-0.5B", "15b_tqh": "Qwen/Qwen2.5-1.5B"}
E57_PACKED_HF = {"05b_tqh": 3, "15b_tqh": 10}

# quality arms: LUT is here ONLY as G-E59a(i)'s identity control, never as a candidate.
Q_ARMS = [("PACKED", []), ("LUT", ["--lut"]), ("LUTBLK", ["--lutblk"]),
          ("LUTBLK32", ["--lutblk", "--lut-group", "32"])]
# speed arms: plain --lut is already refuted end-to-end by E11/E13 and is not a candidate.
S_ARMS = [("PACKED", []), ("LUTBLK", ["--lutblk"]),
          ("LUTBLK32", ["--lutblk", "--lut-group", "32"])]


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


def iqr_pct(v):
    m = statistics.median(v)
    return 100.0 * (quantile(v, 0.75) - quantile(v, 0.25)) / m if m else float("nan")


def boot_ci(v, rng, n=2000):
    b = []
    for _ in range(n):
        b.append(statistics.median([v[rng.randrange(len(v))] for _ in v]))
    b.sort()
    return b[int(0.025 * n)], b[int(0.975 * n) - 1]


# ======================================================================================
def paired_agreement(a, b):
    """Token-level agreement between two arms' generations, over every scored position."""
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


def g_e59a(ident_same, ident_tot, diff_same, diff_tot):
    """Two-sided planted control: lut==lutblk must be identical AND lutblk!=packed must differ."""
    i_ok = ident_tot > 0 and ident_same == ident_tot
    d_ok = diff_tot > 0 and diff_same < diff_tot
    if i_ok and d_ok:
        return "FIRES"
    if not i_ok:
        return "DEAD -- lut and lutblk are NOT identical, contradicting E13 G-M0"
    return "DEAD -- lutblk and packed are identical; the instrument sees nothing"


def g_e59c(a, b):
    """A ratio is quotable when the two bootstrap intervals do not overlap."""
    return "SEPARATED" if (a[1] < b[0] or b[1] < a[0]) else "OVERLAPPING"


def selftest():
    n = [0]

    def chk(name, cond):
        n[0] += 1
        log("  %-60s %s" % (name, "fires" if cond else "*** DEAD ***"))
        if not cond:
            raise SystemExit("a decision function did not fire on a known-positive.  STOP.")

    chk("A-1 identical generations score 100%", paired_agreement([[1, 2, 3]], [[1, 2, 3]]) == (3, 3))
    chk("A-2 one changed token is counted", paired_agreement([[1, 2, 3]], [[1, 9, 3]]) == (2, 3))
    chk("A-3 G-E59a FIRES when lut==lutblk and lutblk!=packed",
        g_e59a(160, 160, 70, 160) == "FIRES")
    chk("A-4 and it is DEAD when lutblk==packed (an instrument seeing nothing)",
        g_e59a(160, 160, 160, 160).startswith("DEAD"))
    chk("A-5 and DEAD when lut!=lutblk, contradicting E13 G-M0",
        g_e59a(159, 160, 70, 160).startswith("DEAD"))
    chk("B-1 first divergence is located", first_divergence([1, 2, 3], [1, 9, 3]) == 1)
    chk("B-2 identical sequences report none", first_divergence([1, 2], [1, 2]) == -1)
    chk("C-1 disjoint intervals SEPARATE", g_e59c((10.0, 11.0), (12.0, 13.0)) == "SEPARATED")
    chk("C-2 overlapping intervals say so", g_e59c((10.0, 12.5), (12.0, 13.0)) == "OVERLAPPING")
    log("")
    log("  %d of %d fire." % (n[0], n[0]))


# ======================================================================================
def one_cell(weights, flags):
    sp = E44.Split()
    cmd = [ENGINE, "--weights", weights, "--threads", str(THREADS)] + list(flags) + \
          ["--bench", str(BENCH_N)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sout, serr = p.communicate()
    child = E44._process_times(int(p._handle))
    if p.returncode != 0:
        sys.stderr.write(serr.decode("utf-8", "replace")[-1200:])
        raise SystemExit("engine failed: " + " ".join(cmd))
    occ, foc = sp.close(child)
    m = E44.RE_BENCH.search(sout.decode("utf-8", "replace"))
    if not m:
        raise SystemExit("no BENCH line")
    cell = {"rate": float(m.group(3)), "occ": occ, "foreign": foc}
    cell.update(sp.record())
    return cell


def generate(weights, flags, n_new, n_prompts, tag):
    """Greedy generations for every prompt, as E57 did, with the prompt index carried."""
    runs = []
    for i in range(n_prompts):
        pfx = os.path.join(E51.TMP, "e59_%s_p%d" % (tag, i))
        ids = os.path.join(E51.TMP, "p%d.bin" % i)
        base = [ENGINE, "--weights", weights, "--threads", str(THREADS)] + list(flags)
        r = E51.parse_gen(E51.run(base + ["--generate", ids, str(n_new), pfx], "gen")[0])
        runs.append({"prompt": i, "ids": r["ids"], "decode_toks": r.get("decode_toks")})
        for f in (pfx + ".prefill.bin", pfx + ".ids.bin"):
            if os.path.exists(f):
                os.remove(f)
    return runs


def main():
    log("E59 -- does the fastest kernel this programme has survive a TRAINED model?")
    log("brief: BRIEF_E59_DOES_THE_FAST_KERNEL_SURVIVE_A_TRAINED_MODEL.md")
    log("")
    log("== self-test ==")
    selftest()
    if "--selftest" in sys.argv:
        return

    e6 = json.load(open(os.path.join(E51.E6_RES, "engine.json")))
    ref = json.load(open(os.path.join(E51.E6_RES, "ref.json")))
    n_new = e6["n_new"]
    n_p = len(e6["prompt_ids"])
    E51.write_prompt_ids(e6["prompt_ids"])
    os.makedirs(E51.TMP, exist_ok=True)
    rng = random.Random(SEED)
    out = {"brief": "BRIEF_E59_DOES_THE_FAST_KERNEL_SURVIVE_A_TRAINED_MODEL.md",
           "engine": os.path.basename(ENGINE), "bench_n": BENCH_N, "reps": REPS,
           "threads": THREADS, "n_new": n_new, "n_prompts": n_p, "arms": {}}

    for art, wp in ARTIFACTS:
        if not os.path.exists(wp):
            log("SKIP %s -- not on disk" % art)
            continue
        log("")
        log("=" * 88)
        log("== %s  --  %s ==" % (art, os.path.basename(wp)))
        log("=" * 88)

        # ---------------- fidelity ----------------
        log("")
        log("-- G-E59a / G-E59b : fidelity, %d prompts x %d new tokens, greedy --" % (n_p, n_new))
        gens, recs = {}, {}
        for name, flags in Q_ARMS:
            recs[name] = generate(wp, flags, n_new, n_p, "%s_%s" % (art, name))
            gens[name] = [r["ids"][-n_new:] for r in recs[name]]

        i_same, i_tot = paired_agreement(gens["LUT"], gens["LUTBLK"])
        d_same, d_tot = paired_agreement(gens["PACKED"], gens["LUTBLK"])
        va = g_e59a(i_same, i_tot, d_same, d_tot)
        log("   control (i)  LUT vs LUTBLK   %d/%d identical   (E13 G-M0 says bit-identical)"
            % (i_same, i_tot))
        log("   control (ii) PACKED vs LUTBLK %d/%d equal      (must DIFFER: int8 activations)"
            % (d_same, d_tot))
        log("   G-E59a : %s" % va)
        if va != "FIRES":
            raise SystemExit("the planted control did not fire.  No number below is read.  STOP.")
        log("")

        hf = HF_KEY[art]
        rr = {x["prompt"]: x for x in ref[hf]}
        log("   %-9s %14s %10s %8s   %s"
            % ("arm", "vs PACKED", "vs HF", "1st div", "note"))
        fid = {}
        for name, _ in Q_ARMS:
            s, t = paired_agreement(gens["PACKED"], gens[name])
            sc = E51.score_against_ref({name: {"weights": wp, "hf": hf, "runs": recs[name]}},
                                       ref, n_new)[name]
            divs = [first_divergence(g, rr[i]["ids"][-n_new:]) for i, g in enumerate(gens[name])]
            dp = [first_divergence(g, gens["PACKED"][i]) for i, g in enumerate(gens[name])]
            note = ""
            if name == "PACKED":
                note = ("E57 says %d -> %s" % (E57_PACKED_HF[art],
                        "REPRODUCED" if sc["matched"] == E57_PACKED_HF[art] else "*** DEVIATED ***"))
            fid[name] = {"vs_packed": [s, t], "vs_hf": [sc["matched"], sc["counted"]],
                         "first_div_vs_hf": divs, "first_div_vs_packed": dp}
            log("   %-9s %8d/%-5d %6d/%-4d %8s   %s"
                % (name, s, t, sc["matched"], sc["counted"], str(dp), note))
        log("")
        log("   The `vs HF` column is REPORTED, NOT DECIDING: packed already sits at %d/%d, a"
            % (E57_PACKED_HF[art], n_p * n_new))
        log("   floor no further damage can show through.  `vs PACKED` is the load-bearing one.")

        # ---------------- speed, interleaved ----------------
        log("")
        log("-- G-E59c : speed, %d reps, --bench %d, arms INTERLEAVED in one sweep --"
            % (REPS, BENCH_N))
        cells = dict((n, []) for n, _ in S_ARMS)
        for name, flags in S_ARMS:
            log("   warm-up %-9s DISCARDED: %.2f tok/s" % (name, one_cell(wp, flags)["rate"]))
        for r in range(REPS):
            for name, flags in S_ARMS:
                cells[name].append(one_cell(wp, flags))
        log("")
        log("   %-9s %8s %8s %18s %7s %9s %8s"
            % ("arm", "median", "p25", "boot 95% CI", "IQR", "foreign", "vs PACKED"))
        stats = {}
        for name, _ in S_ARMS:
            v = [c["rate"] for c in cells[name]]
            ci = boot_ci(v, rng)
            stats[name] = {"median": statistics.median(v), "p25": quantile(v, 0.25), "ci": ci,
                           "iqr_pct": iqr_pct(v), "rates": v,
                           "foreign": statistics.median([c["foreign"] for c in cells[name]]),
                           "clock": statistics.median([c["clock_pct"] for c in cells[name]])}
        base = stats["PACKED"]
        for name, _ in S_ARMS:
            st = stats[name]
            log("   %-9s %8.2f %8.2f  [%7.2f,%7.2f] %6.2f%% %8.2f%% %8.3f"
                % (name, st["median"], st["p25"], st["ci"][0], st["ci"][1], st["iqr_pct"],
                   st["foreign"], st["median"] / base["median"]))
        log("")
        for name, _ in S_ARMS:
            if name == "PACKED":
                continue
            v = g_e59c(stats[name]["ci"], base["ci"])
            log("   %s / PACKED = %.3f   intervals %s"
                % (name, stats[name]["median"] / base["median"], v))
            stats[name]["separation"] = v

        log("")
        log("-- G-E59d : the joint law (E13 sec.7) --")
        for name, _ in S_ARMS:
            if name == "PACKED":
                continue
            s, t = fid[name]["vs_packed"]
            log("   %-9s x%.3f on speed, and it changes %d of %d tokens (%.1f%% agreement)."
                % (name, stats[name]["median"] / base["median"], t - s, t, 100.0 * s / t))
        log("   Neither ratio may be quoted without the token count beside it.")

        out["arms"][art] = {"weights": wp, "fidelity": fid, "g_e59a": va, "speed": stats}

    log("")
    log("== nothing is promoted.  donor_engine.c untouched; packed stays the default. ==")
    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    path = os.path.join(OUTDIR, "e59_lut.json")
    json.dump(out, open(path, "w"), indent=1)
    log("wrote %s" % path)


if __name__ == "__main__":
    main()
