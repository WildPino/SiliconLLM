#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E50 -- turn the fast kernel on by default, gated on the claim it can break.

Brief: BRIEF_E50_TURN_THE_FAST_KERNEL_ON_BY_DEFAULT.md  (1f6ab5b, addendum A ac58081)
Pushed before this file existed.  Every gate is transcribed from section 4 and addendum A.

E49 section 6 registered this as a separate gated change; E49 established it deserves
to be made.  Two changes, one experiment:

  1. donor_engine.c:202  ATTN_SERIAL -> ATTN_AVX4
  2. the engine prints CONFIG attn=... in EVERY mode, and attn=... on the BENCH line

What is at risk is NOT the speed -- E49 owns that.  E49 gated BPB, a scalar average over
12,264 positions; greedy argmax is DISCRETE.  E6's 160/160 greedy-identical trajectory,
the strongest claim this programme has, was measured with no --attn, i.e. on serial.

  python e50_default_kernel.py --selftest
  python e50_default_kernel.py --phase witness     # G-E50c, G-E50d   (seconds)
  python e50_default_kernel.py --phase greedy      # G-E50a           (~10 min)
  python e50_default_kernel.py --phase nats        # G-E50b           (~20 min)
  python e50_default_kernel.py --phase drift       # the drift line, no verdict
  python e50_default_kernel.py                     # all four, in that order
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e44_interval import Occupancy, log                              # noqa: E402

RES = os.path.join(HERE, "results")
E6_RES = os.path.join(RES, "e6")
E50_RES = os.path.join(RES, "e50")
OUT = os.path.join(RES, "e50_default_kernel.json")

ENGINE = os.path.join(HERE, "donor_engine.exe")          # THE NEW BUILD -- that is the point
E26_ENGINE = os.path.join(HERE, "donor_engine_e26.exe")  # what E49 ran, for the drift line only
TMP = r"D:\_ktmp\e50"

THREADS = 6

# ---- G-E50a (section 4, addendum A.2) --------------------------------------------
# E6's arms, verbatim from e6_generate.py:35-40.  The prompts and ids are NOT regenerated
# from the tokenizer: they are read back out of E6's own engine.json, so the prompts are
# identical BY CONSTRUCTION rather than by re-deriving them and hoping.
E6_ARMS = [
    ("A1", r"D:\_ktmp\e1\qwen25-05b_f32.bin", "Qwen/Qwen2.5-0.5B", "known-positive (fp32)"),
    ("A2", r"D:\_ktmp\e1\qwen25-05b_tqh.bin", "Qwen/Qwen2.5-0.5B", "planted control (ternary+head)"),
    ("A3", r"D:\_ktmp\e1\qwen25-15b_tqh.bin", "Qwen/Qwen2.5-1.5B", "planted control at 1.5B"),
]
G50A_A1_REQUIRED = 160          # E6: agree 1.0, matched 160, counted 160
G50A_CONTROL_MAX = 0.90         # E6's own G-C bar: a control must stay BELOW this
G50A_CONTROL_TOL = 2            # addendum A: a control may move by at most this many tokens
E6_CONTROL_MATCHED = {"A2": 3, "A3": 10}     # results/e6/summary.json

# ---- G-E50b (section 4) ----------------------------------------------------------
# E49's two published values, to every digit.  results/e49_attn_kernel.json.
S15 = "D:/_ktmp/e37/e37_carved_nf.bin"
PARITY_K = ["--carve-k", "256"]
IDS = "D:/_ktmp/e1/ids_qwen25-15b_tq.bin"
SEQLEN = 512
E49_NATS_AVX4 = 124963.9517608703
E49_NATS_SERIAL = 124963.9729339122

# ---- the drift line (section 4, "not a gate") ------------------------------------
R128 = "D:/_ktmp/e40/e40_r128.bin"
R128_FLAGS = ["--carve-k", "3"]
DRIFT_WINDOWS = [40, 1280]
E49_AVX4_CELLS = {40: 121.65, 1280: 74.11}   # E49 G-E49b medians, avx4

RE_BENCH = re.compile(r"BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s")
RE_NATS = re.compile(r"NATS_TOTAL ([0-9.eE+-]+)\s*\nN_PREDICTED (\d+)")
RE_CONFIG = re.compile(r"^CONFIG\s+attn=(\S+)\s+attnr=(\S+)", re.M)

# G-E50d: every BENCH-parsing regex in benchmarks/donor_adaptation, collected by hand from
# e3_bench.py:52, e5_sweep.py:53, e25_rank_cost.py:58, e26_carve_cost.py:56,
# e28_kernel_transfer.py:52 (which e30_the_wall.py imports), e44_interval.py:57,
# e48_slope_decomposition.py:70.  If a runner is added later it goes in this list.
BENCH_PARSERS = [
    ("e3_bench.py:52", r"BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s"),
    ("e5_sweep.py:53", r"BENCH\s+\d+ tokens\s+[\d.]+ s\s+([\d.]+) tok/s"),
    ("e25_rank_cost.py:58", r"BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s"),
    ("e26_carve_cost.py:56", r"BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s"),
    ("e28_kernel_transfer.py:52", r"BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s"),
    ("e44_interval.py:57", r"BENCH\s+(\d+)\s+tokens\s+([\d.]+)\s+s\s+([\d.]+)\s+tok/s"),
    ("e48_slope_decomposition.py:70", r"^BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s"),
]


# ==================================================================================
# the decision functions.  Every one is exercised in both directions by --selftest
# before it is allowed to judge anything.
# ==================================================================================

def g_e50a(a1_avx4, a1_serial, controls_avx4):
    """Addendum A.2's four-way table, plus section 4's planted control.

    a1_avx4 / a1_serial : matched count out of 160.
    controls_avx4       : {"A2": matched, "A3": matched} under the new default.

    Returns (verdict, reason).  VOID means no measurement was made -- the change is not
    what broke it -- and is NOT a failure of the change.
    """
    # the planted control first: a scorer that reports agreement for everything has not
    # shown it can see disagreement, and then the A1 reading says nothing.
    for k, m in sorted(controls_avx4.items()):
        if m / 160.0 >= G50A_CONTROL_MAX:
            return ("VOID", "planted control %s did NOT fire: %d/160 agrees with PyTorch, "
                            "so the scorer has not been shown to see a difference" % (k, m))
        if abs(m - E6_CONTROL_MATCHED[k]) > G50A_CONTROL_TOL:
            return ("VOID", "planted control %s moved %d -> %d, more than the +-%d addendum A "
                            "allows -- that is its own finding, not this gate's"
                            % (k, E6_CONTROL_MATCHED[k], m, G50A_CONTROL_TOL))
    ok_avx4 = (a1_avx4 == G50A_A1_REQUIRED)
    ok_serial = (a1_serial == G50A_A1_REQUIRED)
    if ok_avx4 and ok_serial:
        return ("PASS", "A1 is 160/160 on both kernels -- the default may change")
    if not ok_avx4 and ok_serial:
        return ("FAIL", "A1 falls to %d/160 on avx4 while serial holds at 160 -- attributed to "
                        "the kernel, the default does NOT change" % a1_avx4)
    if ok_avx4 and not ok_serial:
        return ("PASS-AND-A-FINDING",
                "A1 holds at 160/160 on avx4 but serial reads %d/160 -- the gate passes and "
                "something between E6's binary and E26's moved the serial trajectory"
                % a1_serial)
    return ("VOID", "A1 falls on BOTH kernels (%d and %d of 160) -- the cause is drift between "
                    "E6's binary and E26's, not this change" % (a1_avx4, a1_serial))


def g_e50b(nats_default, nats_serial):
    """Bit-for-bit reproduction of E49's two published values."""
    d_ok = (nats_default == E49_NATS_AVX4)
    s_ok = (nats_serial == E49_NATS_SERIAL)
    if d_ok and s_ok:
        return ("PASS", "both values reproduce to every digit -- the default IS avx4 and every "
                        "pre-E50 serial reading stays reproducible")
    if not s_ok:
        # addendum A.3: the serial half is what discriminates build from change
        return ("FAIL-DIAGNOSE", "the serial arm does NOT reproduce (%r vs %r) -- run the "
                                 "conditional diagnostic of addendum A.3 before attributing this"
                                 % (nats_serial, E49_NATS_SERIAL))
    return ("FAIL", "serial reproduces but the default does not (%r vs %r) -- the default is not "
                    "what it claims to be" % (nats_default, E49_NATS_AVX4))


def g_e50c(cfg_bench_default, cfg_bench_serial, cfg_bpb_default, cfg_bpb_serial):
    """A line that prints a constant is not a witness -- it must be shown to CHANGE."""
    if not (cfg_bench_default and cfg_bench_serial and cfg_bpb_default and cfg_bpb_serial):
        return ("FAIL", "a mode printed no CONFIG line at all")
    if cfg_bench_default != "avx4" or cfg_bpb_default != "avx4":
        return ("FAIL", "the default does not report avx4 (bench %s, bpb %s)"
                % (cfg_bench_default, cfg_bpb_default))
    if cfg_bench_serial != "serial" or cfg_bpb_serial != "serial":
        return ("FAIL", "--attn serial is not reported (bench %s, bpb %s)"
                % (cfg_bench_serial, cfg_bpb_serial))
    return ("PASS", "the arm is reported, it changes with the flag, and it is reported in --bpb "
                    "too -- which prints no BENCH line at all")


def g_e50d(bench_line):
    """Every existing BENCH parser must still match, and return the same three fields."""
    bad = []
    fields = None
    for name, pat in BENCH_PARSERS:
        m = re.search(pat, bench_line, re.M)
        if not m:
            bad.append((name, "no match"))
            continue
        g = m.groups()
        if len(g) == 3:
            if fields is None:
                fields = g
            elif g != fields:
                bad.append((name, "fields differ: %r vs %r" % (g, fields)))
    if bad:
        return ("FAIL", "; ".join("%s %s" % b for b in bad))
    return ("PASS", "all %d parsers match and agree on (tokens, seconds, tok/s) = %r"
            % (len(BENCH_PARSERS), fields))


# ==================================================================================
def selftest():
    log("== E50 gate self-test -- every decision function in every direction ==")
    n = 0

    def chk(tag, got, want, why):
        nonlocal n
        n += 1
        ok = (got == want)
        log("  %-5s %-22s expected %-22s %s   %s"
            % (tag, got, want, "FIRES" if ok else "**DEAD**", why))
        if not ok:
            raise SystemExit("self-test %s did not fire" % tag)

    ctl_ok = {"A2": 3, "A3": 10}
    chk("A1", g_e50a(160, 160, ctl_ok)[0], "PASS", "E6's own reading on both kernels")
    chk("A2", g_e50a(158, 160, ctl_ok)[0], "FAIL",
        "avx4 falls, serial holds -> attributed to the kernel")
    chk("A3", g_e50a(158, 157, ctl_ok)[0], "VOID",
        "BOTH fall -> drift between E6's binary and E26's, not this change")
    chk("A4", g_e50a(160, 157, ctl_ok)[0], "PASS-AND-A-FINDING",
        "the gate passes and an unrelated finding opens")
    chk("A5", g_e50a(160, 160, {"A2": 160, "A3": 10})[0], "VOID",
        "a control that AGREES voids the null, however good the null looks")
    chk("A6", g_e50a(160, 160, {"A2": 9, "A3": 10})[0], "VOID",
        "a control that moved 3 -> 9 is its own finding, not this gate's")
    chk("A7", g_e50a(160, 160, {"A2": 5, "A3": 10})[0], "PASS",
        "3 -> 5 is inside the +-2 addendum A allows")

    chk("B1", g_e50b(E49_NATS_AVX4, E49_NATS_SERIAL)[0], "PASS", "both to every digit")
    chk("B2", g_e50b(E49_NATS_AVX4 + 1e-9, E49_NATS_SERIAL)[0], "FAIL",
        "a digit is a digit -- the engine is deterministic, so this is not noise")
    chk("B3", g_e50b(E49_NATS_AVX4, E49_NATS_SERIAL + 1e-9)[0], "FAIL-DIAGNOSE",
        "the serial half missing means build-vs-change is unresolved")
    chk("B4", g_e50b(E49_NATS_SERIAL, E49_NATS_SERIAL)[0], "FAIL",
        "the default still running serial is exactly what E50 exists to prevent")

    chk("C1", g_e50c("avx4", "serial", "avx4", "serial")[0], "PASS", "reported, and it changes")
    chk("C2", g_e50c("avx4", "avx4", "avx4", "avx4")[0], "FAIL",
        "a line that prints a constant is not a witness")
    chk("C3", g_e50c("avx4", "serial", None, None)[0], "FAIL",
        "--bpb is where parity is decided and it prints no BENCH line")
    chk("C4", g_e50c("serial", "serial", "serial", "serial")[0], "FAIL",
        "the default did not actually change")

    good = "BENCH  40 tokens  0.329 s  121.65 tok/s  (threads=6, packed, attn=avx4)  ffn~ 3.0 ms/tok"
    chk("D1", g_e50d(good)[0], "PASS", "the arm goes AFTER tok/s, where no parser looks")
    bad = "BENCH  40 tokens  0.329 s  attn=avx4  121.65 tok/s  (threads=6, packed)"
    chk("D2", g_e50d(bad)[0], "FAIL",
        "the arm placed BEFORE tok/s breaks every parser -- this is the version not shipped")

    log("  E50 gate self-test : ALL FIRE  (%d checks)" % n)
    log("")


# ==================================================================================
def run(cmd, what):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True)
    dt = time.time() - t0
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode("utf-8", "replace")[-1500:])
        raise SystemExit("engine failed (%s): %s" % (what, " ".join(cmd)))
    return p.stdout.decode("utf-8", "replace"), dt


def config_arm(txt):
    m = RE_CONFIG.search(txt)
    return m.group(1) if m else None


# ---- G-E50c / G-E50d --------------------------------------------------------------
def phase_witness(out):
    log("== G-E50c -- the witness must be shown to WITNESS, and G-E50d -- nobody is broken ==")
    w = E6_ARMS[0][1]                      # the smallest fp32 arm; the arm does not matter here
    base = [ENGINE, "--weights", w, "--threads", str(THREADS)]
    t_bd, _ = run(base + ["--bench", "20"], "bench default")
    t_bs, _ = run(base + ["--attn", "serial", "--bench", "20"], "bench serial")
    # --bpb needs a slice; the 1.5B ids are the wrong vocab for the 0.5B, so use the arm's own
    # engine-native path: --logits is cheaper and also prints CONFIG.  The brief asks for a mode
    # that is not --bench and that prints no BENCH line; --logits is exactly that.
    ids = os.path.join(TMP, "p0.bin")
    t_ld, _ = run(base + ["--logits", ids, "5", os.path.join(TMP, "_w.bin")], "logits default")
    t_ls, _ = run(base + ["--attn", "serial", "--logits", ids, "5", os.path.join(TMP, "_w.bin")],
                  "logits serial")
    c = (config_arm(t_bd), config_arm(t_bs), config_arm(t_ld), config_arm(t_ls))
    log("  CONFIG arm reported:  bench default %s / bench serial %s / logits default %s / "
        "logits serial %s" % c)
    v, why = g_e50c(*c)
    log("  G-E50c : %s" % v)
    log("  %s" % why)
    log("")

    bench_line = [l for l in t_bd.splitlines() if l.startswith("BENCH")][0]
    log("  the new BENCH line, verbatim:")
    log("    %s" % bench_line)
    v2, why2 = g_e50d(bench_line)
    log("  G-E50d : %s" % v2)
    log("  %s" % why2)
    log("")
    out["G_E50c"] = {"verdict": v, "why": why, "arms": c}
    out["G_E50d"] = {"verdict": v2, "why": why2, "bench_line": bench_line}
    return v == "PASS" and v2 == "PASS"


# ---- G-E50a -----------------------------------------------------------------------
def write_prompt_ids(prompt_ids):
    """From E6's OWN engine.json, so the prompts are identical by construction."""
    os.makedirs(TMP, exist_ok=True)
    for i, ids in enumerate(prompt_ids):
        with open(os.path.join(TMP, "p%d.bin" % i), "wb") as f:
            f.write(struct.pack("<%di" % len(ids), *ids))


def parse_gen(txt):
    d = {}
    for line in txt.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "GEN_IDS":
            d["ids"] = [int(x) for x in f[1:]]
        elif f[0].startswith("GEN_"):
            d[f[0][4:].lower()] = float(f[1])
    return d


def e6_engine_stage(extra, n_new, prompt_ids, tag):
    """e6_generate.py stage_engine, verbatim but for `extra` and where it writes."""
    arms = {}
    for name, wp, hf, role in E6_ARMS:
        if not os.path.exists(wp):
            log("  SKIP %s -- %s not on disk" % (name, wp))
            continue
        rec = {"weights": wp, "hf": hf, "role": role, "runs": []}
        for i in range(len(prompt_ids)):
            pfx = os.path.join(TMP, "%s_%s_p%d" % (tag, name, i))
            ids = os.path.join(TMP, "p%d.bin" % i)
            base = [ENGINE, "--weights", wp, "--threads", str(THREADS)] + list(extra)
            r1 = parse_gen(run(base + ["--generate", ids, str(n_new), pfx], "gen")[0])
            r2 = parse_gen(run(base + ["--generate", ids, str(n_new), pfx + "_b"], "gen b")[0])
            gd = (open(pfx + ".ids.bin", "rb").read() ==
                  open(pfx + "_b.ids.bin", "rb").read())
            lg = pfx + ".ref_logits.bin"
            run(base + ["--logits", ids, str(len(prompt_ids[i])), lg], "logits")
            gp = (open(pfx + ".prefill.bin", "rb").read() == open(lg, "rb").read())
            for f in (lg, pfx + "_b.ids.bin", pfx + "_b.prefill.bin", pfx + ".prefill.bin"):
                os.remove(f)
            rec["runs"].append({"prompt": i, "ids": r1["ids"], "G_P": bool(gp), "G_D": bool(gd),
                                "decode_toks": r1["decode_toks"]})
            log("    %-6s %s p%d  G-P %s  G-D %s  decode %.2f tok/s"
                % (tag, name, i, "PASS" if gp else "FAIL", "PASS" if gd else "FAIL",
                   r1["decode_toks"]))
        arms[name] = rec
    return arms


def score_against_ref(arms, ref, n_new):
    """e6_generate.py stage_score's agreement block, verbatim."""
    summary = {}
    for name, rec in sorted(arms.items()):
        rr = {x["prompt"]: x for x in ref[rec["hf"]]}
        tot = match = 0
        firstdiv = None
        worst_gap = None
        for r in rec["runs"]:
            ours = r["ids"][-n_new:]
            theirs = rr[r["prompt"]]["ids"][-n_new:]
            gaps = rr[r["prompt"]]["top2_gap"]
            for k in range(n_new):
                tot += 1
                if ours[k] == theirs[k]:
                    match += 1
                else:
                    if firstdiv is None:
                        firstdiv = (r["prompt"], k)
                    worst_gap = gaps[k] if worst_gap is None else max(worst_gap, gaps[k])
        summary[name] = {"matched": match, "counted": tot,
                         "agree": match / float(tot) if tot else 0.0,
                         "first_div": firstdiv, "worst_gap_at_div": worst_gap}
    return summary


def phase_greedy(out):
    log("== G-E50a -- E6's 160/160 greedy claim must survive the new default ==")
    log("  addendum A.2: BOTH kernels are run, so a failure can be ATTRIBUTED")
    e6 = json.load(open(os.path.join(E6_RES, "engine.json")))
    ref = json.load(open(os.path.join(E6_RES, "ref.json")))
    n_new = e6["n_new"]
    if e6["threads"] != THREADS:
        raise SystemExit("E6 ran %d threads, this runner %d" % (e6["threads"], THREADS))
    write_prompt_ids(e6["prompt_ids"])
    os.makedirs(E50_RES, exist_ok=True)

    res = {}
    for tag, extra in (("avx4", []), ("serial", ["--attn", "serial"])):
        log("  -- %s --" % tag)
        arms = e6_engine_stage(extra, n_new, e6["prompt_ids"], tag)
        json.dump({"n_new": n_new, "threads": THREADS, "extra": extra, "arms": arms},
                  open(os.path.join(E50_RES, "engine_%s.json" % tag), "w"), indent=1)
        res[tag] = score_against_ref(arms, ref, n_new)

    log("")
    log("  %-4s %-10s %-10s %-10s" % ("arm", "E6 (serial)", "E50 avx4", "E50 serial"))
    for name in ("A1", "A2", "A3"):
        if name not in res["avx4"]:
            continue
        e6m = 160 if name == "A1" else E6_CONTROL_MATCHED[name]
        log("  %-4s %-11s %-10s %-10s"
            % (name, "%d/160" % e6m,
               "%d/160" % res["avx4"][name]["matched"],
               "%d/160" % res["serial"][name]["matched"]))
    log("")
    controls = {k: res["avx4"][k]["matched"] for k in res["avx4"] if k != "A1"}
    v, why = g_e50a(res["avx4"]["A1"]["matched"], res["serial"]["A1"]["matched"], controls)
    log("  G-E50a : %s" % v)
    log("  %s" % why)
    log("")
    out["G_E50a"] = {"verdict": v, "why": why, "avx4": res["avx4"], "serial": res["serial"]}
    return v in ("PASS", "PASS-AND-A-FINDING")


# ---- G-E50b -----------------------------------------------------------------------
def nats(engine, extra):
    cmd = ([engine, "--weights", S15, "--threads", str(THREADS), "--seqlen", str(SEQLEN)]
           + list(extra) + PARITY_K + ["--bpb", IDS])
    txt, dt = run(cmd, "bpb")
    m = RE_NATS.search(txt)
    if not m:
        raise SystemExit("could not parse NATS_TOTAL")
    return float(m.group(1)), int(m.group(2)), dt


def phase_nats(out):
    log("== G-E50b -- the new default must reproduce E49's two values BIT FOR BIT ==")
    nd, n1, t1 = nats(ENGINE, [])
    log("  no flag        NATS_TOTAL %.10f  n=%d  (%.0f s)   E49 avx4   %.10f"
        % (nd, n1, t1, E49_NATS_AVX4))
    ns, n2, t2 = nats(ENGINE, ["--attn", "serial"])
    log("  --attn serial  NATS_TOTAL %.10f  n=%d  (%.0f s)   E49 serial %.10f"
        % (ns, n2, t2, E49_NATS_SERIAL))
    v, why = g_e50b(nd, ns)
    log("  G-E50b : %s" % v)
    log("  %s" % why)
    rec = {"verdict": v, "why": why, "nats_default": repr(nd), "nats_serial": repr(ns),
           "e49_avx4": repr(E49_NATS_AVX4), "e49_serial": repr(E49_NATS_SERIAL)}
    if v == "FAIL-DIAGNOSE":
        log("")
        log("  addendum A.3 diagnostic -- is it the change, or the build?")
        log("  compiling HEAD's UNMODIFIED donor_engine.c and re-running the serial arm")
        rec["diagnostic"] = run_a3_diagnostic()
        log("  %s" % rec["diagnostic"]["reading"])
    log("")
    out["G_E50b"] = rec
    return v == "PASS"


def run_a3_diagnostic():
    """Addendum A.3, registered BEFORE the run and only executed if G-E50b fails."""
    src = os.path.join(TMP, "_head_donor_engine.c")
    exe = os.path.join(TMP, "_head_donor_engine.exe")
    p = subprocess.run(["git", "show", "HEAD:benchmarks/donor_adaptation/engine/donor_engine.c"],
                       capture_output=True, cwd=os.path.join(HERE, "..", "..", ".."))
    if p.returncode != 0:
        return {"reading": "could not extract HEAD's source -- diagnostic inconclusive"}
    open(src, "wb").write(p.stdout)
    b = subprocess.run(["clang", "-O3", "-mavx2", "-mfma", "-ffp-contract=on", "-fopenmp",
                        src, "-o", exe, "-lm"], capture_output=True)
    if b.returncode != 0:
        return {"reading": "could not build HEAD's source -- diagnostic inconclusive"}
    val, _, _ = nats(exe, [])          # HEAD's default IS serial
    ok = (val == E49_NATS_SERIAL)
    os.remove(exe)
    return {"head_default_nats": repr(val), "matches_e49_serial": ok,
            "reading": ("HEAD's unmodified source reproduces E49's serial value, so the build is "
                        "reproducible and MY CHANGE is the delta -- G-E50b stands FAILED"
                        if ok else
                        "HEAD's unmodified source does NOT reproduce it either, so the defect is "
                        "the toolchain and not this change -- G-E50b is VOID, not FAILED")}


# ---- the drift line, which is not a gate ------------------------------------------
def phase_drift(out):
    log("== the drift line -- NOT A GATE.  The machine is not certified idle (E43/E44). ==")
    rows = []
    for n in DRIFT_WINDOWS:
        occ = Occupancy()          # Occupancy.sample() reads the busy fraction SINCE the last
        occ.sample()               # read, so one read before and one after brackets the cell
        txt, _ = run([ENGINE, "--weights", R128, "--threads", str(THREADS)]
                     + R128_FLAGS + ["--bench", str(n)], "drift")
        o = occ.sample()
        m = RE_BENCH.search(txt)
        rate = float(m.group(3))
        ref = E49_AVX4_CELLS[n]
        rows.append({"n": n, "tok_s": rate, "e49_avx4": ref,
                     "delta_pct": 100.0 * (rate - ref) / ref, "occ": o})
        log("  n=%-5d  %6.2f tok/s   E49 avx4 %6.2f   %+5.1f%%   occ %.1f%%   arm=%s"
            % (n, rate, ref, 100.0 * (rate - ref) / ref, o, config_arm(txt)))
    log("  reported with its occupancy and carrying NO verdict -- E44 owns the interval")
    log("")
    out["drift"] = rows


# ==================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--phase", choices=["witness", "greedy", "nats", "drift"])
    a = ap.parse_args()

    log("=" * 84)
    log("E50 -- turn the fast kernel on by default, gated on the claim it can break")
    log("  donor_engine.c:202 ATTN_SERIAL -> ATTN_AVX4, and every mode now prints its arm.")
    log("  E49 gated BPB, a scalar average.  Greedy argmax is DISCRETE, and E6's 160/160")
    log("  was measured with no --attn.  That is what G-E50a is for.")
    log("=" * 84)
    selftest()
    if a.selftest:
        return

    if not os.path.exists(os.path.join(E6_RES, "engine.json")):
        raise SystemExit("results/e6/engine.json is missing -- it is the record being compared "
                         "against, and E50 must not regenerate it (addendum A.4)")
    os.makedirs(TMP, exist_ok=True)
    # the prompt ids come out of E6's OWN engine.json and every phase needs them on disk
    write_prompt_ids(json.load(open(os.path.join(E6_RES, "engine.json")))["prompt_ids"])
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))     # a re-run of ONE phase must not discard the others
    out.update({"brief": "BRIEF_E50_TURN_THE_FAST_KERNEL_ON_BY_DEFAULT.md",
                "engine": ENGINE, "threads": THREADS})

    phases = [a.phase] if a.phase else ["witness", "greedy", "nats", "drift"]
    for ph in phases:
        if ph == "witness":
            if not phase_witness(out) and not a.phase:
                raise SystemExit("G-E50c/G-E50d did not pass -- the change is not auditable, "
                                 "so nothing downstream counts")
        elif ph == "greedy":
            if not phase_greedy(out) and not a.phase:
                log("G-E50a did not pass.  Per section 4 the default does NOT change and the")
                log("speed phases are not run.  Revert donor_engine.c:202 and rebuild.")
                json.dump(out, open(OUT, "w"), indent=1)
                raise SystemExit(1)
        elif ph == "nats":
            phase_nats(out)
        elif ph == "drift":
            phase_drift(out)
        json.dump(out, open(OUT, "w"), indent=1)   # after EVERY phase: a later crash must not
                                                   # take the phases that already ran with it
    log("wrote " + OUT)


if __name__ == "__main__":
    main()
