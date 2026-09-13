#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E61 -- the only kernel that works is the only kernel still running a single accumulator.

Brief: `docs/research/donor_adaptation/briefs/BRIEF_E61_THE_CHAIN_ON_THE_BYTE_THAT_WORKS.md`
Pre-registered and pushed (ab2a537) before this file existed.

E8 wired --mvacc into matvec_sel's packed and fp32 branches and left the third alone, for a
reason it wrote down: in 2026-09-07 that branch was dead code.  E60 made it the measured path --
it is the int8 rung, the only artefact this programme has that is both above the 50 tok/s bar
and faithful -- and the assembly still read one unbroken ymm0->ymm1->ymm0 chain, 4 FMAs per 32
weight bytes (.LBB17_77).  At 5-cycle FMA latency that caps the loop at 36.4 GB/s and E60
measured 31.7-33.1: 87-91% of its own latency wall, against fp32 at 6.5% of its and packed at
30% of its.  So SPEED_LEDGER 59.2's middle denominator may be a kernel, not a box.

WHAT THIS RUNNER IS CAREFUL ABOUT.

1. TWO planted controls, in the same sweep, on the same binary, interleaved with the treatment:
   packed --mvacc 4/1 must FIRE (E8 G-Z3 read 1.2301) and fp32 must stay NULL (E8 G-Z5 read
   1.0029).  A dead instrument voids every null under it; a live fp32 arm voids the mechanism.
2. The fidelity metric is the one E60 registered as the replacement for its own defective gate:
   per-position top-1 under TEACHER FORCING over all 12264 positions, not 5x32 free-running
   tokens.  Its planted control runs FIRST, because ~100% agreement is a CEILING and a saturated
   counter cannot show damage in either direction (E59's law, both halves).
3. --mvacc 1 is a byte-for-byte copy of the pre-E61 loop, so G-E61a is an sha256 gate against
   donor_engine_e60.exe and every E60 number stays reproducible on this binary.
4. Results are written after every phase (E44's rule), and the runner prints numbers and gate
   verdicts but decides no conclusion of its own.
"""
import hashlib
import json
import math
import os
import random
import statistics
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e44_interval as E44                                          # noqa: E402
import e51_math_errno as E51                                        # noqa: E402

OUTDIR = os.path.join(HERE, "results")
ENGINE = os.path.join(HERE, "donor_engine_e61.exe")
FROZEN = os.path.join(HERE, "donor_engine_e60.exe")     # E60's binary, for G-E61a
THREADS = 6
BENCH_N = 160
REPS = 9
SEED = 6101
LOGIT_N = 8                       # logit vectors for G-E61a / G-E61d

E1 = os.path.join("D:", os.sep, "_ktmp", "e1")
E60D = os.path.join("D:", os.sep, "_ktmp", "e60")

BPB_IDS = os.path.join(E1, "ids_qwen25-05b_tqh.bin")
BPB_SEQLEN = 512
BPB_BYTES = 51870
BPB_NPRED = 12264
CHANCE_BPB = 4.069819
OCC_BAR = 4.39

# E60's measured BPB on this slice, for the cross-check that the fidelity pass is the same pass.
E60_BPB = {("05b", "F32"): 0.871810465558026, ("05b", "PACKED"): 4.531233733626962,
           ("05b", "I8"): 0.8718768840274047, ("15b", "I8"): 0.7688588381536873}
# E60's measured rates at mvacc=1-in-effect, for the sanity line (NOT a gate: E60 ran a
# different binary and every absolute tok/s carries +-5% between sweeps; the ratios do not).
E60_RATE = {("05b", "F32"): 19.08, ("05b", "PACKED"): 86.23, ("05b", "I8"): 63.87,
            ("05b", "T1"): 61.89, ("15b", "I8"): 21.39}

WEIGHTS = {
    ("05b", "F32"):    os.path.join(E1, "qwen25-05b_f32.bin"),
    ("05b", "PACKED"): os.path.join(E1, "qwen25-05b_tqh.bin"),
    ("05b", "I8"):     os.path.join(E60D, "qwen25-05b_i8h.bin"),
    ("05b", "T1"):     os.path.join(E60D, "qwen25-05b_t1h.bin"),
    ("15b", "I8"):     os.path.join(E60D, "qwen25-15b_i8h.bin"),
}
# Which arms are timed at which size, and in which role.  Registered in the brief 6:
#   packed = the planted POSITIVE, fp32 = the planted NULL, I8 = the headline, T1 = diagnostic.
SPEED = {"05b": ["PACKED", "F32", "I8", "T1"], "15b": ["I8"]}
MVACCS = [1, 4]


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def rel_l2(pa, pb):
    """||a-b|| / ||a||, streamed, so a 4.9 MB logit dump never needs numpy."""
    num = den = 0.0
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        while True:
            ba, bb = fa.read(1 << 20), fb.read(1 << 20)
            if len(ba) != len(bb):
                raise SystemExit("logit dumps differ in length")
            if not ba:
                break
            va = struct.unpack("<%df" % (len(ba) // 4), ba)
            vb = struct.unpack("<%df" % (len(bb) // 4), bb)
            for x, y in zip(va, vb):
                num += (x - y) * (x - y)
                den += x * x
    return math.sqrt(num) / math.sqrt(den if den > 0 else 1.0)


def top1_agree(pa, pb):
    """(same, total) over int32 LE top-1 ids."""
    a = open(pa, "rb").read()
    b = open(pb, "rb").read()
    if len(a) != len(b):
        raise SystemExit("top-1 dumps differ in length: %d vs %d" % (len(a), len(b)))
    n = len(a) // 4
    va = struct.unpack("<%di" % n, a)
    vb = struct.unpack("<%di" % n, b)
    return sum(1 for x, y in zip(va, vb) if x == y), n


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


# ============================================================ the decision functions
# Thresholds are the brief's, section 6.  Nothing here is computed from the data.
def g_e61a(pairs):
    """PLANTED: --mvacc 1 must be byte-for-byte the pre-E61 loop.  pairs = [(name, ok), ...]."""
    if not pairs:
        return "DEAD -- nothing compared"
    bad = [n for n, ok in pairs if not ok]
    if bad:
        return "DEAD -- e61 --mvacc 1 differs from e60 on " + ",".join(bad)
    return "FIRES"


def g_e61b(ratio):
    """PLANTED POSITIVE: packed m4/m1 >= 1.10, or the sweep cannot see a chain effect at all."""
    if ratio is None:
        return "DEAD -- not measured"
    return "FIRES" if ratio >= 1.10 else "DEAD -- every null below this is void"


def g_e61c(ratio):
    """PLANTED NULL: fp32 must buy nothing.  >=1.10 refutes the mechanism outright."""
    if ratio is None:
        return "DEAD -- not measured"
    if ratio >= 1.10:
        return "MECHANISM-REFUTED -- the int8 result may not be attributed to the chain"
    if 0.97 <= ratio <= 1.05:
        return "NULL"
    return "OUT-OF-BAND -- the brief registered no verdict here; reported, not adjudicated"


def g_e61d(rels):
    """Admissibility: rel L2 of m4 against m1 <= 1.0e-05 (E8 G-Z2a's threshold)."""
    if not rels:
        return "DEAD -- nothing compared"
    bad = ["%s=%.3e" % (k, v) for k, v in sorted(rels.items()) if v > 1.0e-05]
    return "REJECTED -- " + ",".join(bad) if bad else "ADMISSIBLE"


def g_e61e0(same_pair, diff_pair):
    """The metric's own planted control.  same_pair/diff_pair are (same, total)."""
    if not same_pair or not diff_pair or not same_pair[1] or not diff_pair[1]:
        return "VOID -- control not measured"
    hi = same_pair[0] / float(same_pair[1])
    lo = diff_pair[0] / float(diff_pair[1])
    if hi != 1.0:
        return "VOID -- the instrument does not agree with itself (%.6f)" % hi
    if lo >= 0.50:
        return "VOID -- the instrument reads a known-broken arm as %.1f%%" % (100.0 * lo)
    return "BRACKETS"


def g_e61e(agrees, dbpbs, need=0.990, dmax=0.0005):
    """agrees = {size: (same,total)}, dbpbs = {size: |dBPB|}.  Both clauses must hold."""
    if not agrees or not dbpbs:
        return "DEAD -- not measured"
    ok = all(s / float(t) >= need for s, t in agrees.values())
    ok = ok and all(abs(d) <= dmax for d in dbpbs.values())
    return "FAITHFUL" if ok else "DEGRADED"


def g_e61f(ratio):
    """The headline.  Bands fixed in the brief 6 before anything ran."""
    if ratio is None:
        return "DEAD -- not measured"
    if ratio >= 1.35:
        return "OVERSHOOT -- above the derivation; the mechanism is NOT the published explanation"
    if ratio >= 1.12:
        return "CHAIN-CONFIRMED-ON-THE-BYTE"
    if ratio > 1.06:
        return "UNDECIDED"
    return "REFUTED -- 31.7-33.1 GB/s is a property of the box, and 59.2 stands as written"


def selftest():
    n = 0

    def ck(cond, what):
        nonlocal n
        n += 1
        if not cond:
            raise SystemExit("SELFTEST FAILED: " + what)

    ck(g_e61a([]).startswith("DEAD"), "a: empty is dead")
    ck(g_e61a([("05b/I8", True), ("15b/I8", True)]) == "FIRES", "a: all identical fires")
    ck(g_e61a([("05b/I8", True), ("15b/I8", False)]).startswith("DEAD"), "a: one mismatch kills")

    ck(g_e61b(None).startswith("DEAD"), "b: unmeasured is dead")
    ck(g_e61b(1.2301) == "FIRES", "b: E8's own reading fires")
    ck(g_e61b(1.10) == "FIRES", "b: the boundary is inclusive")
    ck(g_e61b(1.0999).startswith("DEAD"), "b: just under does not fire")

    ck(g_e61c(1.0029) == "NULL", "c: E8's own reading is null")
    ck(g_e61c(1.05) == "NULL", "c: upper boundary inclusive")
    ck(g_e61c(1.10).startswith("MECHANISM-REFUTED"), "c: 1.10 refutes")
    ck(g_e61c(1.07).startswith("OUT-OF-BAND"), "c: the unregistered zone is named, not judged")
    ck(g_e61c(0.90).startswith("OUT-OF-BAND"), "c: a slowdown is also out of band")

    ck(g_e61d({}).startswith("DEAD"), "d: empty is dead")
    ck(g_e61d({"05b": 3.3e-06, "15b": 4.0e-06}) == "ADMISSIBLE", "d: E8's magnitude passes")
    ck(g_e61d({"05b": 3.3e-06, "15b": 2.0e-05}).startswith("REJECTED"), "d: one bad rejects")
    ck(g_e61d({"05b": 1.0e-05}) == "ADMISSIBLE", "d: the boundary is inclusive")

    ck(g_e61e0((100, 100), (30, 100)) == "BRACKETS", "e0: brackets")
    ck(g_e61e0((99, 100), (30, 100)).startswith("VOID"), "e0: no self-agreement is void")
    ck(g_e61e0((100, 100), (80, 100)).startswith("VOID"), "e0: a blind low end is void")
    ck(g_e61e0(None, (30, 100)).startswith("VOID"), "e0: missing control is void")

    ck(g_e61e({"05b": (12264, 12264)}, {"05b": 0.0}) == "FAITHFUL", "e: perfect is faithful")
    ck(g_e61e({"05b": (12200, 12264)}, {"05b": 0.0}) == "FAITHFUL", "e: 99.5% passes")
    ck(g_e61e({"05b": (12100, 12264)}, {"05b": 0.0}) == "DEGRADED", "e: 98.7% fails the count")
    ck(g_e61e({"05b": (12264, 12264)}, {"05b": 0.002}) == "DEGRADED", "e: dBPB clause bites")
    ck(g_e61e({"05b": (12264, 12264), "15b": (12000, 12264)},
              {"05b": 0.0, "15b": 0.0}) == "DEGRADED", "e: either size can fail it")

    ck(g_e61f(None).startswith("DEAD"), "f: unmeasured is dead")
    ck(g_e61f(1.40).startswith("OVERSHOOT"), "f: overshoot band")
    ck(g_e61f(1.35).startswith("OVERSHOOT"), "f: overshoot boundary inclusive")
    ck(g_e61f(1.19) == "CHAIN-CONFIRMED-ON-THE-BYTE", "f: the predicted reading confirms")
    ck(g_e61f(1.12) == "CHAIN-CONFIRMED-ON-THE-BYTE", "f: confirm boundary inclusive")
    ck(g_e61f(1.10) == "UNDECIDED", "f: undecided band")
    ck(g_e61f(1.06).startswith("REFUTED"), "f: the dispersion floor refutes")
    ck(g_e61f(1.00).startswith("REFUTED"), "f: no effect refutes")

    ck(abs(quantile([1, 2, 3, 4], 0.5) - 2.5) < 1e-12, "quantile")
    ck(top1_agree.__doc__ is not None, "top1_agree documented")
    log("   self-test: %d/%d PASS" % (n, n))


# ============================================================ measurement primitives
def logits(engine, weights, mvacc, out, extra=()):
    cmd = [engine, "--weights", weights, "--threads", str(THREADS),
           "--seqlen", str(BPB_SEQLEN)]
    if mvacc is not None:
        cmd += ["--mvacc", str(mvacc)]
    cmd += list(extra) + ["--logits", BPB_IDS, str(LOGIT_N), out]
    E51.run(cmd, "logits")
    return out


def bpb_top1(weights, mvacc, t1out):
    """(bpb, npred, seconds).  One pass gives both clauses of G-E61e."""
    cmd = [ENGINE, "--weights", weights, "--threads", str(THREADS),
           "--seqlen", str(BPB_SEQLEN), "--mvacc", str(mvacc),
           "--bpb", BPB_IDS, "--top1", t1out]
    txt, dt = E51.run(cmd, "bpb")
    m = E51.RE_NATS.search(txt)
    if not m:
        raise SystemExit("no NATS_TOTAL for " + weights)
    tot, npred = float(m.group(1)), int(m.group(2))
    if npred != BPB_NPRED:
        raise SystemExit("slice changed: %d predicted, expected %d" % (npred, BPB_NPRED))
    return tot / (math.log(2.0) * BPB_BYTES), npred, dt


def one_cell(weights, mvacc):
    sp = E44.Split()
    cmd = [ENGINE, "--weights", weights, "--threads", str(THREADS),
           "--mvacc", str(mvacc), "--bench", str(BENCH_N)]
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
    cfg = [l for l in txt.splitlines() if l.startswith("CONFIG")]
    cfg = cfg[0] if cfg else ""
    # E61 exists because this line printed mvacc=4 on a kernel that read 1.  Now that all three
    # branches honour it, the claim is checkable, so it is CHECKED rather than trusted.
    if ("mvacc=%d" % mvacc) not in cfg:
        raise SystemExit("CONFIG does not report mvacc=%d: %s" % (mvacc, cfg))
    cell = {"rate": float(m.group(3)), "occ": occ, "foreign": foc, "config": cfg}
    cell.update(sp.record())
    return cell


def save(out):
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, "e61_one_chain.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return p


# ============================================================
def main():
    log("E61 -- the only kernel that works is the only kernel still running a single accumulator")
    log("brief: BRIEF_E61_THE_CHAIN_ON_THE_BYTE_THAT_WORKS.md  (pre-registered, ab2a537)")
    log("")
    log("== self-test ==")
    selftest()
    if "--selftest" in sys.argv:
        return
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    sizes = [s for s in ("05b", "15b") if only is None or s == only]

    for k, w in WEIGHTS.items():
        if not os.path.exists(w):
            raise SystemExit("missing weights for %s: %s" % (k, w))
    os.makedirs(E51.TMP, exist_ok=True)
    rng = random.Random(SEED)
    t_start = time.time()

    out = {"brief": "BRIEF_E61_THE_CHAIN_ON_THE_BYTE_THAT_WORKS.md",
           "engine": os.path.basename(ENGINE), "frozen": os.path.basename(FROZEN),
           "bench_n": BENCH_N, "reps": REPS, "threads": THREADS, "occ_bar": OCC_BAR,
           "slice": {"n_seq": 24, "seq_len": BPB_SEQLEN, "bytes": BPB_BYTES,
                     "n_predicted": BPB_NPRED, "chance_bpb": CHANCE_BPB},
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "sizes": {}}

    def mark(tag):
        out.setdefault("timeline", []).append([tag, round(time.time() - t_start, 1)])

    # ------------------------------------------------ phase 1: G-E61a, the change is inert
    log("")
    log("== G-E61a -- PLANTED: --mvacc 1 must be BYTE-FOR-BYTE the pre-E61 loop ==")
    log("   e61 --mvacc 1  vs  e60 (which had no --mvacc on this branch at all).")
    pairs, hashes = [], {}
    for size in sizes:
        w = WEIGHTS[(size, "I8")]
        a = logits(ENGINE, w, 1, os.path.join(E51.TMP, "e61_%s_m1.bin" % size))
        b = logits(FROZEN, w, None, os.path.join(E51.TMP, "e60_%s.bin" % size))
        ha, hb = sha256(a), sha256(b)
        hashes["%s/e61_m1" % size] = ha
        hashes["%s/e60" % size] = hb
        pairs.append(("%s/I8" % size, ha == hb))
        log("   %-4s I8   e61 m1 %s\n        I8   e60    %s   %s"
            % (size, ha[:16], hb[:16], "IDENTICAL" if ha == hb else "*** DIFFERS ***"))
    va = g_e61a(pairs)
    log("   G-E61a : %s" % va)
    out["G_E61a"] = {"pairs": [[n, ok] for n, ok in pairs], "sha256": hashes, "verdict": va}
    mark("G-E61a")
    save(out)
    if va != "FIRES":
        raise SystemExit("the instrumentation is not inert.  The change is rejected.  STOP.")

    # ------------------------------------------------ phase 2: G-E61d, admissibility
    log("")
    log("== G-E61d -- parity: rel L2 of --mvacc 4 against --mvacc 1, <= 1.0e-05 ==")
    rels = {}
    for size in sizes:
        w = WEIGHTS[(size, "I8")]
        p1 = os.path.join(E51.TMP, "e61_%s_m1.bin" % size)
        p4 = logits(ENGINE, w, 4, os.path.join(E51.TMP, "e61_%s_m4.bin" % size))
        rels[size] = rel_l2(p1, p4)
        log("   %-4s I8   rel L2 = %.4e" % (size, rels[size]))
    vd = g_e61d(rels)
    log("   G-E61d : %s" % vd)
    out["G_E61d"] = {"rel_l2": rels, "verdict": vd}
    mark("G-E61d")
    save(out)
    if vd != "ADMISSIBLE":
        raise SystemExit("the 4-chain arm is not admissible.  Its speed is not read.  STOP.")

    # ------------------------------------------------ phase 3: G-E61e0, the metric's control
    log("")
    log("== G-E61e0 -- PLANTED, on the fidelity metric itself ==")
    log("   ~100%% agreement is a CEILING; a counter on its ceiling cannot show damage either.")
    log("   F32 twice must read 100.000%%; PACKED against F32 must read under 50%%.")
    ctl = {}
    fa = os.path.join(E51.TMP, "t1_05b_f32_a.bin")
    fb = os.path.join(E51.TMP, "t1_05b_f32_b.bin")
    fp = os.path.join(E51.TMP, "t1_05b_packed.bin")
    b1, _, d1 = bpb_top1(WEIGHTS[("05b", "F32")], 4, fa)
    log("   05b F32  run A  BPB %.9f   (%.0f s)" % (b1, d1))
    b2, _, d2 = bpb_top1(WEIGHTS[("05b", "F32")], 4, fb)
    log("   05b F32  run B  BPB %.9f   (%.0f s)" % (b2, d2))
    bp, _, dp = bpb_top1(WEIGHTS[("05b", "PACKED")], 4, fp)
    log("   05b PACKED     BPB %.9f   (%.0f s)" % (bp, dp))
    same_pair = top1_agree(fa, fb)
    diff_pair = top1_agree(fp, fa)
    log("   F32 vs F32     %6d/%d = %.6f%%" % (same_pair[0], same_pair[1],
                                               100.0 * same_pair[0] / same_pair[1]))
    log("   PACKED vs F32  %6d/%d = %.4f%%" % (diff_pair[0], diff_pair[1],
                                               100.0 * diff_pair[0] / diff_pair[1]))
    for nm, b in (("F32_a", b1), ("F32_b", b2), ("PACKED", bp)):
        k = ("05b", nm.split("_")[0])
        if k in E60_BPB:
            log("        %-7s vs E60's %.9f -> |d| = %.2e" % (nm, E60_BPB[k], abs(b - E60_BPB[k])))
    v0 = g_e61e0(same_pair, diff_pair)
    log("   G-E61e0 : %s" % v0)
    ctl = {"f32_a_bpb": b1, "f32_b_bpb": b2, "packed_bpb": bp,
           "f32_vs_f32": list(same_pair), "packed_vs_f32": list(diff_pair), "verdict": v0}
    out["G_E61e0"] = ctl
    mark("G-E61e0")
    save(out)
    if v0 != "BRACKETS":
        raise SystemExit("the fidelity metric does not bracket.  G-E61e is VOID.  STOP.")

    # ------------------------------------------------ phase 4: G-E61e, the treatment
    log("")
    log("== G-E61e -- per-position top-1 under teacher forcing, %d positions ==" % BPB_NPRED)
    log("   I8 --mvacc 4 against I8 --mvacc 1.  Pass: >= 99.0%% and |dBPB| <= 0.0005.")
    agrees, dbpbs = {}, {}
    for size in sizes:
        w = WEIGHTS[(size, "I8")]
        pa = os.path.join(E51.TMP, "t1_%s_i8_m1.bin" % size)
        pb = os.path.join(E51.TMP, "t1_%s_i8_m4.bin" % size)
        ba, _, da = bpb_top1(w, 1, pa)
        bb, _, db = bpb_top1(w, 4, pb)
        ag = top1_agree(pa, pb)
        agrees[size] = ag
        dbpbs[size] = bb - ba
        known = E60_BPB.get((size, "I8"))
        note = "" if known is None else "  (E60 m1: %.9f, |d| = %.2e)" % (known, abs(ba - known))
        log("   %-4s m1 BPB %.9f (%.0f s)%s" % (size, ba, da, note))
        log("        m4 BPB %.9f (%.0f s)   dBPB = %+.9f" % (bb, db, dbpbs[size]))
        log("        top-1 %6d/%d = %.4f%%" % (ag[0], ag[1], 100.0 * ag[0] / ag[1]))
        out["sizes"].setdefault(size, {})["fidelity"] = {
            "m1_bpb": ba, "m4_bpb": bb, "dbpb": dbpbs[size],
            "top1_same": ag[0], "top1_total": ag[1], "m1_seconds": da, "m4_seconds": db}
        save(out)
    ve = g_e61e(agrees, dbpbs)
    log("   G-E61e : %s" % ve)
    out["G_E61e"] = {"agree": dict((k, list(v)) for k, v in agrees.items()),
                     "dbpb": dbpbs, "verdict": ve}
    mark("G-E61e")
    save(out)

    # ------------------------------------------------ phase 5: the rates
    log("")
    log("== G-E61b / G-E61c / G-E61f / G-E61g -- speed, %d reps, --bench %d ==" % (REPS, BENCH_N))
    log("   arms AND mvacc settings interleaved inside every rep, one binary, one sweep.")
    for size in sizes:
        arms = SPEED[size]
        cellsp = [(a, m) for a in arms for m in MVACCS]
        log("")
        log("-- %s --" % size)
        for a, m in cellsp:
            c = one_cell(WEIGHTS[(size, a)], m)
            log("   warm-up %-7s m%d DISCARDED: %7.2f tok/s" % (a, m, c["rate"]))
        acc = dict((k, []) for k in cellsp)
        for r in range(REPS):
            for a, m in cellsp:
                c = one_cell(WEIGHTS[(size, a)], m)
                acc[(a, m)].append(c)
                log("   rep %d  %-7s m%d %7.2f tok/s  foreign %5.2f%%"
                    % (r + 1, a, m, c["rate"], c["foreign"]))
        log("")
        log("   %-7s %3s %9s %13s %18s %9s %7s" % ("arm", "mv", "median", "min..max",
                                                   "95% CI", "foreign", ">bar"))
        med, ci, rates = {}, {}, {}
        sz = out["sizes"].setdefault(size, {})
        sz.setdefault("speed", {})
        for a, m in cellsp:
            v = [c["rate"] for c in acc[(a, m)]]
            fo = [c["foreign"] for c in acc[(a, m)]]
            rates[(a, m)] = v
            med[(a, m)] = statistics.median(v)
            ci[(a, m)] = boot_ci(v, rng)
            sz["speed"]["%s_m%d" % (a, m)] = {
                "rates": v, "median": med[(a, m)], "ci": list(ci[(a, m)]), "foreign": fo,
                "foreign_mean": sum(fo) / len(fo),
                "over_occ_bar": sum(1 for x in fo if x > OCC_BAR)}
            log("   %-7s %3d %9.2f %5.1f..%-5.1f %8.2f..%-8.2f %8.2f%% %5d/%d"
                % (a, m, med[(a, m)], min(v), max(v), ci[(a, m)][0], ci[(a, m)][1],
                   sum(fo) / len(fo), sum(1 for x in fo if x > OCC_BAR), len(fo)))
            k60 = E60_RATE.get((size, a))
            if k60 is not None and m == 1:
                log("           E60 read %.2f on this arm (different binary, +-5%% between "
                    "sweeps -- context, not a gate)" % k60)

        log("")
        log("   %-7s %14s %14s %20s" % ("arm", "paired median", "median ratio", "paired 95% CI"))
        ratios = {}
        for a in arms:
            pr = [x / y for x, y in zip(rates[(a, 4)], rates[(a, 1)])]
            ratios[a] = statistics.median(pr)
            rci = boot_ci(pr, rng)
            sz["speed"]["%s_ratio" % a] = {"paired": pr, "paired_median": ratios[a],
                                           "paired_ci": list(rci),
                                           "median_ratio": med[(a, 4)] / med[(a, 1)]}
            log("   %-7s %14.4f %14.4f %9.4f..%-9.4f"
                % (a, ratios[a], med[(a, 4)] / med[(a, 1)], rci[0], rci[1]))
        save(out)

        if size == "05b":
            vb = g_e61b(ratios.get("PACKED"))
            vc = g_e61c(ratios.get("F32"))
            log("")
            log("   G-E61b : %s   (packed m4/m1 = %.4f; E8 G-Z3 read 1.2301)"
                % (vb, ratios.get("PACKED", float("nan"))))
            log("   G-E61c : %s   (fp32 m4/m1 = %.4f; E8 G-Z5 read 1.0029)"
                % (vc, ratios.get("F32", float("nan"))))
            sz["G_E61b"] = {"ratio": ratios.get("PACKED"), "verdict": vb}
            sz["G_E61c"] = {"ratio": ratios.get("F32"), "verdict": vc}
            if "T1" in ratios:
                g1 = med[("I8", 1)] - med[("T1", 1)]
                g4 = med[("I8", 4)] - med[("T1", 4)]
                log("   G-E61g : I8-T1 gap  m1 %+.2f tok/s -> m4 %+.2f tok/s   "
                    "(E60 read +1.98; descriptive, no verdict)" % (g1, g4))
                sz["G_E61g"] = {"gap_m1": g1, "gap_m4": g4, "e60_gap": 1.98}

        vf = g_e61f(ratios.get("I8"))
        log("   G-E61f : %s" % vf)
        log("            I8 %.2f -> %.2f tok/s, paired median %.4f"
            % (med[("I8", 1)], med[("I8", 4)], ratios["I8"]))
        sz["G_E61f"] = {"ratio": ratios.get("I8"), "verdict": vf,
                        "m1_median": med[("I8", 1)], "m4_median": med[("I8", 4)]}
        save(out)

    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["wall_s"] = round(time.time() - t_start, 1)
    mark("done")
    p = save(out)
    log("")
    log("wrote %s  (%.1f min)" % (p, out["wall_s"] / 60.0))


if __name__ == "__main__":
    main()
