#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E63 Part A -- the carved int8 path, proved correct before anything is timed.

Brief: `docs/research/donor_adaptation/briefs/BRIEF_E63_THE_TEN_BILLION_CELL_AT_ONE_BYTE.md`
Pre-registered and PUSHED before a line of engine code existed; ADDENDUM A (A.1, A.2, A.4)
pushed before this file ran, re-specifying two gates that could not be answered as written.

WHY PART A IS A SEPARATE RUNNER.  Part A is deterministic and may run on a dirty box; Part B is
a rate and may not.  E40 addendum A forbids re-running a fired gate to a pass, so the two must
not share a process that could tempt anyone to repeat the cheap half to rescue the expensive one.
Part A ships alone if the idle window never comes, and says so.

WHAT THIS RUNNER IS CAREFUL ABOUT.

  * It plants twice, in two directions.  `G-E63a` asks whether the new engine changed anything
    that already existed (it must not); `G-E63b` asks whether the fidelity instrument can see a
    change at all (it must).  E59's law: a fidelity gate measured against a reference the
    treatment cannot move is not a gate.  A near-100% reading on G-E63b VOIDS G-E63c -- the
    verdict word is `VOID`, not `PASS`.

  * It checks every witness.  E61 exists because a `CONFIG` line named a kernel it was not
    running.  Every carved cell here has its file's SIDECAR checked (seed, codes, head, carve_E,
    the FFN byte width) and the engine's own `--carve-k` echo checked against the k asked for.

  * It compares KERNELS, not weights (addendum A.4).  `w_i8` and `w_packed` draw the same trits
    off the same rng stream, so packed-carved and int8-carved at one seed are the same matrix
    under two kernels.  The cost is stated where it is charged: an error the exporter makes
    IDENTICALLY in both formats is invisible here, and on noise weights that was never on offer.

  * It carries NO rate.  Every `GEN_*_TOKS` the engine prints is recorded as conduct and is
    explicitly not a measurement: this box is contended and Part B is the only place a rate may
    be read.  A contended timing is not a timing.

  * It writes results after EVERY phase.
"""
import hashlib
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e44_interval as E44                                          # noqa: E402

OUTDIR = os.path.join(HERE, "results")
OUTFILE = os.path.join(OUTDIR, "e63_part_a.json")
BRIEF = "BRIEF_E63_THE_TEN_BILLION_CELL_AT_ONE_BYTE.md"

ENGINE_NEW = os.path.join(HERE, "donor_engine_e63.exe")
ENGINE_OLD = os.path.join(HERE, "donor_engine_e61.exe")     # frozen, pre-E63
THREADS = 6

E1 = os.path.join("D:", os.sep, "_ktmp", "e1")
E36 = os.path.join("D:", os.sep, "_ktmp", "e36")
E60 = os.path.join("D:", os.sep, "_ktmp", "e60")
E63 = os.path.join("D:", os.sep, "_ktmp", "e63")
WORK = os.path.join(E63, "parta")

# ---------------------------------------------------------------- the prompt (addendum A.2)
# The standard slice cannot be used: its ids run to 151,936 and A10B has V = 32,768.  A
# deterministic in-range prompt instead -- no rng, no file to lose, reproducible from this line.
PROMPT_TOKENS = 256
NEW_TOKENS = 24
PROMPT_STRIDE = 7919          # coprime with both vocabs, so the prompt is not a short cycle


def prompt_ids(vocab, n=PROMPT_TOKENS):
    return [(i * PROMPT_STRIDE + 13) % vocab for i in range(n)]


# ---------------------------------------------------------------- registered thresholds
G63B_MAX_AGREE = 0.60         # <= 0.60 => FIRES (addendum A.1)
G63C_MIN_AGREE = 0.9990       # >= 99.90% (addendum A.2)
G63C_MAX_ABSD = 1.0e-3

# ---------------------------------------------------------------- G-E63a's five artefacts
# Every artefact this programme still uses.  Two new enum values must change none of them.
PLANTED = [
    ("05b_f32",        os.path.join(E1,  "qwen25-05b_f32.bin"), 151936, []),
    ("05b_tqh",        os.path.join(E1,  "qwen25-05b_tqh.bin"), 151936, []),
    ("05b_i8h",        os.path.join(E60, "qwen25-05b_i8h.bin"), 151936, []),
    ("15b_i8h",        os.path.join(E60, "qwen25-15b_i8h.bin"), 151936, []),
    ("a10b_packed_k3", os.path.join(E36, "e36_a10b.bin"),        32768, ["--carve-k", "3"]),
]

# ---------------------------------------------------------------- the carved pairs
# (shape, packed file, int8 file, vocab, carve_E).  The two files of a pair were written from
# ONE seed with ONE codes mode: addendum A.4's whole argument rests on that, so it is CHECKED
# against both sidecars below and the pair is refused if they disagree.
PAIRS = [
    ("S05",  os.path.join(E63, "smoke_pk.bin"),  os.path.join(E63, "smoke_i8.bin"),    151936, 16),
    ("A10B", os.path.join(E36, "e36_a10b.bin"),  os.path.join(E63, "e63_a10b_i8.bin"),  32768, 256),
]
SIDECAR_MUST_MATCH = ("shape", "codes", "seed", "head", "carve_E", "carve_k_in_file", "vocab")


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================ the decision functions
def g_e63a(rows):
    """PLANTED: the new engine is inert on everything that already exists.

    `rows` = [{"name":..., "ids_sha_new":..., "ids_sha_old":...}, ...].  Zero tolerance: two
    added enum values and one relaxed guard must change nothing, and if they do the loader was
    not additive.  FAIL => nothing below is read.
    """
    if len(rows) < len(PLANTED):
        return "DEAD -- %d of %d artefacts measured" % (len(rows), len(PLANTED))
    bad = [r["name"] for r in rows if r["ids_sha_new"] != r["ids_sha_old"]]
    if bad:
        return "DEAD -- ids differ on: " + ", ".join(sorted(bad))
    return "FIRES"


def g_e63b(agreements):
    """PLANTED: the instrument must DISCRIMINATE before it certifies (addendum A.1).

    `agreements` = {shape: fraction}, k=3 against k=E on the SAME int8 carved file.  A near-1.0
    reading means the row list is not selecting anything, and then G-E63c is VOID, not passed.
    """
    if not agreements:
        return "DEAD -- no discrimination cell"
    bad = ["%s=%.4f" % (s, a) for s, a in sorted(agreements.items()) if a > G63B_MAX_AGREE]
    if bad:
        return "DEAD -- agreement above %.2f on: %s" % (G63B_MAX_AGREE, ", ".join(bad))
    return "FIRES"


def g_e63c(cells, discriminates):
    """The new path computes the right thing (addendum A.2 + A.4).

    `cells` = [{"shape","k","ids_identical","top1_agree","max_absd"}, ...].
    Gated on G-E63b: an instrument that cannot see a change cannot certify the absence of one.
    """
    if not discriminates:
        return "VOID -- G-E63b did not fire; the instrument cannot see a format change"
    if not cells:
        return "DEAD -- no equivalence cell"
    bad = []
    for c in cells:
        tag = "%s k=%s" % (c["shape"], c["k"])
        if not c["ids_identical"]:
            bad.append(tag + " ids differ")
        if c["top1_agree"] < G63C_MIN_AGREE:
            bad.append("%s top1 %.6f" % (tag, c["top1_agree"]))
        if c["max_absd"] > G63C_MAX_ABSD:
            bad.append("%s max|d| %.3e" % (tag, c["max_absd"]))
    return "EQUIVALENT" if not bad else "DEAD -- " + "; ".join(bad)


# ============================================================ measurement primitives
def sidecar(path):
    p = path + ".json"
    if not os.path.exists(p):
        raise SystemExit("no sidecar beside %s -- refusing to trust an unlabelled artefact" % p)
    with open(p, "r") as fh:
        return json.load(fh)


def check_pair(shape, pk, i8, vocab, carve_e):
    """Addendum A.4's premise, checked instead of assumed."""
    a, b = sidecar(pk), sidecar(i8)
    for f in SIDECAR_MUST_MATCH:
        if a.get(f) != b.get(f):
            raise SystemExit("pair %s disagrees on %r: packed=%r int8=%r"
                             % (shape, f, a.get(f), b.get(f)))
    if a.get("vocab") != vocab or a.get("carve_E") != carve_e:
        raise SystemExit("pair %s: sidecar says vocab=%r carve_E=%r, runner says %r/%r"
                         % (shape, a.get("vocab"), a.get("carve_E"), vocab, carve_e))
    if a.get("ffn_bytes_per_weight", 0.5) != 0.5 or b.get("ffn_bytes_per_weight") != 1.0:
        raise SystemExit("pair %s: FFN byte widths are %r and %r, want 0.5 and 1.0"
                         % (shape, a.get("ffn_bytes_per_weight"), b.get("ffn_bytes_per_weight")))
    return a, b


def write_ids(path, ids):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<%di" % len(ids), *ids))
    return path


def generate(engine, weights, idsfile, prefix, flags, k=None):
    """One --generate run.  Returns the parsed witness; the rates are CONDUCT, not results."""
    cmd = [engine, "--weights", weights, "--threads", str(THREADS),
           "--generate", idsfile, str(NEW_TOKENS), prefix] + list(flags)
    sp = E44.Split()
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    # G-E44b's defect, caught in THIS run's own output: the first Part A pass read
    # sp.close(0), so `foreign` counted the engine itself and every row printed
    # sys == foreign (33.7-55.2%).  Harmless here -- Part A gates nothing on conduct and its
    # axis is deterministic -- but Part B's G-E63d admissibility is foreign < OCC_BAR = 4.39%,
    # and with child_ticks = 0 NO box could ever pass it.  The handle must be read BEFORE the
    # Popen object releases it, exactly as e44_interval.one_rep does.
    child = E44._process_times(int(p._handle))
    wall = time.time() - t0
    if p.returncode != 0:
        raise SystemExit("engine failed (%d) on %s:\n%s"
                         % (p.returncode, weights, err.decode()[-4000:]))
    txt = out.decode("utf-8", "replace")
    etxt = err.decode("utf-8", "replace")
    sysb, foreign = sp.close(child)
    # The witness, checked.  A silently ignored --carve-k would make two different k read as
    # one file and turn G-E63b into a tautology that passes.
    if k is not None:
        want = "--carve-k %d" % k
        if want not in etxt:
            raise SystemExit("engine did not echo %r for %s; stderr was:\n%s"
                             % (want, weights, etxt[-2000:]))
        if "IGNORED" in etxt:
            raise SystemExit("engine IGNORED --carve-k on %s: %s" % (weights, etxt[-2000:]))
    rec = {"wall_s": round(wall, 3), "sys_pct": round(sysb, 2), "foreign_pct": round(foreign, 2)}
    for key, field in (("GEN_PREFILL_TOKS", "prefill_toks_conduct"),
                       ("GEN_DECODE_TOKS", "decode_toks_conduct")):
        for line in txt.splitlines():
            if line.startswith(key):
                rec[field] = float(line.split()[1])
    rec.update(sp.record())
    return rec


def read_prefill(prefix, vocab, npos=None):
    npos = PROMPT_TOKENS if npos is None else npos
    a = np.fromfile(prefix + ".prefill.bin", dtype="<f4")
    if a.size != npos * vocab:
        raise SystemExit("prefill %s is %d floats, want %d x %d" % (prefix, a.size, npos, vocab))
    return a.reshape(npos, vocab)


def compare(prefix_a, prefix_b, vocab):
    """Per-position top-1 agreement and max abs logit difference between two prefills."""
    A = read_prefill(prefix_a, vocab)
    B = read_prefill(prefix_b, vocab)
    agree = float(np.mean(np.argmax(A, axis=1) == np.argmax(B, axis=1)))
    absd = float(np.max(np.abs(A - B)))
    den = float(np.linalg.norm(A))
    rel = float(np.linalg.norm(A - B) / den) if den > 0 else float("nan")
    npos = A.shape[0]
    del A, B
    ids_a = open(prefix_a + ".ids.bin", "rb").read()
    ids_b = open(prefix_b + ".ids.bin", "rb").read()
    return {"top1_agree": agree, "max_absd": absd, "rel_l2": rel,
            "ids_identical": ids_a == ids_b,
            "n_positions": npos, "n_generated": NEW_TOKENS}


# ============================================================ self-test
def selftest():
    n = [0]

    def ok(cond, what):
        n[0] += 1
        if not cond:
            raise SystemExit("SELFTEST FAILED: %s" % what)

    # -- the prompt is in range, deterministic, and not a short cycle
    for V in (151936, 32768):
        ids = prompt_ids(V)
        ok(len(ids) == PROMPT_TOKENS, "prompt length at V=%d" % V)
        ok(all(0 <= i < V for i in ids), "prompt in range at V=%d" % V)
        ok(ids == prompt_ids(V), "prompt is deterministic at V=%d" % V)
    ok(len(set(prompt_ids(151936))) == PROMPT_TOKENS, "prompt has no repeat at the large vocab")
    ok(len(set(prompt_ids(32768))) > PROMPT_TOKENS // 2, "prompt is not a short cycle at V=32768")
    ok(prompt_ids(32768) != prompt_ids(151936), "the two prompts differ")

    # -- G-E63a: zero tolerance, and it needs all five
    five = [{"name": p[0], "ids_sha_new": "x", "ids_sha_old": "x"} for p in PLANTED]
    ok(g_e63a(five) == "FIRES", "G-E63a fires when every artefact matches")
    ok(g_e63a(five[:4]).startswith("DEAD"), "G-E63a is DEAD on four of five")
    ok(g_e63a([]).startswith("DEAD"), "G-E63a is DEAD on nothing")
    one = [dict(r) for r in five]
    one[2]["ids_sha_old"] = "y"
    ok(g_e63a(one).startswith("DEAD"), "G-E63a dies on one differing artefact")
    ok(PLANTED[2][0] in g_e63a(one), "G-E63a names who differed")

    # -- G-E63b: the threshold is an upper bound, and a perfect agreement must NOT pass
    ok(g_e63b({"A10B": 0.10}) == "FIRES", "G-E63b fires at 10% agreement")
    ok(g_e63b({"A10B": G63B_MAX_AGREE}) == "FIRES", "G-E63b fires exactly at the bar")
    ok(g_e63b({"A10B": 0.601}).startswith("DEAD"), "G-E63b dies just above the bar")
    ok(g_e63b({"A10B": 1.0}).startswith("DEAD"), "G-E63b dies on a perfect agreement")
    ok(g_e63b({"S05": 0.2, "A10B": 0.9}).startswith("DEAD"), "G-E63b dies if ANY shape fails")
    ok(g_e63b({}).startswith("DEAD"), "G-E63b is DEAD on nothing")

    # -- G-E63c: gated on the instrument, and each clause bites alone
    good = [{"shape": "A10B", "k": 3, "ids_identical": True,
             "top1_agree": 1.0, "max_absd": 1e-6}]
    ok(g_e63c(good, True) == "EQUIVALENT", "G-E63c passes a clean cell")
    ok(g_e63c(good, False).startswith("VOID"), "G-E63c is VOID when G-E63b did not fire")
    ok(g_e63c([], True).startswith("DEAD"), "G-E63c is DEAD on nothing")
    ok(g_e63c([dict(good[0], ids_identical=False)], True).startswith("DEAD"), "ids clause bites")
    ok(g_e63c([dict(good[0], top1_agree=0.998)], True).startswith("DEAD"), "top1 clause bites")
    ok(g_e63c([dict(good[0], max_absd=2e-3)], True).startswith("DEAD"), "max|d| clause bites")
    ok(g_e63c([dict(good[0], top1_agree=G63C_MIN_AGREE)], True) == "EQUIVALENT",
       "top1 bar is inclusive")
    ok(g_e63c([dict(good[0], max_absd=G63C_MAX_ABSD)], True) == "EQUIVALENT",
       "max|d| bar is inclusive")
    ok("k=3" in g_e63c([dict(good[0], max_absd=1.0)], True), "G-E63c names the failing cell")
    two = good + [dict(good[0], shape="S05", top1_agree=0.5)]
    ok(g_e63c(two, True).startswith("DEAD"), "G-E63c dies if ANY shape fails")

    # -- VOID is not PASS: the two verdicts must never be spelled the same
    ok(g_e63c(good, False) != g_e63c(good, True), "VOID and EQUIVALENT are different words")

    # -- the comparator does what it says
    os.makedirs(WORK, exist_ok=True)
    p = os.path.join(WORK, "_st")
    A = np.zeros((4, 3), dtype="<f4")
    A[:, 0] = 1.0
    B = A.copy()
    A.tofile(p + "_a.prefill.bin")
    B.tofile(p + "_b.prefill.bin")
    open(p + "_a.ids.bin", "wb").write(b"\x01\x02")
    open(p + "_b.ids.bin", "wb").write(b"\x01\x02")
    saved = globals()["PROMPT_TOKENS"]
    globals()["PROMPT_TOKENS"] = 4
    c = compare(p + "_a", p + "_b", 3)
    ok(c["top1_agree"] == 1.0 and c["max_absd"] == 0.0 and c["ids_identical"],
       "compare reads identity as identity")
    B[2, 1] = 5.0
    B.tofile(p + "_b.prefill.bin")
    open(p + "_b.ids.bin", "wb").write(b"\x01\x03")
    c = compare(p + "_a", p + "_b", 3)
    ok(abs(c["top1_agree"] - 0.75) < 1e-9, "compare counts one flipped position of four")
    ok(abs(c["max_absd"] - 5.0) < 1e-6, "compare reports the max abs difference")
    ok(not c["ids_identical"], "compare sees differing ids")
    globals()["PROMPT_TOKENS"] = saved
    for s in ("_a", "_b"):
        for e in (".prefill.bin", ".ids.bin"):
            os.remove(p + s + e)

    # -- sha256 of a known string
    ok(hashlib.sha256(b"abc").hexdigest().startswith("ba7816bf"), "sha256 is sha256")

    # -- the artefacts and binaries this run needs actually exist
    for nm, path, _, _ in PLANTED:
        ok(os.path.exists(path), "planted artefact %s exists (%s)" % (nm, path))
    for e in (ENGINE_NEW, ENGINE_OLD):
        ok(os.path.exists(e), "engine %s exists" % os.path.basename(e))

    log("   selftest: %d checks passed" % n[0])
    return n[0]


def save(out):
    os.makedirs(OUTDIR, exist_ok=True)
    out["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(OUTFILE, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)


# ============================================================ main
def main():
    os.makedirs(WORK, exist_ok=True)
    t_start = time.time()
    out = {"brief": BRIEF, "part": "A",
           "addendum": "A.1 re-specifies G-E63b, A.2 re-specifies G-E63c's positions, "
                       "A.4 names G-E63c's comparator; all three pushed before this run",
           "engine_new": os.path.basename(ENGINE_NEW),
           "engine_old": os.path.basename(ENGINE_OLD),
           "prompt": {"tokens": PROMPT_TOKENS, "new": NEW_TOKENS, "stride": PROMPT_STRIDE},
           "thresholds": {"G63B_MAX_AGREE": G63B_MAX_AGREE,
                          "G63C_MIN_AGREE": G63C_MIN_AGREE,
                          "G63C_MAX_ABSD": G63C_MAX_ABSD},
           "note_no_rate": "Part A carries no rate.  Every GEN_*_TOKS here is conduct on a "
                           "contended box and is not a measurement; Part B is the only place a "
                           "rate may be read.",
           "planted": [], "discrimination": {}, "equivalence": [], "conduct": {}}
    save(out)

    log("== E63 Part A -- the carved int8 path, proved correct before anything is timed")
    log("   brief %s (+ addendum A.1/A.2/A.4, pushed before this run)" % BRIEF)
    selftest()

    # ------------------------------------------------------------ phase 1: G-E63a (PLANTED)
    log("")
    log("-- phase 1: G-E63a -- is the new engine inert on everything that already exists?")
    log("   five artefacts, e63 vs frozen e61, sha256 of the generated ids, ZERO tolerance.")
    log("   FAIL => nothing below is read.")
    for name, path, vocab, flags in PLANTED:
        idsf = write_ids(os.path.join(WORK, "prompt_v%d.bin" % vocab), prompt_ids(vocab))
        row = {"name": name, "weights": path, "vocab": vocab, "flags": flags,
               "bytes": os.path.getsize(path)}
        k = 3 if "--carve-k" in flags else None
        for tag, eng in (("new", ENGINE_NEW), ("old", ENGINE_OLD)):
            pref = os.path.join(WORK, "a_%s_%s" % (name, tag))
            row["run_" + tag] = generate(eng, path, idsf, pref, flags, k=k)
            row["ids_sha_" + tag] = sha256_file(pref + ".ids.bin")
            row["prefill_sha_" + tag] = sha256_file(pref + ".prefill.bin")
            os.remove(pref + ".prefill.bin")     # 38 MB each; the digest is the evidence
        same_ids = row["ids_sha_new"] == row["ids_sha_old"]
        same_pre = row["prefill_sha_new"] == row["prefill_sha_old"]
        out["planted"].append(row)
        save(out)
        log("   %-16s ids %s   prefill %s   (%.0f s, foreign %.1f%%)"
            % (name, "IDENTICAL" if same_ids else "*** DIFFER ***",
               "identical" if same_pre else "differ  <- reported, not a gate",
               row["run_new"]["wall_s"], row["run_new"]["foreign_pct"]))

    va = g_e63a(out["planted"])
    out["G-E63a"] = va
    save(out)
    log("   G-E63a: %s" % va)
    if va != "FIRES":
        log("")
        log("   STOP.  The loader was not additive.  Nothing below is read, by the registered rule.")
        out["stopped_at"] = "G-E63a"
        save(out)
        return 1

    # ------------------------------------------------------------ phase 2+3: the carved pairs
    log("")
    log("-- phase 2: G-E63b -- can the instrument SEE the carve?  (k=3 against k=E, same file)")
    log("-- phase 3: G-E63c -- does the int8 carve compute what the packed carve computes?")
    log("   comparator = the packed carve at the same seed/shape/k (addendum A.4).")

    agreements = {}
    for shape, pk, i8, vocab, carve_e in PAIRS:
        if not os.path.exists(i8):
            log("   %-5s SKIPPED: %s does not exist" % (shape, i8))
            out.setdefault("skipped", []).append(shape)
            save(out)
            continue
        sc_pk, sc_i8 = check_pair(shape, pk, i8, vocab, carve_e)
        idsf = write_ids(os.path.join(WORK, "prompt_v%d.bin" % vocab), prompt_ids(vocab))
        log("   %-5s packed %s (%.2f GB) vs int8 %s (%.2f GB), seed %s codes %s head %s, E=%d"
            % (shape, os.path.basename(pk), os.path.getsize(pk) / 1e9,
               os.path.basename(i8), os.path.getsize(i8) / 1e9,
               sc_pk["seed"], sc_pk["codes"], sc_pk["head"], carve_e))

        prefs = {}
        for fmt, wp in (("pk", pk), ("i8", i8)):
            for k in (3, carve_e):
                pref = os.path.join(WORK, "c_%s_%s_k%d" % (shape, fmt, k))
                r = generate(ENGINE_NEW, wp, idsf, pref, ["--carve-k", str(k)], k=k)
                out["conduct"]["%s_%s_k%d" % (shape, fmt, k)] = r
                prefs[(fmt, k)] = pref
                save(out)

        # phase 2 -- the discrimination control, on the treatment's own file
        d_i8 = compare(prefs[("i8", 3)], prefs[("i8", carve_e)], vocab)
        d_pk = compare(prefs[("pk", 3)], prefs[("pk", carve_e)], vocab)
        agreements[shape] = d_i8["top1_agree"]
        out["discrimination"][shape] = {"int8_k3_vs_kE": d_i8, "packed_k3_vs_kE": d_pk,
                                        "carve_E": carve_e}
        save(out)
        log("     G-E63b  int8 k=3 vs k=%-3d  top1 agree %6.2f%%   (packed, for reference %6.2f%%)"
            % (carve_e, 100 * d_i8["top1_agree"], 100 * d_pk["top1_agree"]))

        # phase 3 -- the equivalence, at both k
        for k in (3, carve_e):
            c = compare(prefs[("pk", k)], prefs[("i8", k)], vocab)
            c.update({"shape": shape, "k": k})
            out["equivalence"].append(c)
            save(out)
            log("     G-E63c  k=%-3d  ids %s  top1 %8.4f%%  max|d| %.3e  rel l2 %.3e"
                % (k, "IDENTICAL" if c["ids_identical"] else "DIFFER",
                   100 * c["top1_agree"], c["max_absd"], c["rel_l2"]))

        for pref in prefs.values():
            os.remove(pref + ".prefill.bin")

    vb = g_e63b(agreements)
    vc = g_e63c(out["equivalence"], vb == "FIRES")
    out["G-E63b"] = vb
    out["G-E63c"] = vc
    out["wall_min"] = round((time.time() - t_start) / 60.0, 1)
    save(out)

    # ------------------------------------------------------------ the reading
    log("")
    log("== gates")
    log("   G-E63a  PLANTED, inert on five artefacts : %s" % va)
    log("   G-E63b  PLANTED, instrument discriminates: %s" % vb)
    log("   G-E63c  the int8 carve computes it right : %s" % vc)
    log("")
    log("   Part B (G-E63d, G-E63e) is NOT attempted here.  It needs an idle box "
        "(COMMUNICATION.md item 1) and it is the only place a rate may be read.")
    log("   %.1f min.  results: %s" % (out["wall_min"], OUTFILE))
    return 0 if (va == "FIRES" and vb == "FIRES" and vc == "EQUIVALENT") else 2


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    sys.exit(main())
