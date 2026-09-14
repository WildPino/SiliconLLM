# -*- coding: utf-8 -*-
"""E64 -- what does the CARVE cost on an INT8 FFN?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E64_WHAT_DOES_THE_CARVE_COST_ON_AN_INT8_FFN.md
plus addendum A, both pushed before this file ran.

QUALITY ONLY.  Not one tok/s is measured here.  `G-E63d` stays VOID and OWED and E64 does not
touch it.

THE DESIGN, in one paragraph.  Every carve cost this programme has published was measured on a
TERNARY FFN.  E64 runs E37's own carve ladder twice from ONE exported file per format -- the
engine's runtime `--carve-k` sweeps k without re-exporting, so within a format the arms differ
in nothing but k, and across formats they differ in nothing but the matrix kind.  The ternary
file is E37's own `e37_carved_nf.bin`, untouched on disk since 12 September, which makes the
ternary ladder a REPLICATION and not a finding (brief section 6 item 5).

WHY THE REPLICATION IS ALSO THE CONTROL.  E37 measured on `donor_engine_e26.exe`; E64 must use
`donor_engine_e63.exe`, the first binary that reads MK_I8/MK_I8_T.  If the ternary ladder
reproduces E37's published numbers on the new binary, the binary change is inert on the old
path and the int8 cell can be read.  If it does not, nothing in E64 may be read -- that is
`G-E64a`, and it is a planted control in the strict sense: a known positive the instrument must
FIRE on before its novel cell counts.
"""
import argparse, json, math, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e64_carve_on_int8.json")

ENGINE = os.path.join(HERE, "donor_engine_e63.exe")

# E1's protocol and convention -- the same constants E32 and E37 used, not re-derived here.
SEQLEN = 512
SCORED_BYTES, N_PRED = 51870, 12264
BYTES_PER_TOK = SCORED_BYTES / float(N_PRED)
V = 151936
CHANCE_BPB = math.log(V, 2) / BYTES_PER_TOK                  # 4.069819
ANCHOR_IDS = r"D:\_ktmp\e1\ids_qwen25-15b_tqh.bin"

TERN = r"D:\_ktmp\e37\e37_carved_nf.bin"                     # E37's own carved artefact
TERN_DENSE = r"D:\_ktmp\e37\e37_dense_nf.bin"                # its uncarved counterpart
I8 = r"D:\_ktmp\e64\e64_carved_i8.bin"                       # E64's, --ffn-rule R8

E_GROUPS = 256
KS = [256, 64, 32, 16, 8, 4, 3, 2, 1]                        # addendum A.3: E37's own ladder

# ---- E37's published readings, results/e37_sparsity_cost.json, for G-E64a.  Transcribed, not
# recomputed: a replication that recomputes its own target is not a replication.
E37_LADDER = {
    256: 3.4757066520304316, 64: 3.9273837335403514, 32: 4.074431016419263,
    16: 3.9868009530008397, 8: 3.9966933605102373, 4: 4.0234390393278865,
    3: 4.029398350226611, 2: 4.0148660778430445, 1: 3.989838501154876,
}
E37_DENSE = 3.47570637184527
G64A_TOL = 1.0e-06          # brief section 4
G64B_TOL = 1.0e-04          # addendum A.1: E37's own G-E37A tolerance, not one I invented
SIGMA_SEED_15B = 0.250      # E62's published value for this donor; NOT re-derived here


def run_bpb(weights, flags):
    """E37's run_bpb, unchanged except for the binary.  Refuses anything that is not E1's
    protocol -- N_PREDICTED must be 12,264 or the slice is not the frozen one."""
    cmd = [ENGINE, "--weights", weights, "--threads", "6", "--seqlen", str(SEQLEN),
           "--bpb", ANCHOR_IDS] + list(flags)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("engine --bpb failed (%d): %s" % (r.returncode, r.stderr[-2000:]))
    m = re.search(r"NATS_PER_TOKEN\s+([0-9.]+)", r.stdout)
    p = re.search(r"N_PREDICTED\s+(\d+)", r.stdout)
    if not m:
        raise SystemExit("no NATS_PER_TOKEN:\n" + r.stdout[-2000:])
    npred = int(p.group(1)) if p else None
    if npred != N_PRED:
        raise SystemExit("N_PREDICTED %r != %d -- not E1's protocol" % (npred, N_PRED))
    nats = float(m.group(1))
    return {"nats_per_token": nats, "bpb": nats / math.log(2) / BYTES_PER_TOK,
            "n_predicted": npred, "seconds": time.time() - t0,
            "config": (re.search(r"CONFIG[^\n]*", r.stdout) or [""])[0]
                      if re.search(r"CONFIG[^\n]*", r.stdout) else ""}


def confirm_format(weights, want_bpw, log):
    """REFUSE an arm whose sidecar does not confirm the format this arm claims to be.

    feedback_config_must_appear_in_output: it is not enough that I passed --ffn-rule; the
    ARTEFACT must say what it is, and the runner must check that rather than trust the command
    line it was handed.  E37's ternary artefact predates the field, so a missing value is
    accepted ONLY for the 0.5 arm and only when the sidecar still says quant=carved.  An int8
    arm with no field is refused outright -- that is precisely the case where a silent fallback
    to packed would come back looking like a result.
    """
    side = weights + ".json"
    if not os.path.exists(side):
        raise SystemExit("no sidecar for %s -- cannot confirm its format" % weights)
    m = json.load(open(side, encoding="utf-8"))
    got = m.get("ffn_bytes_per_weight")
    if got is None and want_bpw == 0.5 and m.get("quant") == "carved":
        log("  %s: sidecar predates ffn_bytes_per_weight; accepted as 0.5 "
            "(quant=carved, no ffn_rule)" % os.path.basename(weights))
        return {"source": "legacy-implied", "bytes_per_weight": 0.5,
                "ffn_rule": m.get("ffn_rule"), "ffn_kinds": m.get("ffn_kinds")}
    if got != want_bpw:
        raise SystemExit("FORMAT MISMATCH for %s: sidecar says ffn_bytes_per_weight=%r, this "
                         "arm requires %r. The cell is REFUSED." % (weights, got, want_bpw))
    log("  %s: sidecar confirms ffn_bytes_per_weight=%.1f, kinds=%s"
        % (os.path.basename(weights), got, m.get("ffn_kinds")))
    return {"source": "sidecar", "bytes_per_weight": got, "ffn_rule": m.get("ffn_rule"),
            "ffn_kinds": m.get("ffn_kinds")}


def ladder(weights, tag, log):
    """The same file at every k.  One export, nine readings."""
    out = {}
    for k in KS:
        r = run_bpb(weights, ["--carve-k", str(k)])
        out[k] = r
        log("  %-6s k=%-4d BPB %.12f   (%.0f s)" % (tag, k, r["bpb"], r["seconds"]))
    return out


def g_e64a(tern):
    """PLANTED CONTROL.  The ternary ladder on the NEW binary must reproduce E37's published
    ladder, which was taken on donor_engine_e26.exe.  Fires => the binary is inert on the old
    path and the int8 cell may be read."""
    rows, worst = {}, 0.0
    for k in KS:
        got, want = tern[k]["bpb"], E37_LADDER[k]
        d = abs(got - want)
        worst = max(worst, d)
        rows[k] = {"got": got, "e37": want, "abs_d": d, "agrees": bool(d <= G64A_TOL)}
    return {"rows": rows, "max_abs_d": worst, "tol": G64A_TOL,
            "fires": bool(worst <= G64A_TOL)}


def g_e64b(tern, dense_bpb):
    """The carve machinery is correctly wired: at k = E = 256 the carved file reproduces the
    SAME-FORMAT uncarved BPB.  Addendum A.1 sets the tolerance to E37's 1e-4 (the carved writer
    permutes F, so summation order differs and bit-identity is impossible).  Addendum A.2: this
    fires on the TERNARY arm only, because the int8 arm has no uncarved counterpart -- a version
    that 'passed' on int8 would compare the file to itself."""
    d = abs(tern[E_GROUPS]["bpb"] - dense_bpb)
    return {"carved_kE": tern[E_GROUPS]["bpb"], "dense": dense_bpb, "abs_d": d,
            "tol": G64B_TOL, "fires": bool(d <= G64B_TOL),
            "scope": "TERNARY ARM ONLY -- the int8 arm has no uncarved counterpart (addendum A.2)"}


def carve_cost(lad):
    """THE FLOOR RULE, written into the code because I broke it on 2026-09-14 (ledger 62.12):
    a format's carve cost is measured against ITS OWN uncarved baseline (k = E), never against
    another format's and never against fp32."""
    base = lad[E_GROUPS]["bpb"]
    return {k: lad[k]["bpb"] - base for k in KS}, base


def g_e64d(ct, ci):
    """ORDINAL, no tolerance on the difference itself (feedback_gate_vs_measured_dispersion:
    no dispersion has been measured for this cell, so none may be registered).  Compares
    LADDERS, not cells -- addendum A.3, because E37's ladder is non-monotone and a single k
    could show either sign."""
    per = {}
    for k in KS:
        if k == E_GROUPS:
            continue
        d = ci[k] - ct[k]                       # >0 => the carve costs MORE on int8
        per[k] = {"tern": ct[k], "i8": ci[k], "i8_minus_tern": d,
                  "sign": (0 if abs(d) <= SIGMA_SEED_15B else (1 if d > 0 else -1))}
    signs = set(v["sign"] for v in per.values())
    if signs == {0}:
        band = "CARVE-IS-PRECISION-BLIND"
    elif signs == {1}:
        band = "CARVE-IS-DEARER-ON-INT8"
    elif signs == {-1}:
        band = "CARVE-IS-CHEAPER-ON-INT8"
    else:
        band = "THE-LADDERS-DISAGREE"
    return {"per_k": per, "sigma_seed": SIGMA_SEED_15B, "signs": sorted(signs), "band": band,
            "max_abs_diff": max(abs(v["i8_minus_tern"]) for v in per.values())}


# ------------------------------------------------------------------ self-tests
def selftest():
    n = [0]

    def ok(c, m):
        n[0] += 1
        if not c:
            raise SystemExit("SELFTEST FAILED: " + m)

    ok(abs(CHANCE_BPB - 4.069819) < 1e-6, "chance line drifted: %r" % CHANCE_BPB)
    ok(abs(BYTES_PER_TOK - 4.22945205479452) < 1e-12, "bytes/token drifted")
    ok(KS[0] == E_GROUPS, "the ladder must start at k=E so the baseline exists")
    ok(len(set(KS)) == len(KS), "duplicate k")
    ok(set(E37_LADDER) == set(KS), "E37 ladder and KS disagree")

    # carve_cost must measure each format against ITS OWN k=E baseline
    lad = {k: {"bpb": 10.0 + k} for k in KS}
    cc, base = carve_cost(lad)
    ok(base == 10.0 + E_GROUPS, "baseline is not k=E")
    ok(cc[E_GROUPS] == 0.0, "the baseline's own cost must be exactly zero")
    ok(cc[1] == (10.0 + 1) - (10.0 + E_GROUPS), "carve cost is not a difference from k=E")

    # G-E64a fires only on an exact reproduction
    tern = {k: {"bpb": E37_LADDER[k]} for k in KS}
    ok(g_e64a(tern)["fires"], "G-E64a must fire on an exact copy of E37's ladder")
    bad = dict((k, {"bpb": v["bpb"]}) for k, v in tern.items())
    bad[8] = {"bpb": E37_LADDER[8] + 1e-5}
    ok(not g_e64a(bad)["fires"], "G-E64a must NOT fire at 1e-5 against a 1e-6 bar")
    ok(abs(g_e64a(bad)["max_abs_d"] - 1e-5) < 1e-12, "max_abs_d wrong")

    # G-E64b uses E37's tolerance and discriminates
    ok(g_e64b({E_GROUPS: {"bpb": 3.4757066520304316}}, E37_DENSE)["fires"],
       "G-E64b must fire on E37's own pair (diff 2.8e-07 vs tol 1e-4)")
    ok(not g_e64b({E_GROUPS: {"bpb": E37_DENSE + 1e-3}}, E37_DENSE)["fires"],
       "G-E64b must not fire at 1e-3")

    # G-E64d bands
    flat = dict((k, 0.5) for k in KS)
    ok(g_e64d(flat, flat)["band"] == "CARVE-IS-PRECISION-BLIND", "identical ladders")
    dear = dict((k, 0.5 + 10.0) for k in KS)
    ok(g_e64d(flat, dear)["band"] == "CARVE-IS-DEARER-ON-INT8", "uniformly dearer")
    cheap = dict((k, 0.5 - 10.0) for k in KS)
    ok(g_e64d(flat, cheap)["band"] == "CARVE-IS-CHEAPER-ON-INT8", "uniformly cheaper")
    mixed = dict((k, 0.5) for k in KS)
    mixed[3] = 0.5 + 10.0
    mixed[8] = 0.5 - 10.0
    ok(g_e64d(flat, mixed)["band"] == "THE-LADDERS-DISAGREE", "sign changes with k")
    # a difference inside sigma_seed is NOT a sign
    near = dict((k, 0.5 + 0.9 * SIGMA_SEED_15B) for k in KS)
    ok(g_e64d(flat, near)["band"] == "CARVE-IS-PRECISION-BLIND",
       "a difference under sigma_seed must not count as a sign")
    ok(g_e64d(flat, dear)["max_abs_diff"] == 10.0, "max_abs_diff wrong")
    # k=E is excluded from the verdict: its cost is zero by construction in BOTH formats, so
    # including it would guarantee a 'same' vote and dilute the ladder.
    ok(E_GROUPS not in g_e64d(flat, dear)["per_k"], "k=E must be excluded from the verdict")

    print("selftest: %d checks passed" % n[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    selftest()
    os.makedirs(RES, exist_ok=True)
    logf = open(os.path.join(HERE, "e64_run.log"), "w", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    for f in (ENGINE, ANCHOR_IDS, TERN, TERN_DENSE, I8):
        if not os.path.exists(f):
            raise SystemExit("missing: " + f)

    log("E64 -- the carve on an int8 FFN.  QUALITY ONLY, no rate.")
    log("engine %s" % os.path.basename(ENGINE))
    log("ternary arm  %s  (E37's own artefact -- REPLICATION)" % os.path.basename(TERN))
    log("int8 arm     %s  (--ffn-rule R8)" % os.path.basename(I8))

    t0 = time.time()
    log("\n-- ternary ladder (replication + binary-inertness control)")
    fmt_t = confirm_format(TERN, 0.5, log)
    fmt_i = confirm_format(I8, 1.0, log)
    tern = ladder(TERN, "tern", log)
    log("\n-- uncarved ternary reference")
    dense = run_bpb(TERN_DENSE, [])
    log("  dense      BPB %.12f" % dense["bpb"])
    log("\n-- int8 ladder (THE CELL)")
    i8 = ladder(I8, "i8", log)

    ga = g_e64a(tern)
    gb = g_e64b(tern, dense["bpb"])
    ct, base_t = carve_cost(tern)
    ci, base_i = carve_cost(i8)
    gd = g_e64d(ct, ci)

    log("\nG-E64a  replication of E37's ladder on the e63 binary: max|d| %.3e vs %.0e -> %s"
        % (ga["max_abs_d"], ga["tol"], "FIRES" if ga["fires"] else "DOES NOT FIRE"))
    log("G-E64b  carved k=E vs uncarved (ternary only): |d| %.3e vs %.0e -> %s"
        % (gb["abs_d"], gb["tol"], "FIRES" if gb["fires"] else "DOES NOT FIRE"))
    log("\nG-E64c  each format against ITS OWN k=E baseline (the floor rule):")
    log("  ternary baseline %.12f   int8 baseline %.12f" % (base_t, base_i))
    log("  %-5s %-16s %-16s %s" % ("k", "carve cost tern", "carve cost i8", "i8 - tern"))
    for k in KS:
        if k == E_GROUPS:
            continue
        log("  %-5d %-16.6f %-16.6f %+.6f" % (k, ct[k], ci[k], ci[k] - ct[k]))
    log("\nG-E64d  %s   (sigma_seed %.3f, max|diff| %.6f)"
        % (gd["band"], gd["sigma_seed"], gd["max_abs_diff"]))

    if not ga["fires"]:
        log("\n!! G-E64a DID NOT FIRE -- by the brief, NO CELL IN E64 MAY BE READ.")

    res = {"experiment": "E64", "brief": "BRIEF_E64_WHAT_DOES_THE_CARVE_COST_ON_AN_INT8_FFN.md"
                                          " + addendum A",
           "quality_only": True, "no_rate_measured": True,
           "g_e63d_status": "VOID and OWED -- untouched by E64",
           "engine": os.path.basename(ENGINE), "chance_bpb": CHANCE_BPB,
           "protocol": {"seqlen": SEQLEN, "scored_bytes": SCORED_BYTES, "n_pred": N_PRED},
           "files": {"tern": TERN, "tern_dense": TERN_DENSE, "i8": I8},
           "format_confirmation": {"tern": fmt_t, "i8": fmt_i},
           "ladders": {"tern": {str(k): tern[k] for k in KS},
                       "i8": {str(k): i8[k] for k in KS}, "tern_dense": dense},
           "carve_cost": {"tern": {str(k): ct[k] for k in KS},
                          "i8": {str(k): ci[k] for k in KS},
                          "baselines": {"tern": base_t, "i8": base_i}},
           "G_E64a": ga, "G_E64b": gb, "G_E64d": gd,
           "wall_s": time.time() - t0}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    log("\nwrote %s  (%.0f s)" % (a.out, res["wall_s"]))
    log("NOTE: the ternary ladder is a REPLICATION of E37, not a finding.")


if __name__ == "__main__":
    main()
