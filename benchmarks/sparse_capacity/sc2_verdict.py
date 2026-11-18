#!/usr/bin/env python3
# Sparse Capacity, phase 2 (D-0, D-R): the verdict, computed from the run directories without choosing anything.
#   python benchmarks/sparse_capacity/sc2_verdict.py [results/sparse_capacity] [--dry]
# Rules: docs/research/SPARSE_CAPACITY/F2_BANK_PROTOCOL.md, section 6 (registered before the runs).
import glob, json, math, os, statistics, sys

GATE_ARMS = ("m1", "m4", "m16", "m64")   # m1 = the bank's quantization and gating with no extra capacity
READ_ARMS = ("dense", "dense-fp")        # readings, one seed each
SEEDS = (0, 1)
STEPS = 7324
SIGMA_BPB_FLOOR = 0.005                  # R1 calibration (sigma_seed of BPB), the floor under the measured sigma
EFF_MIN = 0.25                           # effective experts / E below this in any layer: not the declared bank, VOID
R_MIN_ABOVE = 0.05                       # R measurable only if some gate arm beats the strongest control by this much
R_CEIL = 0.95


def load(root, dry):
    out, skipped = {}, []
    for d in sorted(glob.glob(os.path.join(root, "f2_*"))):
        name = os.path.basename(d)
        if name.startswith("f2_dry_") != dry:
            continue
        p, c = os.path.join(d, "result.json"), os.path.join(d, "config.json")
        if not (os.path.exists(p) and os.path.exists(c)):
            skipped.append(f"{name}: no result"); continue
        r, cfg = json.load(open(p)), json.load(open(c))
        if not dry:
            why = [w for w, bad in (("mode", cfg["mode"] != "RUN"), ("not pushed", not cfg["pushed"]),
                                    ("dirty", bool(cfg["dirty_files"])), ("steps", r["steps"] != STEPS)) if bad]
            if why:
                skipped.append(f"{name}: {', '.join(why)}"); continue
        out[(r["arm"], r["seed"])] = r
    return out, skipped


def pooled_sd(vals_by_arm):
    """Pooled seed standard deviation from arms with two seeds each: s^2 = mean over arms of (d^2 / 2)."""
    ds = [(v[0] - v[1]) ** 2 / 2 for v in vals_by_arm.values() if len(v) == 2]
    return math.sqrt(sum(ds) / len(ds)) if ds else float("nan")


def paired_se(R, a, b, key):
    """Standard error of mean(b) - mean(a) from item-level paired scores (same items in every run), averaged over the
    seeds as if fully correlated (the items are the same in both seeds): the floor under the seed sigma."""
    ses = []
    for s in SEEDS:
        if (a, s) in R and (b, s) in R:
            x, y = R[(a, s)]["probes"][key], R[(b, s)]["probes"][key]
            d = [yi - xi for xi, yi in zip(x, y)]
            n = len(d); m = sum(d) / n
            ses.append(math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n) if n > 1 else float("nan"))
    return statistics.mean(ses) if ses else float("nan")


def binom_floor(R, key):
    """Binomial SE of one arm's mean on the item set, p clamped to [1/n, 1 - 1/n] (so a floor at 0 is not sigma 0)."""
    r = next(iter(R.values()))["probes"][key]
    n = len(r); ps = [statistics.mean(x["probes"][key]) for x in R.values()]
    p = min(max(statistics.mean(ps), 1 / n), 1 - 1 / n)
    return math.sqrt(p * (1 - p) / n)


def ratio(a, b):
    return a / b if b > 0 else float("nan")


def nanmax(*v):
    v = [x for x in v if not math.isnan(x)]
    return max(v) if v else float("nan")


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    dry = "--dry" in sys.argv
    root = args[0] if args else os.path.join(os.path.dirname(__file__), "..", "..", "results", "sparse_capacity")
    R, skipped = load(root, dry)
    print(f"runs found: {sorted(R)}{'  (DRY: not a measurement, no reading)' if dry else ''}")
    for s_ in skipped: print("  skipped", s_)
    bad = []
    heads = {r["git_head"] for r in R.values()}
    if len(heads) > 1: bad.append(f"runs from different commits: {heads}")
    for a in GATE_ARMS:
        for s in SEEDS:
            if (a, s) not in R: bad.append(f"missing run {a} seed {s}")
    for (a, s), r in R.items():
        ex = r.get("experts")
        if ex and a != "m1" and ex["eff_min"] < EFF_MIN:
            bad.append(f"{a} s{s}: effective experts {ex['eff_min']:.0%} of E in some layer (< {EFF_MIN:.0%})")
        if r["probes"]["k_unseen"] > 0.01:
            bad.append(f"{a} s{s}: K sanity check on unseen names {r['probes']['k_unseen']:.3f} (> 0.01: scorer bug)")

    def per_arm(f, arms=GATE_ARMS):
        return {a: [f(R[(a, s)]) for s in SEEDS if (a, s) in R] for a in arms}

    mean = lambda d: {a: (statistics.mean(v) if v else float("nan")) for a, v in d.items()}
    bpb = per_arm(lambda r: r["val"]["bpb"])
    mb, mT, m1_, mt, mk, mr = map(mean, (bpb, per_arm(lambda r: r["val"]["bpb_B_tfit"]),
                                         per_arm(lambda r: r["val"]["bpb_B_t1"]), per_arm(lambda r: r["val"]["top1"]),
                                         per_arm(lambda r: r["probes"]["k_primary"]),
                                         per_arm(lambda r: r["probes"]["r_primary"])))
    s_meas = pooled_sd(bpb)
    sb = nanmax(s_meas, SIGMA_BPB_FLOOR)
    sk = nanmax(pooled_sd(per_arm(lambda r: r["probes"]["k_primary"])), paired_se(R, "m1", "m64", "k_items"),
                binom_floor(R, "k_items"))
    sr_seed = pooled_sd(per_arm(lambda r: r["probes"]["r_primary"]))
    sr_pair = paired_se(R, "m1", "m64", "r_primary_items")
    sr = nanmax(sr_seed, sr_pair, binom_floor(R, "r_primary_items"))

    print("\n  arm       active   BPB (seeds)        mean    top1    K prim  R prim  shuffl  lookah  walk   R ood   eff.exp")
    for a in GATE_ARMS + READ_ARMS:
        rs = [R[(a, s)] for s in SEEDS if (a, s) in R]
        if not rs: continue
        m = lambda f: statistics.mean(f(r) for r in rs)
        ood = [k_ for k_ in rs[0]["probes"]["r"] if k_ == "n12_L5"]
        ex = rs[0].get("experts")
        print(f"  {a:<9} {rs[0]['params']['active'] / 1e6:5.1f} M  {' '.join(f'{r['val']['bpb']:.4f}' for r in rs):<18} {m(lambda r: r['val']['bpb']):.4f}  "
              f"{m(lambda r: r['val']['top1']):.4f}  {m(lambda r: r['probes']['k_primary']):.3f}   "
              f"{m(lambda r: r['probes']['r_primary']):.3f}   {m(lambda r: r['probes']['r_primary_shuffled']):.3f}   "
              f"{m(lambda r: r['probes']['r_primary_lookahead']):.3f}   {m(lambda r: r['probes']['r_primary_walk']):.3f}  "
              f"{(m(lambda r: r['probes']['r']['n12_L5']['acc']) if ood else float('nan')):.3f}   "
              f"{'-' if not ex else f'{ex['eff_min']:.2f}'}")
    print(f"\n  sigma BPB: measured {s_meas:.4f}, used {sb:.4f} (floor {SIGMA_BPB_FLOOR}); sigma K {sk:.4f}; "
          f"sigma R {sr:.4f} (seed {sr_seed:.4f}, paired items {sr_pair:.4f})")

    # ---- D-0: from m1 (same quantization, no extra capacity) to m64
    gain, gainT = mb["m1"] - mb["m64"], mT["m1"] - mT["m64"]
    mono = all(mb[GATE_ARMS[i + 1]] <= mb[GATE_ARMS[i]] + sb for i in range(len(GATE_ARMS) - 1))
    print(f"\n  D-0: gain m1 -> m64 {gain:+.4f} BPB = {ratio(gain, sb):.1f} sigma; no step rises by more than 1 sigma: {mono}")
    print(f"       temperature control (half B): gain at T=1 {m1_['m1'] - m1_['m64']:+.4f}, at the fitted T "
          f"{gainT:+.4f} = {ratio(gainT, sb):.1f} sigma; top-1 gain {mt['m64'] - mt['m1']:+.4f}")
    act = {a: R[(a, 0)]["params"]["active"] / 1e6 for a in GATE_ARMS if (a, 0) in R}
    print("       active parameters, router included: " + ", ".join(f"{a} {v:.1f} M" for a, v in act.items()))
    d0 = "PASS" if (mono and gain >= 4 * sb and gainT >= 4 * sb) else "FAIL"
    # ---- K planted positive, then D-R
    k_up = mk["m64"] - mk["m1"]
    k_ok = k_up > 2 * sk
    ctrl = nanmax(*[statistics.mean(R[(a, s)]["probes"][c] for a in ("m1", "m64") for s in SEEDS if (a, s) in R)
                    for c in ("r_primary_lookahead", "r_primary_shuffled", "r_primary_walk")]) if R else float("nan")
    r_meas = (nanmax(mr["m1"], mr["m64"]) >= ctrl + R_MIN_ABOVE) and (mr["m1"] < R_CEIL)
    r_up = mr["m64"] - mr["m1"]
    print(f"\n  K (planted positive): m64 - m1 {k_up:+.3f} = {ratio(k_up, sk):.1f} sigma ({'rises' if k_ok else 'DOES NOT RISE'})")
    print(f"  D-R: R primary m64 - m1 {r_up:+.3f} = {ratio(r_up, sr):.1f} sigma; strongest control {ctrl:.3f}; "
          f"R measurable: {r_meas}")
    if not k_ok:
        dr = "NON LEGGIBILE (the bank does not store: the planted positive did not fire)"
    elif not r_meas:
        dr = "NON MISURABILE (R at floor or ceiling, or not above its strongest control)"
    else:
        dr = "VIVA" if r_up >= 2 * sr else "PIATTA"
    for a in READ_ARMS:
        if (a, 0) in R:
            print(f"  reading {a}: BPB {R[(a, 0)]['val']['bpb']:.4f} (m1 {mb['m1']:.4f})")
    print("\nVALIDITY:", "ok" if not bad else "; ".join(bad))
    if dry:
        print("VERDICT: none (dry run)"); return
    if bad:
        print("VERDICT: VOID"); return
    print(f"VERDICT: D-0 {d0}; D-R {dr}")
    if d0 == "FAIL":
        print("  -> h8 top-128 fails: before closing, one h128 top-8 arm at the m64 size (protocol section 6)")
    elif dr == "VIVA":
        print("  -> PLAN: phases 3 and 4 measure D-R again on larger banks")
    elif dr == "PIATTA":
        print("  -> not shown at this scale (4.6 M core, 60 M tokens): D-R stays open; the method goes on (phase 3),"
              " K and R stay as readings in phases 3 and 4")
    else:
        print("  -> phase 3 may go on for the method; D-R needs a new probe before phase 4 (protocol section 6)")


if __name__ == "__main__":
    main()
