#!/usr/bin/env python3
"""S1 tables -- every number in the write-up is emitted from the run JSONs by THIS script.

Nothing here recomputes science; it only formats what `s1_sparsity_bpb.py` already wrote to a
completed artefact.  Run it after the runs finish and paste the output into the probe report,
so the prose cannot drift from the measurements.

Usage:  python s1_tables.py [key ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

ARM_LABEL = {
    "A": "A published (top-p L1 on i)",
    "B": "B ours (eps-threshold on W_g x)",
    "C": "C NULL (random)",
    "D": "D oracle |h_i| (registered)",
    "D2": "D2 oracle |h_i|*||W_d[:,i]|| (Builder)",
}
THRESH = (0.005, 0.01, 0.05, 0.10, 0.50)


def load(key):
    p = os.path.join(RESULTS, f"s1_run_{key}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, n=6):
    return "n/a" if x is None else f"{x:.{n}f}"


def curve_rows(d):
    """(p, achieved, {arm: rec}) sorted by achieved sparsity ascending."""
    rows = []
    for ps, rec in d["results"].items():
        arms = {k: v for k, v in rec.items() if k in ARM_LABEL}
        if not arms:
            continue
        ach = arms["A"]["achieved"]["aggregate"]
        rows.append((float(ps), ach, arms, rec.get("sparsity_match_max_abs_dev")))
    return sorted(rows, key=lambda r: r[1])


def crossing(rows, arm, thr):
    """ACHIEVED sparsity at which |delta| first exceeds thr, linearly interpolated.

    Reported as an INTERVAL as well as a point: the point is an interpolation between two
    measured p values and is not itself a measurement.
    """
    prev = None
    for _, ach, arms, _ in rows:
        if arm not in arms:
            continue
        d = arms[arm]["delta_bpb"]
        if d > thr:
            if prev is None:
                return None, (None, ach)
            a0, d0 = prev
            t = (thr - d0) / (d - d0) if d != d0 else 0.0
            return a0 + t * (ach - a0), (a0, ach)
        prev = (ach, d)
    return None, (rows[-1][1] if rows else None, None)


def main(keys):
    if not keys:
        keys = [os.path.basename(p)[len("s1_run_"):-len(".json")]
                for p in sorted(glob.glob(os.path.join(RESULTS, "s1_run_*.json")))]
    loaded = [(k, load(k)) for k in keys]
    loaded = [(k, d) for k, d in loaded if d]

    for key, d in loaded:
        m = d["donor"]
        print(f"\n\n### {key}  --  {m['repo_id']} @ {m['revision_achieved']}")
        print(f"- params ACHIEVED {m['n_params_achieved']:,} | layers {m['n_layers_achieved']} | "
              f"D {m['d_model_achieved']} | F {m['d_ffn_achieved']} | act {m['hidden_act_achieved']} | "
              f"dtype {m['dtype_achieved']}")
        print(f"- baseline BPB **{d['baseline_bpb']:.9f}** "
              f"(marginal slice-resampling SE {d['baseline_slice_se']:.4f})")
        print(f"- slice: {d['slice']['part']} {d['slice']['n_seq']}x{d['slice']['seq_len']} "
              f"seed {d['slice']['seed']}, {d['slice']['total_scored_bytes']:,} scored bytes, "
              f"ids_sha256 {d['slice']['ids_sha256'][:16]}")
        print(f"- peak RSS {d.get('peak_rss_gb')} GB | elapsed {d.get('elapsed_s')} s")

        rows = curve_rows(d)
        arms = [a for a in ARM_LABEL if any(a in r[2] for r in rows)]

        print("\n**BPB against ACHIEVED FFN sparsity** (delta vs unmodified baseline; "
              "paired SE in brackets)\n")
        print("| nominal p | ACHIEVED sparsity | " + " | ".join(arms) + " | arms matched? |")
        print("|---|---|" + "---|" * (len(arms) + 1))
        for ps, ach, ar, match in rows:
            cells = []
            for a in arms:
                if a not in ar:
                    cells.append("--")
                    continue
                r = ar[a]
                cells.append(f"{r['delta_bpb']:+.6f} [{r.get('delta_se_PAIRED', float('nan')):.6f}]")
            mm = "yes" if (match is not None and match == 0.0) else f"DEV {match}"
            print(f"| {ps} | {ach:.6f} | " + " | ".join(cells) + f" | {mm} |")

        print("\n**Absolute BPB**\n")
        print("| ACHIEVED sparsity | " + " | ".join(arms) + " |")
        print("|---|" + "---|" * len(arms))
        for ps, ach, ar, _ in rows:
            print(f"| {ach:.6f} | " + " | ".join(
                (f"{ar[a]['bpb']:.6f}" if a in ar else "--") for a in arms) + " |")

        print("\n**Marginal vs paired SE** (why the paired one is the right bar)\n")
        print("| ACHIEVED sparsity | arm | delta | PAIRED SE | marginal SE | ratio | "
              "per-seq corr w/ baseline |")
        print("|---|---|---|---|---|---|---|")
        for ps, ach, ar, _ in rows:
            for a in arms:
                if a not in ar:
                    continue
                r = ar[a]
                sp, sm = r.get("delta_se_PAIRED"), r.get("bpb_marginal_se")
                ratio = (sm / sp) if (sp and sp > 0) else None
                print(f"| {ach:.6f} | {a} | {r['delta_bpb']:+.6f} | {fmt(sp)} | {fmt(sm, 4)} | "
                      f"{'n/a' if ratio is None else f'{ratio:.0f}x'} | "
                      f"{fmt(r.get('per_seq_corr_with_baseline'), 5)} |")

        print("\n**Where each arm crosses a BPB budget** (ACHIEVED sparsity; "
              "interpolated point, and the measured bracket it lies in)\n")
        print("| budget | " + " | ".join(arms) + " |")
        print("|---|" + "---|" * len(arms))
        for thr in THRESH:
            cells = []
            for a in arms:
                pt, (lo, hi) = crossing(rows, a, thr)
                if pt is None and lo is not None and hi is None:
                    cells.append(f"never (>{lo:.3f} tested)")
                elif pt is None:
                    cells.append("already at 0")
                else:
                    cells.append(f"{pt:.4f}  ({lo:.4f}-{hi:.4f})")
            lab = "sigma_seed 0.005" if thr == 0.005 else f"{thr:g}"
            print(f"| {lab} | " + " | ".join(cells) + " |")

        c = d.get("controls", {})
        print("\n**Controls**\n")
        print("| control | measured | verdict |")
        print("|---|---|---|")
        if "C1_IDENTITY" in c:
            x = c["C1_IDENTITY"]
            print(f"| C1 IDENTITY (p=1.0) | BPB {x['bpb_masked']:.9f} vs unmodified "
                  f"{x['bpb_unmodified']:.9f}, delta {x['delta']:+.2e}, bitwise_equal="
                  f"{x['bitwise_equal']}, achieved sparsity {x['achieved_sparsity']:.9f}, "
                  f"max\\|h_masked-h\\| {x['h_mask_max_abs_dev_achieved']} | **{x['verdict']}** |")
        if "C2_PLANTED" in c:
            x = c["C2_PLANTED"]
            print(f"| C2 PLANTED (drop top-\\|h\\| at p={x['p_budget']} budget, achieved "
                  f"{x['achieved_sparsity']:.6f}) | BPB {x['bpb']:.6f}, delta "
                  f"{x['delta_vs_baseline']:+.6f} = {x['ratio_to_sigma_seed']:.0f}x sigma_seed; "
                  f"vs arm A same budget {x['delta_vs_armA_same_budget']:+.6f} | "
                  f"**{x['verdict']}** |")
        # C3 and C4 from the curve itself
        worst = rows[-1] if rows else None
        if worst and "C" in worst[2]:
            dC = worst[2]["C"]["delta_bpb"]
            dA = worst[2]["A"]["delta_bpb"]
            dB = worst[2]["B"]["delta_bpb"] if "B" in worst[2] else float("nan")
            print(f"| C3 NULL FIRES (at max ACHIEVED sparsity {worst[1]:.6f}) | "
                  f"C {dC:+.6f} vs A {dA:+.6f} vs B {dB:+.6f} | "
                  f"**{'FIRES' if (dC > dA and dC > dB) else 'DID NOT FIRE'}** |")
        devs = [r[3] for r in rows if r[3] is not None]
        print(f"| C4 ACHIEVED matched across arms | max across-arm deviation in achieved "
              f"sparsity = {max(devs) if devs else 'n/a'} | "
              f"**{'PASS' if devs and max(devs) == 0.0 else 'CHECK'}** |")

        pr = d.get("prereg_check", {})
        if pr:
            print(f"\n- pre-registration on remote: {pr.get('brief_on_remote_achieved')}, "
                  f"remote==worktree: {pr.get('brief_remote_equals_worktree')}, "
                  f"sha256 {str(pr.get('brief_sha256_on_remote'))[:16]}, "
                  f"commit `{str(pr.get('brief_commit_on_remote'))[:60]}`, "
                  f"run started {pr.get('run_started_utc')}")

    # ------------------------------------------------------------------ the scale table
    if len(loaded) > 1:
        print("\n\n### SCALE -- ACHIEVED sparsity tolerated at a fixed BPB budget, by donor size\n")
        print("| donor | params | F | baseline BPB | " +
              " | ".join(f"arm A @ +{t:g}" for t in THRESH) + " |")
        print("|---|---|---|---|" + "---|" * len(THRESH))
        for key, d in sorted(loaded, key=lambda kv: kv[1]["donor"]["n_params_achieved"]):
            rows = curve_rows(d)
            cells = []
            for t in THRESH:
                pt, (lo, hi) = crossing(rows, "A", t)
                cells.append("--" if pt is None else f"{pt:.4f}")
            print(f"| {key} | {d['donor']['n_params_achieved']:,} | {d['donor']['d_ffn_achieved']} "
                  f"| {d['baseline_bpb']:.6f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")])


# ================================================================== token-budget sensitivity
# Added after the F1 audit found a(eps) moving by 0.044 between a 256- and a 1024-token budget
# -- larger than the cross-size effects the scale arm is trying to resolve.  The resolution of
# THIS instrument is therefore measured and published beside every number it produces.
#
# Nothing here runs a model.  It resamples the PER-SEQUENCE nats already stored in the run JSON,
# so it costs no forward passes and it is exact for each subset it evaluates.
import math

import numpy as np


def _bpb(nats, byts, idx):
    return float(nats[idx].sum() / (math.log(2.0) * byts[idx].sum()))


def sensitivity(key, budgets_seq=(2, 4, 8, 16, 24), n_draw=400, seed=5, thr=0.01):
    d = load(key)
    if not d or "baseline_per_seq_nats" not in d:
        print(f"\n(no per-sequence nats stored for {key}; token-sensitivity unavailable)")
        return
    byts = np.array(d["slice_bytes_per_seq"])
    nb = np.array(d["baseline_per_seq_nats"])
    n = len(byts)
    seqlen = d["slice"]["seq_len"]
    rows = curve_rows(d)
    arms = [a for a in ARM_LABEL if all(a in r[2] for r in rows)]
    have = [(ach, {a: np.array(ar[a]["per_seq_nats"]) for a in arms})
            for _, ach, ar, _ in rows if all("per_seq_nats" in ar[a] for a in arms)]
    if not have:
        print(f"\n(no per-arm per-sequence nats for {key})")
        return
    rng = np.random.default_rng(seed)

    print(f"\n\n### TOKEN-BUDGET SENSITIVITY -- the resolution of this instrument ({key})\n")
    print(f"Resampled from the stored per-sequence nats of the pinned {n}x{seqlen} slice; "
          f"{n_draw} random subsets per budget, drawn WITHOUT replacement. "
          f"No forward pass is re-run, so each subset value is exact for that subset.\n")

    print("**(a) BPB delta of each arm, by token budget** (mean +- sd across subsets)\n")
    print("| tokens | n_seq | " + " | ".join(arms) + " |")
    print("|---|---|" + "---|" * len(arms))
    # report at the densest p that is still benign-ish: pick the mid achieved sparsity
    mid = have[len(have) // 2]
    ach_mid, nats_mid = mid
    for ns in budgets_seq:
        if ns > n:
            continue
        vals = {a: [] for a in arms}
        for _ in range(n_draw if ns < n else 1):
            idx = rng.choice(n, ns, replace=False) if ns < n else np.arange(n)
            b0 = _bpb(nb, byts, idx)
            for a in arms:
                vals[a].append(_bpb(nats_mid[a], byts, idx) - b0)
        cells = [f"{np.mean(vals[a]):+.4f} +- {np.std(vals[a]):.4f}" for a in arms]
        print(f"| {ns*seqlen} | {ns} | " + " | ".join(cells) + " |")
    print(f"\n(at ACHIEVED sparsity {ach_mid:.4f})\n")

    print("**(b) The headline quantity: ACHIEVED sparsity at which arm A costs "
          f"+{thr:g} BPB** -- spread across subsets\n")
    print("| tokens | n_seq | mean | sd | p05 | p95 | spread (p95-p05) |")
    print("|---|---|---|---|---|---|---|")
    res = {}
    for ns in budgets_seq:
        if ns > n:
            continue
        xs = []
        for _ in range(n_draw if ns < n else 1):
            idx = rng.choice(n, ns, replace=False) if ns < n else np.arange(n)
            b0 = _bpb(nb, byts, idx)
            prev = None
            hit = None
            for ach, nats in have:
                dd = _bpb(nats["A"], byts, idx) - b0
                if dd > thr:
                    if prev is None:
                        hit = ach
                    else:
                        a0, d0 = prev
                        hit = a0 + (thr - d0) / (dd - d0) * (ach - a0) if dd != d0 else a0
                    break
                prev = (ach, dd)
            if hit is not None:
                xs.append(hit)
        if not xs:
            continue
        xs = np.array(xs)
        res[ns] = xs
        print(f"| {ns*seqlen} | {ns} | {xs.mean():.4f} | {xs.std():.4f} | "
              f"{np.quantile(xs,0.05):.4f} | {np.quantile(xs,0.95):.4f} | "
              f"{np.quantile(xs,0.95)-np.quantile(xs,0.05):.4f} |")

    if res:
        big = max(k for k in res if k < n) if any(k < n for k in res) else None
        if big:
            print(f"\n**RESOLUTION:** at the {big*seqlen}-token budget the headline sparsity moves "
                  f"+-{res[big].std():.4f} (1 sd) purely from which sequences were drawn. "
                  f"A cross-size difference smaller than about {2*res[big].std():.3f} in achieved "
                  f"sparsity is NOT resolvable at that budget.")
        print("\n**Calibration against the published trend** (arXiv:2509.00454 [T] Table 1): "
              "S_inter 46.54% (0.5B) -> 50.49% (1.5B) -> 71.66% (14B). "
              "The 0.5B->1.5B step is ~0.040; the 1.5B->14B step is ~0.212.")


if __name__ == "__main__" and "--sens" in sys.argv:
    for k in [a for a in sys.argv[1:] if not a.startswith("--")] or ["qwen2.5-1.5b"]:
        sensitivity(k)
