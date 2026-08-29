#!/usr/bin/env python3
"""Turn f1_analyse.json / f1_ann.json into the tables the write-up quotes.

Nothing is computed here that is not already in the JSON -- this is a formatter, so the
write-up cannot drift from the measurement.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

ARMS = ["fit_actw", "fit_plain", "C2_random_proj", "C2b_random_out",
        "C3_wrong_layer_actw", "C3_wrong_layer_plain"]
LABEL = {"fit_actw": "rank-r activation-weighted (the claim)",
         "fit_plain": "rank-r plain SVD",
         "C2_random_proj": "C2 random projection (B random, A solved)",
         "C2b_random_out": "C2b random output subspace (Builder addition)",
         "C3_wrong_layer_actw": "C3 wrong-layer, activation-weighted",
         "C3_wrong_layer_plain": "C3 wrong-layer, plain"}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def get(an, l, arm, r, tgt, field):
    try:
        return an["layers"][str(l)]["arms"][arm][str(r)]["at_recall_target"][tgt][field]
    except (KeyError, TypeError):
        return float("nan")


def main():
    an = load("f1_analyse.json")
    if an is None or not an.get("layers"):
        sys.exit("no f1_analyse.json with layers yet")
    Ls = sorted(int(k) for k in an["layers"])
    RANKS = [int(r) for r in an["pre_registered"]["ranks"]]
    D = an["cost_model"]["D_achieved"]; F = an["cost_model"]["F_achieved"]
    out = []
    P = out.append

    P("## Donor and geometry (ACHIEVED, read off the artefact)\n")
    d = an["donor"]
    for k in ("repo_id", "revision_pinned", "snapshot_leaf", "config_sha256",
              "n_layers_achieved", "D_achieved", "F_achieved", "stored_dtype_achieved",
              "hidden_act_config"):
        P(f"- `{k}` = `{d[k]}`")
    P(f"- layers measured: {len(Ls)} of {d['n_layers_achieved']}")
    P("")

    # ---------------- 1. the donor's own sparsity: the gate on everything else
    P("## 1. The donor's true active fraction `a` (the number the whole brief depends on)\n")
    P("| layer | theta (eps=0.01) | rel-err ACHIEVED | `a` calib | `a` held-out | `a` per-token min/med/max | oracle-|h| `a` | frac(g>0) | dense-gate FFN_active |")
    P("|---|---|---|---|---|---|---|---|---|")
    for l in Ls:
        R = an["layers"][str(l)]
        tr = R["threshold_rule"]
        orc = tr["oracle_magnitude"]["0.01"]["oracle_active_frac"]
        pt = R["a_eval_per_token"]
        P(f"| {l} | {R['theta_primary_achieved']:.4f} | {R['rel_err_at_theta_achieved']:.4f} | "
          f"{R['a_calib_achieved']:.4f} | **{R['a_eval_achieved']:.4f}** | "
          f"{pt['min']:.3f}/{pt['median']:.3f}/{pt['max']:.3f} | {orc:.4f} | "
          f"{R['gate_distribution_eval']['frac_gate_positive']:.4f} | "
          f"{R['baseline_dense_gate_FFN_active']:.4f} |")
    a_all = np.array([an["layers"][str(l)]["a_eval_achieved"] for l in Ls])
    orc_all = np.array([an["layers"][str(l)]["threshold_rule"]["oracle_magnitude"]["0.01"]
                        ["oracle_active_frac"] for l in Ls])
    P("")
    P(f"**mean `a` (held-out, eps=0.01) = {a_all.mean():.4f}**, min {a_all.min():.4f}, "
      f"max {a_all.max():.4f}.  Oracle-|h| mean {orc_all.mean():.4f}.")
    P(f"**Ideal FFN_active with a FREE, PERFECT predictor = mean `a` = {a_all.mean():.4f}.**  "
      f"Dense-gate baseline = {(1+3*a_all.mean())/3:.4f}.  Dense-gate floor = 0.3333.")
    P("")
    P("### eps ladder (the threshold is a CHOICE; here is its sensitivity)\n")
    P("| eps | mean theta | mean `a` calib | mean oracle-|h| `a` |")
    P("|---|---|---|---|")
    for eps in an["pre_registered"]["eps_ladder"]:
        th = [an["layers"][str(l)]["threshold_rule"]["theta_by_eps"][str(eps)] for l in Ls]
        oo = [an["layers"][str(l)]["threshold_rule"]["oracle_magnitude"][str(eps)]
              ["oracle_active_frac"] for l in Ls]
        P(f"| {eps} | {np.mean([t['theta_achieved'] for t in th]):.4f} | "
          f"{np.mean([t['active_frac_achieved'] for t in th]):.4f} | {np.mean(oo):.4f} |")
    tz = [an["layers"][str(l)]["threshold_rule"]["theta_zero"] for l in Ls]
    P(f"| theta=0 (the dReLU-analogue sign rule) | 0.0 | "
      f"{np.mean([t['active_frac_achieved'] for t in tz]):.4f} | "
      f"(rel-err {np.mean([t['rel_err_achieved'] for t in tz]):.4f}) |")
    P("")

    # ---------------- 2. the headline: recall / |S| / FFN_active
    for tgt in ("0.99", "0.999"):
        P(f"## 2. Recall {tgt}: mean |S|/F and FFN_active, per rank, per arm "
          f"(held-out, {len(Ls)} layers)\n")
        P("| arm | " + " | ".join(f"r={r}" for r in RANKS) + " |")
        P("|---|" + "---|" * len(RANKS))
        for arm in ARMS:
            cells = []
            for r in RANKS:
                s = np.array([get(an, l, arm, r, tgt, "S_frac") for l in Ls])
                fa = np.array([get(an, l, arm, r, tgt, "FFN_active") for l in Ls])
                cells.append(f"{np.nanmean(s):.3f} / {np.nanmean(fa):.3f}")
            P(f"| {LABEL[arm]} | " + " | ".join(cells) + " |")
        P("")
        P("cells are **mean |S|/F  /  mean FFN_active** (n=3 accounting, achieved width "
          f"D={D}, F={F}).")
        P("")
        P("### margin over the nulls (mean |S|/F, fitted MINUS null; negative = fitted better)\n")
        P("| comparison | " + " | ".join(f"r={r}" for r in RANKS) + " |")
        P("|---|" + "---|" * len(RANKS))
        for a, b in (("fit_actw", "C2_random_proj"), ("fit_actw", "C2b_random_out"),
                     ("fit_actw", "C3_wrong_layer_actw"), ("fit_actw", "fit_plain"),
                     ("fit_plain", "C2b_random_out")):
            cells = []
            for r in RANKS:
                sa = np.array([get(an, l, a, r, tgt, "S_frac") for l in Ls])
                sb = np.array([get(an, l, b, r, tgt, "S_frac") for l in Ls])
                dd = sa - sb
                win = int((dd < 0).sum())
                cells.append(f"{np.nanmean(dd):+.3f} ({win}/{len(Ls)})")
            P(f"| {a} - {b} | " + " | ".join(cells) + " |")
        P("")
        P("`(k/n)` = layers on which the first arm needs a STRICTLY SMALLER |S|.")
        P("")

    # ---------------- 3. FFN_active at target width
    P("## 3. FFN_active at the brief's target width (D=8192, F=28672), recall 0.99\n")
    P("| arm | " + " | ".join(f"r={r}" for r in RANKS) + " |")
    P("|---|" + "---|" * len(RANKS))
    for arm in ARMS:
        cells = []
        for r in RANKS:
            f3 = np.nanmean([get(an, l, arm, r, "0.99", "FFN_active_target_width") for l in Ls])
            f2 = np.nanmean([get(an, l, arm, r, "0.99", "FFN_active_target_width_n2")
                             for l in Ls])
            cells.append(f"{f3:.3f} / {f2:.3f}")
        P(f"| {LABEL[arm]} | " + " | ".join(cells) + " |")
    P("")
    P("cells are **n=3 / n=2** accounting.  Dense-gate baseline "
      f"{(1+3*a_all.mean())/3:.3f}, dense-gate floor 0.333, brief's 2% target 0.020.")
    P("")

    # ---------------- 4. controls
    P("## 4. Controls\n")
    c1 = [an["layers"][str(l)]["arms"]["C1_full_rank"][str(D)] for l in Ls]
    P(f"**C1 full rank (r = D = {D}) -- A TAUTOLOGY, tests the code path only.**  "
      f"max |ghat-g| over layers = {max(c['max_abs_ghat_minus_g'] for c in c1):.3e}, "
      f"max relative deviation = {max(c['rel_dev'] for c in c1):.3e}.  "
      f"recall at every target = "
      f"{min(c['at_recall_target']['1.0']['recall_micro'] for c in c1):.6f}, "
      f"mean |S|/F at recall 1.0 = "
      f"{np.mean([c['at_recall_target']['1.0']['S_frac'] for c in c1]):.6f} vs true "
      f"a = {a_all.mean():.6f}.")
    P("")
    P("**C4 planted positive -- the instrument must FIRE at the planted rank and NOT before.**\n")
    P("| layer | planted rank ACHIEVED | " + " | ".join(f"rel-dev r={r}" for r in RANKS) +
      " | FIRES |")
    P("|---|---|" + "---|" * len(RANKS) + "---|")
    for l in Ls:
        c4 = an["layers"][str(l)]["arms"]["C4_planted"]
        cells = [f"{c4['by_rank'][str(r)]['rel_dev']:.2e}" for r in RANKS]
        P(f"| {l} | {c4['planted_rank_achieved']} | " + " | ".join(cells) +
          f" | {'YES' if c4['FIRES'] else 'NO'} |")
    P("")
    P("**C4 recall at tau = theta (same active fraction as the real layer):**\n")
    P("| layer | " + " | ".join(f"r={r}" for r in RANKS) + " |")
    P("|---|" + "---|" * len(RANKS))
    for l in Ls:
        c4 = an["layers"][str(l)]["arms"]["C4_planted"]
        P(f"| {l} | " + " | ".join(
            f"{c4['by_rank'][str(r)]['recall_at_tau_eq_theta']:.4f}" for r in RANKS) + " |")
    P("")

    # ---------------- 5. spectrum facts
    P("## 5. What `W_g` actually looks like (distributional facts, reported in their own right)\n")
    P("| layer | eff-rank (participation) actw / plain | rank for 90% energy actw / plain | "
      "rank for 99% actw / plain | cond(W_g) | energy@r=64 actw / plain |")
    P("|---|---|---|---|---|---|")
    for l in Ls:
        sp = an["spectra"][str(l)]
        P(f"| {l} | {sp['actw']['participation_ratio_effective_rank']:.1f} / "
          f"{sp['plain']['participation_ratio_effective_rank']:.1f} | "
          f"{sp['actw']['rank_for_energy_90pct']} / {sp['plain']['rank_for_energy_90pct']} | "
          f"{sp['actw']['rank_for_energy_99pct']} / {sp['plain']['rank_for_energy_99pct']} | "
          f"{sp['plain']['condition_number']:.1f} | "
          f"{sp['energy_captured_actw']['64']:.3f} / {sp['energy_captured_plain']['64']:.3f} |")
    P("")
    if "direct_svd_crosscheck" in an:
        cc = an["direct_svd_crosscheck"]
        P(f"eigh(M^T M) shortcut vs direct SVD, layer {cc['layer']}, k={cc['k']}: "
          f"subspace alignment error {cc['subspace_alignment_err']:.2e}.")
    P("")

    # ---------------- 6. ANN
    ann = load("f1_ann.json")
    if ann and ann.get("layers"):
        AL = sorted(int(k) for k in ann["layers"])
        P("## 6. Amendment 1 -- IVF-PQ over `W_g`'s rows\n")
        pr = ann["pre_registered"]
        P(f"nlist ACHIEVED {pr['nlist_achieved']} (rule `{pr['nlist_rule']}`), "
          f"m={pr['m_subq']} subquantizers of dsub={pr['dsub_achieved']}, "
          f"{2**pr['nbits']} codewords each.  Layers: {len(AL)}.")
        P("")
        P("| nprobe | reachable frac | ceiling recall | mean |S|/F @rec .99 | FFN_active MACs | "
          "FFN_active BYTES | at target width (MACs / bytes) | build s/layer |")
        P("|---|---|---|---|---|---|---|---|")
        for npb in pr["nprobe"]:
            rows = [ann["layers"][str(l)]["by_nprobe"].get(str(npb)) for l in AL]
            rows = [r for r in rows if r]
            if not rows:
                continue
            def m(f, tgt="0.99"):
                return np.nanmean([r["at_recall_target"][tgt][f] for r in rows])
            P(f"| {npb} | {np.mean([r['reachable_frac'] for r in rows]):.3f} | "
              f"{np.mean([r['ceiling_recall_if_all_reached'] for r in rows]):.4f} | "
              f"{m('S_frac'):.3f} | {m('FFN_active'):.3f} | {m('FFN_active_bytes'):.3f} | "
              f"{m('FFN_active_target_width'):.3f} / "
              f"{m('FFN_active_bytes_target_width'):.3f} | "
              f"{np.mean([ann['layers'][str(l)]['build_seconds_achieved'] for l in AL]):.0f} |")
        P("")
        ic = ann["index_cost_model"]
        P(f"PQ reconstruction relative error ACHIEVED (mean over layers): "
          f"{np.mean([ann['layers'][str(l)]['pq_reconstruction_rel_err_achieved'] for l in AL]):.3f}")
        P(f"Index resident bytes at ACHIEVED width: "
          f"{ic['achieved_width_example_nprobe8_s0.05']['index_resident_bytes']/1e6:.1f} MB "
          f"({ic['achieved_width_index_bytes_vs_L3']:.2f} x the 16 MB L3).")
        P(f"Index resident bytes at TARGET width: "
          f"{ic['target_width_example_nprobe8_s0.05']['index_resident_bytes']/1e6:.1f} MB "
          f"({ic['target_width_index_bytes_vs_L3']:.2f} x the 16 MB L3).")
        P("")

    txt = "\n".join(out)
    open(os.path.join(RES, "f1_tables.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
