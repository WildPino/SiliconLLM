#!/usr/bin/env python3
"""D4 completion -- finishes the sweep the killed session left at 6/9 points, relabels the
ROW-structured (gate_proj) points that were mislabelled INCONCLUSIVE, adds the conditioning
diagnostic (condition number / lambda_max / lambda_min / count-below-Tikhonov of H_reg[S,S])
that was briefed twice and still missing, and adds a wrong-layer-H arm to the Hessian ablation.

APPENDS to results/d4_reconstruction.json -- loads the existing log in full (preserving the
prereg, all four fired controls, the 6 already-recorded sweep points, and the rank diagnostics
exactly as they are) and only adds/patches what this session is responsible for. Does NOT
re-run gate_proj or o_proj's sweep points (they already exist and re-running risks introducing
noise into numbers that are already final) and does NOT touch d4_reconstruction.py's own logic
(that file's ZERO_BY_CONSTRUCTION override, added after the killed run captured its JSON, is
correct for any *future* full run; this script reproduces the same relabelling directly on the
already-written records instead of re-deriving them).

Reuses, unmodified: common.py, d1_pruning.py (Snapshot, _struct_axis, paired_se, N_SEQ/SEQ_LEN/
SEED), d1b_organ_sweep_completion.py (block_size_for), d4_reconstruction.py (select_zero_blocks,
surviving_units, capture_organ_inputs_multi, reconstruct_organ_level, bootstrap_recovery,
classify_recovery, check_hessian_rank_by_layer -- every piece of apparatus, none re-derived).
"""
from __future__ import annotations

import sys, os, json, math, time, gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import d1_pruning as D1
import d1b_organ_sweep_completion as D1B
import d4_reconstruction as D4   # sets torch.set_num_threads(10) on import, same as the killed run

OUTP = D4.OUTP
LEVELS = D4.LEVELS
LAMBDA_SCALE = D4.LAMBDA_SCALE

WRONG_LAYER_OFFSET = 14   # n_layers // 2 -- a perfect derangement on 28 layers, no layer maps to itself


def H_of(X: torch.Tensor) -> torch.Tensor:
    return X.double().T @ X.double()


def conditioning_for_layer(H_raw: torch.Tensor, S: torch.Tensor, lam: float):
    """condition number / lambda_max / lambda_min of H_reg[S,S] = H[S,S] + lam*I, plus the
    count of eigendirections of the RAW (unregularised) H[S,S] that fall below lam -- i.e. the
    directions where the Tikhonov term, not the calibration data, sets the effective scale.
    H_reg and H_raw share eigenvectors (adding lam*I only shifts eigenvalues by +lam), so a
    single eigvalsh of the raw block gives both the regularised spectrum (shift by lam) and the
    raw-vs-lam comparison, without a second D^3 pass."""
    if S.numel() == 0:
        return None
    A = H_raw.index_select(0, S).index_select(1, S)
    eig_raw = torch.linalg.eigvalsh(A).clamp_min(0)
    eig_reg = eig_raw + lam
    lambda_max = float(eig_reg.max())
    lambda_min = float(eig_reg.min())
    cond = lambda_max / lambda_min if lambda_min > 0 else float("inf")
    n_below = int((eig_raw < lam).sum())
    return {
        "lambda_used": lam,
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "condition_number": cond,
        "n_eigendirections_below_tikhonov_lambda": n_below,
        "n_surviving": int(S.numel()),
    }


def main():
    run_t0 = time.time()
    print("== D4 finish: loading existing log ==", flush=True)
    log = json.load(open(OUTP))
    assert log["organ_sweep"], "expected the killed session's 6 points to already be present"
    n_existing = len(log["organ_sweep"])
    print(f"  loaded log with {n_existing} existing sweep points, "
          f"{len(log['controls'])} control groups", flush=True)

    m, tok = C.load_model()
    A = C.arch(m)
    all_layers = list(range(A["n_layers"]))
    assert A == log["arch"], f"arch mismatch vs existing log: {A} != {log['arch']}"

    ids_eval, byts_eval, meta_eval = C.get_slice(tok, "heldout", D4.N_EVAL, D4.SEQ_LEN_EVAL, D4.SEED_EVAL)
    assert meta_eval["ids_sha256"] == log["eval_slice"]["ids_sha256"], "eval slice drifted from the logged one"
    ids_cal, byts_cal, meta_cal = C.get_slice(tok, "calib", D4.N_CAL, D4.SEQ_LEN_CAL, D4.SEED_CAL)
    assert meta_cal["ids_sha256"] == log["calib_slice"]["ids_sha256"], "calib slice drifted from the logged one"

    t0 = time.time()
    base_bpb, base_per = C.bpb(m, ids_eval, byts_eval, return_per_seq=True)
    print(f"baseline BPB {base_bpb:.6f} (logged: {log['baseline_bpb']:.6f})  {time.time()-t0:.0f}s", flush=True)
    assert base_bpb == log["baseline_bpb"], "baseline BPB does not reproduce the logged value bit-exactly"

    def evaluate(tag, **extra):
        b, per = C.bpb(m, ids_eval, byts_eval, return_per_seq=True)
        se = D1.paired_se(base_per, per, byts_eval)
        delta = b - base_bpb
        rec = {"tag": tag, "bpb": b, "delta": delta, "paired_se": se, **extra}
        print(f"  {tag:46s} BPB {b:.5f}  d={delta:+.5f} +-{se:.5f}", flush=True)
        return rec, per

    block_sizes = {}
    for organ in D4.ORGANS:
        bs, _ = D1B.block_size_for(m, organ)
        assert bs == log["block_sizes"][organ], f"{organ} block size drifted"
        block_sizes[organ] = bs

    snap = D1.Snapshot()
    d1 = json.load(open(os.path.join(C.RESULTS, "d1_pruning.json")))

    # ============================================================ activation capture (down_proj + o_proj, ONE forward pass -- same combined-hook trick as the killed run)
    print("\n== capturing calibration activations (down_proj + o_proj) ==", flush=True)
    t0 = time.time()
    X_cal = D4.capture_organ_inputs_multi(m, ids_cal, ["down_proj", "o_proj"], all_layers)
    X_cal_down, X_cal_o = X_cal["down_proj"], X_cal["o_proj"]
    T_cal = ids_cal.shape[0] * ids_cal.shape[1]
    assert T_cal == log["calib_tokens_T"], "calib token count drifted"
    print(f"  combined calib capture: {time.time()-t0:.0f}s, T={T_cal}", flush=True)

    print("  forming down_proj real-H (once, reused for item1/3/4)...", flush=True)
    t0 = time.time()
    H_real_down = {L: H_of(X_cal_down[L]) for L in all_layers}
    H_real_down["_N"] = T_cal
    print(f"  done: {time.time()-t0:.0f}s", flush=True)
    del X_cal_down

    print("  forming o_proj real-H (once, reused for item3)...", flush=True)
    t0 = time.time()
    H_real_o = {L: H_of(X_cal_o[L]) for L in all_layers}
    H_real_o["_N"] = T_cal
    print(f"  done: {time.time()-t0:.0f}s", flush=True)
    del X_cal_o
    del X_cal
    gc.collect()

    down_masks = {}   # level -> mask_by_layer, reused by item1, item3, item4

    # ============================================================ ITEM 1: down_proj @ 25/50/75%
    print("\n== item 1: down_proj sweep (25/50/75%) ==", flush=True)
    new_points = []
    for level in LEVELS:
        mask = D4.select_zero_blocks(m, all_layers, "down_proj", level, block_sizes["down_proj"])
        down_masks[level] = mask
        n_units = C.get_linear(m, 0, "down_proj").weight.shape[1]
        zeroed_total = sum(len(mask[L]["idx_zero_blocks"]) * block_sizes["down_proj"] for L in all_layers)
        total_units = n_units * len(all_layers)
        zero_frac_achieved = zeroed_total / total_units

        dn = dn_se = None
        for rec in d1["organ_sweep"]:
            if rec.get("organ") == "down_proj" and rec.get("mode") == "block_structured" and abs(rec.get("level", -1) - level) < 1e-9:
                dn, dn_se = rec["delta"], rec["paired_se"]; break
        assert dn is not None, f"no D1 naive record for down_proj@{level}"

        Wnew, rmeta = D4.reconstruct_organ_level(m, all_layers, "down_proj", block_sizes["down_proj"], mask, H_real_down, LAMBDA_SCALE)
        lam_mean = float(np.mean([rmeta[L]["lambda"] for L in all_layers]))
        any_fb = any(rmeta[L]["fell_back"] for L in all_layers)

        for L in all_layers:
            W = C.get_linear(m, L, "down_proj").weight
            snap.take((L, "down_proj"), W)
            W.copy_(Wnew[L].to(W.dtype))
        rec, per = evaluate(f"down_proj/reconstructed/{int(level*100)}%", organ="down_proj", level=level,
                             axis="col", zero_frac_requested=level, zero_frac_achieved=zero_frac_achieved,
                             zero_frac_divergence_pct=abs(zero_frac_achieved - level) * 100,
                             delta_naive_d1=dn, delta_naive_d1_paired_se=dn_se,
                             lambda_mean=lam_mean, any_fell_back=any_fb)
        snap.restore(m)

        recovery = 1.0 - rec["delta"] / dn
        rec_se, _ = D4.bootstrap_recovery(base_per, per, byts_eval, dn, seed=11 + n_existing + len(new_points))
        rec["recovery"] = recovery
        rec["recovery_se"] = rec_se
        rec["recovery_verdict"] = D4.classify_recovery(recovery, rec_se)
        new_points.append(rec)
        print(f"    recovery={recovery:+.3f} +-{rec_se:.3f}  [{rec['recovery_verdict']}]  "
              f"zero_frac_div={rec['zero_frac_divergence_pct']:.3f}%", flush=True)

    log["organ_sweep"] = log["organ_sweep"] + new_points
    C.dump("d4_reconstruction.json", log)

    # ============================================================ ITEM 2: relabel gate_proj points
    print("\n== item 2: relabelling gate_proj (ROW-structured) points ==", flush=True)
    n_relabelled = 0
    for rec in log["organ_sweep"]:
        if rec.get("organ") == "gate_proj" and rec.get("axis") == "row":
            old_verdict = rec.get("recovery_verdict")
            rec["recovery_verdict"] = "ZERO_BY_CONSTRUCTION"
            rec["verdict_override_reason"] = (
                f"Empirical classify_recovery() output was {old_verdict!r} (the [-2SE,+2SE] band spans 0 "
                "-> INCONCLUSIVE by the generic rule); overridden because PREREG.derivation_note_mask_geometry "
                "proves recovery=0 algebraically for ROW-structured organs, independent of H's content -- "
                "every kept row has full support (S=all columns) and every dropped row has none, so the "
                "row-separable closed form returns exactly W or exactly 0 regardless of the calibration data. "
                "This is a theorem being confirmed, not a datum about the reconstruction method: INCONCLUSIVE "
                "would mischaracterise a confirmed algebraic fact as unresolved measurement noise and wrongly "
                "invite a reader to think more samples could resolve it."
            )
            rec["excluded_from_summary_stats"] = True
            n_relabelled += 1
            print(f"  {rec['tag']}: {old_verdict!r} -> ZERO_BY_CONSTRUCTION (excluded from summary stats)", flush=True)
    assert n_relabelled == 3, f"expected 3 gate_proj points, relabelled {n_relabelled}"
    C.dump("d4_reconstruction.json", log)

    # ============================================================ column-structured summary (excludes ROW-structured gate_proj by construction)
    col_points = [r for r in log["organ_sweep"] if r.get("axis") == "col"]
    col_recoveries = [r["recovery"] for r in col_points]
    log["organ_sweep_summary"] = {
        "note": "Summary over COLUMN-structured organs only (o_proj, down_proj). gate_proj's "
                "ROW-structured points are excluded by construction -- their recovery=0 is a "
                "theorem (PREREG.derivation_note_mask_geometry), not a datum, and averaging them "
                "in would silently drag any 'does reconstruction work' summary toward a result "
                "that was true before the run started.",
        "n_column_structured_points": len(col_points),
        "n_row_structured_excluded": sum(1 for r in log["organ_sweep"] if r.get("excluded_from_summary_stats")),
        "recovery_mean": float(np.mean(col_recoveries)),
        "recovery_min": float(np.min(col_recoveries)),
        "recovery_max": float(np.max(col_recoveries)),
    }
    C.dump("d4_reconstruction.json", log)

    # ============================================================ ITEM 4: wrong-layer-H ablation arm, down_proj@50%
    print(f"\n== item 4: wrong-layer-H ablation arm, down_proj@50% (offset={WRONG_LAYER_OFFSET}) ==", flush=True)
    n_layers = len(all_layers)
    H_wrong = {L: H_real_down[(L + WRONG_LAYER_OFFSET) % n_layers] for L in all_layers}
    H_wrong["_N"] = T_cal
    mask_50 = down_masks[0.50]
    Wnew, wmeta = D4.reconstruct_organ_level(m, all_layers, "down_proj", block_sizes["down_proj"], mask_50, H_wrong, LAMBDA_SCALE)
    lam_mean_wrong = float(np.mean([wmeta[L]["lambda"] for L in all_layers]))
    any_fb_wrong = any(wmeta[L]["fell_back"] for L in all_layers)
    for L in all_layers:
        W = C.get_linear(m, L, "down_proj").weight
        snap.take((L, "down_proj"), W)
        W.copy_(Wnew[L].to(W.dtype))
    r, _ = evaluate("ABLATION wrong_layer_H down_proj@50%", organ="down_proj", arm="wrong_layer_H",
                     lambda_mean=lam_mean_wrong, any_fell_back=any_fb_wrong,
                     construction="layer L reconstructed using layer (L+14 mod 28)'s real H -- real "
                                  "spectrum and real column means (unlike shuffled_H), wrong layer "
                                  "correspondence (unlike real_H). offset=14 is a perfect derangement "
                                  "on 28 layers: no layer ever uses its own H.")
    snap.restore(m)
    dn_50 = log["controls"]["hessian_ablation"]["delta_naive_d1"]
    r["recovery_point"] = 1.0 - r["delta"] / dn_50
    log["controls"]["hessian_ablation"]["wrong_layer_H"] = r
    log["controls_summary"]["hessian_ablation_arms"]["wrong_layer_H"] = r["recovery_point"]
    log["rank_diagnostics"]["down_proj_wrong_layer_H"] = {
        "label": "down_proj_wrong_layer_H", "method": "derived_from_real_H_permuted",
        "note": "identical matrices to down_proj_real_H, only reassigned to a different layer index "
                "-- rank per matrix is therefore identical to down_proj_real_H's (already exact-checked, "
                "ratio 1.0 at every layer), so re-running eigvalsh would recompute a number already known.",
        "ratio_min": log["rank_diagnostics"]["down_proj_real_H"]["ratio_min"],
        "ratio_mean": log["rank_diagnostics"]["down_proj_real_H"]["ratio_mean"],
        "ratio_max": log["rank_diagnostics"]["down_proj_real_H"]["ratio_max"],
    }
    print(f"  recovery: real_H={log['controls']['hessian_ablation']['real_H']['recovery_point']:.3f}  "
          f"identity_H={log['controls']['hessian_ablation']['identity_H']['recovery_point']:.3f}  "
          f"shuffled_H={log['controls']['hessian_ablation']['shuffled_H']['recovery_point']:.3f}  "
          f"wrong_layer_H={r['recovery_point']:.3f}", flush=True)
    C.dump("d4_reconstruction.json", log)

    # ============================================================ ITEM 3: conditioning diagnostic (report only -- no gate)
    print("\n== item 3: conditioning diagnostic (report-only, no gating threshold) ==", flush=True)
    cond_diag = {}
    for organ in ("o_proj", "down_proj"):
        H_by = H_real_o if organ == "o_proj" else H_real_down
        bs = block_sizes[organ]
        by_level = {}
        for level in LEVELS:
            if organ == "down_proj":
                mask = down_masks[level]
            else:
                mask = D4.select_zero_blocks(m, all_layers, organ, level, bs)
            per_layer = {}
            for L in all_layers:
                W = C.get_linear(m, L, organ).weight
                n_in = W.shape[1]
                S = D4.surviving_units(mask[L], n_in, bs)
                Hraw = H_by[L]
                N = H_by["_N"]
                lam = LAMBDA_SCALE * float(torch.trace(Hraw)) / max(1, N)
                d = conditioning_for_layer(Hraw, S, lam)
                if d is not None:
                    per_layer[str(L)] = d
            conds = [v["condition_number"] for v in per_layer.values() if math.isfinite(v["condition_number"])]
            nbelows = [v["n_eigendirections_below_tikhonov_lambda"] for v in per_layer.values()]
            worst_L = max(per_layer, key=lambda k: per_layer[k]["condition_number"]) if per_layer else None
            by_level[str(level)] = {
                "per_layer": per_layer,
                "condition_number_min": min(conds) if conds else None,
                "condition_number_mean": float(np.mean(conds)) if conds else None,
                "condition_number_max": max(conds) if conds else None,
                "worst_layer_by_condition_number": worst_L,
                "n_eigendirections_below_tikhonov_lambda_min": min(nbelows) if nbelows else None,
                "n_eigendirections_below_tikhonov_lambda_mean": float(np.mean(nbelows)) if nbelows else None,
                "n_eigendirections_below_tikhonov_lambda_max": max(nbelows) if nbelows else None,
            }
            n_surv_repr = per_layer[next(iter(per_layer))]["n_surviving"] if per_layer else 0
            print(f"  {organ}@{int(level*100)}%: condition_number mean={by_level[str(level)]['condition_number_mean']:.3e} "
                  f"max={by_level[str(level)]['condition_number_max']:.3e} (worst layer {worst_L})  "
                  f"n_below_tikhonov mean={by_level[str(level)]['n_eigendirections_below_tikhonov_lambda_mean']:.1f}/"
                  f"{n_surv_repr}", flush=True)
        cond_diag[organ] = by_level
    cond_diag["gate_proj"] = {
        "note": "N/A by construction: gate_proj is ROW-structured, reconstruct_row_axis_level performs "
                "no solve and forms no H (see PREREG.derivation_note_mask_geometry) -- there is no "
                "H_reg[S,S] to condition."
    }
    cond_diag["method_note"] = (
        "condition_number/lambda_max/lambda_min are of H_reg[S,S] = H[S,S] + lambda*I (the matrix "
        "actually inverted by the solve). n_eigendirections_below_tikhonov_lambda counts eigenvalues "
        "of the RAW (unregularised) H[S,S] that are < lambda -- directions where the Tikhonov term "
        "dominates the effective scale rather than the calibration data. Reported for visibility only, "
        "per the brief: NOT used to gate/abort any point (no threshold has been justified)."
    )
    log["conditioning_diagnostics"] = cond_diag
    C.dump("d4_reconstruction.json", log)

    log["done_finish"] = True
    log["finish_wallclock_seconds"] = time.time() - run_t0
    C.dump("d4_reconstruction.json", log)
    print(f"\nwrote {OUTP}", flush=True)
    print(f"finish wallclock: {log['finish_wallclock_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
