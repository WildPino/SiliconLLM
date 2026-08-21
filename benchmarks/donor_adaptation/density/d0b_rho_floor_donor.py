#!/usr/bin/env python3
"""D0b -- re-derive the rho-law block floor at DONOR dimensions.

The transplant this probe checks: `neurons_per_48KB_interleaved = 21.33` was MEASURED at our
engine's own D=1536 (`results/d0_layout.json`) and then quoted in a 100B-donor analysis as
"B_block >= 22" without re-derivation.  bytes_per_neuron for the 3-organ interleaved layout is
analytically `3 * d_model * BYTES_PER_WEIGHT`, i.e. it scales with d_model alone -- it does NOT
depend on d_ffn.  So the constant is NOT a universal engine floor; it is a D=1536-specific number
that must be recomputed at every donor's own d_model.

This script does two, and only two, kinds of thing, and never lets them blur:

  (a) ANALYTIC  -- closed-form arithmetic from the layout formula.  No model, no data, no code
                   path beyond arithmetic.  Every number in section 1 and 2 is this.
  (b) MEASURED  -- the actual `d0_layout.py` functions (`run_lengths`, `layout_stats`), imported
                   unmodified and RUN on synthetic masks sized to the donor's own d_ffn.  This is
                   a measurement of the LAYOUT INSTRUMENT at donor scale (its block/run-length
                   accounting code path), not a measurement of any real donor's co-activation
                   structure -- no donor weights or activations are loaded anywhere in this
                   script.  Every number in section 3 is this, and is labelled as such.

PLANTED CONTROL (required before any donor-scale number from the code path is trusted):
  feed `layout_stats` a synthetic donor-scale (N=28672) mask with runs planted at a KNOWN length
  and confirm it reports that exact length, exactly as `d0_layout.py`'s own D=1536 control does,
  at the new scale.  If this does not fire, the code path is not trustworthy at donor scale and
  nothing else in this script means anything.
"""
from __future__ import annotations

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import d0_layout as D0          # REUSE: run_lengths, layout_stats, BYTES_PER_WEIGHT -- unmodified

BYTES_PER_WEIGHT = D0.BYTES_PER_WEIGHT   # 0.5 B/weight, the engine design point -- pulled from D0,
assert BYTES_PER_WEIGHT == 0.5           # not re-typed, so the two probes cannot silently diverge
BLOCK_BYTES = 49152                      # 48 KiB, the rho-law granularity


# ------------------------------------------------------------------ (a) ANALYTIC: sweep over D
def analytic_row_D(d_model: float) -> dict:
    """bytes/neuron and neurons/48KB as a pure function of d_model, both layouts.  ANALYTIC."""
    bpn_organ = d_model * BYTES_PER_WEIGHT
    bpn_interleaved = 3.0 * d_model * BYTES_PER_WEIGHT
    return {
        "d_model": d_model,
        "bytes_per_neuron_per_organ": bpn_organ,
        "neurons_per_48KB_per_organ": BLOCK_BYTES / bpn_organ,
        "bytes_per_neuron_interleaved": bpn_interleaved,
        "neurons_per_48KB_interleaved": BLOCK_BYTES / bpn_interleaved,
    }


# ------------------------------------------------------------------ (a) ANALYTIC: real donor matrix shapes
def analytic_donor_matrices(d_model: int, d_ffn: int, n_heads: int, n_kv_heads: int) -> dict:
    """Llama-3-70B-class geometry, real per-matrix shapes, read as (out_features, in_features).

    row-structured organs (q,k,v,gate,up): the prunable/skippable unit is a ROW; its length is
    in_features.  col-structured organs (o,down): the unit is a COLUMN; its length is
    out_features.  Both q/k/v/gate/up have in_features = d_model, and both o/down have
    out_features = d_model -- so the block granularity is driven by d_model in EVERY organ, never
    by d_ffn, regardless of the GQA head-count asymmetry or the FFN expansion ratio.  This is
    arithmetic, stated and then checked against the actual shapes below (not assumed).
    """
    head_dim = d_model / n_heads
    shapes = {
        "q_proj": {"out": d_model, "in": d_model, "axis": "row"},
        "k_proj": {"out": n_kv_heads * head_dim, "in": d_model, "axis": "row"},
        "v_proj": {"out": n_kv_heads * head_dim, "in": d_model, "axis": "row"},
        "o_proj": {"out": d_model, "in": d_model, "axis": "col"},
        "gate_proj": {"out": d_ffn, "in": d_model, "axis": "row"},
        "up_proj": {"out": d_ffn, "in": d_model, "axis": "row"},
        "down_proj": {"out": d_model, "in": d_ffn, "axis": "col"},
    }
    out = {}
    for organ, sh in shapes.items():
        unit_len = sh["in"] if sh["axis"] == "row" else sh["out"]
        bytes_per_unit = unit_len * BYTES_PER_WEIGHT
        out[organ] = {
            **sh,
            "structured_unit_length": unit_len,
            "bytes_per_unit": bytes_per_unit,
            "units_per_48KB": BLOCK_BYTES / bytes_per_unit,
            "params": sh["out"] * sh["in"],
        }
    # the 3-organ interleaved FFN neuron: gate row (d_model) + up row (d_model)
    # + down COLUMN (d_model, since down_proj out_features = d_model, NOT d_ffn) -- the
    # asymmetric d_ffn cancels out of the per-neuron cost entirely.
    bpn_interleaved = (out["gate_proj"]["bytes_per_unit"] + out["up_proj"]["bytes_per_unit"]
                       + out["down_proj"]["bytes_per_unit"])
    out["_ffn_interleaved_neuron"] = {
        "bytes_per_neuron": bpn_interleaved,
        "neurons_per_48KB": BLOCK_BYTES / bpn_interleaved,
        "note": "d_ffn does not appear in this formula -- only d_model does (gate/up row length "
                "and down column length are both d_model, regardless of the FFN expansion ratio)",
    }
    return out


# ------------------------------------------------------------------ (b) MEASURED: run D0's own code at donor N
def measured_controls_at_donor_scale(d_ffn_donor: int, bytes_per_neuron_organ: float,
                                      bytes_per_neuron_interleaved: float) -> list:
    """Run d0_layout.run_lengths / layout_stats UNMODIFIED on synthetic masks sized to the
    donor's own d_ffn.  This exercises the actual block-accounting code path at the new scale;
    it is a measurement of the INSTRUMENT, not of any real donor activation pattern (no donor
    weights are loaded here -- there are none available in this repo)."""
    rng = np.random.default_rng(0)
    T, N = 256, d_ffn_donor
    p = D0.P_ACTIVE   # 0.10, same design point as D0's own controls, unmodified
    out = []

    # --- known-positive: runs planted at length == the donor's own interleaved block size,
    #     rounded to nearest int; the instrument must recover that EXACT length.
    donor_block = int(round(BLOCK_BYTES / bytes_per_neuron_interleaved))
    B = np.zeros((T, N), dtype=bool)
    nslots = N // (2 * donor_block)
    nruns = max(1, int(round(p * N / donor_block)))
    for t in range(T):
        starts = rng.choice(nslots, nruns, replace=False) * 2 * donor_block
        for s in starts:
            B[t, s:s + donor_block] = True
    st = D0.layout_stats(B, bytes_per_neuron_organ, block_sizes=[donor_block])
    out.append({
        "name": f"planted_runs_{donor_block}_at_donor_Nffn_{N}",
        "expect": f"mean_run_neurons == {donor_block} exactly",
        "measured_mean_run": st["mean_run_neurons"],
        "measured_active_fraction": st["active_fraction"],
        "fired": abs(st["mean_run_neurons"] - donor_block) < 1e-9,
    })

    # --- known-negative: i.i.d. mask at the same density and donor N must give the analytic
    #     geometric-run-length null 1/(1-p), same check as d0_layout.py's own control.
    Bn = rng.random((T, N)) < p
    stn = D0.layout_stats(Bn, bytes_per_neuron_organ, block_sizes=[donor_block])
    analytic = 1.0 / (1.0 - p)
    out.append({
        "name": f"iid_mask_same_density_at_donor_Nffn_{N}",
        "expect": f"mean_run_neurons ~= 1/(1-p) = {analytic:.3f}",
        "measured_mean_run": stn["mean_run_neurons"],
        "analytic": analytic,
        "fired": abs(stn["mean_run_neurons"] - analytic) < 0.05 * analytic,
    })

    # --- the actual point of the probe: run the SAME synthetic i.i.d.-active mask (a
    #     deliberately unstructured null -- no claim about real donor co-activation) through
    #     the block accounting at the DONOR's own 12/4-neuron granularity and report what
    #     changes purely from the coarser block, all else equal.  This isolates the effect of
    #     block size shrinking, holding the activation pattern fixed.
    block_organ = int(round(BLOCK_BYTES / bytes_per_neuron_organ))
    st_organ = D0.layout_stats(Bn, bytes_per_neuron_organ, block_sizes=[block_organ])
    st_interleaved = D0.layout_stats(Bn, bytes_per_neuron_interleaved, block_sizes=[donor_block])
    out.append({
        "name": "iid_mask_block_accounting_at_donor_granularity",
        "note": "SYNTHETIC i.i.d. mask, NOT a real donor co-activation pattern -- isolates the "
                "effect of the donor's coarser block size on the instrument's own accounting",
        "block_size_per_organ_neurons": block_organ,
        "block_skippable_fraction_per_organ": st_organ["blocks"][str(block_organ)]["block_skippable_fraction"],
        "block_size_interleaved_neurons": donor_block,
        "block_skippable_fraction_interleaved": st_interleaved["blocks"][str(donor_block)]["block_skippable_fraction"],
    })
    return out


# ------------------------------------------------------------------ main
def main():
    log = {
        "bytes_per_weight": BYTES_PER_WEIGHT,
        "block_bytes": BLOCK_BYTES,
    }
    import subprocess
    log["git_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=C.REPO).decode().strip()
    log["git_branch"] = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=C.REPO).decode().strip()
    log["command_line"] = "python " + " ".join(sys.argv)
    log["source_constant_checked"] = {
        "quoted_as": "B_block >= 22 (transplanted from D=1536 without re-derivation)",
        "source_artefact": "results/d0_layout.json",
        "source_field": "neurons_per_48KB_interleaved",
        "source_value": 21.333333333333332,
        "source_d_model": 1536,
    }

    # ---------------------------------------------------------- (a) analytic sweep over D
    D_SWEEP = [1536, 2048, 4096, 5120, 8192]
    print("== (a) ANALYTIC: bytes_per_neuron / neurons_per_48KB vs d_model ==", flush=True)
    log["analytic_sweep_over_d_model"] = []
    for D in D_SWEEP:
        row = analytic_row_D(D)
        log["analytic_sweep_over_d_model"].append(row)
        print(f"  D={D:5d}  bpn_organ={row['bytes_per_neuron_per_organ']:7.1f}B "
              f"neurons/48KB_organ={row['neurons_per_48KB_per_organ']:6.2f}  "
              f"bpn_interleaved={row['bytes_per_neuron_interleaved']:7.1f}B "
              f"neurons/48KB_interleaved={row['neurons_per_48KB_interleaved']:6.2f}", flush=True)

    # sanity: D=1536 row must reproduce the D0 artefact's own numbers exactly (bit-for-bit
    # arithmetic identity, not a re-measurement -- both are the same closed form).
    d1536 = log["analytic_sweep_over_d_model"][0]
    d0_artefact = json.load(open(os.path.join(C.RESULTS, "d0_layout.json")))
    check = {
        "name": "reproduces_d0_layout_json_at_D1536",
        "d0_artefact_neurons_per_48KB_interleaved": d0_artefact["neurons_per_48KB_interleaved"],
        "this_scripts_value": d1536["neurons_per_48KB_interleaved"],
        "d0_artefact_neurons_per_48KB_per_organ": d0_artefact["neurons_per_48KB_per_organ"],
        "this_scripts_value_per_organ": d1536["neurons_per_48KB_per_organ"],
        "fired": (d1536["neurons_per_48KB_interleaved"] == d0_artefact["neurons_per_48KB_interleaved"]
                  and d1536["neurons_per_48KB_per_organ"] == d0_artefact["neurons_per_48KB_per_organ"]),
    }
    print(f"  [check] {check['name']}: {'FIRED' if check['fired'] else 'DID NOT FIRE'}", flush=True)
    log["reproduces_d0_artefact_check"] = check

    # ---------------------------------------------------------- (a) analytic Llama-3-70B-class shapes
    print("\n== (a) ANALYTIC: Llama-3-70B-class real per-matrix shapes (d_model=8192, "
          "d_ffn=28672, GQA 64/8 heads) ==", flush=True)
    donor = analytic_donor_matrices(d_model=8192, d_ffn=28672, n_heads=64, n_kv_heads=8)
    log["analytic_donor_70b_matrices"] = donor
    for organ, v in donor.items():
        if organ.startswith("_"):
            continue
        print(f"  {organ:10s} axis={v['axis']:3s} shape=({v['out']:.0f}x{v['in']:.0f}) "
              f"unit_len={v['structured_unit_length']:.0f} bytes/unit={v['bytes_per_unit']:.0f} "
              f"units/48KB={v['units_per_48KB']:.2f}", flush=True)
    fi = donor["_ffn_interleaved_neuron"]
    print(f"  [interleaved FFN neuron] bytes/neuron={fi['bytes_per_neuron']:.0f} "
          f"neurons/48KB={fi['neurons_per_48KB']:.3f}", flush=True)

    # cross-check against the task's own hand arithmetic: at D=8192 (uniform d_model, ignore
    # d_ffn) the formula 32768/D predicts 4.0 -- confirm the real-shape calc lands on the same
    # number even though gate/up/down are NOT equal-shaped matrices.
    cross = {
        "name": "real_donor_shapes_match_uniform_D_formula",
        "uniform_formula_32768_over_D_at_8192": 32768.0 / 8192.0,
        "real_shape_calc_neurons_per_48KB_interleaved": fi["neurons_per_48KB"],
        "fired": abs(fi["neurons_per_48KB"] - 32768.0 / 8192.0) < 1e-9,
    }
    print(f"  [check] {cross['name']}: {'FIRED' if cross['fired'] else 'DID NOT FIRE'}  "
          f"({cross['real_shape_calc_neurons_per_48KB_interleaved']:.4f} vs "
          f"{cross['uniform_formula_32768_over_D_at_8192']:.4f})", flush=True)
    log["real_shapes_match_uniform_formula_check"] = cross

    # ---------------------------------------------------------- (a) hand-derivable synthetic control
    # D for which the by-hand answer is trivial: D=1024 -> bpn_organ=512B, neurons/48KB_organ=96;
    # bpn_interleaved=1536B, neurons/48KB_interleaved=32.
    hand = analytic_row_D(1024)
    hand_check = {
        "name": "hand_derivable_D1024_control",
        "by_hand": {"bytes_per_neuron_per_organ": 512.0, "neurons_per_48KB_per_organ": 96.0,
                    "bytes_per_neuron_interleaved": 1536.0, "neurons_per_48KB_interleaved": 32.0},
        "code_output": hand,
        "fired": (hand["bytes_per_neuron_per_organ"] == 512.0
                  and hand["neurons_per_48KB_per_organ"] == 96.0
                  and hand["bytes_per_neuron_interleaved"] == 1536.0
                  and hand["neurons_per_48KB_interleaved"] == 32.0),
    }
    print(f"\n  [control] D=1024 by-hand vs code: "
          f"{'FIRED' if hand_check['fired'] else 'DID NOT FIRE'}", flush=True)
    log["hand_derivable_control"] = hand_check

    # ---------------------------------------------------------- (b) MEASURED: D0 code path at donor N
    print("\n== (b) MEASURED: d0_layout.py's own run_lengths/layout_stats, run unmodified on "
          "SYNTHETIC masks sized to the donor's own d_ffn=28672 ==", flush=True)
    bpn_organ_8192 = donor["gate_proj"]["bytes_per_unit"]        # 4096.0
    bpn_interleaved_8192 = donor["_ffn_interleaved_neuron"]["bytes_per_neuron"]  # 12288.0
    controls = measured_controls_at_donor_scale(28672, bpn_organ_8192, bpn_interleaved_8192)
    log["measured_controls_donor_scale"] = controls
    all_fired = True
    for c in controls:
        if "fired" in c:
            print(f"  {c['name']:48s} -> {'FIRED' if c['fired'] else 'DID NOT FIRE'}", flush=True)
            all_fired = all_fired and c["fired"]
        else:
            print(f"  {c['name']:48s} block_skip@organ={c['block_skippable_fraction_per_organ']:.4f} "
                  f"(bs={c['block_size_per_organ_neurons']}) "
                  f"block_skip@interleaved={c['block_skippable_fraction_interleaved']:.4f} "
                  f"(bs={c['block_size_interleaved_neurons']})", flush=True)
    log["all_measured_controls_fired"] = all_fired
    if not all_fired:
        print("\n!!! a planted control at donor scale DID NOT FIRE -- STOP, do not trust the "
              "donor-scale numbers above the failing control !!!", flush=True)

    # ---------------------------------------------------------- summary against the transplant
    log["verdict"] = {
        "transplanted_constant": "B_block >= 22 (from D=1536, neurons_per_48KB_interleaved=21.33)",
        "correct_constant_at_D8192": fi["neurons_per_48KB"],
        "ratio_D1536_over_D8192": d1536["neurons_per_48KB_interleaved"] / fi["neurons_per_48KB"],
        "interleave_assumption_survives_donor_asymmetry": True,
        "interleave_survival_reason": "bytes_per_neuron for gate/up (row, in_features=d_model) "
            "and down (col, out_features=d_model) are ALL d_model-driven; d_ffn cancels out of "
            "the formula entirely regardless of the FFN expansion ratio (28672/8192=3.5x here).",
        "interleave_physical_caveat": "down_proj's 'column' is contiguous only if down_proj is "
            "stored TRANSPOSED (shape [d_ffn, d_model] instead of [d_model, d_ffn]) so that row j "
            "of the transposed store equals column j of down_proj. This requirement is "
            "scale-independent -- it exists identically at D=1536 -- and is a pre-existing "
            "assumption of the interleaved layout, not a new problem introduced by donor scale.",
    }
    print(f"\n== VERDICT: transplanted B_block>=22 vs re-derived at D=8192 = "
          f"{fi['neurons_per_48KB']:.2f} (ratio {log['verdict']['ratio_D1536_over_D8192']:.2f}x) ==",
          flush=True)

    print("wrote", C.dump("d0b_rho_floor_donor.json", log))


if __name__ == "__main__":
    main()
