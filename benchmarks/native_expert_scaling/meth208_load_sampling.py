#!/usr/bin/env python3
"""Frozen-router sequence bootstrap of the METH-207 raw max-load screen."""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth207_causal_mode_bias as M207

P = M207.P
M136 = P.M136
PRIOR = P.DOC / "meth207_causal_mode_bias_result.json"
PRIOR_SHA = "4e7910008f3bc07d80fd48a22c89d2430e3199b5b9116e651623ae7610540558"
ROUTER_SHA = "a5b367cc5902e745d8929d4c206161c698ad5610f9cfb9cae181f1eceadf3701"
REPLICATES = 2048
SEED = 208208
MAX_SECONDS = 20 * 60
CHILD_SLOTS = 1280 * 9


def bootstrap_weights():
    draws = np.random.default_rng(SEED).integers(0, 1024, (REPLICATES, 256))
    weights = np.zeros((REPLICATES, 1024), dtype=np.float32)
    np.add.at(weights, (np.repeat(np.arange(REPLICATES), 256), draws.ravel()), 1)
    assert np.all(weights.sum(1) == 256)
    return weights


def sequence_counts(q, parent, lengths, keys, bias, device):
    assert sum(lengths) == parent.size
    with torch.inference_mode():
        features = torch.from_numpy(q).to(device)
        parents = torch.from_numpy(parent.astype(np.int64)).to(device)
        scores = features @ torch.from_numpy(keys).to(device).T
        grand = parents * 9 + (scores + torch.from_numpy(bias).to(device)[parents]).argmax(-1)
        grand = grand.cpu().numpy()
    seq = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    counts = np.bincount(seq * CHILD_SLOTS + grand,
                         minlength=len(lengths) * CHILD_SLOTS).reshape(len(lengths), CHILD_SLOTS)
    assert np.all(counts.sum(1) == lengths) and int(counts.max()) <= 65535
    return counts.astype(np.uint16)


def load_ratio(counts):
    values = np.asarray(counts, dtype=np.int64).reshape(-1, 1280, 9)
    return 9 * values.max((1, 2)) / values.sum(2).max(1)


def resample_ratios(counts, weights, device):
    matrix = torch.from_numpy(counts.astype(np.float32)).to(device)
    all_ratios = []
    with torch.inference_mode():
        for first in range(0, REPLICATES, 256):
            block = torch.from_numpy(weights[first:first + 256]).to(device)
            sums = (block @ matrix).reshape(-1, 1280, 9)
            child_max = sums.amax((1, 2)).cpu().numpy().astype(np.int64)
            parent_max = sums.sum(2).amax(1).cpu().numpy().astype(np.int64)
            ratios = 9 * child_max / parent_max
            if first == 0:
                for i in range(3):
                    direct = weights[i].astype(np.int64) @ counts.astype(np.int64)
                    assert ratios[i] == load_ratio(direct)[0]
            all_ratios.append(ratios)
    return np.concatenate(all_ratios)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--counts", required=True, type=Path)
    args = ap.parse_args()
    assert not args.out.exists() and not args.counts.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.counts.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage, device = "bindings", None
    cells = {}
    try:
        assert M136.digest(PRIOR) == PRIOR_SHA
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        router = Path(prior["router_artifact"]["path"])
        assert M136.digest(router) == ROUTER_SHA
        with np.load(router, allow_pickle=False) as archive:
            projection_host = archive["projection"].copy()
            keys = archive["content_keys"].copy()
            biases = archive["parent_bias"].copy()
        table = P.M175.load_table()
        parent, child = M136.bind_inputs()
        source_a, source_b, prefixes = M136.load_factor_bank()
        third = M136.load_third()
        assert all(np.array_equal(projection_host[li], item[0].numpy())
                   for li, item in enumerate(third))
        raw_ids, chat, draws = P.D175.load(P.M175.DRAWS_SHA)
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        matches = [i for i in range(torch.cuda.device_count())
                   if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
        assert len(matches) == 1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        P.MAX_SECONDS = M136.MAX_SECONDS = MAX_SECONDS
        stage = "parity"
        model = M136.model_shell(device).eval()
        original_mlps = [layer.mlp for layer in model.model.layers]
        teacher, _ = M136.make_wrappers(model, parent["expert_state"],
            child["expert_state"], source_a, source_b, prefixes, device, "teacher")
        items = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
        with torch.inference_mode():
            reference = [model(torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                device=device)[None], use_cache=False).logits.cpu() for item in items]
        for layer, original in zip(model.model.layers, original_mlps):
            layer.mlp = original
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        wrappers, _ = M136.make_wrappers(model, parent["expert_state"],
            child["expert_state"], source_a, source_b, prefixes, device, "control")
        with torch.inference_mode():
            error = max(float((model(torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                device=device)[None], use_cache=False).logits.cpu().float() - old.float()).abs().max())
                for item, old in zip(items, reference))
        assert error == 0
        projections = [torch.from_numpy(row).to(device) for row in projection_host]
        del parent, child, source_a, source_b, third, reference
        gc.collect()
        counts_by_cell = {}
        stage = "raw_capture_and_reconcile"
        for name, subset in (("fit_raw", draws[:1024]), ("reserved_raw", draws[1024:1280])):
            sequences, _ = P.sequences_for(subset, raw_ids, chat)
            lengths = []
            for seq in sequences:
                modes, mode_sequences, _ = M207.selection_modes([seq], 512, table)
                assert mode_sequences == [1, 0] and not modes.any()
                lengths.append(modes.size)
            data, totals = P.collect(name, sequences, 512, model, wrappers,
                projections, table, started, device)
            expected_cell = prior["cells"][name]
            assert all(totals[field] == expected_cell[field] for field in totals)
            stored = np.empty((24, len(sequences), CHILD_SLOTS), dtype=np.uint16)
            rows = []
            for li, (q, parents) in enumerate(data):
                replay = M207.summarize(q, parents, np.zeros(parents.size, dtype=np.uint8),
                    keys[li], biases[li], totals["structural_selections"], device)
                expected = expected_cell["layers"][li]
                assert replay["gates"] == expected["gates"]
                for field, value in replay.items():
                    if field != "gates":
                        assert math.isclose(value, expected[field], rel_tol=1e-6, abs_tol=1e-6)
                stored[li] = sequence_counts(q, parents, lengths, keys[li], biases[li, 0], device)
                ratio = float(load_ratio(stored[li].sum(0, dtype=np.int64))[0])
                assert ratio == replay["candidate_to_control_max_load_ratio"]
                rows.append({"layer": li, "max_load_ratio": ratio})
                P.budget(started, device)
            cells[name] = {"baseline_reconciled": True, "totals": totals, "layers": rows}
            counts_by_cell[name] = stored
            del data, q, parents, sequences
            gc.collect()
        stage = "draw_bootstrap"
        weights = bootstrap_weights()
        ratios = np.empty((REPLICATES, 24), dtype=np.float64)
        for li in range(24):
            ratios[:, li] = resample_ratios(counts_by_cell["fit_raw"][li], weights, device)
            P.budget(started, device)
            if (li + 1) % 4 == 0:
                print(json.dumps({"bootstrapped_layers": li + 1,
                                  "budget": P.budget(started, device)}), flush=True)
        reserved = np.asarray([r["max_load_ratio"] for r in cells["reserved_raw"]["layers"]])
        worst = ratios.max(1)
        failures = (ratios > 1.25).sum(1)
        observed_worst = float(reserved.max())
        observed_failures = int((reserved > 1.25).sum())
        quantiles = [0.05, 0.5, 0.95, 0.99]
        worst_q = np.quantile(worst, quantiles)
        failures_q = np.quantile(failures, quantiles)
        if observed_worst > worst_q[-1] or observed_failures > failures_q[-1]:
            decision = "beyond_99_percent_fit_sampling_changed_calibration_candidate"
        elif observed_worst <= worst_q[-2] and observed_failures <= failures_q[-2]:
            decision = "fit_sampling_compatible_larger_route_adjudication_needed"
        else:
            decision = "sampling_diagnostic_inconclusive"
        layer_rows = [{"layer": li, "reserved_ratio": float(reserved[li]),
            "bootstrap_quantiles": np.quantile(ratios[:, li], quantiles).tolist(),
            "bootstrap_gate_failure_fraction": float((ratios[:, li] > 1.25).mean()),
            "bootstrap_at_least_reserved_fraction": float((ratios[:, li] >= reserved[li]).mean())}
            for li in range(24)]
        stage = "artifact"
        np.savez(args.counts, fit_raw=counts_by_cell["fit_raw"],
                 reserved_raw=counts_by_cell["reserved_raw"],
                 bootstrap_weights=weights, bootstrap_ratios=ratios)
        assert args.counts.stat().st_size < 1_000_000_000
        with np.load(args.counts, allow_pickle=False) as readback:
            assert np.array_equal(readback["fit_raw"], counts_by_cell["fit_raw"])
            assert np.array_equal(readback["reserved_raw"], counts_by_cell["reserved_raw"])
            assert np.array_equal(readback["bootstrap_weights"], weights)
            assert np.array_equal(readback["bootstrap_ratios"], ratios)
        result = {"experiment": "METH-208-frozen-raw-load-sampling",
            "meth207_result_sha256": PRIOR_SHA, "router_sha256": ROUTER_SHA,
            "source_bank_sha256": M136.EXACT_SHA, "draws_sha256": P.M175.DRAWS_SHA,
            "structural_table_sha256": P.M175.TABLE_SHA,
            "teacher_control_bf16_logit_max_abs_error": error,
            "cells": cells, "sampling": {"unit": "sequence draw with replacement",
                "fit_draws": 1024, "draws_per_replicate": 256, "replicates": REPLICATES,
                "seed": SEED, "shared_weights_across_layers": True, "quantiles": quantiles,
                "worst_layer_ratio_quantiles": worst_q.tolist(),
                "failing_layer_count_quantiles": failures_q.tolist(),
                "all_layer_gate_pass_fraction": float((failures == 0).mean()),
                "at_least_observed_worst_fraction": float((worst >= observed_worst).mean()),
                "at_least_observed_failures_fraction": float((failures >= observed_failures).mean()),
                "joint_at_least_observed_fraction": float(((worst >= observed_worst) &
                                                            (failures >= observed_failures)).mean()),
                "layers": layer_rows},
            "observed": {"worst_layer_ratio": observed_worst,
                         "failing_layer_count": observed_failures},
            "counts_artifact": {"path": str(args.counts.resolve()),
                "sha256": M136.digest(args.counts), "bytes": args.counts.stat().st_size,
                "readback_exact": True}, "decision": decision,
            "runtime": {**P.budget(started, device), "gpu": torch.cuda.get_device_name(device)},
            "scope": "Conditional empirical fit sampling diagnostic; METH-207 stays failed"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.counts.stat().st_size + args.out.stat().st_size < 1_000_000_000
        print(json.dumps({"decision": decision, "sampling": result["sampling"]}), flush=True)
    except BaseException as exc:
        args.out.with_suffix(".failure.json").write_text(json.dumps({"experiment": "METH-208-failure",
            "stage": stage, "error": repr(exc), "completed_cells": cells,
            "seconds": time.monotonic() - started,
            "rss_bytes": psutil.Process().memory_info().rss}, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
