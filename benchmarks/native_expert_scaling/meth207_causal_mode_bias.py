#!/usr/bin/env python3
"""Calibrate two bias banks selected only by the observed leading token."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

import meth204_shared_content_keys as P
import meth206_annealed_bias as A

PREVIOUS = P.DOC / "meth206_annealed_bias_result.json"
PREVIOUS_SHA = "464db925b3a6d2367a6112b4cd1a8d4d41225255c56c3a5a6c8ddc225cf4e6ca"
CHAT_START = 151644


def selection_modes(sequences, width, table):
    """Mode is initialized after reading token zero and persists across windows."""
    parts = []
    counts = [0, 0]
    structural = 0
    for sequence in sequences:
        assert len(sequence) > 0
        mode = int(int(sequence[0]) == CHAT_START)
        counts[mode] += 1
        for first in range(0, len(sequence), width):
            tokens = np.asarray(sequence[first:first + width], dtype=np.int64)
            previous = np.r_[0, tokens[:-1]]
            positions = np.arange(tokens.size, dtype=np.int64)
            shared = (P.R150.shared_mask(tokens, previous, positions, table) |
                      (tokens == 151644) | (tokens == 151645))
            structural += 4 * int(shared.sum())
            parts.append(np.full(4 * int((~shared).sum()), mode, dtype=np.uint8))
    return np.concatenate(parts), counts, structural


def summarize(q, parent, modes, keys, biases, structural, device):
    assert q.shape == (parent.size, 32) and modes.shape == parent.shape
    with torch.inference_mode():
        features = torch.from_numpy(q).to(device)
        parents = torch.from_numpy(parent.astype(np.int64)).to(device)
        mode_ids = torch.from_numpy(modes.astype(np.int64)).to(device)
        centers = torch.from_numpy(keys).to(device)
        offsets = torch.from_numpy(biases).to(device)
        scores = features @ centers.T
        normalized = ((scores - scores.mean(-1, keepdim=True)) /
                      scores.std(-1, unbiased=False, keepdim=True).clamp_min(1e-6))
        local = (scores + offsets[mode_ids, parents]).argmax(-1)
        counts_parent = torch.bincount(parents, minlength=1280).cpu().numpy()
        counts_child = torch.bincount(parents * 9 + local,
                                     minlength=1280 * 9).cpu().numpy()
        hot = counts_parent >= 250
        split = counts_child.reshape(1280, 9)
        share = float(np.max(split[hot].max(1) / counts_parent[hot])) if hot.any() else 0.
        row = {"content_selections": int(parent.size),
               "structural_selections": structural,
               "content_coverage": int((counts_child > 0).sum()),
               "hot_parent_count": int(hot.sum()),
               "candidate_to_control_max_load_ratio":
                   float(9 * counts_child.max() / counts_parent.max()),
               "hot_parent_worst_child_share": share,
               "mean_standardized_score_advantage":
                   float(normalized.gather(1, local[:, None]).mean()),
               "raw_argmax_agreement": float((local == scores.argmax(-1)).float().mean()),
               "bias_max_abs": float(np.abs(biases).max())}
        assert parent.size == int(counts_parent.sum()) == int(counts_child.sum())
        assert all(math.isfinite(value) for value in row.values())
        row["gates"] = {"load_ratio": row["candidate_to_control_max_load_ratio"] <= 1.25,
                        "hot_parent_share": share <= .25,
                        "coverage": row["content_coverage"] >= 4000,
                        "score_advantage": row["mean_standardized_score_advantage"] >= .05,
                        "raw_argmax_agreement": row["raw_argmax_agreement"] >= .15}
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists() and not args.router.exists()
    assert P.M136.digest(PREVIOUS) == PREVIOUS_SHA
    assert P.M136.digest(A.PRIOR) == A.PRIOR_SHA
    prior = json.loads(A.PRIOR.read_text(encoding="utf-8"))
    router = Path(prior["router_artifact"]["path"])
    assert P.M136.digest(router) == A.ROUTER_SHA
    with np.load(router, allow_pickle=False) as archive:
        saved_keys = archive["content_keys"].copy()
        saved_bias = archive["parent_bias"].copy()
        saved_projection = archive["projection"].copy()
    stages = []
    mode_cells = {}
    original_collect = P.collect

    def collect_mode(name, sequences, width, model, wrappers, projections,
                     table, start, device):
        modes, counts, structural = selection_modes(sequences, width, table)
        if name == "fit_raw":
            assert counts == [1024, 0]
        elif name == "fit_chat":
            assert counts == [0, 1024]
        data, totals = original_collect(name, sequences, width, model, wrappers,
                                       projections, table, start, device)
        assert structural == totals["structural_selections"]
        assert modes.size == totals["content_selections_per_layer"]
        mode_cells[name] = {"sequences_by_mode": counts,
                           "content_selections_by_mode": np.bincount(
                               modes, minlength=2).tolist()}
        return [(q, parent, modes) for q, parent in data], totals

    def fit_mode(q, parent, layer, device):
        raw_n = prior["cells"]["fit_raw"]["content_selections_per_layer"]
        terminal = []
        row = {"layer": layer, "saved_baseline_reconciled": True, "modes": []}
        for mode, (name, start, end) in enumerate(
                (("fit_raw", 0, raw_n), ("fit_chat", raw_n, parent.size))):
            cell = prior["cells"][name]
            replay = P.summarize_layer(q[start:end], parent[start:end],
                saved_keys[layer], saved_bias[layer], cell["structural_selections"], device)
            expected = cell["layers"][layer]
            assert replay["gates"] == expected["gates"]
            for key, value in replay.items():
                if key != "gates":
                    assert math.isclose(value, expected[key], rel_tol=1e-6, abs_tol=1e-6)
            with torch.inference_mode():
                features = torch.from_numpy(q[start:end]).to(device)
                parents = torch.from_numpy(parent[start:end].astype(np.int64)).to(device)
                keys = torch.from_numpy(saved_keys[layer]).to(device)
                bias = torch.from_numpy(saved_bias[layer]).to(device).clone()
                scores = features @ keys.T
                target = torch.bincount(parents, minlength=1280).float()[:, None] / 9
                mode_row = {"mode": mode, "stages": []}
                for temperature in A.TEMPERATURES:
                    for _ in range(A.STEPS):
                        probabilities = F.softmax((scores + bias[parents]) / temperature, -1)
                        counts = torch.zeros_like(bias)
                        counts.index_add_(0, parents, probabilities)
                        bias += temperature * torch.log((target + 1) / (counts + 1))
                        bias -= bias.mean(-1, keepdim=True)
                    assert bool(torch.isfinite(bias).all())
                    mode_row["stages"].append({"temperature": temperature,
                        **A.hard_summary(scores, parents, bias)})
                terminal.append(bias.cpu().numpy().copy())
                row["modes"].append(mode_row)
            del features, parents, scores
            torch.cuda.empty_cache()
        stages.append(row)
        return saved_keys[layer].copy(), np.stack(terminal)

    def evaluate(data, totals, centers, biases, device):
        layers = [summarize(q, parent, modes, centers[li], biases[li],
                           totals["structural_selections"], device)
                  for li, (q, parent, modes) in enumerate(data)]
        return {**totals, "layers": layers,
                "all_layer_gates_pass": all(all(row["gates"].values()) for row in layers)}

    apparatus_out = args.out.with_name(args.out.stem + ".apparatus.json")
    assert not apparatus_out.exists()
    P.collect = collect_mode
    P.fit_one = fit_mode
    P.evaluate_cell = evaluate
    P.MAX_SECONDS = 30 * 60
    original_argv = sys.argv
    sys.argv = [original_argv[0], "--out", str(apparatus_out), "--router", str(args.router)]
    try:
        P.main()
    finally:
        sys.argv = original_argv
    result = json.loads(apparatus_out.read_text(encoding="utf-8"))
    with np.load(args.router, allow_pickle=False) as archive:
        assert np.array_equal(archive["projection"], saved_projection)
        assert np.array_equal(archive["content_keys"], saved_keys)
        assert archive["parent_bias"].shape == (24, 2, 1280, 9)
    assert len(stages) == 24
    result["experiment"] = "METH-207-causal-mode-bias-route-screen"
    result["initial_router_sha256"] = A.ROUTER_SHA
    result["meth204_result_sha256"] = A.PRIOR_SHA
    result["meth206_result_sha256"] = PREVIOUS_SHA
    result["fit_rule"] = {"draws": 1024, "reserved_draws": 256,
        "frozen_keys": True, "temperatures": A.TEMPERATURES,
        "steps_per_temperature": A.STEPS, "initial_bias": "METH-204",
        "mode_rule": "leading token == 151644; persist across windows",
        "mode_labels": ["text", "ChatML"]}
    result["mode_cells"] = mode_cells
    result["calibration_stages"] = stages
    result["apparatus_result"] = {"path": str(apparatus_out.resolve()),
                                  "sha256": P.M136.digest(apparatus_out)}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiment": result["experiment"],
                      "decision": result["decision"], "gates": result["gates"]}), flush=True)


if __name__ == "__main__":
    main()
