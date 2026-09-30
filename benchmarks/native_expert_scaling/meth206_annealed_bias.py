#!/usr/bin/env python3
"""Reuse METH-204 apparatus with frozen keys and annealed parent biases."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

import meth204_shared_content_keys as P

PRIOR = P.DOC / "meth204_shared_content_keys_result.json"
PRIOR_SHA = "af36a9d1ff119b1407c395c769afcf40b6023ef64ce430ca9a062c89ae59e160"
DIAGNOSTIC = P.DOC / "meth205_deterministic_route_floor_result.json"
DIAGNOSTIC_SHA = "f297107e589c5152e27a6f4facb86f5e5ed3129acd71ede3ff38f73434055745"
ROUTER_SHA = "10a8d695f51bff84bf1fcc0e90971a9fb332bfd3226eb059af0256770ae9d6e1"
TEMPERATURES = (.05, .01, .002, .0004)
STEPS = 100


def hard_summary(scores, parents, bias):
    counts = torch.bincount(parents * 9 + (scores + bias[parents]).argmax(-1),
                            minlength=1280 * 9).reshape(1280, 9)
    parent_counts = torch.bincount(parents, minlength=1280)
    hot = parent_counts >= 250
    shares = counts[hot].max(1).values.float() / parent_counts[hot]
    return {"hot_parents_above_25_percent": int((shares > .25).sum()),
            "worst_hot_parent_share": float(shares.max()),
            "max_load_ratio": float(9 * counts.max() / parent_counts.max()),
            "coverage": int((counts > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists() and not args.router.exists()
    assert P.M136.digest(PRIOR) == PRIOR_SHA
    assert P.M136.digest(DIAGNOSTIC) == DIAGNOSTIC_SHA
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    initial_router = Path(prior["router_artifact"]["path"])
    assert P.M136.digest(initial_router) == ROUTER_SHA
    with np.load(initial_router, allow_pickle=False) as archive:
        saved_keys = archive["content_keys"].copy()
        saved_bias = archive["parent_bias"].copy()
        saved_projection = archive["projection"].copy()
    stages = []

    def fit_annealed(q, parent, layer, device):
        raw_n = prior["cells"]["fit_raw"]["content_selections_per_layer"]
        for name, start, end in (("fit_raw", 0, raw_n),
                                 ("fit_chat", raw_n, parent.size)):
            cell = prior["cells"][name]
            replay = P.summarize_layer(q[start:end], parent[start:end],
                saved_keys[layer], saved_bias[layer],
                cell["structural_selections"], device)
            expected = cell["layers"][layer]
            assert replay["gates"] == expected["gates"]
            for key, value in replay.items():
                if key != "gates":
                    assert math.isclose(value, expected[key], rel_tol=1e-6, abs_tol=1e-6)
        with torch.inference_mode():
            features = torch.from_numpy(q).to(device)
            parents = torch.from_numpy(parent.astype(np.int64)).to(device)
            keys = torch.from_numpy(saved_keys[layer]).to(device)
            bias = torch.from_numpy(saved_bias[layer]).to(device).clone()
            scores = features @ keys.T
            target = torch.bincount(parents, minlength=1280).float()[:, None] / 9
            row = {"layer": layer, "saved_baseline_reconciled": True, "stages": []}
            for temperature in TEMPERATURES:
                for _ in range(STEPS):
                    probabilities = F.softmax((scores + bias[parents]) / temperature, -1)
                    counts = torch.zeros_like(bias)
                    counts.index_add_(0, parents, probabilities)
                    bias += temperature * torch.log((target + 1) / (counts + 1))
                    bias -= bias.mean(-1, keepdim=True)
                assert bool(torch.isfinite(bias).all())
                row["stages"].append({"temperature": temperature,
                                      **hard_summary(scores, parents, bias)})
            stages.append(row)
            terminal = bias.cpu().numpy().copy()
        del features, parents, scores
        torch.cuda.empty_cache()
        return saved_keys[layer].copy(), terminal

    apparatus_out = args.out.with_name(args.out.stem + ".apparatus.json")
    assert not apparatus_out.exists()
    P.fit_one = fit_annealed
    P.MAX_SECONDS = 30 * 60
    original_argv = sys.argv
    sys.argv = [original_argv[0], "--out", str(apparatus_out),
                "--router", str(args.router)]
    try:
        P.main()
    finally:
        sys.argv = original_argv
    result = json.loads(apparatus_out.read_text(encoding="utf-8"))
    with np.load(args.router, allow_pickle=False) as archive:
        assert np.array_equal(archive["projection"], saved_projection)
        assert np.array_equal(archive["content_keys"], saved_keys)
    assert len(stages) == 24
    result["experiment"] = "METH-206-annealed-bias-route-screen"
    result["initial_router_sha256"] = ROUTER_SHA
    result["meth204_result_sha256"] = PRIOR_SHA
    result["meth205_result_sha256"] = DIAGNOSTIC_SHA
    result["fit_rule"] = {"draws": 1024, "reserved_draws": 256,
                          "frozen_keys": True, "temperatures": TEMPERATURES,
                          "steps_per_temperature": STEPS, "initial_bias": "METH-204"}
    result["calibration_stages"] = stages
    result["apparatus_result"] = {"path": str(apparatus_out.resolve()),
                                  "sha256": P.M136.digest(apparatus_out)}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiment": result["experiment"],
                      "decision": result["decision"], "gates": result["gates"]}), flush=True)


if __name__ == "__main__":
    main()
