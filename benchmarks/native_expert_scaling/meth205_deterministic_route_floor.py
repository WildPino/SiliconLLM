#!/usr/bin/env python3
"""Diagnose exact q-state load floors and soft versus hard route counts."""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import meth204_shared_content_keys as P

M136 = P.M136
DOC = P.DOC
PRIOR = DOC / "meth204_shared_content_keys_result.json"
PRIOR_SHA = "af36a9d1ff119b1407c395c769afcf40b6023ef64ce430ca9a062c89ae59e160"
ROUTER_SHA = "10a8d695f51bff84bf1fcc0e90971a9fb332bfd3226eb059af0256770ae9d6e1"
MAX_SECONDS = 20 * 60


def diagnose(q, parent, keys, bias, device):
    with torch.inference_mode():
        features = torch.from_numpy(q).to(device)
        parents = torch.from_numpy(parent.astype(np.int64)).to(device)
        scores = features @ torch.from_numpy(keys).to(device).T
        logits = scores + torch.from_numpy(bias).to(device)[parents]
        hard = torch.bincount(parents * 9 + logits.argmax(-1),
                              minlength=1280 * 9).reshape(1280, 9).cpu().numpy()
        soft = torch.zeros((1280, 9), device=device)
        soft.index_add_(0, parents, F.softmax(logits / .05, dim=-1))
        soft = soft.cpu().numpy()
    totals = np.bincount(parent, minlength=1280)
    assert np.array_equal(hard.sum(1), totals)
    assert np.allclose(soft.sum(1), totals, rtol=1e-4, atol=.05)
    order = np.argsort(parent, kind="stable")
    boundaries = np.r_[0, np.cumsum(totals)]
    hot_rows = []
    for pid in np.flatnonzero(totals >= 250):
        vectors = np.ascontiguousarray(q[order[boundaries[pid]:boundaries[pid + 1]]])
        byte_rows = vectors.view(np.dtype((np.void, vectors.dtype.itemsize * 32))).ravel()
        _, counts = np.unique(byte_rows, return_counts=True)
        largest = int(counts.max())
        amount = int(totals[pid])
        hot_rows.append({"parent": int(pid), "selections": amount,
                         "unique_exact_q_vectors": int(counts.size),
                         "largest_identical_group": largest,
                         "exact_state_floor": largest / amount,
                         "hard_max_share": float(hard[pid].max() / amount),
                         "soft_max_share": float(soft[pid].max() / amount)})
    return {"hot_parents": hot_rows,
            "hot_parent_count": len(hot_rows),
            "parents_above_25_percent": {
                key: sum(row[key] > .25 for row in hot_rows)
                for key in ("exact_state_floor", "hard_max_share", "soft_max_share")},
            "worst_share": {key: max((row[key] for row in hot_rows), default=0)
                            for key in ("exact_state_floor", "hard_max_share", "soft_max_share")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage, device = "bindings", None
    cells = {}
    try:
        assert M136.digest(PRIOR) == PRIOR_SHA
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        router = Path(prior["router_artifact"]["path"])
        assert M136.digest(router) == ROUTER_SHA
        with np.load(router, allow_pickle=False) as archive:
            projections_host = archive["projection"].copy()
            keys = archive["content_keys"].copy()
            biases = archive["parent_bias"].copy()
        assert projections_host.shape == (24, 32, 896)
        assert keys.shape == (24, 9, 32) and biases.shape == (24, 1280, 9)
        table = P.M175.load_table()
        parent, child = M136.bind_inputs()
        source_a, source_b, prefixes = M136.load_factor_bank()
        third = M136.load_third()
        assert all(np.array_equal(projections_host[li], entry[0].numpy())
                   for li, entry in enumerate(third))
        raw_ids, chat, draws = P.D175.load(P.M175.DRAWS_SHA)
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
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
                         device=device)[None], use_cache=False).logits.cpu().float()
                         - old.float()).abs().max()) for item, old in zip(items, reference))
        assert error == 0
        projections = [torch.from_numpy(row).to(device) for row in projections_host]
        del parent, child, source_a, source_b, third, reference
        gc.collect()
        raw, chats = P.sequences_for(draws[:1024], raw_ids, chat)
        stage = "capture_and_reconcile"
        data_by_cell = {}
        for name, sequences in (("fit_raw", raw), ("fit_chat", chats)):
            data, totals = P.collect(name, sequences, 512, model, wrappers,
                                    projections, table, started, device)
            replay = P.evaluate_cell(data, totals, keys, biases, device)
            old = prior["cells"][name]
            assert all(replay[field] == old[field] for field in
                       ("tokens", "windows", "sequences", "structural_selections",
                        "content_selections_per_layer"))
            for row, expected in zip(replay["layers"], old["layers"]):
                assert row["gates"] == expected["gates"]
                for field, value in row.items():
                    if field != "gates":
                        assert math.isclose(value, expected[field], rel_tol=1e-6, abs_tol=1e-6)
            data_by_cell[name] = data
            cells[name] = {"baseline_reconciled": True, "totals": totals, "layers": []}
        stage = "exact_duplicate_and_surrogate_analysis"
        for li in range(24):
            raw_q, raw_parent = data_by_cell["fit_raw"][li]
            chat_q, chat_parent = data_by_cell["fit_chat"][li]
            for name, q, parent_ids in (("fit_raw", raw_q, raw_parent),
                                        ("fit_chat", chat_q, chat_parent),
                                        ("fit_pooled", np.concatenate((raw_q, chat_q)),
                                         np.concatenate((raw_parent, chat_parent)))):
                cells.setdefault(name, {"layers": []})["layers"].append(
                    diagnose(q, parent_ids, keys[li], biases[li], device))
            if (li + 1) % 4 == 0:
                print(json.dumps({"diagnosed_layers": li + 1,
                                  "budget": P.budget(started, device)}), flush=True)
        floor_cells = {name: [li for li, row in enumerate(cells[name]["layers"])
                             if row["parents_above_25_percent"]["exact_state_floor"]]
                       for name in ("fit_raw", "fit_chat")}
        decision = ("recurrent_projected_input_must_change" if any(floor_cells.values())
                    else "hard_load_parent_calibration_candidate")
        result = {"experiment": "METH-205-deterministic-route-floor",
                  "meth204_result_sha256": PRIOR_SHA, "router_sha256": ROUTER_SHA,
                  "source_bank_sha256": M136.EXACT_SHA,
                  "draws_sha256": P.M175.DRAWS_SHA,
                  "structural_table_sha256": P.M175.TABLE_SHA,
                  "teacher_control_bf16_logit_max_abs_error": error,
                  "cells": cells, "raw_chat_layers_with_exact_floor": floor_cells,
                  "decision": decision, "runtime": P.budget(started, device),
                  "scope": "Exact projected-input floor and saved soft/hard route counts only"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < 1_000_000_000
        print(json.dumps({"decision": decision, "exact_floor_layers": floor_cells,
                          "runtime": result["runtime"]}), flush=True)
    except BaseException as exc:
        args.out.with_suffix(".failure.json").write_text(json.dumps({
            "stage": stage, "error": repr(exc), "completed_cells": cells,
            "seconds": time.monotonic() - started,
            "rss_bytes": psutil.Process().memory_info().rss}, indent=2) + "\n",
            encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
