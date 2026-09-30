#!/usr/bin/env python3
"""Increase only raw bias-calibration support; preserve the passing ChatML bank."""
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

import meth208_load_sampling as S
import meth206_annealed_bias as A

P, M207, M136 = S.P, S.M207, S.M136
DIAGNOSTIC = P.DOC / "meth208_load_sampling_result.json"
DIAGNOSTIC_SHA = "7d069774448ed3fd92e6cc412d38ec99471644671d2b984ac16a211640d77c2d"
MAX_SECONDS = 30 * 60


def partition(draws):
    fit = draws[:1024] + draws[1280:3328]
    old = draws[1024:1280]
    fresh = draws[3328:3840]
    ids = [set(d["update"] for d in group) for group in (fit, old, fresh)]
    assert [len(group) for group in ids] == [3072, 256, 512]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == set(range(1, 3841))
    return fit, old, fresh


def setup(device, projection_host):
    parent, child = M136.bind_inputs()
    source_a, source_b, prefixes = M136.load_factor_bank()
    third = M136.load_third()
    assert all(np.array_equal(projection_host[li], item[0].numpy())
               for li, item in enumerate(third))
    model = M136.model_shell(device).eval()
    originals = [layer.mlp for layer in model.model.layers]
    teacher, _ = M136.make_wrappers(model, parent["expert_state"], child["expert_state"],
        source_a, source_b, prefixes, device, "teacher")
    items = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
    with torch.inference_mode():
        reference = [model(torch.as_tensor(item["prompt_ids"], dtype=torch.long,
            device=device)[None], use_cache=False).logits.cpu() for item in items]
    for layer, original in zip(model.model.layers, originals):
        layer.mlp = original
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    wrappers, _ = M136.make_wrappers(model, parent["expert_state"], child["expert_state"],
        source_a, source_b, prefixes, device, "control")
    with torch.inference_mode():
        error = max(float((model(torch.as_tensor(item["prompt_ids"], dtype=torch.long,
            device=device)[None], use_cache=False).logits.cpu().float() - old.float()).abs().max())
            for item, old in zip(items, reference))
    assert error == 0
    return model, wrappers, [torch.from_numpy(row).to(device) for row in projection_host], error


def check_row(row, expected):
    assert row["gates"] == expected["gates"]
    for field, value in row.items():
        if field != "gates":
            assert math.isclose(value, expected[field], rel_tol=1e-6, abs_tol=1e-6)


def evaluate(data, modes, totals, keys, biases, device):
    rows = [M207.summarize(q, parent, modes, keys[li], biases[li],
            totals["structural_selections"], device) for li, (q, parent) in enumerate(data)]
    return {**totals, "layers": rows,
            "all_layer_gates_pass": all(all(row["gates"].values()) for row in rows)}


def reuse_chat(cell, biases):
    reused = json.loads(json.dumps(cell))
    for li, row in enumerate(reused["layers"]):
        row["bias_max_abs"] = float(np.abs(biases[li]).max())
    reused["evidence_reused_from_sha256"] = S.PRIOR_SHA
    reused["bias_max_abs_recomputed_for_candidate"] = True
    return reused


def fit_raw(q, parent, keys, initial, device):
    with torch.inference_mode():
        features = torch.from_numpy(q).to(device)
        parents = torch.from_numpy(parent.astype(np.int64)).to(device)
        scores = features @ torch.from_numpy(keys).to(device).T
        bias = torch.from_numpy(initial).to(device).clone()
        target = torch.bincount(parents, minlength=1280).float()[:, None] / 9
        stages = []
        for temperature in A.TEMPERATURES:
            for _ in range(A.STEPS):
                probabilities = F.softmax((scores + bias[parents]) / temperature, -1)
                counts = torch.zeros_like(bias)
                counts.index_add_(0, parents, probabilities)
                bias += temperature * torch.log((target + 1) / (counts + 1))
                bias -= bias.mean(-1, keepdim=True)
            assert bool(torch.isfinite(bias).all())
            stages.append({"temperature": temperature, **A.hard_summary(scores, parents, bias)})
        return bias.cpu().numpy().copy(), stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists() and not args.router.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.router.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage, device = "bindings", None
    cells, mode_cells = {}, {}
    try:
        assert M136.digest(S.PRIOR) == S.PRIOR_SHA
        assert M136.digest(DIAGNOSTIC) == DIAGNOSTIC_SHA
        assert M136.digest(A.PRIOR) == A.PRIOR_SHA
        prior = json.loads(S.PRIOR.read_text(encoding="utf-8"))
        initial_record = json.loads(A.PRIOR.read_text(encoding="utf-8"))
        old_router = Path(prior["router_artifact"]["path"])
        initial_router = Path(initial_record["router_artifact"]["path"])
        assert M136.digest(old_router) == S.ROUTER_SHA
        assert M136.digest(initial_router) == A.ROUTER_SHA
        with np.load(old_router, allow_pickle=False) as archive:
            projection = archive["projection"].copy()
            keys = archive["content_keys"].copy()
            biases = archive["parent_bias"].copy()
        old_biases = biases.copy()
        with np.load(initial_router, allow_pickle=False) as archive:
            initial_bias = archive["parent_bias"].copy()
            assert np.array_equal(archive["projection"], projection)
            assert np.array_equal(archive["content_keys"], keys)
        assert M136.digest(P.P.MANIFEST) == P.P.MANIFEST_SHA
        manifest = json.loads(P.P.MANIFEST.read_text(encoding="utf-8"))
        assert len(manifest["items"]) == 24
        for item in manifest["items"]:
            assert M136.M17.sha(item["text"].encode()) == item["text_sha256"]
            assert M136.M17.sha(np.asarray(item["document_ids"], dtype=np.int32).tobytes()) == item["document_ids_sha256"]
        table = P.M175.load_table()
        raw_ids, chat, draws = P.D175.load(P.M175.DRAWS_SHA)
        fit_draws, old_draws, fresh_draws = partition(draws)
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
        model, wrappers, projections, parity_error = setup(device, projection)
        gc.collect()
        P.budget(started, device)

        def collect(name, sequences, width):
            modes, mode_sequences, structural = M207.selection_modes(sequences, width, table)
            data, totals = P.collect(name, sequences, width, model, wrappers,
                projections, table, started, device)
            assert modes.size == totals["content_selections_per_layer"]
            assert structural == totals["structural_selections"]
            mode_cells[name] = {"sequences_by_mode": mode_sequences,
                              "content_selections_by_mode": np.bincount(modes, minlength=2).tolist()}
            return data, modes, totals

        stage = "expanded_raw_fit_capture"
        sequences, _ = P.sequences_for(fit_draws, raw_ids, chat)
        data, modes, totals = collect("fit_raw", sequences, 512)
        assert mode_cells["fit_raw"]["sequences_by_mode"] == [3072, 0]
        original_n = prior["cells"]["fit_raw"]["content_selections_per_layer"]
        stages = []
        stage = "expanded_raw_bias_fit"
        for li, (q, parents) in enumerate(data):
            row = M207.summarize(q[:original_n], parents[:original_n], modes[:original_n],
                keys[li], old_biases[li], prior["cells"]["fit_raw"]["structural_selections"], device)
            check_row(row, prior["cells"]["fit_raw"]["layers"][li])
            biases[li, 0], layer_stages = fit_raw(q, parents, keys[li], initial_bias[li], device)
            stages.append({"layer": li, "old_fit_baseline_reconciled": True, "stages": layer_stages})
            P.budget(started, device)
            if (li + 1) % 4 == 0:
                print(json.dumps({"fitted_layers": li + 1, "budget": P.budget(started, device)}), flush=True)
        assert np.array_equal(biases[:, 1], old_biases[:, 1])
        np.savez(args.router, projection=projection, content_keys=keys, parent_bias=biases)
        with np.load(args.router, allow_pickle=False) as archive:
            assert np.array_equal(archive["projection"], projection)
            assert np.array_equal(archive["content_keys"], keys)
            assert np.array_equal(archive["parent_bias"], biases)
            assert np.array_equal(archive["parent_bias"][:, 1], old_biases[:, 1])
        stage = "fit_screen"
        cells["fit_raw"] = evaluate(data, modes, totals, keys, biases, device)
        cells["fit_chat"] = reuse_chat(prior["cells"]["fit_chat"], biases)
        del data, q, parents, sequences
        gc.collect()
        fit_pass = all(cells[n]["all_layer_gates_pass"] for n in ("fit_raw", "fit_chat"))
        old_pass, fresh_pass, source_pass = False, False, False
        print(json.dumps({"fit_pass": fit_pass, "budget": P.budget(started, device)}), flush=True)
        if fit_pass:
            stage = "old_reserved_screen"
            sequences, _ = P.sequences_for(old_draws, raw_ids, chat)
            data, modes, totals = collect("old_reserved_raw", sequences, 512)
            for li, (q, parents) in enumerate(data):
                row = M207.summarize(q, parents, modes, keys[li], old_biases[li],
                    totals["structural_selections"], device)
                check_row(row, prior["cells"]["reserved_raw"]["layers"][li])
            cells["old_reserved_raw"] = evaluate(data, modes, totals, keys, biases, device)
            cells["old_reserved_chat"] = reuse_chat(prior["cells"]["reserved_chat"], biases)
            del data, q, parents, sequences
            gc.collect()
            old_pass = all(cells[n]["all_layer_gates_pass"] for n in ("old_reserved_raw", "old_reserved_chat"))
            print(json.dumps({"old_reserved_pass": old_pass, "budget": P.budget(started, device)}), flush=True)
        if old_pass:
            stage = "fresh_reserved_screen"
            raw, chats = P.sequences_for(fresh_draws, raw_ids, chat)
            for name, sequences in (("fresh_reserved_raw", raw), ("fresh_reserved_chat", chats)):
                data, modes, totals = collect(name, sequences, 512)
                cells[name] = evaluate(data, modes, totals, keys, biases, device)
                del data
                gc.collect()
                P.budget(started, device)
            fresh_pass = all(cells[n]["all_layer_gates_pass"] for n in ("fresh_reserved_raw", "fresh_reserved_chat"))
            print(json.dumps({"fresh_reserved_pass": fresh_pass, "budget": P.budget(started, device)}), flush=True)
        if fresh_pass:
            stage = "source_separated_screen"
            sequences = [item["document_ids"] for item in manifest["items"]]
            for width in (128, 512):
                name = f"source_w{width}"
                data, modes, totals = collect(name, sequences, width)
                cells[name] = evaluate(data, modes, totals, keys, biases, device)
                del data
                gc.collect()
                P.budget(started, device)
            source_pass = all(cells[f"source_w{w}"]["all_layer_gates_pass"] for w in (128, 512))
        stage = "report"
        result = {"experiment": "METH-209-expanded-raw-bias",
            "meth207_result_sha256": S.PRIOR_SHA, "meth208_result_sha256": DIAGNOSTIC_SHA,
            "initial_router_sha256": A.ROUTER_SHA, "chat_router_source_sha256": S.ROUTER_SHA,
            "source_bank_sha256": M136.EXACT_SHA, "draws_sha256": P.M175.DRAWS_SHA,
            "structural_table_sha256": P.M175.TABLE_SHA, "source_manifest_sha256": P.P.MANIFEST_SHA,
            "teacher_control_bf16_logit_max_abs_error": parity_error,
            "fit_rule": {"raw_draw_ranges": [[0,1024],[1280,3328]], "raw_fit_draws": 3072,
                "old_reserved_range": [1024,1280], "fresh_reserved_range": [3328,3840],
                "chat_bias_byte_identical": True, "frozen_projection_keys": True,
                "temperatures": A.TEMPERATURES, "steps_per_temperature": A.STEPS},
            "calibration_stages": stages, "mode_cells": mode_cells, "cells": cells,
            "gates": {"fit": fit_pass, "old_reserved": old_pass, "fresh_reserved": fresh_pass,
                      "source_separated": source_pass},
            "router_artifact": {"path": str(args.router.resolve()), "sha256": M136.digest(args.router),
                "bytes": args.router.stat().st_size, "readback_exact": True},
            "ledger": prior["ledger"],
            "decision": "route_screen_pass_native_cost_next" if source_pass else "route_screen_fail",
            "runtime": {**P.budget(started, device), "gpu": torch.cuda.get_device_name(device)},
            "scope": "Route-only screen; old chat evidence reused by exact bank identity"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size + args.router.stat().st_size < 1_000_000_000
        print(json.dumps({"decision": result["decision"], "gates": result["gates"],
                          "runtime": result["runtime"]}), flush=True)
    except BaseException as exc:
        args.out.with_suffix(".failure.json").write_text(json.dumps({"experiment": "METH-209-failure",
            "stage": stage, "error": repr(exc), "completed_cells": cells,
            "seconds": time.monotonic() - started, "rss_bytes": psutil.Process().memory_info().rss},
            indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
