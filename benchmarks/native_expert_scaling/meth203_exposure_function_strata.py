#!/usr/bin/env python3
"""Stratify E12800 child function spread by prior training exposure."""

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

import meth201_child_function_spread as P201


P = P201.P
M136, M151, M156, M175, M176 = P201.M136, P201.M151, P201.M156, P201.M175, P201.M176
DOC = P.DOC
BASELINE = DOC / "meth201_child_function_spread_result.json"
BASELINE_SHA = "1b12c7db45edd66c3f0e59c26ebc9c88501c66da3fddf5a3989db17d45833a24"
BINS = (("low", 0, 2048), ("middle", 2048, 8192), ("high", 8192, None))
MAX_SECONDS = 20 * 60
MAX_RSS = 20 * (1 << 30)
MAX_GPU = int(10.5 * (1 << 30))
MAX_DISK = 1_000_000_000


def budget(start, device):
    row = {"seconds": time.monotonic() - start,
           "rss_bytes": psutil.Process().memory_info().rss,
           "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if row["seconds"] > MAX_SECONDS or row["rss_bytes"] > MAX_RSS or row[
            "gpu_peak_allocated_bytes"] > MAX_GPU:
        raise RuntimeError(f"METH-203 resource stop: {row}")
    return row


def exposure_summary(counts):
    assert counts.shape == (24, 1280, 10)
    assert np.all(counts[:, :, 0] == 0)
    content = counts[:, :, 1:].reshape(-1)
    active = content[content > 0]
    assert active.size > 0
    return {"total_content_selections": int(content.sum()),
            "all_content_slots": int(content.size),
            "active_content_slots": int(active.size),
            "active_child_quantiles_0_10_25_50_75_90_99_100":
                np.quantile(active, [0, .1, .25, .5, .75, .9, .99, 1]).tolist(),
            "active_child_below": {str(limit): {"count": int((active < limit).sum()),
                                                "fraction": float((active < limit).mean())}
                                   for limit in (128, 512, 1024)}}


def stratify_layer(wrapper, captured, parent_counts, device):
    x, observed = captured
    assert x.shape == observed.shape == (1, P201.TOKENS, 896)
    flat = x.reshape(P201.TOKENS, 896)
    parents, scores = wrapper.routes(flat)
    children = wrapper.child_route(flat, parents)
    selected = wrapper.last_selected
    assert torch.equal(children, wrapper.last_children)
    assert selected.shape == children.shape == (P201.TOKENS, 4)
    assert torch.equal(selected.div(10, rounding_mode="floor"), children)
    content = selected.remainder(10) != 0
    exposure = parent_counts[children]
    gate = F.softmax(scores, dim=-1).to(flat.dtype)
    shared_a = wrapper.shared_a[parents].to(flat.dtype)
    hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, shared_a))
    ids = children[..., None] * 10 + torch.arange(1, 10, device=device)
    sibling_b = wrapper.cpu_bank.index_select(0, ids.reshape(-1).cpu()).to(
        device).reshape(P201.TOKENS, 4, 9, 896, 8).to(flat.dtype)
    outputs = torch.einsum("nkr,nkcdr->nkcd", hidden, sibling_b).float()
    mean = outputs.mean(dim=2)
    deviation = (outputs - mean.unsqueeze(2)).square().sum(dim=(2, 3)) / 9
    mean_energy = mean.square().sum(dim=-1)
    mlp_energy = observed.float().square().sum(dim=-1)[:, None].expand_as(mean_energy)
    gate_sq = gate.float().square()
    out = {}
    for name, lo, hi in BINS:
        mask = content & (exposure >= lo)
        if hi is not None:
            mask = mask & (exposure < hi)
        weighted = mask.float() * gate_sq
        row = {"content_selections": int(mask.sum()),
               "variation_energy": float((deviation * weighted).sum()),
               "content_mean_energy": float((mean_energy * weighted).sum()),
               "selected_path_mlp_energy": float((mlp_energy * mask.float()).sum())}
        assert all(math.isfinite(value) for value in row.values())
        out[name] = row
    return out


def summarize(rows, exposure):
    by_layer = []
    for layer in range(24):
        item = {"layer": layer, "bins": {}}
        for name, _, _ in BINS:
            sub = [row["layers"][layer][name] for row in rows]
            total = {key: sum(entry[key] for entry in sub) for key in sub[0]}
            total["sibling_rms_over_content_mean_rms"] = math.sqrt(
                total["variation_energy"] / total["content_mean_energy"])
            total["sibling_rms_over_selected_path_mlp_rms"] = math.sqrt(
                total["variation_energy"] / total["selected_path_mlp_energy"])
            item["bins"][name] = total
        by_layer.append(item)
    pooled = {}
    for name, _, _ in BINS:
        sub = [row["bins"][name] for row in by_layer]
        total = {key: sum(entry[key] for entry in sub)
                 for key in ("content_selections", "variation_energy",
                             "content_mean_energy", "selected_path_mlp_energy")}
        total["sibling_rms_over_content_mean_rms"] = math.sqrt(
            total["variation_energy"] / total["content_mean_energy"])
        total["sibling_rms_over_selected_path_mlp_rms"] = math.sqrt(
            total["variation_energy"] / total["selected_path_mlp_energy"])
        pooled[name] = total
    high_pass = [row["layer"] for row in by_layer
                 if row["bins"]["high"]["content_selections"] >= 128 and
                 row["bins"]["high"]["sibling_rms_over_content_mean_rms"] >= .10 and
                 row["bins"]["high"]["sibling_rms_over_selected_path_mlp_rms"] >= .01]
    low_pass = [row["layer"] for row in by_layer
                if row["bins"]["low"]["content_selections"] >= 128 and
                row["bins"]["low"]["sibling_rms_over_content_mean_rms"] >= .10 and
                row["bins"]["low"]["sibling_rms_over_selected_path_mlp_rms"] >= .01]
    exposure_only_warrants_pilot = len(high_pass) >= 18 and len(low_pass) < 18
    return {"training_exposure": exposure, "pooled_bins": pooled,
            "layers": by_layer, "high_passing_layers": high_pass,
            "low_passing_layers": low_pass,
            "decision": "longer_concentrated_training_pilot_warranted"
                        if exposure_only_warrants_pilot else
                        "change_route_training_coupling_first"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    stage = "bindings"
    started = time.monotonic()
    device = None
    try:
        assert P.digest(P.MANIFEST) == P.MANIFEST_SHA
        assert P.digest(P.CANDIDATE) == P.CANDIDATE_SHA
        assert P.digest(BASELINE) == BASELINE_SHA
        manifest = json.loads(P.MANIFEST.read_text(encoding="utf-8"))
        candidate = json.loads(P.CANDIDATE.read_text(encoding="utf-8"))
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        bank = Path(candidate["artifact"]["path"])
        assert P.digest(bank) == candidate["artifact"]["sha256"]
        assert baseline["candidate_bank_sha256"] == candidate["artifact"]["sha256"]
        assert tuple(baseline["document_indices"]) == P201.INDICES
        counts = np.asarray(candidate["training"]["content_route"]["route_counts_by_layer"],
                            dtype=np.int64).reshape(24, 1280, 10)
        exposure = exposure_summary(counts)
        source_rows = manifest["document_rows"]
        selected_rows = [source_rows[i] for i in P201.INDICES]
        for item, old in zip(selected_rows, baseline["rows"]):
            ids = np.asarray(item["span_ids"], dtype=np.int32)
            assert ids.shape == (1024,)
            assert P.M17.sha(ids.tobytes()) == item["span_ids_sha256"]
            assert item["source_row"] == old["source_row"]
            assert item["span_ids_sha256"] == old["span_ids_sha256"]
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count())
                   if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
        assert len(matches) == 1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        P.M17.MAX_SECONDS = M136.MAX_SECONDS = M176.MAX_SECONDS = MAX_SECONDS
        P.M17.MAX_RSS_BYTES = M136.MAX_RSS = M176.MAX_RSS = MAX_RSS
        P.M17.MAX_GPU_BYTES = M136.MAX_GPU = M176.MAX_GPU = MAX_GPU
        table = M175.load_table()
        stage = "initial_parity"
        parent, child = M136.bind_inputs()
        source_a, source_b, prefixes = M136.load_factor_bank()
        parity = M176.initial_parity(parent, child, source_a, source_b,
                                     prefixes, table, device, started)
        stage = "load_candidate"
        model = M136.model_shell(device).eval()
        original_class = M151.SharedStructureSparseExperts
        M176.LowRecurrenceEvalExperts.route_table = table
        M151.SharedStructureSparseExperts = M176.LowRecurrenceEvalExperts
        try:
            wrappers, _ = M136.make_wrappers(
                model, parent["expert_state"], child["expert_state"],
                source_a, source_b, prefixes, device, "shared")
        finally:
            M151.SharedStructureSparseExperts = original_class
        binding = M156.install_bank(wrappers, bank, 12800)
        assert binding["sha256"] == candidate["artifact"]["sha256"]
        del parent, child, source_a, source_b
        gc.collect()
        parent_counts = [torch.from_numpy(counts[li, :, 1:].sum(axis=1).copy()).to(device)
                         for li in range(24)]
        captures = [None] * 24
        hooks = []
        for layer, wrapper in enumerate(wrappers):
            def record(module, inputs, output, li=layer):
                captures[li] = (inputs[0].detach(), output.detach())
            hooks.append(wrapper.register_forward_hook(record))
        stage = "stratified_function"
        rows = []
        with torch.inference_mode():
            for item, old in zip(selected_rows, baseline["rows"]):
                ids = item["span_ids"]
                prefix = torch.as_tensor([model.config.eos_token_id] +
                                         ids[:P201.TOKENS], dtype=torch.long,
                                         device=device)
                window = prefix[:P201.TOKENS][None]
                positions = torch.arange(P201.TOKENS, dtype=torch.long,
                                         device=device)[None]
                tokens = window[0].cpu().numpy().astype(np.int64)
                previous = np.r_[0, tokens[:-1]]
                route_positions = np.arange(P201.TOKENS, dtype=np.int64)
                for wrapper in wrappers:
                    wrapper.enabled = True
                    wrapper.set_explicit_token_context(tokens, previous,
                                                       route_positions)
                model(window, position_ids=positions, use_cache=False)
                assert all(capture is not None for capture in captures)
                layers = []
                for li, (wrapper, captured) in enumerate(zip(wrappers, captures)):
                    check = P201.analyze_layer(wrapper, captured, device)
                    prior = old["layers"][li]
                    assert check["content_selections"] == prior["content_selections"]
                    for key in ("variation_energy", "content_mean_energy",
                                "mlp_output_energy"):
                        assert math.isclose(check[key], prior[key], rel_tol=1e-6), (li, key)
                    bins = stratify_layer(wrapper, captured, parent_counts[li], device)
                    assert sum(row["content_selections"] for row in bins.values()) == \
                        check["content_selections"]
                    for key in ("variation_energy", "content_mean_energy"):
                        assert math.isclose(sum(row[key] for row in bins.values()),
                                            check[key], rel_tol=1e-5, abs_tol=1e-3), (li, key)
                    layers.append(bins)
                    captures[li] = None
                    budget(started, device)
                rows.append({"source_row": item["source_row"], "layers": layers})
                print(json.dumps({"completed_sources": len(rows),
                                  "budget": budget(started, device)}), flush=True)
        for hook in hooks:
            hook.remove()
        stage = "summary"
        summary = summarize(rows, exposure)
        result = {"experiment": "METH-203-exposure-function-strata",
                  "manifest_sha256": P.MANIFEST_SHA,
                  "candidate_result_sha256": P.CANDIDATE_SHA,
                  "candidate_bank_sha256": binding["sha256"],
                  "meth201_result_sha256": BASELINE_SHA,
                  "input_tokens_per_document": P201.TOKENS,
                  "document_indices": P201.INDICES,
                  "initial_bf16_parity": parity,
                  "bank_readback": binding,
                  "rows": rows, "summary": summary,
                  "runtime": {**budget(started, device),
                              "gpu": torch.cuda.get_device_name(device)},
                  "scope": "Viewed METH-173 real states; no new quality data"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < MAX_DISK
        print(json.dumps({"decision": summary["decision"],
                          "pooled_bins": summary["pooled_bins"],
                          "high_passing_layers": summary["high_passing_layers"],
                          "runtime": result["runtime"]}), flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-203-failure",
                                       "stage": stage, "error": repr(error),
                                       "elapsed_seconds": time.monotonic() - started,
                                       "rss_bytes": psutil.Process().memory_info().rss,
                                       "gpu_peak_allocated_bytes":
                                           torch.cuda.max_memory_allocated(device)
                                           if device is not None else None},
                                      indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
