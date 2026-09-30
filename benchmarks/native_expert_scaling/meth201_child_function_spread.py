#!/usr/bin/env python3
"""Measure trained E12800 sibling output spread on viewed real states."""

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import meth179_content_route_function as P


M136, M151, M156, M175, M176 = P.M136, P.M151, P.M156, P.M175, P.M176
DOC = P.DOC
INDICES = (0, 3, 6, 9, 12, 15, 18, 21)
TOKENS = 128
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
        raise RuntimeError(f"METH-201 resource stop: {row}")
    return row


def analyze_layer(wrapper, captured, device):
    x, observed = captured
    assert x.shape == observed.shape == (1, TOKENS, M136.M15.M13.D)
    flat = x.reshape(TOKENS, M136.M15.M13.D)
    parents, scores = wrapper.routes(flat)
    children = wrapper.child_route(flat, parents)
    assert torch.equal(children, wrapper.last_children)
    selected = wrapper.last_selected
    assert selected.shape == children.shape == parents.shape == (TOKENS, 4)
    assert torch.equal(selected.div(10, rounding_mode="floor"), children)
    content = selected.remainder(10) != 0
    gate = F.softmax(scores, dim=-1).to(flat.dtype)
    a = wrapper.shared_a[parents].to(flat.dtype)
    hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, a))
    bank = wrapper.cpu_bank
    assert bank.device.type == "cpu" and bank.shape == (12800, 896, 8)
    all_ids = children[..., None] * 10 + torch.arange(1, 10, device=device)
    assert all_ids.shape == (TOKENS, 4, 9)
    sibling_b = bank.index_select(0, all_ids.reshape(-1).cpu()).to(
        device).reshape(TOKENS, 4, 9, 896, 8).to(flat.dtype)
    sibling_out = torch.einsum("nkr,nkcdr->nkcd", hidden, sibling_b)
    assert sibling_out.shape == (TOKENS, 4, 9, 896)
    direct_b = bank.index_select(0, selected.reshape(-1).cpu()).to(
        device).reshape(TOKENS, 4, 896, 8).to(flat.dtype)
    direct_out = torch.einsum("nkr,nkdr->nkd", hidden, direct_b)
    predicted = wrapper.base(x) + (direct_out * gate.unsqueeze(-1)).sum(
        dim=1).reshape_as(observed)
    assert torch.equal(predicted, observed), float((predicted.float() -
                                                    observed.float()).abs().max())
    chosen_index = (selected.remainder(10) - 1).clamp_min(0)
    chosen_b = sibling_b.gather(2, chosen_index[..., None, None, None].expand(
        TOKENS, 4, 1, 896, 8)).squeeze(2)
    assert torch.equal(chosen_b[content], direct_b[content])
    chosen_direct = torch.einsum("nkr,nkdr->nkd", hidden, chosen_b)
    assert torch.equal(chosen_direct[content], direct_out[content])
    chosen = sibling_out.gather(2, chosen_index[..., None, None].expand(
        TOKENS, 4, 1, 896)).squeeze(2)

    mean = sibling_out.float().mean(dim=2)
    diff = sibling_out.float() - mean.unsqueeze(2)
    weights = gate.float().square() * content.float()
    variation = float((diff.square().sum(dim=(2, 3)) / 9 * weights).sum())
    mean_energy = float((mean.square().sum(dim=-1) * weights).sum())
    chosen_energy = float(((chosen.float() - mean).square().sum(dim=-1)
                           * weights).sum())
    mlp_energy = float(observed.float().square().sum())
    assert variation >= 0 and mean_energy > 0 and mlp_energy > 0
    return {"selections": selected.numel(),
            "content_selections": int(content.sum()),
            "structural_selections": int((~content).sum()),
            "variation_energy": variation, "content_mean_energy": mean_energy,
            "selected_deviation_energy": chosen_energy,
            "mlp_output_energy": mlp_energy,
            "sibling_rms_over_content_mean_rms": math.sqrt(variation / mean_energy),
            "sibling_rms_over_mlp_output_rms": math.sqrt(variation / mlp_energy),
            "selected_over_average_deviation_energy": chosen_energy / variation
            if variation else 0.0,
            "selected_output_parity_exact": True}


def aggregate(rows):
    by_layer = []
    for layer in range(24):
        subset = [row["layers"][layer] for row in rows]
        total = {key: sum(row[key] for row in subset) for key in (
            "selections", "content_selections", "structural_selections",
            "variation_energy", "content_mean_energy",
            "selected_deviation_energy", "mlp_output_energy")}
        assert total["selections"] == len(rows) * TOKENS * 4
        assert total["content_selections"] + total["structural_selections"] == total[
            "selections"]
        total["layer"] = layer
        total["sibling_rms_over_content_mean_rms"] = math.sqrt(
            total["variation_energy"] / total["content_mean_energy"])
        total["sibling_rms_over_mlp_output_rms"] = math.sqrt(
            total["variation_energy"] / total["mlp_output_energy"])
        total["selected_over_average_deviation_energy"] = (
            total["selected_deviation_energy"] / total["variation_energy"])
        by_layer.append(total)
    total = {key: sum(row[key] for row in by_layer) for key in (
        "selections", "content_selections", "structural_selections",
        "variation_energy", "content_mean_energy",
        "selected_deviation_energy", "mlp_output_energy")}
    total["sibling_rms_over_content_mean_rms"] = math.sqrt(
        total["variation_energy"] / total["content_mean_energy"])
    total["sibling_rms_over_mlp_output_rms"] = math.sqrt(
        total["variation_energy"] / total["mlp_output_energy"])
    total["selected_over_average_deviation_energy"] = (
        total["selected_deviation_energy"] / total["variation_energy"])
    pass_layers = [row["layer"] for row in by_layer
                   if row["sibling_rms_over_content_mean_rms"] >= .10 and
                   row["sibling_rms_over_mlp_output_rms"] >= .01]
    return {"all": total, "layers": by_layer,
            "median_sibling_over_mean": statistics.median(
                row["sibling_rms_over_content_mean_rms"] for row in by_layer),
            "median_sibling_over_mlp": statistics.median(
                row["sibling_rms_over_mlp_output_rms"] for row in by_layer),
            "both_thresholds_layers": pass_layers,
            "decision": "router_first_mechanism_candidate" if len(pass_layers) >= 18
                        else "specialist_training_signal_first"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    assert P.digest(P.MANIFEST) == P.MANIFEST_SHA
    assert P.digest(P.CANDIDATE) == P.CANDIDATE_SHA
    manifest = json.loads(P.MANIFEST.read_text(encoding="utf-8"))
    candidate = json.loads(P.CANDIDATE.read_text(encoding="utf-8"))
    assert candidate["decision"] == "matched_long_candidate_pass_fresh_quality_pending"
    bank = Path(candidate["artifact"]["path"])
    assert P.digest(bank) == candidate["artifact"]["sha256"]
    source_rows = manifest["document_rows"]
    assert len(source_rows) == 24
    selected_rows = [source_rows[i] for i in INDICES]
    for item in selected_rows:
        ids = np.asarray(item["span_ids"], dtype=np.int32)
        assert ids.shape == (1024,)
        assert P.M17.sha(ids.tobytes()) == item["span_ids_sha256"]
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    M136.MAX_SECONDS = M176.MAX_SECONDS = MAX_SECONDS
    M136.MAX_RSS = M176.MAX_RSS = MAX_RSS
    M136.MAX_GPU = M176.MAX_GPU = MAX_GPU
    table = M175.load_table()
    parent, child = M136.bind_inputs()
    source_a, source_b, prefixes = M136.load_factor_bank()
    parity = M176.initial_parity(parent, child, source_a, source_b,
                                 prefixes, table, device, started)
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
    assert len(wrappers) == 24
    binding = M156.install_bank(wrappers, bank, 12800)
    assert binding["sha256"] == candidate["artifact"]["sha256"]
    del parent, child, source_a, source_b
    gc.collect()
    budget(started, device)

    captures = [None] * 24
    hooks = []
    for layer, wrapper in enumerate(wrappers):
        def record(module, inputs, output, li=layer):
            captures[li] = (inputs[0].detach(), output.detach())
        hooks.append(wrapper.register_forward_hook(record))
    rows = []
    with torch.inference_mode():
        for item in selected_rows:
            ids = item["span_ids"]
            prefix = torch.as_tensor([model.config.eos_token_id] + ids[:TOKENS],
                                     dtype=torch.long, device=device)
            window = prefix[:TOKENS][None]
            positions = torch.arange(TOKENS, dtype=torch.long,
                                     device=device)[None]
            tokens = window[0].cpu().numpy().astype(np.int64)
            previous = np.r_[0, tokens[:-1]]
            route_positions = np.arange(TOKENS, dtype=np.int64)
            for wrapper in wrappers:
                wrapper.enabled = True
                wrapper.set_explicit_token_context(tokens, previous,
                                                   route_positions)
            model(window, position_ids=positions, use_cache=False)
            assert all(capture is not None for capture in captures)
            layer_rows = []
            for layer, (wrapper, captured) in enumerate(zip(wrappers, captures)):
                row = analyze_layer(wrapper, captured, device)
                row["layer"] = layer
                layer_rows.append(row)
                captures[layer] = None
                budget(started, device)
            rows.append({"source_row": item["source_row"],
                         "span_ids_sha256": item["span_ids_sha256"],
                         "layers": layer_rows})
            print(json.dumps({"completed_sources": len(rows),
                              "budget": budget(started, device)}), flush=True)
    for hook in hooks:
        hook.remove()
    summary = aggregate(rows)
    result = {"experiment": "METH-201-E12800-child-function-spread",
              "manifest_sha256": P.MANIFEST_SHA,
              "candidate_result_sha256": P.CANDIDATE_SHA,
              "candidate_bank_sha256": binding["sha256"],
              "document_indices": INDICES,
              "input_tokens_per_document": TOKENS,
              "initial_bf16_parity": parity,
              "bank_readback": binding,
              "rows": rows, "summary": summary,
              "runtime": {**budget(started, device),
                          "gpu": torch.cuda.get_device_name(device)},
              "scope": "Viewed METH-173 states; functional spread only, no target labels"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.out.stat().st_size < MAX_DISK
    print(json.dumps({"decision": summary["decision"],
                      "all": summary["all"],
                      "both_thresholds_layers": summary["both_thresholds_layers"],
                      "runtime": result["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
