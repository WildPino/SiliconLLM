#!/usr/bin/env python3
"""METH-39: compare fine-router exact-rescore arithmetic paths."""
import argparse
import json
from pathlib import Path
import time

import psutil
from safetensors import safe_open
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth24_export_r8_core as M24
import meth25_fresh_r8_artifact_audit as M25
import meth36_int8_sketch_audit as M36
import meth37_route_replacement as M37
import meth38_candidate_cause as M38


ROOT = Path(__file__).resolve().parents[3]
M37_MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth37_route_quality_manifest.json"
M37_MANIFEST_SHA = "0c9004f36803a600deda1f884591e61a1a5ffb2b1ac0558ba75727565d30a913"
M37_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth37_route_replacement_result.json"
M37_RESULT_SHA = "d3d69a24c767987b38b57bf6dc240194259174ad5476ab59f757661d12b4b5d5"
M38_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth38_candidate_cause_result.json"
M38_RESULT_SHA = "a14e5a04475ffac972a45ad405e7382bcbe883ae61ed07f7c07faedc9132bbe4"
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or peak > MAX_GPU_BYTES or rss > MAX_RSS_BYTES:
        raise RuntimeError(f"METH-39 budget: {elapsed:.1f}s, GPU {peak}, RSS {rss}")
    return {"elapsed_seconds": elapsed, "gpu_peak_allocated_bytes": peak,
            "rss_end_bytes": rss}


def empty_count():
    return {"positions": 0, "exact_ids": 0, "included_exact_ids": 0,
            "full_set_matches": 0}


def finalize(row):
    return {**row,
            "exact_id_inclusion_fraction": row["included_exact_ids"] / row["exact_ids"],
            "full_set_match_fraction": row["full_set_matches"] / row["positions"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert M17.sha(M37_MANIFEST.read_bytes()) == M37_MANIFEST_SHA
    assert M17.sha(M37_RESULT.read_bytes()) == M37_RESULT_SHA
    assert M17.sha(M38_RESULT.read_bytes()) == M38_RESULT_SHA
    assert M15.M13.sha256(M25.CORE) == M25.CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    assert M15.M13.sha256(M36.SKETCH_ARTIFACT) == M36.SKETCH_SHA
    previous = json.loads(M37_RESULT.read_text(encoding="utf-8"))
    previous38 = json.loads(M38_RESULT.read_text(encoding="utf-8"))
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    start = time.monotonic()
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                             revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M15.M13.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    manifest, items = M37.select(tokenizer)
    assert json.loads(M37_MANIFEST.read_text(encoding="utf-8")) == manifest
    assert all(a["source_id"] == b["source_id"]
               for a, b in zip(items, previous["rows"]))
    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    original_params = dict(model.named_parameters())
    assert len(original_params) == 290
    adapter = load_file(str(M20.ADAPTER), device="cpu")
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M37.ShortlistExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(adapter[f"layers.{li}.{key}"].to(device))
        wrappers.append(wrapper)
    del adapter
    with safe_open(str(M36.SKETCH_ARTIFACT), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_E128_ROUTER_R64_INT8_V1"
        assert meta["adapter_sha256"] == M20.ADAPTER_SHA
        assert len(archive.keys()) == 3 * M15.M13.L
        for li, wrapper in enumerate(wrappers):
            basis = archive.get_tensor(f"layers.{li}.basis")
            codes = archive.get_tensor(f"layers.{li}.codes")
            scales = archive.get_tensor(f"layers.{li}.scales")
            assert basis.shape == (M15.M13.D, 64) and basis.dtype == torch.float32
            assert codes.shape == (M15.E, 64) and codes.dtype == torch.int8
            assert scales.shape == (M15.E,) and scales.dtype == torch.float32
            wrapper.basis.copy_(basis.to(device))
            wrapper.sketch.copy_((codes.float() * scales[:, None]).to(device))
    with safe_open(str(M25.CORE), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_R8_CORE_V1"
        assert meta["model_sha256"] == M15.M13.MODEL_SHA
        assert meta["adapter_sha256"] == M20.ADAPTER_SHA
        assert len(archive.keys()) == 459
        matrices = controls = 0
        for name, param in original_params.items():
            if M24.classify(name, tuple(param.shape)) == "control":
                assert torch.equal(archive.get_tensor(name).to(torch.bfloat16),
                                   param.detach().cpu())
                controls += 1
            else:
                codes = archive.get_tensor(name + ".q")
                scales = archive.get_tensor(name + ".scale")
                assert codes.dtype == torch.int8 and tuple(codes.shape) == tuple(param.shape)
                assert scales.dtype == torch.float32 and tuple(scales.shape) == (param.shape[0],)
                param.copy_((codes.to(device).float() *
                             scales.to(device).unsqueeze(1)).to(torch.bfloat16))
                matrices += 1
        assert matrices == 169 and controls == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    check_budget(start, device)
    print("stored model and router index reconstructed", flush=True)

    top1_streams = {}
    route_counts = {}
    arms = (("exact", None, "sum"), ("sum64", 64, "sum"),
            ("sum96", 96, "sum"), ("sum128", 128, "sum"),
            ("bmm128", 128, "bmm"), ("gather128", 128, "full_gather"),
            ("bmm96", 96, "bmm"))
    for label, candidate_count, mode in arms:
        for wrapper in wrappers:
            wrapper.enabled = True
            wrapper.route_mode = "exact" if candidate_count is None else "stored_int8"
            wrapper.candidate_count = candidate_count or 64
            wrapper.rescore_mode = mode
        counts = [empty_count() for _ in wrappers]
        handles = []
        if candidate_count is not None:
            for li, wrapper in enumerate(wrappers):
                def hook(module, input_tuple, layer=li):
                    w = wrappers[layer]
                    x = input_tuple[0].detach().reshape(-1, M15.M13.D).float()
                    assert x.shape[0] == 256
                    full_scores = F.linear(x, w.router)
                    exact_ids = torch.topk(full_scores, M15.K, dim=-1).indices
                    coarse = F.linear(x @ w.basis, w.sketch)
                    candidates = torch.topk(coarse, candidate_count, dim=-1).indices
                    if mode == "sum":
                        exact_scores = (w.router[candidates] * x.unsqueeze(1)).sum(dim=-1)
                    elif mode == "bmm":
                        exact_scores = torch.bmm(w.router[candidates],
                                                 x.unsqueeze(-1)).squeeze(-1)
                    else:
                        assert mode == "full_gather"
                        exact_scores = full_scores.gather(1, candidates)
                    chosen = candidates.gather(
                        1, torch.topk(exact_scores, M15.K, dim=-1).indices)
                    hits = (exact_ids.unsqueeze(2) == candidates.unsqueeze(1)).any(dim=2)
                    matches = (torch.sort(exact_ids, dim=-1).values ==
                               torch.sort(chosen, dim=-1).values).all(dim=-1)
                    row = counts[layer]
                    row["positions"] += x.shape[0]
                    row["exact_ids"] += x.shape[0] * M15.K
                    row["included_exact_ids"] += int(hits.sum())
                    row["full_set_matches"] += int(matches.sum())
                handles.append(wrapper.register_forward_pre_hook(hook))
        streams = []
        for i, item in enumerate(items):
            streams.append(M37.score_top1(model, item["prompt_ids"], device))
            check_budget(start, device)
        for handle in handles:
            handle.remove()
        top1_streams[label] = streams
        if label in ("exact", "sum64"):
            previous_arm = "r8_exact" if label == "exact" else "r8_int8_route"
            assert all(stream == row["top1"][previous_arm]
                       for stream, row in zip(streams, previous["rows"]))
        if candidate_count is not None:
            pooled = empty_count()
            for row in counts:
                for key in pooled:
                    pooled[key] += row[key]
            route_counts[label] = {"pooled": finalize(pooled),
                                   "layers": [finalize(row) for row in counts]}
        print(f"{label} scored 24 top1 prompts", flush=True)

    ranking = {}
    exact = top1_streams["exact"]
    for label, _, _ in arms[1:]:
        matching = sum(sum(a == b for a, b in zip(ref, candidate))
                       for ref, candidate in zip(exact, top1_streams[label]))
        ranking[label] = {"positions": 6144, "matching": matching,
                          "agreement_fraction": matching / 6144}
    for prior_label, label in (("c64", "sum64"), ("c96", "sum96"),
                               ("c128", "sum128")):
        assert ranking[label] == previous38["ranking"][prior_label]
        assert route_counts[label]["pooled"] == previous38["route_counts"][prior_label]["pooled"]
    gates = {"gather128_oracle":
             route_counts["gather128"]["pooled"]["full_set_match_fraction"] == 1.0
             and ranking["gather128"]["agreement_fraction"] == 1.0,
             "bmm128_numeric":
             route_counts["bmm128"]["pooled"]["full_set_match_fraction"] == 1.0
             and ranking["bmm128"]["agreement_fraction"] >= 0.9999,
             "bmm96_diagnostic": ranking["bmm96"]["agreement_fraction"] >= 0.99
             and route_counts["bmm96"]["pooled"]["exact_id_inclusion_fraction"] >= 0.999}
    gates["bmm96_interpretable"] = gates["gather128_oracle"] and gates["bmm128_numeric"]
    gates["bmm96_followup"] = gates["bmm96_interpretable"] and gates["bmm96_diagnostic"]
    runtime = check_budget(start, device)
    result = {"experiment": "METH-39", "prior_result_sha256": M37_RESULT_SHA,
              "prior_diagnostic_sha256": M38_RESULT_SHA,
              "manifest_sha256": M37_MANIFEST_SHA,
              "core_sha256": M25.CORE_SHA,
              "adapter_sha256": M20.ADAPTER_SHA,
              "sketch_sha256": M36.SKETCH_SHA,
              "ranking": ranking, "route_counts": route_counts,
              "gates": gates,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ranking": ranking,
                      "pooled_routes": {k: v["pooled"] for k, v in route_counts.items()},
                      "gates": gates, "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
