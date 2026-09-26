#!/usr/bin/env python3
"""METH-36: stored one-byte router sketch on disjoint real Qwen inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth24_export_r8_core as M24
import meth25_fresh_r8_artifact_audit as M25
import meth25_fresh_r8_manifest as M25M
import meth27_r8_fresh_generation as M27
import meth35_low_rank_router as M35
import meth36_export_int8_sketch as M36


ROOT = Path(__file__).resolve().parents[3]
PROMPT_MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth36_route_prompt_manifest.json"
PROMPT_MANIFEST_SHA = "4bd2bcd666418bfc1b558a4ed6e57650dc31972b29f9a3f6671b99e573dad5ab"
SKETCH_ARTIFACT = ROOT / "results/native_expert_scaling/meth36_e128_router_r64_int8.safetensors"
SKETCH_SHA = "133287eb87cf993d9498fd18c1efafbf703bf10a29cc757a5a5ec73cadbb674d"
M35_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth35_low_rank_router_result.json"
M35_RESULT_SHA = M36.M35_SHA
RANK = 64
CANDIDATE_COUNT = 64
ARMS = ("fp32", "stored_int8")
MAX_SECONDS = 20 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-36 wall-time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-36 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-36 GPU stop: {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def empty_count():
    return {"positions": 0, "exact_ids": 0, "included_exact_ids": 0,
            "full_set_matches": 0, "missed_gate_mass_sum": 0.0,
            "worst_missed_gate_mass": 0.0}


def add_count(dst, src):
    for key in ("positions", "exact_ids", "included_exact_ids", "full_set_matches"):
        dst[key] += src[key]
    dst["missed_gate_mass_sum"] += src["missed_gate_mass_sum"]
    dst["worst_missed_gate_mass"] = max(dst["worst_missed_gate_mass"],
                                        src["worst_missed_gate_mass"])


def finalize_count(row):
    row = dict(row)
    row["exact_id_inclusion_fraction"] = row["included_exact_ids"] / row["exact_ids"]
    row["full_set_match_fraction"] = row["full_set_matches"] / row["positions"]
    row["mean_missed_gate_mass"] = row["missed_gate_mass_sum"] / row["positions"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert M17.sha(PROMPT_MANIFEST.read_bytes()) == PROMPT_MANIFEST_SHA
    assert M17.sha(M35_RESULT.read_bytes()) == M35_RESULT_SHA
    assert M15.M13.sha256(SKETCH_ARTIFACT) == SKETCH_SHA
    assert M15.M13.sha256(M25.CORE) == M25.CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
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
    prompt_manifest, prompts = M36.select_prompts(tokenizer)
    assert json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8")) == prompt_manifest
    assert len(prompts) == 24
    with safe_open(str(M20.ADAPTER), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["model_sha256"] == M15.M13.MODEL_SHA
        assert meta["output_factor"] == "0.5"
    adapter = load_file(str(M20.ADAPTER), device="cpu")
    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    original_params = dict(model.named_parameters())
    assert len(original_params) == 290
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(adapter[f"layers.{li}.{key}"].to(device))
        wrapper.enabled = True
        wrappers.append(wrapper)
    assert len(adapter) == 3 * M15.M13.L
    del adapter
    with safe_open(str(M25.CORE), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_R8_CORE_V1"
        assert meta["model_sha256"] == M15.M13.MODEL_SHA
        assert meta["adapter_sha256"] == M20.ADAPTER_SHA
        assert meta["tied_head"] == "true"
        assert len(archive.keys()) == 459
        matrix_count = control_count = 0
        for name, param in original_params.items():
            organ = M24.classify(name, tuple(param.shape))
            if organ == "control":
                stored = archive.get_tensor(name)
                assert stored.dtype == torch.float32
                assert torch.equal(stored.to(torch.bfloat16), param.detach().cpu())
                control_count += 1
            else:
                codes = archive.get_tensor(name + ".q")
                scale = archive.get_tensor(name + ".scale")
                assert codes.dtype == torch.int8 and tuple(codes.shape) == tuple(param.shape)
                assert scale.dtype == torch.float32 and tuple(scale.shape) == (param.shape[0],)
                assert bool((codes >= -127).all()) and bool((codes <= 127).all())
                param.copy_((codes.to(device).float() *
                             scale.to(device).unsqueeze(1)).to(torch.bfloat16))
                matrix_count += 1
        assert matrix_count == 169 and control_count == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    check_budget(start, device)
    print("stored R8 core and E128 adapter loaded", flush=True)

    indices = []
    with safe_open(str(SKETCH_ARTIFACT), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_E128_ROUTER_R64_INT8_V1"
        assert meta["rank"] == str(RANK)
        assert meta["adapter_sha256"] == M20.ADAPTER_SHA
        assert meta["core_sha256"] == M25.CORE_SHA
        assert len(archive.keys()) == 3 * M15.M13.L
        for li, wrapper in enumerate(wrappers):
            basis = archive.get_tensor(f"layers.{li}.basis")
            codes = archive.get_tensor(f"layers.{li}.codes")
            scales = archive.get_tensor(f"layers.{li}.scales")
            assert basis.dtype == torch.float32 and basis.shape == (M15.M13.D, RANK)
            assert codes.dtype == torch.int8 and codes.shape == (M15.E, RANK)
            assert scales.dtype == torch.float32 and scales.shape == (M15.E,)
            assert bool((codes >= -127).all()) and bool((codes <= 127).all())
            assert bool((scales >= 0).all())
            router_cpu = wrapper.router.detach().cpu()
            fp32_sketch = router_cpu @ basis
            decoded = codes.float() * scales[:, None]
            indices.append({"basis": basis.to(device),
                            "sketches": {"fp32": fp32_sketch.to(device),
                                         "stored_int8": decoded.to(device)}})
            check_budget(start, device)
    print("stored sketch reloaded and decoded", flush=True)

    counts = {arm: {category: [empty_count() for _ in range(M15.M13.L)]
                    for category in M25M.COUNTS}
              for arm in ARMS}
    stream_hashes = {arm: [hashlib.sha256() for _ in range(M15.M13.L)]
                     for arm in ("exact", *ARMS)}
    active_category = [None]

    def hook_for_layer(li):
        index = indices[li]
        router = wrappers[li].router.detach()

        def hook(module, input_tuple):
            x = input_tuple[0].detach().reshape(-1, M15.M13.D).float()
            n = x.shape[0]
            assert n == 256
            scores = F.linear(x, router)
            exact = torch.topk(scores, M15.K, dim=-1)
            exact_ids = exact.indices
            exact_gate = F.softmax(exact.values, dim=-1)
            stream_hashes["exact"][li].update(
                exact_ids.cpu().numpy().astype("<i2", copy=False).tobytes())
            projected_x = x @ index["basis"]
            for arm in ARMS:
                coarse = F.linear(projected_x, index["sketches"][arm])
                candidates = torch.topk(coarse, CANDIDATE_COUNT, dim=-1).indices
                selected_scores = (router[candidates] * x.unsqueeze(1)).sum(dim=-1)
                approximate = candidates.gather(
                    1, torch.topk(selected_scores, M15.K, dim=-1).indices)
                hits = (exact_ids.unsqueeze(2) == candidates.unsqueeze(1)).any(dim=2)
                matches = (torch.sort(exact_ids, dim=-1).values ==
                           torch.sort(approximate, dim=-1).values).all(dim=-1)
                mass = (exact_gate * (~hits)).sum(dim=-1)
                row = counts[arm][active_category[0]][li]
                row["positions"] += n
                row["exact_ids"] += n * M15.K
                row["included_exact_ids"] += int(hits.sum())
                row["full_set_matches"] += int(matches.sum())
                row["missed_gate_mass_sum"] += float(mass.sum())
                row["worst_missed_gate_mass"] = max(
                    row["worst_missed_gate_mass"], float(mass.max()))
                stream_hashes[arm][li].update(
                    approximate.cpu().numpy().astype("<i2", copy=False).tobytes())
        return hook

    handles = [wrapper.register_forward_pre_hook(hook_for_layer(li))
               for li, wrapper in enumerate(wrappers)]
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            active_category[0] = prompt["category"]
            ids = torch.as_tensor(prompt["prompt_ids"], dtype=torch.long,
                                  device=device).unsqueeze(0)
            model(ids, use_cache=False)
            check_budget(start, device)
            if (i+1) % 4 == 0:
                print(f"audited {i+1}/{len(prompts)} real prompts", flush=True)
    for handle in handles:
        handle.remove()
    assert all(sum(row["positions"] for cat in M25M.COUNTS
                   for row in counts[arm][cat]) == 24 * 256 * 24
               for arm in ARMS)

    summaries = {}
    for arm in ARMS:
        categories = {}
        pooled = empty_count()
        layers = [empty_count() for _ in range(M15.M13.L)]
        for category in M25M.COUNTS:
            aggregate = empty_count()
            for li, row in enumerate(counts[arm][category]):
                add_count(aggregate, row)
                add_count(pooled, row)
                add_count(layers[li], row)
            categories[category] = finalize_count(aggregate)
        pooled = finalize_count(pooled)
        passed = (pooled["exact_id_inclusion_fraction"] >= 0.999
                  and pooled["full_set_match_fraction"] >= 0.99
                  and all(r["exact_id_inclusion_fraction"] >= 0.99
                          for r in categories.values()))
        multiply_terms = RANK * M15.M13.D + M15.E * RANK + CANDIDATE_COUNT * M15.M13.D
        summaries[arm] = {"rank": RANK,
                          "candidate_rows": CANDIDATE_COUNT,
                          "multiply_terms_per_layer": multiply_terms,
                          "ratio_to_exhaustive_fp32_multiplies":
                          multiply_terms / (M15.E * M15.M13.D),
                          "hypothetical_e273547_int8_sketch_code_bytes_per_token":
                          M15.M13.L * 273547 * RANK,
                          "pooled": pooled, "categories": categories,
                          "layers": [finalize_count(r) for r in layers],
                          "diagnostic_route_gate_pass": passed}
    runtime = check_budget(start, device)
    result = {"experiment": "METH-36", "prompt_manifest_sha256": PROMPT_MANIFEST_SHA,
              "sketch_artifact_sha256": SKETCH_SHA,
              "meth35_result_sha256": M35_RESULT_SHA,
              "core_sha256": M25.CORE_SHA,
              "adapter_sha256": M20.ADAPTER_SHA,
              "donor_sha256": M15.M13.MODEL_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "index_rule": "stored int8 per-row SVD sketch, rank-64 score, exact 64-row rescore",
              "summary": summaries,
              "route_stream_sha256": {arm: [h.hexdigest() for h in hashes]
                                      for arm, hashes in stream_hashes.items()},
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__},
              "decision": "stored_int8_route_gate_pass_test_quality_and_cpu_next"
              if summaries["stored_int8"]["diagnostic_route_gate_pass"]
              else "stored_int8_route_gate_fail"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summaries, "decision": result["decision"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
