#!/usr/bin/env python3
"""METH-84: test each affordable BF16 FFN-layer rescue on viewed prompts."""

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import psutil
from safetensors import safe_open
import torch

import meth15_zero_residual_expert_smoke as M15
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57
import meth82_export_q4_ffn_core as M82
import meth83_q4_core_composition as M83


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
M83_RESULT = DIR / "meth83_q4_core_composition_result.json"
M83_RESULT_SHA = "443f1877ccb1b3edf7703af0e95b6bf58743f4bb899df01092daccc2d73fd528"
BASE_IDEAL_BYTES = 535_739_904
LAYER_EXTRA_BYTES = 19_203_072
MAX_SECONDS = 5 * 60


def budget(start, device):
    report = {"elapsed_seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (report["elapsed_seconds"] > MAX_SECONDS
            or report["rss_bytes"] > 20 * (1 << 30)
            or report["gpu_peak_allocated_bytes"] > int(10.5 * (1 << 30))):
        raise RuntimeError(f"METH-84 budget exceeded: {report}")
    return report


def score_prompts(model, wrappers, items, donor_top, device, start):
    M44.set_experts(wrappers, True)
    matches = positions = 0
    per_prompt = []
    with torch.inference_mode():
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                  device=device).unsqueeze(0)
            top = model(ids, use_cache=False).logits.argmax(dim=-1)[0].cpu()
            same = int((top == donor_top[item["source_id"]]).sum())
            matches += same
            positions += len(top)
            per_prompt.append({"source_id": item["source_id"], "same": same,
                               "positions": len(top)})
            budget(start, device)
    return {"matching": matches, "positions": positions,
            "agreement": matches / positions, "per_prompt": per_prompt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.stem + ".partial.json")
    assert M15.M13.sha256(M83.MANIFEST) == M83.MANIFEST_SHA
    assert M15.M13.sha256(M83.CORE) == M83.CORE_SHA
    assert M15.M13.sha256(M83_RESULT) == M83_RESULT_SHA
    assert M15.M13.sha256(M83.TRAINING) == M57.TRAINING_SHA
    reference = json.loads(M83_RESULT.read_text(encoding="utf-8"))
    assert reference["decision"] == "stop_q4_core_composition_development"
    items = json.loads(M83.MANIFEST.read_text(encoding="utf-8"))["items"]
    assert len(items) == 24
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert M15.M13.sha256(source) == M57.MODEL_SHA
    training = json.loads(M83.TRAINING.read_text(encoding="utf-8"))
    checkpoint = training["checkpoints"]["512"]
    assert checkpoint["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(checkpoint["path"]) == M57.CHECKPOINT_SHA

    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    original_params = dict(model.named_parameters())
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    assert state["updates"] == 512 and state["source_sha256"] == M57.MODEL_SHA
    wrappers = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
            layer.mlp = wrapper
            for key in ("a", "b", "router"):
                getattr(wrapper, key).copy_(state["expert_state"][li][key].to(device))
            wrappers.append(wrapper)
    del state
    model.config.use_cache = False
    donor_top = {}
    M44.set_experts(wrappers, False)
    with torch.inference_mode():
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                  device=device).unsqueeze(0)
            donor_top[item["source_id"]] = model(
                ids, use_cache=False).logits.argmax(dim=-1)[0].cpu()
            budget(start, device)
    bf16_control = score_prompts(model, wrappers, items, donor_top, device, start)
    assert (bf16_control["matching"], bf16_control["positions"]) == (4045, 4199)
    assert bf16_control["matching"] == reference["prompt_summary"]["pooled"][
        "matching"]["bf16_e128"]
    q4_by_layer = {}
    with safe_open(str(M83.CORE), framework="pt", device="cpu") as archive:
        with torch.no_grad():
            for li in range(24):
                q4_by_layer[li] = {}
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    name = f"model.layers.{li}.mlp.{projection}.weight"
                    packed = archive.get_tensor(name + ".q4").to(device)
                    scale = archive.get_tensor(name + ".scale").to(device)
                    restored = M82.reconstruct(packed, scale)
                    original_params[name].copy_(restored)
                    q4_by_layer[li][name] = restored.cpu()
                    budget(start, device)
    q4_baseline = score_prompts(model, wrappers, items, donor_top, device, start)
    assert (q4_baseline["matching"], q4_baseline["positions"]) == (2875, 4199)
    assert q4_baseline["matching"] == reference["prompt_summary"]["pooled"][
        "matching"]["q4_e128"]

    arms = []
    with safe_open(str(source), framework="pt", device="cpu") as archive:
        for li in range(24):
            with torch.no_grad():
                for name in q4_by_layer[li]:
                    original = archive.get_tensor(name)
                    assert original.dtype == torch.bfloat16
                    original_params[name].copy_(original.to(device))
            scored = score_prompts(model, wrappers, items, donor_top, device, start)
            arms.append({"layer": li,
                         "ideal_addressed_total_bytes": BASE_IDEAL_BYTES + LAYER_EXTRA_BYTES,
                         "extra_bytes": LAYER_EXTRA_BYTES,
                         "delta_matching_vs_q4": scored["matching"] - q4_baseline["matching"],
                         **scored})
            with torch.no_grad():
                for name, q4 in q4_by_layer[li].items():
                    original_params[name].copy_(q4.to(device))
            partial.write_text(json.dumps({"experiment": "METH-84",
                                           "completed_layers": len(arms),
                                           "bf16_control": bf16_control,
                                           "q4_baseline": q4_baseline,
                                           "arms": arms}, indent=2) + "\n",
                               encoding="utf-8")
            budget(start, device)
            print(json.dumps({"layer": li, "matching": scored["matching"],
                              "delta": arms[-1]["delta_matching_vs_q4"]}), flush=True)
    final_baseline = score_prompts(model, wrappers, items, donor_top, device, start)
    assert final_baseline["matching"] == q4_baseline["matching"]
    best = max(arms, key=lambda arm: (arm["matching"], -arm["layer"]))
    decision = ("candidate_only_for_fresh_single_layer_quality_test"
                if best["agreement"] >= 0.95
                and best["ideal_addressed_total_bytes"] <= 560_000_000
                else "single_full_ffn_layer_restore_insufficient_on_viewed_prompts")
    result = {"experiment": "METH-84-single-FFN-layer-viewed-diagnostic",
              "manifest_sha256": M83.MANIFEST_SHA,
              "meth83_result_sha256": M83_RESULT_SHA,
              "core_sha256": M83.CORE_SHA,
              "checkpoint_sha256": M57.CHECKPOINT_SHA,
              "bf16_control": bf16_control, "q4_baseline": q4_baseline,
              "arms": arms, "best_layer": best["layer"],
              "best_matching": best["matching"], "decision": decision,
              "runtime": {**budget(start, device),
                          "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "best_layer": best["layer"],
                      "best_matching": best["matching"],
                      "q4_baseline_matching": q4_baseline["matching"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
