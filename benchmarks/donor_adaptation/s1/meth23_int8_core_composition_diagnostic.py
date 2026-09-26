#!/usr/bin/env python3
"""METH-23: R8 head/body precision diagnosis with the bound E128 adapter."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from safetensors import safe_open
from safetensors.torch import load_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth19_half_residual_independent_audit as M19
import meth20_half_adapter_generation as M20


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/donor_adaptation/ternary"))
import t2_rules as T2  # noqa: E402 -- reuse the exporter's exact R8 rule


M19_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth19_half_residual_independent_result.json"
M19_RESULT_SHA = "eabf4bbe95e326656f9c65589c4a35da1527855dcb645f733d0020ca431e437a"
M20_MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth20_half_adapter_generation_manifest.json"
M20_MANIFEST_SHA = "0e8641615a3e29fba7d9a04eb568268719cc4ff036fac1f0b7ed5cf7d5240284"
CASES = ("original", "head_r8", "body_r8", "head_body_r8")
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-23 wall-time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-23 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-23 GPU stop: {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def summarize(rows):
    groups = {"pooled": rows}
    groups.update({category: [r for r in rows if r["category"] == category]
                   for category in M19.COUNTS})
    out = {}
    for name, group in groups.items():
        denom = math.log(2) * sum(r["bytes"] for r in group)
        original_donor = sum(r["nats"]["original"]["donor"] for r in group) / denom
        out[name] = {"docs": len(group), "bytes": sum(r["bytes"] for r in group),
                     "original_donor_bpb": original_donor, "cases": {}}
        for case in CASES:
            donor = sum(r["nats"][case]["donor"] for r in group) / denom
            adapter = sum(r["nats"][case]["adapter"] for r in group) / denom
            out[name]["cases"][case] = {
                "donor_bpb": donor, "adapter_bpb": adapter,
                "donor_delta_to_original_donor": donor-original_donor,
                "adapter_delta_to_original_donor": adapter-original_donor,
                "adapter_delta_to_same_core_donor": adapter-donor}
    return out


def quantize_param(param):
    assert param.ndim == 2 and param.dtype == torch.bfloat16
    original = param.detach().float()
    q, scale = T2.r8_int8_rtn(original)
    assert q.shape == original.shape and scale.shape == (original.shape[0], 1)
    assert bool((q >= -127).all()) and bool((q <= 127).all())
    reconstructed = (q * scale).to(param.dtype)
    error = reconstructed.float() - original
    stats = {"rows": original.shape[0], "elements": original.numel(),
             "int8_code_bytes": original.numel(),
             "fp32_row_scale_bytes": original.shape[0] * 4,
             "relative_l2_error": float(torch.linalg.vector_norm(error) /
                                        torch.linalg.vector_norm(original).clamp_min(1e-20)),
             "max_abs_error": float(error.abs().max())}
    param.copy_(reconstructed)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    assert M17.sha(M20_MANIFEST.read_bytes()) == M20_MANIFEST_SHA
    m19_result_hash = M15.M13.sha256(M19_RESULT)
    assert m19_result_hash == M19_RESULT_SHA
    prior = json.loads(M19_RESULT.read_text(encoding="utf-8"))
    assert prior["manifest_sha256"] == M20.DOCUMENT_MANIFEST_SHA
    _, items = M19.select()
    start = time.monotonic()
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
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
    prompt_manifest, prompts = M20.selected_prompts(tokenizer)
    assert json.loads(M20_MANIFEST.read_text(encoding="utf-8")) == prompt_manifest
    with safe_open(str(M20.ADAPTER), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
    assert meta["model_sha256"] == M15.M13.MODEL_SHA
    assert meta["source_checkpoint_sha256"] == M17.CHECKPOINT_SHA
    assert meta["output_factor"] == "0.5"
    adapter = load_file(str(M20.ADAPTER), device="cpu")

    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert model.config.eos_token_id == tokenizer.eos_token_id
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.model.embed_tokens.weight
    body = [(name, p) for name, p in model.named_parameters()
            if name.startswith("model.layers.") and p.ndim == 2
            and (".mlp." in name or ".self_attn." in name)]
    assert len(body) == 168
    assert sum(p.numel() for _, p in body) == 357826560
    head_original = head.detach().cpu().clone()
    body_original = [(name, p, p.detach().cpu().clone()) for name, p in body]
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(adapter[f"layers.{li}.{key}"].to(device))
        wrappers.append(wrapper)
    assert len(adapter) == 3 * M15.M13.L
    del adapter

    rows = [{"index": i, "category": item["category"],
             "source_id": item["source_id"],
             "text_sha256": M17.sha(item["text"].encode("utf-8")),
             "bytes": len(item["text"].encode("utf-8")),
             "nats": {}}
            for i, item in enumerate(items)]
    ranking = [{"index": i, "category": prompt["category"],
                "source_id": prompt["source_id"],
                "prompt_ids_sha256": prompt["prompt_ids_sha256"],
                "top1": {}}
               for i, prompt in enumerate(prompts)]
    quant_stats = {}
    for case in CASES:
        with torch.no_grad():
            head.copy_(head_original.to(device))
            for _, p, original in body_original:
                p.copy_(original.to(device))
            head_stat = None
            body_stats = []
            if case in ("head_r8", "head_body_r8"):
                head_stat = quantize_param(head)
            if case in ("body_r8", "head_body_r8"):
                body_stats = [{"name": name, **quantize_param(p)}
                              for name, p, _ in body_original]
        quant_stats[case] = {"head": head_stat,
                             "body": {"matrices": len(body_stats),
                                      "int8_code_bytes": sum(x["int8_code_bytes"] for x in body_stats),
                                      "fp32_row_scale_bytes": sum(x["fp32_row_scale_bytes"] for x in body_stats),
                                      "max_relative_l2_error": max((x["relative_l2_error"] for x in body_stats), default=0),
                                      "max_abs_error": max((x["max_abs_error"] for x in body_stats), default=0)}}
        for i, item in enumerate(items):
            ids = tokenizer.encode(item["text"], add_special_tokens=False)
            for enabled, arm in ((False, "donor"), (True, "adapter")):
                rows[i]["nats"].setdefault(case, {})[arm] = M17.score_doc(
                    model, ids, wrappers, enabled, device, start)
            if (i+1) % 12 == 0:
                print(f"{case} scored {i+1}/{len(items)} documents", flush=True)
                check_budget(start, device)
        with torch.inference_mode():
            for i, prompt in enumerate(prompts):
                ids = torch.as_tensor(prompt["prompt_ids"], dtype=torch.long,
                                      device=device).unsqueeze(0)
                for enabled, arm in ((False, "donor"), (True, "adapter")):
                    for w in wrappers:
                        w.enabled = enabled
                    top = model(ids, use_cache=False).logits.argmax(dim=-1)[0].tolist()
                    ranking[i]["top1"].setdefault(case, {})[arm] = top
                check_budget(start, device)
        print(f"completed {case}", flush=True)

    summary = summarize(rows)
    original = prior["summary"]["pooled"]
    assert abs(summary["pooled"]["cases"]["original"]["donor_bpb"] -
               original["donor_bpb"]) <= 1e-5
    assert abs(summary["pooled"]["cases"]["original"]["adapter_bpb"] -
               original["half_bpb"]) <= 1e-5
    agreement = {}
    for case in CASES:
        agreement[case] = {}
        for arm in ("donor", "adapter"):
            n = sum(len(r["top1"][case][arm]) for r in ranking)
            same = sum(sum(a == b for a, b in zip(r["top1"][case][arm],
                                                  r["top1"]["original"][arm]))
                       for r in ranking)
            agreement[case][arm] = {"positions": n, "matching_top1": same,
                                    "fraction": same/n}
    combined = summary["pooled"]["cases"]["head_body_r8"]
    diagnostic_pass = (
        combined["donor_delta_to_original_donor"] <= 0.01
        and combined["adapter_delta_to_original_donor"] <= 0.01
        and all(summary[c]["cases"]["head_body_r8"]["adapter_delta_to_original_donor"]
                <= 0.03 for c in M19.COUNTS)
        and agreement["head_body_r8"]["donor"]["fraction"] >= 0.95
        and agreement["head_body_r8"]["adapter"]["fraction"] >= 0.95)
    runtime = check_budget(start, device)
    result = {"experiment": "METH-23", "decision": "diagnostic_quality_pass_export_next"
              if diagnostic_pass else "diagnostic_quality_fail_or_inconclusive",
              "document_manifest_sha256": M20.DOCUMENT_MANIFEST_SHA,
              "m19_result_sha256": m19_result_hash,
              "prompt_manifest_sha256": M20_MANIFEST_SHA,
              "donor_sha256": M15.M13.MODEL_SHA,
              "adapter_sha256": M20.ADAPTER_SHA,
              "rule": "t2_rules.r8_int8_rtn per output row; dequantize to BF16",
              "cases": CASES, "quantization": quant_stats,
              "document_rows": rows, "summary": summary,
              "ranking_rows": ranking, "top1_agreement": agreement,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "top1_agreement": agreement,
                      "decision": result["decision"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
