#!/usr/bin/env python3
"""METH-24: export a packed-only R8 int8 Qwen core for quality/C follow-up."""
import argparse
import json
from pathlib import Path
import sys
import time

from huggingface_hub import hf_hub_download
import psutil
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoConfig

import meth15_zero_residual_expert_smoke as M15
import meth20_half_adapter_generation as M20


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/donor_adaptation/ternary"))
import t2_rules as T2  # noqa: E402 -- same R8 rule as diagnostic and native exporter


MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-24 wall-time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-24 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-24 GPU stop: {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def classify(name, shape):
    if name == "model.embed_tokens.weight":
        return "tied_head"
    if name.startswith("model.layers.") and ".mlp." in name and len(shape) == 2:
        return "ffn"
    if name.startswith("model.layers.") and ".self_attn." in name and len(shape) == 2:
        return "attention"
    assert len(shape) == 1, (name, shape)
    return "control"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    start = time.monotonic()
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    source_path = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                                  revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source_path) == M15.M13.MODEL_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    config = AutoConfig.from_pretrained(M15.M13.MODEL, revision=M15.M13.REV,
                                        local_files_only=True)
    assert config.tie_word_embeddings is True
    tensors = {}
    organs = {key: {"matrices": 0, "parameters": 0,
                    "int8_code_bytes": 0, "fp32_row_scale_bytes": 0,
                    "max_relative_l2_error": 0.0, "max_abs_error": 0.0}
              for key in ("tied_head", "ffn", "attention", "control")}
    with safe_open(source_path, framework="pt", device="cpu") as source:
        names = source.keys()
        assert "model.embed_tokens.weight" in names
        assert "lm_head.weight" not in names
        for i, name in enumerate(names):
            original_cpu = source.get_tensor(name)
            shape = tuple(original_cpu.shape)
            organ = classify(name, shape)
            count = original_cpu.numel()
            organs[organ]["parameters"] += count
            if organ == "control":
                tensors[name] = original_cpu.float().contiguous()
                continue
            assert original_cpu.dtype == torch.bfloat16
            original = original_cpu.to(device).float()
            q, scale = T2.r8_int8_rtn(original)
            assert q.shape == original.shape and scale.shape == (shape[0], 1)
            codes = q.to(torch.int8)
            reconstructed = (q * scale).to(torch.bfloat16).float()
            error = reconstructed - original
            relative = float(torch.linalg.vector_norm(error) /
                             torch.linalg.vector_norm(original).clamp_min(1e-20))
            max_error = float(error.abs().max())
            tensors[name + ".q"] = codes.cpu().contiguous()
            tensors[name + ".scale"] = scale.flatten().cpu().contiguous()
            record = organs[organ]
            record["matrices"] += 1
            record["int8_code_bytes"] += count
            record["fp32_row_scale_bytes"] += shape[0] * 4
            record["max_relative_l2_error"] = max(record["max_relative_l2_error"], relative)
            record["max_abs_error"] = max(record["max_abs_error"], max_error)
            del original_cpu, original, q, scale, codes, reconstructed, error
            if (i+1) % 32 == 0:
                print(f"encoded {i+1}/{len(names)} source tensors", flush=True)
                check_budget(start, device)
    assert organs["tied_head"]["matrices"] == 1
    assert organs["ffn"]["matrices"] == 72
    assert organs["attention"]["matrices"] == 96
    assert sum(v["parameters"] for v in organs.values()) == 494032768
    assert sum(v["int8_code_bytes"] for v in organs.values()) == 493961216
    assert len(tensors) == 169 * 2 + 121
    metadata = {"experiment": "METH-24", "format": "QWEN25_R8_CORE_V1",
                "model": M15.M13.MODEL, "model_revision": M15.M13.REV,
                "model_sha256": M15.M13.MODEL_SHA,
                "adapter_sha256": M20.ADAPTER_SHA,
                "tokenizer_fingerprint": M15.M13.TOK_FP,
                "rule": "t2_rules.r8_int8_rtn per output row",
                "matrix_codes": "signed int8 [-127,127]",
                "row_scale_dtype": "float32",
                "control_dtype": "float32",
                "reconstruction_dtype": "bfloat16",
                "tied_head": "true", "matrix_count": "169"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=metadata)
    reloaded = load_file(str(out), device="cpu")
    assert reloaded.keys() == tensors.keys()
    assert all(torch.equal(reloaded[key], value) for key, value in tensors.items())
    with safe_open(str(out), framework="pt", device="cpu") as archive:
        assert archive.metadata() == metadata
    runtime = check_budget(start, device)
    report = {"experiment": "METH-24", "path": str(out.resolve()),
              "sha256": M15.M13.sha256(out), "bytes": out.stat().st_size,
              "tensor_count": len(tensors), "metadata": metadata,
              "organs": organs, "exact_reload_equal": True,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__},
              "decision": "int8_core_artifact_ready_for_independent_quality_not_native"}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": report["sha256"], "bytes": report["bytes"],
                      "tensor_count": report["tensor_count"],
                      "organs": organs, "runtime": report["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
