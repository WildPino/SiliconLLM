#!/usr/bin/env python3
"""METH-82: export Qwen Instruct BF16 head/attention and grouped-Q4 FFN."""

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import psutil
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth24_export_r8_core as M24
import meth42_instruct_prompt_manifest as M42
import meth57_product_key_external_audit as M57


ROOT = Path(__file__).resolve().parents[3]
GROUP = 64
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)
MAX_FILE_BYTES = 528_000_000


def budget(start, device):
    report = {"elapsed_seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (report["elapsed_seconds"] > MAX_SECONDS or report["rss_bytes"] > MAX_RSS_BYTES
            or report["gpu_peak_allocated_bytes"] > MAX_GPU_BYTES):
        raise RuntimeError(f"METH-82 export budget exceeded: {report}")
    return report


def quantize(weight):
    assert weight.ndim == 2 and weight.shape[1] % GROUP == 0
    rows, width = weight.shape
    blocks = weight.float().reshape(rows, width // GROUP, GROUP)
    maxima = blocks.abs().amax(dim=-1)
    scale = torch.where(maxima == 0, torch.ones_like(maxima), maxima / 7.0)
    scale = scale.to(torch.float16)
    assert bool(torch.isfinite(scale).all()) and bool((scale > 0).all())
    signed = torch.round(blocks / scale.float().unsqueeze(-1)).clamp(-7, 7)
    signed = signed.to(torch.int16).reshape(rows, width)
    assert bool((signed >= -7).all()) and bool((signed <= 7).all())
    nibble = (signed + 8).to(torch.uint8)
    packed = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).contiguous()
    return packed, scale.contiguous()


def reconstruct(packed, scale):
    assert packed.ndim == 2 and scale.ndim == 2
    rows, half_width = packed.shape
    width = 2 * half_width
    assert width % GROUP == 0
    assert scale.shape == (rows, width // GROUP)
    low = (packed & 15).to(torch.int16) - 8
    high = (packed >> 4).to(torch.int16) - 8
    signed = torch.stack((low, high), dim=-1).reshape(rows, width)
    assert bool((signed >= -7).all()) and bool((signed <= 7).all())
    return (signed.reshape(rows, width // GROUP, GROUP).float()
            * scale.float().unsqueeze(-1)).reshape(rows, width).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    source = Path(hf_hub_download(M42.MODEL, "model.safetensors",
                                  revision=M42.REV, local_files_only=True))
    assert M15.M13.sha256(source) == M57.MODEL_SHA
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(M42.MODEL, revision=M42.REV,
                                        local_files_only=True)
    assert config.tie_word_embeddings is True
    training_path = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth56_product_key_retention_result.json"
    assert M15.M13.sha256(training_path) == M57.TRAINING_SHA
    training = json.loads(training_path.read_text(encoding="utf-8"))
    checkpoint = training["checkpoints"]["512"]
    assert checkpoint["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(checkpoint["path"]) == M57.CHECKPOINT_SHA

    # Exercise code order and the all-zero-group rule before donor export.
    sample = torch.zeros((2, 128), dtype=torch.float32, device=device)
    sample[0, :64] = torch.linspace(-1.0, 1.0, 64, device=device)
    sample[1, 64:] = torch.linspace(-0.25, 0.25, 64, device=device)
    sample_codes, sample_scale = quantize(sample)
    sample_rec = reconstruct(sample_codes, sample_scale)
    assert sample_codes.shape == (2, 64) and sample_scale.shape == (2, 2)
    assert bool((sample_rec[0, 64:] == 0).all())
    assert bool((sample_rec[1, :64] == 0).all())

    tensors = {}
    recon_reference = {}
    organs = {name: {"tensors": 0, "parameters": 0, "stored_payload_bytes": 0,
                     "max_relative_l2_error": 0.0, "max_abs_error": 0.0}
              for name in ("tied_head", "ffn", "attention", "control")}
    with safe_open(str(source), framework="pt", device="cpu") as archive:
        names = list(archive.keys())
        assert "model.embed_tokens.weight" in names and "lm_head.weight" not in names
        for index, name in enumerate(names):
            original_cpu = archive.get_tensor(name)
            shape = tuple(original_cpu.shape)
            organ = M24.classify(name, shape)
            record = organs[organ]
            record["tensors"] += 1
            record["parameters"] += original_cpu.numel()
            if organ == "control":
                stored = original_cpu.float().contiguous()
                tensors[name] = stored
                record["stored_payload_bytes"] += stored.numel() * stored.element_size()
            elif organ in ("tied_head", "attention"):
                assert original_cpu.dtype == torch.bfloat16
                stored = original_cpu.contiguous()
                tensors[name] = stored
                record["stored_payload_bytes"] += stored.numel() * stored.element_size()
            else:
                assert organ == "ffn" and original_cpu.dtype == torch.bfloat16
                original = original_cpu.to(device).float()
                assert bool(torch.isfinite(original).all())
                packed, scale = quantize(original)
                recon = reconstruct(packed, scale)
                error = recon.float() - original
                relative = float(torch.linalg.vector_norm(error)
                                 / torch.linalg.vector_norm(original).clamp_min(1e-20))
                record["max_relative_l2_error"] = max(
                    record["max_relative_l2_error"], relative)
                record["max_abs_error"] = max(record["max_abs_error"],
                                               float(error.abs().max()))
                codes_cpu = packed.cpu()
                scale_cpu = scale.cpu()
                tensors[name + ".q4"] = codes_cpu
                tensors[name + ".scale"] = scale_cpu
                recon_reference[name] = recon.cpu()
                record["stored_payload_bytes"] += (
                    codes_cpu.numel() * codes_cpu.element_size()
                    + scale_cpu.numel() * scale_cpu.element_size())
                del original, packed, scale, recon, error, codes_cpu, scale_cpu
            del original_cpu
            if (index + 1) % 32 == 0:
                budget(start, device)
                print(f"encoded {index + 1}/{len(names)} source tensors", flush=True)
    assert {name: row["tensors"] for name, row in organs.items()} == {
        "tied_head": 1, "ffn": 72, "attention": 96, "control": 121}
    assert sum(row["parameters"] for row in organs.values()) == 494_032_768
    payload_bytes = sum(row["stored_payload_bytes"] for row in organs.values())
    assert payload_bytes == 527_334_912
    assert len(tensors) == 362 and len(recon_reference) == 72
    metadata = {
        "experiment": "METH-82", "format": "QWEN25_INSTRUCT_Q4_FFN_BF16_HEAD_ATTN_V1",
        "model": M42.MODEL, "model_revision": M42.REV,
        "model_sha256": M57.MODEL_SHA,
        "product_key_checkpoint_sha256": M57.CHECKPOINT_SHA,
        "ffn_codes": "signed symmetric int4 [-7,7] stored as unsigned code + 8",
        "nibble_order": "even input index low; odd input index high",
        "group_width": str(GROUP), "scale_dtype": "float16",
        "attention_dtype": "bfloat16", "tied_head_dtype": "bfloat16",
        "control_dtype": "float32", "reconstruction_dtype": "bfloat16",
        "tied_head": "true", "source_tensor_count": "290",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=metadata)
    assert args.out.stat().st_size <= MAX_FILE_BYTES
    reloaded = load_file(str(args.out), device="cpu")
    assert reloaded.keys() == tensors.keys()
    assert all(torch.equal(reloaded[key], value) for key, value in tensors.items())
    for name, reference in recon_reference.items():
        codes = reloaded[name + ".q4"].to(device)
        scale = reloaded[name + ".scale"].to(device)
        readback = reconstruct(codes, scale).cpu()
        assert torch.equal(readback, reference), name
        budget(start, device)
    with safe_open(str(args.out), framework="pt", device="cpu") as archive:
        assert archive.metadata() == metadata
    report = {
        "experiment": "METH-82-Q4-FFN-packed-core-export",
        "path": str(args.out.resolve()), "sha256": M15.M13.sha256(args.out),
        "physical_file_bytes": args.out.stat().st_size,
        "stored_tensor_payload_bytes": payload_bytes,
        "format_header_and_alignment_bytes": args.out.stat().st_size - payload_bytes,
        "source_model_sha256": M57.MODEL_SHA,
        "expert_checkpoint_sha256": M57.CHECKPOINT_SHA,
        "tensor_count": len(tensors), "metadata": metadata,
        "organs": organs, "exact_reload_equal": True,
        "all_ffn_bf16_reconstruction_readbacks_equal": True,
        "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                    "cuda_index": matches[0], "torch": torch.__version__},
        "decision": "stored_core_ready_for_fresh_composition_development",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"],
                      "file_bytes": report["physical_file_bytes"],
                      "sha256": report["sha256"],
                      "ffn_max_relative_l2_error": organs["ffn"]["max_relative_l2_error"],
                      "runtime": report["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
