#!/usr/bin/env python3
"""Export a Qwen Instruct R8-head/BF16-attention/grouped-Q8-FFN core."""

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
M85_CORE = ROOT / "results/native_expert_scaling/meth85_qwen05b_instruct_group64_r8_core.safetensors"
M85_SHA = "c484a1130e495342d1b8644ff156cc002d0fd68fe12230604e59f35d3af6433a"
GROUP = 64
MAX_SECONDS = 15 * 60
MAX_RSS = 20 * (1 << 30)
MAX_GPU = int(10.5 * (1 << 30))
MAX_FILE = 1_000_000_000
EXPECTED_CORE_PAYLOAD = 548_701_184


def budget(start, device):
    record = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if record["seconds"] > MAX_SECONDS or record["rss_bytes"] > MAX_RSS or record[
            "gpu_peak_allocated_bytes"] > MAX_GPU:
        raise RuntimeError(f"METH-193 resource stop: {record}")
    return record


def quantize(weight):
    assert weight.ndim == 2 and weight.shape[1] % GROUP == 0
    rows, width = weight.shape
    blocks = weight.float().reshape(rows, width // GROUP, GROUP)
    maxima = blocks.abs().amax(dim=-1)
    scales = torch.where(maxima == 0, torch.ones_like(maxima), maxima / 127.0).half()
    assert bool(torch.isfinite(scales).all()) and bool((scales > 0).all())
    signed = torch.round(blocks / scales.float().unsqueeze(-1)).clamp(-127, 127)
    return signed.reshape(rows, width).to(torch.int8).contiguous(), scales.contiguous()


def reconstruct(packed, scales):
    assert packed.ndim == scales.ndim == 2 and packed.dtype == torch.int8
    rows, width = packed.shape
    assert width % GROUP == 0 and tuple(scales.shape) == (rows, width // GROUP)
    assert bool(((packed >= -127) & (packed <= 127)).all())
    return (packed.reshape(rows, width // GROUP, GROUP).float() *
            scales.float().unsqueeze(-1)).reshape(rows, width).bfloat16()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.report.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    stage = "bindings"
    device = None
    try:
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count())
                   if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
        assert len(matches) == 1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        source = Path(hf_hub_download(M42.MODEL, "model.safetensors",
                                      revision=M42.REV, local_files_only=True))
        assert M15.M13.sha256(source) == M57.MODEL_SHA
        assert M15.M13.sha256(M85_CORE) == M85_SHA
        sample = torch.zeros((2, 128), dtype=torch.bfloat16, device=device)
        sample[0, :64] = torch.linspace(-1, 1, 64, device=device).bfloat16()
        q_sample, s_sample = quantize(sample)
        assert tuple(q_sample.shape) == (2, 128)
        assert tuple(s_sample.shape) == (2, 2)
        assert bool((reconstruct(q_sample, s_sample)[0, 64:] == 0).all())
        assert bool((reconstruct(q_sample, s_sample)[1] == 0).all())

        stage = "pack_source_tensors"
        tensors = {}
        references = {}
        organs = {name: {"tensors": 0, "parameters": 0, "payload_bytes": 0,
                         "max_relative_l2_error": 0.0}
                  for name in ("tied_head", "attention", "ffn", "control")}
        with safe_open(str(source), framework="pt", device="cpu") as donor, \
             safe_open(str(M85_CORE), framework="pt", device="cpu") as old:
            assert old.metadata()["model_sha256"] == M57.MODEL_SHA
            names = list(donor.keys())
            assert len(names) == 290
            for index, name in enumerate(names):
                original = donor.get_tensor(name)
                organ = M24.classify(name, tuple(original.shape))
                record = organs[organ]
                record["tensors"] += 1
                record["parameters"] += original.numel()
                if organ == "ffn":
                    assert original.dtype == torch.bfloat16
                    packed, scales = quantize(original.to(device))
                    reference = reconstruct(packed, scales)
                    error = reference.float() - original.to(device).float()
                    relative = float(torch.linalg.vector_norm(error) /
                                     torch.linalg.vector_norm(original.float()).clamp_min(1e-20))
                    record["max_relative_l2_error"] = max(
                        record["max_relative_l2_error"], relative)
                    packed_cpu, scales_cpu = packed.cpu(), scales.cpu()
                    tensors[name + ".q8"] = packed_cpu
                    tensors[name + ".scale"] = scales_cpu
                    references[name] = reference.cpu()
                    record["payload_bytes"] += packed_cpu.numel() + scales_cpu.numel() * 2
                elif organ == "tied_head":
                    codes = old.get_tensor(name + ".q")
                    scales = old.get_tensor(name + ".scale")
                    assert codes.dtype == torch.int8 and scales.dtype == torch.float32
                    assert tuple(codes.shape) == tuple(original.shape)
                    assert tuple(scales.shape) == (original.shape[0],)
                    tensors[name + ".q"] = codes.contiguous()
                    tensors[name + ".scale"] = scales.contiguous()
                    record["payload_bytes"] += codes.numel() + scales.numel() * 4
                else:
                    stored = old.get_tensor(name)
                    assert stored.dtype == (torch.float32 if organ == "control"
                                            else torch.bfloat16)
                    assert torch.equal(stored.to(original.dtype), original)
                    tensors[name] = stored.contiguous()
                    record["payload_bytes"] += stored.numel() * stored.element_size()
                if (index + 1) % 32 == 0:
                    print(json.dumps({"packed_tensors": index + 1,
                                      "budget": budget(start, device)}), flush=True)
        assert {k: v["tensors"] for k, v in organs.items()} == {
            "tied_head": 1, "attention": 96, "ffn": 72, "control": 121}
        assert sum(v["parameters"] for v in organs.values()) == 494_032_768
        payload = sum(v["payload_bytes"] for v in organs.values())
        assert payload == EXPECTED_CORE_PAYLOAD
        assert len(tensors) == 363 and len(references) == 72
        route_bytes = {"parent_router": 24 * (64 * 896 + (8 + 16) * 64) * 4,
                       "child_projection": 24 * 32 * 896 * 4,
                       "selected_child_keys": 24 * 4 * 10 * 32 * 4,
                       "selected_shared_A_B": 24 * 4 * 8 * 896 * 2 * 2}
        ideal = payload + sum(route_bytes.values())
        assert ideal == 559_981_568 and ideal <= 560_000_000

        stage = "save_and_readback"
        metadata = {"experiment": "METH-193", "format": "QWEN25_INSTRUCT_R8H_GROUP64_Q8FFN_BF16ATTN_V1",
                    "model": M42.MODEL, "model_revision": M42.REV,
                    "model_sha256": M57.MODEL_SHA, "meth85_core_sha256": M85_SHA,
                    "ffn_codes": "signed symmetric int8 [-127,127]",
                    "ffn_group_width": "64", "ffn_scale_dtype": "float16",
                    "ffn_pack_order": "one signed byte per input weight",
                    "attention_dtype": "bfloat16", "head_rule": "exact METH-85 R8 head codes and scales",
                    "control_dtype": "float32", "tied_head": "true"}
        save_file(tensors, str(args.out), metadata=metadata)
        assert args.out.stat().st_size < MAX_FILE
        reread = load_file(str(args.out), device="cpu")
        assert reread.keys() == tensors.keys()
        assert all(torch.equal(reread[name], value) for name, value in tensors.items())
        for name, reference in references.items():
            actual = reconstruct(reread[name + ".q8"].to(device),
                                 reread[name + ".scale"].to(device)).cpu()
            assert torch.equal(actual, reference), name
            budget(start, device)
        with safe_open(str(args.out), framework="pt", device="cpu") as archive:
            assert archive.metadata() == metadata
        report = {"experiment": "METH-193-group64-Q8-stored-core",
                  "path": str(args.out.resolve()), "sha256": M15.M13.sha256(args.out),
                  "physical_bytes": args.out.stat().st_size,
                  "payload_bytes": payload, "header_bytes": args.out.stat().st_size - payload,
                  "source_sha256": M57.MODEL_SHA, "meth85_core_sha256": M85_SHA,
                  "tensor_count": len(tensors), "metadata": metadata,
                  "organs": organs, "route_factor_ideal_bytes": route_bytes,
                  "ideal_addressed_bytes_per_token": ideal,
                  "exact_reload_equal": True,
                  "all_ffn_bf16_reconstructions_equal": True,
                  "decision": "stored_q8_core_ready_for_e1280_development",
                  "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"decision": report["decision"],
                          "sha256": report["sha256"],
                          "physical_bytes": report["physical_bytes"],
                          "ideal_addressed_bytes_per_token": ideal,
                          "ffn_max_relative_l2": organs["ffn"]["max_relative_l2_error"],
                          "runtime": report["runtime"]}), flush=True)
    except BaseException as error:
        failure = args.report.with_name(args.report.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-193-export-failure",
                                       "stage": stage, "error": repr(error),
                                       "runtime": {"seconds": time.monotonic() - start,
                                                   "rss_bytes": psutil.Process().memory_info().rss,
                                                   "gpu_peak_allocated_bytes":
                                                       torch.cuda.max_memory_allocated(device)
                                                       if device is not None else None}},
                                      indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
