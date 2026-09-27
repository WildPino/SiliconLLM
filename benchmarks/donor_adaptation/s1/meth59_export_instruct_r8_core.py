"""Export the bound Qwen0.5B-Instruct donor as a packed-only R8 core."""

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import psutil
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoConfig

import meth15_zero_residual_expert_smoke as M15
import meth24_export_r8_core as M24
import meth42_instruct_prompt_manifest as M42
import meth57_product_key_external_audit as M57


ROOT = Path(__file__).resolve().parents[3]
MAX_SECONDS = 15 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES or peak > MAX_GPU_BYTES:
        raise RuntimeError(f"METH-59 export budget: {elapsed:.1f}s, RSS {rss}, GPU {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
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
    config = AutoConfig.from_pretrained(M42.MODEL, revision=M42.REV,
                                         local_files_only=True)
    assert config.tie_word_embeddings is True
    assert M15.M13.sha256(M57.EXTERNAL) == M57.EXTERNAL_SHA
    training_path = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth56_product_key_retention_result.json"
    assert M15.M13.sha256(training_path) == M57.TRAINING_SHA
    training = json.loads(training_path.read_text(encoding="utf-8"))
    checkpoint = training["checkpoints"]["512"]
    assert checkpoint["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(checkpoint["path"]) == M57.CHECKPOINT_SHA
    tensors = {}
    organs = {key: {"matrices": 0, "parameters": 0,
                    "int8_code_bytes": 0, "fp32_row_scale_bytes": 0,
                    "max_relative_l2_error": 0.0, "max_abs_error": 0.0}
              for key in ("tied_head", "ffn", "attention", "control")}
    with safe_open(str(source), framework="pt", device="cpu") as archive:
        names = list(archive.keys())
        assert "model.embed_tokens.weight" in names and "lm_head.weight" not in names
        for i, name in enumerate(names):
            original_cpu = archive.get_tensor(name)
            shape = tuple(original_cpu.shape)
            organ = M24.classify(name, shape)
            organs[organ]["parameters"] += original_cpu.numel()
            if organ == "control":
                tensors[name] = original_cpu.float().contiguous()
                continue
            assert original_cpu.dtype == torch.bfloat16
            original = original_cpu.to(device).float()
            q, scale = M24.T2.r8_int8_rtn(original)
            assert q.shape == original.shape and scale.shape == (shape[0], 1)
            assert bool((q >= -127).all()) and bool((q <= 127).all())
            reconstructed = (q * scale).to(torch.bfloat16).float()
            error = reconstructed - original
            relative = float(torch.linalg.vector_norm(error) /
                             torch.linalg.vector_norm(original).clamp_min(1e-20))
            codes = q.to(torch.int8)
            tensors[name + ".q"] = codes.cpu().contiguous()
            tensors[name + ".scale"] = scale.flatten().cpu().contiguous()
            record = organs[organ]
            record["matrices"] += 1
            record["int8_code_bytes"] += original.numel()
            record["fp32_row_scale_bytes"] += shape[0] * 4
            record["max_relative_l2_error"] = max(record["max_relative_l2_error"], relative)
            record["max_abs_error"] = max(record["max_abs_error"], float(error.abs().max()))
            del original_cpu, original, q, scale, codes, reconstructed, error
            if (i + 1) % 32 == 0:
                print(f"encoded {i+1}/{len(names)} tensors", flush=True)
                budget(start, device)
    assert organs["tied_head"]["matrices"] == 1
    assert organs["ffn"]["matrices"] == 72
    assert organs["attention"]["matrices"] == 96
    assert sum(v["parameters"] for v in organs.values()) == 494032768
    assert sum(v["int8_code_bytes"] for v in organs.values()) == 493961216
    assert len(tensors) == 459
    metadata = {"experiment": "METH-59", "format": "QWEN25_INSTRUCT_R8_CORE_V1",
                "model": M42.MODEL, "model_revision": M42.REV,
                "model_sha256": M57.MODEL_SHA,
                "product_key_checkpoint_sha256": M57.CHECKPOINT_SHA,
                "source_manifest_sha256": M57.EXTERNAL_SHA,
                "rule": "t2_rules.r8_int8_rtn per output row",
                "matrix_codes": "signed int8 [-127,127]",
                "row_scale_dtype": "float32", "control_dtype": "float32",
                "reconstruction_dtype": "bfloat16", "tied_head": "true",
                "matrix_count": "169"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=metadata)
    assert args.out.stat().st_size <= 520_000_000
    reloaded = load_file(str(args.out), device="cpu")
    assert reloaded.keys() == tensors.keys()
    assert all(torch.equal(reloaded[key], value) for key, value in tensors.items())
    with safe_open(str(args.out), framework="pt", device="cpu") as check:
        assert check.metadata() == metadata
    report = {"experiment": "METH-59-Instruct-R8-packed-core-export",
              "path": str(args.out.resolve()),
              "sha256": M15.M13.sha256(args.out),
              "bytes": args.out.stat().st_size,
              "tensor_count": len(tensors), "metadata": metadata,
              "organs": organs, "exact_reload_equal": True,
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__},
              "decision": "stored_core_ready_for_composition_diagnostic"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": report["sha256"], "bytes": report["bytes"],
                      "runtime": report["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
