"""Compose exact METH-59 R8 head/FFN with source BF16 attention."""

import argparse
import json
from pathlib import Path
import time

import psutil
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth24_export_r8_core as M24
import meth42_instruct_prompt_manifest as M42
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57
import meth59_instruct_r8_composition as M59


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MAX_SECONDS = 15 * 60
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES:
        raise RuntimeError(f"METH-62 export budget: {elapsed:.1f}s, RSS {rss}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    torch.set_num_threads(6)
    assert [i for i in range(torch.cuda.device_count())
            if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    from huggingface_hub import hf_hub_download
    source_path = Path(hf_hub_download(M42.MODEL, "model.safetensors",
                                       revision=M42.REV, local_files_only=True))
    assert M15.M13.sha256(source_path) == M57.MODEL_SHA
    assert M15.M13.sha256(M59.CORE) == M59.CORE_SHA
    training_path = DIR / "meth56_product_key_retention_result.json"
    assert M15.M13.sha256(training_path) == M57.TRAINING_SHA
    training = json.loads(training_path.read_text(encoding="utf-8"))
    checkpoint = training["checkpoints"]["512"]
    assert checkpoint["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(checkpoint["path"]) == M57.CHECKPOINT_SHA
    tensors = {}
    organs = {key: {"matrices": 0, "parameters": 0,
                    "stored_bytes": 0, "format": ""}
              for key in ("tied_head", "attention", "ffn", "control")}
    with safe_open(str(source_path), framework="pt", device="cpu") as source, \
         safe_open(str(M59.CORE), framework="pt", device="cpu") as r8:
        assert r8.metadata()["model_sha256"] == M57.MODEL_SHA
        assert r8.metadata()["product_key_checkpoint_sha256"] == M57.CHECKPOINT_SHA
        names = list(source.keys())
        assert len(names) == 290
        for i, name in enumerate(names):
            original = source.get_tensor(name)
            organ = M24.classify(name, tuple(original.shape))
            record = organs[organ]
            record["parameters"] += original.numel()
            if organ == "attention":
                assert original.dtype == torch.bfloat16
                tensors[name] = original.contiguous()
                record["stored_bytes"] += original.numel() * 2
                record["format"] = "bfloat16"
            elif organ == "control":
                stored = r8.get_tensor(name)
                assert stored.dtype == torch.float32
                assert torch.equal(stored.to(original.dtype), original)
                tensors[name] = stored.contiguous()
                record["stored_bytes"] += stored.numel() * 4
                record["format"] = "float32"
            else:
                assert original.dtype == torch.bfloat16
                codes = r8.get_tensor(name + ".q")
                scale = r8.get_tensor(name + ".scale")
                assert codes.dtype == torch.int8 and tuple(codes.shape) == tuple(original.shape)
                assert scale.dtype == torch.float32 and tuple(scale.shape) == (original.shape[0],)
                tensors[name + ".q"] = codes.contiguous()
                tensors[name + ".scale"] = scale.contiguous()
                record["stored_bytes"] += codes.numel() + scale.numel() * 4
                record["format"] = "R8_int8_row_scale_fp32"
            record["matrices"] += int(original.ndim == 2)
            if (i + 1) % 64 == 0:
                budget(start)
    assert [organs[k]["matrices"] for k in ("tied_head", "attention", "ffn", "control")] == [1, 96, 72, 0]
    assert sum(v["parameters"] for v in organs.values()) == 494032768
    assert len(tensors) == 363
    core_bytes = sum(v["stored_bytes"] for v in organs.values())
    router_bytes = M15.M13.L * (M55.ROUTER_RANK * M15.M13.D +
                    (M55.AXIS_A + M55.AXIS_B) * M55.ROUTER_RANK) * 4
    selected_bytes = M15.M13.L * M15.K * 2 * M15.R * M15.M13.D * 2
    assert (core_bytes, router_bytes, selected_bytes) == (539915264, 5652480, 2752512)
    assert core_bytes + router_bytes + selected_bytes == 548320256 <= 560000000
    metadata = {"experiment": "METH-62", "format": "QWEN25_INSTRUCT_R8H_FFN_BF16ATTN_V1",
                "model": M42.MODEL, "model_revision": M42.REV,
                "model_sha256": M57.MODEL_SHA,
                "product_key_checkpoint_sha256": M57.CHECKPOINT_SHA,
                "r8_core_sha256": M59.CORE_SHA,
                "head_ffn_rule": "exact stored METH-59 R8 rows",
                "attention_dtype": "bfloat16", "control_dtype": "float32",
                "tied_head": "true", "matrix_count": "169"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=metadata)
    reloaded = load_file(str(args.out), device="cpu")
    assert reloaded.keys() == tensors.keys()
    assert all(torch.equal(reloaded[key], value) for key, value in tensors.items())
    with safe_open(str(args.out), framework="pt", device="cpu") as archive:
        assert archive.metadata() == metadata and len(archive.keys()) == 363
    report = {"experiment": "METH-62-stored-mixed-Instruct-core-export",
              "source_sha256": M57.MODEL_SHA,
              "checkpoint_sha256": M57.CHECKPOINT_SHA,
              "r8_core_sha256": M59.CORE_SHA,
              "path": str(args.out.resolve()),
              "sha256": M15.M13.sha256(args.out),
              "physical_bytes": args.out.stat().st_size,
              "tensor_count": len(tensors), "metadata": metadata,
              "organs": organs, "exact_reload_equal": True,
              "ideal_addressed_bytes": {"core": core_bytes,
                  "router": router_bytes, "selected_factors": selected_bytes,
                  "total": core_bytes + router_bytes + selected_bytes},
              "runtime": {**budget(start), "available_gpu": "NVIDIA GeForce RTX 3060",
                          "torch": torch.__version__},
              "decision": "stored_mixed_core_ready_for_new_external_audit"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": report["sha256"],
                      "physical_bytes": report["physical_bytes"],
                      "ideal_addressed_bytes": report["ideal_addressed_bytes"],
                      "runtime": report["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
