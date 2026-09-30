#!/usr/bin/env python3
"""Compare METH-182 native grouped-R8 FFN with the stored-core BF16 oracle."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct
import time

import numpy as np
import psutil
from safetensors import safe_open
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
CORE = ROOT / "results/native_expert_scaling/meth85_qwen05b_instruct_group64_r8_core.safetensors"
CORE_SHA = "c484a1130e495342d1b8644ff156cc002d0fd68fe12230604e59f35d3af6433a"
VECTORS = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth125_e1280_vectors.bin"
VECTORS_SHA = "f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699"
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reconstruct(codes, scales, device):
    rows, width = codes.shape
    assert width % 64 == 0 and scales.shape == (rows, width // 64)
    return (codes.to(device).reshape(rows, width // 64, 64).float() *
            scales.to(device).float().unsqueeze(-1)).reshape(rows, width).bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    start = time.monotonic()
    assert digest(CORE) == CORE_SHA and digest(VECTORS) == VECTORS_SHA
    export = json.loads(args.export_report.read_text(encoding="utf-8"))
    assert export["source_core_sha256"] == CORE_SHA
    assert digest(Path(export["binary"]["path"])) == export["binary"]["sha256"]
    assert len(export["rows"]) == 72
    raw = VECTORS.read_bytes()
    assert struct.unpack("<8s4I", raw[:24]) == (b"M125HX01", 24, 256, 896, 1280)
    ids = np.frombuffer(raw, dtype="<u2", offset=24).reshape(256, 24, 896)
    native = args.native_output.read_bytes()
    assert len(native) == 20 + 16 * 24 * 896 * 4
    assert struct.unpack("<8sIII", native[:20]) == (b"M182OUT1", 16, 24, 896)
    observed = np.frombuffer(native, dtype="<f4", offset=20).reshape(16, 24, 896)
    assert np.isfinite(observed).all()
    timing = json.loads(args.timing.read_text(encoding="utf-8"))
    executable_sha = digest(args.exe)
    assert timing["threads"] == 6 and timing["tokens"] == 256 and timing["layers"] == 24
    assert len(timing["pass_ms_per_token"]) == 3
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(6)
    rows = []
    with torch.no_grad(), safe_open(CORE, framework="pt", device="cpu") as archive:
        for layer in range(24):
            matrices = {}
            for organ in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.{layer}.mlp.{organ}.weight"
                q = archive.get_tensor(name + ".q")
                scale = archive.get_tensor(name + ".scale")
                assert q.dtype == torch.int8 and scale.dtype == torch.float16
                matrices[organ] = reconstruct(q, scale, device)
            x = torch.from_numpy(ids[:16, layer].copy()).view(torch.bfloat16).to(device)
            gate = F.linear(x, matrices["gate_proj"])
            up = F.linear(x, matrices["up_proj"])
            reference = F.linear(F.silu(gate) * up, matrices["down_proj"])
            expected = reference.float().cpu().numpy().astype(np.float64)
            actual = observed[:, layer].astype(np.float64)
            for token in range(16):
                denominator = float(np.linalg.norm(expected[token]))
                assert denominator > 0
                relative = float(np.linalg.norm(actual[token] - expected[token]) / denominator)
                rows.append({"token": token, "layer": layer, "relative_l2": relative,
                             "reference_l2": denominator,
                             "output_l2": float(np.linalg.norm(actual[token]))})
            del matrices, x, gate, up, reference
            if time.monotonic() - start > MAX_SECONDS or torch.cuda.max_memory_allocated(device) > MAX_GPU:
                raise RuntimeError("METH-182 reference resource stop")
    errors = [row["relative_l2"] for row in rows]
    median_ms = statistics.median(timing["pass_ms_per_token"])
    numeric_pass = statistics.median(errors) <= .01 and max(errors) <= .05
    rate_pass = median_ms <= 10.0
    report = {"experiment": "METH-182-actual-stored-group64-R8-native-FFN",
              "core_sha256": CORE_SHA, "vectors_sha256": VECTORS_SHA,
              "export_binary_sha256": export["binary"]["sha256"],
              "native_executable_sha256": executable_sha,
              "native_output_sha256": digest(args.native_output),
              "summary": {"comparison_rows": len(rows),
                          "median_relative_l2": statistics.median(errors),
                          "maximum_relative_l2": max(errors),
                          "median_24_layer_ffn_ms_per_token": median_ms,
                          "numeric_gate_pass": numeric_pass,
                          "cost_gate_pass": rate_pass},
              "timing": timing, "rows": rows,
              "decision": ("component_feasible_quality_and_full_rate_pending" if numeric_pass and rate_pass
                           else "stop_this_group64_cpu_path_at_component_gate"),
              "runtime": {"seconds": time.monotonic() - start,
                          "rss_bytes": psutil.Process().memory_info().rss,
                          "gpu_peak_bytes": torch.cuda.max_memory_allocated(device)},
              "scope": "Actual stored FFN and BF16 model states, but component only; no true METH-85 states or full-model rate"}
    assert report["runtime"]["seconds"] <= MAX_SECONDS
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "decision": report["decision"],
                      "result_sha256": digest(args.out)}), flush=True)


if __name__ == "__main__":
    main()
