#!/usr/bin/env python3
"""Measure donor FFN output retained by activation-aware channel top-K."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

from huggingface_hub import hf_hub_download
import numpy as np
import psutil
from safetensors import safe_open
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
VECTORS = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth125_e1280_vectors.bin"
VECTORS_SHA = "f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699"
KS = (128, 256, 512, 1024, 2048)
STATIC_KS = (512, 1024, 2048)
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 12 * (1 << 30)
MAX_DISK = 1_000_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def budget(start, device):
    record = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if record["seconds"] > MAX_SECONDS or record["rss_bytes"] > MAX_RSS or record[
            "gpu_peak_allocated_bytes"] > MAX_GPU:
        raise RuntimeError(f"METH-185 resource stop: {record}")
    return record


def relative_errors(reference, candidate):
    ref = reference.float()
    got = candidate.float()
    norm = torch.linalg.vector_norm(ref, dim=-1)
    assert bool((norm > 0).all()) and bool(torch.isfinite(norm).all())
    relative = torch.linalg.vector_norm(got - ref, dim=-1) / norm
    assert bool(torch.isfinite(relative).all())
    return relative.cpu().numpy().astype(np.float64).tolist()


def masked_output(activation, down, indices):
    mask = torch.zeros(activation.shape, dtype=torch.bool, device=activation.device)
    mask.scatter_(1, indices, True)
    selected = torch.where(mask, activation, torch.zeros_like(activation))
    assert torch.count_nonzero(selected).item() <= indices.numel()
    return F.linear(selected, down)


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    assert a.ndim == 1 and a.size == 256 * 24 and np.isfinite(a).all()
    return {"count": int(a.size), "median_relative_l2": float(np.median(a)),
            "p95_relative_l2": float(np.quantile(a, .95)),
            "max_relative_l2": float(a.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    stage = "source_binding"
    device = None
    layer_rows = []
    try:
        assert digest(VECTORS) == VECTORS_SHA
        raw = VECTORS.read_bytes()
        assert struct.unpack("<8s4I", raw[:24]) == (b"M125HX01", 24, 256, 896, 1280)
        assert len(raw) == 24 + 256 * 24 * 896 * 2
        vectors = np.frombuffer(raw, dtype="<u2", offset=24).reshape(256, 24, 896)
        source = Path(hf_hub_download(MODEL, "model.safetensors",
                                      revision=REVISION, local_files_only=True))
        assert digest(source) == SOURCE_SHA
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count())
                   if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
        assert len(matches) == 1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        stage = "layer_outputs"
        with torch.inference_mode(), safe_open(str(source), framework="pt", device="cpu") as archive:
            for li in range(24):
                matrices = {}
                for organ in ("gate_proj", "up_proj", "down_proj"):
                    name = f"model.layers.{li}.mlp.{organ}.weight"
                    matrix = archive.get_tensor(name)
                    shape = ((896, 4864) if organ == "down_proj" else (4864, 896))
                    assert matrix.dtype == torch.bfloat16 and tuple(matrix.shape) == shape
                    matrices[organ] = matrix.to(device)
                x = torch.from_numpy(vectors[:, li, :].copy()).view(torch.bfloat16).to(device)
                gate = F.linear(x, matrices["gate_proj"])
                up = F.linear(x, matrices["up_proj"])
                activation = F.silu(gate) * up
                down = matrices["down_proj"]
                reference = F.linear(activation, down)
                assert bool(torch.isfinite(reference).all())
                colnorm = torch.linalg.vector_norm(down.float(), dim=0)
                scores = activation.float().abs() * colnorm[None, :]
                assert bool(torch.isfinite(scores).all())
                dynamic_order = torch.argsort(scores, dim=-1, descending=True, stable=True)
                static_order = torch.argsort(colnorm, descending=True, stable=True)
                full_indices = dynamic_order[:, :4864]
                full_output = masked_output(activation, down, full_indices)
                assert torch.equal(full_output, reference)
                errors = {"dynamic": {}, "static_norm": {}}
                for k in KS:
                    output = masked_output(activation, down, dynamic_order[:, :k])
                    errors["dynamic"][str(k)] = relative_errors(reference, output)
                for k in STATIC_KS:
                    indices = static_order[:k].expand(activation.shape[0], -1)
                    output = masked_output(activation, down, indices)
                    errors["static_norm"][str(k)] = relative_errors(reference, output)
                layer_rows.append({"layer": li, "state_count": 256,
                                   "full_k4864_exact_equal": True,
                                   "errors": errors,
                                   "median_relative_l2": {
                                       arm: {key: float(np.median(value))
                                             for key, value in choices.items()}
                                       for arm, choices in errors.items()}})
                print(json.dumps({"completed_layer": li,
                                  "budget": budget(start, device)}), flush=True)
                del matrices, x, gate, up, activation, down, reference
                del colnorm, scores, dynamic_order, static_order, full_output

        stage = "summary"
        summaries = {}
        for arm, ks in (("dynamic", KS), ("static_norm", STATIC_KS)):
            summaries[arm] = {}
            for k in ks:
                values = [error for row in layer_rows for error in row["errors"][arm][str(k)]]
                summaries[arm][str(k)] = summarize(values)
        gates = {}
        for k in STATIC_KS:
            dynamic = summaries["dynamic"][str(k)]["median_relative_l2"]
            static = summaries["static_norm"][str(k)]["median_relative_l2"]
            p95 = summaries["dynamic"][str(k)]["p95_relative_l2"]
            gates[str(k)] = {"dynamic_median_le_1pct": dynamic <= .01,
                             "dynamic_p95_le_5pct": p95 <= .05,
                             "dynamic_at_most_half_static_median": dynamic <= static / 2,
                             "joint_pass": dynamic <= .01 and p95 <= .05 and dynamic <= static / 2}
        chosen = next((k for k in STATIC_KS if gates[str(k)]["joint_pass"]), None)
        report = {"experiment": "METH-185-pretrained-FFN-channel-sparsity-upper-bound",
                  "source_sha256": SOURCE_SHA, "vectors_sha256": VECTORS_SHA,
                  "state_shape": [256, 24, 896], "ffn_width": 4864,
                  "candidate_ks": KS, "static_control_ks": STATIC_KS,
                  "layer_rows": layer_rows, "summary": summaries, "gates": gates,
                  "chosen_k": chosen,
                  "ideal_bf16_selected_ffn_bytes_per_token":
                      {str(k): 24 * 3 * 896 * k * 2 for k in KS},
                  "decision": "channel_oracle_pass_router_required" if chosen is not None
                              else "channel_oracle_fail",
                  "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)},
                  "scope": "Actual BF16 E1280 states; oracle uses full gate/up; no model quality or speed"}
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < MAX_DISK
        print(json.dumps({"decision": report["decision"], "chosen_k": chosen,
                          "dynamic": summaries["dynamic"],
                          "static": summaries["static_norm"],
                          "runtime": report["runtime"]}), flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-185-failure",
                                       "stage": stage, "error": repr(error),
                                       "partial_layer_rows": layer_rows,
                                       "elapsed_seconds": time.monotonic() - start,
                                       "rss_bytes": psutil.Process().memory_info().rss,
                                       "gpu_peak_allocated_bytes":
                                           torch.cuda.max_memory_allocated(device)
                                           if device is not None else None},
                                      indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
