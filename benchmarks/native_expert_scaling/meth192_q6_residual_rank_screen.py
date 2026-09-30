#!/usr/bin/env python3
"""Exact Gram-spectrum screen of stored-Q6 FFN quantization residuals."""

import argparse
import json
import math
from pathlib import Path
import statistics
import time

from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch

import meth187_q6_core_e1280_development as P


RANKS = (64, 94, 128, 256)
BYTES_PER_RANK = 829_440


def summarize(rows):
    result = {}
    for organ in ("all", "gate_proj", "up_proj", "down_proj"):
        subset = rows if organ == "all" else [row for row in rows if row["organ"] == organ]
        assert len(subset) == (72 if organ == "all" else 24)
        energy = sum(row["residual_squared_norm"] for row in subset)
        captures = {}
        for rank in RANKS:
            key = str(rank)
            fractions = [row["captured_fraction"][key] for row in subset]
            captures[key] = {"median": statistics.median(fractions),
                             "minimum": min(fractions),
                             "weighted": sum(row["residual_squared_norm"] *
                                             row["captured_fraction"][key]
                                             for row in subset) / energy}
        result[organ] = {"matrices": len(subset),
                         "residual_squared_norm": energy,
                         "captured_fraction": captures,
                         "rank_for_95pct_median": statistics.median(
                             row["rank_for_95pct"] for row in subset)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assert not args.out.exists()
    report = json.loads(P.EXPORT.read_text(encoding="utf-8"))
    assert P.digest(P.CORE) == report["sha256"]
    assert report["ideal_addressed_bytes_per_token"] == 481_534_976
    source_path = Path(hf_hub_download(P.M42.MODEL, "model.safetensors",
                                       revision=P.M42.REV, local_files_only=True))
    assert P.digest(source_path) == P.M57.MODEL_SHA
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    rows = []
    with safe_open(str(source_path), framework="pt", device="cpu") as source, \
            safe_open(str(P.CORE), framework="pt", device="cpu") as core:
        assert core.metadata()["format"] == "QWEN25_INSTRUCT_R8H_GROUP64_Q6FFN_BF16ATTN_V1"
        assert core.metadata()["model_sha256"] == P.M57.MODEL_SHA
        names = [name for name in source.keys() if
                 any(name.endswith(".mlp." + organ + ".weight") for organ in
                     ("gate_proj", "up_proj", "down_proj"))]
        assert len(names) == 72
        for index, name in enumerate(sorted(names), 1):
            original = source.get_tensor(name)
            assert original.dtype == torch.bfloat16
            assert P.M24.classify(name, tuple(original.shape)) == "ffn"
            packed = core.get_tensor(name + ".q6").to(device)
            scale = core.get_tensor(name + ".scale").to(device)
            reconstructed = P.M186.reconstruct(packed, scale)
            assert reconstructed.shape == original.shape
            residual = original.to(device).float() - reconstructed.float()
            direct_energy = float(residual.square().sum())
            assert direct_energy > 0
            gram = residual @ residual.T if residual.shape[0] <= residual.shape[1] \
                   else residual.T @ residual
            assert gram.shape == (896, 896)
            eigen = torch.linalg.eigvalsh(gram).cpu().double()
            minimum_eigen = float(eigen.min())
            eigen.clamp_(min=0)
            eigen = eigen.flip(0)
            total = float(eigen.sum())
            relative_sum_error = abs(total - direct_energy) / direct_energy
            assert relative_sum_error <= .001, (name, relative_sum_error)
            cumulative = eigen.cumsum(0)
            fractions = {str(rank): float(cumulative[rank - 1] / total)
                         for rank in RANKS}
            rank95 = int(torch.searchsorted(cumulative,
                                             .95 * total).item()) + 1
            organ = next(value for value in ("gate_proj", "up_proj", "down_proj")
                         if name.endswith("." + value + ".weight"))
            rows.append({"name": name, "organ": organ,
                         "shape": list(original.shape),
                         "residual_squared_norm": direct_energy,
                         "relative_weight_l2": math.sqrt(
                             direct_energy / float(original.to(device).float().square().sum())),
                         "captured_fraction": fractions,
                         "rank_for_95pct": rank95,
                         "relative_gram_sum_error": relative_sum_error,
                         "minimum_raw_eigenvalue": minimum_eigen})
            P.budget(start, device)
            if index % 12 == 0:
                print(json.dumps({"completed_matrices": index,
                                  "runtime": P.budget(start, device)}), flush=True)
    summary = summarize(rows)
    result = {"experiment": "METH-192-Q6-FFN-residual-rank-screen",
              "source_sha256": P.M57.MODEL_SHA,
              "core_sha256": report["sha256"],
              "ranks": RANKS,
              "bytes_per_rank": BYTES_PER_RANK,
              "max_rank_under_560mb": 94,
              "rank94_ideal_addressed_bytes_per_token":
                  481_534_976 + 94 * BYTES_PER_RANK,
              "rows": rows, "summary": summary,
              "gate": {"rank94_median_capture_ge_50pct":
                       summary["all"]["captured_fraction"]["94"]["median"] >= .5},
              "runtime": {**P.budget(start, device),
                          "gpu": torch.cuda.get_device_name(device)},
              "scope": "Weight residual only; no model quality inference"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.out.stat().st_size < P.MAX_DISK
    print(json.dumps({"all": summary["all"], "gate": result["gate"],
                      "runtime": result["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
