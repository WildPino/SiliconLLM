#!/usr/bin/env python3
"""Measure C-engine expert selection on NES-01 free-generation trajectories."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from score_nes01_greedy import max_ngram_count, read_generations, sha256


def read_routes(path: Path, experts: int) -> list[tuple[int, np.ndarray]]:
    with path.open("rb") as stream:
        header = stream.read(24)
        if len(header) != 24:
            raise ValueError("short route header")
        magic, count, gen_len, layers, k, found_experts = struct.unpack("<6I", header)
        if (magic, count, gen_len, layers, k, found_experts) != (0x4E523031, 16, 128, 6, 8, experts):
            raise ValueError("unexpected route shape or expert count")
        rows = []
        for index in range(count):
            raw = stream.read(4 + 2 * gen_len * layers * k)
            if len(raw) != 4 + 2 * gen_len * layers * k:
                raise ValueError("short route row")
            offset = struct.unpack_from("<I", raw)[0]
            route = np.frombuffer(raw, dtype="<u2", offset=4).reshape(gen_len, layers, k)
            if offset != 512 + 2048 * index or int(route.max()) >= experts:
                raise ValueError("bad route offset or expert ID")
            if np.any(np.diff(np.sort(route, axis=-1), axis=-1) == 0):
                raise ValueError("duplicate selected expert in one token/layer")
            rows.append((offset, route.copy()))
        if stream.read(1):
            raise ValueError("trailing route bytes")
    return rows


def route_stats(route: np.ndarray, experts: int) -> dict[str, float | int]:
    # route: positions x top-k; positions from one continuation or all 16 pooled.
    k = route.shape[-1]
    counts = np.bincount(route.reshape(-1), minlength=experts)
    chosen = np.zeros((len(route), experts), dtype=bool)
    chosen[np.arange(len(route))[:, None], route] = True
    overlap = (chosen[1:] & chosen[:-1]).sum(axis=1) / k
    return {
        "dead_experts": int((counts == 0).sum()),
        "max_to_mean_load": float(counts.max() / counts.mean()),
        "distinct_experts": int((counts > 0).sum()),
        "mean_adjacent_selected_overlap": float(overlap.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("e32", "e128"):
        parser.add_argument(f"--{arm}-routes", type=Path, required=True)
        parser.add_argument(f"--{arm}-generation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    result = {"schema": "nes01_c_free_generation_routes_v1",
              "scope": "C fp32+exact free-generation route IDs for 16 frozen validation prefixes; descriptive, no causal inference",
              "arms": {}}
    for name, experts in (("e32", 32), ("e128", 128)):
        route_path = getattr(args, f"{name}_routes")
        generation_path = getattr(args, f"{name}_generation")
        routes = read_routes(route_path, experts)
        generations = read_generations(generation_path, 1024)
        samples = []
        for (offset, route), (gen_offset, tokens) in zip(routes, generations):
            if offset != gen_offset:
                raise ValueError("route/generation offsets differ")
            samples.append({
                "validation_offset": offset,
                "triple_8gram": max_ngram_count(tokens) >= 3,
                "max_8gram_count": max_ngram_count(tokens),
                "layers": [route_stats(route[:, layer, :], experts) for layer in range(6)],
            })
        all_routes = np.stack([route for _, route in routes])
        per_layer = [route_stats(all_routes[:, :, layer, :].reshape(-1, 8), experts) for layer in range(6)]
        # For overall persistence, remove adjacency between independent samples.
        for layer, stats in enumerate(per_layer):
            stats["mean_adjacent_selected_overlap"] = float(np.mean([
                sample["layers"][layer]["mean_adjacent_selected_overlap"] for sample in samples]))
        result["arms"][name] = {
            "route_file_sha256": sha256(route_path),
            "generation_file_sha256": sha256(generation_path),
            "triple_8gram_continuations": sum(sample["triple_8gram"] for sample in samples),
            "validation_iid_expected_adjacent_overlap": 8 / experts,
            "per_layer": per_layer,
            "samples": samples,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for arm, row in result["arms"].items():
        print(arm, "loops", row["triple_8gram_continuations"],
              "dead", [layer["dead_experts"] for layer in row["per_layer"]],
              "max/mean", [round(layer["max_to_mean_load"], 2) for layer in row["per_layer"]],
              "overlap", [round(layer["mean_adjacent_selected_overlap"], 3) for layer in row["per_layer"]])


if __name__ == "__main__":
    main()
