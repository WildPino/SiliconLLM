#!/usr/bin/env python3
"""Best-case singular-spectrum screen of bound real 10B MoE expert weights."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import psutil
import scipy.linalg
from safetensors import safe_open
import torch
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
SOURCE = ROOT / "benchmarks/donor_adaptation/density/results/strat01_gigachat_source_189fff27"
BINDING = ROOT / "benchmarks/donor_adaptation/density/results/strat01_gigachat_source_binding_v1/source_binding_report.json"
BINDING_SHA = "0cd0699fb100e66eb89b37ed088a942228446c5b20939dcc682eec186772182f"
REVISION = "189fff27a1dee68473960c3d5bca53e0e07a3191"
LAYERS = (1, 13, 25)
EXPERTS = (0, 32, 63)
ORGANS = ("gate_proj", "up_proj", "down_proj")
RANKS = (64, 128, 192, 384, 512, 768)
EXPECTED_SHARD_SIZES = {
    "model-00000-of-00005.safetensors": 4986065392,
    "model-00002-of-00005.safetensors": 4085022040,
    "model-00005-of-00005.safetensors": 1634008560,
}
MAX_SECONDS = 15 * 60
MAX_RSS = 6 * (1 << 30)
MAX_OUTPUT = 10_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def budget(start):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss}
    if result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS:
        raise RuntimeError(f"METH-180 resource stop: {result}")
    return result


def row(matrix, layer, expert, organ, bf16_sha, file_name, start):
    assert matrix.dtype == np.float32
    assert matrix.shape == ((1536, 1280) if organ == "down_proj" else (1280, 1536))
    original_sumsq = float(np.square(matrix.astype(np.float64)).sum())
    values = scipy.linalg.svdvals(matrix, overwrite_a=False,
                                  check_finite=False, lapack_driver="gesdd")
    assert values.size == 1280 and np.all(np.isfinite(values))
    assert np.all(values[:-1] >= values[1:]) and values[-1] >= 0
    singular_squares = np.square(values.astype(np.float64))
    svd_sumsq = float(singular_squares.sum())
    assert math.isclose(original_sumsq, svd_sumsq, rel_tol=1e-4)
    captured = np.cumsum(singular_squares)
    ranks = {}
    for rank in RANKS:
        energy = float(captured[rank - 1] / svd_sumsq)
        ranks[str(rank)] = {"optimal_frobenius_energy_fraction": energy,
                            "optimal_residual_rms": math.sqrt(
                                max(0.0, svd_sumsq - float(captured[rank - 1])) /
                                matrix.size)}
    return {"layer": layer, "expert": expert, "projection": organ,
            "source_shard": file_name, "tensor_bf16_sha256": bf16_sha,
            "shape": list(matrix.shape), "input_frobenius_sumsq": original_sumsq,
            "svd_frobenius_sumsq": svd_sumsq, "ranks": ranks,
            "budget_after": budget(start)}


def describe(rows, field, value):
    fractions = [item["ranks"]["192"]["optimal_frobenius_energy_fraction"]
                 for item in rows if item[field] == value]
    assert fractions
    return {"count": len(fractions), "min": min(fractions),
            "median": statistics.median(fractions), "max": max(fractions)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    started = time.monotonic()
    stage = "bindings"
    rows = []
    shard_bindings = {}
    try:
        assert digest(BINDING) == BINDING_SHA
        binding = json.loads(BINDING.read_text(encoding="utf-8"))
        assert binding["status"] == "PASS_SOURCE_BINDING"
        index_path = SOURCE / "model.safetensors.index.json"
        config_path = SOURCE / "config.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["num_hidden_layers"] == 26
        assert config["n_routed_experts"] == 64
        assert config["moe_intermediate_size"] == 1280
        assert config["hidden_size"] == 1536
        weight_map = index["weight_map"]
        assert index["metadata"]["total_size"] == 22959501568
        chosen = {}
        for layer in LAYERS:
            names = [f"model.layers.{layer}.mlp.experts.{expert}.{organ}.weight"
                     for expert in EXPERTS for organ in ORGANS]
            files = {weight_map[name] for name in names}
            assert len(files) == 1
            name = next(iter(files))
            assert name in EXPECTED_SHARD_SIZES
            chosen[layer] = (name, names)
        with threadpool_limits(limits=6):
            for layer in LAYERS:
                file_name, names = chosen[layer]
                shard = SOURCE / file_name
                assert shard.stat().st_size == EXPECTED_SHARD_SIZES[file_name]
                shard_bindings[file_name] = {"bytes": shard.stat().st_size,
                                             "sha256": digest(shard)}
                stage = f"layer_{layer}_svd"
                with safe_open(shard, framework="pt", device="cpu") as source:
                    for expert in EXPERTS:
                        for organ in ORGANS:
                            key = f"model.layers.{layer}.mlp.experts.{expert}.{organ}.weight"
                            assert key in names
                            tensor = source.get_tensor(key)
                            assert tensor.dtype == torch.bfloat16
                            bf16_sha = hashlib.sha256(
                                tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()
                            matrix = tensor.float().numpy()
                            record = row(matrix, layer, expert, organ,
                                         bf16_sha, file_name, started)
                            rows.append(record)
                            print(json.dumps({"completed": [layer, expert, organ],
                                              "rank192_energy": record["ranks"]["192"][
                                                  "optimal_frobenius_energy_fraction"],
                                              "seconds": record["budget_after"]["seconds"]}),
                                  flush=True)
                            del tensor, matrix
        stage = "summary"
        assert len(rows) == len(LAYERS) * len(EXPERTS) * len(ORGANS) == 27
        summary = {"rank192_all": {"count": len(rows),
                                   "min": min(r["ranks"]["192"]["optimal_frobenius_energy_fraction"]
                                              for r in rows),
                                   "median": statistics.median(
                                       r["ranks"]["192"]["optimal_frobenius_energy_fraction"]
                                       for r in rows),
                                   "max": max(r["ranks"]["192"]["optimal_frobenius_energy_fraction"]
                                              for r in rows)},
                   "by_projection": {organ: describe(rows, "projection", organ)
                                     for organ in ORGANS},
                   "by_layer": {str(layer): describe(rows, "layer", layer)
                                for layer in LAYERS}}
        gate = (summary["rank192_all"]["median"] >= .95 and
                summary["rank192_all"]["min"] >= .90)
        result = {"experiment": "METH-180-GigaChat10B-expert-rank-screen",
                  "source_repo": "ai-sage/GigaChat3.1-10B-A1.8B-bf16",
                  "source_revision": REVISION,
                  "source_binding_report_sha256": BINDING_SHA,
                  "source_index_sha256": digest(index_path),
                  "source_config_sha256": digest(config_path),
                  "source_shards": shard_bindings,
                  "sample": {"layers": list(LAYERS), "experts": list(EXPERTS),
                             "projections": list(ORGANS), "ranks": list(RANKS)},
                  "byte_arithmetic": {"source_elements_per_projection": 1966080,
                                      "rank192_factor_elements": 540672,
                                      "rank192_int8_over_ideal_q4_0_55": .5,
                                      "scope": "idealized payload only; no scales or kernel"},
                  "rows": rows, "summary": summary,
                  "rank192_weight_only_screen_pass": gate,
                  "decision": ("eligible_for_full_model_quality_cost_test" if gate
                               else "reject_direct_weight_only_rank192_int8_export"),
                  "runtime": budget(started),
                  "scope": "Optimal unweighted matrix reconstruction only; no activation, model-quality or native-rate result"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < MAX_OUTPUT
        print(json.dumps({"decision": result["decision"],
                          "rank192": summary["rank192_all"],
                          "result_sha256": digest(args.out)}), flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-180-failure",
                                       "stage": stage, "error": repr(error),
                                       "source_shards": shard_bindings,
                                       "partial_rows": rows,
                                       "runtime": {"seconds": time.monotonic() - started,
                                                   "rss_bytes": psutil.Process().memory_info().rss}},
                                      indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
