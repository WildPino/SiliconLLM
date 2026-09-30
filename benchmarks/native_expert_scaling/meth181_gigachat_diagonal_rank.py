#!/usr/bin/env python3
"""Disjoint-domain diagonal activation rank screen for bound GigaChat experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
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
RANK_RESULT = DOC / "meth180_gigachat_expert_rank_result.json"
RANK_RESULT_SHA = "ecc9a3432a25a4a7fc0f2881c421947aeb097b7a4cf32a746ebcdefa10081b8d"
TRAIN = ROOT / "results/native_expert_scaling/meth05_gigachat_bf16_interleaved_106chunks.gguf"
TEST = ROOT / "results/native_expert_scaling/meth06_cyrillic_bf16_125chunks.gguf"
TRAIN_SHA = "ef5de87fc4fa0d1382ed2988e59e9472fb1d7fb2c9bed0e8edcced6ff9bd4f9c"
TEST_SHA = "5c529f9009f0f241dfc50d0a1f598b0e183c137693a951f04963dc729a344d08"
LAYERS = (1, 13, 25)
EXPERTS = (0, 32, 63)
ORGANS = {"gate_proj": "ffn_gate_exps", "up_proj": "ffn_up_exps",
          "down_proj": "ffn_down_exps"}
RANK = 192
MIN_COUNT = 64
MAX_SECONDS = 15 * 60
MAX_RSS = 6 * (1 << 30)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resource(start: float) -> dict:
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss}
    if result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS:
        raise RuntimeError(f"METH-181 resource stop: {result}")
    return result


class Imatrix:
    """Read only the F32 imatrix tensors from a pinned GGUF v3 file."""

    def __init__(self, path: Path, expected_sha: str, expected_chunks: int):
        self.path = path
        self.sha256 = digest(path)
        assert self.sha256 == expected_sha
        self.size = path.stat().st_size
        with path.open("rb") as stream:
            def read(n):
                value = stream.read(n)
                if len(value) != n:
                    raise ValueError("short GGUF header")
                return value

            def uint32():
                return struct.unpack("<I", read(4))[0]

            def uint64():
                return struct.unpack("<Q", read(8))[0]

            def string():
                length = uint64()
                assert length < (1 << 20)
                return read(length).decode("utf-8")

            def value(kind):
                formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                           6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
                if kind == 8:
                    return string()
                if kind == 9:
                    element_kind, count = uint32(), uint64()
                    assert count < (1 << 20)
                    return [value(element_kind) for _ in range(count)]
                assert kind in formats
                fmt = "<" + formats[kind]
                return struct.unpack(fmt, read(struct.calcsize(fmt)))[0]

            assert read(4) == b"GGUF"
            assert uint32() == 3
            n_tensors, n_fields = uint64(), uint64()
            assert n_tensors == 616 and n_fields < 100
            self.fields = {string(): value(uint32()) for _ in range(n_fields)}
            assert self.fields["general.type"] == "imatrix"
            assert self.fields["imatrix.chunk_count"] == expected_chunks
            assert self.fields["imatrix.chunk_size"] == 512
            alignment = self.fields.get("general.alignment", 32)
            assert alignment == 32
            self.tensors = {}
            for _ in range(n_tensors):
                name = string()
                nd = uint32()
                assert 1 <= nd <= 4
                dims = tuple(uint64() for _ in range(nd))
                dtype, offset = uint32(), uint64()
                assert name not in self.tensors
                self.tensors[name] = (dims, dtype, offset)
            self.data_base = (stream.tell() + 31) & ~31
        assert self.data_base < self.size

    def array(self, name: str, dimensions: tuple[int, ...]) -> np.ndarray:
        dims, dtype, offset = self.tensors[name]
        assert dims == dimensions and dtype == 0  # GGML F32
        size = math.prod(dims) * 4
        assert offset % 32 == 0 and self.data_base + offset + size <= self.size
        return np.memmap(self.path, dtype="<f4", mode="r",
                         offset=self.data_base + offset,
                         shape=tuple(reversed(dims)))

    def moment(self, layer: int, expert: int, organ: str, width: int) -> tuple[np.ndarray, int]:
        base = f"blk.{layer}.{ORGANS[organ]}.weight"
        counts = self.array(base + ".counts", (1, 64)).reshape(64)
        sums = self.array(base + ".in_sum2", (width, 64))
        assert np.isfinite(counts).all() and np.isfinite(sums).all()
        assert (counts >= 0).all() and (sums >= 0).all()
        count = int(counts[expert])
        assert counts[expert] == count and count >= MIN_COUNT
        moments = np.asarray(sums[expert], dtype=np.float64) / count
        assert moments.shape == (width,) and np.isfinite(moments).all()
        assert (moments > 0).all()
        return moments, count


def energy(matrix: np.ndarray, approximation: np.ndarray,
           sqrt_moment: np.ndarray) -> tuple[float, float]:
    reference = matrix.astype(np.float64) * sqrt_moment[None, :]
    residual = (matrix.astype(np.float64) - approximation.astype(np.float64)) * sqrt_moment[None, :]
    denom = float(np.square(reference).sum())
    error = float(np.square(residual).sum())
    assert denom > 0 and math.isfinite(error)
    return 1.0 - error / denom, math.sqrt(error / matrix.shape[0])


def factor(matrix: np.ndarray, sqrt_moment: np.ndarray):
    weighted = np.asarray(matrix * sqrt_moment.astype(np.float32)[None, :],
                          dtype=np.float32, order="C")
    u, singular, vh = scipy.linalg.svd(weighted, full_matrices=False,
                                       compute_uv=True, overwrite_a=True,
                                       check_finite=False, lapack_driver="gesdd")
    svd_norm = float(np.square(singular.astype(np.float64)).sum())
    direct_norm = float(np.square(weighted.astype(np.float64)).sum())
    assert math.isclose(svd_norm, direct_norm, rel_tol=1e-4)
    reconstructed = (u[:, :RANK] * singular[:RANK]) @ vh[:RANK, :]
    approximation = reconstructed.astype(np.float64) / sqrt_moment[None, :]
    train_optimum = float(np.square(singular[:RANK].astype(np.float64)).sum() / svd_norm)
    return approximation, train_optimum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    start = time.monotonic()
    stage = "bindings"
    rows = []
    try:
        assert digest(RANK_RESULT) == RANK_RESULT_SHA
        prior = json.loads(RANK_RESULT.read_text(encoding="utf-8"))
        assert prior["decision"] == "reject_direct_weight_only_rank192_int8_export"
        assert prior["sample"] == {"layers": list(LAYERS), "experts": list(EXPERTS),
                                    "projections": list(ORGANS),
                                    "ranks": [64, 128, 192, 384, 512, 768]}
        assert digest(SOURCE / "model.safetensors.index.json") == prior["source_index_sha256"]
        assert digest(SOURCE / "config.json") == prior["source_config_sha256"]
        train = Imatrix(TRAIN, TRAIN_SHA, 106)
        test = Imatrix(TEST, TEST_SHA, 125)
        prior_rows = {(r["layer"], r["expert"], r["projection"]): r for r in prior["rows"]}
        moments = {}
        for layer in LAYERS:
            for expert in EXPERTS:
                for organ in ORGANS:
                    width = 1280 if organ == "down_proj" else 1536
                    train_v, train_n = train.moment(layer, expert, organ, width)
                    test_v, test_n = test.moment(layer, expert, organ, width)
                    moments[layer, expert, organ] = (train_v, train_n, test_v, test_n)
        stage = "source_shards"
        for name, binding in prior["source_shards"].items():
            path = SOURCE / name
            assert path.stat().st_size == binding["bytes"] and digest(path) == binding["sha256"]
        stage = "factorizations"
        with threadpool_limits(limits=6):
            for layer in LAYERS:
                keys = [key for key in prior_rows if key[0] == layer]
                shard_names = {prior_rows[key]["source_shard"] for key in keys}
                assert len(shard_names) == 1
                with safe_open(SOURCE / next(iter(shard_names)), framework="pt", device="cpu") as source:
                    for expert in EXPERTS:
                        for organ in ORGANS:
                            reference = prior_rows[layer, expert, organ]
                            key = f"model.layers.{layer}.mlp.experts.{expert}.{organ}.weight"
                            tensor = source.get_tensor(key)
                            assert tensor.dtype == torch.bfloat16
                            bf16_sha = hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()
                            assert bf16_sha == reference["tensor_bf16_sha256"]
                            matrix = tensor.float().numpy()
                            assert list(matrix.shape) == reference["shape"]
                            train_v, train_n, test_v, test_n = moments[layer, expert, organ]
                            sqrt_train, sqrt_test = np.sqrt(train_v), np.sqrt(test_v)
                            weighted_hat, train_optimum = factor(matrix, sqrt_train)
                            weighted_train, train_rms = energy(matrix, weighted_hat, sqrt_train)
                            assert math.isclose(weighted_train, train_optimum, abs_tol=2e-5)
                            weighted_test, test_rms = energy(matrix, weighted_hat, sqrt_test)
                            ordinary_hat, _ = factor(matrix, np.ones(matrix.shape[1], dtype=np.float64))
                            ordinary_test, _ = energy(matrix, ordinary_hat, sqrt_test)
                            test_weighted = np.asarray(matrix * sqrt_test.astype(np.float32)[None, :],
                                                       dtype=np.float32, order="C")
                            test_singular = scipy.linalg.svd(test_weighted, full_matrices=False,
                                                              compute_uv=False, overwrite_a=True,
                                                              check_finite=False, lapack_driver="gesdd")
                            test_optimum = float(np.square(test_singular[:RANK].astype(np.float64)).sum() /
                                                 np.square(test_singular.astype(np.float64)).sum())
                            assert weighted_test <= test_optimum + 2e-5
                            assert ordinary_test <= test_optimum + 2e-5
                            row = {"layer": layer, "expert": expert, "projection": organ,
                                   "source_tensor_bf16_sha256": bf16_sha,
                                   "train_count": train_n, "test_count": test_n,
                                   "train_moment_min": float(train_v.min()),
                                   "train_moment_max": float(train_v.max()),
                                   "test_over_train_moment_median": float(statistics.median(test_v / train_v)),
                                   "train_optimal_diagonal_energy": train_optimum,
                                   "train_direct_diagonal_energy": weighted_train,
                                   "test_weighted_factor_diagonal_energy": weighted_test,
                                   "test_ordinary_factor_diagonal_energy": ordinary_test,
                                   "test_optimal_diagonal_energy": test_optimum,
                                   "train_residual_output_rms": train_rms,
                                   "test_residual_output_rms": test_rms,
                                   "budget_after": resource(start)}
                            rows.append(row)
                            print(json.dumps({"complete": [layer, expert, organ],
                                              "test_weighted_energy": weighted_test,
                                              "seconds": row["budget_after"]["seconds"]}), flush=True)
        assert len(rows) == 27
        scores = [r["test_weighted_factor_diagonal_energy"] for r in rows]
        comparator_gaps = [r["test_weighted_factor_diagonal_energy"] -
                           r["test_ordinary_factor_diagonal_energy"] for r in rows]
        gate = statistics.median(scores) >= .95 and min(scores) >= .90 and min(comparator_gaps) >= -.005
        result = {"experiment": "METH-181-GigaChat10B-disjoint-diagonal-rank",
                  "prior_result_sha256": RANK_RESULT_SHA,
                  "train_imatrix": {"path": str(TRAIN.relative_to(ROOT)), "sha256": train.sha256,
                                     "chunks": 106},
                  "test_imatrix": {"path": str(TEST.relative_to(ROOT)), "sha256": test.sha256,
                                    "chunks": 125},
                  "rank": RANK, "rows": rows,
                  "summary": {"count": len(scores), "min_test_energy": min(scores),
                              "median_test_energy": statistics.median(scores),
                              "max_test_energy": max(scores),
                              "min_weighted_minus_ordinary": min(comparator_gaps),
                              "median_weighted_minus_ordinary": statistics.median(comparator_gaps),
                              "min_train_count": min(r["train_count"] for r in rows),
                              "min_test_count": min(r["test_count"] for r in rows)},
                  "diagonal_rank192_screen_pass": gate,
                  "decision": ("eligible_for_full_model_quality_cost_test" if gate else
                               "reject_diagonal_activation_weighted_rank192_export"),
                  "runtime": resource(start),
                  "scope": "Diagonal input-second-moment proxy only; no full-covariance, model-quality, native-rate or 10x expert-count result"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < 100_000
        print(json.dumps({"decision": result["decision"], "summary": result["summary"],
                          "result_sha256": digest(args.out)}), flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-181-failure", "stage": stage,
                                       "error": repr(error), "partial_rows": rows,
                                       "runtime": {"seconds": time.monotonic() - start,
                                                   "rss_bytes": psutil.Process().memory_info().rss}},
                                      indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
