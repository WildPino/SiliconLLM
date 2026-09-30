#!/usr/bin/env python3
"""Stream exact trained BF16 banks and partition child versus parent shifts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
CONTROL = DOC / "meth175_control_long_result.json"
CANDIDATE = DOC / "meth175_candidate_long_result.json"
CONTROL_SHA = "21ba9c2752e97cae292a3f5d995ea1d205a3448b0a95142d4a995d4a67d9a802"
CANDIDATE_SHA = "5b8de7a93c353b83393b0d2189a2cd594593c298dd867a998ca97d5d35ee4c64"
HEADER = struct.Struct("<8s4I")
WIDTH, RANK, PARENTS, CHILDREN, LAYERS = 896, 8, 1280, 10, 24
CHUNK = 32
MAX_SECONDS = 15 * 60
MAX_RSS = 8 * (1 << 30)
PARTS = ("control_reference", "total_difference", "mean_shift",
         "child_variance", "weighted_control_reference", "weighted_total_difference",
         "weighted_mean_shift", "weighted_child_variance", "structural_difference")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_result(path, expected_sha):
    assert digest(path) == expected_sha
    record = json.loads(path.read_text(encoding="utf-8"))
    assert all(record["gates"].values())
    bank = Path(record["artifact"]["path"])
    assert bank.is_file() and bank.stat().st_size == record["artifact"]["bytes"]
    assert digest(bank) == record["artifact"]["sha256"]
    return record, bank


def map_bank(path, rows):
    data = np.memmap(path, dtype=np.uint8, mode="r")
    assert HEADER.unpack_from(data) == (b"M136BF01", LAYERS, WIDTH, RANK, rows)
    assert data.size == HEADER.size + LAYERS * rows * WIDTH * RANK * 2
    return data


def values(data, layer, rows, first_parent, count):
    offset = HEADER.size + layer * rows * WIDTH * RANK * 2
    first_row = first_parent * (CHILDREN if rows == PARENTS * CHILDREN else 1)
    nrows = count * (CHILDREN if rows == PARENTS * CHILDREN else 1)
    bits = np.frombuffer(data, dtype="<u2", count=nrows * WIDTH * RANK,
                         offset=offset + first_row * WIDTH * RANK * 2)
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(
        (count, CHILDREN, WIDTH, RANK) if rows == PARENTS * CHILDREN
        else (count, WIDTH, RANK))


def blank():
    return {key: 0.0 for key in PARTS}


def accumulate(target, control, candidate, route_counts):
    c = control.astype(np.float64)
    content = candidate[:, 1:].astype(np.float64)
    structural = candidate[:, 0].astype(np.float64)
    mean = content.mean(axis=1)
    shift = mean - c
    child = content - mean[:, None]
    total = content - c[:, None]
    weights = route_counts[:, 1:].astype(np.float64)
    per_parent = weights.sum(axis=1)
    weighted_mean = ((content * weights[:, :, None, None]).sum(axis=1) /
                     np.maximum(per_parent, 1)[:, None, None])
    weighted_shift = weighted_mean - c
    weighted_child = content - weighted_mean[:, None]
    w = weights[:, :, None, None]
    terms = {
        "control_reference": 9 * np.square(c).sum(),
        "total_difference": np.square(total).sum(),
        "mean_shift": 9 * np.square(shift).sum(),
        "child_variance": np.square(child).sum(),
        "weighted_control_reference": (per_parent[:, None, None] * np.square(c)).sum(),
        "weighted_total_difference": (w * np.square(total)).sum(),
        "weighted_mean_shift": (per_parent[:, None, None] * np.square(weighted_shift)).sum(),
        "weighted_child_variance": (w * np.square(weighted_child)).sum(),
        "structural_difference": np.square(structural - c).sum(),
    }
    for key, value in terms.items():
        target[key] += float(value)


def finish(parts, selections):
    content_values = PARENTS * 9 * WIDTH * RANK
    weighted_values = selections * WIDTH * RANK
    structural_values = PARENTS * WIDTH * RANK
    assert math.isclose(parts["total_difference"],
                        parts["mean_shift"] + parts["child_variance"], rel_tol=1e-10)
    assert math.isclose(parts["weighted_total_difference"],
                        parts["weighted_mean_shift"] + parts["weighted_child_variance"],
                        rel_tol=1e-10)
    return {"squared_sums": parts,
            "content_rms": {key: math.sqrt(parts[key] / content_values) for key in
                            ("control_reference", "total_difference", "mean_shift",
                             "child_variance")},
            "weighted_content_rms": {key: math.sqrt(parts[key] / weighted_values) for key in
                                     ("weighted_control_reference", "weighted_total_difference",
                                      "weighted_mean_shift", "weighted_child_variance")},
            "structural_difference_rms": math.sqrt(parts["structural_difference"] /
                                                   structural_values),
            "child_fraction_of_total_squared_difference":
                parts["child_variance"] / parts["total_difference"],
            "weighted_child_fraction_of_total_squared_difference":
                parts["weighted_child_variance"] / parts["weighted_total_difference"],
            "content_selections": selections}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    start = time.monotonic()
    control, control_path = read_result(CONTROL, CONTROL_SHA)
    candidate, candidate_path = read_result(CANDIDATE, CANDIDATE_SHA)
    assert control["decision"] == "matched_long_control_complete"
    assert candidate["decision"] == "matched_long_candidate_pass_fresh_quality_pending"
    source = map_bank(control_path, PARENTS)
    expanded = map_bank(candidate_path, PARENTS * CHILDREN)
    counts_by_layer = candidate["training"]["route_counts_by_layer"]
    assert len(counts_by_layer) == LAYERS
    layers = []
    pooled = blank()
    total_selections = 0
    for li in range(LAYERS):
        counts = np.asarray(counts_by_layer[li], dtype=np.int64).reshape(PARENTS, CHILDREN)
        assert np.all(counts >= 0)
        assert int(counts.sum()) == candidate["training"]["route_selections_by_layer"][li]
        assert int(counts[:, 0].sum()) == candidate["expected_structural_selections_per_layer"]
        selections = int(counts[:, 1:].sum())
        assert selections == 4200432
        parts = blank()
        for first in range(0, PARENTS, CHUNK):
            n = min(CHUNK, PARENTS - first)
            a = values(source, li, PARENTS, first, n)
            b = values(expanded, li, PARENTS * CHILDREN, first, n)
            accumulate(parts, a, b, counts[first:first + n])
            assert time.monotonic() - start <= MAX_SECONDS
            assert psutil.Process().memory_info().rss <= MAX_RSS
        layers.append({"layer": li, **finish(parts, selections)})
        for key in PARTS:
            pooled[key] += parts[key]
        total_selections += selections
        print(json.dumps({"completed_layer": li,
                          "weighted_child_fraction":
                              layers[-1]["weighted_child_fraction_of_total_squared_difference"]}),
              flush=True)
    pooled_view = finish({key: pooled[key] / LAYERS for key in PARTS},
                         total_selections // LAYERS)
    result = {"experiment": "METH-178-trained-content-child-diversity",
              "control_result_sha256": CONTROL_SHA,
              "candidate_result_sha256": CANDIDATE_SHA,
              "control_bank_sha256": control["artifact"]["sha256"],
              "candidate_bank_sha256": candidate["artifact"]["sha256"],
              "layers": layers, "pooled_per_layer": pooled_view,
              "runtime": {"seconds": time.monotonic() - start,
                          "rss_bytes": psutil.Process().memory_info().rss},
              "interpretation_scope": "Weight geometry and training-route-weighted diagnostic only; no model quality inference"}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result_sha256": digest(args.out),
                      "pooled_per_layer": pooled_view}), flush=True)


if __name__ == "__main__":
    main()
