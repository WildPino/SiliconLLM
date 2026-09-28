#!/usr/bin/env python3
"""Postfailure B-bank decomposition without using fresh quality rows."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
M152 = DOC / "meth152_matched_shared_train_result.json"
M152_SHA = "6beb41e673f70ca9cbd496b59ba9572d7ef646d7950ec93d88609d007ac6901c"
M153 = DOC / "meth153_shared_prediction_result.json"
M153_SHA = "5f02275315b2112abdd59b7efa5e434485fe5a170b126242aa466ed19d9130fb"
CONTROL = M136.ART / "meth136_control_bf16.bin"
CONTROL_SHA = "d5d9e6b3753ab55ecbb20a8f60d1ba11a6d9bdd41f66bb34909207b9a29ba299"
CANDIDATE = M136.ART / "meth152_shared_candidate_bf16.bin"
CANDIDATE_SHA = "17196e7a320a6887ba4a04283e673f3b53b961a0062cb15b05e75d558df58a9e"


def bank(path, rows):
    data = np.memmap(path, dtype=np.uint8, mode="r")
    assert M136.OUT_HEADER.unpack_from(data) == (b"M136BF01", 24, 896, 8, rows)
    stride = rows * 896 * 8 * 2
    assert data.size == M136.OUT_HEADER.size + 24 * stride
    return data, stride


def layer(data, stride, index, rows):
    bits = np.frombuffer(data, dtype="<u2", count=rows * 896 * 8,
                         offset=M136.OUT_HEADER.size + index * stride)
    return M136.bf16_to_f32(bits).reshape(rows, 896, 8)


def rms(value):
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    for path, sha in ((M152, M152_SHA), (M153, M153_SHA),
                      (CONTROL, CONTROL_SHA), (CANDIDATE, CANDIDATE_SHA),
                      (M136.EXACT_BANK, M136.EXACT_SHA)):
        assert M136.digest(path) == sha, path
    assert json.loads(M153.read_text(encoding="utf-8"))["decision"] == "stop_document_prompt_gate"
    start = time.monotonic()
    source_a, source_b, _ = M136.load_factor_bank()
    del source_a
    control, control_stride = bank(CONTROL, 1280)
    candidate, candidate_stride = bank(CANDIDATE, 12800)
    layers = []
    for li in range(24):
        source = source_b[li].numpy()
        base = layer(control, control_stride, li, 1280)
        grand = layer(candidate, candidate_stride, li, 12800).reshape(1280, 10, 896, 8)
        grand_mean = grand.mean(axis=1)
        control_shift = base - source
        structural_delta = grand[:, 0] - source
        content_delta = grand[:, 1:] - source[:, None]
        layers.append({"layer": li,
            "control_minus_source_rms": rms(control_shift),
            "control_minus_source_max_abs": float(np.max(np.abs(control_shift))),
            "control_bf16_changed_rows": int(np.any(base != source, axis=(1, 2)).sum()),
            "candidate_mean_minus_source_rms": rms(grand_mean - source),
            "candidate_mean_minus_source_max_abs": float(np.max(np.abs(grand_mean - source))),
            "candidate_structural_minus_source_rms": rms(structural_delta),
            "candidate_content_minus_source_rms": rms(content_delta),
            "candidate_content_minus_control_rms": rms(grand[:, 1:] - base[:, None]),
            "candidate_structural_minus_control_rms": rms(grand[:, 0] - base)})
    result = {"experiment": "METH-154-postfailure-B-bank-decomposition",
              "scope": "training artifacts only; no selection on METH-153 sources",
              "meth152_result_sha256": M152_SHA, "meth153_result_sha256": M153_SHA,
              "exact_source_bank_sha256": M136.EXACT_SHA,
              "control_bank_sha256": CONTROL_SHA,
              "candidate_bank_sha256": CANDIDATE_SHA,
              "layers": layers,
              "summary": {
                  "control_minus_source_rms_min": min(x["control_minus_source_rms"] for x in layers),
                  "control_minus_source_rms_max": max(x["control_minus_source_rms"] for x in layers),
                  "candidate_mean_minus_source_rms_max": max(x["candidate_mean_minus_source_rms"] for x in layers),
                  "candidate_content_minus_source_rms_min": min(x["candidate_content_minus_source_rms"] for x in layers),
                  "candidate_content_minus_source_rms_max": max(x["candidate_content_minus_source_rms"] for x in layers)},
              "elapsed_seconds": time.monotonic() - start}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
