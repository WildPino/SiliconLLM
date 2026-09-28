#!/usr/bin/env python3
"""Continue E12800 hash grandchildren against the frozen METH-136 E1280 control."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth142_token_hash_route_screen as M142


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
M144_MODEL = DOC / "meth144_integrated_hash_parity_result.json"
M144_MODEL_SHA = "f5f4c52a114c947b2bb5b3df10d7136ac445485782239e398848bcd5c0c6d65d"
M144_NATIVE = DOC / "meth144_native_hash_route_result.json"
M144_NATIVE_SHA = "acdd5c0bb13bbd2d33f67199ef108ea698baa0007ceeeb8f790561e9406b190c"
M136_PARTIAL = DOC / "meth136_matched_sparse_train_result.partial.json"
M136_PARTIAL_SHA = "068c19a2fa34b95cca2a357674a1f7e1185ea04095593d54eec1e892ecb29be6"
CONTROL_BANK = M136.ART / "meth136_control_bf16.bin"
CONTROL_SHA = "d5d9e6b3753ab55ecbb20a8f60d1ba11a6d9bdd41f66bb34909207b9a29ba299"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.bank.exists()
    for path, expected in ((M144_MODEL, M144_MODEL_SHA),
                           (M144_NATIVE, M144_NATIVE_SHA),
                           (M136_PARTIAL, M136_PARTIAL_SHA),
                           (CONTROL_BANK, CONTROL_SHA)):
        assert M136.digest(path) == expected, path
    assert M142.hash_grandchildren(np.array([123]), np.array([45]),
        np.array([67]), np.array([[89, 89, 89, 89]]), 3)[0, 0] == 899
    bound = json.loads(M136_PARTIAL.read_text(encoding="utf-8"))
    assert len(bound["completed_arms"]) == 1
    control = bound["completed_arms"][0]
    assert control["arm"] == "control" and control["initial_parity"]["logit_max_abs_error"] == 0
    assert len(control["records"]) == 256
    assert control["artifact"]["sha256"] == CONTROL_SHA
    assert control["expected_selections_per_layer"] > 0
    assert len(control["route_max_to_mean_by_layer"]) == 24
    assert M136.UPDATES == 256 and M136.SEED == 136136

    M136.MAX_SECONDS = 30 * 60
    M136.MAX_DISK = 5_000_000_000
    torch.set_num_threads(6)
    torch.set_grad_enabled(True)
    gpu = [i for i in range(torch.cuda.device_count())
           if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(gpu) == 1
    device = torch.device(f"cuda:{gpu[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    parent_state, child_state = M136.bind_inputs()
    source_a, source_b, prefixes = M136.load_factor_bank()
    raw_ids, chat, draws = M136.training_data()
    prompts = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"]
    teacher = M136.model_shell(device).eval()
    M136.make_wrappers(teacher, parent_state["expert_state"],
                       child_state["expert_state"], source_a, source_b,
                       prefixes, device, "teacher")
    teacher.eval()
    M136.budget(start, device)
    progress = args.out.with_name(args.out.stem + ".hash.progress.json")
    try:
        candidate = M136.train_arm("hash", teacher, parent_state["expert_state"],
            child_state["expert_state"], source_a, source_b, prefixes,
            None, raw_ids, chat, draws, prompts, device, start,
            args.bank, progress)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-145-matched-hash-training",
            "error": repr(error), "elapsed_seconds": time.monotonic() - start,
            "rss_bytes": psutil.Process().memory_info().rss,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)},
            indent=2) + "\n", encoding="utf-8")
        raise
    relative = [candidate["route_max_to_mean_by_layer"][li] /
                control["route_max_to_mean_by_layer"][li] for li in range(24)]
    assert candidate["expected_selections_per_layer"] == control["expected_selections_per_layer"]
    gates = {
        "initial_bf16_logit_parity": candidate["initial_parity"]["logit_max_abs_error"] == 0,
        "coverage": min(candidate["route_coverage_by_layer"]) >= 6400,
        "bf16_distinct": min(candidate["artifact"]["bf16_distinct_rows_by_layer"]) >= 3200,
        "relative_load": max(relative) <= 1.25,
        "hot_parent_share": max(candidate["hot_parent_worst_grandchild_share_by_layer"]) <= 0.25,
        "artifact_readback": candidate["artifact"]["readback_exact"],
    }
    result = {"experiment": "METH-145-matched-hash-training",
              "meth144_model_sha256": M144_MODEL_SHA,
              "meth144_native_sha256": M144_NATIVE_SHA,
              "meth136_control_result_sha256": M136_PARTIAL_SHA,
              "control_bank_sha256": CONTROL_SHA,
              "control_reused": True,
              "exact_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "seed": M136.SEED, "updates": M136.UPDATES,
              "draws": draws, "candidate": candidate,
              "control_max_to_mean_by_layer": control["route_max_to_mean_by_layer"],
              "candidate_to_control_load_ratio_by_layer": relative,
              "gates": gates,
              "runtime": {**M136.budget(start, device),
                          "gpu": torch.cuda.get_device_name(device)},
              "decision": ("matched_training_pass_fresh_quality_pending"
                           if all(gates.values()) else "matched_training_gate_fail")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "minimum_coverage": min(candidate["route_coverage_by_layer"]),
                      "minimum_bf16_distinct": min(candidate["artifact"]["bf16_distinct_rows_by_layer"]),
                      "worst_relative_load": max(relative),
                      "worst_hot_parent_share": max(candidate["hot_parent_worst_grandchild_share_by_layer"]),
                      "candidate_sha256": candidate["artifact"]["sha256"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
