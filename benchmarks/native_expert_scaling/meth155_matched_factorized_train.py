#!/usr/bin/env python3
"""Matched shared B base plus routed residuals versus frozen E1280 control."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth150_shared_structure_route as R150


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
M151_MODEL = DOC / "meth151_shared_model_parity_result.json"
M151_MODEL_SHA = "ded7f7c5663b0199d8ef61436abf02713a4b194efd33c574ea1051082c23a419"
M151_NATIVE = DOC / "meth151_native_shared_route_result.json"
M151_NATIVE_SHA = "41ee01e0510a5016d1d269a3c00777a8f526a959dc0c6f5ceeb45e4eb7631682"
M154 = DOC / "meth154_bank_decomposition_result.json"
M154_SHA = "7d7d5ab76418c5cb0b38eb7fe8473baa00b856c8e2633de0a8610dd12b28cd94"
M153 = DOC / "meth153_shared_prediction_result.json"
M153_SHA = "5f02275315b2112abdd59b7efa5e434485fe5a170b126242aa466ed19d9130fb"
M136_PARTIAL = DOC / "meth136_matched_sparse_train_result.partial.json"
M136_PARTIAL_SHA = "068c19a2fa34b95cca2a357674a1f7e1185ea04095593d54eec1e892ecb29be6"
CONTROL_BANK = M136.ART / "meth136_control_bf16.bin"
CONTROL_SHA = "d5d9e6b3753ab55ecbb20a8f60d1ba11a6d9bdd41f66bb34909207b9a29ba299"


def expected_structural_selections(raw_ids, chat, draws, table):
    count = 0
    for draw in draws:
        raw = raw_ids[draw["raw_row"], draw["raw_offset"]:
                      draw["raw_offset"] + M136.M15.SEQ - 1].astype(np.int64)
        chat_ids = np.asarray(chat[draw["chat_index"]][1][:-1], dtype=np.int64)
        for ids in (raw, chat_ids):
            previous = np.r_[0, ids[:-1]]
            positions = np.arange(ids.size, dtype=np.int64)
            count += 4 * int(R150.shared_mask(ids, previous, positions, table).sum())
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.bank.exists()
    for path, expected in ((M151_MODEL, M151_MODEL_SHA),
                           (M151_NATIVE, M151_NATIVE_SHA),
                           (M154, M154_SHA), (M153, M153_SHA),
                           (M136_PARTIAL, M136_PARTIAL_SHA),
                           (CONTROL_BANK, CONTROL_SHA)):
        assert M136.digest(path) == expected, path
    table = R150.load_table()
    assert R150.golden(table) == {"content": 899, "structural": 890}
    bound = json.loads(M136_PARTIAL.read_text(encoding="utf-8"))
    assert len(bound["completed_arms"]) == 1
    control = bound["completed_arms"][0]
    assert control["arm"] == "control" and len(control["records"]) == 256
    assert control["initial_parity"]["logit_max_abs_error"] == 0
    assert control["artifact"]["sha256"] == CONTROL_SHA
    assert M136.UPDATES == 256 and M136.SEED == 136136

    M136.MAX_SECONDS = 35 * 60
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
    expected_structural = expected_structural_selections(raw_ids, chat, draws, table)
    prompts = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"]
    teacher = M136.model_shell(device).eval()
    M136.make_wrappers(teacher, parent_state["expert_state"],
                       child_state["expert_state"], source_a, source_b,
                       prefixes, device, "teacher")
    teacher.eval()
    M136.budget(start, device)
    progress = args.out.with_name(args.out.stem + ".factorized.progress.json")
    try:
        candidate = M136.train_arm("factorized", teacher, parent_state["expert_state"],
            child_state["expert_state"], source_a, source_b, prefixes,
            None, raw_ids, chat, draws, prompts, device, start,
            args.bank, progress)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-155-matched-factorized-training",
            "error": repr(error), "elapsed_seconds": time.monotonic() - start,
            "rss_bytes": psutil.Process().memory_info().rss,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)},
            indent=2) + "\n", encoding="utf-8")
        raise
    content = candidate["content_route"]
    assert content is not None
    assert candidate["expected_selections_per_layer"] == control["expected_selections_per_layer"]
    relative_all = [candidate["route_max_to_mean_by_layer"][li] /
                    control["route_max_to_mean_by_layer"][li] for li in range(24)]
    gates = {
        "complete_256_updates": len(candidate["records"]) == 256,
        "initial_bf16_logit_parity": candidate["initial_parity"]["logit_max_abs_error"] == 0,
        "structural_count_exact": content["structural_selections_per_layer"] == expected_structural,
        "content_coverage": min(content["route_coverage_by_layer"]) >= 6000,
        "content_load_ratio": max(content["to_own_parent_max_load_ratio_by_layer"]) <= 1.25,
        "content_hot_parent_share": max(content["hot_parent_worst_grandchild_share_by_layer"]) <= 0.25,
        "bf16_distinct": min(candidate["artifact"]["bf16_distinct_rows_by_layer"]) >= 3200,
        "base_coverage": min(candidate["final_base_optimizer_rows_by_layer"]) >= 1100,
        "base_bf16_changed": min(candidate["artifact"]["base_bf16_changed_rows_by_layer"]) > 0,
        "combined_bank_step_audits": min(candidate["final_combined_audit_checks_by_layer"]) == 256,
        "artifact_readback": candidate["artifact"]["readback_exact"],
    }
    result = {"experiment": "METH-155-matched-factorized-training",
              "meth151_model_sha256": M151_MODEL_SHA,
              "meth151_native_sha256": M151_NATIVE_SHA,
              "meth153_failure_sha256": M153_SHA,
              "meth154_bank_diagnostic_sha256": M154_SHA,
              "meth150_table_sha256": R150.TABLE_SHA,
              "meth136_control_result_sha256": M136_PARTIAL_SHA,
              "control_bank_sha256": CONTROL_SHA,
              "control_reused": True,
              "exact_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "seed": M136.SEED, "updates": M136.UPDATES,
              "draws": draws, "candidate": candidate,
              "expected_structural_selections_per_layer": expected_structural,
              "control_max_to_mean_by_layer": control["route_max_to_mean_by_layer"],
              "candidate_to_control_all_token_load_ratio_by_layer": relative_all,
              "gates": gates,
              "runtime": {**M136.budget(start, device),
                          "gpu": torch.cuda.get_device_name(device)},
              "decision": ("matched_training_pass_fresh_quality_pending"
                           if all(gates.values()) else "matched_training_gate_fail")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
        "minimum_content_coverage": min(content["route_coverage_by_layer"]),
        "minimum_bf16_distinct": min(candidate["artifact"]["bf16_distinct_rows_by_layer"]),
        "minimum_base_rows": min(candidate["final_base_optimizer_rows_by_layer"]),
        "minimum_base_bf16_changed": min(candidate["artifact"]["base_bf16_changed_rows_by_layer"]),
        "minimum_combined_audits": min(candidate["final_combined_audit_checks_by_layer"]),
        "worst_content_load_ratio": max(content["to_own_parent_max_load_ratio_by_layer"]),
        "worst_content_hot_share": max(content["hot_parent_worst_grandchild_share_by_layer"]),
        "structural_selections_per_layer": content["structural_selections_per_layer"],
        "candidate_sha256": candidate["artifact"]["sha256"],
        "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
