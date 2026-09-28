#!/usr/bin/env python3
"""Run one frozen matched E1,280 or E12,800 training arm."""

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
import meth155_factorized_shared_experts as F155
import meth175_training_draws as D175


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
TABLE = DOC / "meth172_threshold8_shared_table.json"
TABLE_SHA = "e89ff2cecbf887b85b5e361d47a8b918d8ccb93bd5c385a580ba993b93ac72db"
DRAWS_SHA = "9d4e06eced3cc1c51bac4e6219013f37f4495acd10c87827b567f0966f5c490c"
ROUTE = DOC / "meth172_low_recurrence_route_result.json"
ROUTE_SHA = "3e71f5df77c54081e7d39b106c2cc1138902e9b91cf5b836d18eee350714dba9"
CONTROL_RESULT = DOC / "meth175_control_long_result.json"
MAX_RSS = 52 * (1 << 30)
MAX_DISK = 7_000_000_000


class LowRecurrenceFactorizedExperts(F155.FactorizedSharedExperts):
    route_table = None

    def selected_ids(self, flat, children):
        if self.token_context is None:
            raise RuntimeError("token context must be set before expert forward")
        tokens, previous, positions = self.token_context
        if flat.shape[0] != tokens.size or children.shape != (tokens.size, 4):
            raise RuntimeError("token context shape differs from expert route")
        self.last_children = children.detach()
        source = children.detach().cpu().numpy().astype(np.int64)
        selected, base_shared = R150.route(tokens, previous, positions,
                                           source, self.layer_id, self.route_table)
        shared = base_shared | (tokens == 151644) | (tokens == 151645)
        selected = np.where(shared[:, None], source * 10, selected)
        result = torch.from_numpy(selected).to(device=children.device)
        assert torch.equal(result.div(10, rounding_mode="floor"), children)
        self.last_shared = shared
        return result


def load_table():
    assert M136.digest(TABLE) == TABLE_SHA
    record = json.loads(TABLE.read_text(encoding="utf-8"))
    table = {(r["token"], r["previous"], r["position"])
             for r in record["tuples"]}
    assert len(table) == 1685 and R150.golden(table) == {
        "content": 899, "structural": 890}
    tokens = np.asarray([151644, 151645, 11, 123], dtype=np.int64)
    previous = np.asarray([123, 123, 198, 45], dtype=np.int64)
    positions = np.asarray([67, 67, 56, 67], dtype=np.int64)
    children = np.full((4, 4), 89, dtype=np.int64)
    chosen, base_shared = R150.route(
        tokens, previous, positions, children, 3, table)
    shared = base_shared | (tokens == 151644) | (tokens == 151645)
    chosen = np.where(shared[:, None], children * 10, chosen)
    assert chosen[:, 0].tolist() == [890, 890, 890, 899]
    return table


def expected_structural(raw_ids, chat, draws, table):
    count = 0
    for draw in draws:
        raw = raw_ids[draw["raw_row"], draw["raw_offset"]:
                      draw["raw_offset"] + 127].astype(np.int64)
        chat_ids = np.asarray(chat[draw["chat_index"]][1][:-1], dtype=np.int64)
        for ids in (raw, chat_ids):
            previous = np.r_[0, ids[:-1]]
            positions = np.arange(ids.size, dtype=np.int64)
            shared = R150.shared_mask(ids, previous, positions, table)
            shared |= (ids == 151644) | (ids == 151645)
            count += 4 * int(shared.sum())
    return count


def content_metrics(rows, hot_threshold=250):
    coverage, median, under32, load, hot = [], [], [], [], []
    for item in rows:
        values = np.asarray(item, dtype=np.int64).reshape(1280, 10)
        slots = values[:, 1:].reshape(-1)
        active = slots[slots > 0]
        parent = values.sum(axis=1)
        hot_mask = parent >= hot_threshold
        assert np.any(hot_mask)
        coverage.append(int((slots > 0).sum()))
        median.append(float(np.quantile(active, 0.5)))
        under32.append(float((slots < 32).mean()))
        load.append(float(9 * slots.max() / parent.max()))
        hot.append(float(max(values[j, 1:].max() / parent[j]
                             for j in np.where(hot_mask)[0])))
    return {"minimum_content_coverage": min(coverage),
            "minimum_active_median": min(median),
            "maximum_under_32_fraction": max(under32),
            "maximum_own_parent_load_ratio": max(load),
            "maximum_hot_parent_share": max(hot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("control", "candidate"), required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--control-result-sha")
    ap.add_argument("--control-bank-sha")
    args = ap.parse_args()
    assert not args.out.exists() and not args.bank.exists()
    assert M136.digest(ROUTE) == ROUTE_SHA
    assert M136.digest(DOC / "meth175_training_draws.json") == DRAWS_SHA
    table = load_table()
    raw_ids, chat, draws = D175.load(DRAWS_SHA)
    if args.pilot:
        assert args.arm == "candidate"
        draws = draws[:16]
    else:
        assert len(draws) == 3840
    if args.arm == "candidate" and not args.pilot:
        assert args.control_result_sha and args.control_bank_sha
        assert M136.digest(CONTROL_RESULT) == args.control_result_sha
        control = json.loads(CONTROL_RESULT.read_text(encoding="utf-8"))
        assert control["decision"] == "matched_long_control_complete"
        assert control["draws_sha256"] == DRAWS_SHA
        assert control["artifact"]["sha256"] == args.control_bank_sha
        assert M136.digest(Path(control["artifact"]["path"])) == args.control_bank_sha
    expected_shared = expected_structural(raw_ids, chat, draws, table)
    if not args.pilot:
        prior = json.loads(ROUTE.read_text(encoding="utf-8"))
        assert expected_shared == prior["cells"]["training"]["layers"][0][
            "structural_selections"]
    M136.UPDATES = len(draws)
    M136.MAX_SECONDS = (20 * 60 if args.pilot else
                        4.5 * 60 * 60 if args.arm == "control" else 12 * 60 * 60)
    M136.MAX_RSS = MAX_RSS
    M136.MAX_DISK = MAX_DISK
    M136.HOT_PARENT_MIN_COUNT = 1 if args.pilot else 250
    torch.set_num_threads(6)
    torch.set_grad_enabled(True)
    gpus = [i for i in range(torch.cuda.device_count())
            if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(gpus) == 1
    device = torch.device(f"cuda:{gpus[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    parent, child = M136.bind_inputs()
    source_a, source_b, prefixes = M136.load_factor_bank()
    parity = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"]
    teacher = M136.model_shell(device).eval()
    M136.make_wrappers(teacher, parent["expert_state"], child["expert_state"],
                       source_a, source_b, prefixes, device, "teacher")
    teacher.eval()
    if args.arm == "candidate":
        LowRecurrenceFactorizedExperts.route_table = table
        F155.FactorizedSharedExperts = LowRecurrenceFactorizedExperts
    progress = args.out.with_name(args.out.stem + ".progress.json")
    try:
        result = M136.train_arm(
            "control" if args.arm == "control" else "factorized",
            teacher, parent["expert_state"], child["expert_state"],
            source_a, source_b, prefixes, None, raw_ids, chat, draws,
            parity, device, started, args.bank, progress,
            project_following_arm=False)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({
            "experiment": "METH-175-matched-long-training-failure",
            "arm": args.arm, "pilot": args.pilot,
            "error": repr(error), "elapsed_seconds": time.monotonic() - started,
            "rss_bytes": psutil.Process().memory_info().rss,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)},
            indent=2) + "\n", encoding="utf-8")
        raise
    assert len(result["records"]) == len(draws)
    assert result["initial_parity"]["logit_max_abs_error"] == 0
    assert result["expected_selections_per_layer"] == 4 * sum(
        127 + len(chat[d["chat_index"]][1]) - 1 for d in draws)
    assert result["artifact"]["readback_exact"]
    gates = {"complete_updates": True, "initial_bf16_parity": True,
             "artifact_readback": True}
    metrics = None
    if args.arm == "candidate":
        content = result["content_route"]
        assert content is not None
        assert content["structural_selections_per_layer"] == expected_shared
        metrics = content_metrics(content["route_counts_by_layer"],
                                  hot_threshold=1 if args.pilot else 250)
        expected_total = result["expected_selections_per_layer"]
        gates.update({
            "structural_count_exact": True,
            "content_coverage": metrics["minimum_content_coverage"] >=
                                (10000 if not args.pilot else 1),
            "active_median": metrics["minimum_active_median"] >=
                             (50 if not args.pilot else 1),
            "under_32": metrics["maximum_under_32_fraction"] <=
                        (0.50 if not args.pilot else 1.0),
            "own_parent_load_ratio": (args.pilot or
                                      metrics["maximum_own_parent_load_ratio"] <= 1.25),
            "hot_parent_share": (args.pilot or
                                 metrics["maximum_hot_parent_share"] <= 0.25),
            "structural_share": expected_shared <= 0.25 * expected_total,
            "bf16_distinct": min(result["artifact"]["bf16_distinct_rows_by_layer"]) >=
                             (11000 if not args.pilot else 1),
            "base_coverage": min(result["final_base_optimizer_rows_by_layer"]) >=
                             (1200 if not args.pilot else 1),
            "base_shift": min(result["artifact"]["combined_mean_shift_rms_by_layer"]) > 0,
            "combined_step_audits": min(result["final_combined_audit_checks_by_layer"]) ==
                                    len(draws)})
    decision = ("pilot_pass" if args.pilot and all(gates.values()) else
                "pilot_fail" if args.pilot else
                "matched_long_control_complete" if args.arm == "control" else
                "matched_long_candidate_pass_fresh_quality_pending"
                if all(gates.values()) else "matched_long_candidate_gate_fail")
    output = {"experiment": "METH-175-matched-long-low-recurrence-training",
              "arm": args.arm, "pilot": args.pilot,
              "draws_sha256": DRAWS_SHA, "route_replay_sha256": ROUTE_SHA,
              "table_sha256": TABLE_SHA,
              "expected_structural_selections_per_layer": expected_shared,
              "updates": len(draws), "candidate_metrics": metrics,
              "artifact": result["artifact"], "training": result,
              "gates": gates, "decision": decision,
              "runtime": {**M136.budget(started, device),
                          "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "gates": gates,
                      "metrics": metrics,
                      "bank_sha256": result["artifact"]["sha256"],
                      "result_sha256": M136.digest(args.out),
                      "runtime": output["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
