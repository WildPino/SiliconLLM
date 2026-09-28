#!/usr/bin/env python3
"""Replay METH-136 final BF16 banks to diagnose route-load skew."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth136_sparse_experts  # inserts the S1 import directory
import meth136_matched_sparse_train as M136


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
ART = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling"
PARTIAL = DOC / "meth136_matched_sparse_train_result.partial.json"
PROGRESS = DOC / "meth136_matched_sparse_train_result.candidate.progress.json"
CONTROL = ART / "meth136_control_bf16.bin"
CANDIDATE = ART / "meth136_candidate_bf16.bin"
CONTROL_SHA = "d5d9e6b3753ab55ecbb20a8f60d1ba11a6d9bdd41f66bb34909207b9a29ba299"
CANDIDATE_SHA = "61570f9efb291fcd43e5bb83c4f0ac18b7fbeb8e44e4ce80791efbd80b56808a"
MAX_SECONDS = 20 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 30 * (1 << 30)


def budget(start, device):
    report = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (report["seconds"] > MAX_SECONDS or report["rss_bytes"] > MAX_RSS
            or report["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-138 resource stop: {report}")
    return report


def bank_bits(path, rows):
    data = np.memmap(path, dtype=np.uint8, mode="r")
    assert M136.OUT_HEADER.unpack_from(data) == (b"M136BF01", 24, 896, 8, rows)
    assert data.size == M136.OUT_HEADER.size + 24 * rows * 896 * 8 * 2
    return data


def layer_bits(data, li, rows):
    offset = M136.OUT_HEADER.size + li * rows * 896 * 8 * 2
    return np.frombuffer(data, dtype="<u2", count=rows * 896 * 8,
                         offset=offset).reshape(rows, 896, 8)


def summarize(counts):
    total = int(counts.sum())
    mean = total / counts.numel()
    hot = torch.topk(counts, 5)
    return {"coverage": int((counts > 0).sum()), "selections": total,
            "max_to_mean": float(counts.max() / mean),
            "top_five": [{"slot": int(slot), "selections": int(value)}
                         for value, slot in zip(hot.values, hot.indices)]}


def main():
    start = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    for path, sha in ((CONTROL, CONTROL_SHA), (CANDIDATE, CANDIDATE_SHA)):
        assert M136.digest(path) == sha, path
    partial_sha, progress_sha = M136.digest(PARTIAL), M136.digest(PROGRESS)
    partial = json.loads(PARTIAL.read_text(encoding="utf-8"))
    progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
    assert len(partial["completed_arms"]) == 1
    assert partial["completed_arms"][0]["arm"] == "control"
    assert partial["completed_arms"][0]["artifact"]["sha256"] == CONTROL_SHA
    assert progress["arm"] == "candidate" and progress["completed_update"] == 256
    assert progress["initial_parity"] == {"prompts": 8, "logit_max_abs_error": 0.0}
    assert M136.digest(M136.EXACT_BANK) == M136.EXACT_SHA
    assert M136.digest(M136.THIRD) == M136.THIRD_SHA
    source_a, source_b, prefixes = M136.load_factor_bank()
    control_bits = bank_bits(CONTROL, 1280)
    candidate_bits = bank_bits(CANDIDATE, 12800)
    distinct_control, distinct_candidate, mean_errors = [], [], []
    for li, reference in enumerate(source_b):
        source = reference.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        cbits = layer_bits(control_bits, li, 1280)
        gbits = layer_bits(candidate_bits, li, 12800).reshape(1280, 10, 896, 8)
        distinct_control.append(int(np.count_nonzero(np.any(cbits != source, axis=(1, 2)))))
        distinct_candidate.append(int(np.count_nonzero(np.any(
            gbits != source[:, None], axis=(2, 3)))))
        candidate_float = M136.bf16_to_f32(gbits)
        mean_errors.append(float(np.max(np.abs(candidate_float.mean(axis=1) -
                                                reference.numpy()))))
    del control_bits, candidate_bits

    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    gpu = [i for i in range(torch.cuda.device_count())
           if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(gpu) == 1
    device = torch.device(f"cuda:{gpu[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    parent_state, child_state = M136.bind_inputs()
    raw_ids, chat, draws = M136.training_data()
    third = M136.load_third()
    model = M136.model_shell(device).eval()
    original_mlps = [layer.mlp for layer in model.model.layers]
    arms = {}
    for name, mode, sidecar, path, rows in (
            ("control", "control", None, CONTROL, 1280),
            ("candidate", "candidate", third, CANDIDATE, 12800)):
        for layer, base in zip(model.model.layers, original_mlps):
            layer.mlp = base
        wrappers, banks = M136.make_wrappers(
            model, parent_state["expert_state"], child_state["expert_state"],
            source_a, source_b, prefixes, device, mode, sidecar)
        bits = bank_bits(path, rows)
        for li, bank in enumerate(banks):
            with torch.no_grad():
                bank.copy_(torch.from_numpy(M136.bf16_to_f32(
                    layer_bits(bits, li, rows)).copy()))
        arms[name] = wrappers
    del parent_state, child_state, source_b
    budget(start, device)

    def save(stage, details):
        result = {"experiment": "METH-138-final-artifact-route-load-diagnostic",
                  "scope": "Final-state replay, not the exact METH-136 train-time gate",
                  "input_sha256": {"control_bank": CONTROL_SHA,
                                   "candidate_bank": CANDIDATE_SHA,
                                   "partial_report": partial_sha,
                                   "candidate_progress": progress_sha,
                                   "source_bank": M136.EXACT_SHA,
                                   "third_sidecar": M136.THIRD_SHA},
                  "train_time_control_max_to_mean_by_layer":
                      partial["completed_arms"][0]["route_max_to_mean_by_layer"],
                  "train_time_candidate_coverage_by_layer": progress["coverage_by_layer"],
                  "bf16_distinct_rows_by_layer": {"control": distinct_control,
                                                  "candidate": distinct_candidate},
                  "candidate_postround_sibling_mean_max_abs_by_layer": mean_errors,
                  "completed_stage": stage, "final_state_replay": details,
                  "runtime": {**budget(start, device),
                              "gpu": torch.cuda.get_device_name(device)}}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    replay = {}
    for name, rows in (("control", 1280), ("candidate", 12800)):
        wrappers = arms[name]
        for layer, wrapper in zip(model.model.layers, wrappers):
            layer.mlp = wrapper
        model.eval()
        counts = [torch.zeros(rows, dtype=torch.int64) for _ in range(24)]
        with torch.inference_mode():
            for draw in draws:
                raw = raw_ids[draw["raw_row"],
                              draw["raw_offset"]:draw["raw_offset"] + M136.M15.SEQ]
                _, chat_ids, _ = chat[draw["chat_index"]]
                for sequence in (raw, chat_ids):
                    ids = torch.as_tensor(sequence, dtype=torch.long, device=device)[None]
                    model(ids[:, :-1], use_cache=False)
                    for li, wrapper in enumerate(wrappers):
                        selected = wrapper.last_selected.flatten().cpu().long()
                        counts[li] += torch.bincount(selected, minlength=rows)
                if draw["update"] in (64, 128, 192, 256):
                    print(json.dumps({"arm": name, "replayed_updates": draw["update"],
                                      "budget": budget(start, device)}), flush=True)
        expected = 4 * sum((M136.M15.SEQ - 1) +
                           (len(chat[d["chat_index"]][1]) - 1) for d in draws)
        assert all(int(c.sum()) == expected for c in counts)
        replay[name] = {"by_layer": [summarize(c) for c in counts],
                        "expected_selections_per_layer": expected,
                        "forward_B_transfer_bytes":
                            sum(w.forward_B_transfer_bytes for w in wrappers)}
        if name == "candidate":
            replay[name]["source_child_aggregate_by_layer"] = [
                summarize(c.view(1280, 10).sum(dim=1)) for c in counts]
        save(name, replay)
    result = save("complete", replay)
    print(json.dumps({"decision": "METH-136-remains-rejected",
                      "control_max_skew": max(x["max_to_mean"] for x in replay["control"]["by_layer"]),
                      "candidate_max_skew": max(x["max_to_mean"] for x in replay["candidate"]["by_layer"]),
                      "candidate_min_distinct": min(distinct_candidate),
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
