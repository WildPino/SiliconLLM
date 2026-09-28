#!/usr/bin/env python3
"""Replay the rejected hash bank and split route load by input segment."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth139_quantile_third_router as M139


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
TRAIN = DOC / "meth145_matched_hash_train_result.json"
TRAIN_SHA = "14687c03bf16b552afbdfef2e402e3acc54ff4eea8dd50e3a2140f3e8f2800e7"
BANK = M136.ART / "meth145_hash_candidate_bf16.bin"
BANK_SHA = "881a43fed3b1132a75bb4a4cfce2ada64cc13c40993248412bbadf52adf64c4f"
SEGMENTS = ("raw", "fixed_template", "variable_chat")
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 25 * (1 << 30)


def budget(start, device):
    report = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if (report["seconds"] > MAX_SECONDS or report["rss_bytes"] > MAX_RSS
            or report["gpu_peak_allocated_bytes"] > MAX_GPU):
        raise RuntimeError(f"METH-147 resource stop: {report}")
    return report


def summarize(counts):
    rows = []
    for li, row in enumerate(counts):
        stats = M139.summarize(row)
        parent = row.reshape(1280, 10).sum(axis=1)
        hot = parent >= 250
        share = (float(np.max(row.reshape(1280, 10)[hot].max(axis=1) / parent[hot]))
                 if hot.any() else None)
        rows.append({"layer": li, "selections": int(row.sum()),
                     "coverage": stats["coverage"],
                     "max_to_mean": stats["max_to_mean"],
                     "top_five": stats["top_five"],
                     "hot_parent_count": int(hot.sum()),
                     "worst_hot_parent_share": share})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.counts.exists()
    assert M136.digest(TRAIN) == TRAIN_SHA
    assert M136.digest(BANK) == BANK_SHA
    trained = json.loads(TRAIN.read_text(encoding="utf-8"))
    assert trained["decision"] == "matched_training_gate_fail"
    assert trained["candidate"]["artifact"]["sha256"] == BANK_SHA
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    gpu = [i for i in range(torch.cuda.device_count())
           if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(gpu) == 1
    device = torch.device(f"cuda:{gpu[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    parent, child = M136.bind_inputs()
    source_a, source_b, prefixes = M136.load_factor_bank()
    raw_ids, chat, draws = M136.training_data()
    assert draws == trained["draws"]
    chat_ids = [item[1] for item in chat]
    common = 0
    for position in range(min(map(len, chat_ids))):
        if len({ids[position] for ids in chat_ids}) != 1:
            break
        common += 1
    assert common == 55 and len(chat_ids) == 256
    model = M136.model_shell(device).eval()
    wrappers, banks = M136.make_wrappers(
        model, parent["expert_state"], child["expert_state"],
        source_a, source_b, prefixes, device, "hash")
    data = np.memmap(BANK, dtype=np.uint8, mode="r")
    assert M136.OUT_HEADER.unpack_from(data) == (b"M136BF01", 24, 896, 8, 12800)
    stride = 12800 * 896 * 8 * 2
    assert data.size == M136.OUT_HEADER.size + 24 * stride
    for li, wrapper in enumerate(wrappers):
        bits = np.frombuffer(data, dtype="<u2", count=12800 * 896 * 8,
                             offset=M136.OUT_HEADER.size + li * stride).reshape(12800, 896, 8)
        wrapper.cpu_bank.copy_(torch.from_numpy(M136.bf16_to_f32(bits).copy()))
        assert np.array_equal(wrapper.cpu_bank.to(torch.bfloat16).contiguous().view(
            torch.uint16).numpy(), bits)
        budget(start, device)
    del parent, child, source_a, source_b
    gc.collect()
    counts = {name: np.zeros((24, 12800), dtype=np.int64) for name in SEGMENTS}
    token_counts = {name: 0 for name in SEGMENTS}
    with torch.inference_mode():
        for di, draw in enumerate(draws):
            raw = raw_ids[draw["raw_row"],
                          draw["raw_offset"]:draw["raw_offset"] + 128]
            chat_sequence = chat[draw["chat_index"]][1]
            for kind, sequence in (("raw", raw), ("chat", chat_sequence)):
                context = np.asarray(sequence[:-1], dtype=np.int64)
                for wrapper in wrappers:
                    wrapper.set_token_context(context)
                inputs = torch.as_tensor(context, dtype=torch.long, device=device)[None]
                model(inputs, use_cache=False)
                for li, wrapper in enumerate(wrappers):
                    selected = wrapper.last_selected.cpu().numpy().astype(np.int64)
                    assert selected.shape == (context.size, 4)
                    if kind == "raw":
                        counts["raw"][li] += np.bincount(selected.flatten(), minlength=12800)
                    else:
                        counts["fixed_template"][li] += np.bincount(
                            selected[:common].flatten(), minlength=12800)
                        counts["variable_chat"][li] += np.bincount(
                            selected[common:].flatten(), minlength=12800)
                if kind == "raw":
                    token_counts["raw"] += context.size
                else:
                    token_counts["fixed_template"] += common
                    token_counts["variable_chat"] += context.size - common
            if (di + 1) % 32 == 0:
                print(json.dumps({"draws": di + 1, "budget": budget(start, device)}), flush=True)
    combined = sum(counts.values())
    original = np.asarray(trained["candidate"]["route_counts_by_layer"], dtype=np.int64)
    assert original.shape == combined.shape == (24, 12800)
    assert np.array_equal(original.sum(axis=1), combined.sum(axis=1))
    stage_rows = {segment: summarize(values) for segment, values in counts.items()}
    combined_rows = summarize(combined)
    hotspots = []
    for li in range(24):
        selected = np.argsort(combined[li])[-5:][::-1]
        for slot in selected:
            total = int(combined[li, slot])
            hotspots.append({"layer": li, "slot": int(slot), "total": total,
                             "segment_counts": {
                                 segment: int(counts[segment][li, slot])
                                 for segment in SEGMENTS},
                             "training_time_count": int(original[li, slot])})
    args.counts.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.counts, **counts)
    with np.load(args.counts, allow_pickle=False) as saved:
        assert set(saved.files) == set(SEGMENTS)
        assert all(np.array_equal(saved[name], counts[name]) for name in SEGMENTS)
    result = {"experiment": "METH-147-template-route-attribution",
              "meth145_result_sha256": TRAIN_SHA,
              "candidate_bank_sha256": BANK_SHA,
              "common_chat_prefix_tokens": common,
              "training_chat_sequences": len(chat_ids),
              "draws": len(draws), "token_counts": token_counts,
              "segments": stage_rows, "combined": combined_rows,
              "combined_top_five_per_layer": hotspots,
              "training_to_final_count_l1_fraction_by_layer": [
                  float(np.abs(original[li] - combined[li]).sum() / original[li].sum())
                  for li in range(24)],
              "counts_npz_sha256": M136.digest(args.counts),
              "counts_npz_bytes": args.counts.stat().st_size,
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)},
              "decision": "diagnostic_only_meth145_rejected"}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.counts.stat().st_size + args.out.stat().st_size < 1_000_000_000
    print(json.dumps({"decision": result["decision"],
                      "tokens": token_counts,
                      "worst_hot_share": {
                          name: max(x["worst_hot_parent_share"] or 0 for x in rows)
                          for name, rows in stage_rows.items()},
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
