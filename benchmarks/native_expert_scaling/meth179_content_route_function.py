#!/usr/bin/env python3
"""Score one frozen trained bank under exact versus rotated content routes."""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from transformers import AutoTokenizer

import meth136_sparse_experts  # inserts the donor-adaptation S1 path
import meth136_matched_sparse_train as M136
import meth17_fresh_transfer_audit as M17
import meth151_shared_sparse_experts as M151
import meth156_factorized_quality_prediction as M156
import meth175_matched_long_train as M175
import meth176_long_quality_prediction as M176


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DOC / "meth173_fresh_recurrence_route_manifest.json"
MANIFEST_SHA = "6ebfc36fd452f2a20cfcc9e8210505e36b725ef95992949655c19f3c94eaee20"
CANDIDATE = DOC / "meth175_candidate_long_result.json"
CANDIDATE_SHA = "5b8de7a93c353b83393b0d2189a2cd594593c298dd867a998ca97d5d35ee4c64"
MAX_SECONDS = 20 * 60
MAX_RSS = 40 * (1 << 30)
MAX_GPU = int(10.5 * (1 << 30))
MAX_DISK = 1_000_000_000


class ShiftedContentExperts(M176.LowRecurrenceEvalExperts):
    route_shift = 0

    def selected_ids(self, flat, children):
        selected = super().selected_ids(flat, children)
        if self.route_shift:
            local = selected.remainder(10)
            rotated = selected - local + 1 + (local - 1 + self.route_shift).remainder(9)
            selected = torch.where(local == 0, selected, rotated)
            assert torch.equal(selected.div(10, rounding_mode="floor"), children)
            assert torch.equal(selected.remainder(10) == 0, local == 0)
        return selected


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS or result[
            "gpu_peak_allocated_bytes"] > MAX_GPU:
        raise RuntimeError(f"METH-179 resource stop: {result}")
    return result


def bootstrap(rows):
    rng = np.random.default_rng(179179)
    draws = []
    for _ in range(10000):
        sampled = [rows[int(i)] for i in rng.integers(0, len(rows), len(rows))]
        gain = sum((sum(row["nats"][str(shift)] for shift in range(1, 9)) / 8
                    - row["nats"]["0"]) for row in sampled)
        draws.append(gain / (math.log(2) * sum(row["bytes"] for row in sampled)))
    return {"seed": 179179, "draws": 10000,
            "mean_shift_minus_exact_p05_bpb": float(np.quantile(draws, .05)),
            "mean_shift_minus_exact_median_bpb": float(np.quantile(draws, .5))}


def summarize(rows):
    total_bytes = sum(row["bytes"] for row in rows)
    bpb = {str(shift): sum(row["nats"][str(shift)] for row in rows) /
           (math.log(2) * total_bytes) for shift in range(9)}
    exact = bpb["0"]
    shifts = [bpb[str(shift)] for shift in range(1, 9)]
    ci = bootstrap(rows)
    gates = {"exact_better_than_each_rotation": all(exact < value for value in shifts),
             "bootstrap_mean_rotation_worse": ci["mean_shift_minus_exact_p05_bpb"] > 0}
    return {"bytes": total_bytes, "bpb_by_shift": bpb,
            "exact_minus_each_shift_bpb": {str(shift): exact - bpb[str(shift)]
                                           for shift in range(1, 9)},
            "mean_rotation_minus_exact_bpb": sum(shifts) / 8 - exact,
            "bootstrap": ci, "diagnostic_gates": gates,
            "decision": ("route_alignment_diagnostic_pass" if all(gates.values())
                         else "route_alignment_diagnostic_fail")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    started = time.monotonic()
    rows = []
    stage = "bindings"
    device = None
    try:
        assert digest(MANIFEST) == MANIFEST_SHA
        assert digest(CANDIDATE) == CANDIDATE_SHA
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        assert candidate["decision"] == "matched_long_candidate_pass_fresh_quality_pending"
        assert all(candidate["gates"].values())
        assert manifest["table_sha256"] == M175.TABLE_SHA
        assert manifest["tokenizer_fingerprint"] == M136.M15.M13.TOK_FP
        bank = Path(candidate["artifact"]["path"])
        assert digest(bank) == candidate["artifact"]["sha256"]
        source_rows = manifest["document_rows"]
        assert len(source_rows) == len({row["source_row"] for row in source_rows}) == 24
        tokenizer = AutoTokenizer.from_pretrained(M136.M42.MODEL,
                                                  revision=M136.M42.REV,
                                                  local_files_only=True)
        assert M136.M15.M13.C.tok_fingerprint(tokenizer) == manifest[
            "tokenizer_fingerprint"]
        for item in source_rows:
            ids = np.asarray(item["span_ids"], dtype=np.int32)
            assert len(ids) == 1024 and M17.sha(ids.tobytes()) == item["span_ids_sha256"]
            byte_count = len(tokenizer.decode(ids.tolist(),
                                              skip_special_tokens=False).encode("utf-8"))
            assert byte_count > 0
            rows.append({"source_row": item["source_row"],
                         "span_ids_sha256": item["span_ids_sha256"],
                         "bytes": byte_count, "nats": {}})
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count())
                   if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
        assert len(matches) == 1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        M17.MAX_SECONDS = M176.MAX_SECONDS = M136.MAX_SECONDS = MAX_SECONDS
        M17.MAX_RSS_BYTES = M176.MAX_RSS = M136.MAX_RSS = MAX_RSS
        M17.MAX_GPU_BYTES = M176.MAX_GPU = M136.MAX_GPU = MAX_GPU
        table = M175.load_table()
        stage = "source_parity"
        parent, child = M136.bind_inputs()
        source_a, source_b, prefixes = M136.load_factor_bank()
        parity = M176.initial_parity(parent, child, source_a, source_b,
                                     prefixes, table, device, started)
        stage = "candidate_bank"
        model = M136.model_shell(device).eval()
        original_class = M151.SharedStructureSparseExperts
        ShiftedContentExperts.route_table = table
        M151.SharedStructureSparseExperts = ShiftedContentExperts
        try:
            wrappers, _ = M136.make_wrappers(
                model, parent["expert_state"], child["expert_state"],
                source_a, source_b, prefixes, device, "shared")
        finally:
            M151.SharedStructureSparseExperts = original_class
        assert all(isinstance(wrapper, ShiftedContentExperts) for wrapper in wrappers)
        binding = M156.install_bank(wrappers, bank, 12800)
        assert binding["sha256"] == candidate["artifact"]["sha256"]
        del parent, child, source_a, source_b
        gc.collect()
        stage = "nine_route_scores"
        with torch.inference_mode():
            for shift in range(9):
                assert sorted([1 + ((j - 1 + shift) % 9) for j in range(1, 10)]) == list(range(1, 10))
                for wrapper in wrappers:
                    wrapper.route_shift = shift
                for output, item in zip(rows, source_rows):
                    assert output["source_row"] == item["source_row"]
                    output["nats"][str(shift)] = M176.score_candidate_doc(
                        model, item["span_ids"], wrappers, device, started)
                    budget(started, device)
                print(json.dumps({"completed_shift": shift,
                                  "budget": budget(started, device)}), flush=True)
        stage = "summary"
        summary = summarize(rows)
        result = {"experiment": "METH-179-consumed-route-functional-ablation",
                  "manifest_sha256": MANIFEST_SHA,
                  "candidate_result_sha256": CANDIDATE_SHA,
                  "candidate_bank_sha256": binding["sha256"],
                  "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
                  "initial_bf16_parity": parity, "bank_readback": binding,
                  "document_rows": rows, "summary": summary,
                  "runtime": {**budget(started, device),
                              "gpu": torch.cuda.get_device_name(device)},
                  "scope": "Consumed METH-173 route cohort; mechanism diagnostic only"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < MAX_DISK
        print(json.dumps({"decision": summary["decision"],
                          "exact_bpb": summary["bpb_by_shift"]["0"],
                          "mean_rotation_minus_exact_bpb":
                              summary["mean_rotation_minus_exact_bpb"],
                          "bootstrap_p05":
                              summary["bootstrap"]["mean_shift_minus_exact_p05_bpb"]}),
              flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-179-ablation-failure",
                                       "stage": stage, "error": repr(error),
                                       "partial_rows": rows,
                                       "elapsed_seconds": time.monotonic() - started,
                                       "rss_bytes": psutil.Process().memory_info().rss,
                                       "gpu_peak_allocated_bytes":
                                           torch.cuda.max_memory_allocated(device)
                                           if device is not None else None},
                                      indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
