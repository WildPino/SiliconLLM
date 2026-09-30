#!/usr/bin/env python3
"""Frozen METH-176 held-out BPB and donor-prompt top-1 gate."""

import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth17_fresh_transfer_audit as M17
import meth151_shared_sparse_experts as M151
import meth175_matched_long_train as M175
import meth156_factorized_quality_prediction as M156


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
CONTROL = DOC / "meth175_control_long_result.json"
CANDIDATE = DOC / "meth175_candidate_long_result.json"
CATEGORIES = ("code", "prose", "technical_general")
ARMS = ("donor", "original_e1280", "continued_e1280", "factorized_e12800")
MAX_SECONDS = 70 * 60
MAX_RSS = 40 * (1 << 30)
MAX_GPU = int(10.5 * (1 << 30))
MAX_DISK = 10_000_000_000


class LowRecurrenceEvalExperts(M151.SharedStructureSparseExperts):
    """Read the learned combined bank with the exact METH-175 route."""

    route_table = None
    selected_ids = M175.LowRecurrenceFactorizedExperts.selected_ids


def check_budget(start, device):
    elapsed, rss, gpu = M17.budget(start, device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS or gpu > MAX_GPU:
        raise RuntimeError(f"METH-176 prediction resource stop: {elapsed}, {rss}, {gpu}")
    return {"seconds": elapsed, "rss_bytes": rss,
            "gpu_peak_allocated_bytes": gpu}


@torch.inference_mode()
def score_candidate_doc(model, ids, wrappers, device, start):
    for wrapper in wrappers:
        wrapper.enabled = True
    prefix = torch.as_tensor([model.config.eos_token_id] + ids,
                             dtype=torch.long, device=device)
    nats = 0.0
    for first in range(0, len(ids), M17.STRIDE):
        end = min(first + M17.STRIDE, len(ids))
        lo = max(0, first - M17.CONTEXT)
        window = prefix[lo:end].unsqueeze(0)
        positions = torch.arange(lo, end, dtype=torch.long, device=device).unsqueeze(0)
        tokens = window[0].cpu().numpy().astype(np.int64)
        previous = np.r_[int(prefix[lo - 1]) if lo else 0, tokens[:-1]]
        route_positions = np.arange(lo, end, dtype=np.int64)
        for wrapper in wrappers:
            wrapper.set_explicit_token_context(tokens, previous, route_positions)
        logits = model(window, position_ids=positions, use_cache=False).logits
        selected = logits[0, first - lo:end - lo].float()
        targets = torch.as_tensor(ids[first:end], dtype=torch.long, device=device)
        nats += float(-F.log_softmax(selected, dim=-1).gather(1, targets[:, None]).sum())
        check_budget(start, device)
    return nats


def summarize(documents, prompts):
    result = {"document": {}, "prompt": {}}
    for category in ("pooled", *CATEGORIES):
        docs = documents if category == "pooled" else [
            row for row in documents if row["category"] == category]
        asked = prompts if category == "pooled" else [
            row for row in prompts if row["category"] == category]
        assert len(docs) == len(asked) == (24 if category == "pooled" else 8)
        amount = sum(row["bytes"] for row in docs)
        positions = sum(row["positions"] for row in asked)
        bpb = {arm: sum(row["nats"][arm] for row in docs) /
               (math.log(2) * amount) for arm in ARMS}
        agreement = {arm: sum(row["matching"][arm] for row in asked) / positions
                     for arm in ARMS}
        result["document"][category] = {
            "documents": len(docs), "bytes": amount, "bpb": bpb,
            "candidate_minus_control": bpb["factorized_e12800"] - bpb["continued_e1280"]}
        result["prompt"][category] = {
            "prompts": len(asked), "positions": positions,
            "donor_top1_agreement": agreement,
            "candidate_minus_control": (agreement["factorized_e12800"] -
                                        agreement["continued_e1280"])}
    return result


def bootstrap_bound(rows):
    grouped = [[row for row in rows if row["category"] == category]
               for category in CATEGORIES]
    assert all(len(group) == 8 for group in grouped)
    rng = np.random.default_rng(176176)
    gains = []
    for _ in range(10000):
        sampled = [group[int(index)] for group in grouped
                   for index in rng.integers(0, len(group), len(group))]
        nats = sum(row["nats"]["continued_e1280"] -
                   row["nats"]["factorized_e12800"] for row in sampled)
        gains.append(nats / (math.log(2) * sum(row["bytes"] for row in sampled)))
    return {"seed": 176176, "draws": 10000,
            "control_minus_candidate_p05": float(np.quantile(gains, 0.05))}


def bind_inputs(args):
    assert not args.out.exists()
    for path, sha in ((args.manifest, args.manifest_sha),
                      (args.answerability, args.answerability_sha),
                      (CONTROL, args.control_sha), (CANDIDATE, args.candidate_sha)):
        assert M136.digest(path) == sha, path
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    answerability = json.loads(args.answerability.read_text(encoding="utf-8"))
    control = json.loads(CONTROL.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    assert control["decision"] == "matched_long_control_complete"
    assert candidate["decision"] == "matched_long_candidate_pass_fresh_quality_pending"
    assert all(control["gates"].values()) and all(candidate["gates"].values())
    assert manifest["training_bindings"]["control"]["result_sha256"] == args.control_sha
    assert manifest["training_bindings"]["candidate"]["result_sha256"] == args.candidate_sha
    assert manifest["exact_bank_sha256"] == M136.EXACT_SHA
    assert manifest["child_checkpoint_sha256"] == M136.CHILD_SHA
    assert manifest["parent_checkpoint_sha256"] == M136.M57.CHECKPOINT_SHA
    assert manifest["donor_sha256"] == M136.M57.MODEL_SHA
    assert manifest["draws_sha256"] == M175.DRAWS_SHA
    assert manifest["route_table_sha256"] == M175.TABLE_SHA
    assert manifest["selected_counts"] == dict.fromkeys(CATEGORIES, 8)
    items = manifest["items"]
    assert len(items) == 24
    assert answerability["manifest_sha256"] == args.manifest_sha
    assert answerability["answerable_count"] == 24
    assert [row["source_id"] for row in answerability["rows"]] == [
        row["source_id"] for row in items]
    assert all(row["answerable"] and row["anchor"] in item["excerpt"]
               for row, item in zip(answerability["rows"], items))
    for item in items:
        assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        for key in ("document_ids", "prompt_ids"):
            assert M17.sha(np.asarray(item[key], dtype=np.int32).tobytes()) == item[key + "_sha256"]
    banks = {}
    for arm, result in (("control", control), ("candidate", candidate)):
        bank = Path(manifest["training_bindings"][arm]["bank_path"])
        assert bank == Path(result["artifact"]["path"])
        assert manifest["training_bindings"][arm]["bank_sha256"] == result["artifact"]["sha256"]
        assert manifest["training_bindings"][arm]["bank_bytes"] == bank.stat().st_size
        assert M136.digest(bank) == result["artifact"]["sha256"]
        banks[arm] = bank
    return items, banks


def initial_parity(parent, child, source_a, source_b, prefixes,
                   table, device, start):
    teacher = M136.model_shell(device).eval()
    M136.make_wrappers(teacher, parent["expert_state"], child["expert_state"],
                       source_a, source_b, prefixes, device, "teacher")
    teacher.eval()
    student = M136.model_shell(device).eval()
    original_class = M151.SharedStructureSparseExperts
    LowRecurrenceEvalExperts.route_table = table
    M151.SharedStructureSparseExperts = LowRecurrenceEvalExperts
    try:
        M136.make_wrappers(student, parent["expert_state"], child["expert_state"],
                           source_a, source_b, prefixes, device, "shared")
        parity_items = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"]
        parity = M136.parity_check(teacher, student, parity_items, device)
    finally:
        M151.SharedStructureSparseExperts = original_class
    del teacher, student
    gc.collect(); torch.cuda.empty_cache()
    check_budget(start, device)
    return parity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--answerability", type=Path, required=True)
    parser.add_argument("--answerability-sha", required=True)
    parser.add_argument("--control-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stage = "bindings"
    documents, prompts = [], []
    device = None
    started = time.monotonic()
    try:
        items, banks = bind_inputs(args)
        table = M175.load_table()
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count())
                   if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
        assert len(matches) == 1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        M17.MAX_SECONDS = MAX_SECONDS
        M17.MAX_RSS_BYTES = MAX_RSS
        M17.MAX_GPU_BYTES = MAX_GPU
        M136.MAX_SECONDS = MAX_SECONDS
        M136.MAX_RSS = MAX_RSS
        stage = "source_parity"
        parent, child = M136.bind_inputs()
        source_a, source_b, prefixes = M136.load_factor_bank()
        parity = initial_parity(parent, child, source_a, source_b,
                                prefixes, table, device, started)
        stage = "four_arm_prediction"
        model = M136.model_shell(device).eval()
        base_mlps = [layer.mlp for layer in model.model.layers]
        documents = [{"source_id": row["source_id"], "category": row["category"],
                      "bytes": row["bytes"], "nats": {}} for row in items]
        prompts = [{"source_id": row["source_id"], "category": row["category"],
                    "positions": len(row["prompt_ids"]), "matching": {}}
                   for row in items]
        donor_top = {}
        bindings = {}
        for arm in ARMS:
            if arm == "donor":
                wrappers = []
            else:
                for layer, base in zip(model.model.layers, base_mlps):
                    layer.mlp = base
                gc.collect(); torch.cuda.empty_cache()
                mode = "teacher" if arm == "original_e1280" else (
                    "control" if arm == "continued_e1280" else "shared")
                if arm == "factorized_e12800":
                    original_class = M151.SharedStructureSparseExperts
                    LowRecurrenceEvalExperts.route_table = table
                    M151.SharedStructureSparseExperts = LowRecurrenceEvalExperts
                try:
                    wrappers, sparse_banks = M136.make_wrappers(
                        model, parent["expert_state"], child["expert_state"],
                        source_a, source_b, prefixes, device, mode)
                finally:
                    if arm == "factorized_e12800":
                        M151.SharedStructureSparseExperts = original_class
                if arm == "continued_e1280":
                    bindings[arm] = M156.install_bank(wrappers, banks["control"], 1280)
                elif arm == "factorized_e12800":
                    bindings[arm] = M156.install_bank(wrappers, banks["candidate"], 12800)
                    assert all(isinstance(wrapper, LowRecurrenceEvalExperts)
                               for wrapper in wrappers)
            model.eval()
            for output, item in zip(documents, items):
                output["nats"][arm] = (
                    score_candidate_doc(model, item["document_ids"], wrappers,
                                        device, started)
                    if arm == "factorized_e12800" else
                    M17.score_doc(model, item["document_ids"], wrappers,
                                  bool(wrappers), device, started))
            with torch.inference_mode():
                for output, item in zip(prompts, items):
                    ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                          device=device)[None]
                    if arm == "factorized_e12800":
                        for wrapper in wrappers:
                            wrapper.set_token_context(item["prompt_ids"])
                    top = model(ids, use_cache=False).logits.argmax(-1)[0].cpu()
                    if arm == "donor":
                        donor_top[item["source_id"]] = top
                    output["matching"][arm] = int((
                        top == donor_top[item["source_id"]]).sum())
                    check_budget(started, device)
            print(json.dumps({"completed_arm": arm,
                              "budget": check_budget(started, device)}), flush=True)
            if arm != "donor":
                for layer, base in zip(model.model.layers, base_mlps):
                    layer.mlp = base
                del wrappers, sparse_banks
                gc.collect(); torch.cuda.empty_cache()
        stage = "quality_gates"
        summary = summarize(documents, prompts)
        bootstrap = bootstrap_bound(documents)
        gates = {
            "pooled_bpb_gain": summary["document"]["pooled"]["candidate_minus_control"] <= -0.00005,
            "bootstrap_gain": bootstrap["control_minus_candidate_p05"] > 0,
            "category_bpb": all(summary["document"][category]["candidate_minus_control"] <= 0.02
                                for category in CATEGORIES),
            "pooled_donor_top1": summary["prompt"]["pooled"]["candidate_minus_control"] >= -0.01,
            "category_donor_top1": all(summary["prompt"][category]["candidate_minus_control"] >= -0.02
                                       for category in CATEGORIES)}
        result = {"experiment": "METH-176-long-E12800-fresh-prediction",
                  "manifest_sha256": args.manifest_sha,
                  "answerability_sha256": args.answerability_sha,
                  "control_result_sha256": args.control_sha,
                  "candidate_result_sha256": args.candidate_sha,
                  "initial_bf16_parity": parity,
                  "bank_bindings": bindings,
                  "document_rows": documents, "prompt_rows": prompts,
                  "summary": summary, "bootstrap": bootstrap, "gates": gates,
                  "runtime": {**check_budget(started, device),
                              "gpu": torch.cuda.get_device_name(device)},
                  "decision": "prediction_pass_generation_pending" if all(gates.values())
                              else "stop_document_prompt_gate"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < MAX_DISK
        print(json.dumps({"decision": result["decision"], "gates": gates,
                          "pooled_delta_bpb": summary["document"]["pooled"]["candidate_minus_control"],
                          "bootstrap_p05": bootstrap["control_minus_candidate_p05"],
                          "runtime": result["runtime"]}, indent=2), flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({
            "experiment": "METH-176-long-prediction-failure",
            "stage": stage, "error": repr(error),
            "document_rows": documents, "prompt_rows": prompts,
            "elapsed_seconds": time.monotonic() - started,
            "rss_bytes": psutil.Process().memory_info().rss,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                                        if device is not None else None},
            indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
