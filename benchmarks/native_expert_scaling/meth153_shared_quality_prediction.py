#!/usr/bin/env python3
"""First frozen quality gate: new-source BPB and donor prompt top-1."""

import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136
import meth17_fresh_transfer_audit as M17


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
TRAIN = DOC / "meth152_matched_shared_train_result.json"
CATEGORIES = ("code", "prose", "technical_general")
ARMS = ("donor", "original_e1280", "continued_e1280", "shared_e12800")
MAX_SECONDS = 70 * 60
MAX_RSS = 40 * (1 << 30)
MAX_GPU = int(10.5 * (1 << 30))


def check_budget(start, device):
    elapsed, rss, gpu = M17.budget(start, device)
    assert elapsed <= MAX_SECONDS and rss <= MAX_RSS and gpu <= MAX_GPU
    return {"seconds": elapsed, "rss_bytes": rss, "gpu_peak_allocated_bytes": gpu}


def install_bank(wrappers, path, rows):
    assert path.is_file()
    data = np.memmap(path, dtype=np.uint8, mode="r")
    assert M136.OUT_HEADER.unpack_from(data) == (b"M136BF01", 24, 896, 8, rows)
    stride = rows * 896 * 8 * 2
    assert data.size == M136.OUT_HEADER.size + 24 * stride
    for li, wrapper in enumerate(wrappers):
        bits = np.frombuffer(data, dtype="<u2", count=rows * 896 * 8,
                             offset=M136.OUT_HEADER.size + li * stride).reshape(rows, 896, 8)
        values = torch.from_numpy(M136.bf16_to_f32(bits).copy())
        wrapper.cpu_bank.copy_(values)
        readback = wrapper.cpu_bank.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        assert np.array_equal(readback, bits), li
    return {"rows_per_layer": rows, "bf16_readback_exact": True,
            "sha256": M136.digest(path), "bytes": path.stat().st_size}


@torch.inference_mode()
def score_shared_doc(model, ids, wrappers, device, start):
    prefix = torch.as_tensor([model.config.eos_token_id] + ids,
                             dtype=torch.long, device=device)
    nats = 0.0
    for first in range(0, len(ids), M17.STRIDE):
        end = min(first + M17.STRIDE, len(ids))
        lo = max(0, first - M17.CONTEXT)
        window = prefix[lo:end].unsqueeze(0)
        positions = torch.arange(lo, end, dtype=torch.long, device=device).unsqueeze(0)
        for wrapper in wrappers:
            wrapper.set_token_context(window[0].cpu().numpy())
        logits = model(window, position_ids=positions, use_cache=False).logits
        selected = logits[0, first-lo:end-lo].float()
        targets = torch.as_tensor(ids[first:end], dtype=torch.long, device=device)
        nats += float(-F.log_softmax(selected, dim=-1).gather(1, targets[:, None]).sum())
        check_budget(start, device)
    return nats


def summarize(document_rows, prompt_rows):
    result = {"document": {}, "prompt": {}}
    for category in ("pooled", *CATEGORIES):
        documents = document_rows if category == "pooled" else [
            row for row in document_rows if row["category"] == category]
        prompts = prompt_rows if category == "pooled" else [
            row for row in prompt_rows if row["category"] == category]
        assert len(documents) == len(prompts) == (24 if category == "pooled" else 8)
        amount = sum(row["bytes"] for row in documents)
        count = sum(row["positions"] for row in prompts)
        bpb = {arm: sum(row["nats"][arm] for row in documents) /
               (math.log(2) * amount) for arm in ARMS}
        agreement = {arm: sum(row["matching"][arm] for row in prompts) / count
                     for arm in ARMS}
        result["document"][category] = {
            "documents": len(documents), "bytes": amount, "bpb": bpb,
            "candidate_minus_control": bpb["shared_e12800"] - bpb["continued_e1280"]}
        result["prompt"][category] = {
            "prompts": len(prompts), "positions": count,
            "donor_top1_agreement": agreement,
            "candidate_minus_control": agreement["shared_e12800"] - agreement["continued_e1280"]}
    return result


def bootstrap_bound(rows):
    grouped = [[row for row in rows if row["category"] == category]
               for category in CATEGORIES]
    assert all(len(group) == 8 for group in grouped)
    rng = np.random.default_rng(153153)
    gains = []
    for _ in range(10000):
        sample = [group[int(index)] for group in grouped
                  for index in rng.integers(0, len(group), len(group))]
        nats = sum(row["nats"]["continued_e1280"] -
                   row["nats"]["shared_e12800"] for row in sample)
        gains.append(nats / (math.log(2) * sum(row["bytes"] for row in sample)))
    return {"seed": 153153, "draws": 10000,
            "control_minus_candidate_p05": float(np.quantile(gains, 0.05))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--answerability", type=Path, required=True)
    parser.add_argument("--answerability-sha", required=True)
    parser.add_argument("--train-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    for path, sha in ((args.manifest, args.manifest_sha),
                      (args.answerability, args.answerability_sha),
                      (TRAIN, args.train_sha)):
        assert M136.digest(path) == sha, path
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    screen = json.loads(args.answerability.read_text(encoding="utf-8"))
    assert train["decision"] == "matched_training_pass_fresh_quality_pending"
    assert all(train["gates"].values())
    assert manifest["meth152_result_sha256"] == args.train_sha
    assert manifest["selected_counts"] == dict.fromkeys(CATEGORIES, 8)
    items = manifest["items"]
    assert len(items) == 24
    assert screen["manifest_sha256"] == args.manifest_sha
    assert screen["answerable_count"] == 24
    assert [row["source_id"] for row in screen["rows"]] == [row["source_id"] for row in items]
    for item in items:
        assert M136.M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        for key in ("document_ids", "prompt_ids"):
            assert M136.M17.sha(np.asarray(item[key], dtype=np.int32).tobytes()) == item[key+"_sha256"]
    control_bank = Path(manifest["meth152_artifacts"]["control"]["path"])
    candidate_bank = Path(manifest["meth152_artifacts"]["candidate"]["path"])
    assert M136.digest(control_bank) == manifest["meth152_artifacts"]["control"]["sha256"]
    assert M136.digest(candidate_bank) == manifest["meth152_artifacts"]["candidate"]["sha256"]
    assert train["candidate"]["artifact"]["sha256"] == M136.digest(candidate_bank)

    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    M17.MAX_SECONDS = MAX_SECONDS
    M17.MAX_RSS_BYTES = MAX_RSS
    M17.MAX_GPU_BYTES = MAX_GPU
    M136.MAX_SECONDS = MAX_SECONDS
    M136.MAX_RSS = MAX_RSS
    parent, child = M136.bind_inputs()
    source_a, source_b, prefixes = M136.load_factor_bank()
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
            wrappers, banks = M136.make_wrappers(model, parent["expert_state"],
                child["expert_state"], source_a, source_b, prefixes,
                device, mode)
            if arm == "continued_e1280":
                bindings[arm] = install_bank(wrappers, control_bank, 1280)
            elif arm == "shared_e12800":
                bindings[arm] = install_bank(wrappers, candidate_bank, 12800)
        model.eval()
        for row, item in zip(documents, items):
            row["nats"][arm] = (score_shared_doc(model, item["document_ids"],
                wrappers, device, start) if arm == "shared_e12800" else
                M17.score_doc(model, item["document_ids"], wrappers,
                              bool(wrappers), device, start))
        with torch.inference_mode():
            for row, item in zip(prompts, items):
                ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
                if arm == "shared_e12800":
                    for wrapper in wrappers:
                        wrapper.set_token_context(item["prompt_ids"])
                top = model(ids, use_cache=False).logits.argmax(-1)[0].cpu()
                if arm == "donor":
                    donor_top[item["source_id"]] = top
                row["matching"][arm] = int((top == donor_top[item["source_id"]]).sum())
                check_budget(start, device)
        print(json.dumps({"completed_arm": arm, "budget": check_budget(start, device)}), flush=True)
        if arm != "donor":
            for layer, base in zip(model.model.layers, base_mlps):
                layer.mlp = base
            del wrappers, banks
            gc.collect(); torch.cuda.empty_cache()
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
    result = {"experiment": "METH-153-shared-E12800-fresh-prediction",
              "manifest_sha256": args.manifest_sha,
              "answerability_sha256": args.answerability_sha,
              "meth152_result_sha256": args.train_sha,
              "bank_bindings": bindings,
              "document_rows": documents, "prompt_rows": prompts,
              "summary": summary, "bootstrap": bootstrap, "gates": gates,
              "runtime": {**check_budget(start, device),
                          "gpu": torch.cuda.get_device_name(device)},
              "decision": "prediction_pass_generation_pending" if all(gates.values())
                          else "stop_document_prompt_gate"}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "pooled_delta_bpb": summary["document"]["pooled"]["candidate_minus_control"],
                      "bootstrap_p05": bootstrap["control_minus_candidate_p05"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
