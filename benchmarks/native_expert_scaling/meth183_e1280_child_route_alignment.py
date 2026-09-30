#!/usr/bin/env python3
"""Score the learned E1280 bank under exact and cyclic child routes."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/donor_adaptation/s1"))
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57
import meth95_hierarchical_e1280_parity as M95
import meth122_zero_mean_child_external_audit as M122


DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
M122_RESULT = DOC / "meth122_zero_mean_child_external_audit_result.json"
MAX_SECONDS = 20 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
MAX_DISK = 1_000_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ShiftedHierarchicalExperts(M95.HierarchicalExperts):
    route_shift = 0
    route_mode = "capture"

    def routes(self, flat):
        if self.route_mode == "capture":
            parents, scores = super().routes(flat)
            self.frozen_parents = parents.detach().clone()
            self.frozen_scores = scores.detach().clone()
            return parents, scores
        assert self.route_mode == "replay"
        assert self.frozen_parents.shape[0] == flat.shape[0]
        return self.frozen_parents, self.frozen_scores

    def child_route(self, flat, parent_ids):
        if self.route_mode == "capture":
            exact = super().child_route(flat, parent_ids)
            self.frozen_child = exact.detach().clone()
            return exact
        assert self.route_mode == "replay"
        assert torch.equal(parent_ids, self.frozen_parents)
        local = self.frozen_child.remainder(M95.CHILDREN)
        shifted = parent_ids * M95.CHILDREN + (
            local + self.route_shift).remainder(M95.CHILDREN)
        assert torch.equal(shifted.div(M95.CHILDREN, rounding_mode="floor"),
                           parent_ids)
        return shifted


def budget(start, device):
    record = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if record["seconds"] > MAX_SECONDS or record["rss_bytes"] > MAX_RSS or record[
            "gpu_peak_allocated_bytes"] > MAX_GPU:
        raise RuntimeError(f"METH-183 resource stop: {record}")
    return record


def score_replayed_docs(model, items, rows, wrappers, device, started):
    for wrapper in wrappers:
        wrapper.enabled = True
    with torch.inference_mode():
        for row, item in zip(rows, items):
            assert row["source_id"] == item["source_id"]
            ids = item["document_ids"]
            prefix = torch.tensor([model.config.eos_token_id] + ids,
                                  dtype=torch.long, device=device)
            totals = [0.0] * 10
            for first in range(0, len(ids), M17.STRIDE):
                end = min(first + M17.STRIDE, len(ids))
                lo = max(0, first - M17.CONTEXT)
                window = prefix[lo:end].unsqueeze(0)
                positions = torch.arange(lo, end, dtype=torch.long,
                                         device=device).unsqueeze(0)
                targets = torch.as_tensor(ids[first:end], dtype=torch.long,
                                          device=device)
                for shift in range(10):
                    for wrapper in wrappers:
                        wrapper.route_mode = "capture" if shift == 0 else "replay"
                        wrapper.route_shift = shift
                    logits = model(window, position_ids=positions,
                                   use_cache=False).logits
                    selected = logits[0, first-lo:end-lo].float()
                    lp = F.log_softmax(selected, dim=-1)
                    totals[shift] += float(-lp.gather(1, targets[:, None]).sum())
                budget(started, device)
            row["nats"] = {str(shift): total for shift, total in enumerate(totals)}
            print(json.dumps({"completed_document": len([r for r in rows if r["nats"]]),
                              "budget": budget(started, device)}), flush=True)


def group_summary(rows):
    groups = {"pooled": rows}
    for category in M122.CATEGORIES:
        groups[category] = [r for r in rows if r["category"] == category]
        assert len(groups[category]) == 8
    out = {}
    for group, items in groups.items():
        total_bytes = sum(r["bytes"] for r in items)
        bpb = {str(shift): sum(r["nats"][str(shift)] for r in items) /
               (math.log(2) * total_bytes) for shift in range(10)}
        out[group] = {"documents": len(items), "bytes": total_bytes,
                      "bpb_by_shift": bpb,
                      "mean_shift_minus_exact_bpb":
                      sum(bpb[str(s)] for s in range(1, 10)) / 9 - bpb["0"]}
    return out


def bootstrap(rows):
    rng = np.random.default_rng(183183)
    differences = np.array([
        sum(row["nats"][str(s)] for s in range(1, 10)) / 9 -
        row["nats"]["0"] for row in rows], dtype=np.float64)
    byte_counts = np.array([row["bytes"] for row in rows], dtype=np.float64)
    indices = rng.integers(0, len(rows), size=(10000, len(rows)))
    values = differences[indices].sum(axis=1) / (
        math.log(2) * byte_counts[indices].sum(axis=1))
    return {"seed": 183183, "draws": 10000,
            "mean_shift_minus_exact_p05_bpb": float(np.quantile(values, .05)),
            "mean_shift_minus_exact_median_bpb": float(np.quantile(values, .5))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage = "bind"
    device = None
    rows = []
    try:
        bindings = ((M122.MANIFEST, M122.MANIFEST_SHA),
                    (M122.SPECIALIZED, M122.SPECIALIZED_SHA),
                    (M122.TRAIN_REPORT, M122.TRAIN_REPORT_SHA),
                    (M122.TRAINING, M57.TRAINING_SHA))
        for path, sha in bindings:
            assert digest(path) == sha, path
        manifest = json.loads(M122.MANIFEST.read_text(encoding="utf-8"))
        prior = json.loads(M122_RESULT.read_text(encoding="utf-8"))
        assert prior["specialized_checkpoint_sha256"] == M122.SPECIALIZED_SHA
        items = manifest["items"]
        assert len(items) == len(prior["document_rows"]) == 24
        assert [item["source_id"] for item in items] == [
            row["source_id"] for row in prior["document_rows"]]
        for item in items:
            assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
            assert M17.sha(np.asarray(item["document_ids"], dtype=np.int32).tobytes()) == item[
                "document_ids_sha256"]
            rows.append({"source_id": item["source_id"], "category": item["category"],
                         "bytes": item["bytes"], "nats": {}})
        parent_report = json.loads(M122.TRAINING.read_text(encoding="utf-8"))
        parent = parent_report["checkpoints"]["512"]
        assert digest(parent["path"]) == M57.CHECKPOINT_SHA
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        donor_path = hf_hub_download(M42.MODEL, "model.safetensors",
                                     revision=M42.REV, local_files_only=True)
        assert digest(donor_path) == M57.MODEL_SHA
        tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                                   local_files_only=True)
        assert M122.M15.M13.C.tok_fingerprint(tokenizer) == M122.M15.M13.TOK_FP
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

        stage = "load_model_and_exact_bank"
        model = AutoModelForCausalLM.from_pretrained(
            M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).to(device).eval()
        model.config.use_cache = False
        parent_state = torch.load(parent["path"], map_location="cpu", weights_only=False)
        child_state = torch.load(M122.SPECIALIZED, map_location="cpu", weights_only=False)
        assert parent_state["updates"] == 512 and child_state["updates"] == 256
        assert child_state["parent_checkpoint_sha256"] == M57.CHECKPOINT_SHA
        wrappers = []
        max_center_error = 0.0
        with torch.no_grad():
            for li, layer in enumerate(model.model.layers):
                source = M55.ProductKeyExperts(layer.mlp, li).to(device)
                for key in ("a", "b", "router"):
                    getattr(source, key).copy_(parent_state["expert_state"][li][key].to(device))
                wrapper = ShiftedHierarchicalExperts(source, li).to(device)
                saved = child_state["expert_state"][li]
                for key in ("a", "b", "router", "child_projection", "child_keys"):
                    getattr(wrapper, key).copy_(saved[key].to(device))
                raw_b = wrapper.b.view(M122.M15.E, M95.CHILDREN,
                                       M122.M15.M13.D, M122.M15.R)
                centered = source.b[:, None] + raw_b - raw_b.mean(dim=1, keepdim=True)
                wrapper.b.copy_(centered.reshape_as(wrapper.b))
                error = (wrapper.b.view_as(centered).mean(dim=1) - source.b).abs().max()
                max_center_error = max(max_center_error, float(error))
                layer.mlp = wrapper
                wrappers.append(wrapper)
        assert len(wrappers) == 24 and max_center_error <= 1e-7
        del parent_state, child_state
        budget(started, device)

        stage = "score_routes"
        for shift in range(10):
            assert sorted((i + shift) % 10 for i in range(10)) == list(range(10))
        score_replayed_docs(model, items, rows, wrappers, device, started)
        errors = [abs(row["nats"]["0"] - old["nats"]["bf16_e1280"])
                  for row, old in zip(rows, prior["document_rows"])]
        assert max(errors) <= 0.02, {"max_exact_nats_error": max(errors)}

        stage = "summary"
        summary = group_summary(rows)
        boot = bootstrap(rows)
        exact = summary["pooled"]["bpb_by_shift"]["0"]
        gates = {"exact_beats_all_shifts": all(
                     exact < summary["pooled"]["bpb_by_shift"][str(s)] for s in range(1, 10)),
                 "paired_bootstrap_positive": boot["mean_shift_minus_exact_p05_bpb"] > 0}
        result = {"experiment": "METH-183-E1280-learned-child-route-alignment",
                  "manifest_sha256": M122.MANIFEST_SHA,
                  "specialized_checkpoint_sha256": M122.SPECIALIZED_SHA,
                  "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
                  "prior_result_sha256": digest(M122_RESULT),
                  "source_sha256": M57.MODEL_SHA,
                  "centered_parent_mean_max_abs_error": max_center_error,
                  "exact_nats_max_abs_error": max(errors),
                  "document_rows": rows, "summary_by_category": summary,
                  "bootstrap": boot, "gates": gates,
                  "decision": "route_alignment_diagnostic_pass" if all(gates.values())
                              else "route_alignment_diagnostic_fail",
                  "runtime": {**budget(started, device), "gpu": torch.cuda.get_device_name(device)},
                  "scope": "Previously consumed METH-121 documents; mechanism diagnostic only"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < MAX_DISK
        print(json.dumps({"decision": result["decision"],
                          "exact_bpb": exact,
                          "mean_shift_minus_exact_bpb":
                              summary["pooled"]["mean_shift_minus_exact_bpb"],
                          "paired_p05": boot["mean_shift_minus_exact_p05_bpb"]}),
              flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-183-failure", "stage": stage,
                                       "error": repr(error), "partial_document_rows": rows,
                                       "elapsed_seconds": time.monotonic() - started,
                                       "rss_bytes": psutil.Process().memory_info().rss,
                                       "gpu_peak_allocated_bytes":
                                           torch.cuda.max_memory_allocated(device)
                                           if device is not None else None},
                                      indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
