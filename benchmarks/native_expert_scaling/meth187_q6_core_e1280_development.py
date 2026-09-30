#!/usr/bin/env python3
"""Four-arm development score of stored Q6 donor core plus centered E1280."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from huggingface_hub import hf_hub_download
import numpy as np
import psutil
from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/donor_adaptation/s1"))
import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth24_export_r8_core as M24
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57
import meth95_hierarchical_e1280_parity as M95
import meth122_zero_mean_child_external_audit as M122
import meth186_export_group64_q6_core as M186


DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
CORE = ROOT / "results/native_expert_scaling/meth186_qwen05b_instruct_group64_q6_core.safetensors"
EXPORT = DOC / "meth186_group64_q6_core_export.json"
PRIOR = DOC / "meth122_zero_mean_child_external_audit_result.json"
ARMS = ("bf16_donor", "bf16_e1280", "q6_donor", "q6_e1280")
CATEGORIES = ("code", "prose", "technical_general")
MAX_SECONDS = 15 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)
MAX_DISK = 1_000_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def budget(start, device):
    record = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if record["seconds"] > MAX_SECONDS or record["rss_bytes"] > MAX_RSS or record[
            "gpu_peak_allocated_bytes"] > MAX_GPU:
        raise RuntimeError(f"METH-187 resource stop: {record}")
    return record


def summaries(rows, prompt_rows):
    out = {}
    for category in ("pooled", *CATEGORIES):
        documents = rows if category == "pooled" else [
            row for row in rows if row["category"] == category]
        prompts = prompt_rows if category == "pooled" else [
            row for row in prompt_rows if row["category"] == category]
        assert len(documents) == len(prompts) == (24 if category == "pooled" else 8)
        byte_count = sum(row["bytes"] for row in documents)
        positions = sum(row["positions"] for row in prompts)
        bpb = {arm: sum(row["nats"][arm] for row in documents) /
               (math.log(2) * byte_count) for arm in ARMS}
        matching = {arm: sum(row["matching"][arm] for row in prompts) for arm in ARMS}
        agreement = {arm: matching[arm] / positions for arm in ARMS}
        out[category] = {"documents": len(documents), "bytes": byte_count,
                         "positions": positions, "bpb": bpb,
                         "matching": matching, "agreement": agreement,
                         "q6_minus_bf16_e1280_bpb": bpb["q6_e1280"] - bpb["bf16_e1280"],
                         "q6_minus_bf16_e1280_top1":
                             agreement["q6_e1280"] - agreement["bf16_e1280"]}
    return out


def write_progress(path, stage, rows, prompts, start, device):
    path.write_text(json.dumps({"experiment": "METH-187-progress",
                                "stage": stage, "document_rows": rows,
                                "prompt_rows": prompts,
                                "runtime": budget(start, device)}, indent=2) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.stem + ".partial.json")
    start = time.monotonic()
    device = None
    stage = "bindings"
    rows = []
    prompts = []
    try:
        assert digest(M122.MANIFEST) == M122.MANIFEST_SHA
        assert digest(M122.SPECIALIZED) == M122.SPECIALIZED_SHA
        assert digest(M122.TRAINING) == M57.TRAINING_SHA
        assert digest(CORE) == json.loads(EXPORT.read_text(encoding="utf-8"))["sha256"]
        report = json.loads(EXPORT.read_text(encoding="utf-8"))
        assert report["source_sha256"] == M57.MODEL_SHA
        assert report["ideal_addressed_bytes_per_token"] == 481_534_976
        manifest = json.loads(M122.MANIFEST.read_text(encoding="utf-8"))
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        items = manifest["items"]
        assert len(items) == len(prior["document_rows"]) == len(prior["prompt_rows"]) == 24
        assert [item["source_id"] for item in items] == [
            row["source_id"] for row in prior["document_rows"]]
        for item in items:
            assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
            for field in ("document_ids", "prompt_ids"):
                assert M17.sha(np.asarray(item[field], dtype=np.int32).tobytes()) == item[
                    field + "_sha256"]
            rows.append({"source_id": item["source_id"], "category": item["category"],
                         "bytes": item["bytes"], "nats": {}})
            prompts.append({"source_id": item["source_id"], "category": item["category"],
                            "positions": len(item["prompt_ids"]), "matching": {}})
        parent_report = json.loads(M122.TRAINING.read_text(encoding="utf-8"))
        parent = parent_report["checkpoints"]["512"]
        assert digest(parent["path"]) == M57.CHECKPOINT_SHA
        source_path = Path(hf_hub_download(M42.MODEL, "model.safetensors",
                                           revision=M42.REV, local_files_only=True))
        assert digest(source_path) == M57.MODEL_SHA
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                                   local_files_only=True)
        assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
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

        stage = "load_exact_e1280"
        model = AutoModelForCausalLM.from_pretrained(
            M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).to(device).eval()
        model.config.use_cache = False
        original_params = dict(model.named_parameters())
        assert len(original_params) == 290
        parent_state = torch.load(parent["path"], map_location="cpu", weights_only=False)
        child_state = torch.load(M122.SPECIALIZED, map_location="cpu", weights_only=False)
        assert parent_state["updates"] == 512 and child_state["updates"] == 256
        assert child_state["parent_checkpoint_sha256"] == M57.CHECKPOINT_SHA
        wrappers = []
        mean_error = 0.0
        with torch.no_grad():
            for li, layer in enumerate(model.model.layers):
                base = M55.ProductKeyExperts(layer.mlp, li).to(device)
                for key in ("a", "b", "router"):
                    getattr(base, key).copy_(parent_state["expert_state"][li][key].to(device))
                wrapper = M95.HierarchicalExperts(base, li).to(device)
                saved = child_state["expert_state"][li]
                for key in ("a", "b", "router", "child_projection", "child_keys"):
                    getattr(wrapper, key).copy_(saved[key].to(device))
                raw_b = wrapper.b.view(M15.E, M95.CHILDREN, M15.M13.D, M15.R)
                centered = base.b[:, None] + raw_b - raw_b.mean(dim=1, keepdim=True)
                wrapper.b.copy_(centered.reshape_as(wrapper.b))
                mean_error = max(mean_error, float((wrapper.b.view_as(centered).mean(dim=1)
                                                   - base.b).abs().max()))
                layer.mlp = wrapper
                wrappers.append(wrapper)
        assert mean_error <= 1e-7
        del parent_state, child_state
        budget(start, device)

        stage = "bf16_arms"
        donor_top = {}
        with torch.inference_mode():
            for arm, enabled in (("bf16_donor", False), ("bf16_e1280", True)):
                for row, item in zip(rows, items):
                    row["nats"][arm] = M17.score_doc(
                        model, item["document_ids"], wrappers, enabled, device, start)
                    budget(start, device)
                M44.set_experts(wrappers, enabled)
                for row, item in zip(prompts, items):
                    ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                          device=device)[None]
                    top = model(ids, use_cache=False).logits.argmax(-1)[0].cpu()
                    if arm == "bf16_donor":
                        donor_top[item["source_id"]] = top
                    row["matching"][arm] = int((top == donor_top[item["source_id"]]).sum())
                write_progress(partial, arm, rows, prompts, start, device)
                print(json.dumps({"completed_arm": arm,
                                  "budget": budget(start, device)}), flush=True)
        for row, old in zip(rows, prior["document_rows"]):
            assert row["nats"]["bf16_donor"] == old["nats"]["bf16_donor"]
            assert row["nats"]["bf16_e1280"] == old["nats"]["bf16_e1280"]
        for row, old in zip(prompts, prior["prompt_rows"]):
            assert row["matching"]["bf16_e1280"] == old["matching"]["bf16_e1280"]

        stage = "load_q6_core"
        with safe_open(str(CORE), framework="pt", device="cpu") as archive:
            metadata = archive.metadata()
            assert metadata["format"] == "QWEN25_INSTRUCT_R8H_GROUP64_Q6FFN_BF16ATTN_V1"
            assert metadata["model_sha256"] == M57.MODEL_SHA
            assert len(archive.keys()) == 363
            counts = {key: 0 for key in ("tied_head", "attention", "ffn", "control")}
            with torch.no_grad():
                for name, param in original_params.items():
                    organ = M24.classify(name, tuple(param.shape))
                    counts[organ] += 1
                    if organ == "ffn":
                        packed = archive.get_tensor(name + ".q6").to(device)
                        scales = archive.get_tensor(name + ".scale").to(device)
                        effective = M186.reconstruct(packed, scales)
                        assert tuple(effective.shape) == tuple(param.shape)
                        param.copy_(effective)
                    elif organ == "tied_head":
                        codes = archive.get_tensor(name + ".q").to(device)
                        scales = archive.get_tensor(name + ".scale").to(device)
                        assert codes.dtype == torch.int8 and scales.dtype == torch.float32
                        param.copy_((codes.float() * scales[:, None]).bfloat16())
                    else:
                        stored = archive.get_tensor(name)
                        assert torch.equal(stored.to(param.dtype), param.detach().cpu())
                    budget(start, device)
            assert counts == {"tied_head": 1, "attention": 96, "ffn": 72, "control": 121}
        assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()

        stage = "q6_arms"
        with torch.inference_mode():
            for arm, enabled in (("q6_donor", False), ("q6_e1280", True)):
                for row, item in zip(rows, items):
                    row["nats"][arm] = M17.score_doc(
                        model, item["document_ids"], wrappers, enabled, device, start)
                    budget(start, device)
                M44.set_experts(wrappers, enabled)
                for row, item in zip(prompts, items):
                    ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                          device=device)[None]
                    top = model(ids, use_cache=False).logits.argmax(-1)[0].cpu()
                    row["matching"][arm] = int((top == donor_top[item["source_id"]]).sum())
                write_progress(partial, arm, rows, prompts, start, device)
                print(json.dumps({"completed_arm": arm,
                                  "budget": budget(start, device)}), flush=True)

        stage = "summary"
        summary = summaries(rows, prompts)
        gates = {"pooled_bpb": summary["pooled"]["q6_minus_bf16_e1280_bpb"] <= .01,
                 "category_bpb": all(summary[c]["q6_minus_bf16_e1280_bpb"] <= .02
                                     for c in CATEGORIES),
                 "pooled_top1": summary["pooled"]["q6_minus_bf16_e1280_top1"] >= -.01,
                 "category_top1": all(summary[c]["q6_minus_bf16_e1280_top1"] >= -.02
                                      for c in CATEGORIES)}
        result = {"experiment": "METH-187-Q6-core-E1280-development",
                  "source_sha256": M57.MODEL_SHA,
                  "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
                  "child_checkpoint_sha256": M122.SPECIALIZED_SHA,
                  "manifest_sha256": M122.MANIFEST_SHA,
                  "prior_result_sha256": digest(PRIOR),
                  "core_sha256": report["sha256"],
                  "core_ideal_addressed_bytes_per_token":
                      report["ideal_addressed_bytes_per_token"],
                  "centered_parent_mean_max_abs_error": mean_error,
                  "document_rows": rows, "prompt_rows": prompts,
                  "summary": summary, "gates": gates,
                  "decision": "q6_core_development_pass_fresh_quality_pending" if all(gates.values())
                              else "q6_core_development_fail_train_correction_next",
                  "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)},
                  "scope": "Previously consumed METH-121 sources; development only"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < MAX_DISK
        print(json.dumps({"decision": result["decision"], "gates": gates,
                          "pooled": summary["pooled"], "runtime": result["runtime"]}),
              flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-187-failure",
                                       "stage": stage, "error": repr(error),
                                       "document_rows": rows, "prompt_rows": prompts,
                                       "runtime": {"seconds": time.monotonic() - start,
                                                   "rss_bytes": psutil.Process().memory_info().rss,
                                                   "gpu_peak_allocated_bytes":
                                                       torch.cuda.max_memory_allocated(device)
                                                       if device is not None else None}},
                                      indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
