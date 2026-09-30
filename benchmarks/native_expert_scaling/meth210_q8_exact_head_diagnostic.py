#!/usr/bin/env python3
"""Separate Q8-core embedding/head loss and screen a priced exact-head proposal."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from huggingface_hub import hf_hub_download

import meth194_q8_core_e1280_development as P
import meth188_core_organ_attribution as E

sys.path.insert(0, str(P.ROOT / "benchmarks/donor_adaptation/s1"))
import meth128_exact_head_rerank as H

PRIOR = P.DOC / "meth194_q8_core_e1280_development_result.json"
PRIOR_SHA = "9063af7040e063070a3f590e5be19001ffb48f8b007a2375db8dd17aa656ec1b"
CORE_SHA = "ab38c6bf591b59b993a273b112f0e3bdc889dbde35d576f62933f2843c4aa396"
ARMS = ("bf16_e1280", "q8_r8_tied", "q8_exact_embed_r8_head",
        "q8_r8_embed_exact_head", "q8_exact_tied")
K = 64


def summarize(arms):
    summary = {}
    for category in ("pooled", *P.CATEGORIES):
        values = {}
        for arm, cell in arms.items():
            docs = [r for r in cell["document_rows"] if category == "pooled" or r["category"] == category]
            prompts = [r for r in cell["prompt_rows"] if category == "pooled" or r["category"] == category]
            assert len(docs) == len(prompts) == (24 if category == "pooled" else 8)
            values[arm] = {"bpb": sum(r["nats"] for r in docs) /
                (math.log(2) * sum(r["bytes"] for r in docs)),
                "matching": sum(r["matching"] for r in prompts),
                "positions": sum(r["positions"] for r in prompts)}
            values[arm]["top1"] = values[arm]["matching"] / values[arm]["positions"]
        ref = values["bf16_e1280"]
        for row in values.values():
            row["bpb_minus_bf16_e1280"] = row["bpb"] - ref["bpb"]
            row["top1_minus_bf16_e1280"] = row["top1"] - ref["top1"]
        summary[category] = values
    return summary


def reconcile(cell, old, name):
    for row, previous in zip(cell["document_rows"], old["document_rows"]):
        assert row["source_id"] == previous["source_id"]
        assert row["nats"] == previous["nats"][name]
    for row, previous in zip(cell["prompt_rows"], old["prompt_rows"]):
        assert row["source_id"] == previous["source_id"]
        assert row["matching"] == previous["matching"][name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage, device = "bindings", None
    arms = {}
    try:
        assert P.digest(PRIOR) == PRIOR_SHA
        assert P.digest(P.CORE) == CORE_SHA
        assert P.digest(P.M122.MANIFEST) == P.M122.MANIFEST_SHA
        assert P.digest(P.M122.SPECIALIZED) == P.M122.SPECIALIZED_SHA == H.CHILD_SHA
        assert P.digest(P.M122.TRAINING) == P.M57.TRAINING_SHA
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        report = json.loads(P.EXPORT.read_text(encoding="utf-8"))
        assert report["sha256"] == CORE_SHA and report["ideal_addressed_bytes_per_token"] == 559981568
        manifest = json.loads(P.M122.MANIFEST.read_text(encoding="utf-8"))
        items = manifest["items"]
        assert len(items) == 24
        for item in items:
            assert P.M17.sha(item["text"].encode()) == item["text_sha256"]
            for field in ("document_ids", "prompt_ids"):
                assert P.M17.sha(np.asarray(item[field], dtype=np.int32).tobytes()) == item[field + "_sha256"]
        parent = json.loads(P.M122.TRAINING.read_text(encoding="utf-8"))["checkpoints"]["512"]
        assert P.digest(parent["path"]) == P.M57.CHECKPOINT_SHA
        source = Path(hf_hub_download(P.M42.MODEL, "model.safetensors", revision=P.M42.REV, local_files_only=True))
        assert P.digest(source) == P.M57.MODEL_SHA
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
        assert len(matches) == 1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        P.M17.MAX_SECONDS = P.MAX_SECONDS
        P.M17.MAX_RSS_BYTES = P.MAX_RSS
        P.M17.MAX_GPU_BYTES = P.MAX_GPU
        E.P = P
        stage = "load_centered_model"
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(P.M42.MODEL, revision=P.M42.REV,
            dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device).eval()
        model.config.use_cache = False
        params = dict(model.named_parameters())
        assert len(params) == 290
        H.load_centered(model, device, parent["path"])
        wrappers = [layer.mlp for layer in model.model.layers]
        donor_top = {}
        P.M44.set_experts(wrappers, False)
        with torch.inference_mode():
            for item in items:
                ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
                donor_top[item["source_id"]] = model(ids, use_cache=False).logits.argmax(-1)[0].cpu()
        stage = "baseline"
        arms[ARMS[0]] = E.evaluate(model, wrappers, items, donor_top, ARMS[0], device, started)
        reconcile(arms[ARMS[0]], prior, "bf16_e1280")
        tied = model.model.embed_tokens.weight
        assert tied.data_ptr() == model.lm_head.weight.data_ptr()
        exact = nn.Parameter(tied.detach().clone(), requires_grad=False)
        stage = "load_stored_q8_core"
        with safe_open(str(P.CORE), framework="pt", device="cpu") as archive:
            assert archive.metadata()["format"] == "QWEN25_INSTRUCT_R8H_GROUP64_Q8FFN_BF16ATTN_V1"
            changed = 0
            with torch.no_grad():
                for name, param in params.items():
                    organ = P.M24.classify(name, tuple(param.shape))
                    if organ == "ffn":
                        param.copy_(P.M193.reconstruct(archive.get_tensor(name + ".q8").to(device),
                                                      archive.get_tensor(name + ".scale").to(device)))
                        changed += 1
                    elif organ != "tied_head":
                        assert torch.equal(archive.get_tensor(name).to(param.dtype), param.detach().cpu())
                assert changed == 72
                codes = archive.get_tensor("model.embed_tokens.weight.q").to(device)
                scales = archive.get_tensor("model.embed_tokens.weight.scale").to(device)
                assert codes.shape == exact.shape == (151936,896)
                assert codes.dtype == torch.int8 and scales.dtype == torch.float32
                tied.copy_((codes.float() * scales[:, None]).bfloat16())
        del params
        for arm, embed, head in ((ARMS[1], tied, tied), (ARMS[2], exact, tied),
                                (ARMS[3], tied, exact), (ARMS[4], exact, exact)):
            stage = arm
            model.model.embed_tokens.weight = embed
            model.lm_head.weight = head
            assert (embed.data_ptr() == head.data_ptr()) == (arm in (ARMS[1], ARMS[4]))
            arms[arm] = E.evaluate(model, wrappers, items, donor_top, arm, device, started)
            if arm == ARMS[1]:
                reconcile(arms[arm], prior, "q8_e1280")
            print(json.dumps({"completed_arm": arm, "budget": P.budget(started, device)}), flush=True)
        stage = "fp16_scale_shortlist_screen"
        proposal_scales = scales.half()
        assert bool(torch.isfinite(proposal_scales).all() and (proposal_scales > 0).all())
        proposal = (codes.float() * proposal_scales.float()[:, None]).bfloat16()
        H.KS = (K,)
        shortlist = []
        with torch.inference_mode():
            for item in items:
                ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
                hidden = model.model(ids, use_cache=False).last_hidden_state[0]
                row = H.score_item(hidden, exact, proposal)
                shortlist.append({"source_id": item["source_id"], "category": item["category"], **row})
                P.budget(started, device)
        summary = summarize(arms)
        candidate = ARMS[4]
        gates = {"pooled_bpb": summary["pooled"][candidate]["bpb_minus_bf16_e1280"] <= .01,
            "category_bpb": all(summary[c][candidate]["bpb_minus_bf16_e1280"] <= .02 for c in P.CATEGORIES),
            "pooled_top1": summary["pooled"][candidate]["top1_minus_bf16_e1280"] >= -.01,
            "category_top1": all(summary[c][candidate]["top1_minus_bf16_e1280"] >= -.02 for c in P.CATEGORIES),
            "shortlist_inclusion": all(row["misses"][str(K)] == 0 for row in shortlist),
            "bf16_row_rerank": all(row["rerank_mismatches"][str(K)] == 0 for row in shortlist)}
        ledger = {"m193_ideal_addressed_bytes": 559981568,
            "proposal_scale_bytes_saved": 151936 * 2,
            "exact_selected_head_row_bytes": K * 896 * 2,
            "exact_embedding_lookup_bytes_added_conservatively": 896 * 2,
            "additional_exact_head_stored_bytes": 151936 * 896 * 2}
        ideal = ledger["m193_ideal_addressed_bytes"] - ledger["proposal_scale_bytes_saved"] + ledger["exact_selected_head_row_bytes"] + ledger["exact_embedding_lookup_bytes_added_conservatively"]
        ledger["proposed_two_pass_ideal_addressed_bytes"] = ideal
        ledger["margin_to_560_MB"] = 560000000 - ideal
        gates["proposed_active_payload"] = ideal <= 560000000
        result = {"experiment": "METH-210-Q8-embedding-head-diagnostic",
            "source_sha256": P.M57.MODEL_SHA, "core_sha256": CORE_SHA,
            "meth194_result_sha256": PRIOR_SHA, "manifest_sha256": P.M122.MANIFEST_SHA,
            "parent_checkpoint_sha256": P.M57.CHECKPOINT_SHA,
            "child_checkpoint_sha256": P.M122.SPECIALIZED_SHA,
            "arms": arms, "summary": summary, "shortlist_k": K, "shortlist": shortlist,
            "proposal_codes_sha256": P.M17.sha(codes.cpu().numpy().tobytes()),
            "proposal_fp16_scales_sha256": P.M17.sha(proposal_scales.view(torch.uint16).cpu().numpy().tobytes()),
            "ledger": ledger, "gates": gates,
            "decision": "exact_tied_head_candidate_ready_for_stored_export" if all(gates.values()) else "exact_head_candidate_development_fail",
            "runtime": {**P.budget(started, device), "gpu": torch.cuda.get_device_name(device)},
            "scope": "Viewed sources; offline full-head probabilities and finite-state shortlist only; no new stored artifact or native rate"}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size < P.MAX_DISK
        print(json.dumps({"decision": result["decision"], "gates": gates,
                          "pooled": summary["pooled"], "ledger": ledger}), flush=True)
    except BaseException as error:
        args.out.with_suffix(".failure.json").write_text(json.dumps({"experiment": "METH-210-failure",
            "stage": stage, "error": repr(error), "completed_arms": arms,
            "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
