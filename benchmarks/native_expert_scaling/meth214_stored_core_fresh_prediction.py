#!/usr/bin/env python3
"""Frozen three-arm prediction gate, before generation or task consumption."""
import argparse
import gc
import json
import math
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import meth212_stored_core_loader as L
import meth213_stored_core_fresh_manifest as F

P = L.P
ARMS = ("bf16_donor", "bf16_e1280", "stored_q8_exact_tied")


def bind(manifest, manifest_sha, anchors, anchors_sha):
    assert P.digest(F.LOADER) == F.LOADER_SHA
    assert P.digest(manifest) == manifest_sha and P.digest(anchors) == anchors_sha
    assert P.digest(L.CORE) == L.CORE_SHA and P.digest(L.EXPORT) == L.EXPORT_SHA
    data = json.loads(manifest.read_text(encoding="utf-8"))
    screen = json.loads(anchors.read_text(encoding="utf-8"))
    assert data["experiment"] == "METH-213-stored-core-fresh-manifest"
    assert data["core_sha256"] == L.CORE_SHA and data["loader_result_sha256"] == F.LOADER_SHA
    assert screen["manifest_sha256"] == manifest_sha and screen["answerable_count"] == 24
    items = data["items"]
    assert len(items) == 24 and data["selected_counts"] == F.M.COUNTS
    for item, row in zip(items, screen["rows"]):
        assert row["source_id"] == item["source_id"] and row["anchor"] in item["excerpt"]
        assert P.M17.sha(item["text"].encode()) == item["text_sha256"]
        for key in ("document_ids", "prompt_ids"):
            assert P.M17.sha(np.asarray(item[key], dtype=np.int32).tobytes()) == item[key+"_sha256"]
    assert P.digest(P.M122.SPECIALIZED) == P.M122.SPECIALIZED_SHA
    assert P.digest(P.M122.TRAINING) == P.M57.TRAINING_SHA
    parent = json.loads(P.M122.TRAINING.read_text(encoding="utf-8"))["checkpoints"]["512"]
    assert P.digest(parent["path"]) == P.M57.CHECKPOINT_SHA
    source = hf_hub_download(P.M42.MODEL, "model.safetensors", revision=P.M42.REV, local_files_only=True)
    assert P.digest(source) == P.M57.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(P.M42.MODEL, revision=P.M42.REV, local_files_only=True)
    assert P.M15.M13.C.tok_fingerprint(tokenizer) == P.M15.M13.TOK_FP
    return items, parent, tokenizer


def setup():
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    P.MAX_SECONDS = 70*60
    P.M17.MAX_SECONDS, P.M17.MAX_RSS_BYTES, P.M17.MAX_GPU_BYTES = P.MAX_SECONDS, P.MAX_RSS, P.MAX_GPU
    return device


def summarize(arms):
    result = {}
    for category in ("pooled", *P.CATEGORIES):
        values = {}
        for arm, cell in arms.items():
            docs = [r for r in cell["document_rows"] if category == "pooled" or r["category"] == category]
            prompts = [r for r in cell["prompt_rows"] if category == "pooled" or r["category"] == category]
            assert len(docs) == len(prompts) == (24 if category == "pooled" else 8)
            values[arm] = {"bpb": sum(r["nats"] for r in docs)/(math.log(2)*sum(r["bytes"] for r in docs)),
                          "top1": sum(r["matching"] for r in prompts)/sum(r["positions"] for r in prompts)}
        candidate = values[ARMS[2]]
        result[category] = {"arms": values,
            "candidate_minus_donor_bpb": candidate["bpb"]-values[ARMS[0]]["bpb"],
            "candidate_minus_bf16_e1280_bpb": candidate["bpb"]-values[ARMS[1]]["bpb"],
            "candidate_minus_bf16_e1280_top1": candidate["top1"]-values[ARMS[1]]["top1"]}
    return result


def bootstraps(arms):
    rng = np.random.default_rng(214214)
    draws = rng.integers(0,24,size=(10000,24))
    out = {}
    candidate = arms[ARMS[2]]["document_rows"]
    byte_count = np.asarray([r["bytes"] for r in candidate])
    for control in ARMS[:2]:
        previous = arms[control]["document_rows"]
        assert [r["source_id"] for r in candidate] == [r["source_id"] for r in previous]
        delta = np.asarray([r["nats"]-p["nats"] for r,p in zip(candidate,previous)])
        values = delta[draws].sum(1)/(math.log(2)*byte_count[draws].sum(1))
        out[control] = {"candidate_minus_control_bpb_p05": float(np.quantile(values,.05)),
                        "candidate_minus_control_bpb_p95": float(np.quantile(values,.95)),
                        "seed":214214, "draws":10000, "unit":"source", "decision_gate":False}
    return out


def main():
    ap = argparse.ArgumentParser()
    for key in ("manifest", "anchors", "out"):
        ap.add_argument("--"+key, required=True, type=Path)
    ap.add_argument("--manifest-sha", required=True)
    ap.add_argument("--anchors-sha", required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start, stage, arms = time.monotonic(), "bindings", {}
    def partial():
        args.out.with_suffix(".partial.json").write_text(json.dumps({"stage": stage,"arms":arms,
            "elapsed_seconds":time.monotonic()-start},indent=2)+"\n",encoding="utf-8")
    try:
        items, parent, tokenizer = bind(args.manifest,args.manifest_sha,args.anchors,args.anchors_sha)
        device = setup()
        stage = "bf16_load"
        model = AutoModelForCausalLM.from_pretrained(P.M42.MODEL, revision=P.M42.REV,
            dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device).eval()
        model.config.use_cache = False
        L.D.H.load_centered(model,device,parent["path"])
        wrappers = [layer.mlp for layer in model.model.layers]
        donor_top = {}
        with torch.inference_mode():
            for arm, enabled in ((ARMS[0],False),(ARMS[1],True)):
                stage = arm
                docs, prompts = [], []
                for item in items:
                    nats = P.M17.score_doc(model,item["document_ids"],wrappers,enabled,device,start)
                    docs.append({"source_id":item["source_id"],"category":item["category"],"bytes":item["bytes"],"nats":nats})
                P.M44.set_experts(wrappers,enabled)
                for item in items:
                    ids = torch.as_tensor(item["prompt_ids"],device=device)[None]
                    top = model(ids,use_cache=False).logits.argmax(-1)[0].cpu()
                    if not enabled:
                        donor_top[item["source_id"]] = top
                    prompts.append({"source_id":item["source_id"],"category":item["category"],
                        "positions":len(item["prompt_ids"]),"matching":int((top==donor_top[item["source_id"]]).sum())})
                    P.budget(start,device)
                arms[arm] = {"document_rows":docs,"prompt_rows":prompts}
                partial()
                print(json.dumps({"arm":arm,"runtime":P.budget(start,device)}),flush=True)
        del model, wrappers
        gc.collect()
        torch.cuda.empty_cache()
        stage = "stored_load"
        model, wrappers, proposal, load_record = L.load_stored(device)
        L.D.E.P = P
        stage = ARMS[2]
        arms[ARMS[2]] = L.D.E.evaluate(model,wrappers,items,donor_top,ARMS[2],device,start)
        partial()
        stage = "prompt_shortlist"
        approx = (proposal["codes"].float()*proposal["scales"].float()[:,None]).bfloat16()
        L.D.H.KS = (64,)
        shortlist = []
        with torch.inference_mode():
            for item in items:
                ids = torch.as_tensor(item["prompt_ids"],device=device)[None]
                hidden = model.model(ids,use_cache=False).last_hidden_state[0]
                shortlist.append({"source_id":item["source_id"],"category":item["category"],
                                  **L.D.H.score_item(hidden,model.lm_head.weight,approx)})
                P.budget(start,device)
        summary = summarize(arms)
        gates = {"pooled_bpb_vs_both": all(summary["pooled"][k]<=.01 for k in
                    ("candidate_minus_donor_bpb","candidate_minus_bf16_e1280_bpb")),
                 "category_bpb_vs_both": all(summary[c][k]<=.02 for c in P.CATEGORIES for k in
                    ("candidate_minus_donor_bpb","candidate_minus_bf16_e1280_bpb")),
                 "pooled_top1": summary["pooled"]["candidate_minus_bf16_e1280_top1"]>=-.01,
                 "category_top1": all(summary[c]["candidate_minus_bf16_e1280_top1"]>=-.02 for c in P.CATEGORIES),
                 "prompt_k64_inclusion": all(r["misses"]["64"]==0 for r in shortlist),
                 "prompt_exact_rerank": all(r["rerank_mismatches"]["64"]==0 for r in shortlist)}
        result = {"experiment":"METH-214-stored-core-fresh-prediction", "manifest_sha256":args.manifest_sha,
            "anchors_sha256":args.anchors_sha,"core_sha256":L.CORE_SHA,"load_record":load_record,
            "donor_sha256":P.M57.MODEL_SHA,"parent_checkpoint_sha256":P.M57.CHECKPOINT_SHA,
            "child_checkpoint_sha256":P.M122.SPECIALIZED_SHA,"arms":arms,"summary":summary,
            "bootstrap":bootstraps(arms),"shortlist":shortlist,"gates":gates,
            "decision":"fresh_prediction_pass_generation_next" if all(gates.values()) else "fresh_prediction_fail_stop_generation",
            "runtime":{**P.budget(start,device),"gpu":torch.cuda.get_device_name(device)},
            "scope":"New-source full-head prediction; finite prompt shortlist; no generation/task/blind/native claim"}
        args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"decision":result["decision"],"gates":gates,"summary":summary,"runtime":result["runtime"]}),flush=True)
    except BaseException as error:
        partial()
        args.out.with_suffix(".failure.json").write_text(json.dumps({"stage":stage,"error":repr(error),
            "elapsed_seconds":time.monotonic()-start},indent=2)+"\n",encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
