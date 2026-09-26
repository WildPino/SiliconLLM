#!/usr/bin/env python3
"""METH-41: independent full-model quality audit of the stored C96 route."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
from safetensors import safe_open
from safetensors.torch import load_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth24_export_r8_core as M24
import meth25_fresh_r8_artifact_audit as M25
import meth36_int8_sketch_audit as M36
import meth37_route_replacement as M37
import meth41_fresh_c96_manifest as M41M


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth41_fresh_c96_manifest.json"
MANIFEST_SHA = "8fff79f2502612d1f49083d1c48dd36d2718acec6335d676ccb5b8ded726365f"
M37_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth37_route_replacement_result.json"
M37_RESULT_SHA = "d3d69a24c767987b38b57bf6dc240194259174ad5476ab59f757661d12b4b5d5"
ARMS = ("original_donor", "r8_exact", "r8_int8_route")
CANDIDATES = 96
BOOTSTRAP_SEED = 414141
BOOTSTRAP_DRAWS = 20000
MAX_SECONDS = 20 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or peak > MAX_GPU_BYTES or rss > MAX_RSS_BYTES:
        raise RuntimeError(f"METH-41 budget: {elapsed:.1f}s, GPU {peak}, RSS {rss}")
    return {"elapsed_seconds": elapsed, "gpu_peak_allocated_bytes": peak,
            "rss_end_bytes": rss}


def bpb(rows, arm):
    return sum(r["nats"][arm] for r in rows) / (
        math.log(2) * sum(r["bytes"] for r in rows))


def summarize_docs(rows):
    groups = {"pooled": rows}
    groups.update({cat: [r for r in rows if r["category"] == cat] for cat in
                   ("code", "prose", "technical_general")})
    summary = {}
    for category, group in groups.items():
        scores = {arm: bpb(group, arm) for arm in ARMS}
        summary[category] = {"docs": len(group),
                             "bytes": sum(r["bytes"] for r in group),
                             "bpb": scores,
                             "c96_minus_exact": scores["r8_int8_route"] - scores["r8_exact"],
                             "c96_minus_original_donor":
                             scores["r8_int8_route"] - scores["original_donor"]}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = []
        for cat in ("code", "prose", "technical_general"):
            group = groups[cat]
            sampled.extend(group[i] for i in rng.integers(0, len(group), size=len(group)))
        draws.append(bpb(sampled, "r8_int8_route") - bpb(sampled, "r8_exact"))
    upper = float(np.quantile(draws, 0.95))
    loss_pass = (summary["pooled"]["c96_minus_exact"] <= 0.001
                 and all(summary[c]["c96_minus_exact"] <= 0.002 for c in
                         ("code", "prose", "technical_general"))
                 and upper <= 0.002)
    donor_pass = (summary["pooled"]["c96_minus_original_donor"] <= 0.01
                  and summary["code"]["c96_minus_original_donor"] <= 0.01
                  and all(summary[c]["c96_minus_original_donor"] <= 0.03
                          for c in ("prose", "technical_general")))
    return summary, {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
                     "penalty_ci95_upper": upper}, loss_pass, donor_pass


def set_arm(wrappers, arm):
    M37.set_arm(wrappers, arm)
    for wrapper in wrappers:
        wrapper.candidate_count = CANDIDATES
        wrapper.rescore_mode = "sum"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert M17.sha(MANIFEST.read_bytes()) == MANIFEST_SHA
    assert M17.sha(M37_RESULT.read_bytes()) == M37_RESULT_SHA
    assert M15.M13.sha256(M25.CORE) == M25.CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    assert M15.M13.sha256(M36.SKETCH_ARTIFACT) == M36.SKETCH_SHA
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                             revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M15.M13.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    manifest, items = M41M.select(tokenizer)
    assert json.loads(MANIFEST.read_text(encoding="utf-8")) == manifest
    prior = json.loads(M37_RESULT.read_text(encoding="utf-8"))
    old_manifest, old_items = M37.select(tokenizer)
    assert prior["manifest_sha256"] == M17.sha(
        (M41M.DOCS / "meth37_route_quality_manifest.json").read_bytes())
    assert len(old_manifest["selected"]) == len(old_items) == 24
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    original_params = dict(model.named_parameters())
    assert len(original_params) == 290
    adapter = load_file(str(M20.ADAPTER), device="cpu")
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M37.ShortlistExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(adapter[f"layers.{li}.{key}"].to(device))
        wrappers.append(wrapper)
    assert len(adapter) == 3 * M15.M13.L
    del adapter
    with safe_open(str(M36.SKETCH_ARTIFACT), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_E128_ROUTER_R64_INT8_V1"
        assert meta["adapter_sha256"] == M20.ADAPTER_SHA
        assert meta["core_sha256"] == M25.CORE_SHA
        assert len(archive.keys()) == 3 * M15.M13.L
        for li, wrapper in enumerate(wrappers):
            basis = archive.get_tensor(f"layers.{li}.basis")
            codes = archive.get_tensor(f"layers.{li}.codes")
            scales = archive.get_tensor(f"layers.{li}.scales")
            assert basis.shape == (M15.M13.D, 64) and basis.dtype == torch.float32
            assert codes.shape == (M15.E, 64) and codes.dtype == torch.int8
            assert scales.shape == (M15.E,) and scales.dtype == torch.float32
            wrapper.basis.copy_(basis.to(device))
            wrapper.sketch.copy_((codes.float() * scales[:, None]).to(device))
    check_budget(start, device)
    rows = [{"category": x["category"], "source_id": x["source_id"],
             "text_sha256": x["text_sha256"], "bytes": x["bytes"],
             "tokens": x["tokens"], "nats": {}, "top1": {}, "generation": {}}
            for x in items]
    set_arm(wrappers, "original_donor")
    for item, row in zip(items, rows):
        row["nats"]["original_donor"] = M17.score_doc(
            model, item["ids"], wrappers, False, device, start)
        check_budget(start, device)
    print("BF16 donor documents scored", flush=True)
    with safe_open(str(M25.CORE), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_R8_CORE_V1"
        assert meta["model_sha256"] == M15.M13.MODEL_SHA
        assert meta["adapter_sha256"] == M20.ADAPTER_SHA
        assert meta["tied_head"] == "true"
        assert len(archive.keys()) == 459
        matrices = controls = 0
        for name, param in original_params.items():
            if M24.classify(name, tuple(param.shape)) == "control":
                assert torch.equal(archive.get_tensor(name).to(torch.bfloat16),
                                   param.detach().cpu())
                controls += 1
            else:
                codes = archive.get_tensor(name + ".q")
                scales = archive.get_tensor(name + ".scale")
                assert codes.dtype == torch.int8 and tuple(codes.shape) == tuple(param.shape)
                assert scales.dtype == torch.float32
                assert tuple(scales.shape) == (param.shape[0],)
                param.copy_((codes.to(device).float() *
                             scales.to(device).unsqueeze(1)).to(torch.bfloat16))
                matrices += 1
        assert matrices == 169 and controls == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    set_arm(wrappers, "r8_exact")
    for item, saved in zip(old_items, prior["rows"]):
        assert item["source_id"] == saved["source_id"]
        assert M37.score_top1(model, item["prompt_ids"], device) == saved["top1"]["r8_exact"]
        check_budget(start, device)
    print("24 METH-37 exact-route top1 controls reproduced", flush=True)
    for arm in ("r8_exact", "r8_int8_route"):
        set_arm(wrappers, arm)
        for i, (item, row) in enumerate(zip(items, rows)):
            row["nats"][arm] = M17.score_doc(
                model, item["ids"], wrappers, True, device, start)
            check_budget(start, device)
            if (i + 1) % 8 == 0:
                print(f"{arm} documents {i+1}/{len(items)}", flush=True)
        for item, row in zip(items, rows):
            row["top1"][arm] = M37.score_top1(model, item["prompt_ids"], device)
            check_budget(start, device)
    matching = sum(sum(a == b for a, b in zip(row["top1"]["r8_exact"],
                                            row["top1"]["r8_int8_route"]))
                   for row in rows)
    top1 = {"positions": len(rows) * 256, "matching": matching,
            "agreement_fraction": matching / (len(rows) * 256)}
    print(f"C96 route-replaced top1 {matching}/{len(rows)*256}", flush=True)
    for arm in ("r8_exact", "r8_int8_route"):
        set_arm(wrappers, arm)
        with torch.inference_mode():
            for i, (item, row) in enumerate(zip(items, rows)):
                inp = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                      device=device).unsqueeze(0)
                output = model.generate(input_ids=inp, max_new_tokens=128,
                                        do_sample=False,
                                        pad_token_id=model.config.eos_token_id,
                                        use_cache=True)
                continuation = output[0, 256:].cpu().tolist()
                row["generation"][arm] = {
                    "continuation_ids": continuation,
                    "continuation_text": tokenizer.decode(
                        continuation, skip_special_tokens=False),
                    "repeated_8gram_3x": M20.repeated_8gram(continuation),
                    "distinct2": M20.distinct2(continuation)}
                check_budget(start, device)
                if (i + 1) % 8 == 0:
                    print(f"{arm} generation {i+1}/{len(items)}", flush=True)
    docs, bootstrap, loss_pass, donor_pass = summarize_docs(rows)
    generations, generation_pass = M37.summarize_generation(
        rows, model.config.eos_token_id)
    route_pass = loss_pass and top1["agreement_fraction"] >= 0.99 and generation_pass
    runtime = check_budget(start, device)
    result = {"experiment": "METH-41", "manifest_sha256": MANIFEST_SHA,
              "source_sha256": M15.M13.MODEL_SHA,
              "core_sha256": M25.CORE_SHA, "adapter_sha256": M20.ADAPTER_SHA,
              "sketch_sha256": M36.SKETCH_SHA,
              "prior_control_sha256": M37_RESULT_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "candidate_count": CANDIDATES, "rescore_mode": "sum",
              "rows": rows, "document_summary": docs, "bootstrap": bootstrap,
              "top1": top1, "generation_summary": generations,
              "gates": {"paired_loss": loss_pass,
                        "paired_top1": top1["agreement_fraction"] >= 0.99,
                        "paired_generation": generation_pass,
                        "paired_route_replacement": route_pass,
                        "donor_relative_documents": donor_pass},
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"document_summary": docs, "bootstrap": bootstrap,
                      "top1": top1, "generation_summary": generations,
                      "gates": result["gates"], "runtime": result["runtime"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
