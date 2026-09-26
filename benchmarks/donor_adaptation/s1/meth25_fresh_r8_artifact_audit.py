#!/usr/bin/env python3
"""METH-25: independent quality audit of the stored R8 core plus E128 adapter."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth23_int8_core_composition_diagnostic as M23
import meth24_export_r8_core as M24
import meth25_fresh_r8_manifest as M25


ROOT = M25.ROOT
CORE = ROOT / "results/native_expert_scaling/meth24_qwen05b_r8_core.safetensors"
CORE_SHA = "c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27"
MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth25_fresh_r8_manifest.json"
MANIFEST_SHA = "20890f2c4614287dfdb7527b0935436bb88a035a348e45e0c1e4f13f6d068f77"
ARMS = ("original_donor", "r8_donor", "r8_adapter", "r8_permuted_adapter")
BOOTSTRAP_SEED = 252526
BOOTSTRAP_DRAWS = 20000
PERMUTATION_SEED = 252525
PROMPT_SEED = "252527"


def summarize(rows):
    groups = {"pooled": rows}
    groups.update({c: [r for r in rows if r["category"] == c]
                   for c in M25.COUNTS})

    def bpb(group, arm):
        return sum(r["nats"][arm] for r in group) / (
            math.log(2) * sum(r["bytes"] for r in group))

    summary = {}
    for name, group in groups.items():
        donor = bpb(group, "original_donor")
        summary[name] = {"docs": len(group), "bytes": sum(r["bytes"] for r in group),
                         "tokens": sum(r["tokens"] for r in group),
                         "bpb": {arm: bpb(group, arm) for arm in ARMS},
                         "delta_to_original_donor": {
                             arm: bpb(group, arm)-donor for arm in ARMS},
                         "route_utility_bpb": bpb(group, "r8_permuted_adapter") -
                                              bpb(group, "r8_adapter")}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = []
        for category in M25.COUNTS:
            group = groups[category]
            sampled += [group[i] for i in rng.integers(0, len(group), size=len(group))]
        draws.append(bpb(sampled, "r8_adapter") - bpb(sampled, "original_donor"))
    summary["pooled"]["adapter_delta_ci95_upper"] = float(np.quantile(draws, 0.95))
    summary["pooled"]["bootstrap_seed"] = BOOTSTRAP_SEED
    summary["pooled"]["bootstrap_draws"] = BOOTSTRAP_DRAWS
    document_pass = (
        summary["pooled"]["delta_to_original_donor"]["r8_donor"] <= 0.01
        and summary["pooled"]["delta_to_original_donor"]["r8_adapter"] <= 0.01
        and summary["pooled"]["adapter_delta_ci95_upper"] <= 0.02
        and summary["code"]["delta_to_original_donor"]["r8_adapter"] <= 0.01
        and summary["prose"]["delta_to_original_donor"]["r8_adapter"] <= 0.03
        and summary["technical_general"]["delta_to_original_donor"]["r8_adapter"] <= 0.03
        and summary["pooled"]["route_utility_bpb"] >= 0.002)
    return summary, document_pass


def top1(model, wrappers, enabled, ids, device):
    for w in wrappers:
        w.enabled = enabled
    with torch.inference_mode():
        return model(torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0),
                     use_cache=False).logits.argmax(dim=-1)[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert M17.sha(MANIFEST.read_bytes()) == MANIFEST_SHA
    manifest, items = M25.select()
    assert json.loads(MANIFEST.read_text(encoding="utf-8")) == manifest
    assert M15.M13.sha256(CORE) == CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    start = time.monotonic()
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                             revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M15.M13.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    original_params = dict(model.named_parameters())
    assert len(original_params) == 290
    wrappers = []
    adapter = load_file(str(M20.ADAPTER), device="cpu")
    with safe_open(str(M20.ADAPTER), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
    assert meta["model_sha256"] == M15.M13.MODEL_SHA
    assert meta["output_factor"] == "0.5"
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(adapter[f"layers.{li}.{key}"].to(device))
        wrappers.append(wrapper)
    assert len(adapter) == 3 * M15.M13.L
    del adapter
    perm_rng = np.random.default_rng(PERMUTATION_SEED)
    original_routers = [w.router.detach().clone() for w in wrappers]
    permuted_routers = [r[torch.as_tensor(perm_rng.permutation(M15.E),
                                         dtype=torch.long, device=device)]
                        for r in original_routers]

    prompts = []
    for category in M25.COUNTS:
        pool = [item for item in items if item["category"] == category]
        pool.sort(key=lambda item: M17.sha((PROMPT_SEED + "|" + item["source_id"]).encode()))
        for item in pool[:8]:
            ids = tokenizer.encode(item["text"], add_special_tokens=False)
            assert len(ids) >= 256
            prompts.append({"category": category, "source_id": item["source_id"],
                            "prompt_ids_sha256": M17.sha(np.asarray(ids[:256], dtype=np.int32).tobytes()),
                            "ids": ids[:256], "top1": {}})
    assert len(prompts) == 24
    rows = []
    for i, item in enumerate(items):
        ids = tokenizer.encode(item["text"], add_special_tokens=False)
        rows.append({"index": i, "category": item["category"],
                     "source_id": item["source_id"],
                     "text_sha256": M17.sha(item["text"].encode()),
                     "bytes": len(item["text"].encode()), "tokens": len(ids),
                     "ids": ids, "nats": {}})
    for row in rows:
        row["nats"]["original_donor"] = M17.score_doc(
            model, row["ids"], wrappers, False, device, start)
        M24.check_budget(start, device)
    for prompt in prompts:
        prompt["top1"]["original_donor"] = top1(
            model, wrappers, False, prompt["ids"], device)
        prompt["top1"]["original_adapter"] = top1(
            model, wrappers, True, prompt["ids"], device)
    print("original donor document and top1 controls scored", flush=True)

    with safe_open(str(CORE), framework="pt", device="cpu") as archive:
        core_meta = archive.metadata()
        assert core_meta["model_sha256"] == M15.M13.MODEL_SHA
        assert core_meta["adapter_sha256"] == M20.ADAPTER_SHA
        assert core_meta["format"] == "QWEN25_R8_CORE_V1"
        assert core_meta["tied_head"] == "true"
        assert len(archive.keys()) == 459
        matrix_count = 0
        control_count = 0
        for name, param in original_params.items():
            organ = M24.classify(name, tuple(param.shape))
            if organ == "control":
                stored = archive.get_tensor(name)
                assert stored.dtype == torch.float32
                assert torch.equal(stored.to(torch.bfloat16), param.detach().cpu())
                control_count += 1
            else:
                codes = archive.get_tensor(name + ".q")
                scale = archive.get_tensor(name + ".scale")
                assert codes.dtype == torch.int8 and tuple(codes.shape) == tuple(param.shape)
                assert scale.dtype == torch.float32 and tuple(scale.shape) == (param.shape[0],)
                assert bool((codes >= -127).all()) and bool((codes <= 127).all())
                reconstructed = (codes.to(device).float() *
                                 scale.to(device).unsqueeze(1)).to(torch.bfloat16)
                param.copy_(reconstructed)
                matrix_count += 1
        assert matrix_count == 169 and control_count == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    M24.check_budget(start, device)
    print("stored R8 core reconstructed: 169 matrices, 121 controls", flush=True)

    for i, row in enumerate(rows):
        for arm in ARMS[1:]:
            if arm == "r8_permuted_adapter":
                for w, r in zip(wrappers, permuted_routers):
                    w.router.copy_(r)
            else:
                for w, r in zip(wrappers, original_routers):
                    w.router.copy_(r)
            row["nats"][arm] = M17.score_doc(
                model, row["ids"], wrappers, arm != "r8_donor", device, start)
        M24.check_budget(start, device)
        if (i+1) % 8 == 0:
            print(f"stored R8 scored {i+1}/{len(rows)} documents", flush=True)
    for w, r in zip(wrappers, original_routers):
        w.router.copy_(r)
    for prompt in prompts:
        prompt["top1"]["r8_donor"] = top1(
            model, wrappers, False, prompt["ids"], device)
        prompt["top1"]["r8_adapter"] = top1(
            model, wrappers, True, prompt["ids"], device)
    agreement = {}
    for arm, ref in (("r8_donor", "original_donor"),
                     ("r8_adapter", "original_adapter")):
        same = sum(sum(a == b for a, b in zip(p["top1"][arm], p["top1"][ref]))
                   for p in prompts)
        agreement[arm] = {"positions": 6144, "matching": same,
                          "fraction": same / 6144}
    ranking_pass = all(x["fraction"] >= 0.95 for x in agreement.values())
    summary, document_pass = summarize(rows)
    runtime = M24.check_budget(start, device)
    for row in rows:
        del row["ids"]
    for prompt in prompts:
        del prompt["ids"]
    result = {"experiment": "METH-25", "manifest_sha256": MANIFEST_SHA,
              "core_sha256": CORE_SHA, "adapter_sha256": M20.ADAPTER_SHA,
              "source_sha256": M15.M13.MODEL_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "arms": ARMS, "rows": rows, "summary": summary,
              "document_gate_pass": document_pass,
              "prompt_seed": PROMPT_SEED, "prompt_rows": prompts,
              "top1_agreement": agreement, "ranking_gate_pass": ranking_pass,
              "joint_quality_screen_pass": document_pass and ranking_pass,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "top1_agreement": agreement,
                      "joint_quality_screen_pass": result["joint_quality_screen_pass"],
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
