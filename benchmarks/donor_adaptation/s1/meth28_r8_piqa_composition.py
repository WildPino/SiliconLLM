#!/usr/bin/env python3
"""METH-28: full PIQA task audit of the stored R8 core and E128 adapter."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth21_half_adapter_piqa as M21
import meth24_export_r8_core as M24
import meth25_fresh_r8_artifact_audit as M25


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth21_half_adapter_piqa_manifest.json"
MANIFEST_SHA = "58acfd5882ae5d7e155bdd0ae826728aba510dc6ef4f599b7b807161a498531d"
ARMS = ("original_donor", "r8_donor", "r8_adapter")
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 282828


def arm_result(model, item, device):
    scores = [M21.score_option(model, item["prefix"], suffix, device)
              for suffix in item["suffixes"]]
    return {"choice_mean": 0 if scores[0][1] <= scores[1][1] else 1,
            "choice_total": 0 if scores[0][0] <= scores[1][0] else 1,
            "options": [{"total_nll": total, "mean_nll": mean,
                         "tokens": len(suffix)}
                        for (total, mean), suffix in zip(scores, item["suffixes"])]}


def summarize(rows):
    out = {"items": len(rows)}
    for arm in ARMS:
        primary = [r[arm]["choice_mean"] == r["label"] for r in rows]
        secondary = [r[arm]["choice_total"] == r["label"] for r in rows]
        out[arm] = {"correct_mean": sum(primary),
                    "accuracy_mean": sum(primary)/len(rows),
                    "correct_total": sum(secondary),
                    "accuracy_total": sum(secondary)/len(rows),
                    "gold_mean_nll": sum(r[arm]["options"][r["label"]]["mean_nll"]
                                         for r in rows)/len(rows)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for arm in ARMS[1:]:
        paired = np.asarray([
            int(r[arm]["choice_mean"] == r["label"]) -
            int(r["original_donor"]["choice_mean"] == r["label"])
            for r in rows], dtype=np.int8)
        draws = [float(paired[rng.integers(0, len(rows), len(rows))].mean())
                 for _ in range(BOOTSTRAP_DRAWS)]
        out[arm]["delta_to_original_donor"] = (
            out[arm]["accuracy_mean"] - out["original_donor"]["accuracy_mean"])
        out[arm]["paired_bootstrap_lower95"] = float(np.quantile(draws, 0.05))
        out[arm]["discordance"] = {
            "original_correct_arm_wrong": sum(
                r["original_donor"]["choice_mean"] == r["label"] and
                r[arm]["choice_mean"] != r["label"] for r in rows),
            "original_wrong_arm_correct": sum(
                r["original_donor"]["choice_mean"] != r["label"] and
                r[arm]["choice_mean"] == r["label"] for r in rows)}
    out["r8_adapter"]["delta_to_r8_donor"] = (
        out["r8_adapter"]["accuracy_mean"] - out["r8_donor"]["accuracy_mean"])
    out["bootstrap_seed"] = BOOTSTRAP_SEED
    out["bootstrap_draws_per_comparison"] = BOOTSTRAP_DRAWS
    out["composition_task_gate_pass"] = all(
        out[arm]["delta_to_original_donor"] >= -0.02 and
        out[arm]["paired_bootstrap_lower95"] >= -0.05
        for arm in ARMS[1:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert M17.sha(MANIFEST.read_bytes()) == MANIFEST_SHA
    assert M15.M13.sha256(M25.CORE) == M25.CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
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
    manifest, items = M21.bind_data(tokenizer)
    assert json.loads(MANIFEST.read_text(encoding="utf-8")) == manifest
    assert len(items) == M21.ITEMS
    with safe_open(str(M20.ADAPTER), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["model_sha256"] == M15.M13.MODEL_SHA
        assert meta["source_checkpoint_sha256"] == M17.CHECKPOINT_SHA
        assert meta["output_factor"] == "0.5"
    adapter = load_file(str(M20.ADAPTER), device="cpu")
    start = time.monotonic()
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert model.config.eos_token_id == tokenizer.eos_token_id
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    original_params = dict(model.named_parameters())
    assert len(original_params) == 290
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(adapter[f"layers.{li}.{key}"].to(device))
        wrappers.append(wrapper)
    assert len(adapter) == 3 * M15.M13.L
    del adapter

    rows = [{"index": i, "label": item["label"]}
            for i, item in enumerate(items)]

    def score_arm(arm):
        enabled = arm == "r8_adapter"
        for w in wrappers:
            w.enabled = enabled
        with torch.inference_mode():
            for i, (item, row) in enumerate(zip(items, rows)):
                row[arm] = arm_result(model, item, device)
                M21.check_budget(start, device)
                if (i+1) % 256 == 0:
                    print(f"{arm} scored {i+1}/{len(items)} PIQA items", flush=True)

    score_arm("original_donor")
    assert sum(r["original_donor"]["choice_mean"] == r["label"]
               for r in rows) == 1298
    with safe_open(str(M25.CORE), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_R8_CORE_V1"
        assert meta["model_sha256"] == M15.M13.MODEL_SHA
        assert meta["adapter_sha256"] == M20.ADAPTER_SHA
        assert meta["tied_head"] == "true"
        assert len(archive.keys()) == 459
        matrix_count = control_count = 0
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
                param.copy_((codes.to(device).float() *
                             scale.to(device).unsqueeze(1)).to(torch.bfloat16))
                matrix_count += 1
        assert matrix_count == 169 and control_count == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    M21.check_budget(start, device)
    print("loaded stored R8 core", flush=True)
    score_arm("r8_donor")
    score_arm("r8_adapter")
    summary = summarize(rows)
    runtime = M21.check_budget(start, device)
    result = {"experiment": "METH-28", "task_manifest_sha256": MANIFEST_SHA,
              "core_sha256": M25.CORE_SHA, "adapter_sha256": M20.ADAPTER_SHA,
              "model_sha256": M15.M13.MODEL_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "arms": ARMS, "rows": rows, "summary": summary,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "runtime": result["runtime"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
