"""Data-free least-squares R8 scale diagnostic under BF16 attention."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil
from safetensors import safe_open
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth24_export_r8_core as M24
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57
import meth59_instruct_r8_composition as M59


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
M60_RESULT = DIR / "meth60_r8_core_organ_ablation_result.json"
M60_SHA = "22bac64fa592f6e56c77111082ea51fc10e7476db2d37ed9b445e0fc01d28df3"
TRAINING = DIR / "meth56_product_key_retention_result.json"
ARMS = ("bf16_attention_r8", "l2_head", "l2_ffn", "l2_head_ffn")
MAX_SECONDS = 10 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    gpu = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES or gpu > MAX_GPU_BYTES:
        raise RuntimeError(f"METH-61 budget: {elapsed:.1f}s, RSS {rss}, GPU {gpu}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": gpu}


def l2_codes(weight):
    """Four fixed LS scale updates, three with code refresh and one final."""
    assert weight.dtype == torch.float32 and weight.ndim == 2
    scale = (weight.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    codes = (weight / scale).round().clamp(-127, 127)
    for _ in range(3):
        scale = ((weight * codes).sum(dim=1, keepdim=True) /
                 codes.square().sum(dim=1, keepdim=True).clamp_min(1.0)).clamp_min(1e-12)
        codes = (weight / scale).round().clamp(-127, 127)
    scale = ((weight * codes).sum(dim=1, keepdim=True) /
             codes.square().sum(dim=1, keepdim=True).clamp_min(1.0)).clamp_min(1e-12)
    return codes.to(torch.int8), scale


def category_summary(rows):
    result = {}
    for category in ("pooled", *M57.CATEGORIES):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        positions = sum(r["positions"] for r in group)
        result[category] = {"positions": positions}
        for arm in ("bf16_student", *ARMS):
            matching = sum(r["matches"][arm] for r in group)
            result[category][arm] = {"matching": matching,
                                     "agreement": matching / positions}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert M15.M13.sha256(M60_RESULT) == M60_SHA
    assert M15.M13.sha256(M59.CORE) == M59.CORE_SHA
    assert M15.M13.sha256(TRAINING) == M57.TRAINING_SHA
    assert M15.M13.sha256(M57.EXTERNAL) == M57.EXTERNAL_SHA
    prior = json.loads(M60_RESULT.read_text(encoding="utf-8"))
    assert prior["summary"]["pooled"]["bf16_attention"]["matching"] == 3914
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    checkpoint = training["checkpoints"]["512"]
    assert checkpoint["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(checkpoint["path"]) == M57.CHECKPOINT_SHA
    manifest = json.loads(M57.EXTERNAL.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert len(items) == 24
    for item in items:
        assert M17.sha(np.asarray(item["prompt_ids"], dtype=np.int32).tobytes()) == item["prompt_ids_sha256"]
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source_path = hf_hub_download(M42.MODEL, "model.safetensors",
                                  revision=M42.REV, local_files_only=True)
    assert M15.M13.sha256(source_path) == M57.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    for param in model.parameters():
        param.requires_grad_(False)
    original_params = dict(model.named_parameters())
    assert len(original_params) == 290
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    assert state["updates"] == 512 and state["source_sha256"] == M57.MODEL_SHA
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(state["expert_state"][li][key].to(device))
        wrappers.append(wrapper)
    model.config.use_cache = False
    prompt_rows = [{"source_id": item["source_id"], "category": item["category"],
                    "prompt_ids_sha256": item["prompt_ids_sha256"],
                    "positions": len(item["prompt_ids"]), "matches": {}}
                   for item in items]
    donor_top = []
    with torch.inference_mode():
        for i, item in enumerate(items):
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            M44.set_experts(wrappers, False)
            donor = model(ids, use_cache=False).logits.argmax(dim=-1)
            donor_top.append(donor)
            M44.set_experts(wrappers, True)
            student = model(ids, use_cache=False).logits.argmax(dim=-1)
            prompt_rows[i]["matches"]["bf16_student"] = int((student == donor).sum())
            budget(start, device)
    assert sum(r["matches"]["bf16_student"] for r in prompt_rows) == 3997

    error_totals = {organ: {"original_sse": 0.0, "l2_sse": 0.0,
                            "weight_square_sum": 0.0, "matrices": 0}
                    for organ in ("tied_head", "ffn")}
    with safe_open(str(source_path), framework="pt", device="cpu") as source, \
         safe_open(str(M59.CORE), framework="pt", device="cpu") as packed:
        matrix_names = [name for name, param in original_params.items()
                        if M24.classify(name, tuple(param.shape)) != "control"]
        assert len(matrix_names) == 169
        for name, param in original_params.items():
            if M24.classify(name, tuple(param.shape)) == "control":
                assert torch.equal(packed.get_tensor(name).to(torch.bfloat16),
                                   param.detach().cpu())
        # Cache only one matrix's candidate at a time; each arm is rebuilt
        # directly from the two bound source archives, never prompt outcomes.
        def set_arm(arm):
            for name in matrix_names:
                param = original_params[name]
                organ = M24.classify(name, tuple(param.shape))
                if organ == "attention":
                    param.copy_(source.get_tensor(name).to(device))
                    continue
                use_l2 = ((organ == "tied_head" and arm in ("l2_head", "l2_head_ffn")) or
                          (organ == "ffn" and arm in ("l2_ffn", "l2_head_ffn")))
                if use_l2:
                    original = source.get_tensor(name).to(device).float()
                    codes, scale = l2_codes(original)
                    reconstructed = (codes.float() * scale).to(torch.bfloat16)
                    if arm == "l2_head_ffn":
                        old_codes = packed.get_tensor(name + ".q").to(device)
                        old_scale = packed.get_tensor(name + ".scale").to(device)
                        old = (old_codes.float() * old_scale[:, None]).to(torch.bfloat16)
                        stats = error_totals[organ]
                        stats["original_sse"] += float(((old.float() - original) ** 2).sum())
                        stats["l2_sse"] += float(((reconstructed.float() - original) ** 2).sum())
                        stats["weight_square_sum"] += float((original ** 2).sum())
                        stats["matrices"] += 1
                else:
                    codes = packed.get_tensor(name + ".q").to(device)
                    scale = packed.get_tensor(name + ".scale").to(device)
                    reconstructed = (codes.float() * scale[:, None]).to(torch.bfloat16)
                param.copy_(reconstructed)
                budget(start, device)
            assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()

        for arm in ARMS:
            set_arm(arm)
            with torch.inference_mode():
                M44.set_experts(wrappers, True)
                for i, item in enumerate(items):
                    ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
                    top = model(ids, use_cache=False).logits.argmax(dim=-1)
                    prompt_rows[i]["matches"][arm] = int((top == donor_top[i]).sum())
                    budget(start, device)
            matching = sum(r["matches"][arm] for r in prompt_rows)
            print(f"{arm}: {matching}/4125", flush=True)

    summary = category_summary(prompt_rows)
    assert summary["pooled"]["bf16_attention_r8"]["matching"] == 3914
    bytes_per_token = prior["byte_ledger"]["bf16_attention"]["ideal_addressed_total_bytes"]
    assert bytes_per_token == 548320256
    gates = {arm: {"top1": summary["pooled"][arm]["agreement"] >= 0.95 and
                      all(summary[c][arm]["agreement"] >= 0.90 for c in M57.CATEGORIES),
                   "traffic": bytes_per_token <= 560_000_000}
             for arm in ARMS}
    for arm in ARMS:
        gates[arm]["joint_diagnostic"] = all(gates[arm].values())
    result = {"experiment": "METH-61-weight-only-L2-R8-scale-diagnostic",
              "source_sha256": M57.MODEL_SHA, "core_sha256": M59.CORE_SHA,
              "checkpoint_sha256": M57.CHECKPOINT_SHA,
              "external_manifest_sha256": M57.EXTERNAL_SHA,
              "m60_result_sha256": M60_SHA,
              "prompt_rows": prompt_rows, "summary": summary,
              "error_totals": error_totals,
              "ideal_addressed_bytes_per_token": bytes_per_token,
              "gates": gates,
              "decision": "weight_only_candidate_for_packed_export"
                          if any(g["joint_diagnostic"] for g in gates.values())
                          else "fixed_L2_scale_rule_does_not_rescue_quality",
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": summary["pooled"], "error_totals": error_totals,
                      "gates": gates, "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
