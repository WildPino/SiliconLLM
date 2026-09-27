"""Attribute METH-59 R8 ranking loss to head, attention or FFN bytes."""

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
TRAINING = DIR / "meth56_product_key_retention_result.json"
M59_RESULT = DIR / "meth59_instruct_r8_composition_result.json"
ARMS = ("all_r8", "bf16_head", "bf16_attention", "bf16_ffn")
MAX_SECONDS = 10 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    gpu = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES or gpu > MAX_GPU_BYTES:
        raise RuntimeError(f"METH-60 budget: {elapsed:.1f}s, RSS {rss}, GPU {gpu}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": gpu}


def summaries(rows):
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
    assert M15.M13.sha256(M59.CORE) == M59.CORE_SHA
    assert M15.M13.sha256(TRAINING) == M57.TRAINING_SHA
    assert M15.M13.sha256(M57.EXTERNAL) == M57.EXTERNAL_SHA
    prior = json.loads(M59_RESULT.read_text(encoding="utf-8"))
    assert prior["decision"] == "stop_before_generation_task_early_gate_failed"
    assert prior["core_sha256"] == M59.CORE_SHA
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
    for p in model.parameters():
        p.requires_grad_(False)
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
    baseline_matching = sum(row["matches"]["bf16_student"] for row in prompt_rows)
    assert baseline_matching == 3997
    print("BF16 control 3997/4125 reproduced", flush=True)

    with safe_open(str(source_path), framework="pt", device="cpu") as source, \
         safe_open(str(M59.CORE), framework="pt", device="cpu") as packed:
        assert len(packed.keys()) == 459
        meta = packed.metadata()
        assert meta["model_sha256"] == M57.MODEL_SHA
        assert meta["product_key_checkpoint_sha256"] == M57.CHECKPOINT_SHA
        matrix_names = [name for name, param in original_params.items()
                        if M24.classify(name, tuple(param.shape)) != "control"]
        assert len(matrix_names) == 169
        for name, param in original_params.items():
            if M24.classify(name, tuple(param.shape)) == "control":
                assert torch.equal(packed.get_tensor(name).to(torch.bfloat16),
                                   param.detach().cpu())

        def set_arm(arm):
            for name in matrix_names:
                param = original_params[name]
                organ = M24.classify(name, tuple(param.shape))
                restore = ((arm == "bf16_head" and organ == "tied_head") or
                           (arm == "bf16_attention" and organ == "attention") or
                           (arm == "bf16_ffn" and organ == "ffn"))
                if restore:
                    original = source.get_tensor(name)
                    assert original.dtype == torch.bfloat16
                    param.copy_(original.to(device))
                else:
                    codes = packed.get_tensor(name + ".q")
                    scale = packed.get_tensor(name + ".scale")
                    assert codes.dtype == torch.int8 and scale.dtype == torch.float32
                    param.copy_((codes.to(device).float() *
                                 scale.to(device).unsqueeze(1)).to(torch.bfloat16))
            assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
            budget(start, device)

        for arm in ARMS:
            set_arm(arm)
            with torch.inference_mode():
                M44.set_experts(wrappers, True)
                for i, item in enumerate(items):
                    ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
                    top = model(ids, use_cache=False).logits.argmax(dim=-1)
                    prompt_rows[i]["matches"][arm] = int((top == donor_top[i]).sum())
                    budget(start, device)
            matched = sum(row["matches"][arm] for row in prompt_rows)
            print(f"{arm}: {matched}/{sum(row['positions'] for row in prompt_rows)}", flush=True)

    summary = summaries(prompt_rows)
    assert summary["pooled"]["all_r8"]["matching"] == prior["prompt_summary"]["pooled"]["r8_student_matching"]
    export = json.loads((DIR / "meth59_instruct_r8_core_export.json").read_text(encoding="utf-8"))
    assert export["sha256"] == M59.CORE_SHA
    organs = export["organs"]
    core_bytes = sum(v["int8_code_bytes"] + v["fp32_row_scale_bytes"]
                     for v in organs.values()) + organs["control"]["parameters"] * 4
    router_bytes = M15.M13.L * (M55.ROUTER_RANK * M15.M13.D +
                    (M55.AXIS_A + M55.AXIS_B) * M55.ROUTER_RANK) * 4
    selected_bytes = M15.M13.L * M15.K * 2 * M15.R * M15.M13.D * 2
    byte_ledger = {}
    for arm in ARMS:
        restore = {"all_r8": None, "bf16_head": "tied_head",
                   "bf16_attention": "attention", "bf16_ffn": "ffn"}[arm]
        extra = 0
        if restore:
            organ = organs[restore]
            extra = organ["parameters"] * 2 - organ["int8_code_bytes"] - organ["fp32_row_scale_bytes"]
        total = core_bytes + router_bytes + selected_bytes + extra
        byte_ledger[arm] = {"core_bytes": core_bytes + extra,
                            "router_bytes": router_bytes,
                            "selected_factor_bytes": selected_bytes,
                            "ideal_addressed_total_bytes": total,
                            "under_560m": total <= 560_000_000,
                            "extra_over_all_r8_bytes": extra}
    gates = {arm: {"top1": summary["pooled"][arm]["agreement"] >= 0.95 and
                      all(summary[c][arm]["agreement"] >= 0.90 for c in M57.CATEGORIES),
                   "traffic": byte_ledger[arm]["under_560m"]}
             for arm in ARMS}
    for arm in ARMS:
        gates[arm]["joint_diagnostic"] = all(gates[arm].values())
    result = {"experiment": "METH-60-R8-core-organ-ranking-ablation",
              "source_sha256": M57.MODEL_SHA, "core_sha256": M59.CORE_SHA,
              "checkpoint_sha256": M57.CHECKPOINT_SHA,
              "external_manifest_sha256": M57.EXTERNAL_SHA,
              "m59_result_sha256": M15.M13.sha256(M59_RESULT),
              "prompt_rows": prompt_rows, "summary": summary,
              "byte_ledger": byte_ledger, "gates": gates,
              "decision": "mixed_precision_candidate_for_independent_audit"
                          if any(g["joint_diagnostic"] for g in gates.values())
                          else "no_single_organ_rescue_meets_both_gates",
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": summary["pooled"], "byte_ledger": byte_ledger,
                      "gates": gates, "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
