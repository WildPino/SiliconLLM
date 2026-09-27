#!/usr/bin/env python3
"""Audit row-scaled int8 factors against the bound METH-47 checkpoint."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth48_int8_factor_bank as M48


def quantize_per_row(value):
    assert value.dtype == torch.float32 and value.ndim == 3
    maximum = value.abs().amax(dim=2)
    scale = maximum / 127.0
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(value / safe[:, :, None]).clamp(-127, 127).to(torch.int8)
    restored = codes.float() * scale[:, :, None]
    sqerr = float((value.double() - restored.double()).square().sum())
    sqnorm = float(value.double().square().sum())
    maxerr = float((value - restored).abs().max())
    return codes.contiguous(), scale.contiguous(), sqerr, sqnorm, maxerr


def restore_factors(wrappers, archive, device):
    with torch.no_grad():
        for li, wrapper in enumerate(wrappers):
            for factor, rows in (("a", M15.R), ("b", M15.M13.D)):
                code = archive.get_tensor(f"layers.{li}.{factor}_codes")
                scale = archive.get_tensor(f"layers.{li}.{factor}_scales")
                assert code.dtype == torch.int8 and tuple(code.shape) == tuple(getattr(wrapper, factor).shape)
                assert scale.dtype == torch.float32 and tuple(scale.shape) == (M15.E, rows)
                assert bool(torch.isfinite(scale).all()) and bool((scale >= 0).all())
                value = code.float() * scale[:, :, None]
                getattr(wrapper, factor).copy_(value.to(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path, digest in ((M48.CHECKPOINT, M48.CHECKPOINT_SHA),
                         (M48.MANIFEST, M48.MANIFEST_SHA),
                         (M48.REFERENCE, M48.REFERENCE_SHA)):
        assert M15.M13.sha256(path) == digest, path
    manifest = json.loads(M48.MANIFEST.read_text(encoding="utf-8"))
    reference = json.loads(M48.REFERENCE.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert len(items) == len(reference["generation_rows"]) == 12
    for item in items:
        assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        assert M17.sha(np.asarray(item["document_ids"], dtype=np.int32).tobytes()) == item["document_ids_sha256"]
        assert M17.sha(np.asarray(item["prompt_ids"], dtype=np.int32).tobytes()) == item["prompt_ids_sha256"]
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()

    state = torch.load(M48.CHECKPOINT, map_location="cpu", weights_only=False)
    assert state["updates"] == 512 and state["source_sha256"] == M44.MODEL_SHA
    assert state["teacher_sha256"] == M44.TEACHER_SHA
    assert len(state["expert_state"]) == M15.M13.L
    tensors = {}
    error = {"a": {"squared_error": 0.0, "squared_norm": 0.0, "max_abs_error": 0.0},
             "b": {"squared_error": 0.0, "squared_norm": 0.0, "max_abs_error": 0.0}}
    for li, values in enumerate(state["expert_state"]):
        for factor in ("a", "b"):
            value = values[factor]
            assert tuple(value.shape) == ((M15.E, M15.R, M15.M13.D) if factor == "a"
                                          else (M15.E, M15.M13.D, M15.R))
            code, scale, sqerr, sqnorm, maxerr = quantize_per_row(value)
            tensors[f"layers.{li}.{factor}_codes"] = code
            tensors[f"layers.{li}.{factor}_scales"] = scale
            error[factor]["squared_error"] += sqerr
            error[factor]["squared_norm"] += sqnorm
            error[factor]["max_abs_error"] = max(error[factor]["max_abs_error"], maxerr)
        assert tuple(values["router"].shape) == (M15.E, M15.M13.D)
        assert bool(torch.isfinite(values["router"]).all())
    metadata = {"format": "METH49_E128_AB_INT8_PER_ROW_V1",
                "checkpoint_sha256": M48.CHECKPOINT_SHA,
                "model_sha256": M44.MODEL_SHA,
                "quantization": "round_even_clamp_-127_127_per_expert_per_factor_row"}
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.artifact), metadata=metadata)
    with safe_open(str(args.artifact), framework="pt", device="cpu") as archive:
        assert archive.metadata() == metadata and len(archive.keys()) == 4 * M15.M13.L
        for key, value in tensors.items():
            assert torch.equal(archive.get_tensor(key), value), key
    code_bytes_per_expert = M15.M13.L * 2 * M15.R * M15.M13.D
    scale_bytes_per_expert = M15.M13.L * (M15.R + M15.M13.D) * 4
    payload_bytes_per_expert = code_bytes_per_expert + scale_bytes_per_expert
    assert sum(t.numel() * t.element_size() for t in tensors.values()) == M15.E * payload_bytes_per_expert
    for factor in ("a", "b"):
        error[factor]["relative_squared_error"] = (error[factor]["squared_error"] /
                                                        error[factor]["squared_norm"])

    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert M15.M13.sha256(source) == M44.MODEL_SHA
    tok = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                        local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrappers = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
            layer.mlp = wrapper
            for factor in ("a", "b", "router"):
                getattr(wrapper, factor).copy_(state["expert_state"][li][factor].to(device))
            wrappers.append(wrapper)
    model.config.use_cache = False
    original_generation = M48.score_generations(model, wrappers, items, device, start)
    for row, saved in zip(original_generation, reference["generation_rows"]):
        assert row["source_id"] == saved["source_id"]
        assert row["continuation_ids"] == saved["student"]["continuation_ids"], row["source_id"]
    original_docs = M48.score_documents(model, wrappers, items, device, start)
    original_prompts = M48.score_prompt_ids(model, wrappers, items, device, start)
    with safe_open(str(args.artifact), framework="pt", device="cpu") as archive:
        restore_factors(wrappers, archive, device)
    packed_docs = M48.score_documents(model, wrappers, items, device, start)
    packed_prompts = M48.score_prompt_ids(model, wrappers, items, device, start)
    packed_generation = M48.score_generations(model, wrappers, items, device, start)
    docs = M48.document_summary(original_docs, packed_docs)
    prompts = M48.compare_prompts(original_prompts, packed_prompts)
    generations = M48.compare_generations(original_generation, packed_generation)
    gates = {
        "documents": all(docs[c]["int8_minus_original_bpb"] <= 0.002
                         for c in ("pooled", *M48.CATEGORIES)),
        "prompt_top1": prompts["pooled"]["agreement"] >= 0.99,
        "generation": all(generations[c]["int8_eos"] >= generations[c]["original_eos"] and
                          generations[c]["int8_early_non_eos"] <= generations[c]["original_early_non_eos"] and
                          generations[c]["int8_repeated_8gram"] <= generations[c]["original_repeated_8gram"]
                          for c in ("pooled", *M48.CATEGORIES)),
    }
    gates["joint"] = all(gates.values())
    runtime = M48.budget(start, device)
    result = {"experiment": "METH-49-int8-row-trained-factor-bank",
              "checkpoint_sha256": M48.CHECKPOINT_SHA, "model_sha256": M44.MODEL_SHA,
              "manifest_sha256": M48.MANIFEST_SHA, "reference_sha256": M48.REFERENCE_SHA,
              "artifact_sha256": M15.M13.sha256(args.artifact),
              "artifact_bytes": args.artifact.stat().st_size,
              "factor_code_bytes_per_expert": code_bytes_per_expert,
              "factor_scale_bytes_per_expert": scale_bytes_per_expert,
              "factor_payload_bytes_per_expert": payload_bytes_per_expert,
              "projected_factor_payload_bytes": {str(e): e * payload_bytes_per_expert
                                                 for e in (128, 27355, 273547)},
              "quantization_error": error,
              "original_document_rows": original_docs, "int8_document_rows": packed_docs,
              "document_summary": docs, "prompt_summary": prompts,
              "original_generation_rows": original_generation,
              "int8_generation_rows": packed_generation,
              "generation_summary": generations, "gates": gates,
              "decision": "int8_row_factor_bank_diagnostic_pass" if gates["joint"]
                          else "stop_row_scale_int8_factor_bank",
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("artifact_sha256", "artifact_bytes",
                                            "factor_payload_bytes_per_expert",
                                            "projected_factor_payload_bytes",
                                            "quantization_error", "document_summary",
                                            "prompt_summary", "generation_summary",
                                            "gates", "decision", "runtime")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
