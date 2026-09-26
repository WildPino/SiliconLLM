#!/usr/bin/env python3
"""METH-33: diagnostic A-only/B-only ternary factor precision audit."""
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
import meth24_export_r8_core as M24
import meth25_fresh_r8_manifest as M25M
import meth25_fresh_r8_artifact_audit as M25
import meth32_export_ternary_factors as M32E
import meth32_ternary_factor_audit as M32A


ROOT = Path(__file__).resolve().parents[3]
M32_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth32_ternary_factor_audit.json"
M32_RESULT_SHA = "c378cf06e26934144c603fd9cd2ffe4b12e305fd6af272c5bb9da4bae33ad960"
ARMS = ("a_ternary_b_fp32", "a_fp32_b_ternary")
BOOTSTRAP_SEED = 333333
BOOTSTRAP_DRAWS = 20000


def summary_for(rows, arm):
    summary = {}
    for category in ("pooled",) + tuple(M25M.COUNTS):
        subset = rows if category == "pooled" else [x for x in rows if x["category"] == category]
        nbytes = sum(x["bytes"] for x in subset)
        def score(name):
            return sum(x["nats"][name] for x in subset) / (math.log(2) * nbytes)
        donor = sum(x["donor_nats"] for x in subset) / (math.log(2) * nbytes)
        summary[category] = {"docs": len(subset), "bytes": nbytes,
                             "intact_bpb": score("intact"),
                             "mixed_bpb": score(arm),
                             "mixed_minus_intact_bpb": score(arm) - score("intact"),
                             "mixed_minus_original_donor_bpb": score(arm) - donor}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = {cat: [x for x in rows if x["category"] == cat] for cat in M25M.COUNTS}
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = []
        for group in groups.values():
            sample.extend(group[i] for i in rng.integers(0, len(group), len(group)))
        nbytes = sum(x["bytes"] for x in sample)
        draws.append(sum(x["nats"][arm] - x["nats"]["intact"] for x in sample) /
                     (math.log(2) * nbytes))
    return summary, float(np.quantile(draws, 0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert M15.M13.sha256(M25.CORE) == M25.CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    assert M15.M13.sha256(M32A.PACKED) == M32A.PACKED_SHA
    assert M17.sha(M25.MANIFEST.read_bytes()) == M25.MANIFEST_SHA
    assert M17.sha(M32A.PREVIOUS.read_bytes()) == M32A.PREVIOUS_SHA
    assert M17.sha(M32_RESULT.read_bytes()) == M32_RESULT_SHA
    prior = json.loads(M32A.PREVIOUS.read_text(encoding="utf-8"))
    prior32 = json.loads(M32_RESULT.read_text(encoding="utf-8"))
    assert prior32["packed_adapter_sha256"] == M32A.PACKED_SHA
    manifest, items = M25M.select()
    assert json.loads(M25.MANIFEST.read_text(encoding="utf-8")) == manifest
    assert len(items) == 48
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
    source_adapter = load_file(str(M20.ADAPTER), device="cpu")
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for organ in ("a", "b", "router"):
            getattr(wrapper, organ).copy_(source_adapter[f"layers.{li}.{organ}"].to(device))
        wrappers.append(wrapper)
    with safe_open(str(M25.CORE), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_R8_CORE_V1"
        assert meta["model_sha256"] == M15.M13.MODEL_SHA
        assert meta["adapter_sha256"] == M20.ADAPTER_SHA
        assert meta["tied_head"] == "true"
        assert len(archive.keys()) == 459
        matrix_count = control_count = 0
        for name, param in original_params.items():
            if M24.classify(name, tuple(param.shape)) == "control":
                stored = archive.get_tensor(name)
                assert stored.dtype == torch.float32
                assert torch.equal(stored.to(torch.bfloat16), param.detach().cpu())
                control_count += 1
            else:
                q = archive.get_tensor(name + ".q")
                scale = archive.get_tensor(name + ".scale")
                assert q.dtype == torch.int8 and tuple(q.shape) == tuple(param.shape)
                assert scale.dtype == torch.float32 and tuple(scale.shape) == (param.shape[0],)
                param.copy_((q.to(device).float() *
                             scale.to(device).unsqueeze(1)).to(torch.bfloat16))
                matrix_count += 1
        assert matrix_count == 169 and control_count == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    print("stored R8 core and intact adapter loaded", flush=True)

    rows = []
    for item, previous in zip(items, prior["rows"]):
        ids = tokenizer.encode(item["text"], add_special_tokens=False)
        assert item["source_id"] == previous["source_id"]
        assert item["category"] == previous["category"]
        assert len(item["text"].encode()) == previous["bytes"]
        assert len(ids) == previous["tokens"]
        rows.append({"category": item["category"], "source_id": item["source_id"],
                     "bytes": previous["bytes"], "tokens": len(ids), "ids": ids,
                     "donor_nats": previous["nats"]["original_donor"],
                     "nats": {}})
    prompts = []
    for previous in prior["prompt_rows"]:
        item = next(x for x in items if x["source_id"] == previous["source_id"])
        ids = tokenizer.encode(item["text"], add_special_tokens=False)[:256]
        assert len(ids) == 256
        assert M17.sha(np.asarray(ids, dtype=np.int32).tobytes()) == previous["prompt_ids_sha256"]
        prompts.append({"category": previous["category"], "source_id": previous["source_id"],
                        "ids": ids, "prior_top1": previous["top1"]["r8_adapter"]})
    assert len(prompts) == 24

    for i, row in enumerate(rows):
        row["nats"]["intact"] = M17.score_doc(
            model, row["ids"], wrappers, True, device, start)
        assert abs(row["nats"]["intact"] - prior["rows"][i]["nats"]["r8_adapter"]) <= 0.01
        assert abs(row["nats"]["intact"] - prior32["rows"][i]["nats"]["intact"]) <= 0.01
        M24.check_budget(start, device)
    intact_top1, intact_routes = M32A.top1_and_routes(model, wrappers, prompts, device, start)
    assert all(a == p["prior_top1"] for a, p in zip(intact_top1, prompts))
    print("intact controls reproduced: 48 documents and 24 top-1 streams", flush=True)

    ternary = {"a": [], "b": []}
    with safe_open(str(M32A.PACKED), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_R8_E128_TERNARY_LUT_V1"
        assert meta["core_sha256"] == M25.CORE_SHA
        assert meta["source_adapter_sha256"] == M20.ADAPTER_SHA
        assert len(archive.keys()) == 5 * M15.M13.L
        for li in range(M15.M13.L):
            for organ, outputs, inputs, pad in (("a", M15.R, M15.M13.D, 32),
                                                 ("b", M15.M13.D, M15.R, M15.M13.D)):
                key = f"layers.{li}.{organ}"
                code = archive.get_tensor(key + ".code")
                scale = archive.get_tensor(key + ".scale")
                assert tuple(code.shape) == (M15.E, inputs // 2, pad)
                assert tuple(scale.shape) == (M15.E, outputs)
                ternary[organ].append((M32E.unpack_codes(code, outputs).float() *
                                       scale.unsqueeze(-1)).contiguous())
            assert torch.equal(archive.get_tensor(f"layers.{li}.router"),
                               source_adapter[f"layers.{li}.router"])
    assert all(len(v) == M15.M13.L for v in ternary.values())
    print("stored ternary A/B factors decoded", flush=True)

    code_a = M15.M13.D // 2 * 32
    scale_a = M15.R * 4
    code_b = M15.R // 2 * M15.M13.D
    scale_b = M15.M13.D * 4
    fp_a = fp_b = M15.R * M15.M13.D * 4
    selected_bytes = {"a_ternary_b_fp32": M15.M13.L * M15.K * (code_a + scale_a + fp_b),
                      "a_fp32_b_ternary": M15.M13.L * M15.K * (fp_a + code_b + scale_b)}
    assert selected_bytes == {"a_ternary_b_fp32": 4131840,
                              "a_fp32_b_ternary": 3440640}
    arm_results = {}
    for arm in ARMS:
        quantized = "a" if arm == "a_ternary_b_fp32" else "b"
        unmodified = "b" if quantized == "a" else "a"
        for li, wrapper in enumerate(wrappers):
            getattr(wrapper, quantized).copy_(ternary[quantized][li].to(device))
            getattr(wrapper, unmodified).copy_(source_adapter[f"layers.{li}.{unmodified}"].to(device))
            assert torch.equal(wrapper.router.detach().cpu(),
                               source_adapter[f"layers.{li}.router"])
        for i, row in enumerate(rows):
            row["nats"][arm] = M17.score_doc(
                model, row["ids"], wrappers, True, device, start)
            M24.check_budget(start, device)
            if (i + 1) % 16 == 0:
                print(f"{arm} scored {i+1}/{len(rows)} documents", flush=True)
        mixed_top1, mixed_routes = M32A.top1_and_routes(model, wrappers, prompts, device, start)
        matching = sum(sum(a == b for a, b in zip(original, variant))
                       for original, variant in zip(intact_top1, mixed_top1))
        cat_top1 = {}
        for category in M25M.COUNTS:
            indices = [i for i, p in enumerate(prompts) if p["category"] == category]
            hits = sum(sum(a == b for a, b in zip(intact_top1[i], mixed_top1[i]))
                       for i in indices)
            cat_top1[category] = {"positions": len(indices) * 256,
                                  "matching": hits,
                                  "agreement_fraction": hits / (len(indices) * 256)}
        full_sets = int(np.all(np.sort(intact_routes, axis=-1) ==
                               np.sort(mixed_routes, axis=-1), axis=-1).sum())
        summary, ci95 = summary_for(rows, arm)
        penalty_pass = (summary["pooled"]["mixed_minus_intact_bpb"] <= 0.001
                        and summary["code"]["mixed_minus_intact_bpb"] <= 0.002
                        and summary["prose"]["mixed_minus_intact_bpb"] <= 0.003
                        and summary["technical_general"]["mixed_minus_intact_bpb"] <= 0.003
                        and ci95 <= 0.003)
        donor_pass = (summary["pooled"]["mixed_minus_original_donor_bpb"] <= 0.01
                      and summary["code"]["mixed_minus_original_donor_bpb"] <= 0.01
                      and summary["prose"]["mixed_minus_original_donor_bpb"] <= 0.03
                      and summary["technical_general"]["mixed_minus_original_donor_bpb"] <= 0.03)
        top1_pass = matching / 6144 >= 0.99
        byte_pass = selected_bytes[arm] <= 4200000
        arm_results[arm] = {"selected_factor_bytes_per_token": selected_bytes[arm],
                            "summary": summary,
                            "bootstrap_penalty_ci95_upper": ci95,
                            "top1": {"positions": 6144, "matching": matching,
                                     "agreement_fraction": matching / 6144,
                                     "categories": cat_top1},
                            "route_diagnostic": {"input_layer_cases": int(intact_routes.shape[0] *
                                                                             intact_routes.shape[1]),
                                                 "full_set_matches": full_sets,
                                                 "full_set_match_fraction": full_sets /
                                                 (intact_routes.shape[0] * intact_routes.shape[1])},
                            "gates": {"bytes": byte_pass, "penalty": penalty_pass,
                                      "donor": donor_pass, "top1": top1_pass,
                                      "diagnostic_pass": byte_pass and penalty_pass and
                                      donor_pass and top1_pass}}
        print(f"{arm}: top1={matching}/6144 pooled_penalty="
              f"{summary['pooled']['mixed_minus_intact_bpb']:.6f} pass="
              f"{arm_results[arm]['gates']['diagnostic_pass']}", flush=True)
    passing = [arm for arm in ARMS if arm_results[arm]["gates"]["diagnostic_pass"]]
    chosen = min(passing, key=lambda arm: selected_bytes[arm]) if passing else None
    runtime = M24.check_budget(start, device)
    for row in rows:
        del row["ids"]
    for prompt in prompts:
        del prompt["ids"]
        del prompt["prior_top1"]
    result = {"experiment": "METH-33", "core_sha256": M25.CORE_SHA,
              "source_adapter_sha256": M20.ADAPTER_SHA,
              "ternary_adapter_sha256": M32A.PACKED_SHA,
              "document_manifest_sha256": M25.MANIFEST_SHA,
              "prior_meth32_result_sha256": M32_RESULT_SHA,
              "arms": arm_results, "rows": rows,
              "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS},
              "chosen_diagnostic_arm": chosen,
              "decision": "mixed_candidate_for_stored_export_and_fresh_audit" if chosen else
              "both_mixed_factor_arms_fail_diagnostic",
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arms": arm_results, "chosen_diagnostic_arm": chosen,
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
