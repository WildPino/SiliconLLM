#!/usr/bin/env python3
"""METH-34: reused-document diagnostic of the stored 15-level LUT adapter."""
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
import meth34_export_i4_factors as M34


ROOT = Path(__file__).resolve().parents[3]
PACKED = ROOT / "results/native_expert_scaling/meth34_qwen05b_i4_lut_adapter.safetensors"
PACKED_SHA = "f4b77f2cd47d8c220872e2609e9ddb01068f4766cd25b7f922fd1ec2a32dc6d8"
PREVIOUS = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth25_fresh_r8_artifact_result.json"
PREVIOUS_SHA = "025c333a9edb4dcf73f6e421f91cdf54ead8cd11351e3fa17d8a6956d7e56a08"
BOOTSTRAP_SEED = 343434
BOOTSTRAP_DRAWS = 20000


def bpb(rows, arm):
    return sum(x["nats"][arm] for x in rows) / (
        math.log(2) * sum(x["bytes"] for x in rows))


def top1_and_routes(model, wrappers, prompts, device, start):
    route_chunks = [[] for _ in wrappers]
    handles = []
    for li, wrapper in enumerate(wrappers):
        def hook(module, input_tuple, layer=li):
            x = input_tuple[0].reshape(-1, M15.M13.D).float()
            assert x.shape[0] == 256
            ids = torch.topk(torch.nn.functional.linear(x, wrappers[layer].router),
                             M15.K, dim=-1).indices
            route_chunks[layer].append(ids.cpu().numpy().astype(np.int16))
        handles.append(wrapper.register_forward_pre_hook(hook))
    top1 = []
    with torch.inference_mode():
        for prompt in prompts:
            logits = model(torch.as_tensor(prompt["ids"], dtype=torch.long,
                                           device=device).unsqueeze(0),
                           use_cache=False).logits
            top1.append(logits.argmax(dim=-1)[0].cpu().tolist())
            M24.check_budget(start, device)
    for handle in handles:
        handle.remove()
    routes = np.stack([np.concatenate(chunks, axis=0) for chunks in route_chunks])
    assert routes.shape == (M15.M13.L, len(prompts) * 256, M15.K)
    return top1, routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert args.artifact_sha256 == PACKED_SHA
    assert M15.M13.sha256(PACKED) == PACKED_SHA
    assert M15.M13.sha256(M25.CORE) == M25.CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    assert M17.sha(M25.MANIFEST.read_bytes()) == M25.MANIFEST_SHA
    assert M17.sha(PREVIOUS.read_bytes()) == PREVIOUS_SHA
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    manifest, items = M25M.select()
    assert json.loads(M25.MANIFEST.read_text(encoding="utf-8")) == manifest
    assert len(items) == 48 and len(previous["rows"]) == 48
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
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(source_adapter[f"layers.{li}.{key}"].to(device))
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
    for item, prior in zip(items, previous["rows"]):
        ids = tokenizer.encode(item["text"], add_special_tokens=False)
        assert item["source_id"] == prior["source_id"]
        assert item["category"] == prior["category"]
        assert len(item["text"].encode()) == prior["bytes"]
        assert len(ids) == prior["tokens"]
        rows.append({"category": item["category"], "source_id": item["source_id"],
                     "bytes": prior["bytes"], "tokens": len(ids), "ids": ids,
                     "original_donor_nats": prior["nats"]["original_donor"],
                     "nats": {}})
    prompts = []
    for prior in previous["prompt_rows"]:
        item = next(x for x in items if x["source_id"] == prior["source_id"])
        ids = tokenizer.encode(item["text"], add_special_tokens=False)[:256]
        assert len(ids) == 256
        assert M17.sha(np.asarray(ids, dtype=np.int32).tobytes()) == prior["prompt_ids_sha256"]
        prompts.append({"category": prior["category"], "source_id": prior["source_id"],
                        "ids": ids, "prior_top1": prior["top1"]["r8_adapter"]})
    assert len(prompts) == 24

    for i, row in enumerate(rows):
        row["nats"]["intact"] = M17.score_doc(
            model, row["ids"], wrappers, True, device, start)
        assert abs(row["nats"]["intact"] - previous["rows"][i]["nats"]["r8_adapter"]) <= 0.01
        M24.check_budget(start, device)
    print("intact R8 adapter reproduced all 48 documents", flush=True)
    intact_top1, intact_routes = top1_and_routes(model, wrappers, prompts, device, start)
    assert all(a == p["prior_top1"] for a, p in zip(intact_top1, prompts))
    print("intact R8 adapter reproduced all 24 top-1 streams", flush=True)

    with safe_open(str(PACKED), framework="pt", device="cpu") as archive:
        meta = archive.metadata()
        assert meta["format"] == "QWEN25_R8_E128_I4_LUT_V1"
        assert meta["core_sha256"] == M25.CORE_SHA
        assert meta["source_adapter_sha256"] == M20.ADAPTER_SHA
        assert len(archive.keys()) == 5 * M15.M13.L
        for li, wrapper in enumerate(wrappers):
            for organ, outputs, inputs, pad in (("a", M15.R, M15.M13.D, 32),
                                                 ("b", M15.M13.D, M15.R, M15.M13.D)):
                key = f"layers.{li}.{organ}"
                code = archive.get_tensor(key + ".code")
                scale = archive.get_tensor(key + ".scale")
                assert tuple(code.shape) == (M15.E, inputs, pad)
                assert tuple(scale.shape) == (M15.E, outputs)
                assert scale.dtype == torch.float32
                q = M34.unpack_codes(code, outputs)
                reconstructed = q.float() * scale.unsqueeze(-1)
                getattr(wrapper, organ).copy_(reconstructed.to(device))
            router = archive.get_tensor(f"layers.{li}.router")
            assert torch.equal(router, source_adapter[f"layers.{li}.router"])
            assert torch.equal(wrapper.router.detach().cpu(), router)
            M24.check_budget(start, device)
    print("packed i4 factors reloaded and decoded", flush=True)
    for i, row in enumerate(rows):
        row["nats"]["i4"] = M17.score_doc(
            model, row["ids"], wrappers, True, device, start)
        M24.check_budget(start, device)
        if (i + 1) % 8 == 0:
            print(f"i4 scored {i+1}/{len(rows)} documents", flush=True)
    i4_top1, i4_routes = top1_and_routes(model, wrappers, prompts, device, start)
    top1_matching = sum(sum(a == b for a, b in zip(original, packed))
                        for original, packed in zip(intact_top1, i4_top1))
    route_id_matching = int((intact_routes == i4_routes).sum())
    route_full_matches = int(np.all(np.sort(intact_routes, axis=-1) ==
                                    np.sort(i4_routes, axis=-1), axis=-1).sum())

    summaries = {}
    for category in ("pooled",) + tuple(M25M.COUNTS):
        subset = rows if category == "pooled" else [x for x in rows if x["category"] == category]
        old_bpb = bpb(subset, "intact")
        new_bpb = bpb(subset, "i4")
        donor_bpb = sum(x["original_donor_nats"] for x in subset) / (
            math.log(2) * sum(x["bytes"] for x in subset))
        summaries[category] = {"docs": len(subset), "bytes": sum(x["bytes"] for x in subset),
                               "intact_bpb": old_bpb, "i4_bpb": new_bpb,
                               "i4_minus_intact_bpb": new_bpb - old_bpb,
                               "i4_minus_original_donor_bpb": new_bpb - donor_bpb}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = {c: [x for x in rows if x["category"] == c] for c in M25M.COUNTS}
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = []
        for category, group in groups.items():
            sample.extend(group[i] for i in rng.integers(0, len(group), len(group)))
        draws.append(bpb(sample, "i4") - bpb(sample, "intact"))
    ci95 = float(np.quantile(draws, 0.95))
    penalty_pass = (summaries["pooled"]["i4_minus_intact_bpb"] <= 0.001
                    and summaries["code"]["i4_minus_intact_bpb"] <= 0.002
                    and summaries["prose"]["i4_minus_intact_bpb"] <= 0.003
                    and summaries["technical_general"]["i4_minus_intact_bpb"] <= 0.003
                    and ci95 <= 0.003)
    donor_pass = (summaries["pooled"]["i4_minus_original_donor_bpb"] <= 0.01
                  and summaries["code"]["i4_minus_original_donor_bpb"] <= 0.01
                  and summaries["prose"]["i4_minus_original_donor_bpb"] <= 0.03
                  and summaries["technical_general"]["i4_minus_original_donor_bpb"] <= 0.03)
    top1_pass = top1_matching / 6144 >= 0.99
    selected_bytes = M15.M13.L * M15.K * (
        M15.M13.D * 32 + M15.R * M15.M13.D +
        (M15.R + M15.M13.D) * 4)
    assert selected_bytes == 3787776
    byte_pass = selected_bytes <= 4200000
    diagnostic_pass = byte_pass and penalty_pass and donor_pass and top1_pass
    runtime = M24.check_budget(start, device)
    for row in rows:
        del row["ids"]
    for prompt in prompts:
        del prompt["ids"]
        del prompt["prior_top1"]
    result = {"experiment": "METH-34", "packed_adapter_sha256": PACKED_SHA,
              "source_adapter_sha256": M20.ADAPTER_SHA,
              "core_sha256": M25.CORE_SHA,
              "document_manifest_sha256": M25.MANIFEST_SHA,
              "prior_result_sha256": PREVIOUS_SHA,
              "selected_factor_bytes_per_token": selected_bytes,
              "rows": rows, "summary": summaries,
              "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
                            "penalty_ci95_upper": ci95},
              "top1": {"positions": 6144, "matching": top1_matching,
                       "agreement_fraction": top1_matching / 6144},
              "route_diagnostic": {"exact_ids": int(intact_routes.size),
                                   "matching_ids_same_rank": route_id_matching,
                                   "input_layer_cases": int(intact_routes.shape[0] * intact_routes.shape[1]),
                                   "full_set_matches": route_full_matches},
              "gates": {"bytes": byte_pass, "penalty": penalty_pass,
                        "donor": donor_pass, "top1": top1_pass,
                        "diagnostic_pass": diagnostic_pass},
              "decision": "candidate_for_fresh_quality_and_native_lut_audit" if
              diagnostic_pass else "i4_factor_diagnostic_fail",
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summaries, "bootstrap": result["bootstrap"],
                      "top1": result["top1"], "route_diagnostic": result["route_diagnostic"],
                      "gates": result["gates"], "runtime": result["runtime"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
