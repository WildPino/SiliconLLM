#!/usr/bin/env python3
"""METH-37: compare exact and stored-int8 routes on full documents and generation."""
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
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth19_half_residual_independent_audit as M19
import meth20_half_adapter_generation as M20
import meth24_export_r8_core as M24
import meth25_fresh_r8_artifact_audit as M25
import meth25_fresh_r8_manifest as M25M
import meth36_int8_sketch_audit as M36


ROOT = Path(__file__).resolve().parents[3]
M19_MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth19_half_residual_independent_manifest.json"
M19_MANIFEST_SHA = "1eb4193e521df4d25d6367a84d8478d4ea733841fe9a77d550de01796ddf3d70"
M25_RESULT = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth25_fresh_r8_artifact_result.json"
M25_RESULT_SHA = "025c333a9edb4dcf73f6e421f91cdf54ead8cd11351e3fa17d8a6956d7e56a08"
SEED = "meth37-37037"
ARMS = ("original_donor", "r8_exact", "r8_int8_route")
MAX_SECONDS = 20 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)
BOOTSTRAP_SEED = 373737
BOOTSTRAP_DRAWS = 20000


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS:
        raise TimeoutError(f"METH-37 wall-time stop: {elapsed:.1f}s")
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"METH-37 RSS stop: {rss}")
    if peak > MAX_GPU_BYTES:
        raise MemoryError(f"METH-37 GPU stop: {peak}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": peak}


def select(tokenizer):
    assert M17.sha(M19_MANIFEST.read_bytes()) == M19_MANIFEST_SHA
    previous_manifest, items = M19.select()
    assert json.loads(M19_MANIFEST.read_text(encoding="utf-8")) == previous_manifest
    assert M17.sha(M25M.ROOT.joinpath(
        "docs/research/NATIVE_EXPERT_SCALING_20260925/meth25_fresh_r8_manifest.json"
    ).read_bytes()) == M25.MANIFEST_SHA
    _, m25_items = M25M.select()
    excluded = {item["source_id"] for item in m25_items}
    chosen = []
    for category in ("code", "prose", "technical_general"):
        pool = [x for x in items if x["category"] == category]
        pool.sort(key=lambda x: M17.sha((SEED + "|" + x["source_id"]).encode()))
        assert len(pool) >= 8
        for item in pool[:8]:
            assert item["source_id"] not in excluded
            ids = tokenizer.encode(item["text"], add_special_tokens=False)
            assert len(ids) >= 256
            prompt = ids[:256]
            chosen.append({"category": category, "source_id": item["source_id"],
                           "text_sha256": M17.sha(item["text"].encode("utf-8")),
                           "bytes": len(item["text"].encode("utf-8")),
                           "tokens": len(ids),
                           "prompt_ids_sha256": M17.sha(np.asarray(
                               prompt, dtype=np.int32).tobytes()),
                           "ids": ids, "prompt_ids": prompt})
    assert len(chosen) == 24 and len({x["source_id"] for x in chosen}) == 24
    manifest = {"experiment": "METH-37", "seed": SEED,
                "m19_manifest_sha256": M19_MANIFEST_SHA,
                "core_sha256": M25.CORE_SHA,
                "adapter_sha256": M20.ADAPTER_SHA,
                "sketch_sha256": M36.SKETCH_SHA,
                "tokenizer_fingerprint": M15.M13.TOK_FP,
                "selected": [{k: v for k, v in x.items()
                              if k not in ("ids", "prompt_ids")}
                             for x in chosen]}
    return manifest, chosen


class ShortlistExperts(M15.ResidualExperts):
    def __init__(self, base, layer_id):
        super().__init__(base, layer_id)
        self.route_mode = "exact"
        self.candidate_count = 64
        self.rescore_mode = "sum"
        self.register_buffer("basis", torch.zeros(M15.M13.D, 64, dtype=torch.float32))
        self.register_buffer("sketch", torch.zeros(M15.E, 64, dtype=torch.float32))

    def forward(self, x):
        if not self.enabled or self.route_mode == "exact":
            return super().forward(x)
        assert self.route_mode == "stored_int8"
        dense = self.base(x)
        flat = x.reshape(-1, M15.M13.D)
        projected = flat.float() @ self.basis
        coarse = F.linear(projected, self.sketch)
        candidates = torch.topk(coarse, self.candidate_count, dim=-1).indices
        if self.rescore_mode == "sum":
            candidate_scores = (self.router[candidates] *
                                flat.float().unsqueeze(1)).sum(dim=-1)
        elif self.rescore_mode == "bmm":
            candidate_scores = torch.bmm(
                self.router[candidates], flat.float().unsqueeze(-1)).squeeze(-1)
        elif self.rescore_mode == "full_gather":
            candidate_scores = F.linear(flat.float(), self.router).gather(1, candidates)
        else:
            raise ValueError(self.rescore_mode)
        top = torch.topk(candidate_scores, M15.K, dim=-1)
        chosen = candidates.gather(1, top.indices)
        gate = F.softmax(top.values, dim=-1).to(flat.dtype)
        a = self.a[chosen].to(flat.dtype)
        b = self.b[chosen].to(flat.dtype)
        hidden = F.silu(torch.einsum("nd,nkrd->nkr", flat, a))
        out = torch.einsum("nkr,nkdr->nkd", hidden, b)
        residual = (out * gate.unsqueeze(-1)).sum(dim=1).reshape_as(dense)
        return dense + residual


def set_arm(wrappers, arm):
    for wrapper in wrappers:
        wrapper.enabled = arm != "original_donor"
        wrapper.route_mode = "stored_int8" if arm == "r8_int8_route" else "exact"


def score_top1(model, ids, device):
    with torch.inference_mode():
        return model(torch.as_tensor(ids, dtype=torch.long,
                                     device=device).unsqueeze(0),
                     use_cache=False).logits.argmax(dim=-1)[0].cpu().tolist()


def bpb(rows, arm):
    return sum(r["nats"][arm] for r in rows) / (
        math.log(2) * sum(r["bytes"] for r in rows))


def summarize_docs(rows):
    groups = {"pooled": rows}
    groups.update({cat: [r for r in rows if r["category"] == cat]
                   for cat in ("code", "prose", "technical_general")})
    summary = {}
    for category, group in groups.items():
        scores = {arm: bpb(group, arm) for arm in ARMS}
        summary[category] = {"docs": len(group), "bytes": sum(r["bytes"] for r in group),
                             "bpb": scores,
                             "int8_minus_exact": scores["r8_int8_route"] - scores["r8_exact"],
                             "int8_minus_original_donor":
                             scores["r8_int8_route"] - scores["original_donor"]}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = []
        for category in ("code", "prose", "technical_general"):
            group = groups[category]
            sampled.extend(group[i] for i in rng.integers(0, len(group), size=len(group)))
        draws.append(bpb(sampled, "r8_int8_route") - bpb(sampled, "r8_exact"))
    upper = float(np.quantile(draws, 0.95))
    route_loss_pass = (summary["pooled"]["int8_minus_exact"] <= 0.001
                       and all(summary[c]["int8_minus_exact"] <= 0.002
                               for c in ("code", "prose", "technical_general"))
                       and upper <= 0.002)
    donor_pass = (summary["pooled"]["int8_minus_original_donor"] <= 0.01
                  and summary["code"]["int8_minus_original_donor"] <= 0.01
                  and all(summary[c]["int8_minus_original_donor"] <= 0.03
                          for c in ("prose", "technical_general")))
    return summary, {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
                     "penalty_ci95_upper": upper}, route_loss_pass, donor_pass


def summarize_generation(rows, eos_id):
    summary = {}
    for category in ("pooled", "code", "prose", "technical_general"):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        arms = {}
        for arm in ("r8_exact", "r8_int8_route"):
            seqs = [r["generation"][arm]["continuation_ids"] for r in group]
            arms[arm] = {"prompts": len(group),
                         "repeated_8gram_3x": sum(M20.repeated_8gram(x) for x in seqs),
                         "early_non_eos_under16": sum(
                             len(x) < 16 and (not x or x[-1] != eos_id) for x in seqs),
                         "eos_terminated": sum(bool(x) and x[-1] == eos_id for x in seqs),
                         "generated_tokens": sum(len(x) for x in seqs),
                         "mean_distinct2": sum(M20.distinct2(x) for x in seqs) / len(group)}
        summary[category] = arms
    passed = all(
        summary[c]["r8_int8_route"][metric] <= summary[c]["r8_exact"][metric] + 1
        for c in summary for metric in ("repeated_8gram_3x", "early_non_eos_under16"))
    return summary, passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--manifest-sha256")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
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
    manifest, items = select(tokenizer)
    if args.prepare:
        assert args.manifest_sha256 is None and args.out is None
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest_sha256": M17.sha(args.manifest.read_bytes()),
                          "selected_count": len(items)}, indent=2), flush=True)
        return
    assert args.manifest_sha256 and args.out
    assert M17.sha(args.manifest.read_bytes()) == args.manifest_sha256
    assert json.loads(args.manifest.read_text(encoding="utf-8")) == manifest
    assert M15.M13.sha256(M25.CORE) == M25.CORE_SHA
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    assert M15.M13.sha256(M36.SKETCH_ARTIFACT) == M36.SKETCH_SHA
    assert M17.sha(M25_RESULT.read_bytes()) == M25_RESULT_SHA
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
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    original_params = dict(model.named_parameters())
    assert len(original_params) == 290
    adapter = load_file(str(M20.ADAPTER), device="cpu")
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = ShortlistExperts(layer.mlp, li).to(device)
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
            assert bool((codes >= -127).all()) and bool((codes <= 127).all())
            wrapper.basis.copy_(basis.to(device))
            wrapper.sketch.copy_((codes.float() * scales[:, None]).to(device))
    check_budget(start, device)
    print("original donor and stored router sketch loaded", flush=True)

    rows = [{"category": x["category"], "source_id": x["source_id"],
             "text_sha256": x["text_sha256"], "bytes": x["bytes"],
             "tokens": x["tokens"], "nats": {}, "top1": {}, "generation": {}}
            for x in items]
    set_arm(wrappers, "original_donor")
    for item, row in zip(items, rows):
        row["nats"]["original_donor"] = M17.score_doc(
            model, item["ids"], wrappers, False, device, start)
        check_budget(start, device)
    print("original donor documents scored", flush=True)

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
                stored = archive.get_tensor(name)
                assert torch.equal(stored.to(torch.bfloat16), param.detach().cpu())
                controls += 1
            else:
                codes = archive.get_tensor(name + ".q")
                scale = archive.get_tensor(name + ".scale")
                assert codes.dtype == torch.int8 and tuple(codes.shape) == tuple(param.shape)
                assert scale.dtype == torch.float32 and tuple(scale.shape) == (param.shape[0],)
                param.copy_((codes.to(device).float() *
                             scale.to(device).unsqueeze(1)).to(torch.bfloat16))
                matrices += 1
        assert matrices == 169 and controls == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    check_budget(start, device)
    print("stored R8 core reconstructed", flush=True)

    prior = json.loads(M25_RESULT.read_text(encoding="utf-8"))
    _, m25_items = M25M.select()
    m25_by_id = {x["source_id"]: x for x in m25_items}
    set_arm(wrappers, "r8_exact")
    for saved in prior["prompt_rows"]:
        item = m25_by_id[saved["source_id"]]
        ids = tokenizer.encode(item["text"], add_special_tokens=False)[:256]
        assert M17.sha(np.asarray(ids, dtype=np.int32).tobytes()) == saved["prompt_ids_sha256"]
        assert score_top1(model, ids, device) == saved["top1"]["r8_adapter"]
        check_budget(start, device)
    print("exact-route METH-25 top1 controls reproduced", flush=True)

    for arm in ("r8_exact", "r8_int8_route"):
        set_arm(wrappers, arm)
        for i, (item, row) in enumerate(zip(items, rows)):
            row["nats"][arm] = M17.score_doc(
                model, item["ids"], wrappers, True, device, start)
            check_budget(start, device)
            if (i + 1) % 8 == 0:
                print(f"{arm} scored {i+1}/{len(items)} documents", flush=True)
        for item, row in zip(items, rows):
            row["top1"][arm] = score_top1(model, item["prompt_ids"], device)
            check_budget(start, device)
    top1_matching = sum(sum(a == b for a, b in zip(
        row["top1"]["r8_exact"], row["top1"]["r8_int8_route"])) for row in rows)
    top1 = {"positions": len(rows) * 256, "matching": top1_matching,
            "agreement_fraction": top1_matching / (len(rows) * 256)}
    print(f"route-replaced top1 agreement {top1_matching}/{len(rows)*256}", flush=True)

    for arm in ("r8_exact", "r8_int8_route"):
        set_arm(wrappers, arm)
        with torch.inference_mode():
            for i, (item, row) in enumerate(zip(items, rows)):
                inp = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                      device=device).unsqueeze(0)
                output = model.generate(input_ids=inp, max_new_tokens=128,
                                        do_sample=False, pad_token_id=model.config.eos_token_id,
                                        use_cache=True)
                continuation = output[0, 256:].cpu().tolist()
                row["generation"][arm] = {
                    "continuation_ids": continuation,
                    "continuation_text": tokenizer.decode(continuation,
                                                           skip_special_tokens=False),
                    "repeated_8gram_3x": M20.repeated_8gram(continuation),
                    "distinct2": M20.distinct2(continuation)}
                check_budget(start, device)
                if (i + 1) % 8 == 0:
                    print(f"{arm} generated {i+1}/{len(items)} prompts", flush=True)

    docs, bootstrap, loss_pass, donor_pass = summarize_docs(rows)
    generations, generation_pass = summarize_generation(rows, model.config.eos_token_id)
    route_pass = loss_pass and top1["agreement_fraction"] >= 0.99 and generation_pass
    runtime = check_budget(start, device)
    result = {"experiment": "METH-37", "manifest_sha256": args.manifest_sha256,
              "source_sha256": M15.M13.MODEL_SHA,
              "core_sha256": M25.CORE_SHA, "adapter_sha256": M20.ADAPTER_SHA,
              "sketch_sha256": M36.SKETCH_SHA,
              "prior_control_sha256": M25_RESULT_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "rows": rows, "document_summary": docs, "bootstrap": bootstrap,
              "top1": top1, "generation_summary": generations,
              "gates": {"paired_loss": loss_pass, "paired_top1":
                        top1["agreement_fraction"] >= 0.99,
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
