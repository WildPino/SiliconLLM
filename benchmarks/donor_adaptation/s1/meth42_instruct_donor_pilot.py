#!/usr/bin/env python3
"""Compare BF16 base, Instruct and base-adapter-grafted Instruct generation."""
import argparse
import json
import math
from pathlib import Path
import time

import psutil
from safetensors.torch import load_file
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth37_route_replacement as M37
import meth41_fresh_c96_manifest as M41
import meth42_instruct_prompt_manifest as M42M


ROOT = Path(__file__).resolve().parents[3]
PROMPTS = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth42_instruct_prompt_manifest.json"
PROMPTS_SHA = "09813338debb3ba962c7d2108f743e6adc4f5e6c076542c49b837eacd39858fa"
INSTRUCT_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
MAX_SECONDS = 20 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)
ARMS = ("base", "instruct", "instruct_graft")


def check_budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    peak = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or peak > MAX_GPU_BYTES or rss > MAX_RSS_BYTES:
        raise RuntimeError(f"METH-42 budget: {elapsed:.1f}s GPU={peak} RSS={rss}")
    return {"elapsed_seconds": elapsed, "gpu_peak_allocated_bytes": peak,
            "rss_end_bytes": rss}


def generate_arm(model, rows, arm, tokenizer, device, start):
    with torch.inference_mode():
        for i, row in enumerate(rows):
            ids = row["prompt_ids"]
            inp = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            output = model.generate(input_ids=inp, max_new_tokens=128,
                                    do_sample=False,
                                    pad_token_id=model.config.eos_token_id,
                                    use_cache=True)
            continuation = output[0, len(ids):].cpu().tolist()
            row["generation"][arm] = {
                "continuation_ids": continuation,
                "continuation_text": tokenizer.decode(
                    continuation, skip_special_tokens=False),
                "repeated_8gram_3x": M20.repeated_8gram(continuation),
                "distinct2": M20.distinct2(continuation)}
            check_budget(start, device)
            if (i + 1) % 8 == 0:
                print(f"{arm} generated {i+1}/{len(rows)}", flush=True)


def summarize_generation(rows, eos_id):
    summary = {}
    for cat in ("pooled", "code", "prose", "technical_general"):
        group = rows if cat == "pooled" else [r for r in rows if r["category"] == cat]
        summary[cat] = {}
        for arm in ARMS:
            seqs = [r["generation"][arm]["continuation_ids"] for r in group]
            summary[cat][arm] = {
                "prompts": len(group),
                "repeated_8gram_3x": sum(M20.repeated_8gram(x) for x in seqs),
                "early_non_eos_under16": sum(
                    len(x) < 16 and (not x or x[-1] != eos_id) for x in seqs),
                "eos_terminated": sum(bool(x) and x[-1] == eos_id for x in seqs),
                "generated_tokens": sum(map(len, seqs)),
                "mean_distinct2": sum(M20.distinct2(x) for x in seqs) / len(group)}
    return summary


def summarize_docs(rows):
    summary = {}
    for cat in ("pooled", "code", "prose", "technical_general"):
        group = rows if cat == "pooled" else [r for r in rows if r["category"] == cat]
        total_bytes = sum(r["bytes"] for r in group)
        scores = {arm: sum(r["nats"][arm] for r in group) /
                  (math.log(2) * total_bytes)
                  for arm in ("instruct", "instruct_graft")}
        summary[cat] = {"docs": len(group), "bytes": total_bytes,
                        "bpb": scores,
                        "graft_minus_instruct":
                        scores["instruct_graft"] - scores["instruct"]}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert M17.sha(PROMPTS.read_bytes()) == PROMPTS_SHA
    prompt_manifest = json.loads(PROMPTS.read_text(encoding="utf-8"))
    assert prompt_manifest["model"] == M42M.MODEL
    assert prompt_manifest["revision"] == M42M.REV
    assert prompt_manifest["tokenizer_fingerprint"] == M15.M13.TOK_FP
    assert len(prompt_manifest["items"]) == 24
    assert M15.M13.sha256(M20.ADAPTER) == M20.ADAPTER_SHA
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base_path = hf_hub_download(M15.M13.MODEL, "model.safetensors",
                                revision=M15.M13.REV, local_files_only=True)
    instruct_path = hf_hub_download(M42M.MODEL, "model.safetensors",
                                    revision=M42M.REV, local_files_only=True)
    assert M15.M13.sha256(base_path) == M15.M13.MODEL_SHA
    assert M15.M13.sha256(instruct_path) == INSTRUCT_SHA
    tokenizer = AutoTokenizer.from_pretrained(
        M42M.MODEL, revision=M42M.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    _, docs = M41.select(tokenizer)
    rows = []
    for fixed, item in zip(prompt_manifest["items"], docs):
        assert fixed["source_id"] == item["source_id"]
        assert fixed["category"] == item["category"]
        rows.append({"category": item["category"],
                     "source_id": item["source_id"],
                     "text_sha256": item["text_sha256"],
                     "bytes": item["bytes"],
                     "prompt_ids": fixed["prompt_ids"],
                     "prompt_ids_sha256": fixed["prompt_ids_sha256"],
                     "nats": {}, "top1": {}, "generation": {}})
    assert len({r["source_id"] for r in rows}) == 24
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    base = AutoModelForCausalLM.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert base.model.embed_tokens.weight.data_ptr() == base.lm_head.weight.data_ptr()
    generate_arm(base, rows, "base", tokenizer, device, start)
    del base
    torch.cuda.empty_cache()
    instruct = AutoModelForCausalLM.from_pretrained(
        M42M.MODEL, revision=M42M.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    assert instruct.model.embed_tokens.weight.data_ptr() == instruct.lm_head.weight.data_ptr()
    assert len(dict(instruct.named_parameters())) == 290
    for item, row in zip(docs, rows):
        row["nats"]["instruct"] = M17.score_doc(
            instruct, item["ids"], [], False, device, start)
        check_budget(start, device)
    print("Instruct documents scored", flush=True)
    generate_arm(instruct, rows, "instruct", tokenizer, device, start)
    adapter = load_file(str(M20.ADAPTER), device="cpu")
    assert len(adapter) == 3 * M15.M13.L
    wrappers = []
    for li, layer in enumerate(instruct.model.layers):
        wrapper = M15.ResidualExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(adapter[f"layers.{li}.{key}"].to(device))
        wrappers.append(wrapper)
    del adapter
    check_budget(start, device)
    for item, row in zip(docs, rows):
        row["nats"]["instruct_graft"] = M17.score_doc(
            instruct, item["ids"], wrappers, True, device, start)
        check_budget(start, device)
    print("grafted Instruct documents scored", flush=True)
    for enabled, arm in ((False, "instruct"), (True, "instruct_graft")):
        for wrapper in wrappers:
            wrapper.enabled = enabled
        for row in rows:
            row["top1"][arm] = M37.score_top1(
                instruct, row["prompt_ids"], device)
            check_budget(start, device)
    for wrapper in wrappers:
        wrapper.enabled = True
    generate_arm(instruct, rows, "instruct_graft", tokenizer, device, start)
    docs_summary = summarize_docs(rows)
    gen_summary = summarize_generation(rows, instruct.config.eos_token_id)
    top1_matching = sum(sum(a == b for a, b in zip(
        row["top1"]["instruct"], row["top1"]["instruct_graft"]))
                        for row in rows)
    top1_total = sum(len(row["prompt_ids"]) for row in rows)
    top1 = {"positions": top1_total, "matching": top1_matching,
            "agreement_fraction": top1_matching / top1_total}
    instruct_utility = (gen_summary["pooled"]["instruct"]["repeated_8gram_3x"] <= 4
                        and gen_summary["code"]["instruct"]["repeated_8gram_3x"] <= 2
                        and all(gen_summary[c]["instruct"]["early_non_eos_under16"] == 0
                                for c in gen_summary))
    graft_loss = (docs_summary["pooled"]["graft_minus_instruct"] <= 0.01
                  and all(docs_summary[c]["graft_minus_instruct"] <= 0.02
                          for c in ("code", "prose", "technical_general")))
    graft_generation = all(
        gen_summary[c]["instruct_graft"][metric] <=
        gen_summary[c]["instruct"][metric] + 1
        for c in gen_summary for metric in
        ("repeated_8gram_3x", "early_non_eos_under16"))
    gates = {"instruct_utility": instruct_utility,
             "graft_loss": graft_loss,
             "graft_top1": top1["agreement_fraction"] >= 0.95,
             "graft_generation": graft_generation}
    gates["graft_candidate"] = all(gates.values())
    runtime = check_budget(start, device)
    result = {"experiment": "METH-42", "prompt_manifest_sha256": PROMPTS_SHA,
              "base_sha256": M15.M13.MODEL_SHA,
              "instruct_revision": M42M.REV,
              "instruct_sha256": INSTRUCT_SHA,
              "adapter_sha256": M20.ADAPTER_SHA,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "rows": rows, "document_summary": docs_summary,
              "generation_summary": gen_summary, "top1": top1,
              "gates": gates,
              "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"document_summary": docs_summary,
                      "generation_summary": gen_summary,
                      "top1": top1, "gates": gates,
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
