#!/usr/bin/env python3
"""Attribute the viewed-source METH-187 loss to stored head and FFN organs."""

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch

import meth187_q6_core_e1280_development as P


PRIOR = P.DOC / "meth187_q6_core_e1280_development_result.json"
PRIOR_SHA = "7ec10c7fe1de942bfb40be77f813b83888d6d658d0a8ec844beef96822f46e1c"
ARMS = ("bf16_e1280", "r8_head_only", "q6_ffn_only", "r8_head_q6_ffn")


def evaluate(model, wrappers, items, donor_top, arm, device, start):
    documents, prompts = [], []
    with torch.inference_mode():
        for item in items:
            nats = P.M17.score_doc(model, item["document_ids"], wrappers,
                                   True, device, start)
            documents.append({"source_id": item["source_id"],
                              "category": item["category"],
                              "bytes": item["bytes"], "nats": nats})
            P.budget(start, device)
        P.M44.set_experts(wrappers, True)
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                  device=device)[None]
            top = model(ids, use_cache=False).logits.argmax(-1)[0].cpu()
            prompts.append({"source_id": item["source_id"],
                            "category": item["category"],
                            "positions": len(item["prompt_ids"]),
                            "matching": int((top == donor_top[item["source_id"]]).sum())})
            P.budget(start, device)
    return {"arm": arm, "document_rows": documents, "prompt_rows": prompts}


def summary(arms):
    import math
    result = {}
    for category in ("pooled", *P.CATEGORIES):
        rows = {}
        for arm, data in arms.items():
            documents = [r for r in data["document_rows"]
                         if category == "pooled" or r["category"] == category]
            prompts = [r for r in data["prompt_rows"]
                       if category == "pooled" or r["category"] == category]
            assert len(documents) == len(prompts) == (24 if category == "pooled" else 8)
            rows[arm] = {
                "bpb": sum(r["nats"] for r in documents) /
                       (math.log(2) * sum(r["bytes"] for r in documents)),
                "top1": sum(r["matching"] for r in prompts) /
                        sum(r["positions"] for r in prompts)}
        reference = rows["bf16_e1280"]
        losses = {arm: (reference["top1"] - row["top1"]) * 100
                  for arm, row in rows.items()}
        full = losses["r8_head_q6_ffn"]
        decision = ("head_first" if full > 0 and
                    losses["r8_head_only"] >= .8 * full and
                    losses["q6_ffn_only"] <= .2 * full else
                    "ffn_first" if full > 0 and
                    losses["q6_ffn_only"] >= .8 * full and
                    losses["r8_head_only"] <= .2 * full else "joint")
        result[category] = {"arms": rows, "ranking_loss_points": losses,
                            "ranking_interaction_points": full - losses["r8_head_only"]
                            - losses["q6_ffn_only"], "priority": decision}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assert not args.out.exists()
    assert P.digest(PRIOR) == PRIOR_SHA
    assert P.digest(P.M122.MANIFEST) == P.M122.MANIFEST_SHA
    assert P.digest(P.M122.SPECIALIZED) == P.M122.SPECIALIZED_SHA
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    manifest = json.loads(P.M122.MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(P.EXPORT.read_text(encoding="utf-8"))
    assert P.digest(P.CORE) == report["sha256"]
    assert len(manifest["items"]) == 24
    parent = json.loads(P.M122.TRAINING.read_text(encoding="utf-8"))["checkpoints"]["512"]
    assert P.digest(parent["path"]) == P.M57.CHECKPOINT_SHA
    source = Path(hf_hub_download(P.M42.MODEL, "model.safetensors",
                                  revision=P.M42.REV, local_files_only=True))
    assert P.digest(source) == P.M57.MODEL_SHA

    from transformers import AutoModelForCausalLM
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        P.M42.MODEL, revision=P.M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    model.config.use_cache = False
    original_params = dict(model.named_parameters())
    assert len(original_params) == 290
    parent_state = torch.load(parent["path"], map_location="cpu", weights_only=False)
    child_state = torch.load(P.M122.SPECIALIZED, map_location="cpu", weights_only=False)
    assert parent_state["updates"] == 512 and child_state["updates"] == 256
    wrappers = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            base = P.M55.ProductKeyExperts(layer.mlp, li).to(device)
            for key in ("a", "b", "router"):
                getattr(base, key).copy_(parent_state["expert_state"][li][key].to(device))
            wrapper = P.M95.HierarchicalExperts(base, li).to(device)
            saved = child_state["expert_state"][li]
            for key in ("a", "b", "router", "child_projection", "child_keys"):
                getattr(wrapper, key).copy_(saved[key].to(device))
            raw_b = wrapper.b.view(P.M15.E, P.M95.CHILDREN, P.M15.M13.D, P.M15.R)
            wrapper.b.copy_((base.b[:, None] + raw_b - raw_b.mean(dim=1, keepdim=True))
                            .reshape_as(wrapper.b))
            layer.mlp = wrapper
            wrappers.append(wrapper)
    del parent_state, child_state
    P.budget(start, device)
    items = manifest["items"]
    donor_top = {}
    with torch.inference_mode():
        P.M44.set_experts(wrappers, False)
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                  device=device)[None]
            donor_top[item["source_id"]] = model(ids, use_cache=False).logits.argmax(-1)[0].cpu()
    P.budget(start, device)
    arms = {}
    arms[ARMS[0]] = evaluate(model, wrappers, items, donor_top, ARMS[0], device, start)
    print(json.dumps({"arm": ARMS[0], "budget": P.budget(start, device)}), flush=True)

    head = model.model.embed_tokens.weight
    assert head.data_ptr() == model.lm_head.weight.data_ptr()
    original_head = head.detach().cpu().clone()
    with safe_open(str(P.CORE), framework="pt", device="cpu") as archive:
        assert archive.metadata()["format"] == "QWEN25_INSTRUCT_R8H_GROUP64_Q6FFN_BF16ATTN_V1"
        with torch.no_grad():
            q = archive.get_tensor("model.embed_tokens.weight.q").to(device)
            scale = archive.get_tensor("model.embed_tokens.weight.scale").to(device)
            head.copy_((q.float() * scale[:, None]).bfloat16())
        arms[ARMS[1]] = evaluate(model, wrappers, items, donor_top, ARMS[1], device, start)
        print(json.dumps({"arm": ARMS[1], "budget": P.budget(start, device)}), flush=True)
        with torch.no_grad():
            head.copy_(original_head.to(device))
            assert torch.equal(head.cpu(), original_head)
            changed = 0
            for name, param in original_params.items():
                if P.M24.classify(name, tuple(param.shape)) != "ffn":
                    continue
                param.copy_(P.M186.reconstruct(archive.get_tensor(name + ".q6").to(device),
                                               archive.get_tensor(name + ".scale").to(device)))
                changed += 1
            assert changed == 72
        arms[ARMS[2]] = evaluate(model, wrappers, items, donor_top, ARMS[2], device, start)
        print(json.dumps({"arm": ARMS[2], "budget": P.budget(start, device)}), flush=True)
        with torch.no_grad():
            head.copy_((q.float() * scale[:, None]).bfloat16())
        arms[ARMS[3]] = evaluate(model, wrappers, items, donor_top, ARMS[3], device, start)
        print(json.dumps({"arm": ARMS[3], "budget": P.budget(start, device)}), flush=True)

    for arm, old in ((ARMS[0], "bf16_e1280"), (ARMS[3], "q6_e1280")):
        for new_row, prior_row in zip(arms[arm]["document_rows"], prior["document_rows"]):
            assert new_row["source_id"] == prior_row["source_id"]
            assert new_row["nats"] == prior_row["nats"][old]
        for new_row, prior_row in zip(arms[arm]["prompt_rows"], prior["prompt_rows"]):
            assert new_row["source_id"] == prior_row["source_id"]
            assert new_row["matching"] == prior_row["matching"][old]
    result = {"experiment": "METH-188-core-organ-attribution",
              "source_sha256": P.M57.MODEL_SHA,
              "core_sha256": report["sha256"],
              "prior_sha256": PRIOR_SHA,
              "manifest_sha256": P.M122.MANIFEST_SHA,
              "arms": arms, "summary": summary(arms),
              "runtime": {**P.budget(start, device), "gpu": torch.cuda.get_device_name(device)},
              "scope": "Already viewed METH-121 sources; development attribution only"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.out.stat().st_size < P.MAX_DISK
    print(json.dumps({"pooled": result["summary"]["pooled"],
                      "runtime": result["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
