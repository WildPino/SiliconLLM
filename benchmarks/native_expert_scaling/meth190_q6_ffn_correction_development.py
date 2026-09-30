#!/usr/bin/env python3
"""Viewed-source development gate for stored METH-189 FFN correction."""

import argparse
import json
import math
from pathlib import Path
import time

from safetensors import safe_open
import torch

import meth187_q6_core_e1280_development as P
import meth188_core_organ_attribution as A
import meth189_train_q6_ffn_correction as T


PRIOR = P.DOC / "meth187_q6_core_e1280_development_result.json"
PRIOR_SHA = "7ec10c7fe1de942bfb40be77f813b83888d6d658d0a8ec844beef96822f46e1c"
TRAIN = P.DOC / "meth189_q6_ffn_correction_train_result.json"
ARMS = ("bf16_e1280", "q6_e1280", "q6_e1280_rank64")


def aggregate(arms):
    result = {}
    for category in ("pooled", *P.CATEGORIES):
        values = {}
        for arm in ARMS:
            docs = [r for r in arms[arm]["document_rows"]
                    if category == "pooled" or r["category"] == category]
            prompts = [r for r in arms[arm]["prompt_rows"]
                       if category == "pooled" or r["category"] == category]
            assert len(docs) == len(prompts) == (24 if category == "pooled" else 8)
            values[arm] = {"bpb": sum(r["nats"] for r in docs) /
                           (math.log(2) * sum(r["bytes"] for r in docs)),
                           "top1": sum(r["matching"] for r in prompts) /
                           sum(r["positions"] for r in prompts)}
        reference = values["bf16_e1280"]
        corrected = values["q6_e1280_rank64"]
        original = values["q6_e1280"]
        result[category] = {
            "arms": values,
            "corrected_minus_bf16_bpb": corrected["bpb"] - reference["bpb"],
            "corrected_minus_bf16_top1": corrected["top1"] - reference["top1"],
            "corrected_minus_uncorrected_bpb": corrected["bpb"] - original["bpb"],
            "corrected_minus_uncorrected_top1": corrected["top1"] - original["top1"]}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assert not args.out.exists()
    assert P.digest(PRIOR) == PRIOR_SHA
    assert P.digest(P.M122.MANIFEST) == P.M122.MANIFEST_SHA
    assert P.digest(P.M122.SPECIALIZED) == P.M122.SPECIALIZED_SHA
    assert P.digest(P.CORE) == json.loads(P.EXPORT.read_text(encoding="utf-8"))["sha256"]
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    training = json.loads(TRAIN.read_text(encoding="utf-8"))
    checkpoint = Path(training["checkpoint"]["path"])
    assert P.digest(checkpoint) == training["checkpoint"]["sha256"]
    assert training["checkpoint"]["payload_bytes"] == T.PAYLOAD
    assert training["ideal_addressed_bytes_per_token"] == 534_619_136
    parent = json.loads(P.M122.TRAINING.read_text(encoding="utf-8"))["checkpoints"]["512"]
    assert P.digest(parent["path"]) == P.M57.CHECKPOINT_SHA
    items = json.loads(P.M122.MANIFEST.read_text(encoding="utf-8"))["items"]
    assert len(items) == 24

    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    matches = [i for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    parent_state = torch.load(parent["path"], map_location="cpu", weights_only=False)
    child_state = torch.load(P.M122.SPECIALIZED, map_location="cpu", weights_only=False)
    teacher, teacher_wrappers, _ = T.make_model(device, parent_state, child_state)
    teacher.eval()
    student, student_wrappers, corrections = T.make_model(
        device, parent_state, child_state, core=True)
    student.eval()
    del parent_state, child_state
    donor_top = {}
    with torch.inference_mode():
        P.M44.set_experts(teacher_wrappers, False)
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long,
                                  device=device)[None]
            donor_top[item["source_id"]] = teacher(ids, use_cache=False).logits.argmax(-1)[0].cpu()
    arms = {}
    arms[ARMS[0]] = A.evaluate(teacher, teacher_wrappers, items, donor_top,
                              ARMS[0], device, start)
    arms[ARMS[1]] = A.evaluate(student, student_wrappers, items, donor_top,
                              ARMS[1], device, start)
    for arm in ARMS[:2]:
        for new, old in zip(arms[arm]["document_rows"], prior["document_rows"]):
            assert new["source_id"] == old["source_id"]
            assert new["nats"] == old["nats"][arm]
        for new, old in zip(arms[arm]["prompt_rows"], prior["prompt_rows"]):
            assert new["source_id"] == old["source_id"]
            assert new["matching"] == old["matching"][arm]
    print(json.dumps({"baseline_parity": True,
                      "runtime": T.budget(start, device)}), flush=True)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as archive:
        assert archive.metadata()["format"] == "METH189_Q6_FFN_RANK64_BF16_V1"
        assert archive.metadata()["core_sha256"] == training["core_sha256"]
        assert len(archive.keys()) == 144
        with torch.no_grad():
            for name, module in corrections.items():
                for key in ("a", "b"):
                    stored = archive.get_tensor(name + "." + key)
                    assert stored.dtype == torch.bfloat16
                    parameter = getattr(module, key)
                    assert stored.shape == parameter.shape
                    parameter.copy_(stored.to(device))
    arms[ARMS[2]] = A.evaluate(student, student_wrappers, items, donor_top,
                              ARMS[2], device, start)
    summary = aggregate(arms)
    gates = {"pooled_bpb": summary["pooled"]["corrected_minus_bf16_bpb"] <= .01,
             "category_bpb": all(summary[c]["corrected_minus_bf16_bpb"] <= .02
                                 for c in P.CATEGORIES),
             "pooled_top1": summary["pooled"]["corrected_minus_bf16_top1"] >= -.01,
             "category_top1": all(summary[c]["corrected_minus_bf16_top1"] >= -.02
                                  for c in P.CATEGORIES)}
    result = {"experiment": "METH-190-Q6-FFN-rank64-development",
              "prior_sha256": PRIOR_SHA, "training_sha256": P.digest(TRAIN),
              "core_sha256": training["core_sha256"],
              "checkpoint_sha256": training["checkpoint"]["sha256"],
              "ideal_addressed_bytes_per_token": 534_619_136,
              "arms": arms, "summary": summary, "gates": gates,
              "decision": "development_pass_fresh_quality_pending" if all(gates.values())
                          else "development_fail_do_not_spend_fresh_sources",
              "runtime": {**T.budget(start, device),
                          "gpu": torch.cuda.get_device_name(device)},
              "scope": "Already viewed METH-121 sources; development only"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert args.out.stat().st_size < T.MAX_DISK
    print(json.dumps({"decision": result["decision"], "gates": gates,
                      "summary": summary["pooled"],
                      "runtime": result["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
