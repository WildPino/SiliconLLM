"""Diagnostic paired audit of stored Instruct R8 core plus METH-56 experts."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
from safetensors import safe_open
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth20_half_adapter_generation as M20
import meth21_half_adapter_piqa as M21
import meth24_export_r8_core as M24
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
CORE = ROOT / "results/native_expert_scaling/meth59_qwen05b_instruct_r8_core.safetensors"
CORE_SHA = "5b6ace4a027d127a6eb49ca99197edd4a1b810ebe681d8e3b6ab7edc088ca9bd"
REFERENCE = DIR / "meth57_product_key_external_audit_result.json"
REFERENCE_SHA = "b88e9208db0999a8b3b8550a8e5a1bd4c265910a0b202dc451c1914e2823b996"
TRAINING = DIR / "meth56_product_key_retention_result.json"
MAX_SECONDS = 25 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    gpu = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES or gpu > MAX_GPU_BYTES:
        raise RuntimeError(f"METH-59 diagnostic budget: {elapsed:.1f}s, RSS {rss}, GPU {gpu}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": gpu}


def save_partial(path, stage, payload):
    path.write_text(json.dumps({"experiment": "METH-59", "completed_stage": stage,
                                "core_sha256": CORE_SHA, "reference_sha256": REFERENCE_SHA,
                                "external_manifest_sha256": M57.EXTERNAL_SHA,
                                **payload}, indent=2) + "\n", encoding="utf-8")


def doc_summary(rows):
    result = {}
    for category in ("pooled", *M57.CATEGORIES):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        denom = math.log(2) * sum(r["bytes"] for r in group)
        arm_bpb = {arm: sum(r["nats"][arm] for r in group) / denom
                   for arm in ("bf16_donor", "r8_donor", "r8_student")}
        result[category] = {"documents": len(group), "bytes": sum(r["bytes"] for r in group),
                            "bpb": arm_bpb,
                            "r8_donor_delta": arm_bpb["r8_donor"] - arm_bpb["bf16_donor"],
                            "r8_student_delta": arm_bpb["r8_student"] - arm_bpb["bf16_donor"]}
    return result


def prompt_summary(rows):
    result = {}
    for category in ("pooled", *M57.CATEGORIES):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        positions = sum(r["positions"] for r in group)
        result[category] = {"positions": positions,
                            "r8_donor_matching": sum(r["r8_donor_matching"] for r in group),
                            "r8_student_matching": sum(r["r8_student_matching"] for r in group),
                            "r8_student_agreement": sum(r["r8_student_matching"] for r in group) / positions}
    return result


def generation_summary(rows):
    result = {}
    for category in ("pooled", *M57.CATEGORIES):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        result[category] = {arm: {
            "prompts": len(group),
            "eos_terminated": sum(r[arm]["eos_terminated"] for r in group),
            "early_non_eos_under16": sum(r[arm]["early_non_eos_under16"] for r in group),
            "repeated_8gram_3x": sum(r[arm]["repeated_8gram_3x"] for r in group)}
            for arm in ("bf16_donor", "r8_student")}
    return result


def task_summary(rows):
    donor = np.asarray([int(r["bf16_donor"]["choice_mean"] == r["label"])
                        for r in rows], dtype=np.int8)
    student = np.asarray([int(r["r8_student"]["choice_mean"] == r["label"])
                          for r in rows], dtype=np.int8)
    paired = student.astype(np.int16) - donor.astype(np.int16)
    rng = np.random.default_rng(M21.BOOTSTRAP_SEED)
    draws = [float(paired[rng.integers(0, len(rows), len(rows))].mean())
             for _ in range(M21.BOOTSTRAP_DRAWS)]
    return {"items": len(rows), "donor_correct": int(donor.sum()),
            "student_correct": int(student.sum()),
            "accuracy_delta": float(paired.mean()),
            "paired_bootstrap_lower95": float(np.quantile(draws, 0.05)),
            "bootstrap_seed": M21.BOOTSTRAP_SEED,
            "bootstrap_draws": M21.BOOTSTRAP_DRAWS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.stem + ".partial.json")
    assert M15.M13.sha256(CORE) == CORE_SHA
    assert M15.M13.sha256(REFERENCE) == REFERENCE_SHA
    assert M15.M13.sha256(TRAINING) == M57.TRAINING_SHA
    assert M15.M13.sha256(M57.EXTERNAL) == M57.EXTERNAL_SHA
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    checkpoint = training["checkpoints"]["512"]
    assert checkpoint["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(checkpoint["path"]) == M57.CHECKPOINT_SHA
    manifest = json.loads(M57.EXTERNAL.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert len(items) == 24
    ref_docs = {r["source_id"]: r for r in reference["document_rows"]}
    ref_generations = {r["source_id"]: r for r in reference["generation_rows"]}
    assert set(ref_docs) == set(ref_generations) == {r["source_id"] for r in items}
    for item in items:
        assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        assert M17.sha(np.asarray(item["document_ids"], dtype=np.int32).tobytes()) == item["document_ids_sha256"]
        assert M17.sha(np.asarray(item["prompt_ids"], dtype=np.int32).tobytes()) == item["prompt_ids_sha256"]
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV,
                             local_files_only=True)
    assert M15.M13.sha256(source) == M57.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    _, task_items = M21.bind_data(tokenizer)
    assert len(task_items) == 1838 == len(reference["task_rows"])
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
    assert state["teacher_sha256"] == M44.TEACHER_SHA
    wrappers = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            getattr(wrapper, key).copy_(state["expert_state"][li][key].to(device))
        wrappers.append(wrapper)
    model.config.use_cache = False
    donor_prompt_top = {}
    with torch.inference_mode():
        M44.set_experts(wrappers, False)
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            donor_prompt_top[item["source_id"]] = model(ids, use_cache=False).logits.argmax(dim=-1).cpu().tolist()[0]
    budget(start, device)
    with safe_open(str(CORE), framework="pt", device="cpu") as archive:
        metadata = archive.metadata()
        assert metadata["format"] == "QWEN25_INSTRUCT_R8_CORE_V1"
        assert metadata["model_sha256"] == M57.MODEL_SHA
        assert metadata["product_key_checkpoint_sha256"] == M57.CHECKPOINT_SHA
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
                reconstructed = (codes.to(device).float() *
                                 scale.to(device).unsqueeze(1)).to(torch.bfloat16)
                param.copy_(reconstructed)
                matrix_count += 1
        assert matrix_count == 169 and control_count == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    budget(start, device)
    doc_rows = []
    for item in items:
        source_id = item["source_id"]
        donor = ref_docs[source_id]
        assert donor["document_ids_sha256"] == item["document_ids_sha256"]
        row = {"source_id": source_id, "category": item["category"],
               "bytes": item["bytes"], "document_ids_sha256": item["document_ids_sha256"],
               "nats": {"bf16_donor": donor["donor_nats"]}}
        row["nats"]["r8_donor"] = M17.score_doc(model, item["document_ids"], wrappers, False, device, start)
        row["nats"]["r8_student"] = M17.score_doc(model, item["document_ids"], wrappers, True, device, start)
        doc_rows.append(row)
        budget(start, device)
        print(f"documents {len(doc_rows)}/{len(items)}", flush=True)
    documents = doc_summary(doc_rows)
    save_partial(partial, "documents", {"document_rows": doc_rows, "document_summary": documents})
    prompt_rows = []
    with torch.inference_mode():
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            original = donor_prompt_top[item["source_id"]]
            M44.set_experts(wrappers, False)
            r8_donor = model(ids, use_cache=False).logits.argmax(dim=-1).cpu().tolist()[0]
            M44.set_experts(wrappers, True)
            r8_student = model(ids, use_cache=False).logits.argmax(dim=-1).cpu().tolist()[0]
            assert len(original) == len(r8_donor) == len(r8_student)
            prompt_rows.append({"source_id": item["source_id"], "category": item["category"],
                                "prompt_ids_sha256": item["prompt_ids_sha256"],
                                "positions": len(original),
                                "r8_donor_matching": sum(a == b for a, b in zip(original, r8_donor)),
                                "r8_student_matching": sum(a == b for a, b in zip(original, r8_student))})
            budget(start, device)
    prompts = prompt_summary(prompt_rows)
    early_gates = {
        "documents": (documents["pooled"]["r8_student_delta"] <= 0.02 and
                      all(documents[c]["r8_student_delta"] <= 0.04 for c in M57.CATEGORIES)),
        "prompt_top1": (prompts["pooled"]["r8_student_agreement"] >= 0.95 and
                        all(prompts[c]["r8_student_agreement"] >= 0.90 for c in M57.CATEGORIES))}
    save_partial(partial, "prompts", {"document_rows": doc_rows,
        "document_summary": documents, "prompt_rows": prompt_rows,
        "prompt_summary": prompts, "early_gates": early_gates})
    print(json.dumps({"documents": documents, "prompts": prompts, "early_gates": early_gates}), flush=True)
    if not all(early_gates.values()):
        print("METH-59 early gate failed; stopped before generation/task", flush=True)
        return

    generation_rows = []
    eos = model.config.eos_token_id
    model.config.use_cache = True
    with torch.inference_mode():
        for item in items:
            source_id = item["source_id"]
            original = ref_generations[source_id]
            assert original["prompt_ids_sha256"] == item["prompt_ids_sha256"]
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            M44.set_experts(wrappers, True)
            output = model.generate(input_ids=ids, max_new_tokens=128,
                                    do_sample=False, pad_token_id=eos,
                                    use_cache=True)
            continuation = output[0, ids.shape[1]:].cpu().tolist()
            generated = {"continuation_ids": continuation,
                         "continuation_text": tokenizer.decode(continuation, skip_special_tokens=False),
                         "eos_terminated": bool(continuation) and continuation[-1] == eos,
                         "early_non_eos_under16": len(continuation) < 16 and
                                                (not continuation or continuation[-1] != eos),
                         "repeated_8gram_3x": M20.repeated_8gram(continuation),
                         "distinct2": M20.distinct2(continuation)}
            generation_rows.append({"source_id": source_id, "category": item["category"],
                                    "prompt_ids_sha256": item["prompt_ids_sha256"],
                                    "bf16_donor": original["donor"],
                                    "r8_student": generated})
            budget(start, device)
            print(f"generation {len(generation_rows)}/{len(items)}", flush=True)
    model.config.use_cache = False
    generations = generation_summary(generation_rows)
    generation_gate = (
        generations["pooled"]["r8_student"]["eos_terminated"] >=
        generations["pooled"]["bf16_donor"]["eos_terminated"] - 2 and
        all(generations[c]["r8_student"]["repeated_8gram_3x"] <=
            generations[c]["bf16_donor"]["repeated_8gram_3x"] + 1 and
            generations[c]["r8_student"]["early_non_eos_under16"] <=
            generations[c]["bf16_donor"]["early_non_eos_under16"] + 1
            for c in ("pooled", *M57.CATEGORIES)))
    save_partial(partial, "generations", {"document_rows": doc_rows,
        "document_summary": documents, "prompt_rows": prompt_rows,
        "prompt_summary": prompts, "generation_rows": generation_rows,
        "generation_summary": generations,
        "early_gates": {**early_gates, "generation": generation_gate}})
    if not generation_gate:
        print("METH-59 generation gate failed; stopped before task", flush=True)
        return

    task_rows = []
    with torch.inference_mode():
        M44.set_experts(wrappers, True)
        for index, item in enumerate(task_items):
            source = reference["task_rows"][index]
            assert source["index"] == index and source["label"] == item["label"]
            scores = [M21.score_option(model, item["prefix"], suffix, device)
                      for suffix in item["suffixes"]]
            selected = {"choice_mean": 0 if scores[0][1] <= scores[1][1] else 1,
                        "choice_total": 0 if scores[0][0] <= scores[1][0] else 1,
                        "options": [{"total_nll": total, "mean_nll": mean,
                                     "tokens": len(suffix)}
                                    for (total, mean), suffix in zip(scores, item["suffixes"])]}
            task_rows.append({"index": index, "label": item["label"],
                              "bf16_donor": source["donor"], "r8_student": selected})
            budget(start, device)
            if (index + 1) % 128 == 0:
                print(f"PIQA {index+1}/{len(task_items)}", flush=True)
    task = task_summary(task_rows)
    gates = {**early_gates, "generation": generation_gate,
             "task": (task["accuracy_delta"] >= -0.02 and
                      task["paired_bootstrap_lower95"] >= -0.05)}
    gates["joint_diagnostic"] = all(gates.values())
    result = {"experiment": "METH-59-Instruct-R8-core-product-key-composition",
              "core_sha256": CORE_SHA,
              "checkpoint_sha256": M57.CHECKPOINT_SHA,
              "reference_sha256": REFERENCE_SHA,
              "external_manifest_sha256": M57.EXTERNAL_SHA,
              "piqa_source_sha256": M21.PIQA_SHA,
              "piqa_labels_sha256": M21.LABELS_SHA,
              "document_rows": doc_rows, "document_summary": documents,
              "prompt_rows": prompt_rows, "prompt_summary": prompts,
              "generation_rows": generation_rows, "generation_summary": generations,
              "task_rows": task_rows, "task_summary": task,
              "gates": gates,
              "decision": "diagnostic_pass_new_independent_audit_required"
                          if gates["joint_diagnostic"] else "reject_R8_composition",
              "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps({"document_summary": documents, "prompt_summary": prompts,
                      "generation_summary": generations, "task_summary": task,
                      "gates": gates, "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
