"""METH-90: external audit of grouped-R8 core with adapted E128 bank."""

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
import meth85_export_group64_r8_core as M85


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
MANIFEST = DIR / "meth90_quant_aware_external_manifest.json"
MANIFEST_SHA = "70d48b30de55c41283374223ba2b61edaacad396e6f3410c45c166bbaa885937"
SCREEN = DIR / "meth90_answerability_screen.json"
SCREEN_SHA = "3616e7145363500cd15eac999c575f59a1bf6534d67acc8c4f6507e1f7782c97"
CORE = ROOT / "results/native_expert_scaling/meth85_qwen05b_instruct_group64_r8_core.safetensors"
CORE_SHA = "c484a1130e495342d1b8644ff156cc002d0fd68fe12230604e59f35d3af6433a"
EXPORT = DIR / "meth85_group64_r8_core_export.json"
EXPORT_SHA = "e3976ec0be65e65ce6cead716711847c80a4b2f11b41d5d40a4b651cadcc301e"
ADAPTED = ROOT / "results/native_expert_scaling/meth88_quant_aware_b_e128.pt"
ADAPTED_SHA = "3e39c6557afcd072c8a0f232880f9bd3061baa937b43bc88d48ad68f709a36b5"
TRAIN_REPORT = DIR / "meth88_quant_aware_b_adaptation_result.json"
TRAIN_REPORT_SHA = "d65ac64150a82d61a9fe77d178a8e6fb0486d0d1aaf7897d925c38e3a5870087"
TRAINING = DIR / "meth56_product_key_retention_result.json"
CATEGORIES = ("code", "prose", "technical_general")
MAX_SECONDS = 30 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    gpu = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES or gpu > MAX_GPU_BYTES:
        raise RuntimeError(f"METH-90 audit budget: {elapsed:.1f}s, RSS {rss}, GPU {gpu}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": gpu}


def save_partial(path, stage, **payload):
    path.write_text(json.dumps({"experiment": "METH-90-quant-aware-direct-audit",
                                "completed_stage": stage,
                                "core_sha256": CORE_SHA,
                                "manifest_sha256": MANIFEST_SHA,
                                **payload}, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def document_summary(rows):
    result = {}
    for category in ("pooled", *CATEGORIES):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        denom = math.log(2) * sum(r["bytes"] for r in group)
        donor = sum(r["donor_nats"] for r in group) / denom
        student = sum(r["student_nats"] for r in group) / denom
        result[category] = {"documents": len(group),
                            "bytes": sum(r["bytes"] for r in group),
                            "donor_bpb": donor, "student_bpb": student,
                            "delta_bpb": student - donor}
    return result


def prompt_summary(rows):
    result = {}
    for category in ("pooled", *CATEGORIES):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        positions = sum(r["positions"] for r in group)
        matching = sum(r["matching"] for r in group)
        result[category] = {"positions": positions, "matching": matching,
                            "agreement": matching / positions}
    return result


def generation_summary(rows):
    result = {}
    for category in ("pooled", *CATEGORIES):
        group = rows if category == "pooled" else [r for r in rows if r["category"] == category]
        result[category] = {}
        for arm in ("donor", "student"):
            result[category][arm] = {
                "prompts": len(group),
                "eos_terminated": sum(r[arm]["eos_terminated"] for r in group),
                "early_non_eos_under16": sum(r[arm]["early_non_eos_under16"] for r in group),
                "repeated_8gram_3x": sum(r[arm]["repeated_8gram_3x"] for r in group),
                "mean_distinct2": sum(r[arm]["distinct2"] for r in group) / len(group)}
    return result


def task_summary(rows):
    donor = np.asarray([int(r["donor"]["choice_mean"] == r["label"])
                        for r in rows], dtype=np.int8)
    student = np.asarray([int(r["student"]["choice_mean"] == r["label"])
                          for r in rows], dtype=np.int8)
    paired = student.astype(np.int16) - donor.astype(np.int16)
    rng = np.random.default_rng(M21.BOOTSTRAP_SEED)
    draws = [float(paired[rng.integers(0, len(rows), len(rows))].mean())
             for _ in range(M21.BOOTSTRAP_DRAWS)]
    return {"items": len(rows), "donor_correct": int(donor.sum()),
            "donor_accuracy": float(donor.mean()),
            "student_correct": int(student.sum()),
            "student_accuracy": float(student.mean()),
            "accuracy_delta": float(paired.mean()),
            "paired_bootstrap_lower95": float(np.quantile(draws, 0.05)),
            "bootstrap_seed": M21.BOOTSTRAP_SEED,
            "bootstrap_draws": M21.BOOTSTRAP_DRAWS,
            "donor_correct_student_wrong": int(((donor == 1) & (student == 0)).sum()),
            "donor_wrong_student_correct": int(((donor == 0) & (student == 1)).sum())}


def score_generations(model, wrappers, items, tokenizer, device, start, arm):
    rows = []
    eos = model.config.eos_token_id
    model.config.use_cache = True
    M44.set_experts(wrappers, arm == "student")
    with torch.inference_mode():
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            output = model.generate(input_ids=ids, max_new_tokens=128,
                                    do_sample=False, pad_token_id=eos,
                                    use_cache=True)
            continuation = output[0, ids.shape[1]:].cpu().tolist()
            rows.append({"source_id": item["source_id"],
                         "category": item["category"],
                         "prompt_ids_sha256": item["prompt_ids_sha256"],
                         "continuation_ids": continuation,
                         "continuation_text": tokenizer.decode(continuation, skip_special_tokens=False),
                         "eos_terminated": bool(continuation) and continuation[-1] == eos,
                         "early_non_eos_under16": len(continuation) < 16 and
                            (not continuation or continuation[-1] != eos),
                         "repeated_8gram_3x": M20.repeated_8gram(continuation),
                         "distinct2": M20.distinct2(continuation)})
            budget(start, device)
            print(f"{arm} generation {len(rows)}/{len(items)}", flush=True)
    model.config.use_cache = False
    return rows


def score_task(model, wrappers, task_items, device, start, arm):
    rows = []
    M44.set_experts(wrappers, arm == "student")
    with torch.inference_mode():
        for index, item in enumerate(task_items):
            scores = [M21.score_option(model, item["prefix"], suffix, device)
                      for suffix in item["suffixes"]]
            rows.append({"index": index, "label": item["label"],
                         "choice_mean": 0 if scores[0][1] <= scores[1][1] else 1,
                         "choice_total": 0 if scores[0][0] <= scores[1][0] else 1,
                         "options": [{"total_nll": total, "mean_nll": mean,
                                      "tokens": len(suffix)}
                                     for (total, mean), suffix in zip(scores, item["suffixes"]) ]})
            budget(start, device)
            if (index + 1) % 128 == 0:
                print(f"{arm} PIQA {index+1}/{len(task_items)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.stem + ".partial.json")
    for path, digest in ((MANIFEST, MANIFEST_SHA), (SCREEN, SCREEN_SHA),
                         (CORE, CORE_SHA), (EXPORT, EXPORT_SHA),
                         (ADAPTED, ADAPTED_SHA), (TRAIN_REPORT, TRAIN_REPORT_SHA),
                         (TRAINING, M57.TRAINING_SHA)):
        assert M15.M13.sha256(path) == digest, path
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    export = json.loads(EXPORT.read_text(encoding="utf-8"))
    assert screen["answerable_count"] == 24
    assert export["sha256"] == CORE_SHA
    assert export["ideal_addressed_bytes"]["total"] == 557106176
    items = manifest["items"]
    assert len(items) == 24
    assert [r["source_id"] for r in screen["rows"]] == [r["source_id"] for r in items]
    for item in items:
        assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        for field in ("document_ids", "prompt_ids"):
            assert M17.sha(np.asarray(item[field], dtype=np.int32).tobytes()) == item[field + "_sha256"]
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    checkpoint = training["checkpoints"]["512"]
    assert checkpoint["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(checkpoint["path"]) == M57.CHECKPOINT_SHA
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors",
                             revision=M42.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M57.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    _, task_items = M21.bind_data(tokenizer)
    assert len(task_items) == 1838
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
    doc_rows = []
    for item in items:
        nats = M17.score_doc(model, item["document_ids"], wrappers, False, device, start)
        doc_rows.append({"source_id": item["source_id"], "category": item["category"],
                         "bytes": item["bytes"],
                         "document_ids_sha256": item["document_ids_sha256"],
                         "donor_nats": nats})
        budget(start, device)
        print(f"donor documents {len(doc_rows)}/{len(items)}", flush=True)
    donor_top = {}
    with torch.inference_mode():
        M44.set_experts(wrappers, False)
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            donor_top[item["source_id"]] = model(ids, use_cache=False).logits.argmax(dim=-1).cpu()
            budget(start, device)
    donor_generations = score_generations(model, wrappers, items, tokenizer, device, start, "donor")
    donor_tasks = score_task(model, wrappers, task_items, device, start, "donor")
    save_partial(partial, "donor", document_rows=doc_rows,
                 donor_generation_rows=donor_generations,
                 donor_task_rows=donor_tasks,
                 runtime=budget(start, device))

    with safe_open(str(CORE), framework="pt", device="cpu") as archive:
        metadata = archive.metadata()
        assert metadata["format"] == "QWEN25_INSTRUCT_R8H_GROUP64_R8FFN_BF16ATTN_V1"
        assert metadata["model_sha256"] == M57.MODEL_SHA
        assert metadata["product_key_checkpoint_sha256"] == M57.CHECKPOINT_SHA
        assert len(archive.keys()) == 363
        matrix_count = control_count = 0
        for name, param in original_params.items():
            organ = M24.classify(name, tuple(param.shape))
            if organ == "control":
                stored = archive.get_tensor(name)
                assert stored.dtype == torch.float32
                assert torch.equal(stored.to(torch.bfloat16), param.detach().cpu())
                control_count += 1
            elif organ == "attention":
                stored = archive.get_tensor(name)
                assert stored.dtype == torch.bfloat16
                assert torch.equal(stored, param.detach().cpu())
                matrix_count += 1
            elif organ == "tied_head":
                codes = archive.get_tensor(name + ".q")
                scale = archive.get_tensor(name + ".scale")
                assert codes.dtype == torch.int8 and tuple(codes.shape) == tuple(param.shape)
                assert scale.dtype == torch.float32 and tuple(scale.shape) == (param.shape[0],)
                param.copy_((codes.to(device).float() *
                             scale.to(device).unsqueeze(1)).to(torch.bfloat16))
                matrix_count += 1
            else:
                assert organ == "ffn"
                codes = archive.get_tensor(name + ".q")
                scale = archive.get_tensor(name + ".scale")
                assert codes.dtype == torch.int8 and tuple(codes.shape) == tuple(param.shape)
                assert scale.dtype == torch.float16
                param.copy_(M85.reconstruct_ffn(codes.to(device), scale.to(device)))
                matrix_count += 1
        assert matrix_count == 169 and control_count == 121
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    adapted = torch.load(ADAPTED, map_location="cpu", weights_only=False)
    assert adapted["updates"] == 64 and adapted["parent_updates"] == 512
    assert adapted["core_sha256"] == CORE_SHA
    assert adapted["parent_checkpoint_sha256"] == M57.CHECKPOINT_SHA
    assert adapted["trainable"] == "B_only"
    with torch.no_grad():
        for wrapper, saved in zip(wrappers, adapted["expert_state"]):
            assert torch.equal(wrapper.a.cpu(), saved["a"])
            assert torch.equal(wrapper.router.cpu(), saved["router"])
            wrapper.b.copy_(saved["b"].to(device))
    del adapted
    budget(start, device)
    for row, item in zip(doc_rows, items):
        row["student_nats"] = M17.score_doc(model, item["document_ids"], wrappers, True, device, start)
        budget(start, device)
        print(f"student documents {len([r for r in doc_rows if 'student_nats' in r])}/{len(items)}", flush=True)
    documents = document_summary(doc_rows)
    document_gate = documents["pooled"]["delta_bpb"] <= 0.02 and all(
        documents[c]["delta_bpb"] <= 0.04 for c in CATEGORIES)
    prompt_rows = []
    with torch.inference_mode():
        M44.set_experts(wrappers, True)
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
            top = model(ids, use_cache=False).logits.argmax(dim=-1).cpu()
            donor = donor_top[item["source_id"]]
            assert top.shape == donor.shape
            prompt_rows.append({"source_id": item["source_id"],
                                "category": item["category"],
                                "prompt_ids_sha256": item["prompt_ids_sha256"],
                                "positions": int(top.numel()),
                                "matching": int((top == donor).sum())})
            budget(start, device)
    prompts = prompt_summary(prompt_rows)
    save_partial(partial, "documents_and_prompts", document_rows=doc_rows,
                 document_summary=documents, prompt_rows=prompt_rows,
                 prompt_summary=prompts, donor_generation_rows=donor_generations,
                 donor_task_rows=donor_tasks, document_gate=document_gate,
                 runtime=budget(start, device))
    print(json.dumps({"documents": documents, "prompts": prompts,
                      "document_gate": document_gate}), flush=True)
    if not document_gate:
        result = {"experiment": "METH-90-quant-aware-direct-audit",
                  "core_sha256": CORE_SHA, "manifest_sha256": MANIFEST_SHA,
                  "document_rows": doc_rows, "document_summary": documents,
                  "prompt_rows": prompt_rows, "prompt_summary": prompts,
                  "gates": {"documents": False},
                  "decision": "stop_before_student_generation_task_document_gate_failed",
                  "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    student_generations = score_generations(model, wrappers, items, tokenizer, device, start, "student")
    assert [r["source_id"] for r in donor_generations] == [r["source_id"] for r in student_generations]
    generation_rows = []
    for donor, student in zip(donor_generations, student_generations):
        assert donor["prompt_ids_sha256"] == student["prompt_ids_sha256"]
        generation_rows.append({"source_id": donor["source_id"], "category": donor["category"],
                                "prompt_ids_sha256": donor["prompt_ids_sha256"],
                                "donor": {k: v for k, v in donor.items() if k not in
                                          ("source_id", "category", "prompt_ids_sha256")},
                                "student": {k: v for k, v in student.items() if k not in
                                            ("source_id", "category", "prompt_ids_sha256")}})
    generations = generation_summary(generation_rows)
    generation_gate = (
        generations["pooled"]["student"]["eos_terminated"] >=
        generations["pooled"]["donor"]["eos_terminated"] - 2 and
        all(generations[c]["student"]["repeated_8gram_3x"] <=
            generations[c]["donor"]["repeated_8gram_3x"] + 1 and
            generations[c]["student"]["early_non_eos_under16"] <=
            generations[c]["donor"]["early_non_eos_under16"] + 1
            for c in ("pooled", *CATEGORIES)))
    save_partial(partial, "generations", document_rows=doc_rows,
                 document_summary=documents, prompt_rows=prompt_rows,
                 prompt_summary=prompts, generation_rows=generation_rows,
                 generation_summary=generations,
                 donor_task_rows=donor_tasks,
                 gates={"documents": document_gate, "generation": generation_gate},
                 runtime=budget(start, device))
    if not generation_gate:
        result = {"experiment": "METH-90-quant-aware-direct-audit",
                  "core_sha256": CORE_SHA, "manifest_sha256": MANIFEST_SHA,
                  "document_rows": doc_rows, "document_summary": documents,
                  "prompt_rows": prompt_rows, "prompt_summary": prompts,
                  "generation_rows": generation_rows,
                  "generation_summary": generations,
                  "gates": {"documents": document_gate, "generation": False},
                  "decision": "stop_before_student_task_generation_gate_failed",
                  "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    student_tasks = score_task(model, wrappers, task_items, device, start, "student")
    assert len(donor_tasks) == len(student_tasks) == 1838
    task_rows = []
    for donor, student in zip(donor_tasks, student_tasks):
        assert donor["index"] == student["index"] and donor["label"] == student["label"]
        task_rows.append({"index": donor["index"], "label": donor["label"],
                          "donor": {k: v for k, v in donor.items() if k not in ("index", "label")},
                          "student": {k: v for k, v in student.items() if k not in ("index", "label")}})
    task = task_summary(task_rows)
    gates = {"documents": document_gate, "generation": generation_gate,
             "task": task["accuracy_delta"] >= -0.02 and
                     task["paired_bootstrap_lower95"] >= -0.05}
    gates["joint_automatic"] = all(gates.values())
    result = {"experiment": "METH-90-quant-aware-direct-audit",
              "core_sha256": CORE_SHA,
              "checkpoint_sha256": M57.CHECKPOINT_SHA,
              "adapted_checkpoint_sha256": ADAPTED_SHA,
              "source_sha256": M57.MODEL_SHA,
              "manifest_sha256": MANIFEST_SHA,
              "answerability_sha256": SCREEN_SHA,
              "piqa_source_sha256": M21.PIQA_SHA,
              "piqa_labels_sha256": M21.LABELS_SHA,
              "document_rows": doc_rows, "document_summary": documents,
              "prompt_rows": prompt_rows, "prompt_summary": prompts,
              "generation_rows": generation_rows,
              "generation_summary": generations,
              "task_rows": task_rows, "task_summary": task,
              "gates": gates,
              "decision": "automatic_pass_blind_semantic_review_pending"
                          if gates["joint_automatic"] else "reject_quant_aware_core_direct_quality",
              "runtime": {**budget(start, device),
                          "gpu": torch.cuda.get_device_name(device),
                          "cuda_index": matches[0], "torch": torch.__version__}}
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps({"document_summary": documents,
                      "prompt_summary": prompts,
                      "generation_summary": generations,
                      "task_summary": task, "gates": gates,
                      "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
