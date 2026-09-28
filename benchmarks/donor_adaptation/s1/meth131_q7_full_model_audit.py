#!/usr/bin/env python3
"""Fresh automatic BF16 E1280 versus stored Q7-child-B model quality audit."""

import argparse
import json
import math
from pathlib import Path
import struct
import time

from huggingface_hub import hf_hub_download
import numpy as np
import psutil
import torch

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth21_half_adapter_piqa as M21
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57
import meth90_quant_aware_external_audit as M90
import meth95_hierarchical_e1280_parity as M95


ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
ART = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling"
MANIFEST = DOC / "meth131_q7_external_manifest.json"
MANIFEST_SHA = "84d8dbb400be133b6d1e440d55da5d09afd8ebbe0bd659474bdad8e708868ba9"
ANSWERABILITY = DOC / "meth131_answerability_screen.json"
ANSWERABILITY_SHA = "95480c4e6dd4a621502264f59ca9dc565134e776c18014025535fba2033cf911"
TRAINING = DOC / "meth56_product_key_retention_result.json"
CHILD = ART / "meth107_long_chat_e1280.pt"
CHILD_SHA = "15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520"
OLD_BANK = ART / "meth126_shared_a_factor_bank.bin"
OLD_SHA = "1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1"
Q_BANK = ART / "meth130_q7_shared_a_factor_bank.bin"
Q_SHA = "20329a07f7dfcd3bfee08d4ee64b7e243d0004f4e18ec3543c78516c203b0265"
HEADER = struct.Struct("<8s8I")
CATEGORIES = ("code", "prose", "technical_general")
ARMS = ("exact_e1280", "q7_e1280")
MAX_SECONDS = 35 * 60
MAX_GPU = int(10.5 * (1 << 30))
MAX_RSS = 20 * (1 << 30)


def budget(start, device):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss,
              "gpu_peak_bytes": torch.cuda.max_memory_allocated(device)}
    if result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS or result["gpu_peak_bytes"] > MAX_GPU:
        raise RuntimeError(f"METH-131 resource stop: {result}")
    return result


def write(path, stage, payload, start, device):
    payload.update({"experiment": "METH-131-Q7-E1280-fresh-full-model-audit",
                    "completed_stage": stage, "source_sha256": M57.MODEL_SHA,
                    "parent_checkpoint_sha256": M57.CHECKPOINT_SHA,
                    "child_checkpoint_sha256": CHILD_SHA,
                    "bf16_bank_sha256": OLD_SHA, "q7_bank_sha256": Q_SHA,
                    "manifest_sha256": MANIFEST_SHA,
                    "answerability_sha256": ANSWERABILITY_SHA,
                    "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device),
                                "torch": torch.__version__}})
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summaries(doc_rows, prompt_rows):
    documents, prompts = {}, {}
    for category in ("pooled", *CATEGORIES):
        ds = doc_rows if category == "pooled" else [r for r in doc_rows if r["category"] == category]
        ps = prompt_rows if category == "pooled" else [r for r in prompt_rows if r["category"] == category]
        assert len(ds) == len(ps) == (24 if category == "pooled" else 8)
        amount = sum(r["bytes"] for r in ds)
        bpb = {arm: sum(r["nats"][arm] for r in ds) / (math.log(2) * amount) for arm in ARMS}
        positions = sum(r["positions"] for r in ps)
        agreement = {arm: sum(r["matching"][arm] for r in ps) / positions for arm in ARMS}
        documents[category] = {"documents": len(ds), "bytes": amount,
                               "bpb": bpb, "q7_minus_exact": bpb["q7_e1280"]-bpb["exact_e1280"]}
        prompts[category] = {"prompts": len(ps), "positions": positions,
                             "donor_top1_agreement": agreement,
                             "q7_minus_exact": agreement["q7_e1280"]-agreement["exact_e1280"]}
    return documents, prompts


def verify_and_install_q7(children, start, device):
    old = np.memmap(OLD_BANK, dtype=np.uint8, mode="r")
    quant = np.memmap(Q_BANK, dtype=np.uint8, mode="r")
    old_header = HEADER.unpack_from(old)
    q_header = HEADER.unpack_from(quant)
    dims = (24, 896, 64, 8, 16, 32, 10, 8)
    assert old_header == (b"M126FB01", *dims)
    assert q_header == (b"M130FB01", *dims)
    layers, width, rank, na, nb, cr, nchild, r = dims
    e = na*nb*nchild
    rows = e*width
    parent_router_bytes = (rank*width+(na+nb)*rank)*4
    projection_bytes = cr*width*4
    keys_bytes = e*cr*4
    router_bytes = parent_router_bytes+projection_bytes+keys_bytes
    a_bytes = na*nb*r*width*2
    old_layer = router_bytes+a_bytes+rows*r*2
    q_layer = router_bytes+a_bytes+rows*8
    assert len(old) == HEADER.size+layers*old_layer
    assert len(quant) == HEADER.size+layers*q_layer
    maximum_weight_error = 0.0
    numerator = 0.0
    denominator = 0.0
    for li, wrapper in enumerate(children):
        old_base = HEADER.size+li*old_layer
        q_base = HEADER.size+li*q_layer
        assert np.array_equal(old[old_base:old_base+router_bytes+a_bytes],
                              quant[q_base:q_base+router_bytes+a_bytes])
        cursor = old_base
        for tensor, length in ((wrapper.router, parent_router_bytes),
                               (wrapper.child_projection, projection_bytes),
                               (wrapper.child_keys, keys_bytes)):
            actual = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
            assert len(actual) == length and actual == old[cursor:cursor+length].tobytes(), (li, cursor)
            cursor += length
        a_ref = np.frombuffer(old, dtype="<u2", count=(a_bytes//2),
                              offset=old_base+router_bytes).reshape(na*nb,r,width)
        a_actual = wrapper.a.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16)
        a_actual = a_actual.numpy().reshape(na*nb,nchild,r,width)
        assert np.array_equal(a_ref, a_actual[:, 0])
        assert np.all(a_actual == a_actual[:, :1])
        b_ref = np.frombuffer(old, dtype="<u2", count=rows*r,
                              offset=old_base+router_bytes+a_bytes).reshape(rows,r)
        b_actual = wrapper.b.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16)
        assert np.array_equal(b_ref, b_actual.numpy().reshape(rows,r)), li
        record = np.frombuffer(quant, dtype=np.uint8, count=rows*8,
                               offset=q_base+router_bytes+a_bytes).reshape(rows,8)
        scales = record[:, :4].copy().view("<f4").reshape(rows)
        packed = record[:, 4:]
        low = packed & 15
        high = packed >> 4
        assert np.all(low <= 6) and np.all(high <= 6)
        assert np.isfinite(scales).all() and np.all(scales >= 0)
        codes = np.empty((rows,r), dtype=np.float32)
        codes[:, 0::2] = low.astype(np.float32)-3
        codes[:, 1::2] = high.astype(np.float32)-3
        dequant = codes*scales[:, None]
        bf = (b_ref.astype("<u4") << 16).view("<f4")
        delta = bf-dequant
        numerator += float(np.sum(delta.astype(np.float64)**2))
        denominator += float(np.sum(bf.astype(np.float64)**2))
        maximum_weight_error = max(maximum_weight_error, float(np.max(np.abs(delta))))
        with torch.no_grad():
            wrapper.b.copy_(torch.from_numpy(dequant.reshape(wrapper.b.shape)).to(device))
        budget(start, device)
    return {"all_router_A_B_rows_verified": True,
            "q7_dequant_weight_relative_l2": math.sqrt(numerator/denominator),
            "q7_dequant_weight_max_abs_error": maximum_weight_error,
            "note": "PyTorch BF16 matmul after Q7 dequantization is a quality proxy; native pair-LUT parity is separate"}


def install_exact_b(children,start,device):
    old = np.memmap(OLD_BANK,dtype=np.uint8,mode="r")
    rows=1280*896
    router=((64*896+(8+16)*64)+32*896+1280*32)*4
    a_bytes=128*8*896*2
    layer_bytes=router+a_bytes+rows*8*2
    for li,wrapper in enumerate(children):
        offset=HEADER.size+li*layer_bytes+router+a_bytes
        bits=np.frombuffer(old,dtype="<u2",count=rows*8,offset=offset)
        values=(bits.astype("<u4")<<16).view("<f4").reshape(wrapper.b.shape)
        with torch.no_grad(): wrapper.b.copy_(torch.from_numpy(values).to(device))
        budget(start,device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.stem+".partial.json")
    for path, digest in ((MANIFEST, MANIFEST_SHA), (ANSWERABILITY, ANSWERABILITY_SHA),
                         (TRAINING, M57.TRAINING_SHA), (CHILD, CHILD_SHA),
                         (OLD_BANK, OLD_SHA), (Q_BANK, Q_SHA)):
        assert M15.M13.sha256(path) == digest, path
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = manifest["items"]
    assert len(items) == 24 and manifest["selected_counts"] == dict.fromkeys(CATEGORIES,8)
    answerability = json.loads(ANSWERABILITY.read_text(encoding="utf-8"))
    assert answerability["answerable_count"] == 24
    assert [r["source_id"] for r in answerability["rows"]] == [r["source_id"] for r in items]
    for item in items:
        assert M17.sha(item["text"].encode("utf-8")) == item["text_sha256"]
        for key in ("document_ids", "prompt_ids"):
            assert M17.sha(np.asarray(item[key], dtype=np.int32).tobytes()) == item[key+"_sha256"]
    parent = json.loads(TRAINING.read_text(encoding="utf-8"))["checkpoints"]["512"]
    assert parent["sha256"] == M57.CHECKPOINT_SHA
    assert M15.M13.sha256(parent["path"]) == M57.CHECKPOINT_SHA
    source = hf_hub_download(M42.MODEL,"model.safetensors",revision=M42.REV,
                             local_files_only=True)
    assert M15.M13.sha256(source) == M57.MODEL_SHA
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    _, task_items = M21.bind_data(tok)
    assert len(task_items) == 1838
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    gpu = [i for i in range(torch.cuda.device_count())
           if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(gpu) == 1
    device = torch.device(f"cuda:{gpu[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    M90.MAX_SECONDS=MAX_SECONDS
    M17.MAX_SECONDS=MAX_SECONDS
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        M42.MODEL, revision=M42.REV, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.config.use_cache = False
    saved_parent = torch.load(parent["path"], map_location="cpu", weights_only=False)
    assert saved_parent["updates"] == 512 and saved_parent["source_sha256"] == M57.MODEL_SHA
    parents = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
            layer.mlp = wrapper
            for key in ("a","b","router"):
                getattr(wrapper,key).copy_(saved_parent["expert_state"][li][key].to(device))
            parents.append(wrapper)
    saved_child = torch.load(CHILD, map_location="cpu", weights_only=False)
    assert saved_child["updates"] == 256 and saved_child["source_sha256"] == M57.MODEL_SHA
    assert saved_child["parent_checkpoint_sha256"] == M57.CHECKPOINT_SHA
    children = []
    centered_mean_error = []
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            wrapper = M95.HierarchicalExperts(parents[li],li).to(device)
            saved = saved_child["expert_state"][li]
            for key in ("a","b","router","child_projection","child_keys"):
                assert getattr(wrapper,key).shape == saved[key].shape
                getattr(wrapper,key).copy_(saved[key].to(device))
            raw = wrapper.b.view(M15.E,M95.CHILDREN,M15.M13.D,M15.R)
            mean = raw.mean(dim=1,keepdim=True)
            wrapper.b.copy_((parents[li].b.detach()[:,None]+raw-mean).reshape_as(wrapper.b))
            centered_mean_error.append(float((wrapper.b.view(M15.E,M95.CHILDREN,
                M15.M13.D,M15.R).mean(dim=1)-parents[li].b.detach()).abs().max()))
            children.append(wrapper)
    del saved_child, saved_parent
    assert max(centered_mean_error) <= 1e-7

    def activate(wrappers):
        for layer,wrapper in zip(model.model.layers,wrappers): layer.mlp=wrapper

    doc_rows = [{"source_id": x["source_id"],"category":x["category"],
                 "bytes":x["bytes"],"nats":{}} for x in items]
    prompt_rows = [{"source_id": x["source_id"],"category":x["category"],
                    "positions":len(x["prompt_ids"]),"matching":{}} for x in items]
    donor_top = {}
    activate(parents)
    with torch.inference_mode():
        M44.set_experts(parents,False)
        for row,item in zip(prompt_rows,items):
            ids=torch.as_tensor(item["prompt_ids"],dtype=torch.long,device=device)[None]
            donor_top[item["source_id"]]=model(ids,use_cache=False).logits.argmax(-1)[0].cpu()
            budget(start,device)

    def evaluate_documents_and_prompts(arm):
        activate(children)
        for row,item in zip(doc_rows,items):
            row["nats"][arm]=M17.score_doc(model,item["document_ids"],children,True,device,start)
            budget(start,device)
        with torch.inference_mode():
            M44.set_experts(children,True)
            for row,item in zip(prompt_rows,items):
                ids=torch.as_tensor(item["prompt_ids"],dtype=torch.long,device=device)[None]
                top=model(ids,use_cache=False).logits.argmax(-1)[0].cpu()
                row["matching"][arm]=int((top==donor_top[item["source_id"]]).sum())
                budget(start,device)
        print(json.dumps({"completed_arm":arm,"runtime":budget(start,device)}),flush=True)

    evaluate_documents_and_prompts("exact_e1280")
    binding = verify_and_install_q7(children,start,device)
    evaluate_documents_and_prompts("q7_e1280")
    docs,prompts=summaries(doc_rows,prompt_rows)
    gates={"pooled_bpb":docs["pooled"]["q7_minus_exact"]<=0.005,
           "category_bpb":all(docs[c]["q7_minus_exact"]<=0.02 for c in CATEGORIES),
           "pooled_prompt_top1":prompts["pooled"]["q7_minus_exact"]>=-0.01,
           "category_prompt_top1":all(prompts[c]["q7_minus_exact"]>=-0.02 for c in CATEGORIES)}
    payload={"binding":binding,"centered_parent_mean_max_abs_error_by_layer":centered_mean_error,
             "document_rows":doc_rows,"prompt_rows":prompt_rows,
             "document_summary":docs,"prompt_summary":prompts,"gates":gates}
    write(partial,"documents_and_prompts",payload,start,device)
    if not all(gates.values()):
        payload["decision"]="stop_document_prompt_gate"
        write(args.out,"documents_and_prompts",payload,start,device)
        print(json.dumps({"decision":payload["decision"],"gates":gates,
                          "documents":docs,"prompts":prompts},indent=2),flush=True)
        return

    generations={}
    install_exact_b(children,start,device)
    activate(children)
    generations["exact_e1280"]=M90.score_generations(
        model,children,items,tok,device,start,"student")
    verify_and_install_q7(children,start,device)
    generations["q7_e1280"]=M90.score_generations(
        model,children,items,tok,device,start,"student")
    generation_rows=[]
    for i,item in enumerate(items):
        generation_rows.append({"source_id":item["source_id"],"category":item["category"],
                                "prompt_ids_sha256":item["prompt_ids_sha256"],
                                **{arm:generations[arm][i] for arm in ARMS}})
    comparison=[{"category":r["category"],"donor":r["exact_e1280"],
                 "student":r["q7_e1280"]} for r in generation_rows]
    gen_summary=M90.generation_summary(comparison)
    gen_gate=(gen_summary["pooled"]["student"]["eos_terminated"] >=
              gen_summary["pooled"]["donor"]["eos_terminated"]-2 and
              all(gen_summary[c]["student"]["repeated_8gram_3x"] <=
                  gen_summary[c]["donor"]["repeated_8gram_3x"]+1 and
                  gen_summary[c]["student"]["early_non_eos_under16"] <=
                  gen_summary[c]["donor"]["early_non_eos_under16"]+1
                  for c in ("pooled",*CATEGORIES)))
    gates["generation"]=gen_gate
    payload.update({"generation_rows":generation_rows,"generation_summary":gen_summary})
    write(partial,"generation",payload,start,device)
    if not gen_gate:
        payload["decision"]="stop_generation_gate"
        write(args.out,"generation",payload,start,device)
        print(json.dumps({"decision":payload["decision"],"gates":gates,
                          "generation_summary":gen_summary},indent=2),flush=True)
        return

    task_arms={}
    install_exact_b(children,start,device)
    activate(children)
    task_arms["exact_e1280"]=M90.score_task(model,children,task_items,device,start,"student")
    verify_and_install_q7(children,start,device)
    task_arms["q7_e1280"]=M90.score_task(model,children,task_items,device,start,"student")
    task_rows=[]
    for i in range(len(task_items)):
        task_rows.append({"index":i,"label":task_items[i]["label"],
                          **{arm:task_arms[arm][i] for arm in ARMS}})
    task_comparison=[{"label":r["label"],"donor":r["exact_e1280"],
                      "student":r["q7_e1280"]} for r in task_rows]
    task_summary=M90.task_summary(task_comparison)
    gates["task"]=(task_summary["accuracy_delta"]>=-0.02 and
                   task_summary["paired_bootstrap_lower95"]>=-0.05)
    gates["joint_automatic"]=all(gates.values())
    payload.update({"task_rows":task_rows,"task_summary":task_summary,
                    "piqa_source_sha256":M21.PIQA_SHA,
                    "piqa_labels_sha256":M21.LABELS_SHA,
                    "decision":"automatic_pass_blind_semantic_review_pending"
                    if gates["joint_automatic"] else "stop_task_gate"})
    write(args.out,"all_automatic",payload,start,device)
    partial.unlink(missing_ok=True)
    print(json.dumps({"decision":payload["decision"],"gates":gates,
                      "documents":docs,"prompts":prompts,
                      "generation_summary":gen_summary,"task_summary":task_summary,
                      "runtime":payload["runtime"]},indent=2),flush=True)


if __name__=="__main__":
    main()
