#!/usr/bin/env python3
"""Cached greedy generation and exact-row proposal gate after METH-214."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as TF
from transformers import AutoModelForCausalLM

import meth214_stored_core_fresh_prediction as Q

P, L = Q.P, Q.L
import meth20_half_adapter_generation as G


def choose(model, ids, cache=None, cached=False):
    output = model.model(ids, past_key_values=cache, use_cache=cached)
    hidden = output.last_hidden_state[0,-1:]
    logits = TF.linear(hidden,model.lm_head.weight)
    return int(logits.argmax(-1)), output.past_key_values if cached else None, hidden, logits


def cache_parity(model, items, device, start):
    records = []
    with torch.inference_mode():
        for category in P.CATEGORIES:
            item = next(r for r in items if r["category"]==category)
            ids = torch.as_tensor(item["prompt_ids"],device=device)[None]
            cache = None
            mismatches, max_logit_error, steps = 0, 0.0, []
            for step in range(16):
                full, _, _, full_logits = choose(model,ids)
                current, cache, _, current_logits = choose(model,ids if step==0 else ids[:,-1:],cache,True)
                mismatches += int(full != current)
                max_logit_error = max(max_logit_error,float((full_logits.float()-current_logits.float()).abs().max()))
                steps.append({"step":step,"full_choice":full,"cached_choice":current})
                ids = torch.cat((ids,torch.as_tensor([[full]],device=device)),dim=1)
                P.budget(start,device)
            records.append({"source_id":item["source_id"],"category":category,
                "steps":steps,"choice_mismatches":mismatches,"max_logit_abs_error":max_logit_error})
    return records


def generations(model,items,tokenizer,device,start,approx,save):
    eos = model.config.eos_token_id
    assert isinstance(eos,int)
    rows = []
    with torch.inference_mode():
        for item in items:
            ids = torch.as_tensor(item["prompt_ids"],device=device)[None]
            cache, continuation, shortlist = None, [], []
            for step in range(128):
                token, cache, hidden, logits = choose(model,ids,cache,True)
                if approx is not None:
                    probe = L.D.H.score_item(hidden,model.lm_head.weight,approx)
                    assert probe["positions"] == 1
                    shortlist.append({"step":step,"full_choice":token,**probe})
                continuation.append(token)
                P.budget(start,device)
                if token==eos:
                    break
                ids = torch.as_tensor([[token]],device=device)
            row = {"source_id":item["source_id"],"category":item["category"],
                "prompt_ids_sha256":item["prompt_ids_sha256"],"continuation_ids":continuation,
                "continuation_text":tokenizer.decode(continuation,skip_special_tokens=False),
                "eos_terminated":bool(continuation) and continuation[-1]==eos,
                "early_non_eos_under16":len(continuation)<16 and (not continuation or continuation[-1]!=eos),
                "repeated_8gram_3x":G.repeated_8gram(continuation),"distinct2":G.distinct2(continuation),
                "shortlist":shortlist}
            rows.append(row)
            save(rows)
            print(json.dumps({"generated":len(rows),"total":len(items),"runtime":P.budget(start,device)}),flush=True)
            if approx is not None and any(r["misses"]["64"] or r["rerank_mismatches"]["64"] for r in shortlist):
                return rows, False
    return rows, True


def summarize(arms):
    summary = {}
    for category in ("pooled",*P.CATEGORIES):
        cells = {}
        for arm,rows in arms.items():
            group = [r for r in rows if category=="pooled" or r["category"]==category]
            assert len(group)==(24 if category=="pooled" else 8)
            cells[arm] = {"eos_terminated":sum(r["eos_terminated"] for r in group),
                "early_non_eos_under16":sum(r["early_non_eos_under16"] for r in group),
                "repeated_8gram_3x":sum(r["repeated_8gram_3x"] for r in group)}
        summary[category] = cells
    return summary


def main():
    ap = argparse.ArgumentParser()
    for key in ("manifest","anchors","prediction","out"):
        ap.add_argument("--"+key,required=True,type=Path)
    for key in ("manifest-sha","anchors-sha","prediction-sha"):
        ap.add_argument("--"+key,required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    start, stage, arms, parity = time.monotonic(), "bindings", {}, {}
    gates = {}
    def save():
        args.out.with_suffix(".partial.json").write_text(json.dumps({"stage":stage,"arms":arms,
            "parity":parity,"gates":gates,"elapsed_seconds":time.monotonic()-start},indent=2)+"\n",encoding="utf-8")
    def finish(decision, summary=None):
        result = {"experiment":"METH-215-stored-core-cached-generation", "decision":decision,
            "core_sha256":L.CORE_SHA,"manifest_sha256":args.manifest_sha,
            "anchors_sha256":args.anchors_sha,"prediction_sha256":args.prediction_sha,
            "arms":arms,"parity":parity,"summary":summary,"gates":gates,
            "decoding":"Unpenalized greedy raw BF16 head; lowest ID ties; config EOS; <=128 tokens",
            "runtime":{**P.budget(start,device),"gpu":torch.cuda.get_device_name(device)},
            "scope":"Finite cached/full and K64 generation screen; no task/blind/native quality or rate"}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(json.dumps({"decision":decision,"gates":gates,"summary":summary,"runtime":result["runtime"]}),flush=True)
    try:
        assert P.digest(args.prediction)==args.prediction_sha
        pred = json.loads(args.prediction.read_text(encoding="utf-8"))
        assert pred["decision"]=="fresh_prediction_pass_generation_next" and all(pred["gates"].values())
        assert pred["core_sha256"]==L.CORE_SHA and pred["manifest_sha256"]==args.manifest_sha
        items,parent,tokenizer = Q.bind(args.manifest,args.manifest_sha,args.anchors,args.anchors_sha)
        device = Q.setup()
        model = AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,
            dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).to(device).eval()
        L.D.H.load_centered(model,device,parent["path"])
        wrappers = [layer.mlp for layer in model.model.layers]
        for arm,enabled in ((Q.ARMS[0],False),(Q.ARMS[1],True),(Q.ARMS[2],True)):
            approx = None
            if arm==Q.ARMS[2]:
                del model,wrappers
                gc.collect()
                torch.cuda.empty_cache()
                model,wrappers,proposal,load_record = L.load_stored(device)
                approx = (proposal["codes"].float()*proposal["scales"].float()[:,None]).bfloat16()
                L.D.H.KS = (64,)
            P.M44.set_experts(wrappers,enabled)
            stage = arm+"_cache_parity"
            parity[arm] = cache_parity(model,items,device,start)
            gates[arm+"_cache_choices"] = all(r["choice_mismatches"]==0 for r in parity[arm])
            save()
            if not gates[arm+"_cache_choices"]:
                finish("cache_parity_fail_stop_generation_and_task")
                return
            stage = arm+"_generation"
            def save_rows(rows):
                arms[arm] = rows
                save()
            rows, passed = generations(model,items,tokenizer,device,start,approx,save_rows)
            arms[arm] = rows
            if approx is not None:
                gates["generated_k64_inclusion_and_rerank"] = passed
                if not passed:
                    finish("generated_k64_fail_stop_task")
                    return
        summary = summarize(arms)
        for control in Q.ARMS[:2]:
            gates[control+"_pooled_eos"] = summary["pooled"][Q.ARMS[2]]["eos_terminated"] >= summary["pooled"][control]["eos_terminated"]-2
            for metric in ("early_non_eos_under16","repeated_8gram_3x"):
                gates[control+"_"+metric] = all(summary[c][Q.ARMS[2]][metric] <= summary[c][control][metric]+1
                                              for c in ("pooled",*P.CATEGORIES))
        finish("generation_pass_task_next" if all(gates.values()) else "generation_health_fail_stop_task",summary)
    except BaseException as error:
        save()
        args.out.with_suffix(".failure.json").write_text(json.dumps({"stage":stage,"error":repr(error),
            "elapsed_seconds":time.monotonic()-start},indent=2)+"\n",encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
