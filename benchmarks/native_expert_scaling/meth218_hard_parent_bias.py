#!/usr/bin/env python3
"""Hard-count correction of five fixed raw-mode bias vectors, then staged gates."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

import meth217_local_key_concentration as D

K,P,M136 = D.K,D.P,D.M136
DIAGNOSTIC = P.DOC / "meth217_local_key_concentration_result.json"
DIAGNOSTIC_SHA = "25c79ebe507cb5650fbd434983ec888d678ef3166181f3f50728cf894edcc561"
CAPTURE_SHA = "ce3d11eec323e251deba08f7e70aef15d527f6881e3f35763b0ebcf2b4c541db"
PAIRS = ((12,359),(16,600),(16,889),(17,158),(20,786))


def hard_counts(scores,bias):
    assert scores.dtype==bias.dtype==np.float32
    return np.bincount((scores+bias).argmax(1),minlength=9)


def calibrate(scores,initial):
    n = scores.shape[0]
    target = np.full(9,n//9,dtype=np.int64)
    target[:n%9]+=1
    bias = initial.copy()
    before = hard_counts(scores,bias)
    trace = []
    converged = False
    for sweep in range(1,201):
        for child in range(9):
            logits = scores+bias
            logits[:,child] = -np.inf
            thresholds = logits.max(1).astype(np.float64)-scores[:,child].astype(np.float64)
            thresholds.sort(kind="stable")
            rank = int(target[child])
            midpoint = np.float32((thresholds[rank-1]+thresholds[rank])/2)
            candidates = (np.nextafter(midpoint,np.float32(-np.inf)),midpoint,
                          np.nextafter(midpoint,np.float32(np.inf)))
            options = []
            for value in candidates:
                trial = bias.copy()
                trial[child]=value
                count = int(hard_counts(scores,trial)[child])
                options.append((abs(count-rank),float(value),value))
            bias[child]=min(options,key=lambda x:(x[0],x[1]))[2]
        bias = (bias.astype(np.float64)-bias.astype(np.float64).mean()).astype(np.float32)
        counts = hard_counts(scores,bias)
        trace.append({"sweep":sweep,"counts":counts.tolist(),"max_share":float(counts.max()/n),
                      "min_share":float(counts.min()/n)})
        if counts.max()/n<=.15 and counts.min()/n>=.05:
            converged=True
            break
    assert np.isfinite(bias).all()
    return bias,{"selections":n,"target_counts":target.tolist(),"before_counts":before.tolist(),
        "after_counts":counts.tolist(),"converged":converged,"sweeps":trace,
        "bias_delta_max_abs":float(np.abs(bias-initial).max())}


def setup_model(device):
    parent,child = M136.bind_inputs()
    source_a,source_b,prefixes = M136.load_factor_bank()
    model = M136.model_shell(device).eval()
    originals = [layer.mlp for layer in model.model.layers]
    teacher,_ = M136.make_wrappers(model,parent["expert_state"],child["expert_state"],source_a,source_b,prefixes,device,"teacher")
    items = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
    with torch.inference_mode():
        reference = [model(torch.as_tensor(r["prompt_ids"],device=device)[None],use_cache=False).logits.cpu() for r in items]
    for layer,original in zip(model.model.layers,originals):
        layer.mlp=original
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    wrappers,_ = M136.make_wrappers(model,parent["expert_state"],child["expert_state"],source_a,source_b,prefixes,device,"control")
    with torch.inference_mode():
        error = max(float((model(torch.as_tensor(r["prompt_ids"],device=device)[None],use_cache=False).logits.cpu().float()-old.float()).abs().max()) for r,old in zip(items,reference))
    assert error==0
    return model,wrappers,error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--router",required=True,type=Path)
    args = ap.parse_args()
    assert not args.out.exists() and not args.router.exists()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.router.parent.mkdir(parents=True,exist_ok=True)
    started,stage,cells,records = time.monotonic(),"bindings",{},[]
    try:
        assert M136.digest(D.PRIOR)==D.PRIOR_SHA
        assert M136.digest(DIAGNOSTIC)==DIAGNOSTIC_SHA
        prior = json.loads(D.PRIOR.read_text(encoding="utf-8"))
        diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
        old_router = Path(prior["router_artifact"]["path"])
        capture = Path(diagnostic["capture"]["path"])
        assert M136.digest(old_router)==D.ROUTER_SHA and M136.digest(capture)==CAPTURE_SHA
        found = tuple((int(li),r["parent"]) for li,cell in diagnostic["layers"].items()
                      for r in cell["failing_parent_details"] if r["failing_hard_share"])
        assert found==PAIRS and diagnostic["observed_failure_mechanisms"]==["soft_hard_gap_needs_hard_partition"]
        with np.load(old_router,allow_pickle=False) as archive:
            projection,keys,bias = (archive[k].copy() for k in ("projection","content_keys","parent_bias"))
        original_bias = bias.copy()
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_name(i)=="NVIDIA GeForce RTX 3060"]
        assert len(matches)==1
        device=torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        P.MAX_SECONDS=M136.MAX_SECONDS=25*60
        P.MAX_RSS=M136.MAX_RSS=12*(1<<30)
        M136.MAX_GPU=P.MAX_GPU
        cells = json.loads(json.dumps(prior["cells"]))
        stage="hard_fit_and_four_layer_screen"
        with np.load(capture,allow_pickle=False) as archive:
            assert archive["sequence_offsets"][-1]==prior["cells"]["fit_raw"]["content_selections_per_layer"]
            for li in D.LAYERS:
                q,parent = archive[f"q_{li}"],archive[f"parent_{li}"]
                scores = K.local_scores(q,parent,keys[li],device).cpu().numpy()
                for layer,pid in PAIRS:
                    if layer!=li:
                        continue
                    selected = scores[parent==pid].copy()
                    old = next(r for r in diagnostic["layers"][str(li)]["failing_parent_details"] if r["parent"]==pid)
                    assert np.array_equal(hard_counts(selected,bias[li,0,pid]),old["hard_counts"])
                    bias[li,0,pid],record = calibrate(selected,bias[li,0,pid])
                    records.append({"layer":li,"parent":pid,"numpy_torch_original_counts_equal":True,**record})
                row = K.summarize(q,parent,np.zeros(parent.size,dtype=np.uint8),keys[li],bias[li],
                    prior["cells"]["fit_raw"]["structural_selections"],device)
                cells["fit_raw"]["layers"][li]=row
                P.budget(started,device)
                del q,parent,scores
        cells["fit_raw"]["all_layer_gates_pass"] = all(all(r["gates"].values()) for r in cells["fit_raw"]["layers"])
        cells["fit_raw"]["unchanged_layer_metrics_reused_from"] = D.PRIOR_SHA
        cells["fit_raw"]["changed_layers_recomputed"] = list(D.LAYERS)
        cells["fit_chat"]["metrics_reused_from_byte_identical_router"] = D.PRIOR_SHA
        changed = np.any(bias!=original_bias,axis=-1)
        expected_changed = np.zeros((24,2,1280),dtype=bool)
        for li,pid in PAIRS:
            expected_changed[li,0,pid]=True
        assert np.array_equal(changed,expected_changed)
        assert np.array_equal(bias[:,1],original_bias[:,1])
        stage="router_export"
        np.savez(args.router,projection=projection,content_keys=keys,parent_bias=bias)
        with np.load(args.router,allow_pickle=False) as archive:
            for name,expected in (("projection",projection),("content_keys",keys),("parent_bias",bias)):
                assert np.array_equal(archive[name],expected)
        fit_pass = all(r["converged"] for r in records) and all(cells[n]["all_layer_gates_pass"] for n in ("fit_raw","fit_chat"))
        old_pass,new_pass,source_pass,parity=False,False,False,None
        print(json.dumps({"fit_pass":fit_pass,"calibration":records,"runtime":P.budget(started,device)}),flush=True)
        gc.collect()
        if fit_pass:
            stage="model_parity"
            model,wrappers,parity=setup_model(device)
            assert all(np.array_equal(w.child_projection.cpu().numpy(),projection[li]) for li,w in enumerate(wrappers))
            projections=[w.child_projection for w in wrappers]
            raw_ids,chat,draws=P.D175.load(P.M175.DRAWS_SHA)
            _,old_draws,new_draws=K.R.partition(draws)
            table=P.M175.load_table()
            def evaluate(name,sequences,width):
                modes,counts,structural=K.C.selection_modes(sequences,width,table)
                data,totals=P.collect(name,sequences,width,model,wrappers,projections,table,started,device)
                assert structural==totals["structural_selections"] and modes.size==totals["content_selections_per_layer"]
                rows=[K.summarize(q,parents,modes,keys[li],bias[li],structural,device) for li,(q,parents) in enumerate(data)]
                cells[name]={**totals,"sequences_by_mode":counts,"layers":rows,
                    "all_layer_gates_pass":all(all(r["gates"].values()) for r in rows)}
                del data
                gc.collect()
                P.budget(started,device)
            def passing(names):
                return all(cells[n]["all_layer_gates_pass"] for n in names)
            stage="old_reserved"
            raw,chats=P.sequences_for(old_draws,raw_ids,chat)
            evaluate("old_raw",raw,512)
            evaluate("old_chat",chats,512)
            old_pass=passing(("old_raw","old_chat"))
            if old_pass:
                stage="new_reserved"
                raw,chats=P.sequences_for(new_draws,raw_ids,chat)
                evaluate("new_raw",raw,512)
                evaluate("new_chat",chats,512)
                new_pass=passing(("new_raw","new_chat"))
            if new_pass:
                stage="source_screen"
                assert M136.digest(P.P.MANIFEST)==P.P.MANIFEST_SHA
                items=json.loads(P.P.MANIFEST.read_text(encoding="utf-8"))["items"]
                for width in (128,512):
                    evaluate(f"source_w{width}",[r["document_ids"] for r in items],width)
                source_pass=passing(("source_w128","source_w512"))
        result={"experiment":"METH-218-targeted-hard-local-parent-bias",
            "meth216_result_sha256":D.PRIOR_SHA,"meth217_result_sha256":DIAGNOSTIC_SHA,
            "input_router_sha256":D.ROUTER_SHA,"fit_capture_sha256":CAPTURE_SHA,
            "source_bank_sha256":M136.EXACT_SHA,"child_checkpoint_sha256":M136.CHILD_SHA,
            "draws_sha256":P.M175.DRAWS_SHA,"structural_table_sha256":P.M175.TABLE_SHA,
            "teacher_control_logit_max_abs_error":parity,"calibration":records,
            "only_five_raw_bias_vectors_changed":True,"keys_projection_chat_biases_byte_identical":True,
            "router_artifact":{"path":str(args.router.resolve()),"sha256":M136.digest(args.router),
                "bytes":args.router.stat().st_size,"readback_exact":True},
            "ledger":prior["ledger"],"cells":cells,
            "gates":{"fit":fit_pass,"old_reserved":old_pass,"new_reserved":new_pass,"source":source_pass},
            "decision":"hard_local_route_screen_pass_native_cost_next" if source_pass else "hard_local_route_screen_fail_stop",
            "runtime":{**P.budget(started,device),"gpu":torch.cuda.get_device_name(device)},
            "scope":"Targeted hard fit and consumed route-development gates; no new B/native or fresh quality"}
        args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        assert args.out.stat().st_size+args.router.stat().st_size<1_000_000_000
        print(json.dumps({"decision":result["decision"],"gates":result["gates"],"runtime":result["runtime"]}),flush=True)
    except BaseException as error:
        args.out.with_suffix(".failure.json").write_text(json.dumps({"stage":stage,"error":repr(error),
            "calibration":records,"cells":cells,"elapsed_seconds":time.monotonic()-started},indent=2)+"\n",encoding="utf-8")
        raise


if __name__=="__main__":
    main()
