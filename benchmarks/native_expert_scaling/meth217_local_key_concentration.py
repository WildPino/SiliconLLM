#!/usr/bin/env python3
"""Read-only diagnosis and saved replay data for METH-216's four raw failures."""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import meth216_local_child_keys as K

P, M136 = K.P, K.M136
PRIOR = P.DOC / "meth216_local_child_keys_result.json"
PRIOR_SHA = "b2669d8ad250cc6e980717c94ec95641f8469e04437c1937ec9f643351000f0b"
ROUTER_SHA = "c92d603885513ec30d2eaee36975760df243dffb30521830524e32512ec3397e"
LAYERS = (12,16,17,20)
TEMPERATURE = .0004


def capture(sequences,model,wrappers,table,start,device):
    parts = {li:[] for li in LAYERS}
    parents = {li:[] for li in LAYERS}
    shared_now = [None]
    hooks = []
    totals = {"windows":0,"tokens":0,"structural_selections":0,"sequences":len(sequences),
              "sequences_by_mode":[len(sequences),0]}
    offsets = [0]
    for li in LAYERS:
        def hook(module,inputs,_output,layer=li):
            flat = inputs[0].reshape(-1,896)
            selected = module.last_selected.long()
            assert selected.shape==(flat.shape[0],4)
            q = F.normalize(F.linear(flat.float(),module.child_projection),dim=-1)
            shared = shared_now[0]
            assert shared.shape==(flat.shape[0],)
            parts[layer].append(np.repeat(q.cpu().numpy()[~shared],4,axis=0))
            parents[layer].append(selected.cpu().numpy()[~shared].reshape(-1).astype(np.int16))
        hooks.append(wrappers[li].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            for si,sequence in enumerate(sequences):
                assert int(sequence[0])!=151644 and len(sequence)==127
                tokens = np.asarray(sequence,dtype=np.int64)
                previous = np.r_[0,tokens[:-1]]
                shared = (P.R150.shared_mask(tokens,previous,np.arange(tokens.size),table) |
                          (tokens==151644) | (tokens==151645))
                shared_now[0] = shared
                model(torch.as_tensor(tokens,device=device)[None],use_cache=False)
                totals["windows"]+=1
                totals["tokens"]+=tokens.size
                totals["structural_selections"]+=4*int(shared.sum())
                offsets.append(offsets[-1]+4*int((~shared).sum()))
                P.budget(start,device)
                if (si+1)%256==0:
                    print(json.dumps({"raw_sequences":si+1,"runtime":P.budget(start,device)}),flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    data = {li:(np.concatenate(parts[li]),np.concatenate(parents[li])) for li in LAYERS}
    totals["content_selections_per_layer"] = offsets[-1]
    for q,parent in data.values():
        assert q.shape==(offsets[-1],32) and parent.shape==(offsets[-1],)
        assert np.isfinite(q).all() and np.all((parent>=0)&(parent<1280))
    return data,np.asarray(offsets,dtype=np.int64),totals


def quantiles(values):
    return dict(zip(("min","p05","median","p95","max"),
                    map(float,np.quantile(values,(0,.05,.5,.95,1)))))


def diagnose(q,parent,keys,bias,device):
    with torch.inference_mode():
        scores = K.local_scores(q,parent,keys,device)
        parents = torch.from_numpy(parent.astype(np.int64)).to(device)
        logits = scores+torch.from_numpy(bias).to(device)[parents]
        choices = logits.argmax(-1)
        hard = torch.bincount(parents*9+choices,minlength=1280*9).view(1280,9).cpu().numpy()
        soft = torch.zeros((1280,9),device=device)
        soft.index_add_(0,parents,F.softmax(logits/TEMPERATURE,dim=-1))
        soft = soft.cpu().numpy()
        top = logits.topk(2,dim=-1).values
        margins = (top[:,0]-top[:,1]).cpu().numpy()
        dispersion = scores.std(-1,unbiased=False).cpu().numpy()
    counts = np.bincount(parent,minlength=1280)
    assert np.array_equal(hard.sum(1),counts)
    assert np.allclose(soft.sum(1),counts,atol=.05,rtol=1e-4)
    order = np.argsort(parent,kind="stable")
    boundaries = np.r_[0,counts.cumsum()]
    hot = np.flatnonzero(counts>=250)
    rows = []
    for pid in hot:
        ix = order[boundaries[pid]:boundaries[pid+1]]
        vectors = np.ascontiguousarray(q[ix])
        bytes_view = vectors.view(np.dtype((np.void,128))).ravel()
        _,multiplicity = np.unique(bytes_view,return_counts=True)
        n = int(counts[pid])
        rows.append({"parent":int(pid),"selections":n,"unique_exact_q_vectors":int(multiplicity.size),
            "largest_identical_group":int(multiplicity.max()),"exact_state_floor":float(multiplicity.max()/n),
            "hard_max_share":float(hard[pid].max()/n),"soft_max_share":float(soft[pid].max()/n)})
    failing = [r for r in rows if r["hard_max_share"]>.25]
    controls = sorted((r for r in rows if r["hard_max_share"]<=.25),key=lambda r:(-r["hard_max_share"],r["parent"]))[:2]
    details = []
    for row in failing+controls:
        pid,n = row["parent"],row["selections"]
        ix = order[boundaries[pid]:boundaries[pid+1]]
        key = np.ascontiguousarray(keys[pid])
        pair_cos = key@key.T
        off_diagonal = pair_cos[~np.eye(9,dtype=bool)]
        row_bytes = key.view(np.dtype((np.void,128))).ravel()
        reason = ("feature_or_causal_state_change_required" if row["exact_state_floor"]>.25 else
                  "final_soft_calibration_not_balanced" if row["soft_max_share"]>.25 else
                  "soft_hard_gap_needs_hard_partition")
        details.append({**row,"failing_hard_share":row["hard_max_share"]>.25,
            "classification":reason if row["hard_max_share"]>.25 else "passing_reference_parent",
            "hard_counts":hard[pid].tolist(),"soft_counts":soft[pid].tolist(),
            "biased_top_two_margin":quantiles(margins[ix]),
            "zero_margin_fraction":float((margins[ix]==0).mean()),
            "margin_le_1e_6_fraction":float((margins[ix]<=1e-6).mean()),
            "unbiased_score_std":quantiles(dispersion[ix]),
            "unique_exact_key_rows":int(np.unique(row_bytes).size),
            "off_diagonal_key_cosine":quantiles(off_diagonal)})
    return {"hot_parents":rows,"failing_parent_details":details,
            "hot_parent_count":len(rows),"failing_parent_count":len(failing),
            "temperature":TEMPERATURE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--capture",required=True,type=Path)
    args = ap.parse_args()
    assert not args.out.exists() and not args.capture.exists()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.capture.parent.mkdir(parents=True,exist_ok=True)
    started,stage,layers = time.monotonic(),"bindings",{}
    try:
        assert M136.digest(PRIOR)==PRIOR_SHA
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        router = Path(prior["router_artifact"]["path"])
        assert M136.digest(router)==ROUTER_SHA
        with np.load(router,allow_pickle=False) as archive:
            projection,keys,bias = (archive[k].copy() for k in ("projection","content_keys","parent_bias"))
        assert projection.shape==(24,32,896) and keys.shape==(24,1280,9,32) and bias.shape==(24,2,1280,9)
        failed_layers = tuple(li for li,r in enumerate(prior["cells"]["fit_raw"]["layers"]) if not r["gates"]["hot_parent_share"])
        assert failed_layers==LAYERS
        parent,child = M136.bind_inputs()
        assert all(np.array_equal(projection[li],s["child_projection"].numpy()) for li,s in enumerate(child["expert_state"]))
        source_a,source_b,prefixes = M136.load_factor_bank()
        raw_ids,chat,draws = P.D175.load(P.M175.DRAWS_SHA)
        fit_draws,_,_ = K.R.partition(draws)
        sequences,_ = P.sequences_for(fit_draws,raw_ids,chat)
        table = P.M175.load_table()
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_name(i)=="NVIDIA GeForce RTX 3060"]
        assert len(matches)==1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        P.MAX_SECONDS=M136.MAX_SECONDS=20*60
        P.MAX_RSS=M136.MAX_RSS=12*(1<<30)
        M136.MAX_GPU=P.MAX_GPU
        stage = "parity"
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
        del parent,child,source_a,source_b,reference
        gc.collect()
        stage = "raw_capture"
        data,offsets,totals = capture(sequences,model,wrappers,table,started,device)
        for key,value in totals.items():
            assert value==prior["cells"]["fit_raw"][key],key
        stage = "reconciliation_and_diagnosis"
        payload = {"sequence_offsets":offsets}
        for li,(q,parents) in data.items():
            replay = K.summarize(q,parents,np.zeros(parents.size,dtype=np.uint8),keys[li],bias[li],totals["structural_selections"],device)
            expected = prior["cells"]["fit_raw"]["layers"][li]
            assert replay["gates"]==expected["gates"]
            for key,value in replay.items():
                if key!="gates":
                    assert math.isclose(value,expected[key],rel_tol=1e-6,abs_tol=1e-6),key
            layers[str(li)] = {"baseline_reconciled":True,"replay":replay,**diagnose(q,parents,keys[li],bias[li,0],device)}
            payload[f"q_{li}"]=q
            payload[f"parent_{li}"]=parents
            P.budget(started,device)
            print(json.dumps({"diagnosed_layer":li,"failing_parents":layers[str(li)]["failing_parent_count"]}),flush=True)
        stage = "capture_export"
        assert sum(a.nbytes for a in payload.values())<950_000_000
        np.savez(args.capture,**payload)
        with np.load(args.capture,allow_pickle=False) as archive:
            assert set(archive.files)==set(payload)
            for name,expected in payload.items():
                assert np.array_equal(archive[name],expected)
        assert M136.digest(router)==ROUTER_SHA
        mechanisms = sorted(set(r["classification"] for cell in layers.values() for r in cell["failing_parent_details"] if r["failing_hard_share"]))
        result = {"experiment":"METH-217-fixed-local-key-concentration-diagnostic",
            "meth216_result_sha256":PRIOR_SHA,"router_sha256":ROUTER_SHA,
            "source_bank_sha256":M136.EXACT_SHA,"child_checkpoint_sha256":M136.CHILD_SHA,
            "draws_sha256":P.M175.DRAWS_SHA,"structural_table_sha256":P.M175.TABLE_SHA,
            "teacher_control_logit_max_abs_error":error,"totals":totals,"layers":layers,
            "capture":{"path":str(args.capture.resolve()),"sha256":M136.digest(args.capture),
                "bytes":args.capture.stat().st_size,"readback_exact":True,"layers":LAYERS},
            "observed_failure_mechanisms":mechanisms,"decision":"diagnosed_local_parent_concentration",
            "runtime":{**P.budget(started,device),"gpu":torch.cuda.get_device_name(device)},
            "scope":"Fixed fit inputs/weights; no fitting, reserved/source inference, new B or native quality/rate"}
        args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        assert args.out.stat().st_size+args.capture.stat().st_size<1_000_000_000
        print(json.dumps({"mechanisms":mechanisms,"capture":result["capture"],"runtime":result["runtime"]}),flush=True)
    except BaseException as error:
        args.out.with_suffix(".failure.json").write_text(json.dumps({"stage":stage,"error":repr(error),
            "layers":layers,"elapsed_seconds":time.monotonic()-started},indent=2)+"\n",encoding="utf-8")
        raise


if __name__=="__main__":
    main()
