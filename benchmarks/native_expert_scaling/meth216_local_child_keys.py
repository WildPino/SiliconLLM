#!/usr/bin/env python3
"""Route-only screen of local leaf keys in learned E1280 child coordinates."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import meth204_shared_content_keys as P
import meth207_causal_mode_bias as C
import meth209_expanded_raw_bias as R

M136 = P.M136
BATCH = 16384


def local_scores(q, parent, keys, device):
    out = torch.empty((parent.size,9),device=device)
    key = torch.from_numpy(keys).to(device)
    for first in range(0,parent.size,BATCH):
        last = first+BATCH
        features = torch.from_numpy(q[first:last]).to(device)
        ids = torch.from_numpy(parent[first:last].astype(np.int64)).to(device)
        out[first:last] = torch.bmm(key[ids],features[:,:,None]).squeeze(-1)
    assert bool(torch.isfinite(out).all())
    return out


def fit_keys(q,parent,layer,device):
    rng = np.random.default_rng(216216+layer)
    order = np.argsort(parent,kind="stable")
    counts = np.bincount(parent,minlength=1280)
    boundaries = np.r_[0,counts.cumsum()]
    keys = rng.normal(size=(1280,9,32)).astype(np.float32)
    keys /= np.maximum(np.linalg.norm(keys,axis=-1,keepdims=True),1e-12)
    samples, sample_parent = [], []
    for pi in range(1280):
        rows = order[boundaries[pi]:boundaries[pi+1]]
        if not rows.size:
            continue
        selected = rng.choice(rows,size=min(256,rows.size),replace=False)
        sampled = q[selected].copy()
        samples.append(sampled)
        sample_parent.append(np.full(selected.size,pi,dtype=np.int64))
        initial = [sampled[0]]
        keys[pi,0] = sampled[0]
        for child in range(1,9):
            similarity = (sampled @ np.asarray(initial).T).max(1)
            farthest = int(similarity.argmin())
            if similarity[farthest] >= 1-1e-6:
                break
            keys[pi,child] = sampled[farthest]
            initial.append(sampled[farthest])
    features = torch.from_numpy(np.concatenate(samples)).to(device)
    parents = torch.from_numpy(np.concatenate(sample_parent)).to(device)
    centers = torch.from_numpy(keys).to(device)
    initial = centers.clone()
    with torch.inference_mode():
        for _ in range(16):
            sums = torch.zeros((1280*9,32),device=device)
            counts_device = torch.zeros(1280*9,device=device,dtype=torch.int64)
            for first in range(0,parents.numel(),BATCH):
                qs, ps = features[first:first+BATCH], parents[first:first+BATCH]
                labels = torch.bmm(centers[ps],qs[:,:,None]).squeeze(-1).argmax(-1)
                ids = ps*9+labels
                sums.index_add_(0,ids,qs)
                counts_device += torch.bincount(ids,minlength=1280*9)
            updated = F.normalize(sums.view(1280,9,32),dim=-1)
            centers = torch.where(counts_device.view(1280,9,1)>0,updated,initial)
        assert bool(torch.isfinite(centers).all())
        result = centers.cpu().numpy().copy()
    return result,{"sampled_selections":int(parents.numel()),"observed_parents":int((counts>0).sum()),
        "parents_with_at_least_9_selections":int((counts>=9).sum())}


def fit_bias(q,parent,keys,device):
    scores = local_scores(q,parent,keys,device)
    parents = torch.from_numpy(parent.astype(np.int64)).to(device)
    target = torch.bincount(parents,minlength=1280).float()[:,None]/9
    bias = torch.zeros((1280,9),device=device)
    with torch.inference_mode():
        for temperature in R.A.TEMPERATURES:
            for _ in range(R.A.STEPS):
                probabilities = F.softmax((scores+bias[parents])/temperature,-1)
                counts = torch.zeros_like(bias)
                counts.index_add_(0,parents,probabilities)
                bias += temperature*torch.log((target+1)/(counts+1))
                bias -= bias.mean(-1,keepdim=True)
        assert bool(torch.isfinite(bias).all())
    return bias.cpu().numpy().copy()


def summarize(q,parent,modes,keys,biases,structural,device):
    with torch.inference_mode():
        scores = local_scores(q,parent,keys,device)
        parents = torch.from_numpy(parent.astype(np.int64)).to(device)
        mode = torch.from_numpy(modes.astype(np.int64)).to(device)
        bias = torch.from_numpy(biases).to(device)
        normalized = (scores-scores.mean(-1,keepdim=True))/scores.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)
        local = (scores+bias[mode,parents]).argmax(-1)
        pc = torch.bincount(parents,minlength=1280).cpu().numpy()
        cc = torch.bincount(parents*9+local,minlength=1280*9).cpu().numpy()
        hot = pc>=250
        share = float((cc.reshape(1280,9)[hot].max(1)/pc[hot]).max()) if hot.any() else 0.
        row = {"content_selections":int(parent.size),"structural_selections":structural,
            "content_coverage":int((cc>0).sum()),"hot_parent_count":int(hot.sum()),
            "candidate_to_control_max_load_ratio":float(9*cc.max()/pc.max()),
            "hot_parent_worst_child_share":share,
            "mean_standardized_score_advantage":float(normalized.gather(1,local[:,None]).mean()),
            "raw_argmax_agreement":float((local==scores.argmax(-1)).float().mean()),
            "bias_max_abs":float(np.abs(biases).max())}
        assert int(pc.sum())==int(cc.sum())==parent.size
        assert all(np.isfinite(value) for value in row.values())
        row["gates"] = {"load_ratio":row["candidate_to_control_max_load_ratio"]<=1.25,
            "hot_parent_share":share<=.25,"coverage":row["content_coverage"]>=4000,
            "score_advantage":row["mean_standardized_score_advantage"]>=.05,
            "raw_argmax_agreement":row["raw_argmax_agreement"]>=.15}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--router",required=True,type=Path)
    args = ap.parse_args()
    assert not args.out.exists() and not args.router.exists()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.router.parent.mkdir(parents=True,exist_ok=True)
    started,stage,cells = time.monotonic(),"bindings",{}
    try:
        parent,child = M136.bind_inputs()
        source_a,source_b,prefixes = M136.load_factor_bank()
        projections_host = np.stack([s["child_projection"].numpy() for s in child["expert_state"]])
        assert projections_host.shape==(24,32,896)
        raw_ids,chat,draws = P.D175.load(P.M175.DRAWS_SHA)
        fit_draws,old_draws,fresh_draws = R.partition(draws)
        table = P.M175.load_table()
        assert M136.digest(P.P.MANIFEST)==P.P.MANIFEST_SHA
        manifest = json.loads(P.P.MANIFEST.read_text(encoding="utf-8"))
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_name(i)=="NVIDIA GeForce RTX 3060"]
        assert len(matches)==1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        P.MAX_SECONDS = M136.MAX_SECONDS = 45*60
        M136.MAX_RSS,M136.MAX_GPU = P.MAX_RSS,P.MAX_GPU
        stage = "parity"
        model = M136.model_shell(device).eval()
        originals = [layer.mlp for layer in model.model.layers]
        teacher,_ = M136.make_wrappers(model,parent["expert_state"],child["expert_state"],source_a,source_b,prefixes,device,"teacher")
        items = json.loads(M136.PARITY.read_text(encoding="utf-8"))["items"][:8]
        with torch.inference_mode():
            reference = [model(torch.as_tensor(item["prompt_ids"],device=device)[None],use_cache=False).logits.cpu() for item in items]
        for layer,original in zip(model.model.layers,originals):
            layer.mlp = original
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        wrappers,_ = M136.make_wrappers(model,parent["expert_state"],child["expert_state"],source_a,source_b,prefixes,device,"control")
        with torch.inference_mode():
            parity = max(float((model(torch.as_tensor(item["prompt_ids"],device=device)[None],use_cache=False).logits.cpu().float()-old.float()).abs().max()) for item,old in zip(items,reference))
        assert parity==0
        assert all(np.array_equal(w.child_projection.cpu().numpy(),projections_host[li]) for li,w in enumerate(wrappers))
        projections = [w.child_projection for w in wrappers]
        del parent,child,source_a,source_b,prefixes,reference
        gc.collect()
        def collect(name,sequences,width):
            modes,seq_modes,structural = C.selection_modes(sequences,width,table)
            data,totals = P.collect(name,sequences,width,model,wrappers,projections,table,started,device)
            assert structural==totals["structural_selections"] and modes.size==totals["content_selections_per_layer"]
            totals["sequences_by_mode"] = seq_modes
            return data,modes,totals
        def evaluate(data,modes,totals,keys,bias):
            rows = [summarize(q,parents,modes,keys[li],bias[li],totals["structural_selections"],device) for li,(q,parents) in enumerate(data)]
            return {**totals,"layers":rows,"all_layer_gates_pass":all(all(r["gates"].values()) for r in rows)}
        stage = "fit_capture"
        raw_sequences,_ = P.sequences_for(fit_draws,raw_ids,chat)
        _,chat_sequences = P.sequences_for(draws[:1024],raw_ids,chat)
        raw_data,raw_modes,raw_totals = collect("fit_raw",raw_sequences,512)
        chat_data,chat_modes,chat_totals = collect("fit_chat",chat_sequences,512)
        assert raw_totals["sequences_by_mode"]==[3072,0] and chat_totals["sequences_by_mode"]==[0,1024]
        del raw_sequences,chat_sequences
        stage = "fit_local_keys_bias"
        keys,biases,fit_records = [],[],[]
        for li in range(24):
            q = np.concatenate((raw_data[li][0],chat_data[li][0]))
            parents = np.concatenate((raw_data[li][1],chat_data[li][1]))
            key,record = fit_keys(q,parents,li,device)
            del q,parents
            bias = np.stack([fit_bias(data[li][0],data[li][1],key,device) for data in (raw_data,chat_data)])
            keys.append(key)
            biases.append(bias)
            fit_records.append({"layer":li,**record})
            P.budget(started,device)
            print(json.dumps({"fitted_layers":li+1,"runtime":P.budget(started,device)}),flush=True)
        keys,biases = np.stack(keys),np.stack(biases)
        assert keys.shape==(24,1280,9,32) and biases.shape==(24,2,1280,9)
        np.savez(args.router,projection=projections_host,content_keys=keys,parent_bias=biases)
        with np.load(args.router,allow_pickle=False) as archive:
            for name,expected in (("projection",projections_host),("content_keys",keys),("parent_bias",biases)):
                assert np.array_equal(archive[name],expected)
        stage = "fit_screen"
        cells["fit_raw"] = evaluate(raw_data,raw_modes,raw_totals,keys,biases)
        cells["fit_chat"] = evaluate(chat_data,chat_modes,chat_totals,keys,biases)
        del raw_data,chat_data
        gc.collect()
        def passing(names):
            return all(cells[name]["all_layer_gates_pass"] for name in names)
        fit_pass = passing(("fit_raw","fit_chat"))
        old_pass,fresh_pass,source_pass = False,False,False
        if fit_pass:
            stage = "old_reserved"
            raw,chat_sequences = P.sequences_for(old_draws,raw_ids,chat)
            for name,sequences in (("old_raw",raw),("old_chat",chat_sequences)):
                data,modes,totals = collect(name,sequences,512)
                cells[name] = evaluate(data,modes,totals,keys,biases)
                del data
                gc.collect()
            old_pass = passing(("old_raw","old_chat"))
        if old_pass:
            stage = "new_reserved"
            raw,chat_sequences = P.sequences_for(fresh_draws,raw_ids,chat)
            for name,sequences in (("new_raw",raw),("new_chat",chat_sequences)):
                data,modes,totals = collect(name,sequences,512)
                cells[name] = evaluate(data,modes,totals,keys,biases)
                del data
                gc.collect()
            fresh_pass = passing(("new_raw","new_chat"))
        if fresh_pass:
            stage = "source_screen"
            for width in (128,512):
                name = f"source_w{width}"
                data,modes,totals = collect(name,[r["document_ids"] for r in manifest["items"]],width)
                cells[name] = evaluate(data,modes,totals,keys,biases)
                del data
                gc.collect()
            source_pass = passing(("source_w128","source_w512"))
        result = {"experiment":"METH-216-local-child-feature-keys-route-screen",
            "source_bank_sha256":M136.EXACT_SHA,"child_checkpoint_sha256":M136.CHILD_SHA,
            "draws_sha256":P.M175.DRAWS_SHA,"structural_table_sha256":P.M175.TABLE_SHA,
            "source_manifest_sha256":P.P.MANIFEST_SHA,"teacher_control_logit_max_abs_error":parity,
            "fit_records":fit_records,"fit_rule":{"raw_sequences":3072,"chat_sequences":1024,
                "max_samples_per_parent":256,"key_updates":16,"seed_base":216216,
                "temperatures":R.A.TEMPERATURES,"bias_updates_per_temperature":R.A.STEPS,
                "mode_rule":"leading token ==151644; persist across windows"},
            "router_artifact":{"path":str(args.router.resolve()),"sha256":M136.digest(args.router),
                "bytes":args.router.stat().st_size,"readback_exact":True},
            "ledger":{"local_keys_bytes":keys.nbytes,"two_mode_bias_bytes":biases.nbytes,
                "projection_copy_bytes":projections_host.nbytes,"new_selected_weight_bytes_per_token":114048,
                "extra_projection_read_if_native_feature_reused":0,"native_feature_reuse_verified":False},
            "cells":cells,"gates":{"fit":fit_pass,"old_reserved":old_pass,"new_reserved":fresh_pass,"source":source_pass},
            "decision":"local_route_screen_pass_native_cost_next" if source_pass else "local_route_screen_fail_stop",
            "runtime":{**P.budget(started,device),"gpu":torch.cuda.get_device_name(device)},
            "scope":"Changed route-only geometry; consumed development inputs; no trained new B, fresh quality or native rate"}
        args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        assert args.out.stat().st_size+args.router.stat().st_size<P.MAX_DISK
        print(json.dumps({"decision":result["decision"],"gates":result["gates"],"runtime":result["runtime"]}),flush=True)
    except BaseException as error:
        args.out.with_suffix(".failure.json").write_text(json.dumps({"stage":stage,"error":repr(error),
            "cells":cells,"elapsed_seconds":time.monotonic()-started},indent=2)+"\n",encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
