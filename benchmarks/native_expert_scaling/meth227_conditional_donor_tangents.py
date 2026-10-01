#!/usr/bin/env python3
"""Frozen E16/E160 common-free full-input source-Jacobian function pair."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch.nn import functional as F
import meth222_conditional_function_pilot as M
import meth226_donor_tangent_feasibility as N

P=M.P
SEED,WIDTH=227227,896
NATIVE=P.DOC/'meth226_donor_tangent_native_result.json'
NATIVE_SHA='c8356317ce5d6499660e43b23c09251a2a4196128a60fa2035a15b84c9615667'
PREVIOUS=P.DOC/'meth222_conditional_function_pilot_result.json'
PREVIOUS_SHA='41858ede5147f7c98821987892096acb0aa7abf3611db4a1b849663e1d0ddb34'
CAPTURE=P.ROOT/'results/native_expert_scaling/meth222_layer12_training_states.npz'


def assign_full(x,centers):
    # Direct difference avoids cancellation of squared norms. Lowest ID wins ties.
    return torch.cat([((x[first:first+2048,None,:]-centers[None,:,:]).square().sum(-1)).argmin(-1)
                      for first in range(0,len(x),2048)])


def kmeans_full(x,k,seed,start,device):
    rng=np.random.default_rng(seed)
    centers=x[torch.as_tensor(rng.choice(len(x),k,replace=False),device=device)].clone()
    for _ in range(20):
        labels=assign_full(x,centers)
        for ci in range(k):
            local=x[labels==ci]
            if len(local): centers[ci]=local.mean(0)
        P.budget(start,device)
    return centers,assign_full(x,centers)


def build_bank(centers,matrices,arm,start,device):
    weight=torch.empty((len(centers),WIDTH,WIDTH),dtype=torch.bfloat16)
    bias=torch.empty((len(centers),WIDTH),dtype=torch.float32)
    audits=[]
    for ci,center in enumerate(centers):
        jacobian,value=N.donor_tangent(center,*matrices)
        derivative={}
        if ci==0:
            with torch.enable_grad():
                reference=torch.autograd.functional.jacobian(lambda x:N.donor_value(x,*matrices),center,vectorize=True)
            difference=jacobian.double()-reference.double()
            derivative={'relative_frobenius':float(difference.norm()/reference.double().norm()),
                'max_abs_relative_to_peak':float(difference.abs().max()/reference.double().abs().max())}
            assert derivative['relative_frobenius']<=1e-5 and derivative['max_abs_relative_to_peak']<=1e-4
        rounded=jacobian.bfloat16()
        intercept=value-F.linear(center,rounded.float())
        error=float((F.linear(center,rounded.float(),intercept)-value).double().norm()/value.double().norm())
        assert torch.isfinite(jacobian).all() and torch.isfinite(intercept).all() and error<=1e-5
        weight[ci].copy_(rounded.cpu()); bias[ci].copy_(intercept.cpu())
        audits.append({'arm':arm,'cell':ci,'center_relative_l2':error,
            'weight_sha256':P.M17.sha(weight[ci].view(torch.uint16).numpy().tobytes()),
            'bias_sha256':P.M17.sha(bias[ci].numpy().tobytes()),'autograd':derivative})
        P.budget(start,device)
    return weight,bias,audits


def radius_summary(x,centers,labels):
    distance=(x-centers[labels]).square().sum(-1)
    fraction=distance/x.square().sum(-1).clamp_min(1e-20)
    return {'mean_squared_distance':float(distance.double().mean()),
        'median_fraction_of_input_squared_norm':float(fraction.median()),
        'p95_fraction_of_input_squared_norm':float(torch.quantile(fraction,.95))}


def main():
    ap=argparse.ArgumentParser()
    for key in ('checkpoint','out'): ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args()
    assert not args.checkpoint.exists() and not args.out.exists()
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True); args.out.parent.mkdir(parents=True,exist_ok=True)
    start,stage,progress=time.monotonic(),'bindings',{}
    def partial():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'progress':progress,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
    try:
        assert P.digest(NATIVE)==NATIVE_SHA and P.digest(PREVIOUS)==PREVIOUS_SHA
        native=json.loads(NATIVE.read_text(encoding='utf-8')); old=json.loads(PREVIOUS.read_text(encoding='utf-8'))
        assert all(native['gates'].values()) and P.digest(CAPTURE)==old['capture']['sha256']
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=25*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        with np.load(CAPTURE,allow_pickle=False) as archive:
            x=torch.from_numpy(archive['x_bf16'].reshape(-1,WIDTH)).view(torch.bfloat16).to(device).float()
            y=torch.from_numpy(archive['y_bf16'].reshape(-1,WIDTH)).view(torch.bfloat16).to(device).float()
            raw_rows,offsets=archive['row_ids'].copy(),archive['offsets'].copy()
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        cut=M.FIT*M.SEQ; xf=x[:cut]
        stage='full_input_hierarchical_centroids'
        parents,lp=kmeans_full(xf,16,SEED,start,device)
        counts16=torch.bincount(lp,minlength=16).tolist()
        children=torch.zeros((16,10,WIDTH),device=device)
        lc=torch.full_like(lp,-1)
        for parent in range(16):
            mask=lp==parent
            if counts16[parent]>=10:
                children[parent],local=kmeans_full(xf[mask],10,SEED+1+parent,start,device)
                lc[mask]=parent*10+local
            progress['clustering']={'completed_parents':parent+1,'counts16':counts16}
            partial()
        counts160=torch.bincount(lc[lc>=0],minlength=160).tolist()
        gates={'all_parents_have_at_least_10_fit_states':min(counts16)>=10,'all160_cells_occupied':min(counts160)>0}
        fit={'counts16':counts16,'counts160':counts160,
            'parent_labels_sha256':P.M17.sha(lp.cpu().numpy().astype('<i4').tobytes()),
            'leaf_labels_sha256':P.M17.sha(lc.cpu().numpy().astype('<i4').tobytes())}
        tensors={'router.parents':parents.cpu().contiguous(),'router.children':children.cpu().contiguous()}
        audits=[]
        if all(gates.values()):
            stage='compile_original_donor_tangents'
            matrices=[]; source_hashes={}
            with safe_open(str(source),framework='pt',device='cpu') as archive:
                for organ in ('gate','up','down'):
                    name=f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight'
                    w=archive.get_tensor(name)
                    assert w.dtype==torch.bfloat16 and w.shape==((WIDTH,4864) if organ=='down' else (4864,WIDTH))
                    source_hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes())
                    matrices.append(w.to(device).float())
            fit['source_tensor_sha256']=source_hashes
            for arm,centers in (('e16',parents),('e160',children.reshape(160,WIDTH))):
                w,b,checks=build_bank(centers,matrices,arm,start,device)
                tensors[arm+'.weight'],tensors[arm+'.bias']=w,b
                audits.extend(checks)
                progress['compiled_functions']={'count':len(audits)}; partial()
            gates['all176_functions_have_distinct_weight_bits']=len({r['weight_sha256'] for r in audits})==176
            del matrices
        save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-227','layer':str(M.LAYER),
            'source_sha256':P.M57.MODEL_SHA,'capture_sha256':old['capture']['sha256'],
            'function':'full-input BF16 Jacobian and FP32 bias; direct donor Taylor approximation'})
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
            assert set(archive.keys())==set(tensors)
            for name,value in tensors.items(): assert torch.equal(value,archive.get_tensor(name))
        gates['every_stored_tensor_readback_equal']=True
        rows,summary,distances=[],{},{}
        stage='consumed_validation' if all(gates.values()) else 'fit_gate_stop'
        if all(gates.values()):
            fit['distance']={'e16':radius_summary(xf,parents,lp),
                'e160':radius_summary(xf,children.reshape(160,WIDTH),lc)}
            with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
                saved={name:archive.get_tensor(name).to(device).float() for name in archive.keys()}
            xv,yv=x[cut:],y[cut:]
            pv=assign_full(xv,saved['router.parents']); cv=torch.empty_like(pv)
            for parent in range(16):
                mask=pv==parent
                if bool(mask.any()): cv[mask]=parent*10+assign_full(xv[mask],saved['router.children'][parent])
            rotated=(cv//10)*10+(cv%10+1)%10
            distances={'e16':radius_summary(xv,saved['router.parents'],pv),
                'e160':radius_summary(xv,saved['router.children'].reshape(160,WIDTH),cv)}
            for seq in range(M.VALID):
                span=slice(seq*M.SEQ,(seq+1)*M.SEQ)
                energy=float(yv[span].double().square().sum())
                assert energy==old['validation_rows'][seq]['energy']
                row={'validation_sequence':seq,'raw_row':int(raw_rows[M.FIT+seq]),'offset':int(offsets[M.FIT+seq]),
                    'energy':energy,'sse':{}}
                for arm,bank,labels in (('e16','e16',pv),('e160','e160',cv),('e160_rotated','e160',rotated)):
                    selected=labels[span]
                    predicted=torch.bmm(saved[bank+'.weight'][selected],xv[span,:,None]).squeeze(-1)+saved[bank+'.bias'][selected]
                    assert torch.isfinite(predicted).all()
                    row['sse'][arm]=float((predicted-yv[span]).double().square().sum())
                rows.append(row); P.budget(start,device)
            total_energy=sum(r['energy'] for r in rows)
            summary={arm:{'sse':sum(r['sse'][arm] for r in rows),
                'normalized_sse':sum(r['sse'][arm] for r in rows)/total_energy} for arm in ('e16','e160','e160_rotated')}
            rng=np.random.default_rng(SEED+1); draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
            delta=np.asarray([r['sse']['e16']-r['sse']['e160'] for r in rows])
            energy=np.asarray([r['energy'] for r in rows]); gain=delta[draws].sum(1)/energy[draws].sum(1)
            summary['bootstrap']={'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),
                'seed':SEED+1,'draws':10000,'unit':'consumed_training_corpus_window'}
            gates.update({'full_function_normalized_sse_le_001':summary['e160']['normalized_sse']<=.01,
                'e160_at_least_10_percent_less_sse_than_e16':summary['e160']['sse']<=.9*summary['e16']['sse'],
                'e160_at_least_10_percent_less_sse_than_rotated':summary['e160']['sse']<=.9*summary['e160_rotated']['sse'],
                'paired_window_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-227-conditional-full-input-donor-tangents','layer':M.LAYER,'seed':SEED,
            'source_sha256':P.M57.MODEL_SHA,'source_revision':P.M42.REV,'capture_sha256':old['capture']['sha256'],
            'native_result_sha256':NATIVE_SHA,'previous_result_sha256':PREVIOUS_SHA,'fit_sequences':M.FIT,
            'validation_sequences':M.VALID,'tokens_per_sequence':M.SEQ,'lloyd_steps':20,'active_functions':1,
            'fit':fit,'function_audits':audits,'gates':gates,'summary':summary,'validation_rows':rows,
            'validation_distance':distances,'checkpoint_sha256':P.digest(args.checkpoint),
            'checkpoint_bytes':args.checkpoint.stat().st_size,'runtime':P.budget(start,device),
            'script_sha256':P.digest(Path(__file__)),'torch_version':torch.__version__,
            'decision':'local_tangent_function_and_n_pass_native_routed_composition_next' if all(gates.values()) else 'stop_this_fixed_conditional_tangent_geometry',
            'scope':'One source layer, consumed training windows; no independent LLM quality, causal composition, n-scaled CPU/LUT/DRAM, full rate or second family'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'decision':result['decision'],'gates':gates,'summary':summary,'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'elapsed_seconds':time.monotonic()-start,'progress':progress},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
