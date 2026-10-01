#!/usr/bin/env python3
"""Reuse fitted parents, but compile original-source derivatives into every child prior."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch.nn import functional as F
import meth230_source_anchored_nonlinear_pair as S

P,M,R,N=S.P,S.M,S.R,S.N
D,H=S.D,S.H
PREVIOUS=P.DOC/'meth230_source_anchored_nonlinear_result.json'
PREVIOUS_SHA='724837b3f5093e0d388477755d02f9dbae72ffdda6c10051c64aad14ed436dd7'
PREVIOUS_FILE=P.ROOT/'results/native_expert_scaling/meth230_layer12_source_nonlinear_functions.safetensors'


def child_prior(center,matrices,gate,up,down,check_autograd):
    full_jac,value=N.N.donor_tangent(center,*matrices)
    g,u=F.linear(center,gate),F.linear(center,up)
    sig=torch.sigmoid(g)
    hidden_jac=(u*(sig+g*sig*(1-sig)))[:,None]*gate+F.silu(g)[:,None]*up
    raw=full_jac-down.float()@hidden_jac
    affine=raw.bfloat16()
    phi=F.silu(g)*u
    bias=value-(F.linear(center,affine.float())+F.linear(phi,down.float()))
    error=float((N.combined_value(center,affine.float(),gate,up,down.float(),bias)-value).double().norm()/value.double().norm())
    assert torch.isfinite(affine).all() and torch.isfinite(bias).all() and error<=1e-5
    derivative={}
    if check_autograd:
        raw_bias=value-(F.linear(center,raw)+F.linear(phi,down.float()))
        with torch.enable_grad():
            jac=torch.autograd.functional.jacobian(lambda x:N.combined_value(x,raw,gate,up,down.float(),raw_bias),center,vectorize=True)
        diff=jac.double()-full_jac.double()
        derivative={'relative_frobenius':float(diff.norm()/full_jac.double().norm()),
            'max_abs_relative_to_peak':float(diff.abs().max()/full_jac.double().abs().max())}
        assert derivative['relative_frobenius']<=1e-5 and derivative['max_abs_relative_to_peak']<=1e-4
    return torch.cat((affine,down.bfloat16()),dim=1),bias,{'source_center_relative_l2':error,'source_gradient_check':derivative}


def main():
    ap=argparse.ArgumentParser()
    for key in ('checkpoint','out'): ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args(); assert not args.checkpoint.exists() and not args.out.exists()
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True); args.out.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.checkpoint.parent).free>=4*1024**3
    start,stage,progress=time.monotonic(),'bindings',{}
    def partial():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'progress':progress,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
    try:
        assert P.digest(PREVIOUS)==PREVIOUS_SHA
        old=json.loads(PREVIOUS.read_text(encoding='utf-8'))
        assert P.digest(PREVIOUS_FILE)==old['checkpoint_sha256'] and P.digest(R.CAPTURE)==old['capture_sha256']
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=30*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            x=torch.from_numpy(archive['x_bf16'].reshape(-1,D)).view(torch.bfloat16).to(device).float()
            y=torch.from_numpy(archive['y_bf16'].reshape(-1,D)).view(torch.bfloat16).to(device).float()
        with safe_open(str(PREVIOUS_FILE),framework='pt',device='cpu') as archive:
            original={name:archive.get_tensor(name) for name in archive.keys()}
        tensors=dict(original)
        for name in ('affine','down','bias'):
            tensors['e160.'+name]=torch.empty_like(original['e160.'+name])
        tensors['child_prior.affine']=torch.empty_like(original['e160.affine'])
        tensors['child_prior.bias']=torch.empty_like(original['e160.bias'])
        parents,children=original['router.parents'].to(device),original['router.children'].to(device)
        cut=M.FIT*M.SEQ; xf,yf=x[:cut],y[:cut]
        lp=R.assign_full(xf,parents); lc=torch.empty_like(lp)
        for parent in range(16):
            mask=lp==parent; lc[mask]=parent*10+R.assign_full(xf[mask],children[parent])
        counts16=torch.bincount(lp,minlength=16).tolist(); counts160=torch.bincount(lc,minlength=160).tolist()
        assert counts16==old['counts16'] and counts160==old['counts160']
        route_old=json.loads(S.ROUTE_RESULT.read_text(encoding='utf-8')); assert P.digest(S.ROUTE_RESULT)==S.ROUTE_SHA
        assert P.M17.sha(lp.cpu().numpy().astype('<i4').tobytes())==route_old['fit']['parent_labels_sha256']
        assert P.M17.sha(lc.cpu().numpy().astype('<i4').tobytes())==route_old['fit']['leaf_labels_sha256']
        duplicate_groups={}
        for arm in ('e16','e160'):
            for index,digest in enumerate(old['function_hashes'][arm]): duplicate_groups.setdefault(digest,[]).append([arm,index])
        assert [v for v in duplicate_groups.values() if len(v)>1]==[[['e16',15],['e160',156]]]
        duplicate_inputs=xf[lc==156]
        assert len(duplicate_inputs)==18 and len(torch.unique(duplicate_inputs,dim=0))==1
        diagnosis={'child':156,'states':18,'unique_input_states':1,
            'maximum_input_variance':float(duplicate_inputs.double().var(0,correction=0).max()),
            'duplicated_weight_pairs':[['e16',15],['e160',156]]}
        assert diagnosis['maximum_input_variance']==0
        matrices=[]; source_hashes={}
        with safe_open(str(source),framework='pt',device='cpu') as archive:
            for organ in ('gate','up','down'):
                name=f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight'; w=archive.get_tensor(name)
                source_hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes()); matrices.append(w.to(device).float())
        assert source_hashes==old['source_tensor_sha256']
        audits=[]; fit_sse={arm:0. for arm in ('e16','child_prior','e160')}; fit_energy=0.
        for parent in range(16):
            mask=lp==parent; xp,yp=xf[mask],yf[mask]; local_leaf=lc[mask]
            gate,up=original['input.gate'][parent].to(device).float(),original['input.up'][parent].to(device).float()
            down=original['e16.down'][parent].to(device)
            parent_w=torch.cat((original['e16.affine'][parent],original['e16.down'][parent]),dim=1).to(device)
            parent_b=original['e16.bias'][parent].to(device)
            h=F.silu(F.linear(xp,gate))*F.linear(xp,up); z=torch.cat((xp,h),dim=1)
            variance=z.double().var(0,correction=0); floor=max(float(variance.mean())*1e-4,1e-12)
            assert floor==next(a['variance_floor'] for a in old['readout_audits'] if a['arm']=='e16' and a['cell']==parent)
            variance=variance.clamp_min(floor)
            fit_sse['e16']+=float((S.readout(xp,h,parent_w,parent_b)-yp).double().square().sum())
            fit_energy+=float(yp.double().square().sum())
            for ci in range(10):
                leaf=parent*10+ci; chosen=local_leaf==leaf
                stage='source_derivative_child_prior'
                w0,b0,identity=child_prior(children[parent,ci],matrices,gate,up,down,leaf in (0,156))
                tensors['child_prior.affine'][leaf].copy_(w0[:,:D].contiguous().cpu())
                tensors['child_prior.bias'][leaf].copy_(b0.cpu())
                stage='fixed_anchored_child_fit'
                w,b,check=S.anchored_readout(z[chosen],yp[chosen],w0,b0,variance)
                S.place(tensors,'e160',leaf,w,b)
                audits.append({'child':leaf,**identity,**check})
                fit_sse['child_prior']+=float((S.readout(xp[chosen],h[chosen],w0,b0)-yp[chosen]).double().square().sum())
                fit_sse['e160']+=float((S.readout(xp[chosen],h[chosen],w,b)-yp[chosen]).double().square().sum())
                P.budget(start,device)
            progress['fit']={'completed_parents':parent+1,'children':len(audits),'runtime':P.budget(start,device)}
            partial(); print(json.dumps(progress['fit']),flush=True)
        assert fit_sse['e16']/fit_energy==old['fit_normalized_sse']['e16']
        stage='snapshot_preserved_controls_and_readback'
        unchanged=[]
        for name,value in original.items():
            if not name.startswith('e160.'):
                assert torch.equal(value,tensors[name]); unchanged.append(name)
        save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-231','source_sha256':P.M57.MODEL_SHA,
            'capture_sha256':old['capture_sha256'],'parent_snapshot_sha256':old['checkpoint_sha256'],
            'function':'source derivative child affine prior with unchanged trained parent nonlinear response'})
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
            assert set(archive.keys())==set(tensors)
            for name,value in tensors.items(): assert torch.equal(value,archive.get_tensor(name))
        hashes={arm:[P.M17.sha(tensors[arm+'.affine'][i].view(torch.uint16).numpy().tobytes()+
            tensors[arm+'.down'][i].view(torch.uint16).numpy().tobytes()) for i in range(count)] for arm,count in (('e16',16),('e160',160))}
        gates={'original_duplicate_constant_input_diagnosis':True,'unchanged_routes_selection_input_and_parents':True,
            'e16_fit_metric_exact':True,'all_child_source_value_derivative_and_solve_checks':True,
            'all_tensors_readback_exact':True,'all176_weight_pairs_distinct':len(set(hashes['e16']+hashes['e160']))==176}
        rows,summary=[],{}
        if all(gates.values()):
            stage='consumed_validation_after_all_fitting'
            del matrices,z,h,w0,parent_w
            with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
                saved={name:archive.get_tensor(name).to(device).float() for name in archive.keys() if name!='selection'}
            xv,yv=x[cut:],y[cut:]; pv=R.assign_full(xv,saved['router.parents']); cv=torch.empty_like(pv)
            hv=torch.empty((len(xv),H),device=device)
            for parent in range(16):
                mask=pv==parent
                if bool(mask.any()):
                    cv[mask]=parent*10+R.assign_full(xv[mask],saved['router.children'][parent])
                    hv[mask]=F.silu(F.linear(xv[mask],saved['input.gate'][parent]))*F.linear(xv[mask],saved['input.up'][parent])
            rotated=cv//10*10+(cv%10+1)%10
            for seq in range(M.VALID):
                span=slice(seq*M.SEQ,(seq+1)*M.SEQ); energy=float(yv[span].double().square().sum())
                assert energy==route_old['validation_rows'][seq]['energy']
                row={'validation_sequence':seq,'energy':energy,'sse':{}}
                for arm,bank,ids in (('e16','e16',pv),('child_prior','child_prior',cv),('e160','e160',cv),('e160_rotated','e160',rotated)):
                    selected=ids[span]
                    linear=torch.bmm(saved[bank+'.affine'][selected],xv[span,:,None]).squeeze(-1)
                    down_selected=saved['e16.down'][pv[span]] if bank=='child_prior' else saved[bank+'.down'][selected]
                    nonlinear=torch.bmm(down_selected,hv[span,:,None]).squeeze(-1)
                    pred=(linear+nonlinear)+saved[bank+'.bias'][selected]
                    assert torch.isfinite(pred).all(); row['sse'][arm]=float((pred-yv[span]).double().square().sum())
                rows.append(row); P.budget(start,device)
            energy=sum(r['energy'] for r in rows)
            summary={arm:{'sse':sum(r['sse'][arm] for r in rows),'normalized_sse':sum(r['sse'][arm] for r in rows)/energy}
                for arm in ('e16','child_prior','e160','e160_rotated')}
            rng=np.random.default_rng(231232); draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
            delta=np.asarray([r['sse']['e16']-r['sse']['e160'] for r in rows]); ey=np.asarray([r['energy'] for r in rows])
            gain=delta[draws].sum(1)/ey[draws].sum(1)
            summary['bootstrap']={'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),
                'seed':231232,'draws':10000,'unit':'consumed_training_corpus_window'}
            gates.update({'complete_function_nmse_le_001':summary['e160']['normalized_sse']<=.01,
                'e160_sse_le_90_percent_e16':summary['e160']['sse']<=.9*summary['e16']['sse'],
                'e160_sse_le_90_percent_rotated':summary['e160']['sse']<=.9*summary['e160_rotated']['sse'],
                'e160_sse_le_90_percent_child_prior':summary['e160']['sse']<=.9*summary['child_prior']['sse'],
                'paired_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-231-source-derivative-anchored-nonlinear-children','previous_result_sha256':PREVIOUS_SHA,
            'parent_snapshot_sha256':old['checkpoint_sha256'],'source_sha256':P.M57.MODEL_SHA,'capture_sha256':old['capture_sha256'],
            'hidden':H,'diagnosis':diagnosis,'counts16':counts16,'counts160':counts160,'unchanged_control_tensors':unchanged,
            'child_audits':audits,'function_hashes':hashes,'fit_normalized_sse':{arm:v/fit_energy for arm,v in fit_sse.items()},
            'gates':gates,'summary':summary,'validation_rows':rows,'checkpoint_sha256':P.digest(args.checkpoint),
            'checkpoint_bytes':args.checkpoint.stat().st_size,'runtime':P.budget(start,device),
            'script_sha256':P.digest(Path(__file__)),'torch_version':torch.__version__,
            'decision':'local_source_derivative_nonlinear_function_and_count_pass_native_bank_next' if all(gates.values()) else 'stop_this_source_derivative_child_anchor_geometry',
            'scope':'One-layer consumed training-window source function transfer; no full LLM quality, stored routed C/DRAM/LUT, rate or second family'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'decision':result['decision'],'summary':summary,'gates':gates,'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'elapsed_seconds':time.monotonic()-start,'progress':progress},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
