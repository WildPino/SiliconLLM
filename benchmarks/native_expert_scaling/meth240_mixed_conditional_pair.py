#!/usr/bin/env python3
"""Fit actual fixed32 mixed-output E16/E160 functions over full source features."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import meth239_mixed_output_priors as Q

P,M,R,N=Q.P,Q.M,Q.R,Q.N
D,H,TAU=896,4864,1024.
PRIOR=P.DOC/'meth239_mixed_output_prior_result.json'
PRIOR_SHA='e6e72a440dc18e01411ab85ce79ec3f4e1bfe7e8aa9b1c80a3de43342a437a30'
PRIOR_FILE=P.ROOT/'results/native_expert_scaling/meth239_layer12_mixed_output_priors.safetensors'
FIELDS=('down.q','down.scale','down.ids','down.escape','bias')


def get(tensors,arm,index,device):
    return tuple(tensors[arm+'.'+name][index].to(device) for name in FIELDS)


def place(tensors,arm,index,coeff):
    for name,value in zip(FIELDS,coeff): tensors[arm+'.'+name][index].copy_(value.cpu())


def decode(coeff):
    return Q.L.decode_mixed(*coeff[:4])


def readout(phi,coeff):
    return Q.L.mixed_linear(phi,*coeff[:4])+coeff[4]


def anchored_fit(z,y,prior,variance):
    zd,yd=z.double(),y.double(); wp,bp=decode(prior).double(),prior[4].double()
    mean=zd.mean(0); xc=zd-mean
    residual=yd-(zd@wp.T+bp); mr=residual.mean(0); rc=residual-mr
    n,features=xc.shape
    if n<features:
        scaled=xc/variance[None,:]
        gram=scaled@xc.T+TAU*torch.eye(n,dtype=torch.float64,device=z.device)
        delta_t=scaled.T@torch.linalg.solve(gram,rc); method='dual'
    else:
        gram=xc.T@xc+TAU*torch.diag(variance)
        delta_t=torch.linalg.solve(gram,xc.T@rc); method='primal'
    cross=xc.T@rc
    normal=xc.T@(xc@delta_t-rc)+TAU*variance[:,None]*delta_t
    relative=float(normal.norm()/cross.norm().clamp_min(1e-12))
    assert torch.isfinite(delta_t).all() and relative<=1e-7
    q,s,ids,escape=Q.L.encode_mixed((wp+delta_t.T).float())
    effective=Q.L.decode_mixed(q,s,ids,escape).double()
    shift=mr*(n/(n+TAU)); bias=(bp+shift-(effective-wp)@mean).float()
    error=float(((effective@mean+bias.double())-(wp@mean+bp)-shift).norm()/yd.mean(0).norm().clamp_min(1e-12))
    assert torch.isfinite(bias).all() and error<=1e-5
    return (q,s,ids,escape,bias),{'states':n,'features':features,'solve':method,
        'normal_equation_relative_residual':relative,'mean_shift_rounding_relative_error':error,
        'intercept_shrink':n/(n+TAU)}


def prior_checks(audit):
    complete=audit['complete_unrounded_autograd']; feature=audit['feature_derivatives']
    return audit['condition_number']<=1e8 and audit['relative_coefficient_change']<=.25 and \
        audit['unrounded_gradient_relative_frobenius']<=1e-5 and audit['raw32_center_relative_l2']<=1e-5 and \
        audit['stored_center_relative_l2']<=1e-5 and audit['stored_gradient_relative_frobenius']<=.01 and \
        (not complete or (complete['relative_frobenius']<=1e-5 and complete['max_relative_to_peak']<=1e-4)) and \
        (not feature or (feature['fixed16_row_relative_frobenius']<=1e-5 and all(d['relative_l2']<=1e-5 for d in feature['directions'])))


def predict_bank(phi,labels,tensors,bank,device):
    out=torch.empty((len(phi),D),dtype=torch.float32,device=device)
    for index in range(len(tensors[bank+'.bias'])):
        mask=labels==index
        if bool(mask.any()): out[mask]=readout(phi[mask],get(tensors,bank,index,device))
    assert torch.isfinite(out).all()
    return out


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
        assert P.digest(PRIOR)==PRIOR_SHA and P.digest(Q.NATIVE)==Q.NATIVE_SHA
        old=json.loads(PRIOR.read_text()); native=json.loads(Q.NATIVE.read_text())
        assert all(old['gates'].values()) and all(native['gates'].values())
        assert P.digest(PRIOR_FILE)==old['checkpoint_sha256'] and P.digest(R.CAPTURE)==old['capture_sha256']
        assert P.digest(Q.S.ROUTE_RESULT)==Q.S.ROUTE_SHA
        route=json.loads(Q.S.ROUTE_RESULT.read_text())
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=30*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        values,controls=Q.load_layer(native,device); assert controls==old['controls']
        with safe_open(str(PRIOR_FILE),framework='pt',device='cpu') as archive:
            original={name:archive.get_tensor(name) for name in archive.keys()}
        tensors={('parent_prior.'+name[6:] if name.startswith('prior.') else name):t for name,t in original.items()}
        for arm,count in (('e16',16),('child_prior',160),('e160',160)):
            for field in FIELDS:
                example=original['prior.'+field]
                tensors[arm+'.'+field]=torch.empty((count,*example.shape[1:]),dtype=example.dtype)
        tensors['fit.feature_variance']=torch.empty((16,H),dtype=torch.float64)
        parents=tensors['router.parents'].to(device); children=tensors['router.children'].to(device)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            xf=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,D).copy()).view(torch.bfloat16).to(device).float()
            yf=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,D).copy()).view(torch.bfloat16).to(device).float()
        lp=R.assign_full(xf,parents); lc=torch.empty_like(lp)
        for parent in range(16):
            mask=lp==parent; lc[mask]=parent*10+R.assign_full(xf[mask],children[parent])
        counts16=torch.bincount(lp,minlength=16).tolist(); counts160=torch.bincount(lc,minlength=160).tolist()
        assert counts16==old['fit_counts']==route['fit']['counts16'] and counts160==route['fit']['counts160']
        assert P.M17.sha(lp.cpu().numpy().astype('<i4').tobytes())==route['fit']['parent_labels_sha256']
        assert P.M17.sha(lc.cpu().numpy().astype('<i4').tobytes())==route['fit']['leaf_labels_sha256']
        matrices=[]; source_hashes={}
        with safe_open(str(source),framework='pt',device='cpu') as archive:
            for organ in ('gate','up','down'):
                name=f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight'; w=archive.get_tensor(name)
                source_hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes()); matrices.append(w.to(device).float())
        assert source_hashes==old['source_tensor_sha256']
        fit_sse={arm:0. for arm in ('parent_prior','e16','child_prior','e160')}
        fit_energy=float(yf.double().square().sum()); fit_audits=[]; child_audits=[]
        stage='fixed_parent_output_fit'
        for parent in range(16):
            mask=lp==parent; phi=Q.features(xf[mask],values); y=yf[mask]
            variance=phi.double().var(0,correction=0); floor=max(float(variance.mean())*1e-4,1e-12)
            variance=variance.clamp_min(floor); tensors['fit.feature_variance'][parent].copy_(variance.cpu())
            prior=get(tensors,'parent_prior',parent,device)
            coeff,audit=anchored_fit(phi,y,prior,variance); place(tensors,'e16',parent,coeff)
            fit_audits.append({'arm':'e16','cell':parent,'variance_floor':floor,**audit})
            for arm,c in (('parent_prior',prior),('e16',coeff)):
                fit_sse[arm]+=float((readout(phi,c)-y).double().square().sum())
            progress={'parent_fits':fit_audits}; partial(); P.budget(start,device)
        del phi,prior,coeff,variance,y
        stage='all160_source_derivative_child_priors_before_child_fit'
        for leaf in range(160):
            parent=leaf//10; inherited=get(tensors,'e16',parent,device)
            q,s,ids,escape,bias,audit=Q.projected_prior(children[parent,leaf%10],matrices,values,
                decode(inherited),leaf in (0,156))
            place(tensors,'child_prior',leaf,(q,s,ids,escape,bias))
            child_audits.append({'child':leaf,**audit})
            progress={'parent_fits':fit_audits,'child_priors':child_audits}; partial(); P.budget(start,device)
        child_gate=all(prior_checks(a) for a in child_audits)
        if child_gate:
            stage='fixed_160_child_output_fit'
            for parent in range(16):
                mask=lp==parent; phi=Q.features(xf[mask],values); y=yf[mask]; local=lc[mask]
                variance=tensors['fit.feature_variance'][parent].to(device)
                for child in range(10):
                    leaf=parent*10+child; chosen=local==leaf; prior=get(tensors,'child_prior',leaf,device)
                    coeff,audit=anchored_fit(phi[chosen],y[chosen],prior,variance); place(tensors,'e160',leaf,coeff)
                    fit_audits.append({'arm':'e160','cell':leaf,**audit})
                    for arm,c in (('child_prior',prior),('e160',coeff)):
                        fit_sse[arm]+=float((readout(phi[chosen],c)-y[chosen]).double().square().sum())
                    P.budget(start,device)
                progress={'parent_fits_and_child_fits':fit_audits,'child_priors':child_audits}; partial()
            del phi,prior,coeff,variance,y
        else:
            # Preserve an honest prerequisite-only snapshot; unfitted E160 is omitted.
            for field in FIELDS: del tensors['e160.'+field]
        stage='physical_snapshot_readback'
        save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-240','source_sha256':P.M57.MODEL_SHA,
            'layer':str(M.LAYER),'hidden':str(H),'prior_samples':str(TAU),'child_prior_gate':str(child_gate)})
        assert args.checkpoint.stat().st_size<1700000000
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
            assert set(archive.keys())==set(tensors)
            for name,t in tensors.items(): assert torch.equal(archive.get_tensor(name),t)
        hashes={}; physical_hashes={}
        for arm,count in (('e16',16),('e160',160)):
            if not child_gate and arm=='e160': continue
            hashes[arm]=[]; physical_hashes[arm]=[]
            for index in range(count):
                c=get(tensors,arm,index,device)
                hashes[arm].append(P.M17.sha(decode(c).cpu().numpy().tobytes()))
                physical_hashes[arm].append(P.M17.sha(b''.join((t.view(torch.uint16) if t.dtype==torch.bfloat16 else t).cpu().numpy().tobytes() for t in c[:4])))
        gates={'source_native_capture_route_bindings':True,'all_fit_labels_counts_exact':True,
            'all160_source_child_prior_gates':child_gate,'all_snapshot_tensors_readback_exact':True,
            'all176_readout_solves_and_intercepts':child_gate and len(fit_audits)==176,
            'all176_effective_weights_distinct':child_gate and len(set(hashes.get('e16',[])+hashes.get('e160',[])))==176}
        rows=[]; summary={}
        if all(gates.values()):
            stage='consumed_validation_once_after_all_fitting'
            del matrices,xf,yf,inherited
            with np.load(R.CAPTURE,allow_pickle=False) as archive:
                xv=torch.from_numpy(archive['x_bf16'][M.FIT:].reshape(-1,D).copy()).view(torch.bfloat16).to(device).float()
                yv=torch.from_numpy(archive['y_bf16'][M.FIT:].reshape(-1,D).copy()).view(torch.bfloat16).to(device).float()
            pv=R.assign_full(xv,parents); cv=torch.empty_like(pv)
            for parent in range(16):
                mask=pv==parent
                if bool(mask.any()): cv[mask]=parent*10+R.assign_full(xv[mask],children[parent])
            rotated=cv//10*10+(cv%10+1)%10
            hv=Q.features(xv,values)
            predictions={}
            for arm,bank,labels in (('parent_prior','parent_prior',pv),('e16','e16',pv),
                ('child_prior','child_prior',cv),('e160','e160',cv),('e160_rotated','e160',rotated)):
                predictions[arm]=predict_bank(hv,labels,tensors,bank,device); P.budget(start,device)
            for sequence in range(M.VALID):
                span=slice(sequence*M.SEQ,(sequence+1)*M.SEQ); energy=float(yv[span].double().square().sum())
                assert energy==route['validation_rows'][sequence]['energy']
                rows.append({'validation_sequence':sequence,'energy':energy,
                    'sse':{arm:float((pred[span]-yv[span]).double().square().sum()) for arm,pred in predictions.items()}})
            energy=sum(r['energy'] for r in rows)
            summary={arm:{'sse':sum(r['sse'][arm] for r in rows),'normalized_sse':sum(r['sse'][arm] for r in rows)/energy} for arm in predictions}
            rng=np.random.default_rng(240241); draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
            delta=np.asarray([r['sse']['e16']-r['sse']['e160'] for r in rows]); ey=np.asarray([r['energy'] for r in rows])
            gain=delta[draws].sum(1)/ey[draws].sum(1)
            summary['bootstrap']={'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),
                'seed':240241,'draws':10000,'unit':'consumed_training_corpus_window'}
            gates.update({'complete_function_nmse_le_001':summary['e160']['normalized_sse']<=.01,
                'e160_sse_le_90_percent_e16':summary['e160']['sse']<=.9*summary['e16']['sse'],
                'e160_sse_le_90_percent_rotated':summary['e160']['sse']<=.9*summary['e160_rotated']['sse'],
                'e160_sse_le_90_percent_child_prior':summary['e160']['sse']<=.9*summary['child_prior']['sse'],
                'e160_sse_le_90_percent_parent_prior':summary['e160']['sse']<=.9*summary['parent_prior']['sse'],
                'paired_window_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-240-full-feature-fixed32-mixed-output-E16-E160-pair','layer':M.LAYER,'hidden':H,
            'prior_result_sha256':PRIOR_SHA,'native_result_sha256':Q.NATIVE_SHA,'source_sha256':P.M57.MODEL_SHA,
            'source_tensor_sha256':source_hashes,'controls':controls,'capture_sha256':old['capture_sha256'],
            'route_result_sha256':Q.S.ROUTE_SHA,'counts16':counts16,'counts160':counts160,
            'fit_unique_states':M.FIT*M.SEQ,'readout_fit_state_exposures':(2 if child_gate else 1)*M.FIT*M.SEQ,
            'prior_sample_equivalents':TAU,'child_prior_audits':child_audits,'fit_audits':fit_audits,
            'effective_function_hashes':hashes,'physical_function_hashes':physical_hashes,
            'fit_normalized_sse':{arm:v/fit_energy for arm,v in fit_sse.items() if child_gate or arm in ('parent_prior','e16')},
            'gates':gates,'summary':summary,'validation_rows':rows,'checkpoint_sha256':P.digest(args.checkpoint),
            'checkpoint_bytes':args.checkpoint.stat().st_size,'script_sha256':P.digest(Path(__file__)),
            'runtime':P.budget(start,device),'torch_version':torch.__version__,
            'decision':'local_mixed_function_and_count_pass_native_stored_bank_next' if all(gates.values()) else 'stop_this_fixed_full_feature_mixed_conditional_recipe',
            'scope':'One-layer source-transfer fit/consumed validation only; no full/independent LLM quality, actual routed C/large-RAM n/DRAM/LUT or accepted full rate/second donor'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'progress':progress,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
