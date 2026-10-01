#!/usr/bin/env python3
"""Fixed fit-only source nonlinearity selection and anchored E16/E160 readouts."""
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
import meth227_conditional_donor_tangents as R
import meth229_source_curvature_feasibility as N

P,M=R.P,R.M
D,H,TAU=896,2048,1024.
NATIVE=P.DOC/'meth229_source_curvature_native_result.json'
NATIVE_SHA='f2e94ed991048709ff4c61dbb456ab95077865a36e3dc574dfc3a343d4330dbc'
ROUTE_RESULT=P.DOC/'meth227_conditional_donor_tangent_result.json'
ROUTE_SHA='85f5b41e99850a2eeb32d0c071f32060f35d0e958db4729fffa115681a406fd2'
ROUTE_FILE=P.ROOT/'results/native_expert_scaling/meth227_layer12_tangent_functions.safetensors'


def select_units(x,center,gate,up,down):
    g,u=F.linear(x,gate),F.linear(x,up)
    g0,u0=F.linear(center,gate),F.linear(center,up)
    s=torch.sigmoid(g0)
    jh=(u0*(s+g0*s*(1-s)))[:,None]*gate+F.silu(g0)[:,None]*up
    remainder=F.silu(g)*u-F.silu(g0)*u0-F.linear(x-center,jh)
    score=remainder.double().var(0,correction=0)*down.double().square().sum(0)
    assert torch.isfinite(score).all() and float(score.sum())>0
    ids=torch.argsort(score,descending=True,stable=True)[:H]
    assert len(ids.unique())==H
    return ids,{'score_sum':float(score.sum()),'selected_score_fraction':float(score[ids].sum()/score.sum()),
        'selected_original_ids':ids.cpu().tolist(),'meaning':'Independent unit score, not output-error/retained-energy bound'}


def anchored_readout(z,y,prior_w,prior_b,variance):
    zd,yd=z.double(),y.double()
    wp,bp=prior_w.double(),prior_b.double()
    mean=zd.mean(0); xc=zd-mean
    residual=yd-(zd@wp.T+bp)
    mr=residual.mean(0); rc=residual-mr
    n,features=xc.shape
    if n<features:
        scaled=xc/variance[None,:]
        gram=scaled@xc.T+TAU*torch.eye(n,dtype=torch.float64,device=z.device)
        solution=torch.linalg.solve(gram,rc)
        delta_t=scaled.T@solution
        method='dual'
    else:
        gram=xc.T@xc+TAU*torch.diag(variance)
        delta_t=torch.linalg.solve(gram,xc.T@rc)
        method='primal'
    cross=xc.T@rc
    normal=xc.T@(xc@delta_t-rc)+TAU*variance[:,None]*delta_t
    relative=float(normal.norm()/cross.norm().clamp_min(1e-12))
    assert torch.isfinite(delta_t).all() and relative<=1e-7
    weight=(wp+delta_t.T).bfloat16()
    effective_delta=weight.double()-wp
    shift=mr*(n/(n+TAU))
    bias=(bp+shift-effective_delta@mean).float()
    mean_error=float(((weight.double()@mean+bias.double())-(wp@mean+bp)-shift).norm()/yd.mean(0).norm().clamp_min(1e-12))
    assert torch.isfinite(bias).all() and mean_error<=1e-5
    return weight,bias,{'states':n,'features':features,'solve':method,'normal_equation_relative_residual':relative,
        'intercept_shrink':n/(n+TAU),'mean_shift_rounding_relative_error':mean_error}


def place(tensors,arm,index,w,b):
    tensors[arm+'.affine'][index].copy_(w[:,:D].contiguous().cpu())
    tensors[arm+'.down'][index].copy_(w[:,D:].contiguous().cpu())
    tensors[arm+'.bias'][index].copy_(b.cpu())


def readout(x,h,w,b):
    return (F.linear(x,w[:,:D].float().contiguous())+F.linear(h,w[:,D:].float().contiguous()))+b


def main():
    ap=argparse.ArgumentParser()
    for key in ('checkpoint','out'): ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args(); assert not args.checkpoint.exists() and not args.out.exists()
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True); args.out.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.checkpoint.parent).free>=2*1024**3
    start,stage,progress=time.monotonic(),'bindings',{}
    def partial():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'progress':progress,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
    try:
        assert P.digest(NATIVE)==NATIVE_SHA and P.digest(ROUTE_RESULT)==ROUTE_SHA
        native=json.loads(NATIVE.read_text(encoding='utf-8')); old=json.loads(ROUTE_RESULT.read_text(encoding='utf-8'))
        assert all(native['gates'].values()) and P.digest(ROUTE_FILE)==old['checkpoint_sha256']
        assert P.digest(R.CAPTURE)==old['capture_sha256']
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=30*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            x=torch.from_numpy(archive['x_bf16'].reshape(-1,D)).view(torch.bfloat16).to(device).float()
            y=torch.from_numpy(archive['y_bf16'].reshape(-1,D)).view(torch.bfloat16).to(device).float()
        with safe_open(str(ROUTE_FILE),framework='pt',device='cpu') as archive:
            parents=archive.get_tensor('router.parents').to(device)
            children=archive.get_tensor('router.children').to(device)
        cut=M.FIT*M.SEQ; xf,yf=x[:cut],y[:cut]
        lp=R.assign_full(xf,parents); lc=torch.empty_like(lp)
        for parent in range(16):
            mask=lp==parent; lc[mask]=parent*10+R.assign_full(xf[mask],children[parent])
        counts16=torch.bincount(lp,minlength=16).tolist(); counts160=torch.bincount(lc,minlength=160).tolist()
        assert counts16==old['fit']['counts16'] and counts160==old['fit']['counts160'] and min(counts160)>0
        assert P.M17.sha(lp.cpu().numpy().astype('<i4').tobytes())==old['fit']['parent_labels_sha256']
        assert P.M17.sha(lc.cpu().numpy().astype('<i4').tobytes())==old['fit']['leaf_labels_sha256']
        progress['fixed_routes']={'counts16':counts16,'counts160':counts160,'route_result_sha256':ROUTE_SHA}
        matrices=[]; source_hashes={}
        with safe_open(str(source),framework='pt',device='cpu') as archive:
            for organ in ('gate','up','down'):
                name=f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight'; value=archive.get_tensor(name)
                assert value.dtype==torch.bfloat16 and value.shape==((D,4864) if organ=='down' else (4864,D))
                source_hashes[name]=P.M17.sha(value.view(torch.uint16).numpy().tobytes())
                matrices.append(value.to(device).float())
        assert source_hashes==old['fit']['source_tensor_sha256']
        tensors={'router.parents':parents.cpu().contiguous(),'router.children':children.cpu().contiguous(),
            'selection':torch.empty((16,H),dtype=torch.int64),
            'input.gate':torch.empty((16,H,D),dtype=torch.bfloat16),'input.up':torch.empty((16,H,D),dtype=torch.bfloat16)}
        for arm,count in (('prior',16),('e16',16),('e160',160)):
            tensors[arm+'.affine']=torch.empty((count,D,D),dtype=torch.bfloat16)
            tensors[arm+'.down']=torch.empty((count,D,H),dtype=torch.bfloat16)
            tensors[arm+'.bias']=torch.empty((count,D),dtype=torch.float32)
        audits=[]; selections=[]; fit_sse={arm:0. for arm in ('prior','e16','e160')}; fit_energy=0.
        for parent in range(16):
            mask=lp==parent; xp,yp=xf[mask],yf[mask]; local_leaf=lc[mask]
            stage='fit_only_source_curvature_selection'
            selected,selection=select_units(xp,parents[parent],*matrices); selections.append({'parent':parent,**selection})
            tensors['selection'][parent].copy_(selected.cpu())
            aff,bias,gs,us,ds,raw_aff,full_jac,value=N.source_prior(parents[parent],*matrices,selected)
            tensors['input.gate'][parent].copy_(gs.bfloat16().cpu()); tensors['input.up'][parent].copy_(us.bfloat16().cpu())
            assert torch.equal(gs.bfloat16(),matrices[0][selected].bfloat16()) and torch.equal(us.bfloat16(),matrices[1][selected].bfloat16())
            center_error=float((N.combined_value(parents[parent],aff.float(),gs,us,ds,bias)-value).double().norm()/value.double().norm())
            assert center_error<=1e-5
            derivative={}
            if parent==0:
                rb=value-(F.linear(parents[parent],raw_aff)+N.N.donor_value(parents[parent],gs,us,ds))
                with torch.enable_grad():
                    ref=torch.autograd.functional.jacobian(lambda v:N.combined_value(v,raw_aff,gs,us,ds,rb),parents[parent],vectorize=True)
                diff=ref.double()-full_jac.double()
                derivative={'relative_frobenius':float(diff.norm()/full_jac.double().norm()),
                    'max_abs_relative_to_peak':float(diff.abs().max()/full_jac.double().abs().max())}
                assert derivative['relative_frobenius']<=1e-5 and derivative['max_abs_relative_to_peak']<=1e-4
            prior_w=torch.cat((aff,ds.bfloat16()),dim=1)
            place(tensors,'prior',parent,prior_w,bias)
            h=F.silu(F.linear(xp,gs))*F.linear(xp,us)
            z=torch.cat((xp,h),dim=1)
            variance=z.double().var(0,correction=0)
            floor=max(float(variance.mean())*1e-4,1e-12); variance=variance.clamp_min(floor)
            stage='source_anchored_parent_child_readouts'
            parent_w,parent_b,check=anchored_readout(z,yp,prior_w,bias,variance)
            place(tensors,'e16',parent,parent_w,parent_b)
            audits.append({'arm':'e16','cell':parent,'variance_floor':floor,'source_center_relative_l2':center_error,
                'source_gradient_check':derivative,**check})
            for arm,w,b in (('prior',prior_w,bias),('e16',parent_w,parent_b)):
                fit_sse[arm]+=float((readout(xp,h,w,b)-yp).double().square().sum())
            fit_energy+=float(yp.double().square().sum())
            for ci in range(10):
                leaf=parent*10+ci; chosen=local_leaf==leaf
                w,b,check=anchored_readout(z[chosen],yp[chosen],parent_w,parent_b,variance)
                place(tensors,'e160',leaf,w,b)
                audits.append({'arm':'e160','cell':leaf,**check})
                fit_sse['e160']+=float((readout(xp[chosen],h[chosen],w,b)-yp[chosen]).double().square().sum())
                P.budget(start,device)
            progress['fit']={'completed_parents':parent+1,'readout_checks':len(audits),
                'minimum_cell_states':min(counts160),'runtime':P.budget(start,device)}
            partial(); print(json.dumps(progress['fit']),flush=True)
        stage='physical_snapshot_readback'
        save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-230','source_sha256':P.M57.MODEL_SHA,
            'capture_sha256':old['capture_sha256'],'layer':str(M.LAYER),'hidden':str(H),'prior_samples':str(TAU),
            'compute':'BF16 effective weights,FP32 split affine/nonlinear forward and bias'})
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
            assert set(archive.keys())==set(tensors)
            for name,tensor in tensors.items(): assert torch.equal(tensor,archive.get_tensor(name))
        function_hashes={}
        for arm,count in (('e16',16),('e160',160)):
            function_hashes[arm]=[P.M17.sha(tensors[arm+'.affine'][i].view(torch.uint16).numpy().tobytes()+
                tensors[arm+'.down'][i].view(torch.uint16).numpy().tobytes()) for i in range(count)]
        gates={'fit_routes_counts_and_labels_exact':True,'selected_input_rows_exact_source':True,
            'source_value_gradient_checks':True,'all176_readout_equations_and_intercept_checks':True,
            'every_snapshot_tensor_readback_equal':True,
            'all176_fitted_weight_pairs_distinct':len(set(function_hashes['e16']+function_hashes['e160']))==176}
        rows=[]; summary={}
        if all(gates.values()):
            stage='consumed_validation_once_after_all_fitting'
            del matrices,z,h,prior_w,parent_w
            with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
                saved={name:archive.get_tensor(name).to(device).float() for name in archive.keys() if name!='selection'}
            xv,yv=x[cut:],y[cut:]
            pv=R.assign_full(xv,saved['router.parents']); cv=torch.empty_like(pv)
            hv=torch.empty((len(xv),H),device=device)
            for parent in range(16):
                mask=pv==parent
                if bool(mask.any()):
                    cv[mask]=parent*10+R.assign_full(xv[mask],saved['router.children'][parent])
                    hv[mask]=F.silu(F.linear(xv[mask],saved['input.gate'][parent]))*F.linear(xv[mask],saved['input.up'][parent])
            rotated=cv//10*10+(cv%10+1)%10
            for seq in range(M.VALID):
                span=slice(seq*M.SEQ,(seq+1)*M.SEQ); energy=float(yv[span].double().square().sum())
                assert energy==old['validation_rows'][seq]['energy']
                row={'validation_sequence':seq,'energy':energy,'sse':{}}
                for arm,bank,labels in (('prior','prior',pv),('e16','e16',pv),('e160','e160',cv),('e160_rotated','e160',rotated)):
                    ids=labels[span]
                    linear=torch.bmm(saved[bank+'.affine'][ids],xv[span,:,None]).squeeze(-1)
                    nonlinear=torch.bmm(saved[bank+'.down'][ids],hv[span,:,None]).squeeze(-1)
                    predicted=(linear+nonlinear)+saved[bank+'.bias'][ids]
                    assert torch.isfinite(predicted).all()
                    row['sse'][arm]=float((predicted-yv[span]).double().square().sum())
                rows.append(row); P.budget(start,device)
            energy=sum(r['energy'] for r in rows)
            summary={arm:{'sse':sum(r['sse'][arm] for r in rows),'normalized_sse':sum(r['sse'][arm] for r in rows)/energy}
                for arm in ('prior','e16','e160','e160_rotated')}
            rng=np.random.default_rng(230231); draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
            delta=np.asarray([r['sse']['e16']-r['sse']['e160'] for r in rows]); ey=np.asarray([r['energy'] for r in rows])
            gain=delta[draws].sum(1)/ey[draws].sum(1)
            summary['bootstrap']={'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),
                'seed':230231,'draws':10000,'unit':'consumed_training_corpus_window'}
            gates.update({'complete_function_nmse_le_001':summary['e160']['normalized_sse']<=.01,
                'e160_sse_le_90_percent_e16':summary['e160']['sse']<=.9*summary['e16']['sse'],
                'e160_sse_le_90_percent_rotated':summary['e160']['sse']<=.9*summary['e160_rotated']['sse'],
                'e160_sse_le_90_percent_source_prior':summary['e160']['sse']<=.9*summary['prior']['sse'],
                'paired_window_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-230-source-anchored-nonlinear-E16-E160-pair','layer':M.LAYER,'hidden':H,
            'source_sha256':P.M57.MODEL_SHA,'source_tensor_sha256':source_hashes,'capture_sha256':old['capture_sha256'],
            'native_result_sha256':NATIVE_SHA,'route_result_sha256':ROUTE_SHA,'route_snapshot_sha256':old['checkpoint_sha256'],
            'counts16':counts16,'counts160':counts160,'fit_unique_states':cut,'readout_fit_state_exposures':2*cut,
            'prior_sample_equivalents':TAU,'selection':selections,'readout_audits':audits,'function_hashes':function_hashes,
            'fit_normalized_sse':{arm:v/fit_energy for arm,v in fit_sse.items()},'gates':gates,'summary':summary,
            'validation_rows':rows,'checkpoint_sha256':P.digest(args.checkpoint),'checkpoint_bytes':args.checkpoint.stat().st_size,
            'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),'torch_version':torch.__version__,
            'decision':'local_nonlinear_function_and_count_pass_native_stored_bank_next' if all(gates.values()) else 'stop_this_fixed_source_anchored_nonlinear_readout_geometry',
            'scope':'One-layer stored function fit on consumed raw training windows; no independent LLM quality, n-scaled C/DRAM/LUT, full causal model/rate or second donor'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'decision':result['decision'],'summary':summary,'gates':gates,'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'elapsed_seconds':time.monotonic()-start,'progress':progress},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
