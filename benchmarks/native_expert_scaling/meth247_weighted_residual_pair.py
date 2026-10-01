#!/usr/bin/env python3
"""Encode frozen continuous solutions as separate activity-weighted BF16 factors."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch.nn import functional as F
import meth244_readout_encoding_audit as A

C,P,M,R,Q=A.C,A.P,A.M,A.R,A.Q
NATIVE=P.DOC/'meth246_single_team_residual_native_result.json'
NATIVE_SHA='bb84bdb1544025866e7411dd830ffea3bc5daf4b73447bf194c45c55ba9da202'
AUDIT=P.DOC/'meth244_readout_encoding_result.json'
AUDIT_SHA='5164e4deb8ee6866b9d3b1f62f0d3468f6e7d814c517000876ccbf8305949849'
RANK=32
FIELDS=('right','left','bias')


def encoded_value(phi,base,factors):
    right,left,bias=factors
    correction=F.linear(F.linear(phi,right.float()),left.float())
    return ((Q.L.mixed_linear(phi,*base[:4])+correction)+base[4])+bias


def factorize(z,raw_w,raw_b,base):
    zd=z.double(); mean=zd.mean(0); centered=zd-mean
    delta=raw_w-C.decode(base).double(); response=centered@delta.T
    energy=float(response.square().sum()); covariance=response.T@response
    if energy==0:
        u,s,_=torch.linalg.svd(delta,full_matrices=False,driver='gesvdj')
        basis=u[:,:RANK]; fraction=None; method='zero_response_source_coefficient_SVD'
        residual=0.; orth=float((basis.T@basis-torch.eye(RANK,dtype=torch.float64,device=z.device)).norm())
        assert abs(float(s.square().sum())/float(delta.square().sum().clamp_min(1e-24))-1)<=1e-8 or float(delta.square().sum())==0
    else:
        eigen,u=torch.linalg.eigh(covariance)
        assert float(eigen[0])>=-1e-10*float(eigen[-1])
        assert abs(float(eigen.sum())/energy-1)<=1e-8
        basis=u[:,-RANK:].flip(1); largest=eigen[-RANK:].flip(0)
        residual=float((covariance@basis-basis*largest[None,:]).norm()/covariance.norm().clamp_min(1e-24))
        orth=float((basis.T@basis-torch.eye(RANK,dtype=torch.float64,device=z.device)).norm())
        fraction=float(largest.sum()/eigen.sum()); method='fit_centered_output_covariance'
    assert residual<=1e-8 and orth<=1e-8
    ids=basis.abs().argmax(0); signs=torch.sign(basis[ids,torch.arange(RANK,device=z.device)])
    assert bool((signs.abs()==1).all()); basis=basis*signs[None,:]
    projection=basis.T@delta; norm=projection.norm(dim=1)
    balance=torch.where(norm>0,norm.sqrt(),torch.ones_like(norm))
    left=(basis*balance[None,:]).bfloat16().contiguous()
    right=(projection/balance[:,None]).bfloat16().contiguous()
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    zero=torch.zeros(896,device=z.device)
    before=encoded_value(z,base,(right,left,zero))
    desired=raw_w@mean+raw_b
    bias=(desired-before.double().mean(0)).float()
    prediction=encoded_value(z,base,(right,left,bias))
    mean_error=float((prediction.double().mean(0)-desired).norm()/desired.norm().clamp_min(1e-12))
    assert torch.isfinite(bias).all() and mean_error<=1e-5
    effective=left.float().double()@right.float().double()
    centered_error=float(((centered@right.float().double().T)@left.float().double().T-response).square().sum())
    report={'method':method,'centered_raw_correction_energy':energy,
        'rank32_unrounded_response_energy_fraction':fraction,'eigen_relative_residual':residual,
        'basis_orthogonality_frobenius':orth,'mean_conservation_relative_l2':mean_error,
        'stored_factor_centered_response_relative_squared_error':centered_error/max(energy,1e-24),
        'stored_factor_weight_relative_squared_error':float((effective-delta).square().sum()/delta.square().sum().clamp_min(1e-24))}
    return (right,left,bias),prediction,report


def factor_get(tensors,bank,index,device):
    return tuple(tensors[bank+'.residual.'+field][index].to(device) for field in FIELDS)


def predict(phi,labels,tensors,bank,device):
    output=torch.empty((len(phi),896),device=device)
    count=len(tensors[bank+'.residual.bias'])
    for index in range(count):
        mask=labels==index
        if bool(mask.any()):
            parent=index if bank=='e16' else index//10
            output[mask]=encoded_value(phi[mask],C.get(tensors,'base',parent,device),factor_get(tensors,bank,index,device))
    assert torch.isfinite(output).all(); return output


def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','out'): ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args(); assert not args.checkpoint.exists() and not args.out.exists()
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.checkpoint.parent).free>=2*1024**3
    start,stage=time.monotonic(),'bindings'; fit_rows=[]; validation=[]
    try:
        assert P.digest(NATIVE)==NATIVE_SHA and P.digest(A.D.PAIR)==A.D.PAIR_SHA and P.digest(AUDIT)==AUDIT_SHA
        native=json.loads(NATIVE.read_text()); pair=json.loads(A.D.PAIR.read_text()); audit=json.loads(AUDIT.read_text())
        assert all(native['gates'].values()) and audit['decision']=='continuous_local_count_gain_exposed_change_readout_encoding'
        assert P.digest(A.D.SNAPSHOT)==pair['checkpoint_sha256'] and P.digest(R.CAPTURE)==pair['capture_sha256']
        assert P.digest(Q.NATIVE)==Q.NATIVE_SHA and P.digest(Q.S.ROUTE_RESULT)==Q.S.ROUTE_SHA
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        values,controls=Q.load_layer(json.loads(Q.NATIVE.read_text()),device); assert controls==pair['controls']
        with safe_open(str(A.D.SNAPSHOT),framework='pt',device='cpu') as archive:
            original={name:archive.get_tensor(name) for name in archive.keys()}
        tensors={name:t for name,t in original.items() if not name.startswith(('parent_prior.','e16.','child_prior.','e160.'))}
        for field in C.FIELDS: tensors['base.'+field]=original['e16.'+field]
        for bank,count in (('e16',16),('e160',160)):
            tensors[bank+'.residual.right']=torch.empty((count,RANK,4864),dtype=torch.bfloat16)
            tensors[bank+'.residual.left']=torch.empty((count,896,RANK),dtype=torch.bfloat16)
            tensors[bank+'.residual.bias']=torch.empty((count,896),dtype=torch.float32)
        parents=tensors['router.parents'].to(device); children=tensors['router.children'].to(device)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            xf=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            yf=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        lp=R.assign_full(xf,parents); lc=torch.empty_like(lp)
        for parent in range(16):
            mask=lp==parent; lc[mask]=parent*10+R.assign_full(xf[mask],children[parent])
        route=json.loads(Q.S.ROUTE_RESULT.read_text())
        assert torch.bincount(lp,minlength=16).tolist()==pair['counts16'] and torch.bincount(lc,minlength=160).tolist()==pair['counts160']
        for labels,key in ((lp,'parent_labels_sha256'),(lc,'leaf_labels_sha256')):
            assert P.M17.sha(labels.cpu().numpy().astype('<i4').tobytes())==route['fit'][key]
        stage='fixed_raw_solve_replay_and_rank32_activity_weighted_encoding'
        hashes={'e16':[],'e160':[]}; energy=float(yf.double().square().sum())
        for parent in range(16):
            mask=lp==parent; phi=Q.features(xf[mask],values); y=yf[mask]; local=lc[mask]
            base=C.get(tensors,'base',parent,device); variance=tensors['fit.feature_variance'][parent].to(device)
            tasks=[('e16',parent,phi,y,'parent_prior')]+[('e160',parent*10+c,phi[local==parent*10+c],y[local==parent*10+c],'child_prior') for c in range(10)]
            for bank,index,z,target,prior_bank in tasks:
                prior=C.get(original,prior_bank,index,device)
                old_coeff,raw_w,raw_b,normal=A.replay_fit(z,target,prior,variance)
                assert all(torch.equal(a,b) for a,b in zip(old_coeff,C.get(original,bank,index,device)))
                factors,output,report=factorize(z,raw_w,raw_b,base)
                for field,t in zip(FIELDS,factors): tensors[bank+'.residual.'+field][index].copy_(t.cpu())
                effective=C.decode(base).double()+factors[1].float().double()@factors[0].float().double()
                hashes[bank].append(P.M17.sha(effective.cpu().numpy().tobytes()))
                raw=z.double()@raw_w.T+raw_b
                fit_rows.append({'arm':bank,'cell':index,'states':len(z),'normal_equation_relative_residual':normal,
                    **report,'raw_solution_sse':float((raw-target.double()).square().sum()),
                    'actual_factorized_sse':float((output-target).double().square().sum())})
                P.budget(start,device)
            args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'fit_rows':fit_rows},indent=2)+'\n',encoding='utf-8')
        fit={}
        for bank in ('e16','e160'):
            selected=[r for r in fit_rows if r['arm']==bank]
            fit[bank]={'raw_normalized_sse':sum(r['raw_solution_sse'] for r in selected)/energy,
                'factorized_normalized_sse':sum(r['actual_factorized_sse'] for r in selected)/energy}
            assert abs(fit[bank]['raw_normalized_sse']/audit['summary'][bank]['raw_FP64_solution_sse']-1)<=1e-12
        stage='physical_snapshot_readback'
        save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-247','rank':'32','raw_fit_snapshot_sha256':pair['checkpoint_sha256'],
            'operator_result_sha256':NATIVE_SHA,'base':'actual stored METH-240 fitted E16','factorization':'fit-centered covariance;zero-response source SVD'})
        assert args.checkpoint.stat().st_size<200000000
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
            assert set(archive.keys())==set(tensors)
            for name,t in tensors.items(): assert torch.equal(archive.get_tensor(name),t)
        gates={'all_native_source_snapshot_capture_route_bindings':True,'all_fit_labels_counts_exact':True,
            'all176_original_coefficient_bias_replays_exact':len(fit_rows)==176,
            'all176_factor_basis_mean_checks':True,'raw_fit_scores_replay_within_1e12_relative':True,
            'all_snapshot_tensors_readback_exact':True,'all176_decoded_FP64_coefficients_distinct':len(set(hashes['e16']+hashes['e160']))==176,
            'factorized_fit_functions_nmse_le_001':all(v['factorized_normalized_sse']<=.01 for v in fit.values()),
            'factorized_E16_fit_no_worse_than_stored_base_E16':fit['e16']['factorized_normalized_sse']<=pair['fit_normalized_sse']['e16'],
            'factorized_E160_fit_sse_le_90_percent_E16':fit['e160']['factorized_normalized_sse']<=.9*fit['e16']['factorized_normalized_sse']}
        summary={}
        if all(gates.values()):
            stage='consumed_validation_once_after_all_factorization'
            del xf,yf,phi,y,tasks,z,target,raw,raw_w,raw_b,effective,prior,base,old_coeff,output
            with np.load(R.CAPTURE,allow_pickle=False) as archive:
                xv=torch.from_numpy(archive['x_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
                yv=torch.from_numpy(archive['y_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            pv=R.assign_full(xv,parents); cv=torch.empty_like(pv)
            for parent in range(16):
                mask=pv==parent
                if bool(mask.any()): cv[mask]=parent*10+R.assign_full(xv[mask],children[parent])
            rotated=cv//10*10+(cv%10+1)%10; hv=Q.features(xv,values)
            predictions={bank:predict(hv,labels,tensors,'e160' if bank=='e160_rotated' else bank,device)
                for bank,labels in (('e16',pv),('e160',cv),('e160_rotated',rotated))}
            for bank,labels in (('parent_prior',pv),('child_prior',cv),('stored_base_e16',pv)):
                predictions[bank]=C.predict_bank(hv,labels,original,'e16' if bank=='stored_base_e16' else bank,device)
                P.budget(start,device)
            for sequence in range(M.VALID):
                span=slice(sequence*M.SEQ,(sequence+1)*M.SEQ); ey=float(yv[span].double().square().sum())
                assert ey==pair['validation_rows'][sequence]['energy']
                sse={bank:float((pred[span]-yv[span]).double().square().sum()) for bank,pred in predictions.items()}
                assert sse['stored_base_e16']==pair['validation_rows'][sequence]['sse']['e16']
                assert all(sse[bank]==pair['validation_rows'][sequence]['sse'][bank] for bank in ('parent_prior','child_prior'))
                validation.append({'validation_sequence':sequence,'energy':ey,'sse':sse})
            ey=sum(r['energy'] for r in validation)
            summary={bank:{'sse':sum(r['sse'][bank] for r in validation),'normalized_sse':sum(r['sse'][bank] for r in validation)/ey} for bank in predictions}
            rng=np.random.default_rng(247248); draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
            delta=np.asarray([r['sse']['e16']-r['sse']['e160'] for r in validation]); eys=np.asarray([r['energy'] for r in validation])
            gain=delta[draws].sum(1)/eys[draws].sum(1)
            summary['bootstrap']={'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),'seed':247248,'draws':10000}
            gates.update({'unchanged_prior_and_stored_parent_validation_controls_exact':True,
                'complete_function_nmse_le_001':summary['e160']['normalized_sse']<=.01,
                **{'e160_sse_le_90_percent_'+bank:summary['e160']['sse']<=.9*summary[bank]['sse'] for bank in ('e16','e160_rotated','child_prior','parent_prior','stored_base_e16')},
                'paired_window_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-247-separate-activity-weighted-rank32-BF16-E16-E160-functions','rank':RANK,
            'native_result_sha256':NATIVE_SHA,'pair_result_sha256':A.D.PAIR_SHA,'encoding_audit_sha256':AUDIT_SHA,
            'raw_snapshot_sha256':pair['checkpoint_sha256'],'source_sha256':pair['source_sha256'],'capture_sha256':pair['capture_sha256'],
            'route_result_sha256':Q.S.ROUTE_SHA,'controls':controls,'prior_sample_equivalents':C.TAU,
            'counts16':pair['counts16'],'counts160':pair['counts160'],'fit_rows':fit_rows,'fit_summary':fit,
            'decoded_FP64_coefficient_hashes':hashes,'gates':gates,'summary':summary,'validation_rows':validation,
            'checkpoint_sha256':P.digest(args.checkpoint),'checkpoint_bytes':args.checkpoint.stat().st_size,
            'script_sha256':P.digest(Path(__file__)),'runtime':P.budget(start,device),'torch_version':torch.__version__,
            'decision':'separate_rank32_local_function_count_pass_native_bank_next' if all(gates.values()) else 'stop_this_fixed_activity_weighted_rank32_residual_pair',
            'scope':'Same frozen raw176 solutions/keys/priors/tau,fit-only factor selection and consumed development validation. Actual stored parent plus BF16 factors,not requantized parent. No trained bank C/large-RAM n/new full independent LLM quality or accepted full rate/second donor.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','fit_summary','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'fit_rows':fit_rows,'validation_rows':validation,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
