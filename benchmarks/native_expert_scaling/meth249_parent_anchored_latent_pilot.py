#!/usr/bin/env python3
"""Nondeployable FP64 child residuals preserving the complete actual parent."""
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
import meth248_raw_validation_audit as D

B,A,C,P,M,R,Q=D.B,D.A,D.C,D.P,D.M,D.R,D.Q
DIAG=P.DOC/'meth248_raw_validation_result.json'
DIAG_SHA='dda4fc408cf4b7b51fa2909d0e5d8cec9df999a227266d4823b3ff4b26315e7a'


def source_fallback(center,base,factors,left,pinv,variance,values,matrices):
    phi_j=Q.feature_jacobian(center,values)
    feature_checks=Q.derivative_checks(center,values,phi_j)
    assert feature_checks['fixed16_row_relative_frobenius']<=1e-5
    assert all(r['relative_l2']<=1e-5 for r in feature_checks['directions'])
    effective=C.decode(base).double()+left@factors[0].float().double()
    parent_j=effective@phi_j.double()
    ids=torch.tensor([0,1,127,255,383,511,639,767,895],device=center.device)
    with torch.enable_grad():
        automatic=torch.autograd.functional.jacobian(lambda x:B.encoded_value(Q.features(x,values),base,factors)[ids],center,vectorize=True)
    parent_check=float((automatic.double()-parent_j[ids]).norm()/parent_j[ids].norm())
    assert parent_check<=1e-5
    source_j,_=Q.N.donor_tangent(center,*matrices)
    target=pinv@(source_j.double()-parent_j)
    root=variance.sqrt(); whitened=phi_j.double()/root[:,None]
    qq,tt=torch.linalg.qr(whitened,mode='reduced')
    singular=torch.linalg.svdvals(tt); condition=float(singular[0]/singular[-1])
    assert torch.isfinite(singular).all() and condition<=1e8
    latent=torch.linalg.solve_triangular(tt.T,target.T,upper=False).T@qq.T/root[None,:]
    relative=float((latent@phi_j.double()-target).norm()/target.norm().clamp_min(1e-12))
    growth=float((left@latent).norm()/effective.norm().clamp_min(1e-12))
    assert torch.isfinite(latent).all() and relative<=1e-5 and growth<=.25
    return latent,{'feature_checks':feature_checks,'parent_autograd_relative_check':parent_check,
        'whitened_feature_condition_number':condition,'projected_source_gradient_relative_residual':relative,
        'effective_increment_relative_norm':growth}


def fit_child(z,y,parent_prediction,left,pinv,variance,prior):
    zd=z.double(); mean=zd.mean(0); xc=zd-mean
    residual=y.double()-parent_prediction.double(); mean_residual=residual.mean(0)
    latent_target=residual@pinv.T
    centered=(latent_target-latent_target.mean(0))-xc@prior.T
    n,h=xc.shape
    if n<h:
        scaled=xc/variance[None,:]
        gram=scaled@xc.T+C.TAU*torch.eye(n,dtype=torch.float64,device=z.device)
        delta_t=scaled.T@torch.linalg.solve(gram,centered); method='dual'
    else:
        gram=xc.T@xc+C.TAU*torch.diag(variance)
        delta_t=torch.linalg.solve(gram,xc.T@centered); method='primal'
    cross=xc.T@centered
    normal=xc.T@(xc@delta_t-centered)+C.TAU*variance[:,None]*delta_t
    relative=float(normal.norm()/cross.norm().clamp_min(1e-12)); assert relative<=1e-7
    final=prior+delta_t.T; gamma=n/(n+C.TAU)
    bias=gamma*mean_residual-left@(final@mean)
    prediction=parent_prediction.double()+(zd@final.T)@left.T+bias
    expected=parent_prediction.double().mean(0)+gamma*mean_residual
    mean_error=float((prediction.mean(0)-expected).norm()/expected.norm().clamp_min(1e-12))
    assert torch.isfinite(final).all() and torch.isfinite(bias).all() and mean_error<=1e-10
    return final,bias,prediction,mean,{'states':n,'solve':method,'normal_relative_residual':relative,
        'mean_conservation_relative_l2':mean_error,'mean_intercept_shrink':gamma,
        'centered_feature_energy':float(xc.square().sum())}


def child_predict(phi,parent_prediction,labels,tensors,device):
    out=torch.empty((len(phi),896),dtype=torch.float64,device=device)
    for leaf in range(160):
        mask=labels==leaf
        if bool(mask.any()):
            left=tensors['e16.residual.left'][leaf//10].to(device).float().double()
            delta=tensors['child.delta_right'][leaf].to(device); bias=tensors['child.delta_bias'][leaf].to(device)
            out[mask]=parent_prediction[mask].double()+(phi[mask].double()@delta.T)@left.T+bias
    assert torch.isfinite(out).all(); return out


def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','out'): ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args(); assert not args.checkpoint.exists() and not args.out.exists()
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True); assert shutil.disk_usage(args.checkpoint.parent).free>=2*1024**3
    start,stage=time.monotonic(),'bindings'; fit_rows=[]; rows=[]; parent_replays=[]
    try:
        assert P.digest(D.PAIR)==D.PAIR_SHA and P.digest(DIAG)==DIAG_SHA
        assert P.digest(A.D.PAIR)==A.D.PAIR_SHA
        original_result=json.loads(A.D.PAIR.read_text())
        assert P.digest(A.D.SNAPSHOT)==original_result['checkpoint_sha256']
        split=json.loads(D.PAIR.read_text()); diagnosis=json.loads(DIAG.read_text())
        assert diagnosis['decision']=='raw_consumed_count_gain_fails_change_continuous_function_hierarchy'
        assert P.digest(D.ENCODED)==split['checkpoint_sha256'] and P.digest(R.CAPTURE)==split['capture_sha256']
        assert P.digest(B.NATIVE)==B.NATIVE_SHA and P.digest(Q.NATIVE)==Q.NATIVE_SHA
        assert P.digest(Q.S.ROUTE_RESULT)==Q.S.ROUTE_SHA
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        values,controls=Q.load_layer(json.loads(Q.NATIVE.read_text()),device); assert controls==split['controls']
        with safe_open(str(D.ENCODED),framework='pt',device='cpu') as archive:
            tensors={name:archive.get_tensor(name) for name in archive.keys() if not name.startswith('e160.')}
        parent_keys=set(tensors)
        tensors['child.delta_right']=torch.empty((160,32,4864),dtype=torch.float64)
        tensors['child.delta_bias']=torch.empty((160,896),dtype=torch.float64)
        tensors['diagnostic.child_mean_phi']=torch.empty((160,4864),dtype=torch.float64)
        parents=tensors['router.parents'].to(device); children=tensors['router.children'].to(device)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            xf=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            yf=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        lp=R.assign_full(xf,parents); lc=torch.empty_like(lp)
        for parent in range(16):
            mask=lp==parent; lc[mask]=parent*10+R.assign_full(xf[mask],children[parent])
        route=json.loads(Q.S.ROUTE_RESULT.read_text())
        assert torch.bincount(lp,minlength=16).tolist()==split['counts16'] and torch.bincount(lc,minlength=160).tolist()==split['counts160']
        for label,key in ((lp,'parent_labels_sha256'),(lc,'leaf_labels_sha256')):
            assert P.M17.sha(label.cpu().numpy().astype('<i4').tobytes())==route['fit'][key]
        hashes={'e16':[],'e160':[]}; inverses=[]; source_matrices=None; source_hashes={}
        fit_parent_sse=fit_child_sse=0.; energy=float(yf.double().square().sum())
        stage='full_parent_anchored_continuous_latent_child_fits'
        for parent in range(16):
            mask=lp==parent; phi=Q.features(xf[mask],values); y=yf[mask]; local=lc[mask]
            base=C.get(tensors,'base',parent,device); factors=B.factor_get(tensors,'e16',parent,device)
            parent_prediction=B.encoded_value(phi,base,factors)
            parent_sse=float((parent_prediction-y).double().square().sum());fit_parent_sse+=parent_sse
            previous=next(r for r in split['fit_rows'] if r['arm']=='e16' and r['cell']==parent)
            parent_replays.append({'parent':parent,'sse':parent_sse,'old_sse':previous['actual_factorized_sse'],
                'relative_discrepancy':parent_sse/previous['actual_factorized_sse']-1})
            left=factors[1].float().double(); gram=left.T@left
            condition=float(torch.linalg.cond(gram)); assert condition<=1e8
            pinv=torch.linalg.solve(gram,left.T)
            inverse_error=float((pinv@left-torch.eye(32,dtype=torch.float64,device=device)).norm()); assert inverse_error<=1e-8
            inverses.append({'parent':parent,'left_gram_condition':condition,'left_inverse_identity_error':inverse_error})
            effective=C.decode(base).double()+left@factors[0].float().double()
            parent_hash=P.M17.sha(effective.cpu().numpy().tobytes()); hashes['e16'].append(parent_hash)
            assert parent_hash==split['decoded_FP64_coefficient_hashes']['e16'][parent]
            variance=tensors['fit.feature_variance'][parent].to(device)
            for child in range(10):
                leaf=parent*10+child; chosen=local==leaf; z=phi[chosen]; target=y[chosen]; pp=parent_prediction[chosen]
                supported=bool(torch.count_nonzero(z.double()-z.double().mean(0)))
                prior=torch.zeros((32,4864),dtype=torch.float64,device=device); fallback={}
                if not supported:
                    if source_matrices is None:
                        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
                        assert P.digest(source)==P.M57.MODEL_SHA
                        source_matrices=[]
                        with safe_open(str(source),framework='pt',device='cpu') as archive:
                            for organ in ('gate','up','down'):
                                name=f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight'; w=archive.get_tensor(name)
                                assert w.dtype==torch.bfloat16; source_hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes())
                                source_matrices.append(w.to(device).float())
                        assert source_hashes==original_result['source_tensor_sha256']
                    prior,fallback=source_fallback(children[parent,child],base,factors,left,pinv,variance,values,source_matrices)
                delta,bias,pred,mean,audit=fit_child(z,target,pp,left,pinv,variance,prior)
                growth=float((left@delta).norm()/effective.norm().clamp_min(1e-12)); assert growth<=.25
                tensors['child.delta_right'][leaf].copy_(delta.cpu()); tensors['child.delta_bias'][leaf].copy_(bias.cpu())
                tensors['diagnostic.child_mean_phi'][leaf].copy_(mean.cpu())
                composite=effective+left@delta; hashes['e160'].append(P.M17.sha(composite.cpu().numpy().tobytes()))
                sse=float((pred-target.double()).square().sum()); fit_child_sse+=sse
                fit_rows.append({'child':leaf,'supported_centered_features':supported,'source_fallback':fallback,
                    'effective_increment_relative_norm':growth,'sse':sse,**audit})
                P.budget(start,device)
            args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'fit_rows':fit_rows},indent=2)+'\n',encoding='utf-8')
        stage='continuous_snapshot_and_unchanged_full_parent_readback'
        save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-249','scope':'nondeployable FP64 latent deltas',
            'parent_snapshot_sha256':split['checkpoint_sha256'],'tau':str(C.TAU),'rank':'32'})
        assert args.checkpoint.stat().st_size<320000000
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive, safe_open(str(D.ENCODED),framework='pt',device='cpu') as previous:
            assert set(archive.keys())==set(tensors)
            for name,t in tensors.items():
                assert torch.equal(archive.get_tensor(name),t)
                if name in parent_keys: assert torch.equal(previous.get_tensor(name),t)
        stage='original_exact_parent_fit_score_replay'
        assert fit_parent_sse/energy==split['fit_summary']['e16']['factorized_normalized_sse']
        gates={'all_source_parent_capture_route_native_bindings':True,'all_fit_labels_counts_exact':True,
            'all16_parent_coefficients_predictions_and_fields_unchanged':True,'all16_left_inverses_qualified':True,
            'all160_solve_mean_and_growth_controls':len(fit_rows)==160,'all_snapshot_tensors_readback_exact':True,
            'all176_decoded_coefficients_distinct':len(set(hashes['e16']+hashes['e160']))==176,
            'fit_function_nmse_le_001':fit_child_sse/energy<=.01,
            'fit_child_sse_no_worse_than_parent':fit_child_sse<=fit_parent_sse*(1+1e-12)}
        summary={}
        if all(gates.values()):
            del xf,yf,phi,y,parent_prediction,source_matrices,z,target,pp,pred,delta,prior,bias,composite,effective
            stage='consumed_validation_once_after_all_fit_prerequisites'
            with np.load(R.CAPTURE,allow_pickle=False) as archive:
                xv=torch.from_numpy(archive['x_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
                yv=torch.from_numpy(archive['y_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            pv=R.assign_full(xv,parents); cv=torch.empty_like(pv)
            for parent in range(16):
                chosen=pv==parent
                if bool(chosen.any()): cv[chosen]=parent*10+R.assign_full(xv[chosen],children[parent])
            rotated=cv//10*10+(cv%10+1)%10; hv=Q.features(xv,values)
            pv_prediction=B.predict(hv,pv,tensors,'e16',device)
            predictions={'e16':pv_prediction.double(),'e160':child_predict(hv,pv_prediction,cv,tensors,device),
                'e160_rotated':child_predict(hv,pv_prediction,rotated,tensors,device)}
            with safe_open(str(A.D.SNAPSHOT),framework='pt',device='cpu') as archive:
                priors={name:archive.get_tensor(name) for name in archive.keys() if name.startswith(('parent_prior.','child_prior.'))}
            for bank,labels in (('parent_prior',pv),('child_prior',cv)):
                predictions[bank]=C.predict_bank(hv,labels,priors,bank,device).double()
            for sequence in range(M.VALID):
                span=slice(sequence*M.SEQ,(sequence+1)*M.SEQ); ey=float(yv[span].double().square().sum())
                assert ey==split['validation_rows'][sequence]['energy']
                sse={bank:float((prediction[span]-yv[span].double()).square().sum()) for bank,prediction in predictions.items()}
                # Keep the original FP32-subtraction parent/prior controls separate.
                controls_sse={bank:float((predictions[bank][span].float()-yv[span]).double().square().sum()) for bank in ('e16','parent_prior','child_prior')}
                assert all(controls_sse[bank]==split['validation_rows'][sequence]['sse'][bank] for bank in controls_sse)
                rows.append({'validation_sequence':sequence,'energy':ey,'sse':sse,'unchanged_FP32_control_sse':controls_sse})
            ey=sum(r['energy'] for r in rows)
            summary={bank:{'sse':sum(r['sse'][bank] for r in rows),'normalized_sse':sum(r['sse'][bank] for r in rows)/ey} for bank in predictions}
            rng=np.random.default_rng(249250); draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
            diff=np.asarray([r['sse']['e16']-r['sse']['e160'] for r in rows]); energies=np.asarray([r['energy'] for r in rows])
            gain=diff[draws].sum(1)/energies[draws].sum(1)
            summary['bootstrap']={'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),'seed':249250,'draws':10000}
            gates.update({'all_parent_prior_validation_controls_exact':True,'continuous_function_nmse_le_001':summary['e160']['normalized_sse']<=.01,
                **{'e160_sse_le_90_percent_'+bank:summary['e160']['sse']<=.9*summary[bank]['sse'] for bank in ('e16','e160_rotated','parent_prior','child_prior')},
                'paired_window_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-249-full-actual-parent-anchored-tied-rank32-continuous-child-pilot',
            'parent_result_sha256':D.PAIR_SHA,'diagnosis_result_sha256':DIAG_SHA,'parent_snapshot_sha256':split['checkpoint_sha256'],
            'source_sha256':split['source_sha256'],'source_fallback_tensor_hashes':source_hashes,'capture_sha256':split['capture_sha256'],
            'native_result_sha256':B.NATIVE_SHA,'route_result_sha256':Q.S.ROUTE_SHA,'controls':controls,'tau':C.TAU,
            'counts16':split['counts16'],'counts160':split['counts160'],'left_inverse_controls':inverses,'fit_rows':fit_rows,'parent_replays':parent_replays,
            'fit_summary':{'parent_nmse':fit_parent_sse/energy,'child_nmse':fit_child_sse/energy},
            'decoded_FP64_coefficient_hashes':hashes,'gates':gates,'summary':summary,'validation_rows':rows,
            'checkpoint_sha256':P.digest(args.checkpoint),'checkpoint_bytes':args.checkpoint.stat().st_size,
            'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),'torch_version':torch.__version__,
            'decision':'continuous_parent_anchor_count_pass_freeze_private_codec_and_kernel' if all(gates.values()) else 'stop_this_fixed_full_parent_tied_latent_hierarchy',
            'scope':'Full actual parent preserved;private FP64 residuals in its fixed output space. Source projected sensitivity only for zero support,not BF16 derivative. Consumed one-layer development,nondeployable/no new C,n/DRAM/full independent quality/rate/second donor.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','fit_summary','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'fit_rows':fit_rows,'validation_rows':rows,'parent_replays':parent_replays,
            'parent_nmse':fit_parent_sse/energy if parent_replays else None,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
