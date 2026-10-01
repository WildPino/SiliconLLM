#!/usr/bin/env python3
"""Matched E16/E160 residual geometry with complete fitted-parent anchoring."""
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
import meth249_parent_anchored_latent_pilot as F

B,A,C,P,M,R,Q=F.B,F.A,F.C,F.P,F.M,F.R,F.Q
BASIS_RESULT=P.DOC/'meth250_residual_output_space_result.json'
BASIS_RESULT_SHA='8220ad28bd577730f78bb02827e8b09f66b9d2c010a453da009bbdb69a920bfb'
BASIS=P.ROOT/'results/native_expert_scaling/meth250_layer12_actual_parent_residual_basis.safetensors'
BASIS_SHA='c457466afb56ab19ab5f7872309a8e73ef51db5e726e6cceaff51cafbe4095a0'


def value(phi,reference,left,right,bias):
    return reference.double()+(phi.double()@right.T)@left.T+bias


def predict(phi,labels,reference,tensors,bank,device):
    out=torch.empty_like(reference,dtype=torch.float64)
    for index in range(16 if bank=='anchor' else 160):
        chosen=labels==index
        if bool(chosen.any()):
            parent=index if bank=='anchor' else index//10
            left=tensors['private.left'][parent].to(device).float().double()
            out[chosen]=value(phi[chosen],reference[chosen],left,
                tensors[bank+'.right'][index].to(device),tensors[bank+'.bias'][index].to(device))
    assert torch.isfinite(out).all();return out


def fallback(center,base,factors,left,pinv,variance,values,source,effective,ar,ab):
    jac=Q.feature_jacobian(center,values);checks=Q.derivative_checks(center,values,jac)
    assert checks['fixed16_row_relative_frobenius']<=1e-5 and all(r['relative_l2']<=1e-5 for r in checks['directions'])
    parent_j=effective@jac.double();ids=torch.tensor([0,1,127,255,383,511,639,767,895],device=center.device)
    def function(x):
        phi=Q.features(x,values)
        return value(phi,B.encoded_value(phi,base,factors),left,ar,ab)[ids]
    with torch.enable_grad():automatic=torch.autograd.functional.jacobian(function,center,vectorize=True)
    check=float((automatic.double()-parent_j[ids]).norm()/parent_j[ids].norm());assert check<=1e-5
    source_j,_=Q.N.donor_tangent(center,*source);target=pinv@(source_j.double()-parent_j)
    root=variance.sqrt();qq,tt=torch.linalg.qr(jac.double()/root[:,None],mode='reduced')
    singular=torch.linalg.svdvals(tt);condition=float(singular[0]/singular[-1]);assert torch.isfinite(singular).all() and condition<=1e8
    prior=torch.linalg.solve_triangular(tt.T,target.T,upper=False).T@qq.T/root[None,:]
    error=float((prior@jac.double()-target).norm()/target.norm().clamp_min(1e-12))
    growth=float((left@prior).norm()/effective.norm().clamp_min(1e-12));assert torch.isfinite(prior).all() and error<=1e-5 and growth<=.25
    return prior,{'feature_checks':checks,'complete_matched_parent_autograd_relative_check':check,
        'whitened_feature_condition_number':condition,'projected_source_gradient_relative_residual':error,'effective_increment_relative_norm':growth}


def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','out'):ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args();assert not args.checkpoint.exists() and not args.out.exists()
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(args.checkpoint.parent).free>=2*1024**3
    start=time.monotonic();stage='bindings';fits=[];anchors=[];rows=[]
    try:
        for path,sha in ((BASIS_RESULT,BASIS_RESULT_SHA),(BASIS,BASIS_SHA),(F.D.PAIR,F.D.PAIR_SHA),
                         (A.D.PAIR,A.D.PAIR_SHA),(B.NATIVE,B.NATIVE_SHA),(Q.NATIVE,Q.NATIVE_SHA),(Q.S.ROUTE_RESULT,Q.S.ROUTE_SHA)):
            assert P.digest(path)==sha
        br=json.loads(BASIS_RESULT.read_text());pair=json.loads(F.D.PAIR.read_text());original=json.loads(A.D.PAIR.read_text())
        assert all(br['gates'].values()) and br['decision']=='freeze_full_parent_private_residual_basis_pilot'
        assert P.digest(F.D.ENCODED)==pair['checkpoint_sha256']==br['parent_snapshot_sha256']
        assert P.digest(R.CAPTURE)==pair['capture_sha256']==br['capture_sha256']
        assert P.digest(A.D.SNAPSHOT)==original['checkpoint_sha256']
        device=M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        values,controls=Q.load_layer(json.loads(Q.NATIVE.read_text()),device);assert controls==pair['controls']
        with safe_open(str(F.D.ENCODED),framework='pt',device='cpu') as archive:
            tensors={k:archive.get_tensor(k) for k in archive.keys() if not k.startswith('e160.')}
        old_keys=set(tensors)
        with safe_open(str(BASIS),framework='pt',device='cpu') as archive:tensors['private.left']=archive.get_tensor('residual.left')
        assert tensors['private.left'].shape==(16,896,32) and tensors['private.left'].dtype==torch.bfloat16
        for bank,n in (('anchor',16),('child',160)):
            tensors[bank+'.right']=torch.empty((n,32,4864),dtype=torch.float64)
            tensors[bank+'.bias']=torch.empty((n,896),dtype=torch.float64)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            x=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            y=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        parents=tensors['router.parents'].to(device);children=tensors['router.children'].to(device)
        lp=R.assign_full(x,parents);lc=torch.empty_like(lp)
        for parent in range(16):
            chosen=lp==parent;lc[chosen]=parent*10+R.assign_full(x[chosen],children[parent])
        assert torch.bincount(lp,minlength=16).tolist()==pair['counts16'] and torch.bincount(lc,minlength=160).tolist()==pair['counts160']
        route=json.loads(Q.S.ROUTE_RESULT.read_text())
        for label,key in ((lp,'parent_labels_sha256'),(lc,'leaf_labels_sha256')):
            assert P.M17.sha(label.cpu().numpy().astype('<i4').tobytes())==route['fit'][key]
        energy=float(y.double().square().sum());hashes={'e16':[],'e160':[]};source=None;source_hashes={}
        stage='matched_parent_residual_and_full_matched_parent_anchored_children'
        for parent in range(16):
            chosen=lp==parent;phi=Q.features(x[chosen],values);target=y[chosen];local=lc[chosen]
            base=C.get(tensors,'base',parent,device);factors=B.factor_get(tensors,'e16',parent,device)
            reference=B.encoded_value(phi,base,factors);sse_reference=float((reference-target).double().square().sum())
            previous=next(r for r in pair['fit_rows'] if r['arm']=='e16' and r['cell']==parent);assert sse_reference==previous['actual_factorized_sse']
            reference_coeff=C.decode(base).double()+factors[1].float().double()@factors[0].float().double()
            assert P.M17.sha(reference_coeff.cpu().numpy().tobytes())==pair['decoded_FP64_coefficient_hashes']['e16'][parent]
            left=tensors['private.left'][parent].to(device).float().double();gram=left.T@left
            condition=float(torch.linalg.cond(gram));pinv=torch.linalg.solve(gram,left.T)
            inverse_error=float((pinv@left-torch.eye(32,dtype=torch.float64,device=device)).norm());assert condition<=1e8 and inverse_error<=1e-8
            variance=tensors['fit.feature_variance'][parent].to(device);zero=torch.zeros((32,4864),dtype=torch.float64,device=device)
            ar,ab,anchor,_,audit=F.fit_child(phi,target,reference,left,pinv,variance,zero)
            effective=reference_coeff+left@ar;growth=float((left@ar).norm()/reference_coeff.norm());assert growth<=.25
            tensors['anchor.right'][parent].copy_(ar.cpu());tensors['anchor.bias'][parent].copy_(ab.cpu())
            hashes['e16'].append(P.M17.sha(effective.cpu().numpy().tobytes()))
            anchors.append({'parent':parent,'reference_sse':sse_reference,'sse':float((anchor-target.double()).square().sum()),
                'left_gram_condition':condition,'left_inverse_identity_error':inverse_error,'effective_increment_relative_norm':growth,**audit})
            for child in range(10):
                leaf=parent*10+child;selected=local==leaf;z=phi[selected];yy=target[selected];pp=anchor[selected]
                supported=bool(torch.count_nonzero(z.double()-z.double().mean(0)));prior=zero;fb={}
                if not supported:
                    if source is None:
                        path=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True));assert P.digest(path)==P.M57.MODEL_SHA
                        source=[]
                        with safe_open(str(path),framework='pt',device='cpu') as archive:
                            for organ in ('gate','up','down'):
                                name=f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight';w=archive.get_tensor(name);assert w.dtype==torch.bfloat16
                                source_hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes());source.append(w.to(device).float())
                        assert source_hashes==original['source_tensor_sha256']
                    prior,fb=fallback(children[parent,child],base,factors,left,pinv,variance,values,source,effective,ar,ab)
                delta,bias,pred,_,audit=F.fit_child(z,yy,pp,left,pinv,variance,prior)
                growth=float((left@delta).norm()/effective.norm());assert growth<=.25
                tensors['child.right'][leaf].copy_(delta.cpu());tensors['child.bias'][leaf].copy_(bias.cpu())
                hashes['e160'].append(P.M17.sha((effective+left@delta).cpu().numpy().tobytes()))
                fits.append({'child':leaf,'supported_centered_features':supported,'source_fallback':fb,
                    'sse':float((pred-yy.double()).square().sum()),'effective_increment_relative_norm':growth,**audit});P.budget(start,device)
            args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'anchor_rows':anchors,'fit_rows':fits},indent=2)+'\n',encoding='utf-8')
        stage='continuous_snapshot_and_unchanged_reference_readback'
        save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-251','scope':'nondeployable FP64 matched latent correction hierarchy',
            'parent_snapshot_sha256':pair['checkpoint_sha256'],'basis_sha256':BASIS_SHA,'tau':str(C.TAU),'rank':'32'})
        assert args.checkpoint.stat().st_size<320000000
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive,safe_open(str(F.D.ENCODED),framework='pt',device='cpu') as previous:
            assert set(archive.keys())==set(tensors)
            for name,t in tensors.items():
                assert torch.equal(archive.get_tensor(name),t) and torch.isfinite(t).all()
                if name in old_keys:assert torch.equal(t,previous.get_tensor(name))
        reference_sse=sum(r['reference_sse'] for r in anchors);anchor_sse=sum(r['sse'] for r in anchors);child_sse=sum(r['sse'] for r in fits)
        assert reference_sse/energy==pair['fit_summary']['e16']['factorized_normalized_sse']
        gates={'all_source_basis_capture_route_native_bindings':True,'all_reference_parent_fields_coefficients_scores_unchanged':True,
            'all16_matched_anchor_and160_child_numerical_controls':True,'all_snapshot_tensors_readback_exact':True,
            'all176_matched_decoded_coefficients_distinct':len(set(hashes['e16']+hashes['e160']))==176,
            'fit_function_nmse_le_001':child_sse/energy<=.01,'fit_anchor_no_worse_than_reference':anchor_sse<=reference_sse*(1+1e-12),
            'fit_child_no_worse_than_matched_anchor':child_sse<=anchor_sse*(1+1e-12)}
        summary={}
        if all(gates.values()):
            del x,y,phi,target,reference,anchor,z,yy,pp,pred,source
            stage='consumed_validation_once_after_all_fit_prerequisites'
            with np.load(R.CAPTURE,allow_pickle=False) as archive:
                x=torch.from_numpy(archive['x_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
                y=torch.from_numpy(archive['y_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            pv=R.assign_full(x,parents);cv=torch.empty_like(pv)
            for parent in range(16):
                selected=pv==parent
                if bool(selected.any()):cv[selected]=parent*10+R.assign_full(x[selected],children[parent])
            rotated=cv//10*10+(cv%10+1)%10;phi=Q.features(x,values);ref=B.predict(phi,pv,tensors,'e16',device)
            anchor=predict(phi,pv,ref,tensors,'anchor',device)
            predictions={'e16_reference':ref.double(),'e16':anchor,'e160':predict(phi,cv,anchor,tensors,'child',device),
                'e160_rotated':predict(phi,rotated,anchor,tensors,'child',device)}
            with safe_open(str(A.D.SNAPSHOT),framework='pt',device='cpu') as archive:
                priors={name:archive.get_tensor(name) for name in archive.keys() if name.startswith(('parent_prior.','child_prior.'))}
            for bank,labels in (('parent_prior',pv),('child_prior',cv)):predictions[bank]=C.predict_bank(phi,labels,priors,bank,device).double()
            for sequence in range(M.VALID):
                span=slice(sequence*M.SEQ,(sequence+1)*M.SEQ);ey=float(y[span].double().square().sum());assert ey==pair['validation_rows'][sequence]['energy']
                sse={bank:float((pred[span]-y[span].double()).square().sum()) for bank,pred in predictions.items()}
                controls_sse={bank:float((predictions['e16_reference' if bank=='e16' else bank][span].float()-y[span]).double().square().sum()) for bank in ('e16','parent_prior','child_prior')}
                assert all(controls_sse[bank]==pair['validation_rows'][sequence]['sse'][bank] for bank in controls_sse)
                rows.append({'validation_sequence':sequence,'energy':ey,'sse':sse,'unchanged_FP32_control_sse':controls_sse})
            ey=sum(r['energy'] for r in rows);summary={bank:{'sse':sum(r['sse'][bank] for r in rows),'normalized_sse':sum(r['sse'][bank] for r in rows)/ey} for bank in predictions}
            rng=np.random.default_rng(251252);draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
            diff=np.asarray([r['sse']['e16']-r['sse']['e160'] for r in rows]);energies=np.asarray([r['energy'] for r in rows]);gain=diff[draws].sum(1)/energies[draws].sum(1)
            summary['bootstrap']={'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),'seed':251252,'draws':10000}
            gates.update({'all_reference_prior_validation_controls_exact':True,'continuous_function_nmse_le_001':summary['e160']['normalized_sse']<=.01,
                **{'e160_sse_le_90_percent_'+bank:summary['e160']['sse']<=.9*summary[bank]['sse'] for bank in ('e16','e160_rotated','parent_prior','child_prior')},
                'paired_window_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-251-matched-residual-basis-full-parent-anchored-continuous-child-pilot',
            'basis_result_sha256':BASIS_RESULT_SHA,'basis_snapshot_sha256':BASIS_SHA,'parent_result_sha256':F.D.PAIR_SHA,
            'parent_snapshot_sha256':pair['checkpoint_sha256'],'source_sha256':pair['source_sha256'],'source_fallback_tensor_hashes':source_hashes,
            'capture_sha256':pair['capture_sha256'],'native_result_sha256':B.NATIVE_SHA,'route_result_sha256':Q.S.ROUTE_SHA,'controls':controls,
            'tau':C.TAU,'counts16':pair['counts16'],'counts160':pair['counts160'],'anchor_rows':anchors,'fit_rows':fits,
            'fit_summary':{'reference_nmse':reference_sse/energy,'matched_e16_nmse':anchor_sse/energy,'child_nmse':child_sse/energy},
            'decoded_FP64_coefficient_hashes':hashes,'gates':gates,'summary':summary,'validation_rows':rows,
            'checkpoint_sha256':P.digest(args.checkpoint),'checkpoint_bytes':args.checkpoint.stat().st_size,'runtime':P.budget(start,device),
            'script_sha256':P.digest(Path(__file__)),'torch_version':torch.__version__,
            'decision':'matched_residual_basis_count_pass_freeze_private_codec_and_kernel' if all(gates.values()) else 'stop_this_fixed_matched_residual_basis_hierarchy',
            'scope':'Matched E16/E160 extra rank32 output space. Full old and newly fitted parents retained. FP64 pilot,consumed one-layer development. No encoded/C private bank,n/DRAM/fresh/full quality/accepted rate/family-scale result.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','fit_summary','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'anchor_rows':anchors,'fit_rows':fits,'validation_rows':rows,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':main()
