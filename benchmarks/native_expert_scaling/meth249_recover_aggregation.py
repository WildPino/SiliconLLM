#!/usr/bin/env python3
"""Recover completed M249 coefficients; only new consumed validation, no fits."""
import argparse
import json
import time
import numpy as np
from safetensors import safe_open
import torch
import meth249_parent_anchored_latent_pilot as F

B,A,C,P,M,R,Q=F.B,F.A,F.C,F.P,F.M,F.R,F.Q
FAIL=P.DOC/'meth249_parent_anchored_latent_repair1_result.failure.json'
FAIL_SHA='105535b560e1e0eee2ed7a04eb3750515bbd94d99755c45d0e9c808cad8a42fb'
CHECKPOINT=P.ROOT/'results/native_expert_scaling/meth249_layer12_parent_anchored_continuous_repair1.safetensors'
CHECKPOINT_SHA='5b95ecc272b257929388ac6520cfcbdf3643a9e4aa4aa21c0acd1e69482df7a0'
DIAGNOSTIC=P.DOC/'meth249_parent_replay_diagnostic.json'
DIAGNOSTIC_SHA='21825bbbff61fa58dd4c3f7bfa169690197af14fc96e90b5b4f6bbbc2cc0c36e'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=F.Path)
    args=ap.parse_args();assert not args.out.exists();start=time.monotonic();stage='bindings';rows=[]
    try:
        for path,sha in ((FAIL,FAIL_SHA),(CHECKPOINT,CHECKPOINT_SHA),(DIAGNOSTIC,DIAGNOSTIC_SHA),
                         (F.D.PAIR,F.D.PAIR_SHA),(F.DIAG,F.DIAG_SHA),(A.D.PAIR,A.D.PAIR_SHA),
                         (B.NATIVE,B.NATIVE_SHA),(Q.NATIVE,Q.NATIVE_SHA),(Q.S.ROUTE_RESULT,Q.S.ROUTE_SHA)):
            assert P.digest(path)==sha
        failure=json.loads(FAIL.read_text());diagnostic=json.loads(DIAGNOSTIC.read_text());pair=json.loads(F.D.PAIR.read_text())
        original=json.loads(A.D.PAIR.read_text())
        assert P.digest(F.D.ENCODED)==pair['checkpoint_sha256'] and P.digest(A.D.SNAPSHOT)==original['checkpoint_sha256']
        assert P.digest(R.CAPTURE)==pair['capture_sha256']
        assert failure['stage']=='original_exact_parent_fit_score_replay' and failure['error']=='AssertionError()'
        assert failure['validation_rows']==[] and len(failure['fit_rows'])==160
        assert [r['child'] for r in failure['fit_rows']]==list(range(160))
        assert [r['states'] for r in failure['fit_rows']]==pair['counts160']
        assert [r['parent'] for r in failure['parent_replays']]==list(range(16))
        assert all(r['sse']==r['old_sse'] and r['relative_discrepancy']==0 for r in failure['parent_replays'])
        energy=diagnostic['energy'];parent_sse=sum(r['sse'] for r in failure['parent_replays'])
        assert parent_sse/energy==diagnostic['observed_nmse']==diagnostic['old_nmse']==pair['fit_summary']['e16']['factorized_normalized_sse']
        assert all(r['observed_sse']==failure['parent_replays'][r['parent']]['sse'] for r in diagnostic['rows'])
        fits=failure['fit_rows'];child_sse=sum(r['sse'] for r in fits)
        for r in fits:
            assert r['normal_relative_residual']<=1e-7 and r['mean_conservation_relative_l2']<=1e-10
            assert r['effective_increment_relative_norm']<=.25 and r['mean_intercept_shrink']==r['states']/(r['states']+C.TAU)
            assert r['supported_centered_features']==(r['centered_feature_energy']>0)
            fb=r['source_fallback']
            if not r['supported_centered_features']:
                assert fb['feature_checks']['fixed16_row_relative_frobenius']<=1e-5
                assert all(k['relative_l2']<=1e-5 for k in fb['feature_checks']['directions'])
                assert fb['parent_autograd_relative_check']<=1e-5 and fb['whitened_feature_condition_number']<=1e8
                assert fb['projected_source_gradient_relative_residual']<=1e-5 and fb['effective_increment_relative_norm']<=.25
            else:assert not fb
        device=M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=5*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        values,controls=Q.load_layer(json.loads(Q.NATIVE.read_text()),device);assert controls==pair['controls']
        with safe_open(str(CHECKPOINT),framework='pt',device='cpu') as archive,safe_open(str(F.D.ENCODED),framework='pt',device='cpu') as previous:
            old_keys={k for k in previous.keys() if not k.startswith('e160.')}
            assert set(archive.keys())==old_keys|{'child.delta_right','child.delta_bias','diagnostic.child_mean_phi'}
            tensors={k:archive.get_tensor(k) for k in archive.keys()}
            for name in old_keys:assert torch.equal(tensors[name],previous.get_tensor(name))
            assert all(torch.isfinite(t).all() for t in tensors.values())
        assert CHECKPOINT.stat().st_size<320000000
        hashes={'e16':[],'e160':[]};inverses=[];stage='completed_snapshot_coefficient_and_inverse_audit'
        for parent in range(16):
            base=C.get(tensors,'base',parent,device);right,left,_=B.factor_get(tensors,'e16',parent,device);left=left.float().double()
            gram=left.T@left;condition=float(torch.linalg.cond(gram));pinv=torch.linalg.solve(gram,left.T)
            error=float((pinv@left-torch.eye(32,dtype=torch.float64,device=device)).norm())
            assert condition<=1e8 and error<=1e-8
            inverses.append({'parent':parent,'left_gram_condition':condition,'left_inverse_identity_error':error})
            effective=C.decode(base).double()+left@right.float().double()
            hashes['e16'].append(P.M17.sha(effective.cpu().numpy().tobytes()))
            assert hashes['e16'][-1]==pair['decoded_FP64_coefficient_hashes']['e16'][parent]
            for child in range(10):
                leaf=parent*10+child;delta=tensors['child.delta_right'][leaf].to(device)
                growth=float((left@delta).norm()/effective.norm().clamp_min(1e-12));assert growth<=.25
                assert abs(growth-fits[leaf]['effective_increment_relative_norm'])<=1e-12
                hashes['e160'].append(P.M17.sha((effective+left@delta).cpu().numpy().tobytes()))
            P.budget(start,device)
        gates={'all_source_parent_capture_route_native_bindings':True,'all_retained_fit_labels_counts_and_exact_parent_scores':True,
            'all16_parent_fields_coefficients_unchanged':True,'all16_left_inverses_qualified':True,
            'all160_retained_solve_mean_and_recomputed_growth_controls':True,'all_snapshot_tensors_readback_exact':True,
            'all176_decoded_coefficients_distinct':len(set(hashes['e16']+hashes['e160']))==176,
            'fit_function_nmse_le_001':child_sse/energy<=.01,'fit_child_sse_no_worse_than_parent':child_sse<=parent_sse*(1+1e-12)}
        summary={}
        if all(gates.values()):
            stage='consumed_validation_once_after_all_fit_prerequisites'
            with np.load(R.CAPTURE,allow_pickle=False) as archive:
                x=torch.from_numpy(archive['x_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
                y=torch.from_numpy(archive['y_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            parents=tensors['router.parents'].to(device);children=tensors['router.children'].to(device)
            pv=R.assign_full(x,parents);cv=torch.empty_like(pv)
            for parent in range(16):
                chosen=pv==parent
                if bool(chosen.any()):cv[chosen]=parent*10+R.assign_full(x[chosen],children[parent])
            rotated=cv//10*10+(cv%10+1)%10;phi=Q.features(x,values);pp=B.predict(phi,pv,tensors,'e16',device)
            predictions={'e16':pp.double(),'e160':F.child_predict(phi,pp,cv,tensors,device),
                'e160_rotated':F.child_predict(phi,pp,rotated,tensors,device)}
            with safe_open(str(A.D.SNAPSHOT),framework='pt',device='cpu') as archive:
                priors={name:archive.get_tensor(name) for name in archive.keys() if name.startswith(('parent_prior.','child_prior.'))}
            for bank,labels in (('parent_prior',pv),('child_prior',cv)):
                predictions[bank]=C.predict_bank(phi,labels,priors,bank,device).double()
            for sequence in range(M.VALID):
                span=slice(sequence*M.SEQ,(sequence+1)*M.SEQ);ey=float(y[span].double().square().sum())
                assert ey==pair['validation_rows'][sequence]['energy']
                sse={bank:float((pred[span]-y[span].double()).square().sum()) for bank,pred in predictions.items()}
                old={bank:float((predictions[bank][span].float()-y[span]).double().square().sum()) for bank in ('e16','parent_prior','child_prior')}
                assert all(old[bank]==pair['validation_rows'][sequence]['sse'][bank] for bank in old)
                rows.append({'validation_sequence':sequence,'energy':ey,'sse':sse,'unchanged_FP32_control_sse':old})
            ey=sum(r['energy'] for r in rows)
            summary={bank:{'sse':sum(r['sse'][bank] for r in rows),'normalized_sse':sum(r['sse'][bank] for r in rows)/ey} for bank in predictions}
            rng=np.random.default_rng(249250);draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
            diff=np.asarray([r['sse']['e16']-r['sse']['e160'] for r in rows]);energies=np.asarray([r['energy'] for r in rows])
            gain=diff[draws].sum(1)/energies[draws].sum(1)
            summary['bootstrap']={'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),'seed':249250,'draws':10000}
            gates.update({'all_parent_prior_validation_controls_exact':True,'continuous_function_nmse_le_001':summary['e160']['normalized_sse']<=.01,
                **{'e160_sse_le_90_percent_'+bank:summary['e160']['sse']<=.9*summary[bank]['sse'] for bank in ('e16','e160_rotated','parent_prior','child_prior')},
                'paired_window_gain_p05_positive':float(np.quantile(gain,.05))>0})
        result={'experiment':'METH-249-full-parent-anchored-continuous-latent-aggregation-recovery',
            'producer_failure_sha256':FAIL_SHA,'parent_diagnostic_sha256':DIAGNOSTIC_SHA,'parent_result_sha256':F.D.PAIR_SHA,
            'diagnosis_result_sha256':F.DIAG_SHA,'parent_snapshot_sha256':pair['checkpoint_sha256'],
            'source_sha256':pair['source_sha256'],'capture_sha256':pair['capture_sha256'],'controls':controls,
            'native_result_sha256':B.NATIVE_SHA,'route_result_sha256':Q.S.ROUTE_SHA,'tau':C.TAU,
            'counts16':pair['counts16'],'counts160':pair['counts160'],'left_inverse_controls':inverses,'fit_rows':fits,
            'fit_summary':{'parent_nmse':parent_sse/energy,'child_nmse':child_sse/energy},'decoded_FP64_coefficient_hashes':hashes,
            'gates':gates,'summary':summary,'validation_rows':rows,'checkpoint_sha256':CHECKPOINT_SHA,'checkpoint_bytes':CHECKPOINT.stat().st_size,
            'original_fit_seconds':failure['seconds'],'runtime':P.budget(start,device),'script_sha256':P.digest(F.Path(__file__)),
            'decision':'continuous_parent_anchor_count_pass_freeze_private_codec_and_kernel' if all(gates.values()) else 'stop_this_fixed_full_parent_tied_latent_hierarchy',
            'scope':'No new fits. Full actual parent preserved;private FP64 residuals in fixed parent output space. Source sensitivity only for zero support. Consumed one-layer development,not deployable/fresh/full quality/n/DRAM/accepted rate.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','fit_summary','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'validation_rows':rows,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':main()
