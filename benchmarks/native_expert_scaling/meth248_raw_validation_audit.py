#!/usr/bin/env python3
"""Diagnose raw readout generalization versus the fixed rank32 encoding."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from safetensors import safe_open
import torch
import meth247_weighted_residual_pair as B

A,C,P,M,R,Q=B.A,B.C,B.P,B.M,B.R,B.Q
PAIR=P.DOC/'meth247_weighted_residual_pair_result.json'
PAIR_SHA='a433b502960ee2c50c5ab1fa1b214d61327e4584935d27fa10682b08a4ab7ee9'
ENCODED=P.ROOT/'results/native_expert_scaling/meth247_layer12_weighted_residual_functions.safetensors'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args(); assert not args.out.exists()
    start,stage=time.monotonic(),'bindings'; fits=[]; rows=[]
    try:
        assert P.digest(PAIR)==PAIR_SHA and P.digest(A.D.PAIR)==A.D.PAIR_SHA and P.digest(B.AUDIT)==B.AUDIT_SHA
        split=json.loads(PAIR.read_text()); original_result=json.loads(A.D.PAIR.read_text()); audit=json.loads(B.AUDIT.read_text())
        assert split['decision']=='stop_this_fixed_activity_weighted_rank32_residual_pair'
        assert P.digest(ENCODED)==split['checkpoint_sha256'] and P.digest(A.D.SNAPSHOT)==original_result['checkpoint_sha256']
        assert P.digest(R.CAPTURE)==split['capture_sha256'] and P.digest(B.NATIVE)==B.NATIVE_SHA
        assert P.digest(Q.NATIVE)==Q.NATIVE_SHA and P.digest(Q.S.ROUTE_RESULT)==Q.S.ROUTE_SHA
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        values,controls=Q.load_layer(json.loads(Q.NATIVE.read_text()),device); assert controls==split['controls']
        with safe_open(str(A.D.SNAPSHOT),framework='pt',device='cpu') as archive:
            original={name:archive.get_tensor(name) for name in archive.keys()}
        with safe_open(str(ENCODED),framework='pt',device='cpu') as archive:
            encoded={name:archive.get_tensor(name) for name in archive.keys()}
        parents=original['router.parents'].to(device); children=original['router.children'].to(device)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            xf=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            yf=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            xv=torch.from_numpy(archive['x_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        lp=R.assign_full(xf,parents); lc=torch.empty_like(lp); pv=R.assign_full(xv,parents); cv=torch.empty_like(pv)
        for parent in range(16):
            mf=lp==parent; lc[mf]=parent*10+R.assign_full(xf[mf],children[parent])
            mv=pv==parent
            if bool(mv.any()): cv[mv]=parent*10+R.assign_full(xv[mv],children[parent])
        route=json.loads(Q.S.ROUTE_RESULT.read_text())
        assert torch.bincount(lp,minlength=16).tolist()==original_result['counts16'] and torch.bincount(lc,minlength=160).tolist()==original_result['counts160']
        for label,key in ((lp,'parent_labels_sha256'),(lc,'leaf_labels_sha256')):
            assert P.M17.sha(label.cpu().numpy().astype('<i4').tobytes())==route['fit'][key]
        rotated=cv//10*10+(cv%10+1)%10; hv=Q.features(xv,values)
        raw={name:torch.full((len(xv),896),float('nan'),dtype=torch.float64,device=device) for name in ('e16','e160','e160_rotated')}
        stage='exact176_raw_replays_and_consumed_input_predictions'
        for parent in range(16):
            mask=lp==parent; phi=Q.features(xf[mask],values); y=yf[mask]; local=lc[mask]
            variance=original['fit.feature_variance'][parent].to(device)
            tasks=[('e16',parent,phi,y,'parent_prior')]+[('e160',parent*10+c,phi[local==parent*10+c],y[local==parent*10+c],'child_prior') for c in range(10)]
            for bank,index,z,target,prior_bank in tasks:
                coeff,w,b,normal=A.replay_fit(z,target,C.get(original,prior_bank,index,device),variance)
                assert all(torch.equal(a,c) for a,c in zip(coeff,C.get(original,bank,index,device)))
                fit_pred=z.double()@w.T+b
                fits.append({'arm':bank,'cell':index,'states':len(z),'raw_sse':float((fit_pred-target.double()).square().sum()),'normal_relative_residual':normal})
                for arm,labels in ((bank,pv if bank=='e16' else cv),)+(() if bank=='e16' else (('e160_rotated',rotated),)):
                    chosen=labels==index
                    if bool(chosen.any()): raw[arm][chosen]=hv[chosen].double()@w.T+b
                P.budget(start,device)
            args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'fit_rows':fits},indent=2)+'\n',encoding='utf-8')
        energy_fit=float(yf.double().square().sum())
        for bank in ('e16','e160'):
            observed=sum(r['raw_sse'] for r in fits if r['arm']==bank)/energy_fit
            assert abs(observed/audit['summary'][bank]['raw_FP64_solution_sse']-1)<=1e-12
        assert all(torch.isfinite(t).all() for t in raw.values())
        del xf,yf,phi,y,tasks,z,target,w,b,coeff,fit_pred
        stage='consumed_target_error_and_fixed_factorized_replay'
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            yv=torch.from_numpy(archive['y_bf16'][M.FIT:].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        factored={name:B.predict(hv,labels,encoded,'e160' if name=='e160_rotated' else name,device)
            for name,labels in (('e16',pv),('e160',cv),('e160_rotated',rotated))}
        for sequence in range(M.VALID):
            span=slice(sequence*M.SEQ,(sequence+1)*M.SEQ); ey=float(yv[span].double().square().sum())
            assert ey==split['validation_rows'][sequence]['energy']
            sse={name:float((pred[span]-yv[span].double()).square().sum()) for name,pred in raw.items()}
            encoded_sse={name:float((pred[span]-yv[span]).double().square().sum()) for name,pred in factored.items()}
            assert all(encoded_sse[name]==split['validation_rows'][sequence]['sse'][name] for name in factored)
            rows.append({'validation_sequence':sequence,'energy':ey,'raw_sse':sse,'factorized_sse':encoded_sse})
        energy=sum(r['energy'] for r in rows); summary={}; ledger={}
        for bank in raw:
            summary[bank]={'raw_normalized_sse':sum(r['raw_sse'][bank] for r in rows)/energy,
                'factorized_normalized_sse':sum(r['factorized_sse'][bank] for r in rows)/energy}
            e=raw[bank]-yv.double(); d=factored[bank].double()-raw[bank]
            er=float(e.square().sum()); ds=float(d.square().sum()); cross=2*float((e*d).sum())
            encoded_sse=float((factored[bank].double()-yv.double()).square().sum())
            closure=abs(encoded_sse-(er+ds+cross))/max(encoded_sse,1e-12); assert closure<=1e-9
            ledger[bank]={'raw_sse_over_energy':er/energy,'factorized_FP64_subtract_sse_over_energy':encoded_sse/energy,
                'factorization_distortion_over_energy':ds/energy,'factorization_cross_term_over_energy':cross/energy,'SSE_identity_relative_closure':closure}
        rng=np.random.default_rng(248249); draws=rng.integers(0,M.VALID,size=(10000,M.VALID))
        delta=np.asarray([r['raw_sse']['e16']-r['raw_sse']['e160'] for r in rows]); ey=np.asarray([r['energy'] for r in rows])
        gain=delta[draws].sum(1)/ey[draws].sum(1)
        decisions={'raw_E160_nmse_le_001':summary['e160']['raw_normalized_sse']<=.01,
            'raw_E160_sse_le_90_percent_E16':summary['e160']['raw_normalized_sse']<=.9*summary['e16']['raw_normalized_sse'],
            'raw_E160_sse_le_90_percent_rotated':summary['e160']['raw_normalized_sse']<=.9*summary['e160_rotated']['raw_normalized_sse'],
            'raw_paired_gain_p05_positive':float(np.quantile(gain,.05))>0}
        result={'experiment':'METH-248-raw-continuous-consumed-validation-versus-rank32-factorization-audit',
            'split_result_sha256':PAIR_SHA,'raw_result_sha256':A.D.PAIR_SHA,'audit_result_sha256':B.AUDIT_SHA,
            'raw_snapshot_sha256':original_result['checkpoint_sha256'],'factorized_snapshot_sha256':split['checkpoint_sha256'],
            'source_sha256':split['source_sha256'],'capture_sha256':split['capture_sha256'],'route_result_sha256':Q.S.ROUTE_SHA,
            'fit_replay_rows':fits,'validation_rows':rows,'summary':summary,'factorization_ledger':ledger,
            'bootstrap':{'gain_p05':float(np.quantile(gain,.05)),'gain_p95':float(np.quantile(gain,.95)),'seed':248249,'draws':10000},
            'gates':{'all_snapshot_capture_source_native_route_bindings':True,'all176_original_coefficients_and_biases_exact':True,
                'all_raw_fit_scores_within_1e12_relative':True,'all_factorized_validation_window_scores_exact':True,'all_error_ledger_identities_close':True},
            'diagnostic_decisions':decisions,'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'decision':'raw_consumed_count_gain_exists_change_residual_representation' if all(decisions.values()) else 'raw_consumed_count_gain_fails_change_continuous_function_hierarchy',
            'scope':'Frozen raw readouts over same FP32 row-Q8/LUT features,not all-FP64 ancestors or donor. Fit replays unchanged;already consumed validation only. No rank/strength retry,new model/native bank/full independent quality or accepted rate.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','summary','bootstrap','diagnostic_decisions','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'fit_rows':fits,'validation_rows':rows,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
