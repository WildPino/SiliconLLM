#!/usr/bin/env python3
"""Replay fixed fits and separate continuous fitting from readout encoding."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from safetensors import safe_open
import torch
import meth241_source_precision_replay as D

C,P,M,R,Q=D.C,D.P,D.M,D.R,D.Q
HIER=P.DOC/'meth243_parent_value_prior_result.json'
HIER_SHA='ac40b7453cf6414df62911b7d3cf87b1aeb403b8a4005921506470345af25770'


def replay_fit(z,y,prior,variance):
    zd,yd=z.double(),y.double(); wp,bp=C.decode(prior).double(),prior[4].double()
    mean=zd.mean(0); xc=zd-mean
    residual=yd-(zd@wp.T+bp); mr=residual.mean(0); rc=residual-mr
    n,features=xc.shape
    if n<features:
        scaled=xc/variance[None,:]
        gram=scaled@xc.T+C.TAU*torch.eye(n,dtype=torch.float64,device=z.device)
        delta_t=scaled.T@torch.linalg.solve(gram,rc)
    else:
        gram=xc.T@xc+C.TAU*torch.diag(variance)
        delta_t=torch.linalg.solve(gram,xc.T@rc)
    cross=xc.T@rc
    normal=xc.T@(xc@delta_t-rc)+C.TAU*variance[:,None]*delta_t
    relative=float(normal.norm()/cross.norm().clamp_min(1e-12)); assert relative<=1e-7
    raw_w=wp+delta_t.T; shift=mr*(n/(n+C.TAU))
    raw_b=bp+shift-(raw_w-wp)@mean
    q,s,ids,escape=Q.L.encode_mixed(raw_w.float())
    effective=Q.L.decode_mixed(q,s,ids,escape).double()
    bias=(bp+shift-(effective-wp)@mean).float()
    return (q,s,ids,escape,bias),raw_w,raw_b,relative


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args(); assert not args.out.exists()
    start,stage=time.monotonic(),'bindings'; rows=[]
    try:
        assert P.digest(D.PAIR)==D.PAIR_SHA and P.digest(HIER)==HIER_SHA
        pair=json.loads(D.PAIR.read_text()); hierarchy=json.loads(HIER.read_text())
        assert pair['source_sha256']==P.M57.MODEL_SHA
        assert hierarchy['decision']=='stop_fixed_parent_value_source_slope_hierarchy_as_count_solution'
        assert P.digest(D.SNAPSHOT)==pair['checkpoint_sha256'] and P.digest(R.CAPTURE)==pair['capture_sha256']
        assert P.digest(Q.NATIVE)==Q.NATIVE_SHA and P.digest(Q.S.ROUTE_RESULT)==Q.S.ROUTE_SHA
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=10*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        values,controls=Q.load_layer(json.loads(Q.NATIVE.read_text()),device); assert controls==pair['controls']
        with safe_open(str(D.SNAPSHOT),framework='pt',device='cpu') as archive:
            tensors={name:archive.get_tensor(name) for name in archive.keys()}
        parents=tensors['router.parents'].to(device); children=tensors['router.children'].to(device)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            xf=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            yf=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        lp=R.assign_full(xf,parents); lc=torch.empty_like(lp)
        for parent in range(16):
            mask=lp==parent; lc[mask]=parent*10+R.assign_full(xf[mask],children[parent])
        route=json.loads(Q.S.ROUTE_RESULT.read_text())
        assert torch.bincount(lp,minlength=16).tolist()==pair['counts16']
        assert torch.bincount(lc,minlength=160).tolist()==pair['counts160']
        for labels,key in ((lp,'parent_labels_sha256'),(lc,'leaf_labels_sha256')):
            assert P.M17.sha(labels.cpu().numpy().astype('<i4').tobytes())==route['fit'][key]
        stage='exact_fixed176_solve_replay_and_encoding_error_ledger'
        for parent in range(16):
            mask=lp==parent; phi=Q.features(xf[mask],values); y=yf[mask]; local=lc[mask]
            variance=tensors['fit.feature_variance'][parent].to(device)
            for arm,index,z,target,prior_arm in [('e16',parent,phi,y,'parent_prior')]+[
                ('e160',parent*10+child,phi[local==parent*10+child],y[local==parent*10+child],'child_prior') for child in range(10)]:
                prior=C.get(tensors,prior_arm,index,device)
                coeff,raw_w,raw_b,normal=replay_fit(z,target,prior,variance)
                stored=C.get(tensors,arm,index,device)
                assert all(torch.equal(a,b) for a,b in zip(coeff,stored))
                raw=z.double()@raw_w.T+raw_b
                decoded=z.double()@C.decode(stored).double().T+stored[4].double()
                actual=C.readout(z,stored).double(); target=target.double()
                eraw=raw-target; distortion=decoded-raw
                raw_sse=float(eraw.square().sum()); decoded_sse=float((decoded-target).square().sum())
                distortion_sse=float(distortion.square().sum()); cross=2*float((eraw*distortion).sum())
                closure=abs(decoded_sse-(raw_sse+distortion_sse+cross))/max(decoded_sse,1e-12)
                assert closure<=1e-9 and torch.isfinite(raw).all()
                rows.append({'arm':arm,'cell':index,'states':len(z),'energy':float(target.square().sum()),
                    'raw_FP64_solution_sse':raw_sse,'decoded_FP64_stored_sse':decoded_sse,
                    'actual_mixed_stored_sse':float((actual.float()-target.float()).double().square().sum()),
                    'encoding_distortion_sse':distortion_sse,'encoding_cross_term':cross,
                    'SSE_identity_relative_closure':closure,'normal_equation_relative_residual':normal,
                    'arithmetic_delta_sse':float((actual-decoded).square().sum()),
                    'prior_sse':float((C.readout(z,prior).double()-target).square().sum()),
                    'stored_parent_sse':float((C.readout(z,C.get(tensors,'e16',parent,device)).double()-target).square().sum())})
                P.budget(start,device)
            args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'rows':rows},indent=2)+'\n',encoding='utf-8')
        summary={}
        for arm in ('e16','e160'):
            selected=[r for r in rows if r['arm']==arm]; energy=sum(r['energy'] for r in selected)
            metrics=('raw_FP64_solution_sse','decoded_FP64_stored_sse','actual_mixed_stored_sse',
                'encoding_distortion_sse','encoding_cross_term','arithmetic_delta_sse','prior_sse','stored_parent_sse')
            summary[arm]={'energy':energy,**{key:sum(r[key] for r in selected)/energy for key in metrics}}
            assert summary[arm]['actual_mixed_stored_sse']==pair['fit_normalized_sse'][arm]
        assert summary['e160']['energy']==summary['e16']['energy']
        raw_gain=summary['e160']['raw_FP64_solution_sse']<=.9*summary['e16']['raw_FP64_solution_sse']
        actual_gap=summary['e160']['actual_mixed_stored_sse']-summary['e16']['actual_mixed_stored_sse']
        penalty=summary['e160']['actual_mixed_stored_sse']-summary['e160']['raw_FP64_solution_sse']
        encoding_explains_gap=actual_gap>0 and penalty>=.5*actual_gap
        result={'experiment':'METH-244-fixed-readout-continuous-versus-stored-fit-error-audit',
            'pair_result_sha256':D.PAIR_SHA,'hierarchy_result_sha256':HIER_SHA,
            'snapshot_sha256':pair['checkpoint_sha256'],'source_sha256':pair['source_sha256'],'capture_sha256':pair['capture_sha256'],
            'native_result_sha256':Q.NATIVE_SHA,'route_result_sha256':Q.S.ROUTE_SHA,'controls':controls,
            'prior_sample_equivalents':C.TAU,'rows':rows,'summary':summary,
            'gates':{'all_source_snapshot_capture_route_bindings':True,'exact_fit_labels_counts':True,
                'all176_encoded_coefficients_and_biases_replay_exact':len(rows)==176,
                'actual_stored_fit_scores_replay_exact':True,'all_FP64_SSE_identities_close':True},
            'diagnostic_decisions':{'raw_E160_fit_sse_le_90_percent_raw_E16':raw_gain,
                'encoding_penalty_ge_half_actual_positive_count_gap':encoding_explains_gap},
            'decision':'continuous_local_count_gain_exposed_change_readout_encoding' if raw_gain and encoding_explains_gap else 'continuous_local_count_gain_absent_do_not_retry_codec_only',
            'script_sha256':P.digest(Path(__file__)),'runtime':P.budget(start,device),
            'scope':'All176 frozen METH-240 fit systems replayed,not a new training recipe. FP64 ideal is a nondeployable fit-only diagnostic. No validation/new data/quality/native timing or replacement checkpoint.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','diagnostic_decisions','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'rows':rows,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
