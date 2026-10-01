#!/usr/bin/env python3
"""Fit-only diagnosis of inherited versus actual-parent-error output space."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import meth249_recover_aggregation as F

B,A,C,P,M,R,Q=F.B,F.A,F.C,F.P,F.M,F.R,F.Q
RESULT=P.DOC/'meth249_parent_anchored_latent_result.json'
RESULT_SHA='d4bd08d5e5c0773ef33a5ddbd1fa3aa2292e585bbc3f198697ba543c88a78c0e'


def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','out'):ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args();assert not args.out.exists() and not args.checkpoint.exists()
    start=time.monotonic();stage='bindings';rows=[]
    try:
        assert P.digest(RESULT)==RESULT_SHA and P.digest(F.F.D.PAIR)==F.F.D.PAIR_SHA
        result=json.loads(RESULT.read_text());pair=json.loads(F.F.D.PAIR.read_text())
        assert result['decision']=='stop_this_fixed_full_parent_tied_latent_hierarchy'
        assert P.digest(F.F.D.ENCODED)==pair['checkpoint_sha256'] and P.digest(R.CAPTURE)==pair['capture_sha256']
        assert P.digest(Q.NATIVE)==Q.NATIVE_SHA and P.digest(Q.S.ROUTE_RESULT)==Q.S.ROUTE_SHA
        device=M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=10*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        values,controls=Q.load_layer(json.loads(Q.NATIVE.read_text()),device);assert controls==pair['controls']
        with safe_open(str(F.F.D.ENCODED),framework='pt',device='cpu') as archive:
            parents={k:archive.get_tensor(k) for k in archive.keys() if not k.startswith('e160.')}
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            x=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
            y=torch.from_numpy(archive['y_bf16'][:M.FIT].reshape(-1,896).copy()).view(torch.bfloat16).to(device).float()
        labels=R.assign_full(x,parents['router.parents'].to(device));assert torch.bincount(labels,minlength=16).tolist()==pair['counts16']
        route=json.loads(Q.S.ROUTE_RESULT.read_text())
        assert P.M17.sha(labels.cpu().numpy().astype('<i4').tobytes())==route['fit']['parent_labels_sha256']
        bases=[];stored=[];stage='actual_parent_error_covariance_and_fixed_rank32_projection'
        for parent in range(16):
            chosen=labels==parent;phi=Q.features(x[chosen],values)
            base=C.get(parents,'base',parent,device);factors=B.factor_get(parents,'e16',parent,device)
            prediction=B.encoded_value(phi,base,factors);old=next(r for r in pair['fit_rows'] if r['arm']=='e16' and r['cell']==parent)
            sse32=float((prediction-y[chosen]).double().square().sum());assert sse32==old['actual_factorized_sse']
            residual=y[chosen].double()-prediction.double();mean=residual.mean(0);centered=residual-mean
            total=float(residual.square().sum());energy=float(centered.square().sum());covariance=centered.T@centered
            eigen,u=torch.linalg.eigh(covariance);assert float(eigen[0])>=-1e-10*float(eigen[-1])
            assert abs(float(eigen.sum())/energy-1)<=1e-8
            basis=u[:,-32:].flip(1);largest=eigen[-32:].flip(0)
            error=float((covariance@basis-basis*largest[None,:]).norm()/covariance.norm())
            orth=float((basis.T@basis-torch.eye(32,dtype=torch.float64,device=device)).norm());assert error<=1e-8 and orth<=1e-8
            ids=basis.abs().argmax(0);signs=torch.sign(basis[ids,torch.arange(32,device=device)])
            assert bool((signs.abs()==1).all());basis=(basis*signs[None,:]).contiguous()
            left=basis.bfloat16().contiguous();rounded=left.float().double()
            gram=rounded.T@rounded;condition=float(torch.linalg.cond(gram));pinv=torch.linalg.solve(gram,rounded.T)
            inverse_error=float((pinv@rounded-torch.eye(32,dtype=torch.float64,device=device)).norm())
            assert condition<=1e8 and inverse_error<=1e-8
            old_left=factors[1].float().double();old_pinv=torch.linalg.solve(old_left.T@old_left,old_left.T)
            old_projection=(centered@old_pinv.T)@old_left.T
            new_projection=(centered@pinv.T)@rounded.T
            old_remaining=float((centered-old_projection).square().sum())
            new_remaining=float((centered-new_projection).square().sum());optimal_remaining=energy-float(largest.sum())
            assert new_remaining>=optimal_remaining-1e-10*energy
            assert abs(float((centered@basis).square().sum())/float(largest.sum())-1)<=1e-8
            bases.append(basis.cpu());stored.append(left.cpu())
            rows.append({'parent':parent,'states':len(phi),'exact_old_FP32_sse':sse32,'residual_FP64_sse':total,
                'residual_mean_sse':len(phi)*float(mean.square().sum()),'centered_residual_sse':energy,
                'old_left_captured_centered_sse':energy-old_remaining,'optimal_rank32_captured_centered_sse':float(largest.sum()),
                'stored_new_left_captured_centered_sse':energy-new_remaining,'stored_new_left_optimistic_full_residual_sse':new_remaining,
                'eigen_relative_residual':error,'basis_orthogonality_frobenius':orth,
                'stored_left_gram_condition':condition,'stored_left_inverse_identity_error':inverse_error})
            P.budget(start,device)
            args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'rows':rows},indent=2)+'\n',encoding='utf-8')
        fit_energy=float(y.double().square().sum());assert sum(r['exact_old_FP32_sse'] for r in rows)/fit_energy==pair['fit_summary']['e16']['factorized_normalized_sse']
        weights={'diagnostic.left_FP64':torch.stack(bases),'residual.left':torch.stack(stored)}
        save_file(weights,str(args.checkpoint),metadata={'experiment':'METH-250','scope':'fit-only basis;not learned child functions','parent_snapshot_sha256':pair['checkpoint_sha256']})
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
            assert set(archive.keys())==set(weights)
            assert all(torch.equal(archive.get_tensor(k),t) for k,t in weights.items())
        total=sum(r['residual_FP64_sse'] for r in rows);centered=sum(r['centered_residual_sse'] for r in rows)
        old_captured=sum(r['old_left_captured_centered_sse'] for r in rows);new_captured=sum(r['stored_new_left_captured_centered_sse'] for r in rows)
        floor=sum(r['stored_new_left_optimistic_full_residual_sse'] for r in rows)
        summary={'fit_parent_nmse':sum(r['exact_old_FP32_sse'] for r in rows)/fit_energy,
            'old_left_centered_error_fraction':old_captured/centered,'new_left_centered_error_fraction':new_captured/centered,
            'optimistic_new_left_full_residual_gain':1-floor/total,'old_left_captured_sse':old_captured,'new_left_captured_sse':new_captured}
        gates={'all16_exact_parent_score_and_route_controls':True,'all16_eigen_inverse_projection_controls':True,
            'physical_basis_readback_exact':True,'stored_basis_can_remove_at_least_10_percent_centered_error':new_captured>=.1*centered,
            'stored_basis_improves_old_space_by_at_least_10_percent_centered_error':new_captured-old_captured>=.1*centered}
        output={'experiment':'METH-250-fit-only-actual-parent-residual-output-space-diagnosis','preceding_result_sha256':RESULT_SHA,
            'parent_result_sha256':F.F.D.PAIR_SHA,'parent_snapshot_sha256':pair['checkpoint_sha256'],'capture_sha256':pair['capture_sha256'],
            'source_sha256':pair['source_sha256'],'route_result_sha256':Q.S.ROUTE_SHA,'controls':controls,'rows':rows,'summary':summary,'gates':gates,
            'checkpoint_sha256':P.digest(args.checkpoint),'checkpoint_bytes':args.checkpoint.stat().st_size,'runtime':P.budget(start,device),
            'script_sha256':P.digest(Path(__file__)),
            'decision':'freeze_full_parent_private_residual_basis_pilot' if all(gates.values()) else 'stop_this_fixed_residual_output_basis_before_child_fitting',
            'scope':'Fit-only error projection/optimistic arbitrary correction upper bound. No child training or validation labels,learned count gain,deployable/native/full quality/rate.'}
        args.out.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:output[k] for k in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'rows':rows,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':main()
