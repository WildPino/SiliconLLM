#!/usr/bin/env python3
"""Lower-bound Jacobian error achievable by row scales alone, fixed codes."""
import argparse
import json
from pathlib import Path
import time
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch
import meth235_full_feature_output_priors as Q

P,M,R,N=Q.P,Q.M,Q.R,Q.N
PRIOR=P.DOC/'meth235_full_feature_output_prior_result.json'
PRIOR_SHA='132013d3991622143d268090015900abfcbb88e22d5799eb4df1073b4a97b140'
SNAPSHOT=P.ROOT/'results/native_expert_scaling/meth235_layer12_full_feature_output_priors.safetensors'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); assert not args.out.exists()
    start,stage=time.monotonic(),'bindings'; rows=[]
    try:
        assert P.digest(PRIOR)==PRIOR_SHA
        prior=json.loads(PRIOR.read_text()); assert not prior['gates']['stored_gradient']
        assert P.digest(SNAPSHOT)==prior['checkpoint_sha256'] and P.digest(Q.NATIVE)==Q.NATIVE_SHA
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=10*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        native=json.loads(Q.NATIVE.read_text()); values,controls=Q.load_layer(native,device)
        with safe_open(str(SNAPSHOT),framework='pt',device='cpu') as archive:
            parents=archive.get_tensor('router.parents').to(device)
            codes=archive.get_tensor('prior.down.q'); scales=archive.get_tensor('prior.down.scale')
        matrices=[]
        with safe_open(str(source),framework='pt',device='cpu') as archive:
            for organ in ('gate','up','down'):
                matrices.append(archive.get_tensor(f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight').to(device).float())
        stage='fixed_code_row_scale_lower_bound'
        for parent,center in enumerate(parents):
            source_jac,_=N.donor_tangent(center,*matrices); jd=source_jac.double()
            a=Q.feature_jacobian(center,values).double(); code=codes[parent].to(device).double()
            direction=code@a; denominator=direction.square().sum(1)
            assert (denominator>0).all()
            optimal=(direction*jd).sum(1)/denominator
            assert torch.isfinite(optimal).all() and (optimal>0).all()
            original_error=float(((direction*scales[parent].to(device).double()[:,None])-jd).norm()/jd.norm())
            old=prior['audits'][parent]['stored_gradient_relative_frobenius']
            # Old decoded-FP32 coefficients and FP64 q*scale differ by rounding only.
            assert abs(original_error-old)<=1e-7
            residual=optimal[:,None]*direction-jd
            bound=float(residual.norm()/jd.norm())
            optimal32=optimal.float()
            rounded_error=float((optimal32.double()[:,None]*direction-jd).norm()/jd.norm())
            normal=float((residual*direction).sum(1).norm()/(jd*direction).sum(1).norm())
            assert normal<=1e-10 and bound<=original_error+1e-10
            rows.append({'parent':parent,'original_relative_frobenius':original_error,
                'fp64_scale_lower_bound':bound,'fp32_scale_relative_frobenius':rounded_error,
                'least_squares_normal_relative_residual':normal,
                'positive_scale_minimum':float(optimal.min()),'positive_scale_maximum':float(optimal.max()),
                'optimal_scale_sha256':P.M17.sha(optimal32.cpu().numpy().tobytes())})
            P.budget(start,device)
        failing=[r['parent'] for r in rows if r['fp64_scale_lower_bound']>.01]
        result={'experiment':'METH-236-fixed-output-codes-row-scale-Jacobian-bound',
            'prior_result_sha256':PRIOR_SHA,'snapshot_sha256':prior['checkpoint_sha256'],
            'source_sha256':P.M57.MODEL_SHA,'native_result_sha256':Q.NATIVE_SHA,'controls':controls,
            'rows':rows,'gates':{'all_source_snapshot_bindings':True,'original_error_replay':True,'normal_equations':True},
            'summary':{'parents_unreachable_at_1percent':failing,
                'maximum_fp64_scale_lower_bound':max(r['fp64_scale_lower_bound'] for r in rows),
                'maximum_original_error':max(r['original_relative_frobenius'] for r in rows)},
            'decision':'exclude_scale_only_repair_for_these_fixed_codes' if failing else 'scale_bound_pass_freeze_actual_calibrated_prior_variant',
            'script_sha256':P.digest(Path(__file__)),'runtime':P.budget(start,device),
            'scope':'Analytic row-scale lower bound only on original source centers, fixed physical codes; no calibrated snapshot, prior promotion, fitting, validation/full-model quality or native cost retiming'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'rows':rows,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
