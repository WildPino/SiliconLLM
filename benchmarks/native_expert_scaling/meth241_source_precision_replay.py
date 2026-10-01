#!/usr/bin/env python3
"""Diagnose source-precision mismatch on fit data, without new fitting."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch
from torch.nn import functional as F
import meth240_mixed_conditional_pair as C

P,M,R,N,Q=C.P,C.M,C.R,C.N,C.Q
PAIR=P.DOC/'meth240_mixed_conditional_pair_result.json'
PAIR_SHA='3c02724b1194f305036aae6896a489e1d19b477d7d8fe1d3ddc45fa9579fc505'
SNAPSHOT=P.ROOT/'results/native_expert_scaling/meth240_layer12_mixed_conditional_functions.safetensors'


def bf16_value(x,matrices):
    gate,up,down=matrices; x=x.bfloat16()
    return F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down).float()


def canonical_point(center,matrices):
    return bf16_value(center.repeat(4,128,1),matrices)[0,0]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args(); assert not args.out.exists()
    start,stage=time.monotonic(),'bindings'; chunks=[]; sensitivities=[]
    try:
        assert P.digest(PAIR)==PAIR_SHA and P.digest(Q.NATIVE)==Q.NATIVE_SHA
        pair=json.loads(PAIR.read_text())
        assert pair['decision']=='stop_this_fixed_full_feature_mixed_conditional_recipe'
        assert P.digest(SNAPSHOT)==pair['checkpoint_sha256'] and P.digest(R.CAPTURE)==pair['capture_sha256']
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=10*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        matrices=[]; hashes={}
        with safe_open(str(source),framework='pt',device='cpu') as archive:
            for organ in ('gate','up','down'):
                name=f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight'; w=archive.get_tensor(name)
                assert w.dtype==torch.bfloat16
                hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes()); matrices.append(w.to(device))
        assert hashes==pair['source_tensor_sha256']; fp32=[w.float() for w in matrices]
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            xb=archive['x_bf16'][:M.FIT].copy(); yb=archive['y_bf16'][:M.FIT].copy()
        x=torch.from_numpy(xb.reshape(-1,896)).view(torch.bfloat16).to(device)
        y=torch.from_numpy(yb.reshape(-1,896)).view(torch.bfloat16).to(device).float()
        s16=s32=energy=0.; mismatches=0
        stage='canonical_capture_geometry_replay'
        for first in range(0,len(x),512):
            xx=x[first:first+512].reshape(4,128,896); target=y[first:first+512]
            out16=bf16_value(xx,matrices).reshape(-1,896)
            out32=N.donor_value(xx.float(),*fp32).reshape(-1,896)
            e16=float((out16.double()-target.double()).square().sum())
            e32=float((out32.double()-target.double()).square().sum()); ey=float(target.double().square().sum())
            count=int((out16!=target).sum()); mismatches+=count
            chunks.append({'first_fit_state':first,'states':512,'bf16_capture_sse':e16,
                'fp32_capture_sse':e32,'energy':ey,'different_bf16_output_elements':count})
            s16+=e16; s32+=e32; energy+=ey; P.budget(start,device)
        native=json.loads(Q.NATIVE.read_text()); values,controls=Q.load_layer(native,device)
        with safe_open(str(SNAPSHOT),framework='pt',device='cpu') as archive:
            parents=archive.get_tensor('router.parents').to(device); children=archive.get_tensor('router.children').to(device)
            coeff={arm:tuple(archive.get_tensor(arm+'.'+field)[index].to(device) for field in C.FIELDS)
                for arm,index in (('e16',15),('child_prior',156),('e160',156))}
        lp=R.assign_full(x.float(),parents); mask=lp==15
        local=R.assign_full(x[mask].float(),children[15]); chosen=x[mask][local==6].float()
        target=y[mask][local==6]
        assert len(chosen)==pair['counts160'][156]==18 and len(torch.unique(chosen,dim=0))==1
        center=children[15,6]; assert torch.equal(center.bfloat16(),chosen[0].bfloat16())
        f32=N.donor_value(center,*fp32); f16=canonical_point(center,matrices)
        mean_target=target.double().mean(0); denom=mean_target.norm()
        candidates={'source_fp32':f32,'source_bf16':f16,
            **{arm:C.readout(Q.features(center,values),value) for arm,value in coeff.items()}}
        case={'child':156,'states':18,'unique_input_states':1,'unique_captured_outputs':len(torch.unique(target,dim=0)),
            'center_bf16_equals_observed_input':True,
            'relative_l2_to_captured_target_mean':{arm:float((pred.double()-mean_target).norm()/denom) for arm,pred in candidates.items()}}
        stage='precision_sensitivity_probe'
        for name,center in (('parent0',parents[0]),('parent15',parents[15]),('child156',children[15,6])):
            jac,_=N.donor_tangent(center,*fp32); directions=[]
            for dimension in (0,127,511,895,None):
                direction=torch.zeros_like(center)
                if dimension is None: direction=center/center.norm()
                else: direction[dimension]=1
                with torch.enable_grad():
                    _,sensitivity=torch.autograd.functional.jvp(lambda c:canonical_point(c,matrices),center,direction)
                reference=F.linear(direction,jac)
                assert torch.isfinite(sensitivity).all() and torch.isfinite(reference).all()
                directions.append({'dimension':dimension,'bf16_autograd_sensitivity_relative_difference':
                    float((sensitivity.double()-reference.double()).norm()/reference.double().norm())})
            sensitivities.append({'center':name,'directions':directions}); P.budget(start,device)
        qualifies=s16/energy<=1e-8 and s16<=.1*s32
        result={'experiment':'METH-241-source-BF16-versus-FP32-fit-target-precision-replay',
            'pair_result_sha256':PAIR_SHA,'snapshot_sha256':pair['checkpoint_sha256'],'source_sha256':P.M57.MODEL_SHA,
            'source_tensor_sha256':hashes,'capture_sha256':pair['capture_sha256'],'native_result_sha256':Q.NATIVE_SHA,
            'controls':controls,'chunks':chunks,'constant_input_case':case,'sensitivity_probes':sensitivities,
            'summary':{'bf16_capture_normalized_sse':s16/energy,'fp32_capture_normalized_sse':s32/energy,
                'bf16_mismatched_output_elements':mismatches,'fit_output_elements':len(x)*896,
                'bf16_replay_within_fixed_threshold_and_10x_better':qualifies},
            'gates':{'all_source_snapshot_capture_bindings':True,'canonical_capture_batch_geometry':True,
                'constant_input_child_replay':True,'sensitivity_outputs_finite':True},
            'decision':'precision_matched_source_replay_qualifies_new_prior_method' if qualifies else 'stop_precision_matched_prior_proposal_pending_source_replay_diagnosis',
            'script_sha256':P.digest(Path(__file__)),'runtime':P.budget(start,device),
            'scope':'Fit-only source-precision diagnosis; BF16 autograd is a rounded-backward/straight-through sensitivity, not a mathematical derivative of discrete quantization. No new fitting, validation, prior promotion, native cost, n gain or full quality.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','summary','constant_input_case','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'chunks':chunks,'sensitivities':sensitivities,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
