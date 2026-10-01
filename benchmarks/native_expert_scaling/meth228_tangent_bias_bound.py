#!/usr/bin/env python3
"""Replay frozen functions; bound any bias-only repair on recorded products."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from safetensors import safe_open
import torch
import meth227_conditional_donor_tangents as R

P,M=R.P,R.M
PREVIOUS=P.DOC/'meth227_conditional_donor_tangent_result.json'
PREVIOUS_SHA='85f5b41e99850a2eeb32d0c071f32060f35d0e958db4729fffa115681a406fd2'
CHECKPOINT=P.ROOT/'results/native_expert_scaling/meth227_layer12_tangent_functions.safetensors'


def decompose(recorded,prebias,labels,count,energy,old_sse):
    rows=[]; within,bias_energy,oracle=0.,0.,0.
    for ci in range(count):
        mask=labels==ci; n=int(mask.sum())
        if not n:
            rows.append({'cell':ci,'states':0}); continue
        r,a=recorded[mask],prebias[mask]
        mr,ma=r.mean(0),a.mean(0)
        w=float(np.square(r-mr).sum()); b=float(n*np.square(mr).sum())
        ideal=float(np.square(a-ma).sum())
        within+=w; bias_energy+=b; oracle+=ideal
        rows.append({'cell':ci,'states':n,'recorded_within_cell_sse':w,
            'recorded_cell_mean_error_energy':b,'ideal_real_bias_oracle_sse':ideal})
    discrepancy=abs(within+bias_energy-old_sse)/old_sse
    assert discrepancy<=1e-12
    # A candidate's reported error uses a rounded FP32 subtraction, and its
    # output uses one rounded bias addition. Include both rounding operations
    # and an absolute tiny term covering possible flush-to-zero.
    u=2.**-24; tiny=math.sqrt(recorded.size)*float(np.finfo(np.float32).tiny)/math.sqrt(energy)
    reported_gate=.01
    exact_error_bound=(math.sqrt(reported_gate)+tiny)/(1-u)
    ideal_error_bound=exact_error_bound+(u*(1+exact_error_bound)+tiny)/(1-u)
    # Conservative margin for FP64 squared reductions/mean computations.
    # This affects only diagnostic exclusion, never the candidate .01 gate.
    acceptance_bound=ideal_error_bound**2*(1+1e-8)+1e-12
    return {'cells':rows,'recorded_normalized_sse':old_sse/energy,
        'recorded_within_cell_normalized_sse':within/energy,
        'recorded_mean_error_energy_fraction':bias_energy/old_sse,
        'decomposition_relative_discrepancy':discrepancy,
        'ideal_real_bias_oracle_normalized_sse':oracle/energy,
        'oracle_nmse_necessary_upper_bound_for_fp32_bias_candidate_passing_001':acceptance_bound,
        'bias_only_repair_cannot_pass_fixed_products_and_fp32_addition':oracle/energy>acceptance_bound}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args(); assert not args.out.exists()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    start,stage=time.monotonic(),'bindings'
    try:
        assert P.digest(PREVIOUS)==PREVIOUS_SHA
        old=json.loads(PREVIOUS.read_text(encoding='utf-8'))
        assert P.digest(CHECKPOINT)==old['checkpoint_sha256'] and P.digest(R.CAPTURE)==old['capture_sha256']
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=15*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            x=torch.from_numpy(archive['x_bf16'][M.FIT:].reshape(-1,896)).view(torch.bfloat16).to(device).float()
            y=torch.from_numpy(archive['y_bf16'][M.FIT:].reshape(-1,896)).view(torch.bfloat16).to(device).float()
        with safe_open(str(CHECKPOINT),framework='pt',device='cpu') as archive:
            saved={name:archive.get_tensor(name).to(device).float() for name in archive.keys()}
        stage='fixed_route_and_per_window_replay'
        pv=R.assign_full(x,saved['router.parents']); cv=torch.empty_like(pv)
        for parent in range(16):
            mask=pv==parent
            if bool(mask.any()): cv[mask]=parent*10+R.assign_full(x[mask],saved['router.children'][parent])
        rotated=(cv//10)*10+(cv%10+1)%10
        recorded={arm:np.empty((len(x),896),dtype=np.float64) for arm in ('e16','e160')}
        prebias={arm:np.empty_like(value) for arm,value in recorded.items()}
        replay=[]
        for seq in range(M.VALID):
            span=slice(seq*M.SEQ,(seq+1)*M.SEQ)
            energy=float(y[span].double().square().sum())
            assert energy==old['validation_rows'][seq]['energy']
            row={'validation_sequence':seq,'energy':energy,'sse':{}}
            for arm,bank,labels in (('e16','e16',pv),('e160','e160',cv),('e160_rotated','e160',rotated)):
                selected=labels[span]
                product=torch.bmm(saved[bank+'.weight'][selected],x[span,:,None]).squeeze(-1)
                predicted=product+saved[bank+'.bias'][selected]
                error=predicted-y[span]
                sse=float(error.double().square().sum())
                assert sse==old['validation_rows'][seq]['sse'][arm]
                row['sse'][arm]=sse
                if arm in recorded:
                    recorded[arm][span]=error.double().cpu().numpy()
                    prebias[arm][span]=(product.double()-y[span].double()).cpu().numpy()
            replay.append(row); P.budget(start,device)
        energy=sum(r['energy'] for r in replay)
        stage='nondeployable_bias_oracle_decomposition'
        summary={arm:decompose(recorded[arm],prebias[arm],labels.cpu().numpy(),count,energy,
            old['summary'][arm]['sse']) for arm,labels,count in (('e16',pv,16),('e160',cv,160))}
        decision='bias_only_repair_excluded_for_fixed_tangent_products' if summary['e160']['bias_only_repair_cannot_pass_fixed_products_and_fp32_addition'] else 'bias_oracle_leaves_room_freeze_fit_only_bias_method_before_testing'
        result={'experiment':'METH-228-frozen-tangent-bias-oracle-bound','previous_result_sha256':PREVIOUS_SHA,
            'checkpoint_sha256':old['checkpoint_sha256'],'capture_sha256':old['capture_sha256'],
            'every_original_window_energy_and_all3_sse_exact':True,'replay':replay,'summary':summary,
            'decision':decision,'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'torch_version':torch.__version__,
            'scope':'Nondeployable validation-label bias oracle on fixed FP32 matrix products; no candidate fitting/export, no wider coefficient/operator impossibility, no new LLM quality/rate'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'decision':decision,'e16_oracle_nmse':summary['e16']['ideal_real_bias_oracle_normalized_sse'],
            'e160_oracle_nmse':summary['e160']['ideal_real_bias_oracle_normalized_sse'],
            'e160_mean_error_energy_fraction':summary['e160']['recorded_mean_error_energy_fraction'],
            'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'elapsed_seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
