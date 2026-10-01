#!/usr/bin/env python3
"""Four fixed projection/encoding cycles using actual quantized feedback."""
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
from torch.nn import functional as F
import meth235_full_feature_output_priors as Q
import meth236_output_scale_jacobian_bound as B

P,M,R,N=Q.P,Q.M,Q.R,Q.N
STEPS=4
BOUND=P.DOC/'meth236_output_scale_jacobian_bound_result.json'
BOUND_SHA='d6febbb46d8387935a6c0653ad3df171588a8b0a6ae294b1a457a3b72d478a23'


def encoded_hash(codes,scales):
    return P.M17.sha(codes.cpu().numpy().tobytes()+scales.cpu().numpy().tobytes())


def main():
    ap=argparse.ArgumentParser()
    for key in ('checkpoint','out'): ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args(); assert not args.checkpoint.exists() and not args.out.exists()
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True); args.out.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.checkpoint.parent).free>=2*1024**3
    start,stage,progress=time.monotonic(),'bindings',{}
    def partial():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'progress':progress,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
    try:
        assert P.digest(B.PRIOR)==B.PRIOR_SHA and P.digest(Q.NATIVE)==Q.NATIVE_SHA
        prior=json.loads(B.PRIOR.read_text()); bound=json.loads(BOUND.read_text())
        assert P.digest(BOUND)==BOUND_SHA
        assert all(value for key,value in prior['gates'].items() if key!='stored_gradient')
        assert bound['prior_result_sha256']==B.PRIOR_SHA
        assert bound['decision']=='exclude_scale_only_repair_for_these_fixed_codes'
        assert bound['summary']['parents_unreachable_at_1percent']==list(range(16))
        assert P.digest(B.SNAPSHOT)==prior['checkpoint_sha256'] and P.digest(R.CAPTURE)==prior['capture_sha256']
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        native=json.loads(Q.NATIVE.read_text()); values,controls=Q.load_layer(native,device)
        assert controls==prior['controls']
        with safe_open(str(B.SNAPSHOT),framework='pt',device='cpu') as archive:
            original={name:archive.get_tensor(name) for name in archive.keys()}
        tensors=dict(original)
        for name in ('prior.down.q','prior.down.scale','prior.bias'): tensors[name]=torch.empty_like(original[name])
        parents=original['router.parents'].to(device)
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            xf=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,Q.D).copy()).view(torch.bfloat16).to(device).float()
        labels=R.assign_full(xf,parents)
        assert torch.bincount(labels,minlength=16).tolist()==prior['fit_counts']
        assert P.M17.sha(labels.cpu().numpy().astype('<i4').tobytes())==prior['fit_parent_label_sha256']
        matrices=[]; source_hashes={}
        with safe_open(str(source),framework='pt',device='cpu') as archive:
            for organ in ('gate','up','down'):
                name=f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight'; w=archive.get_tensor(name)
                source_hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes()); matrices.append(w.to(device).float())
        assert source_hashes==prior['source_tensor_sha256']
        base=(values['down.q'].float()*values['down.scale'][:,None]).double()
        audits=[]; sse=old_sse=energy=0.; hashes=[]
        for parent,center in enumerate(parents):
            stage='quantized_feedback_projection'
            source_jac,source_value=N.donor_tangent(center,*matrices); jd=source_jac.double()
            a=Q.feature_jacobian(center,values).double(); qr,triangular=torch.linalg.qr(a,mode='reduced')
            codes=original['prior.down.q'][parent].to(device); scales=original['prior.down.scale'][parent].to(device)
            current=(codes.float()*scales[:,None]).double(); history=[]
            initial_hash=encoded_hash(codes,scales)
            for step in range(1,STEPS+1):
                residual=jd-current@a
                delta=torch.linalg.solve_triangular(triangular.T,residual.T,upper=False).T@qr.T
                raw=current+delta; raw_error=float((raw@a-jd).norm()/jd.norm())
                assert raw_error<=1e-5 and torch.isfinite(raw).all()
                raw32=raw.float(); codes,scales=Q.L.encode_rows(raw32)
                current=(codes.float()*scales[:,None]).double()
                error=float((current@a-jd).norm()/jd.norm())
                history.append({'step':step,'unrounded_gradient_error':raw_error,
                    'stored_gradient_error':error,'function_sha256':encoded_hash(codes,scales)})
                P.budget(start,device)
            phi=Q.features(center,values); bias=source_value-Q.L.row_linear(phi,codes,scales)
            assert torch.isfinite(bias).all()
            point=float((Q.L.row_linear(phi,codes,scales)+bias-source_value).double().norm()/source_value.double().norm())
            change=float((current-base).norm()/base.norm())
            raw_bias=source_value-F.linear(phi,raw32); complete={}
            if parent in (0,15):
                with torch.enable_grad():
                    jac=torch.autograd.functional.jacobian(lambda x:F.linear(Q.features(x,values),raw32,raw_bias),center,vectorize=True)
                complete={'relative_frobenius':float((jac.double()-jd).norm()/jd.norm()),
                    'max_relative_to_peak':float((jac.double()-jd).abs().max()/jd.abs().max())}
            tensors['prior.down.q'][parent].copy_(codes.cpu()); tensors['prior.down.scale'][parent].copy_(scales.cpu())
            tensors['prior.bias'][parent].copy_(bias.cpu()); digest=encoded_hash(codes,scales); hashes.append(digest)
            stage='fit_input_source_function_control'; local=previous=local_energy=0.; xp=xf[labels==parent]
            old_codes=original['prior.down.q'][parent].to(device)
            old_scales=original['prior.down.scale'][parent].to(device); old_bias=original['prior.bias'][parent].to(device)
            for first in range(0,len(xp),512):
                x=xp[first:first+512]; phi=Q.features(x,values); target=N.donor_value(x,*matrices)
                output=Q.L.row_linear(phi,codes,scales)+bias
                old=Q.L.row_linear(phi,old_codes,old_scales)+old_bias
                local+=float((output.double()-target.double()).square().sum())
                previous+=float((old.double()-target.double()).square().sum())
                local_energy+=float(target.double().square().sum())
            assert abs(previous-prior['audits'][parent]['fit_source_sse'])<=max(1e-12,previous*1e-12)
            audit={'parent':parent,'initial_function_sha256':initial_hash,'history':history,
                'final_function_sha256':digest,'repeated_encoding_count':STEPS+1-len({initial_hash,*[r['function_sha256'] for r in history]}),
                'stored_center_relative_l2':point,'relative_stored_coefficient_change':change,
                'complete_unrounded_autograd':complete,'fit_source_sse':local,'old_fit_source_sse':previous,
                'source_energy':local_energy,'fit_source_normalized_sse':local/local_energy}
            audits.append(audit); sse+=local; old_sse+=previous; energy+=local_energy
            progress={'completed_parent':parent,'audits':audits}; partial(); P.budget(start,device)
        unchanged=[name for name in tensors if name not in ('prior.down.q','prior.down.scale','prior.bias')]
        assert all(torch.equal(tensors[name],original[name]) for name in unchanged)
        stage='physical_snapshot_readback'
        save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-237','source_sha256':P.M57.MODEL_SHA,
            'steps':str(STEPS),'native_result_sha256':Q.NATIVE_SHA,'format':'row-Q8-FP32-SiLU-LUT'})
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
            assert set(archive.keys())==set(tensors)
            for name,t in tensors.items(): assert torch.equal(t,archive.get_tensor(name))
        gates={'all_controls_source_fit_replay_exact':True,'every_snapshot_tensor_readback_equal':True,
            'source_center_values':max(a['stored_center_relative_l2'] for a in audits)<=1e-5,
            'coefficient_growth':max(a['relative_stored_coefficient_change'] for a in audits)<=.25,
            'unrounded_gradient':max(h['unrounded_gradient_error'] for a in audits for h in a['history'])<=1e-5,
            'complete_unrounded_autograd':all(a['complete_unrounded_autograd']['relative_frobenius']<=1e-5 and a['complete_unrounded_autograd']['max_relative_to_peak']<=1e-4 for a in audits if a['complete_unrounded_autograd']),
            'stored_gradient':max(a['history'][-1]['stored_gradient_error'] for a in audits)<=.01,
            'fit_source_function':sse/energy<=.01,'all16_coefficient_pairs_distinct':len(set(hashes))==16}
        result={'experiment':'METH-237-fixed-four-cycle-quantized-feedback-output-projection','steps':STEPS,
            'prior_result_sha256':B.PRIOR_SHA,'scale_bound_result_sha256':P.digest(BOUND),'native_result_sha256':Q.NATIVE_SHA,
            'source_sha256':P.M57.MODEL_SHA,'source_tensor_sha256':source_hashes,'controls':controls,
            'unchanged_control_tensors':unchanged,'capture_sha256':prior['capture_sha256'],'audits':audits,'gates':gates,
            'summary':{'fit_source_normalized_sse':sse/energy,'old_fit_source_normalized_sse':old_sse/energy,
                'maximum_final_stored_gradient_error':max(a['history'][-1]['stored_gradient_error'] for a in audits),
                'maximum_coefficient_change':max(a['relative_stored_coefficient_change'] for a in audits)},
            'checkpoint_bytes':args.checkpoint.stat().st_size,'checkpoint_sha256':P.digest(args.checkpoint),
            'script_sha256':P.digest(Path(__file__)),'runtime':P.budget(start,device),
            'decision':'quantized_output_projection_pass_freeze_full_feature_pair' if all(gates.values()) else 'stop_this_four_cycle_quantized_output_projection',
            'scope':'Fixed quantization-aware source prior only; no output-label fitting, validation/new quality, full model, routed bank native cost, n-scale or accepted full rate'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'progress':progress,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
