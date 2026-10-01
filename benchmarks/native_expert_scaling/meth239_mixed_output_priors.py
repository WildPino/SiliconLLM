#!/usr/bin/env python3
"""Qualify the fixed32 BF16 escape codec on output-only source priors."""
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
import meth238_outlier_split_feasibility as L
import meth230_source_anchored_nonlinear_pair as S

P,M,R,N=S.P,S.M,S.R,L.N
D,H=896,4864
NATIVE=P.DOC/'meth238_outlier_split_native_result.json'
NATIVE_SHA='61b496f66455e764039877ddaf17cedaa3aab49c287f09178e085afd56dbe684'
FIXTURE=P.ROOT/'results/native_expert_scaling/meth238_outlier_split_fixture.bin'


def load_layer(native,device):
    assert P.digest(FIXTURE)==native['binary']['sha256']
    values={}; hashes={}
    with FIXTURE.open('rb') as file:
        for segment in native['segments']:
            if segment['layer'] not in (None,M.LAYER): continue
            file.seek(segment['offset']); raw=file.read(segment['bytes'])
            assert P.M17.sha(raw)==segment['sha256']; name=segment['name']; hashes[name]=segment['sha256']
            dtype=('<u2' if name.endswith(('.ids','.escape')) else np.int8 if name.endswith('.q') else '<f4')
            tensor=torch.from_numpy(np.frombuffer(raw,dtype=dtype).copy())
            if name.endswith('.q'): tensor=tensor.reshape((D,H) if name.startswith('down') else (H,D))
            if name.endswith(('.ids','.escape')): tensor=tensor.reshape(D,32)
            if name.endswith('.escape'): tensor=tensor.view(torch.bfloat16)
            values[name]=tensor.to(device)
    L.TABLE=values.pop('silu_table')
    return values,hashes


def features(x,values):
    g=L.row_linear(x,values['gate.q'],values['gate.scale'])
    u=L.row_linear(x,values['up.q'],values['up.scale'])
    return L.silu_lookup(g)*u


def feature_jacobian(center,values):
    g=L.row_linear(center,values['gate.q'],values['gate.scale'])
    u=L.row_linear(center,values['up.q'],values['up.scale'])
    position=((g+16)*16).clamp(0,512)
    index=position.long().clamp(max=511)
    slope=16*(L.TABLE[index+1]-L.TABLE[index])
    slope=torch.where(g<=-16,torch.zeros_like(g),
        torch.where((g>=16)|(position>=512),torch.ones_like(g),slope))
    gate=values['gate.q'].float()*values['gate.scale'][:,None]
    up=values['up.q'].float()*values['up.scale'][:,None]
    return (u*slope)[:,None]*gate+L.silu_lookup(g)[:,None]*up


def derivative_checks(center,values,jac):
    # Fixed rows plus five directions check both reverse and forward derivatives.
    ids=torch.tensor([0,1,127,255,511,767,1023,1535,2047,2559,3071,3583,4095,4351,4607,4863],device=center.device)
    with torch.enable_grad():
        reference=torch.autograd.functional.jacobian(lambda x:features(x,values)[ids],center,vectorize=True)
    rel=float((reference.double()-jac[ids].double()).norm()/reference.double().norm())
    checks=[]
    for dimension in (0,127,511,895,None):
        direction=torch.zeros_like(center)
        if dimension is None: direction=center/center.norm()
        else: direction[dimension]=1
        with torch.enable_grad():
            _,expected=torch.autograd.functional.jvp(lambda x:features(x,values),center,direction)
        actual=F.linear(direction,jac)
        error=float((actual.double()-expected.double()).norm()/expected.double().norm())
        checks.append({'dimension':dimension,'relative_l2':error})
    return {'fixed16_row_relative_frobenius':rel,'directions':checks}


def projected_prior(center,matrices,values,base,check_autograd=False):
    source_jac,source_value=N.donor_tangent(center,*matrices)
    a=feature_jacobian(center,values)
    checks=derivative_checks(center,values,a) if check_autograd else {}
    ad=a.double(); bd=base.double(); jd=source_jac.double()
    q,t=torch.linalg.qr(ad,mode='reduced')
    singular=torch.linalg.svdvals(t)
    condition=float(singular[0]/singular[-1]); assert torch.isfinite(singular).all() and condition<=1e8
    residual=jd-bd@ad
    correction=torch.linalg.solve_triangular(t.T,residual.T,upper=False).T@q.T
    raw=bd+correction
    reconstructed=raw@ad
    gradient_error=float((reconstructed-jd).norm()/jd.norm())
    coefficient_change=float(correction.norm()/bd.norm())
    raw32=raw.float(); phi=features(center,values)
    raw_bias=source_value-F.linear(phi,raw32)
    raw_point=float((F.linear(phi,raw32,raw_bias)-source_value).double().norm()/source_value.double().norm())
    codes,scales,ids,escape=L.encode_mixed(raw32)
    bias=source_value-L.mixed_linear(phi,codes,scales,ids,escape)
    stored_point=float((L.mixed_linear(phi,codes,scales,ids,escape)+bias-source_value).double().norm()/source_value.double().norm())
    stored=L.decode_mixed(codes,scales,ids,escape)
    stored_gradient_error=float((stored.double()@ad-jd).norm()/jd.norm())
    complete={}
    if check_autograd:
        with torch.enable_grad():
            actual=torch.autograd.functional.jacobian(lambda x:F.linear(features(x,values),raw32,raw_bias),center,vectorize=True)
        complete={'relative_frobenius':float((actual.double()-jd).norm()/jd.norm()),
            'max_relative_to_peak':float((actual.double()-jd).abs().max()/jd.abs().max())}
    assert all(torch.isfinite(x).all() for x in (raw,raw_bias,codes,scales,bias))
    return codes,scales,ids,escape,bias,{'condition_number':condition,'smallest_singular_value':float(singular[-1]),
        'largest_singular_value':float(singular[0]),'relative_coefficient_change':coefficient_change,
        'unrounded_gradient_relative_frobenius':gradient_error,'raw32_center_relative_l2':raw_point,
        'stored_center_relative_l2':stored_point,'stored_gradient_relative_frobenius':stored_gradient_error,
        'feature_derivatives':checks,'complete_unrounded_autograd':complete}


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
        assert P.digest(NATIVE)==NATIVE_SHA and P.digest(S.ROUTE_RESULT)==S.ROUTE_SHA
        native=json.loads(NATIVE.read_text()); route=json.loads(S.ROUTE_RESULT.read_text())
        assert all(native['gates'].values()) and P.digest(S.ROUTE_FILE)==route['checkpoint_sha256']
        assert P.digest(R.CAPTURE)==route['capture_sha256']
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        device=M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        values,controls=load_layer(native,device)
        with safe_open(str(S.ROUTE_FILE),framework='pt',device='cpu') as archive:
            parents=archive.get_tensor('router.parents').to(device); children=archive.get_tensor('router.children')
        with np.load(R.CAPTURE,allow_pickle=False) as archive:
            xf=torch.from_numpy(archive['x_bf16'][:M.FIT].reshape(-1,D).copy()).view(torch.bfloat16).to(device).float()
        labels=R.assign_full(xf,parents); counts=torch.bincount(labels,minlength=16).tolist()
        assert counts==route['fit']['counts16']
        assert P.M17.sha(labels.cpu().numpy().astype('<i4').tobytes())==route['fit']['parent_labels_sha256']
        matrices=[]; source_hashes={}
        with safe_open(str(source),framework='pt',device='cpu') as archive:
            for organ in ('gate','up','down'):
                name=f'model.layers.{M.LAYER}.mlp.{organ}_proj.weight'; w=archive.get_tensor(name)
                source_hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes()); matrices.append(w.to(device).float())
        assert source_hashes==route['fit']['source_tensor_sha256']
        base=L.decode_mixed(values['down.q'],values['down.scale'],values['down.ids'],values['down.escape'])
        tensors={'input.'+name:value.cpu().contiguous() for name,value in values.items() if name.startswith(('gate.','up.'))}
        tensors.update({'lookup.table':L.TABLE.cpu(),'source.down.q':values['down.q'].cpu(),
            'source.down.scale':values['down.scale'].cpu(),'source.down.ids':values['down.ids'].cpu(),
            'source.down.escape':values['down.escape'].cpu(),'router.parents':parents.cpu(),'router.children':children,
            'prior.down.q':torch.empty((16,D,H),dtype=torch.int8),'prior.down.scale':torch.empty((16,D)),
            'prior.bias':torch.empty((16,D)),'prior.down.ids':torch.empty((16,D,32),dtype=torch.uint16),
            'prior.down.escape':torch.empty((16,D,32),dtype=torch.bfloat16)})
        audits=[]; sse=baseline_sse=energy=0.; hashes=[]
        for parent,center in enumerate(parents):
            stage='source_output_derivative_projection'
            codes,scales,ids,escape,bias,audit=projected_prior(center,matrices,values,base,parent in (0,15))
            tensors['prior.down.q'][parent].copy_(codes.cpu()); tensors['prior.down.scale'][parent].copy_(scales.cpu())
            tensors['prior.bias'][parent].copy_(bias.cpu())
            tensors['prior.down.ids'][parent].copy_(ids.cpu()); tensors['prior.down.escape'][parent].copy_(escape.cpu())
            digest=P.M17.sha(codes.cpu().numpy().tobytes()+scales.cpu().numpy().tobytes()+ids.cpu().numpy().tobytes()+escape.cpu().view(torch.uint16).numpy().tobytes()); hashes.append(digest)
            local_sse=local_base=local_energy=0.
            stage='fit_input_source_function_screen'; xp=xf[labels==parent]
            for first in range(0,len(xp),512):
                x=xp[first:first+512]; phi=features(x,values); target=N.donor_value(x,*matrices)
                output=L.mixed_linear(phi,codes,scales,ids,escape)+bias
                baseline=L.mixed_linear(phi,values['down.q'],values['down.scale'],values['down.ids'],values['down.escape'])
                local_sse+=float((output.double()-target.double()).square().sum())
                local_base+=float((baseline.double()-target.double()).square().sum())
                local_energy+=float(target.double().square().sum())
            audit.update({'parent':parent,'fit_states':len(xp),'function_sha256':digest,
                'fit_source_sse':local_sse,'baseline_fit_source_sse':local_base,'source_energy':local_energy,
                'fit_source_normalized_sse':local_sse/local_energy,'baseline_fit_source_normalized_sse':local_base/local_energy})
            audits.append(audit); sse+=local_sse; baseline_sse+=local_base; energy+=local_energy
            progress={'completed_parent':parent,'audits':audits}; partial(); P.budget(start,device)
        stage='physical_snapshot_readback'
        save_file(tensors,str(args.checkpoint),metadata={'experiment':'METH-239','source_sha256':P.M57.MODEL_SHA,
            'native_result_sha256':NATIVE_SHA,'layer':str(M.LAYER),'hidden':str(H),'format':'row-Q8-BF16-32-escape-FP32-SiLU-LUT'})
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as archive:
            assert set(archive.keys())==set(tensors)
            for name,t in tensors.items(): assert torch.equal(archive.get_tensor(name),t)
        derivative=[a['feature_derivatives'] for a in audits if a['feature_derivatives']]
        complete=[a['complete_unrounded_autograd'] for a in audits if a['complete_unrounded_autograd']]
        gates={'source_route_native_bindings':True,'all_snapshot_tensors_readback_exact':True,
            'full_rank_condition':all(a['condition_number']<=1e8 for a in audits),
            'coefficient_growth':max(a['relative_coefficient_change'] for a in audits)<=.25,
            'feature_derivatives':all(a['fixed16_row_relative_frobenius']<=1e-5 and all(d['relative_l2']<=1e-5 for d in a['directions']) for a in derivative),
            'unrounded_gradient':max(a['unrounded_gradient_relative_frobenius'] for a in audits)<=1e-5,
            'complete_unrounded_autograd':all(a['relative_frobenius']<=1e-5 and a['max_relative_to_peak']<=1e-4 for a in complete),
            'source_center_values':max(max(a['raw32_center_relative_l2'],a['stored_center_relative_l2']) for a in audits)<=1e-5,
            'stored_gradient':max(a['stored_gradient_relative_frobenius'] for a in audits)<=.01,
            'fit_source_function':sse/energy<=.01,'all16_coefficient_pairs_distinct':len(set(hashes))==16}
        result={'experiment':'METH-239-fixed32-BF16-output-escape-source-prior-qualification','escape_count':32,'layer':M.LAYER,'hidden':H,
            'source_revision':P.M42.REV,'source_sha256':P.M57.MODEL_SHA,'source_tensor_sha256':source_hashes,
            'native_result_sha256':NATIVE_SHA,'fixture_sha256':native['binary']['sha256'],'controls':controls,
            'route_result_sha256':S.ROUTE_SHA,'route_snapshot_sha256':route['checkpoint_sha256'],
            'capture_sha256':route['capture_sha256'],'fit_parent_label_sha256':route['fit']['parent_labels_sha256'],
            'fit_counts':counts,'audits':audits,'gates':gates,'function_hashes':hashes,
            'summary':{'fit_source_normalized_sse':sse/energy,'baseline_fit_source_normalized_sse':baseline_sse/energy,
                'maximum_condition_number':max(a['condition_number'] for a in audits),
                'maximum_coefficient_change':max(a['relative_coefficient_change'] for a in audits),
                'maximum_stored_gradient_error':max(a['stored_gradient_relative_frobenius'] for a in audits)},
            'checkpoint_sha256':P.digest(args.checkpoint),'checkpoint_bytes':args.checkpoint.stat().st_size,
            'script_sha256':P.digest(Path(__file__)),'runtime':P.budget(start,device),'torch_version':torch.__version__,
            'decision':'mixed_output_priors_pass_freeze_full_feature_pair' if all(gates.values()) else 'stop_this_fixed32_mixed_output_prior_projection',
            'scope':'Source derivative prior qualification, fit-input controls only; no output-label fitting, validation, full model/independent quality, routed bank cost, n-scale or complete accepted rate'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'progress':progress,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
