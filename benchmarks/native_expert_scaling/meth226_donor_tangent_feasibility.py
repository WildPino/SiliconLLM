#!/usr/bin/env python3
"""Autograd-qualified original donor tangents and actual stored C arithmetic."""
import argparse
import json
from pathlib import Path
import statistics
import struct
import subprocess
import time
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch
from torch.nn import functional as F
import meth224_common_native_feasibility as N

P,C=N.P,N.C


def donor_value(x,gate,up,down):
    return F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)


def donor_tangent(center,gate,up,down):
    g,u=F.linear(center,gate),F.linear(center,up)
    s=torch.sigmoid(g)
    dg=u*(s+g*s*(1-s))
    du=F.silu(g)
    jacobian=(down*dg[None,:])@gate+(down*du[None,:])@up
    value=F.linear(du*u,down)
    return jacobian,value


def main():
    ap=argparse.ArgumentParser()
    for key in ('binary','exe','out'): ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args()
    assert not args.binary.exists() and not args.out.exists()
    args.binary.parent.mkdir(parents=True,exist_ok=True); args.out.parent.mkdir(parents=True,exist_ok=True)
    start,stage=time.monotonic(),'bindings'
    derivatives,segments=[],[]
    try:
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA and P.digest(C.VECTORS)==C.VECTORS_SHA
        device=N.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        raw_states=C.VECTORS.read_bytes()
        assert struct.unpack_from('<8s4I',raw_states)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(raw_states,dtype='<u2',offset=24).reshape(256,24,896)
        stored=[]
        with safe_open(str(source),framework='pt',device='cpu') as archive,args.binary.open('xb') as output:
            output.write(struct.pack('<8s4I',b'M226BF01',24,896,896,32))
            for li in range(24):
                matrices=[]; source_hashes={}
                for organ in ('gate','up','down'):
                    name=f'model.layers.{li}.mlp.{organ}_proj.weight'
                    weight=archive.get_tensor(name)
                    assert weight.dtype==torch.bfloat16 and weight.shape==((896,4864) if organ=='down' else (4864,896))
                    source_hashes[name]=P.M17.sha(weight.view(torch.uint16).numpy().tobytes())
                    matrices.append(weight.to(device).float())
                center=torch.from_numpy(states[0,li].copy()).view(torch.bfloat16).to(device).float()
                stage='analytic_derivative_against_autograd'
                jacobian,value=donor_tangent(center,*matrices)
                with torch.enable_grad():
                    reference=torch.autograd.functional.jacobian(lambda x:donor_value(x,*matrices),center,vectorize=True)
                difference=jacobian.double()-reference.double()
                relative=float(difference.norm()/reference.double().norm())
                maximum_relative_to_peak=float(difference.abs().max()/reference.double().abs().max())
                derivative={'layer':li,'center_token':0,'center_bf16_sha256':P.M17.sha(states[0,li].tobytes()),
                    'source_tensor_sha256':source_hashes,'relative_frobenius':relative,'max_abs_relative_to_peak':maximum_relative_to_peak}
                derivatives.append(derivative)
                assert torch.isfinite(jacobian).all() and torch.isfinite(value).all()
                assert relative<=1e-5 and maximum_relative_to_peak<=1e-4
                stage='serialize_verified_tangent'
                weight=jacobian.bfloat16()
                bias=value-F.linear(center,weight.float())
                center_error=float((F.linear(center,weight.float(),bias)-value).double().norm()/value.double().norm())
                assert center_error<=1e-5
                derivative['stored_center_relative_l2']=center_error
                wc,bc=weight.cpu().contiguous(),bias.cpu().contiguous()
                stored.append((wc,bc))
                for name,raw in (('jacobian',wc.view(torch.uint16).numpy().tobytes()),('bias',bc.numpy().tobytes())):
                    segments.append({'layer':li,'name':name,'offset':output.tell(),'bytes':len(raw),'sha256':P.M17.sha(raw)})
                    output.write(raw)
                del matrices,jacobian,reference,difference,weight,bias,center,value,wc,bc
                P.budget(start,device)
        assert args.binary.stat().st_size==38621208 and len(segments)==48
        assert len({s['sha256'] for s in segments if s['name']=='jacobian'})==24
        with args.binary.open('rb') as audit:
            assert audit.read(24)==struct.pack('<8s4I',b'M226BF01',24,896,896,32)
            for row in segments:
                assert audit.tell()==row['offset'] and P.M17.sha(audit.read(row['bytes']))==row['sha256']
            assert audit.read(1)==b''
        stage='native_affine_component'
        torch.cuda.synchronize(device)
        check=args.binary.with_suffix('.check.bin'); assert not check.exists()
        run=subprocess.run([str(args.exe),str(args.binary),str(C.VECTORS),str(check)],
            capture_output=True,text=True,check=True,timeout=600)
        timing=json.loads(run.stdout)
        assert timing['threads']==6 and timing['tokens']==256 and timing['layers']==24 and timing['input_width']==896
        raw=check.read_bytes()
        assert len(raw)==20+16*24*896*4 and struct.unpack_from('<8s3I',raw)==(b'M226OUT1',16,24,896)
        actual=np.frombuffer(raw,dtype='<f4',offset=20).reshape(16,24,896)
        assert np.isfinite(actual).all()
        stage='stored_affine_fp32_oracle'; rows=[]
        for li,(weight,bias) in enumerate(stored):
            x=torch.from_numpy(states[:16,li].copy()).view(torch.bfloat16).to(device).float()
            expected=F.linear(x,weight.to(device).float(),bias.to(device)).cpu().numpy().astype(np.float64)
            errors=np.linalg.norm(actual[:,li].astype(np.float64)-expected,axis=1)/np.linalg.norm(expected,axis=1)
            assert np.isfinite(errors).all()
            rows.extend({'token':t,'layer':li,'relative_l2':float(error)} for t,error in enumerate(errors))
            P.budget(start,device)
        errors=[r['relative_l2'] for r in rows]
        gates={'source_derivatives_against_autograd':True,'stored_center_reconciliation':True,
            'all_segments_readback_exact':True,'all24_weight_matrices_distinct':True,
            'native_numeric':statistics.median(errors)<=1e-4 and max(errors)<=5e-4,
            'native_component_cost':statistics.median(timing['pass_ms_per_token'])<=10}
        result={'experiment':'METH-226-full-input-donor-tangent-native-feasibility','source_sha256':P.M57.MODEL_SHA,
            'source_revision':P.M42.REV,'vectors_sha256':C.VECTORS_SHA,'derivative_checks':derivatives,'segments':segments,
            'binary':{'bytes':args.binary.stat().st_size,'sha256':P.digest(args.binary)},'output_sha256':P.digest(check),
            'executable_sha256':P.digest(args.exe),'source_code_sha256':P.digest(args.exe.with_suffix('.c')),
            'included_meth182_source_sha256':P.digest(args.exe.parent/'meth182_group64_ffn_cpu.c'),
            'script_sha256':P.digest(Path(__file__)),'torch_version':torch.__version__,'timing':timing,'rows':rows,
            'summary':{'median_relative_l2':statistics.median(errors),'maximum_relative_l2':max(errors),
                'median_ms_per_token':statistics.median(timing['pass_ms_per_token'])},'gates':gates,'runtime':P.budget(start,device),
            'decision':'apparatus_component_pass_freeze_conditional_tangent_pair' if all(gates.values()) else 'stop_this_tangent_native_operator',
            'scope':'24 source tangents at existing states: operator/derivative fixture, not trained bank, nonlinear accuracy, n scaling, full LLM quality/rate or multiple families'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'decision':result['decision'],'summary':result['summary'],'gates':gates,'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'elapsed_seconds':time.monotonic()-start,'derivative_checks':derivatives,'segments':segments},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
