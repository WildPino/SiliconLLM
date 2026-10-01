#!/usr/bin/env python3
"""Qualify all original source features with row-Q8/FP32 native arithmetic."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import time
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch
from torch.nn import functional as F
import meth226_donor_tangent_feasibility as N

P,C=N.P,N.C


def encode_rows(w):
    w=w.float()
    maximum=w.abs().amax(dim=1)
    scale=torch.where(maximum>0,maximum/127,torch.ones_like(maximum))
    q=torch.round(w/scale[:,None]).clamp(-127,127).to(torch.int8)
    assert torch.isfinite(scale).all() and (scale>0).all() and (q!=-128).all()
    return q.contiguous(),scale.contiguous()


def row_linear(x,q,scale):
    return F.linear(x,q.float())*scale


def row_value(x,values):
    gate=row_linear(x,*values['gate'])
    up=row_linear(x,*values['up'])
    return row_linear(F.silu(gate)*up,*values['down'])+values['bias']


def main():
    ap=argparse.ArgumentParser()
    for key in ('binary','exe','out'): ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args()
    assert not args.binary.exists() and not args.out.exists()
    assert shutil.disk_usage(args.binary.parent).free>=2*1024**3
    args.binary.parent.mkdir(parents=True,exist_ok=True); args.out.parent.mkdir(parents=True,exist_ok=True)
    start,stage=time.monotonic(),'bindings'; segments=[]; fidelity=[]; source_hashes=[]
    try:
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA and P.digest(C.VECTORS)==C.VECTORS_SHA
        device=N.N.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        # Meaningful format edge checks: zero rows, ties, and negative symmetric codes.
        q,s=encode_rows(torch.tensor([[0.,0.,0.,0.],[127.,.5,1.5,-1.5]],device=device))
        assert torch.equal(q.cpu(),torch.tensor([[0,0,0,0],[127,0,2,-2]],dtype=torch.int8))
        assert torch.equal(s.cpu(),torch.ones(2))
        raw=C.VECTORS.read_bytes()
        assert struct.unpack_from('<8s4I',raw)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(raw,dtype='<u2',offset=24).reshape(256,24,896)
        stored=[]
        with safe_open(str(source),framework='pt',device='cpu') as archive,args.binary.open('xb') as output:
            output.write(struct.pack('<8s4I',b'M232RQ01',24,896,4864,32))
            for li in range(24):
                values={'bias':torch.zeros(896,dtype=torch.float32)}; matrices=[]; hashes={}
                for organ in ('gate','up','down'):
                    name=f'model.layers.{li}.mlp.{organ}_proj.weight'
                    w=archive.get_tensor(name)
                    assert w.dtype==torch.bfloat16 and w.shape==((896,4864) if organ=='down' else (4864,896))
                    hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes())
                    matrices.append(w.to(device).float())
                    values[organ]=tuple(t.cpu() for t in encode_rows(matrices[-1]))
                source_hashes.append({'layer':li,'sha256':hashes})
                stage='source_function_screen'
                x=torch.from_numpy(states[:,li].copy()).view(torch.bfloat16).to(device).float()
                v={key:(tuple(t.to(device) for t in value) if key!='bias' else value.to(device)) for key,value in values.items()}
                reference=N.donor_value(x,*matrices)
                expected=row_value(x,v)
                assert torch.isfinite(reference).all() and torch.isfinite(expected).all()
                sse=float((reference.double()-expected.double()).square().sum())
                energy=float(reference.double().square().sum()); assert energy>0
                fidelity.append({'layer':li,'states':256,'sse':sse,'energy':energy,'normalized_sse':sse/energy})
                stage='source_fixture_serialization'
                for organ in ('gate','up','down'):
                    for field,t in zip(('q','scale'),values[organ]):
                        raw=t.numpy().tobytes()
                        segments.append({'layer':li,'name':organ+'.'+field,'offset':output.tell(),'bytes':len(raw),'sha256':P.M17.sha(raw)})
                        output.write(raw)
                raw=values['bias'].numpy().tobytes()
                segments.append({'layer':li,'name':'bias','offset':output.tell(),'bytes':len(raw),'sha256':P.M17.sha(raw)})
                output.write(raw); stored.append(values)
                del matrices,x,v,reference,expected,w
                P.budget(start,device)
        assert args.binary.stat().st_size==314892312 and len(segments)==168
        distinct={organ:len({s['sha256'] for s in segments if s['name']==organ+'.q'}) for organ in ('gate','up','down')}
        assert all(v==24 for v in distinct.values())
        with args.binary.open('rb') as audit:
            assert audit.read(24)==struct.pack('<8s4I',b'M232RQ01',24,896,4864,32)
            for row in segments:
                assert audit.tell()==row['offset'] and P.M17.sha(audit.read(row['bytes']))==row['sha256']
            assert audit.read(1)==b''
        stage='actual_row_q8_native_component'
        torch.cuda.synchronize(device)
        check=args.binary.with_suffix('.check.bin'); assert not check.exists()
        run=subprocess.run([str(args.exe),str(args.binary),str(C.VECTORS),str(check)],
            capture_output=True,text=True,check=True,timeout=600)
        timing=json.loads(run.stdout)
        assert timing['threads']==6 and timing['tokens']==256 and timing['layers']==24 and timing['hidden']==4864
        assert timing['weight_bytes']==314892312 and len(timing['pass_ms_per_token'])==3
        raw=check.read_bytes()
        assert len(raw)==20+16*24*896*4 and struct.unpack_from('<8s3I',raw)==(b'M232OUT1',16,24,896)
        actual=np.frombuffer(raw,dtype='<f4',offset=20).reshape(16,24,896)
        assert np.isfinite(actual).all()
        stage='stored_fp32_row_scaled_oracle'; rows=[]
        for li,values in enumerate(stored):
            v={key:(tuple(t.to(device) for t in value) if key!='bias' else value.to(device)) for key,value in values.items()}
            x=torch.from_numpy(states[:16,li].copy()).view(torch.bfloat16).to(device).float()
            expected=row_value(x,v).cpu().numpy().astype(np.float64)
            errors=np.linalg.norm(actual[:,li].astype(np.float64)-expected,axis=1)/np.linalg.norm(expected,axis=1)
            assert np.isfinite(errors).all()
            rows.extend({'token':t,'layer':li,'relative_l2':float(e)} for t,e in enumerate(errors))
            del v,x,expected
            P.budget(start,device)
        errors=[r['relative_l2'] for r in rows]
        pooled_sse=sum(r['sse'] for r in fidelity)/sum(r['energy'] for r in fidelity)
        gates={'codec_edge_cases':True,'all168_segments_readback_exact':True,'all24_layers_distinct_per_organ':True,
            'native_numeric':statistics.median(errors)<=1e-4 and max(errors)<=5e-4,
            'source_function_error':pooled_sse<=.01,
            'native_component_cost':statistics.median(timing['pass_ms_per_token'])<=10}
        result={'experiment':'METH-232-full-source-row-Q8-native-qualification','hidden':4864,
            'source_sha256':P.M57.MODEL_SHA,'source_revision':P.M42.REV,'vectors_sha256':C.VECTORS_SHA,
            'source_tensor_hashes':source_hashes,'segments':segments,'distinct_code_matrices':distinct,
            'binary':{'bytes':args.binary.stat().st_size,'sha256':P.digest(args.binary)},
            'output_sha256':P.digest(check),'executable_sha256':P.digest(args.exe),
            'source_code_sha256':P.digest(args.exe.with_suffix('.c')),'script_sha256':P.digest(Path(__file__)),
            'included_meth182_source_sha256':P.digest(args.exe.parent/'meth182_group64_ffn_cpu.c'),
            'timing':timing,'rows':rows,'source_fidelity':fidelity,'gates':gates,
            'summary':{'median_relative_l2':statistics.median(errors),'maximum_relative_l2':max(errors),
                'pooled_source_function_normalized_sse':pooled_sse,'maximum_layer_source_normalized_sse':max(r['normalized_sse'] for r in fidelity),
                'median_ms_per_token':statistics.median(timing['pass_ms_per_token'])},
            'runtime':P.budget(start,device),'torch_version':torch.__version__,
            'decision':'row_q8_pass_freeze_full_feature_conditional_method' if all(gates.values()) else 'stop_this_full_source_row_q8_operator',
            'scope':'24 distinct original source fixtures, no activation quantization; not routed learned capacity, actual BF16 whole-model quality, new held-out quality, n-scaled DRAM/LUT or accepted full rate'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'decision':result['decision'],'summary':result['summary'],'gates':gates,'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'elapsed_seconds':time.monotonic()-start,'segments':segments,'source_fidelity':fidelity},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
