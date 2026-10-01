#!/usr/bin/env python3
"""Qualify the full stored affine plus source nonlinear operator before fitting."""
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
import meth226_donor_tangent_feasibility as N

P,C=N.P,N.C
HIDDEN=2048


def source_prior(center,gate,up,down,selected):
    full_jac,value=N.donor_tangent(center,gate,up,down)
    gs,us,ds=gate[selected].contiguous(),up[selected].contiguous(),down[:,selected].contiguous()
    selected_jac,selected_value=N.donor_tangent(center,gs,us,ds)
    raw_affine=full_jac-selected_jac
    affine=raw_affine.bfloat16()
    bias=value-(F.linear(center,affine.float())+selected_value)
    return affine,bias,gs,us,ds,raw_affine,full_jac,value


def combined_value(x,affine,gate,up,down,bias):
    return (F.linear(x,affine)+N.donor_value(x,gate,up,down))+bias


def main():
    ap=argparse.ArgumentParser()
    for key in ('binary','exe','out'): ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args(); assert not args.binary.exists() and not args.out.exists()
    args.binary.parent.mkdir(parents=True,exist_ok=True); args.out.parent.mkdir(parents=True,exist_ok=True)
    start,stage=time.monotonic(),'bindings'; identities,segments=[],[]
    try:
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA and P.digest(C.VECTORS)==C.VECTORS_SHA
        device=N.N.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        raw=C.VECTORS.read_bytes()
        assert struct.unpack_from('<8s4I',raw)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(raw,dtype='<u2',offset=24).reshape(256,24,896)
        stored=[]
        with safe_open(str(source),framework='pt',device='cpu') as archive,args.binary.open('xb') as output:
            output.write(struct.pack('<8s4I',b'M229BF01',24,896,HIDDEN,32))
            for li in range(24):
                matrices=[]; hashes={}
                for organ in ('gate','up','down'):
                    name=f'model.layers.{li}.mlp.{organ}_proj.weight'
                    w=archive.get_tensor(name)
                    assert w.dtype==torch.bfloat16 and w.shape==((896,4864) if organ=='down' else (4864,896))
                    hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes())
                    matrices.append(w.to(device).float())
                center=torch.from_numpy(states[0,li].copy()).view(torch.bfloat16).to(device).float()
                selected=torch.arange(HIDDEN,device=device)
                stage='source_value_gradient_identity'
                affine,bias,gs,us,ds,raw_affine,full_jac,value=source_prior(center,*matrices,selected)
                raw_bias=value-(F.linear(center,raw_affine)+N.donor_value(center,gs,us,ds))
                with torch.enable_grad():
                    reference=torch.autograd.functional.jacobian(
                        lambda x:combined_value(x,raw_affine,gs,us,ds,raw_bias),center,vectorize=True)
                difference=reference.double()-full_jac.double()
                relative=float(difference.norm()/full_jac.double().norm())
                peak=float(difference.abs().max()/full_jac.double().abs().max())
                raw_error=float((combined_value(center,raw_affine,gs,us,ds,raw_bias)-value).double().norm()/value.double().norm())
                stored_error=float((combined_value(center,affine.float(),gs,us,ds,bias)-value).double().norm()/value.double().norm())
                assert relative<=1e-5 and peak<=1e-4 and raw_error<=1e-5 and stored_error<=1e-5
                assert all(torch.isfinite(v).all() for v in (affine,bias,gs,us,ds))
                identities.append({'layer':li,'source_tensor_sha256':hashes,'center_token':0,
                    'center_bf16_sha256':P.M17.sha(states[0,li].tobytes()),'relative_jacobian_frobenius':relative,
                    'max_jacobian_error_relative_to_peak':peak,'raw_center_relative_l2':raw_error,
                    'stored_center_relative_l2':stored_error})
                values={'affine':affine.cpu().contiguous(),'gate':gs.bfloat16().cpu().contiguous(),
                    'up':us.bfloat16().cpu().contiguous(),'down':ds.bfloat16().cpu().contiguous(),
                    'bias':bias.cpu().contiguous()}
                stored.append(values)
                stage='source_fixture_serialization'
                for name,v in values.items():
                    raw=(v.view(torch.uint16) if name!='bias' else v).numpy().tobytes()
                    segments.append({'layer':li,'name':name,'offset':output.tell(),'bytes':len(raw),'sha256':P.M17.sha(raw)})
                    output.write(raw)
                del matrices,affine,bias,gs,us,ds,raw_affine,full_jac,value,reference,difference,center
                P.budget(start,device)
        assert args.binary.stat().st_size==302862360 and len(segments)==120
        assert len({s['sha256'] for s in segments if s['name']=='affine'})==24
        with args.binary.open('rb') as audit:
            assert audit.read(24)==struct.pack('<8s4I',b'M229BF01',24,896,HIDDEN,32)
            for row in segments:
                assert audit.tell()==row['offset'] and P.M17.sha(audit.read(row['bytes']))==row['sha256']
            assert audit.read(1)==b''
        stage='actual_combined_native_component'
        torch.cuda.synchronize(device)
        check=args.binary.with_suffix('.check.bin'); assert not check.exists()
        run=subprocess.run([str(args.exe),str(args.binary),str(C.VECTORS),str(check)],
            capture_output=True,text=True,check=True,timeout=600)
        timing=json.loads(run.stdout)
        assert timing['threads']==6 and timing['tokens']==256 and timing['layers']==24 and timing['hidden']==HIDDEN
        raw=check.read_bytes()
        assert len(raw)==20+16*24*896*4 and struct.unpack_from('<8s3I',raw)==(b'M229OUT1',16,24,896)
        actual=np.frombuffer(raw,dtype='<f4',offset=20).reshape(16,24,896)
        assert np.isfinite(actual).all()
        stage='stored_fp32_combined_oracle'; rows=[]
        for li,values in enumerate(stored):
            v={name:t.to(device).float() for name,t in values.items()}
            x=torch.from_numpy(states[:16,li].copy()).view(torch.bfloat16).to(device).float()
            expected=combined_value(x,v['affine'],v['gate'],v['up'],v['down'],v['bias']).cpu().numpy().astype(np.float64)
            errors=np.linalg.norm(actual[:,li].astype(np.float64)-expected,axis=1)/np.linalg.norm(expected,axis=1)
            assert np.isfinite(errors).all()
            rows.extend({'token':t,'layer':li,'relative_l2':float(e)} for t,e in enumerate(errors))
            P.budget(start,device)
        errors=[r['relative_l2'] for r in rows]
        gates={'source_value_gradient_identity':True,'stored_center_identity':True,
            'all120_segments_readback_exact':True,'all24_affine_fixtures_distinct':True,
            'native_numeric':statistics.median(errors)<=1e-4 and max(errors)<=5e-4,
            'native_combined_cost':statistics.median(timing['pass_ms_per_token'])<=10}
        result={'experiment':'METH-229-affine-plus-source-nonlinearity-native-qualification','hidden':HIDDEN,
            'source_sha256':P.M57.MODEL_SHA,'source_revision':P.M42.REV,'vectors_sha256':C.VECTORS_SHA,
            'identities':identities,'segments':segments,'binary':{'bytes':args.binary.stat().st_size,'sha256':P.digest(args.binary)},
            'output_sha256':P.digest(check),'executable_sha256':P.digest(args.exe),
            'source_code_sha256':P.digest(args.exe.with_suffix('.c')),'script_sha256':P.digest(Path(__file__)),
            'included_meth182_source_sha256':P.digest(args.exe.parent/'meth182_group64_ffn_cpu.c'),
            'timing':timing,'rows':rows,'gates':gates,'summary':{'median_relative_l2':statistics.median(errors),
                'maximum_relative_l2':max(errors),'median_ms_per_token':statistics.median(timing['pass_ms_per_token'])},
            'runtime':P.budget(start,device),'torch_version':torch.__version__,
            'decision':'combined_operator_pass_freeze_source_anchored_nonlinear_pair' if all(gates.values()) else 'stop_this_H2048_combined_native_operator',
            'scope':'Distinct source first2048-row component fixtures only; no selected/trained conditional bank, donor accuracy, n-scaled routing/DRAM, full rate or second family'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'decision':result['decision'],'summary':result['summary'],'gates':gates,'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'elapsed_seconds':time.monotonic()-start,'identities':identities,'segments':segments},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
