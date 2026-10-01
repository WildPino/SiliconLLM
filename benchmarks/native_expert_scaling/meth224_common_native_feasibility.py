#!/usr/bin/env python3
"""One BF16/FP32 H768 common fixture before learned function distillation."""
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
import meth219_native_arithmetic_development as D

P,C=D.P,D.C


def main():
    ap=argparse.ArgumentParser()
    for key in ('binary','exe','out'):
        ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args()
    assert not args.out.exists() and not args.binary.exists()
    args.binary.parent.mkdir(parents=True,exist_ok=True); args.out.parent.mkdir(parents=True,exist_ok=True)
    start,stage,segments=time.monotonic(),'bindings',[]
    try:
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA and P.digest(C.VECTORS)==C.VECTORS_SHA
        device=D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=15*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        stage='actual_donor_fixture_export'
        with safe_open(str(source),framework='pt',device='cpu') as archive,args.binary.open('xb') as out:
            out.write(struct.pack('<8s4I',b'M224BF01',24,896,768,32))
            for li in range(24):
                for organ in ('gate','up','down'):
                    name=f'model.layers.{li}.mlp.{organ}_proj.weight'
                    original=archive.get_tensor(name)
                    assert original.dtype==torch.bfloat16
                    assert original.shape==((896,4864) if organ=='down' else (4864,896))
                    selected=(original[:,:768] if organ=='down' else original[:768,:]).contiguous()
                    raw=selected.view(torch.uint16).numpy().tobytes()
                    segments.append({'layer':li,'name':name,'offset':out.tell(),'bytes':len(raw),
                        'sha256':P.M17.sha(raw),'source_tensor_sha256':P.M17.sha(original.view(torch.uint16).numpy().tobytes()),
                        'selection':'first_768_input_columns' if organ=='down' else 'first_768_output_rows'})
                    out.write(raw)
                raw=np.zeros(896,dtype=np.float32).tobytes()
                segments.append({'layer':li,'name':'constructed_zero_output_bias','offset':out.tell(),
                    'bytes':len(raw),'sha256':P.M17.sha(raw)})
                out.write(raw); P.budget(start,device)
        assert len(segments)==96 and args.binary.stat().st_size==99176472
        with args.binary.open('rb') as audit:
            assert audit.read(24)==struct.pack('<8s4I',b'M224BF01',24,896,768,32)
            for row in segments:
                assert audit.tell()==row['offset'] and P.M17.sha(audit.read(row['bytes']))==row['sha256']
            assert audit.read(1)==b''
        stage='native_component'
        output=args.binary.with_suffix('.check.bin'); assert not output.exists()
        run=subprocess.run([str(args.exe),str(args.binary),str(C.VECTORS),str(output)],
            capture_output=True,text=True,check=True,timeout=600)
        timing=json.loads(run.stdout)
        assert timing['threads']==6 and timing['tokens']==256 and timing['layers']==24 and timing['hidden']==768
        raw=output.read_bytes()
        assert len(raw)==20+16*24*896*4 and struct.unpack_from('<8s3I',raw)==(b'M224OUT1',16,24,896)
        observed=np.frombuffer(raw,dtype='<f4',offset=20).reshape(16,24,896)
        assert np.isfinite(observed).all()
        states=np.frombuffer(C.VECTORS.read_bytes(),dtype='<u2',offset=24).reshape(256,24,896)
        rows=[]; stage='fp32_oracle'
        with safe_open(str(source),framework='pt',device='cpu') as archive,torch.inference_mode():
            for li in range(24):
                matrices=[]
                for organ in ('gate','up','down'):
                    weight=archive.get_tensor(f'model.layers.{li}.mlp.{organ}_proj.weight')
                    selected=weight[:,:768] if organ=='down' else weight[:768,:]
                    matrices.append(selected.contiguous().to(device).float())
                x=torch.from_numpy(states[:16,li].copy()).view(torch.bfloat16).to(device).float()
                expected=F.linear(F.silu(F.linear(x,matrices[0]))*F.linear(x,matrices[1]),matrices[2]).cpu().numpy().astype(np.float64)
                errors=np.linalg.norm(observed[:,li].astype(np.float64)-expected,axis=1)/np.linalg.norm(expected,axis=1)
                assert np.isfinite(errors).all()
                rows.extend({'token':t,'layer':li,'relative_l2':float(error)} for t,error in enumerate(errors))
                P.budget(start,device)
        errors=[r['relative_l2'] for r in rows]
        numeric=statistics.median(errors)<=1e-4 and max(errors)<=5e-4
        cost=statistics.median(timing['pass_ms_per_token'])<=10
        summary={'rows':len(rows),'median_relative_l2':statistics.median(errors),'maximum_relative_l2':max(errors),
            'numeric_pass':numeric,'median_common_ms_per_token':statistics.median(timing['pass_ms_per_token']),'cost_pass':cost}
        result={'experiment':'METH-224-H768-BF16-FP32-common-native-fixture','source_sha256':P.M57.MODEL_SHA,
            'source_revision':P.M42.REV,'vectors_sha256':C.VECTORS_SHA,'segments':segments,
            'binary':{'bytes':args.binary.stat().st_size,'sha256':P.digest(args.binary)},
            'output_sha256':P.digest(output),'executable_sha256':P.digest(args.exe),
            'source_code_sha256':P.digest(args.exe.with_suffix('.c')),'script_sha256':P.digest(Path(__file__)),
            'summary':summary,'timing':timing,'rows':rows,'runtime':P.budget(start,device),
            'decision':'component_pass_freeze_nonlinear_function_distillation' if numeric and cost else 'stop_this_H768_BF16_native_kernel_at_component_gate',
            'scope':'Actual source-weight subsets as shape/precision fixture only; no quality candidate, trained compact core, full rate, useful-n or multi-family claim'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'summary':summary,'decision':result['decision'],'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'elapsed_seconds':time.monotonic()-start,'segments':segments},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':
    main()
