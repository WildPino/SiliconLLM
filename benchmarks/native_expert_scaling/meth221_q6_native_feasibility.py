#!/usr/bin/env python3
"""Export actual Q6 bytes losslessly and check one fixed native FP32 kernel."""
import argparse
import json
from pathlib import Path
import statistics
import struct
import subprocess
import time
import numpy as np
from safetensors import safe_open
import torch
from torch.nn import functional as F
import meth219_native_arithmetic_development as D

P,C=D.P,D.C
CORE=P.ROOT/'results/native_expert_scaling/meth186_qwen05b_instruct_group64_q6_core.safetensors'
CORE_SHA='f68ed7ba06eb38a19d883310c86d9e569bf719e726a664a6287edb463d3ae725'
EXPORT=P.DOC/'meth186_group64_q6_core_export.json'
EXPORT_SHA='70b325aca75eff11f03fe0afbd9aab5cb218d9e41ebf5b79e1627dc05c58f6b2'


def unsigned_codes(packed):
    rows,width=packed.shape
    assert packed.dtype==np.uint8 and width%3==0
    triples=packed.reshape(rows,-1,3).astype(np.uint32)
    value=triples[...,0] | triples[...,1]<<8 | triples[...,2]<<16
    return np.stack([(value>>s)&63 for s in (0,6,12,18)],axis=-1).reshape(rows,width//3*4).astype(np.uint8)


def planar(codes):
    rows,width=codes.shape
    assert width%64==0 and ((codes>=1)&(codes<=63)).all()
    groups=codes.reshape(rows,width//64,64)
    low=(groups[...,::2]&15) | ((groups[...,1::2]&15)<<4)
    high=np.zeros((*groups.shape[:2],16),dtype=np.uint8)
    for j in range(4):
        high|=(groups[...,j::4]>>4)<<(j*2)
    return np.concatenate((low,high),axis=-1).reshape(rows,width//4*3)


def unplanar(packed):
    rows,width=packed.shape
    groups=packed.reshape(rows,-1,48)
    low=groups[...,:32]
    low_codes=np.stack((low&15,low>>4),axis=-1).reshape(rows,-1,64)
    high=groups[...,32:]
    high_codes=np.stack([(high>>(j*2))&3 for j in range(4)],axis=-1).reshape(rows,-1,64)
    return (low_codes | high_codes<<4).reshape(rows,width//3*4)


def main():
    ap=argparse.ArgumentParser()
    for key in ('binary','exe','out'):
        ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args()
    assert not args.binary.exists() and not args.out.exists()
    start,stage,export_rows=time.monotonic(),'bindings',[]
    args.binary.parent.mkdir(parents=True,exist_ok=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    try:
        assert P.digest(CORE)==CORE_SHA and P.digest(EXPORT)==EXPORT_SHA
        old=json.loads(EXPORT.read_text(encoding='utf-8'))
        assert old['sha256']==CORE_SHA and P.digest(C.VECTORS)==C.VECTORS_SHA
        device=D.Q.setup()
        P.MAX_SECONDS=P.M17.MAX_SECONDS=15*60
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        stage='lossless_native_layout_export'
        header=struct.pack('<8s4I',b'M221Q6P1',24,896,4864,64)
        with safe_open(str(CORE),framework='pt',device='cpu') as archive,args.binary.open('xb') as output:
            assert archive.metadata()['format']=='QWEN25_INSTRUCT_R8H_GROUP64_Q6FFN_BF16ATTN_V1'
            output.write(header)
            for li in range(24):
                for organ in ('gate','up','down'):
                    name=f'model.layers.{li}.mlp.{organ}_proj.weight'
                    source=archive.get_tensor(name+'.q6')
                    scales=archive.get_tensor(name+'.scale')
                    codes=unsigned_codes(source.numpy())
                    converted=planar(codes)
                    assert np.array_equal(unplanar(converted),codes)
                    assert scales.dtype==torch.float16 and scales.shape==(codes.shape[0],codes.shape[1]//64)
                    assert ((scales>0)&torch.isfinite(scales)).all()
                    code_bytes=converted.tobytes()
                    scale_bytes=scales.view(torch.uint16).numpy().tobytes()
                    offset=output.tell()
                    output.write(code_bytes); output.write(scale_bytes)
                    export_rows.append({'layer':li,'organ':organ,'offset':offset,
                        'source_q6_sha256':P.M17.sha(source.numpy().tobytes()),
                        'decoded_unsigned_sha256':P.M17.sha(codes.tobytes()),
                        'planar_sha256':P.M17.sha(code_bytes),'scale_sha256':P.M17.sha(scale_bytes),
                        'code_bytes':len(code_bytes),'scale_bytes':len(scale_bytes),
                        'every_code_roundtrip_equal':True})
                    P.budget(start,device)
        assert len(export_rows)==72 and args.binary.stat().st_size==245145624
        with args.binary.open('rb') as audit:
            assert audit.read(24)==header
            for row in export_rows:
                assert audit.tell()==row['offset']
                assert P.M17.sha(audit.read(row['code_bytes']))==row['planar_sha256']
                assert P.M17.sha(audit.read(row['scale_bytes']))==row['scale_sha256']
            assert audit.read(1)==b''
        stage='native_component'
        output_path=args.binary.with_suffix('.check.bin')
        assert not output_path.exists()
        run=subprocess.run([str(args.exe),str(args.binary),str(C.VECTORS),str(output_path)],
            capture_output=True,text=True,check=True,timeout=600)
        timing=json.loads(run.stdout)
        assert timing['threads']==6 and timing['tokens']==256 and timing['layers']==24 and timing['codec_patterns']==64
        raw=output_path.read_bytes()
        assert len(raw)==20+16*24*896*4 and struct.unpack_from('<8s3I',raw)==(b'M221OUT1',16,24,896)
        actual=np.frombuffer(raw,dtype='<f4',offset=20).reshape(16,24,896)
        assert np.isfinite(actual).all()
        states=np.frombuffer(C.VECTORS.read_bytes(),dtype='<u2',offset=24).reshape(256,24,896)
        rows=[]
        stage='fp32_numerical_oracle'
        with safe_open(str(CORE),framework='pt',device='cpu') as archive,torch.inference_mode():
            for li in range(24):
                matrices=[]
                for organ in ('gate','up','down'):
                    name=f'model.layers.{li}.mlp.{organ}_proj.weight'
                    unsigned=unsigned_codes(archive.get_tensor(name+'.q6').numpy())
                    codes=torch.from_numpy(unsigned.astype(np.int16)-32).to(device).float()
                    scales=archive.get_tensor(name+'.scale').to(device).float()
                    rows_count,width=codes.shape
                    matrices.append((codes.reshape(rows_count,width//64,64)*scales.unsqueeze(-1)).reshape(rows_count,width))
                x=torch.from_numpy(states[:16,li].copy()).view(torch.bfloat16).to(device).float()
                expected=F.linear(F.silu(F.linear(x,matrices[0]))*F.linear(x,matrices[1]),matrices[2]).cpu().numpy().astype(np.float64)
                errors=np.linalg.norm(actual[:,li].astype(np.float64)-expected,axis=1)/np.linalg.norm(expected,axis=1)
                assert np.isfinite(errors).all()
                rows.extend({'token':t,'layer':li,'relative_l2':float(v)} for t,v in enumerate(errors))
                P.budget(start,device)
        errors=[r['relative_l2'] for r in rows]
        numeric=statistics.median(errors)<=1e-4 and max(errors)<=5e-4
        cost=statistics.median(timing['pass_ms_per_token'])<=10
        summary={'rows':len(rows),'median_relative_l2':statistics.median(errors),'maximum_relative_l2':max(errors),
            'numeric_pass':numeric,'median_ffn_ms_per_token':statistics.median(timing['pass_ms_per_token']),'cost_pass':cost}
        result={'experiment':'METH-221-Q6-planar-FP32-native-component','core_sha256':CORE_SHA,
            'source_export_sha256':EXPORT_SHA,'vectors_sha256':C.VECTORS_SHA,'export_rows':export_rows,
            'binary':{'bytes':args.binary.stat().st_size,'sha256':P.digest(args.binary)},
            'native_output_sha256':P.digest(output_path),'executable_sha256':P.digest(args.exe),
            'source_sha256':P.digest(args.exe.with_suffix('.c')),'script_sha256':P.digest(Path(__file__)),
            'summary':summary,'timing':timing,'rows':rows,'runtime':P.budget(start,device),
            'hypothetical_exact_head_q6_active_weight_bytes':481347584,
            'decision':'component_pass_freeze_changed_conditional_recovery_pilot' if numeric and cost else 'stop_this_Q6_planar_kernel_at_component_gate',
            'scope':'Actual saved FFN component on consumed BF16-source states only; direct Q6 quality remains failed; no native full rate, transfer recovery, or useful-n result'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'summary':summary,'decision':result['decision'],'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'elapsed_seconds':time.monotonic()-start,'completed_export_rows':export_rows},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':
    main()
