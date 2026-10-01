#!/usr/bin/env python3
"""Localize M219 W8A8 emulator differences; price fixed W8A32 C alternative."""
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

P, L, C = D.P, D.L, D.C
COMPONENT = P.DOC/'meth219_native_arithmetic_development_component.json'
COMPONENT_SHA = 'b00676aa0b8c162386c3f9e38d9acd6a8f4afddd520bd947b2a2d52e5225e5f1'


def quantize(x):
    groups = x.reshape(*x.shape[:-1], x.shape[-1]//64, 64)
    maximum = groups.abs().amax(-1, keepdim=True)
    scales = torch.where(maximum>0,maximum/127,torch.ones_like(maximum))
    normalized = groups/scales
    codes = normalized.round().clamp(-127,127)
    return codes.reshape_as(x).to(torch.int8), scales.squeeze(-1), normalized.reshape_as(x)


def relative(a, b):
    return np.linalg.norm(a.astype(np.float64)-b.astype(np.float64),axis=-1)/np.linalg.norm(b.astype(np.float64),axis=-1)


def main():
    ap = argparse.ArgumentParser()
    for key in ('exe','tape','out'):
        ap.add_argument('--'+key,required=True,type=Path)
    args = ap.parse_args()
    assert not args.out.exists() and not args.tape.exists()
    start=time.monotonic()
    assert P.digest(COMPONENT)==COMPONENT_SHA and P.digest(L.CORE)==L.CORE_SHA
    previous=json.loads(COMPONENT.read_text(encoding='utf-8'))['component']
    binary=P.ROOT/'results/native_expert_scaling/meth219_m211_group64_ffn.bin'
    assert P.digest(binary)==previous['binary_sha256']
    assert P.digest(C.VECTORS)==C.VECTORS_SHA
    native=binary.with_suffix('.check.bin')
    assert P.digest(native)==previous['output_sha256']
    run=subprocess.run([str(args.exe),str(binary),str(C.VECTORS),str(args.tape)],
        capture_output=True,text=True,check=True,timeout=600)
    timing=json.loads(run.stdout)
    raw=args.tape.read_bytes()
    assert struct.unpack_from('<8s5I',raw)==(b'M220INT1',16,24,896,4864,64)
    dimensions=(('qx','i1',(896,)),('si','<f4',(14,)),('gate','<f4',(4864,)),
        ('up','<f4',(4864,)),('hidden','<f4',(4864,)),('qh','i1',(4864,)),
        ('sh','<f4',(76,)),('out','<f4',(896,)),('float_out','<f4',(896,)))
    records=[]
    offset=28
    for t in range(16):
        for li in range(24):
            row={'token':t,'layer':li}
            for name,dtype,shape in dimensions:
                row[name]=np.frombuffer(raw,dtype=dtype,count=int(np.prod(shape)),offset=offset).reshape(shape)
                offset+=row[name].nbytes
            records.append(row)
    assert offset==len(raw)
    taped=np.stack([r['out'] for r in records]).reshape(16,24,896)
    original=np.frombuffer(native.read_bytes(),dtype='<f4',offset=20).reshape(16,24,896)
    assert np.array_equal(taped,original), 'Instrumented C does not reproduce existing executable'
    device=D.Q.setup()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    P.MAX_SECONDS=P.M17.MAX_SECONDS=15*60
    states=np.frombuffer(C.VECTORS.read_bytes(),dtype='<u2',offset=24).reshape(256,24,896)
    rows=[]
    with safe_open(str(L.CORE),framework='pt',device='cpu') as archive, torch.inference_mode():
        for li in range(24):
            layer_rows=[r for r in records if r['layer']==li]
            matrices=[]
            for organ in ('gate','up','down'):
                name=f'model.layers.{li}.mlp.{organ}_proj.weight'
                matrices.append(D.reconstruct(archive.get_tensor(name+'.q8').to(device),
                    archive.get_tensor(name+'.scale').to(device)))
            x=torch.from_numpy(states[:16,li].copy()).view(torch.bfloat16).to(device).float()
            qx,si,_=quantize(x)
            deq=(qx.reshape(16,14,64).float()*si.unsqueeze(-1)).reshape(16,896)
            gate,up=F.linear(deq,matrices[0]),F.linear(deq,matrices[1])
            hidden=F.silu(gate)*up
            qh,sh,normalized=quantize(hidden)
            deq_h=(qh.reshape(16,76,64).float()*sh.unsqueeze(-1)).reshape(16,4864)
            emulated=F.linear(deq_h,matrices[2]).cpu().numpy()
            taped_qh=torch.from_numpy(np.stack([r['qh'] for r in layer_rows])).to(device).float()
            taped_sh=torch.from_numpy(np.stack([r['sh'] for r in layer_rows])).to(device)
            replay_h=(taped_qh.reshape(16,76,64)*taped_sh.unsqueeze(-1)).reshape(16,4864)
            replay=F.linear(replay_h,matrices[2]).cpu().numpy()
            float_expected=F.linear(F.silu(F.linear(x,matrices[0]))*F.linear(x,matrices[1]),matrices[2]).cpu().numpy()
            qx_np,si_np=qx.cpu().numpy(),si.cpu().numpy()
            qh_np,sh_np=qh.cpu().numpy(),sh.cpu().numpy()
            gate_np,up_np=gate.cpu().numpy(),up.cpu().numpy()
            norm_np=normalized.cpu().numpy()
            for t,r in enumerate(layer_rows):
                mismatch=np.flatnonzero(qh_np[t]!=r['qh'])
                boundary=np.abs(np.abs(norm_np[t,mismatch]-np.round(norm_np[t,mismatch]))-.5)
                rows.append({'token':t,'layer':li,
                    'input_code_mismatches':int((qx_np[t]!=r['qx']).sum()),
                    'input_scale_max_abs':float(np.abs(si_np[t]-r['si']).max()),
                    'gate_relative_l2':float(relative(gate_np[t],r['gate'])),
                    'up_relative_l2':float(relative(up_np[t],r['up'])),
                    'hidden_code_mismatches':len(mismatch),
                    'hidden_scale_max_abs':float(np.abs(sh_np[t]-r['sh']).max()),
                    'mismatched_hidden_max_boundary_distance':float(boundary.max()) if len(boundary) else None,
                    'w8a8_output_relative_l2':float(relative(emulated[t],r['out'])),
                    'c_hidden_replay_relative_l2':float(relative(replay[t],r['out'])),
                    'float_input_output_relative_l2':float(relative(r['float_out'],float_expected[t]))})
            P.budget(start,device)
    w8=[r['w8a8_output_relative_l2'] for r in rows]
    replay=[r['c_hidden_replay_relative_l2'] for r in rows]
    fp=[r['float_input_output_relative_l2'] for r in rows]
    failures=[r for r in rows if r['w8a8_output_relative_l2']>5e-4]
    diagnosis=all(r['input_code_mismatches']==0 and r['input_scale_max_abs']==0 for r in rows) and all(
        r['hidden_code_mismatches']>0 and r['mismatched_hidden_max_boundary_distance']<=1e-3 for r in failures) and max(replay)<=5e-4
    numeric=statistics.median(fp)<=1e-4 and max(fp)<=5e-4
    cost=statistics.median(timing['float_input_pass_ms_per_token'])<=10
    summary={'rows':len(rows),'taped_c_outputs_exact':True,'w8a8_median_relative_l2':statistics.median(w8),
        'w8a8_max_relative_l2':max(w8),'w8a8_rows_over_frozen_limit':len(failures),
        'input_code_mismatches':sum(r['input_code_mismatches'] for r in rows),
        'hidden_code_mismatches':sum(r['hidden_code_mismatches'] for r in rows),
        'c_hidden_replay_max_relative_l2':max(replay),'hidden_boundary_diagnosis_pass':diagnosis,
        'float_input_median_relative_l2':statistics.median(fp),'float_input_max_relative_l2':max(fp),
        'float_input_numeric_pass':numeric,'float_input_median_ms_per_token':statistics.median(timing['float_input_pass_ms_per_token']),
        'float_input_cost_pass':cost}
    result={'experiment':'METH-220-native-FFN-intermediate-diagnostic','core_sha256':L.CORE_SHA,
        'component_sha256':COMPONENT_SHA,'ffn_binary_sha256':previous['binary_sha256'],
        'vectors_sha256':C.VECTORS_SHA,'exe_sha256':P.digest(args.exe),'tape_sha256':P.digest(args.tape),
        'script_sha256':P.digest(Path(__file__)),'source_sha256':P.digest(args.exe.with_suffix('.c')),
        'summary':summary,'timing':timing,'rows':rows,'runtime':P.budget(start,device),
        'decision':'float_input_component_pass_freeze_quality_protocol' if numeric and cost else 'stop_this_float_input_native_kernel_at_component_gate',
        'scope':'Existing component states only; no source-quality, generation/task, full native rate or large-n claim'}
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'decision':result['decision'],'runtime':result['runtime']}),flush=True)


if __name__=='__main__':
    main()
