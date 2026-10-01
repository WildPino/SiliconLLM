#!/usr/bin/env python3
"""Price actual rank32 BF16 correction over the byte-unchanged mixed FFN."""
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
import meth239_mixed_output_priors as Q

L,P,N,C=Q.L,Q.P,Q.N,Q.L.C
RANK=32
AUDIT=P.DOC/'meth244_readout_encoding_result.json'
AUDIT_SHA='5164e4deb8ee6866b9d3b1f62f0d3468f6e7d814c517000876ccbf8305949849'
BYTES=326580256


def tensor_from_raw(name,raw):
    dtype='<u2' if name.endswith(('.ids','.escape')) or name.startswith('residual.') and name!='residual.bias' else np.int8 if name.endswith('.q') else '<f4'
    t=torch.from_numpy(np.frombuffer(raw,dtype=dtype).copy())
    if name.endswith('.q'): t=t.reshape((896,4864) if name.startswith('down.') else (4864,896))
    if name.endswith(('.ids','.escape')): t=t.reshape(896,32)
    if name=='residual.right': t=t.reshape(RANK,4864)
    if name=='residual.left': t=t.reshape(896,RANK)
    if name.endswith('.escape') or name in ('residual.right','residual.left'): t=t.view(torch.bfloat16)
    return t


def features(x,v):
    g=L.row_linear(x,v['gate.q'],v['gate.scale']); u=L.row_linear(x,v['up.q'],v['up.scale'])
    return L.silu_lookup(g)*u


def combined_value(x,v):
    phi=features(x,v)
    base=L.mixed_linear(phi,*(v['down.'+name] for name in ('q','scale','ids','escape')))
    correction=F.linear(F.linear(phi,v['residual.right'].float()),v['residual.left'].float())
    return ((base+correction)+v['bias'])+v['residual.bias']


def main():
    ap=argparse.ArgumentParser()
    for name in ('binary','exe','out'): ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args(); assert not args.binary.exists() and not args.out.exists()
    args.binary.parent.mkdir(parents=True,exist_ok=True); args.out.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.binary.parent).free>=2*1024**3
    start,stage=time.monotonic(),'bindings'; segments=[]; fidelity=[]; factors=[]
    try:
        assert P.digest(AUDIT)==AUDIT_SHA and P.digest(Q.NATIVE)==Q.NATIVE_SHA
        audit=json.loads(AUDIT.read_text()); native=json.loads(Q.NATIVE.read_text())
        assert audit['decision']=='continuous_local_count_gain_exposed_change_readout_encoding'
        assert all(native['gates'].values()) and P.digest(Q.FIXTURE)==native['binary']['sha256']
        assert P.digest(C.VECTORS)==C.VECTORS_SHA
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        device=Q.M.D.Q.setup(); P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
        raw=C.VECTORS.read_bytes(); assert struct.unpack_from('<8s4I',raw)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(raw,dtype='<u2',offset=24).reshape(256,24,896)
        stored=[]; source_hashes=[]
        with safe_open(str(source),framework='pt',device='cpu') as archive, Q.FIXTURE.open('rb') as original, args.binary.open('xb') as output:
            assert original.read(24)==struct.pack('<8s4I',b'M238ES01',24,896,4864,32)
            output.write(struct.pack('<8s5I',b'M245RS01',24,896,4864,32,RANK))
            def write(name,li,raw):
                segments.append({'layer':li,'name':name,'offset':output.tell(),'bytes':len(raw),'sha256':P.M17.sha(raw)})
                output.write(raw)
            table=native['segments'][0]; original.seek(table['offset']); raw=original.read(table['bytes'])
            assert P.M17.sha(raw)==table['sha256']; write('silu_table',None,raw)
            L.TABLE=tensor_from_raw('silu_table',raw).to(device)
            for li in range(24):
                v={}
                for segment in native['segments']:
                    if segment['layer']!=li: continue
                    original.seek(segment['offset']); raw=original.read(segment['bytes'])
                    assert P.M17.sha(raw)==segment['sha256']; write(segment['name'],li,raw)
                    v[segment['name']]=tensor_from_raw(segment['name'],raw)
                matrices=[]; hashes={}
                for organ in ('gate','up','down'):
                    name=f'model.layers.{li}.mlp.{organ}_proj.weight'; w=archive.get_tensor(name)
                    assert w.dtype==torch.bfloat16
                    hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes()); matrices.append(w.to(device).float())
                assert hashes==native['source_tensor_hashes'][li]['sha256']
                source_hashes.append({'layer':li,'sha256':hashes})
                stage='fixed_source_residual_SVD_and_BF16_factor_encoding'
                vd={name:t.to(device) for name,t in v.items()}
                decoded=L.decode_mixed(*(vd['down.'+name] for name in ('q','scale','ids','escape')))
                residual=matrices[2]-decoded
                u,s,vh=torch.linalg.svd(residual.double(),full_matrices=False,driver='gesvdj')
                squared=float(residual.double().square().sum()); spectral=float(s.double().square().sum())
                assert squared>0 and abs(spectral/squared-1)<=1e-5 and bool((s[:-1]>=s[1:]).all())
                ur=u[:,:RANK]; vr=vh[:RANK]
                ids=ur.abs().argmax(0); signs=torch.sign(ur[ids,torch.arange(RANK,device=device)])
                assert bool((signs.abs()==1).all())
                root=s[:RANK].sqrt()
                left=(ur*signs[None,:]*root[None,:]).bfloat16().contiguous()
                right=(vr*signs[:,None]*root[:,None]).bfloat16().contiguous()
                assert torch.isfinite(left).all() and torch.isfinite(right).all()
                assert bool((left!=0).any()) and bool((right!=0).any())
                factor_error=float((left.float().double()@right.float().double()-residual.double()).square().sum())/squared
                factors.append({'layer':li,'rank':RANK,'residual_squared_norm':squared,
                    'SVD_energy_relative_check':abs(spectral/squared-1),
                    'unrounded_top32_energy_fraction':float(s[:RANK].double().square().sum())/spectral,
                    'stored_BF16_factor_weight_relative_squared_error':factor_error})
                for name,t in (('residual.right',right.cpu()),('residual.left',left.cpu()),('residual.bias',torch.zeros(896))):
                    raw=(t.view(torch.uint16) if t.dtype==torch.bfloat16 else t).numpy().tobytes()
                    write(name,li,raw); v[name]=t
                vd={name:t.to(device) for name,t in v.items()}
                x=torch.from_numpy(states[:,li].copy()).view(torch.bfloat16).to(device).float()
                reference=N.donor_value(x,*matrices); expected=combined_value(x,vd)
                assert torch.isfinite(reference).all() and torch.isfinite(expected).all()
                sse=float((reference.double()-expected.double()).square().sum()); energy=float(reference.double().square().sum())
                fidelity.append({'layer':li,'states':256,'sse':sse,'energy':energy,'normalized_sse':sse/energy})
                stored.append(v)
                args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'factors':factors,'source_fidelity':fidelity},indent=2)+'\n',encoding='utf-8')
                del matrices,decoded,residual,u,s,vh,ur,vr,left,right,vd,x,reference,expected,w
                P.budget(start,device)
        assert args.binary.stat().st_size==BYTES and len(segments)==289
        base={(r['layer'],r['name']):r for r in native['segments']}
        unchanged=[r for r in segments if not r['name'].startswith('residual.')]
        assert len(unchanged)==217 and all((r['bytes'],r['sha256'])==(base[(r['layer'],r['name'])]['bytes'],base[(r['layer'],r['name'])]['sha256']) for r in unchanged)
        distinct={name:len({s['sha256'] for s in segments if s['name']==name}) for name in ('residual.left','residual.right')}
        assert all(n==24 for n in distinct.values())
        with args.binary.open('rb') as check:
            assert check.read(28)==struct.pack('<8s5I',b'M245RS01',24,896,4864,32,RANK)
            for r in segments: assert check.tell()==r['offset'] and P.M17.sha(check.read(r['bytes']))==r['sha256']
            assert check.read(1)==b''
        stage='combined_native_component_once'
        check_path=args.binary.with_suffix('.check.bin'); assert not check_path.exists()
        torch.cuda.synchronize(device)
        run=subprocess.run([str(args.exe),str(args.binary),str(C.VECTORS),str(check_path)],capture_output=True,text=True,check=True,timeout=600)
        timing=json.loads(run.stdout)
        assert all(timing[name]==value for name,value in (('threads',6),('tokens',256),('layers',24),('hidden',4864),('residual_rank',RANK),('weight_bytes',BYTES)))
        assert len(timing['pass_ms_per_token'])==3
        raw=check_path.read_bytes(); assert len(raw)==20+16*24*896*4 and struct.unpack_from('<8s3I',raw)==(b'M245OUT1',16,24,896)
        actual=np.frombuffer(raw,dtype='<f4',offset=20).reshape(16,24,896); assert np.isfinite(actual).all()
        rows=[]; stage='actual_combined_separate_GPU_oracle'
        for li,v in enumerate(stored):
            vd={name:t.to(device) for name,t in v.items()}
            x=torch.from_numpy(states[:16,li].copy()).view(torch.bfloat16).to(device).float()
            expected=combined_value(x,vd).cpu().numpy().astype(np.float64)
            errors=np.linalg.norm(actual[:,li].astype(np.float64)-expected,axis=1)/np.linalg.norm(expected,axis=1)
            assert np.isfinite(errors).all(); rows.extend({'token':t,'layer':li,'relative_l2':float(e)} for t,e in enumerate(errors))
            del vd,x,expected; P.budget(start,device)
        errors=[r['relative_l2'] for r in rows]; pooled=sum(r['sse'] for r in fidelity)/sum(r['energy'] for r in fidelity)
        gates={'source_native_audit_vector_bindings':True,'all217_base_segments_unchanged':True,
            'all289_segments_readback_exact':True,'all24_nonzero_distinct_BF16_factor_pairs':True,
            'all24_source_SVD_energy_checks':True,'native_numeric':statistics.median(errors)<=1e-4 and max(errors)<=5e-4,
            'source_function_error':pooled<=.01,'native_component_cost':statistics.median(timing['pass_ms_per_token'])<=10}
        result={'experiment':'METH-245-mixed-parent-plus-separate-rank32-BF16-residual-native-qualification',
            'rank':RANK,'native_result_sha256':Q.NATIVE_SHA,'encoding_audit_sha256':AUDIT_SHA,
            'source_sha256':P.M57.MODEL_SHA,'source_revision':P.M42.REV,'vectors_sha256':C.VECTORS_SHA,
            'source_tensor_hashes':source_hashes,'segments':segments,'source_factors':factors,'distinct_factor_hashes':distinct,
            'source_fidelity':fidelity,'rows':rows,'timing':timing,'gates':gates,
            'summary':{'median_relative_l2':statistics.median(errors),'maximum_relative_l2':max(errors),
                'pooled_source_function_normalized_sse':pooled,'maximum_layer_source_normalized_sse':max(r['normalized_sse'] for r in fidelity),
                'median_ms_per_token':statistics.median(timing['pass_ms_per_token'])},
            'binary':{'bytes':BYTES,'sha256':P.digest(args.binary)},'output_sha256':P.digest(check_path),
            'executable_sha256':P.digest(args.exe),'source_code_sha256':P.digest(args.exe.with_suffix('.c')),
            'included_meth182_source_sha256':P.digest(args.exe.parent/'meth182_group64_ffn_cpu.c'),
            'script_sha256':P.digest(Path(__file__)),'runtime':P.budget(start,device),'torch_version':torch.__version__,
            'decision':'separate_rank32_operator_pass_freeze_learned_residual_factorization' if all(gates.values()) else 'stop_this_fixed_rank32_separate_residual_operator',
            'scope':'Source-derived residual fixtures only,all shared/base bytes unchanged. No trained conditional bank,n-scaled routing/DRAM,full independent LLM quality/accepted rate or second donor.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'seconds':time.monotonic()-start,'segments':segments,'factors':factors,'source_fidelity':fidelity},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
