#!/usr/bin/env python3
"""Source-derived private nonlinear feature layout and native qualification."""
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
import meth245_separate_residual_feasibility as A

P,Q,L,N,C=A.P,A.Q,A.L,A.N,A.C
PRIOR=P.DOC/'meth245_separate_residual_native_repair1_result.json'
PRIOR_SHA='7022e2063afef2339342ff515ab4788732b7d795b8b2c690c511b4f5a5e5d61d'
NATIVE=P.DOC/'meth246_single_team_residual_native_result.json'
NATIVE_SHA='bb84bdb1544025866e7411dd830ffea3bc5daf4b73447bf194c45c55ba9da202'
FIXTURE=P.ROOT/'results/native_expert_scaling/meth245_separate_residual_fixture_repair1.bin'
PRIVATE=32
BYTES=329334308


def private_features(x,v):
    phi=A.features(x,v)
    selected=L.silu_lookup(F.linear(x,v['private.gate'].float()))*F.linear(x,v['private.up'].float())
    phi[:,v['private.ids'].long()]=selected
    return phi


def readout(phi,v):
    base=L.mixed_linear(phi,*(v['down.'+name] for name in ('q','scale','ids','escape')))
    correction=F.linear(F.linear(phi,v['residual.right'].float()),v['residual.left'].float())
    return ((base+correction)+v['bias'])+v['residual.bias']


def main():
    ap=argparse.ArgumentParser()
    for name in ('binary','exe','out'):ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args();assert not args.binary.exists() and not args.out.exists()
    args.binary.parent.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(args.binary.parent).free>=2*1024**3
    start=time.monotonic();stage='bindings';segments=[];selection=[];fidelity=[]
    try:
        assert P.digest(PRIOR)==PRIOR_SHA and P.digest(NATIVE)==NATIVE_SHA
        prior=json.loads(PRIOR.read_text());native=json.loads(NATIVE.read_text())
        assert all(native['gates'].values()) and P.digest(FIXTURE)==prior['binary']['sha256']==native['fixture_sha256']
        assert P.digest(C.VECTORS)==C.VECTORS_SHA
        old_check=P.ROOT/'results/native_expert_scaling/meth246_single_team_residual.check.bin'
        assert P.digest(old_check)==native['output_sha256']
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True));assert P.digest(source)==P.M57.MODEL_SHA
        device=Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        vector_raw=C.VECTORS.read_bytes();assert struct.unpack_from('<8s4I',vector_raw)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(vector_raw,dtype='<u2',offset=24).reshape(256,24,896);stored=[];source_hashes=[]
        with safe_open(str(source),framework='pt',device='cpu') as archive,FIXTURE.open('rb') as original,args.binary.open('xb') as output:
            assert original.read(28)==struct.pack('<8s5I',b'M245RS01',24,896,4864,32,32)
            output.write(struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,PRIVATE))
            def write(name,layer,raw):
                segments.append({'layer':layer,'name':name,'offset':output.tell(),'bytes':len(raw),'sha256':P.M17.sha(raw)})
                output.write(raw)
            table=prior['segments'][0];original.seek(table['offset']);raw=original.read(table['bytes']);assert P.M17.sha(raw)==table['sha256']
            write('silu_table',None,raw);L.TABLE=A.tensor_from_raw('silu_table',raw).to(device)
            for li in range(24):
                stage='byte_unchanged_base_and_source_only_private_rows';v={}
                for segment in prior['segments']:
                    if segment['layer']!=li:continue
                    original.seek(segment['offset']);raw=original.read(segment['bytes']);assert P.M17.sha(raw)==segment['sha256']
                    write(segment['name'],li,raw);v[segment['name']]=A.tensor_from_raw(segment['name'],raw)
                source_cpu=[];matrices=[];hashes={}
                for organ in ('gate','up','down'):
                    name=f'model.layers.{li}.mlp.{organ}_proj.weight';w=archive.get_tensor(name);assert w.dtype==torch.bfloat16
                    hashes[name]=P.M17.sha(w.view(torch.uint16).numpy().tobytes());source_cpu.append(w);matrices.append(w.to(device).float())
                assert hashes==prior['source_tensor_hashes'][li]['sha256'];source_hashes.append({'layer':li,'sha256':hashes})
                # Source-only FP64 coefficient discrepancy; no activation/target selection.
                vd={name:t.to(device) for name,t in v.items()};score=torch.zeros(4864,dtype=torch.float64,device=device)
                for index,organ in enumerate(('gate','up')):
                    decoded=vd[organ+'.q'].float()*vd[organ+'.scale'][:,None]
                    score+=(matrices[index].double()-decoded.double()).square().sum(1)
                order=torch.argsort(score,descending=True,stable=True);ids=order[:PRIVATE].sort().values.cpu()
                assert len(set(ids.tolist()))==PRIVATE and float(score[ids.to(device)].min())>0
                v['private.ids']=ids.to(torch.uint16)
                v['private.gate']=source_cpu[0][ids].contiguous();v['private.up']=source_cpu[1][ids].contiguous()
                for name in ('private.ids','private.gate','private.up'):
                    t=v[name];raw=(t.view(torch.uint16) if t.dtype==torch.bfloat16 else t).numpy().tobytes();write(name,li,raw)
                selection.append({'layer':li,'ids':ids.tolist(),'selected_discrepancy_energy':float(score[ids.to(device)].sum()),
                    'total_discrepancy_energy':float(score.sum()),'minimum_selected_score':float(score[ids.to(device)].min())})
                vd={name:t.to(device) for name,t in v.items()};x=torch.from_numpy(states[:,li].copy()).view(torch.bfloat16).to(device).float()
                phi=private_features(x,vd);plain=A.features(x,vd);change=float((phi.double()-plain.double()).square().sum());assert change>0
                expected=readout(phi,vd);reference=N.donor_value(x,*matrices)
                assert torch.isfinite(expected).all() and torch.isfinite(reference).all()
                sse=float((expected.double()-reference.double()).square().sum());energy=float(reference.double().square().sum())
                fidelity.append({'layer':li,'states':256,'sse':sse,'energy':energy,'normalized_sse':sse/energy,
                    'private_feature_change_squared_norm':change})
                stored.append(v);args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'selection':selection,'source_fidelity':fidelity},indent=2)+'\n',encoding='utf-8')
                del vd,x,phi,plain,expected,reference,matrices,source_cpu,w,decoded,score;P.budget(start,device)
        assert args.binary.stat().st_size==BYTES and len(segments)==361
        unchanged=[r for r in segments if not r['name'].startswith('private.')]
        old_segments={(r['layer'],r['name']):r for r in prior['segments']}
        assert len(unchanged)==289 and all((r['bytes'],r['sha256'])==(old_segments[(r['layer'],r['name'])]['bytes'],old_segments[(r['layer'],r['name'])]['sha256']) for r in unchanged)
        with args.binary.open('rb') as check:
            assert check.read(32)==struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,PRIVATE)
            for r in segments:assert check.tell()==r['offset'] and P.M17.sha(check.read(r['bytes']))==r['sha256']
            assert check.read(1)==b''
        stage='private_feature_native_component_once';check_path=args.binary.with_suffix('.check.bin');assert not check_path.exists()
        torch.cuda.synchronize(device)
        run=subprocess.run([str(args.exe),str(args.binary),str(C.VECTORS),str(check_path)],capture_output=True,text=True,check=True,timeout=180)
        timing=json.loads(run.stdout)
        assert all(timing[k]==v for k,v in (('threads',6),('tokens',256),('layers',24),('hidden',4864),('residual_rank',32),('private_units',32),('weight_bytes',BYTES)))
        assert len(timing['pass_ms_per_token'])==3
        raw=check_path.read_bytes();assert len(raw)==20+16*24*896*4 and struct.unpack_from('<8s3I',raw)==(b'M252OUT1',16,24,896)
        actual=np.frombuffer(raw,dtype='<f4',offset=20).reshape(16,24,896);assert np.isfinite(actual).all()
        old_actual=np.frombuffer(old_check.read_bytes(),dtype='<f4',offset=20).reshape(16,24,896)
        changed=[int(np.count_nonzero(actual[:,li]!=old_actual[:,li])) for li in range(24)]
        stage='independent_GPU_private_feature_oracle';rows=[]
        for li,v in enumerate(stored):
            vd={name:t.to(device) for name,t in v.items()};x=torch.from_numpy(states[:16,li].copy()).view(torch.bfloat16).to(device).float()
            expected=readout(private_features(x,vd),vd).cpu().numpy().astype(np.float64)
            errors=np.linalg.norm(actual[:,li].astype(np.float64)-expected,axis=1)/np.linalg.norm(expected,axis=1)
            assert np.isfinite(errors).all();rows.extend({'token':t,'layer':li,'relative_l2':float(e)} for t,e in enumerate(errors))
            del vd,x,expected;P.budget(start,device)
        errors=[r['relative_l2'] for r in rows];pooled=sum(r['sse'] for r in fidelity)/sum(r['energy'] for r in fidelity)
        gates={'all_source_prior_native_vector_bindings':True,'all289_original_segments_unchanged':True,
            'all361_segments_readback_exact':True,'all24_source_only_selected_BF16_rows_and_nonzero_feature_effects':True,
            'all24_native_output_payloads_change':all(n>0 for n in changed),
            'native_numeric':statistics.median(errors)<=1e-4 and max(errors)<=5e-4,
            'source_function_error':pooled<=.01,'native_component_cost':statistics.median(timing['pass_ms_per_token'])<=10}
        result={'experiment':'METH-252-original-BF16-private-nonlinear-feature-native-qualification','private_units':PRIVATE,
            'prior_result_sha256':PRIOR_SHA,'native_prior_sha256':NATIVE_SHA,'source_sha256':P.M57.MODEL_SHA,'source_revision':P.M42.REV,
            'source_tensor_hashes':source_hashes,'vectors_sha256':C.VECTORS_SHA,'segments':segments,'selection':selection,
            'source_fidelity':fidelity,'rows':rows,'changed_native_elements_per_layer':changed,'timing':timing,'gates':gates,
            'summary':{'median_relative_l2':statistics.median(errors),'maximum_relative_l2':max(errors),
                'pooled_source_function_normalized_sse':pooled,'source_fidelity_ratio_to_prior':pooled/prior['summary']['pooled_source_function_normalized_sse'],
                'maximum_layer_source_normalized_sse':max(r['normalized_sse'] for r in fidelity),'median_ms_per_token':statistics.median(timing['pass_ms_per_token'])},
            'binary':{'bytes':BYTES,'sha256':P.digest(args.binary)},'output_sha256':P.digest(check_path),
            'executable_sha256':P.digest(args.exe),'source_code_sha256':P.digest(args.exe.with_suffix('.c')),
            'included_meth182_source_sha256':P.digest(args.exe.parent/'meth182_group64_ffn_cpu.c'),'script_sha256':P.digest(Path(__file__)),
            'runtime':P.budget(start,device),'native_peak_working_set_bytes':timing['peak_working_set_bytes'],
            'decision':'private_source_feature_operator_pass_freeze_matched_unit_selection' if all(gates.values()) else 'stop_this_fixed_private_source_feature_operator',
            'scope':'Source-only static private32 unit choices;actual BF16 nonlinear rows,shared LUT/readout unchanged. Fixed prebuilt maps;no learned cell bank or dynamic route/map cost,large-n/DRAM,new/full quality/accepted rate/family transfer.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),'segments':segments,'selection':selection,'source_fidelity':fidelity,'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':main()
