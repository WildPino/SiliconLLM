#!/usr/bin/env python3
"""Fixed I16/Q8 shared projections: GPU/source,exact CPU integer,then cost."""
import argparse
import json
from pathlib import Path
import statistics
import struct
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as TF
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import meth271_output_aware_rows as A

P,G,Q=A.P,A.G,A.Q
PRIOR=P.DOC/'meth271_output_aware_rows_result.json'
PRIOR_SHA='638594a650189ab45efa62c08be04a3e42b21f7eff00c52d372d5991077120c3'
PHASE=P.DOC/'meth273_private128_phase_result.json'
PHASE_SHA='d8918584b3c7d9ac323347e979600a15099559cacc3016c92c132f1144225643'
FIXTURE=P.ROOT/'results/native_expert_scaling/meth271_output_aware_rows_fixture.bin'
SOURCE=Path(__file__).with_name('meth274_i16_shared_input_cpu.c')


def input_codes(x):
    assert x.dtype==torch.float32 and torch.isfinite(x).all()
    maximum=x.abs().amax(1).cpu().numpy()
    inverse=np.divide(np.float32(32767),np.where(maximum>0,maximum,np.float32(1)),dtype=np.float32)
    inverse[maximum==0]=np.float32(1)
    scale=np.divide(np.float32(1),inverse,dtype=np.float32)
    # CPU IEEE FP32 divisions avoid a different CUDA reciprocal implementation.
    inv=torch.from_numpy(inverse).to(x.device)
    codes=torch.round(x*inv[:,None]).clamp(-32767,32767).to(torch.int16)
    return codes,inverse,scale


def private_features(x,v,codes,scale):
    values=codes.float();s=torch.from_numpy(scale).to(x.device)[:,None]
    gate=(TF.linear(values,v['gate.q'].float())*s)*v['gate.scale'][None,:]
    up=(TF.linear(values,v['up.q'].float())*s)*v['up.scale'][None,:]
    phi=G.L.silu_lookup(gate)*up
    private=G.L.silu_lookup(TF.linear(x,v['private.gate'].float()))*TF.linear(x,v['private.up'].float())
    phi[:,v['private.ids'].long()]=private
    return phi


def from_raw(name,raw):
    if not name.startswith('private.'):return G.A.tensor_from_raw(name,raw)
    t=torch.from_numpy(np.frombuffer(raw,dtype='<u2').copy())
    return t if name=='private.ids' else t.view(torch.bfloat16).reshape(128,896)


def summarize(rows,name):
    cells=[r[name] for r in rows]
    return {'mean_state_relative_squared_error':float(np.mean([c['mean_state_relative_squared_error'] for c in cells])),
        'energy_normalized_squared_error':sum(c['squared_error'] for c in cells)/sum(c['reference_energy'] for c in cells),
        'maximum_layer_energy_normalized_squared_error':max(c['energy_normalized_squared_error'] for c in cells)}


def source_gates(rows,prefix):
    all_values=summarize(rows,prefix);reserve=summarize(rows,prefix+'_reserve')
    baseline=summarize(rows,'baseline32_reserve')
    return {prefix+'_all_mean_at_most85percent269':all_values['mean_state_relative_squared_error']<=.85*A.BASE_MEAN,
        prefix+'_all_maximum_layer_no_worse269':all_values['maximum_layer_energy_normalized_squared_error']<=A.BASE_MAX,
        prefix+'_reserve_mean_at_most85percent_baseline32':reserve['mean_state_relative_squared_error']<=.85*baseline['mean_state_relative_squared_error'],
        prefix+'_reserve_maximum_layer_no_worse32':reserve['maximum_layer_energy_normalized_squared_error']<=baseline['maximum_layer_energy_normalized_squared_error']}


def main():
    ap=argparse.ArgumentParser()
    for name in ('exe','quantizer','check','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();start=time.monotonic();stage='bindings';rows=[];gates={};result={};native_runs={}
    timing_check=args.check.with_name(args.check.stem+'.timing'+args.check.suffix)
    assert all(not p.exists() for p in (args.exe,args.quantizer,args.check,timing_check,args.out,args.out.with_suffix('.failure.json')))
    def partial():
        A.dump(args.out.with_suffix('.partial.json'),{'stage':stage,'layers':rows,'gates':gates,
            'native_runs':native_runs,'seconds':time.monotonic()-start})
    try:
        for path,sha in ((PRIOR,PRIOR_SHA),(PHASE,PHASE_SHA),(A.ASSAY,A.ASSAY_SHA),
                         (Q.D.CORE,Q.D.CORE_SHA),(Q.D.EXPORT,Q.D.EXPORT_SHA),(G.C.VECTORS,G.C.VECTORS_SHA)):
            assert P.digest(path)==sha,str(path)
        prior=json.loads(PRIOR.read_text(encoding='utf-8'));phase=json.loads(PHASE.read_text(encoding='utf-8'))
        assay=json.loads(A.ASSAY.read_text(encoding='utf-8'));export=json.loads(Q.D.EXPORT.read_text(encoding='utf-8'))
        assert prior['decision']=='stop_fixed128_output_aware_at_native_numeric_or_cost'
        assert all(v for k,v in prior['gates'].items() if k!='native_component_median_at_most10ms')
        assert all(phase['gates'].values())
        assert P.digest(Path(A.__file__))==prior['script_sha256']
        assert P.digest(Path(A.A.__file__))==assay['script_sha256']
        assert P.digest(FIXTURE)==prior['binary']['sha256']
        assert P.digest(Path(A.R.__file__))==export['script_sha256']
        for path,sha in export['helper_sha256'].items():assert P.digest(Path(path))==sha
        include=SOURCE.with_name('meth182_group64_ffn_cpu.c')
        assert P.digest(include)==prior['included_meth182_source_sha256']
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        vector_raw=G.C.VECTORS.read_bytes()
        assert struct.unpack_from('<8s4I',vector_raw)==(b'M125HX01',24,256,896,1280)
        states=np.frombuffer(vector_raw,dtype='<u2',offset=24).reshape(256,24,896)
        device=G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=10*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        reserve=torch.arange(1,256,2,device=device)
        expected=np.empty((256,24,896),dtype='<f4');references=np.empty((256,24,896),dtype='<u2')
        codes=np.empty((256,24,896),dtype='<i2');inverse=np.empty((256,24),dtype='<f4');scales=inverse.copy()
        baseline32=np.empty((256,24,896),dtype='<u2')
        stage='fixed6144_GPU_BF16_source_and_I16_reference'
        with torch.inference_mode(),safe_open(str(source),framework='pt',device='cpu') as donor,\
             safe_open(str(Q.D.CORE),framework='pt',device='cpu') as saved,FIXTURE.open('rb') as fixture:
            assert fixture.read(32)==struct.pack('<8s6I',b'M252PF01',24,896,4864,32,32,128)
            table=prior['segments'][0];assert fixture.tell()==table['offset']
            raw=fixture.read(table['bytes']);assert P.M17.sha(raw)==table['sha256']
            assert raw==A.raw_tensor(saved.get_tensor('ffn.silu_table'))
            G.L.TABLE=from_raw('silu_table',raw).to(device)
            for li in range(24):
                v={}
                for segment in prior['segments']:
                    if segment['layer']!=li:continue
                    assert fixture.tell()==segment['offset']
                    raw=fixture.read(segment['bytes']);assert P.M17.sha(raw)==segment['sha256']
                    v[segment['name']]=from_raw(segment['name'],raw)
                assert len(v)==15
                old={name:saved.get_tensor(f'ffn.{li}.{name}') for name in v}
                for name in v:
                    if not name.startswith('private.'):assert torch.equal(v[name],old[name])
                assert set(old['private.ids'].tolist()).issubset(v['private.ids'].tolist())
                w={name:donor.get_tensor(f'model.layers.{li}.mlp.{name}_proj.weight') for name in ('gate','up','down')}
                hashes={f'model.layers.{li}.mlp.{name}_proj.weight':P.M17.sha(A.raw_tensor(t)) for name,t in w.items()}
                assert hashes==prior['source_tensor_hashes'][li]['sha256']
                for name in ('gate','up'):
                    assert torch.equal(w[name][v['private.ids'].long()],v['private.'+name])
                vd={name:t.to(device) for name,t in v.items()};od={name:t.to(device) for name,t in old.items()}
                wd={name:t.to(device) for name,t in w.items()}
                x=torch.from_numpy(states[:,li].copy()).view(torch.bfloat16).to(device);xf=x.float()
                reference=TF.linear(TF.silu(TF.linear(x,wd['gate']))*TF.linear(x,wd['up']),wd['down'])
                original=G.readout(G.private_features(xf,od),od).bfloat16()
                unchanged=G.readout(G.private_features(xf,vd),vd).bfloat16()
                old_measure=A.A.measure(reference,original);assert old_measure==assay['layers'][li]['methods']['candidate_existing_fp32']
                assert A.A.measure(reference,unchanged)==prior['layers'][li]['private128']
                qx,inv,scale=input_codes(xf)
                value=G.readout(private_features(xf,vd,qx,scale),vd);assert torch.isfinite(value).all()
                measure=A.A.measure(reference,value.bfloat16())
                rows.append({'layer':li,'states':256,'baseline32_exact269':True,'unchanged128_exact271':True,
                    'gpu_i16':measure,'gpu_i16_reserve':A.subset_measure(reference,value.bfloat16(),reserve),
                    'baseline32_reserve':A.subset_measure(reference,original,reserve),
                    'drift_vs_unchanged128':A.A.measure(unchanged,value.bfloat16()),
                    'maximum_input_reconstruction_absolute_error':float((xf-qx.float()*torch.from_numpy(scale).to(device)[:,None]).abs().max())})
                expected[:,li]=value.cpu().numpy();references[:,li]=reference.cpu().view(torch.uint16).numpy()
                baseline32[:,li]=original.cpu().view(torch.uint16).numpy()
                codes[:,li]=qx.cpu().numpy();inverse[:,li]=inv;scales[:,li]=scale
                partial();print(json.dumps({'layer':li,'completed_layers':len(rows)}),flush=True)
                del v,old,w,vd,od,wd,x,xf,reference,original,unchanged,qx,value
                P.budget(start,device)
            assert not fixture.read(1)
        args.quantizer.parent.mkdir(parents=True,exist_ok=True)
        with args.quantizer.open('xb') as file:
            file.write(struct.pack('<8s3I',b'M274QX01',256,24,896))
            for array in (codes,inverse,scales):file.write(array.tobytes())
        assert args.quantizer.stat().st_size==11059220
        gates={'all_immutable_source_fixture_helpers_and_states_bound':True,
            'all361_segments_exact_and_old32_source_rows_retained':True,
            'all6144_GPU_baselines_exact269_and271_and_outputs_finite':True,**source_gates(rows,'gpu_i16')}
        result={'experiment':'METH-274-I16-input-Q8-shared-projections',
            'prior_sha256':PRIOR_SHA,'phase_sha256':PHASE_SHA,'source_sha256':P.M57.MODEL_SHA,
            'source_revision':P.M42.REV,'artifact_sha256':Q.D.CORE_SHA,'fixture_sha256':prior['binary']['sha256'],
            'vectors_sha256':G.C.VECTORS_SHA,'quantizer_sha256':P.digest(args.quantizer),
            'quantizer_bytes':args.quantizer.stat().st_size,'script_sha256':P.digest(Path(__file__)),
            'source_code_sha256':P.digest(SOURCE),'included_meth182_sha256':P.digest(include),
            'helper_sha256':export['helper_sha256'],'selection_script_sha256':P.digest(Path(A.__file__)),
            'layers':rows,'gates':gates,'GPU_summary':summarize(rows,'gpu_i16'),
            'GPU_scoring_runtime':P.budget(start,device),'native_runs':native_runs,
            'scope':'Consumed source components with changed shared input arithmetic; unchanged271 rows/bank/router. Not full model/fresh quality/useful n/RAM route-LUT-DRAM/accepted>=50 or family transfer;267/271/272 stops remain.'}
        if not all(gates.values()):
            result['decision']='stop_fixed_I16_shared_operator_at_GPU_source_fidelity'
        else:
            stage='native_compile_integer_qualification';partial();native_start=time.monotonic()
            torch.cuda.synchronize(device)
            command=['clang',*A.COMPILE_FLAGS,str(SOURCE),'-o',str(args.exe),'-lm','-lpsapi']
            compilation=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
            def native(mode,check):
                remaining=300-(time.monotonic()-native_start);assert remaining>0
                run=subprocess.run([str(args.exe),str(FIXTURE),str(G.C.VECTORS),str(check),str(args.quantizer),mode],
                    capture_output=True,text=True,check=True,timeout=min(180,remaining))
                value={'timing':json.loads(run.stdout),'qualification':json.loads(run.stderr)}
                native_runs[mode]=value;partial();return value
            qualification=native('qualify',args.check);record=qualification['qualification']
            assert record['mode']=='qualify' and record['qualified_states']==6144 and record['integer_dots_checked']==59768832
            assert record['quantizer_values_checked']==5505024 and all(record[k]==0 for k in
                ('quantizer_mismatches','parameter_mismatches','integer_dot_mismatches'))
            gates['all6144_CPU_GPU_quantizers_and_I64_integer_dots_exact']=True
            raw=args.check.read_bytes()
            assert len(raw)==20+256*24*896*4 and struct.unpack_from('<8s3I',raw)==(b'M274ALL1',256,24,896)
            actual=np.frombuffer(raw,dtype='<f4',offset=20).reshape(256,24,896);assert np.isfinite(actual).all()
            errors=np.linalg.norm(actual.astype(np.float64)-expected.astype(np.float64),axis=2)/np.linalg.norm(expected.astype(np.float64),axis=2)
            assert np.isfinite(errors).all()
            native_rows=[]
            for li in range(24):
                ref=torch.from_numpy(references[:,li].copy()).view(torch.bfloat16)
                val=torch.from_numpy(actual[:,li].copy()).bfloat16()
                rows[li]['cpu_i16']=A.A.measure(ref,val)
                rows[li]['cpu_i16_reserve']=A.subset_measure(ref,val,torch.arange(1,256,2))
                native_rows.extend({'token':t,'layer':li,'relative_l2':float(e)} for t,e in enumerate(errors[:,li]))
            gates.update(source_gates(rows,'cpu_i16'))
            gates['original384_native_numeric_limits']=float(np.median(errors[:16]))<=1e-4 and float(errors[:16].max())<=5e-4
            gates['all6144_native_numeric_same_limits']=float(np.median(errors))<=1e-4 and float(errors.max())<=5e-4
            result.update({'native_rows':native_rows,'CPU_summary':summarize(rows,'cpu_i16'),
                'native_numeric_summary':{'median384':float(np.median(errors[:16])),'max384':float(errors[:16].max()),
                    'median6144':float(np.median(errors)),'max6144':float(errors.max())},
                'compile_command':command,'compile_stderr':compilation.stderr,'executable_sha256':P.digest(args.exe),
                'native_all_output_sha256':P.digest(args.check)})
            if not all(gates.values()):
                result['decision']='stop_fixed_I16_operator_at_CPU_source_or_numeric_fidelity_no_timing'
            else:
                stage='three_original_native_timing_sweeps';partial();timed=native('timing',timing_check)['timing']
                assert all(timed[k]==v for k,v in (('threads',6),('tokens',256),('layers',24),
                    ('hidden',4864),('residual_rank',32),('private_units',128),('weight_bytes',337596452)))
                assert len(timed['pass_ms_per_token'])==3 and timed['peak_working_set_bytes']<=20*1024**3
                traw=timing_check.read_bytes()
                assert struct.unpack_from('<8s3I',traw)==(b'M274OUT1',16,24,896)
                assert traw[20:]==raw[20:20+16*24*896*4]
                gates['timing384_outputs_bitwise_equal_qualification']=True
                gates['native_component_median_at_most10ms']=statistics.median(timed['pass_ms_per_token'])<=10
                result['median_ms_per_token']=statistics.median(timed['pass_ms_per_token'])
                result['timing_output_sha256']=P.digest(timing_check)
                result['decision']='I16_shared_operator_component_pass_requires_full_archive_and_quality' if all(gates.values()) else 'stop_fixed_I16_shared_operator_at_native_cost'
            result['native_seconds']=time.monotonic()-native_start;assert result['native_seconds']<=300
        A.dump(args.out,result);print(json.dumps({k:result[k] for k in ('decision','gates','GPU_summary')}),flush=True)
    except BaseException as failure:
        partial();A.dump(args.out.with_suffix('.failure.json'),{'stage':stage,
            'error':type(failure).__name__+': '+str(failure),'seconds':time.monotonic()-start,
            'completed_layers':len(rows),'native_runs':native_runs,'result_before_failure':result,
            'subprocess_stdout':getattr(failure,'stdout',None),'subprocess_stderr':getattr(failure,'stderr',None)})
        raise


if __name__=='__main__':main()
