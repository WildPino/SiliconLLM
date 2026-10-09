"""Isolate source-informed group aggregation on retained common operands.

Use initial projected F32 masters in F64, not learned/ternary bank outputs.
No donor/full learner execution, recurrence, fitting updates or generation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))


def bind(a):
    native_path=DOC/'chatbot_hybrid_native_result_repair3_20261008.json'
    native=json.loads(native_path.read_bytes()); terminal_path=native_path.with_suffix('.terminal.json')
    terminal=json.loads(terminal_path.read_bytes())
    assert terminal['exit_code']==0 and terminal['resource_gates'] and terminal['result_sha256']==sha(native_path)
    trace=next(v for v in native['native_outputs'] if Path(v['path']).name=='trace.bin')
    trace_path=Path(trace['path']); query_path=trace_path.parent/'queries.json'
    assert trace_path.stat().st_size==trace['bytes']==39588480 and sha(trace_path)==trace['sha256']
    query_identity=next(v for v in terminal['output_files'] if Path(v['path']).resolve()==query_path.resolve())
    assert query_path.stat().st_size==query_identity['bytes'] and sha(query_path)==query_identity['sha256']
    queries=json.loads(query_path.read_bytes())
    assert len(queries)==6 and sum(len(r['input_ids']) for r in queries)==261
    manifest_path=ROOT/'results/native_expert_scaling/chatbot_hybrid_pilot_repair1_20261008/initial_manifest.json'
    pilot_path=DOC/'chatbot_hybrid_pilot_result_repair1_20261008.json'
    pilot_terminal_path=pilot_path.with_suffix('.terminal.json')
    pilot_terminal=json.loads(pilot_terminal_path.read_bytes())
    assert pilot_terminal['exit_code']==0 and pilot_terminal['resource_gates'] and pilot_terminal['result_sha256']==sha(pilot_path)
    manifest_identity=next(v for v in pilot_terminal['output_files'] if Path(v['path']).resolve()==manifest_path.resolve())
    assert manifest_path.stat().st_size==manifest_identity['bytes'] and sha(manifest_path)==manifest_identity['sha256']
    manifest=json.loads(manifest_path.read_bytes())
    layers=[next(v for v in manifest['components'] if v['component']==f'layers.{i}') for i in range(12)]
    assert all(Path(v['path']).stat().st_size==v['bytes'] and sha(v['path'])==v['sha256'] for v in layers)
    files=[Path(__file__),B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',Path(sys.executable),
           DOC/'CHATBOT_HYBRID_GROUP_SUM_PROTOCOL_20261009.md',native_path,terminal_path,trace_path,query_path,manifest_path,pilot_path,pilot_terminal_path]
    files += [Path(v['path']) for v in layers]
    files += [SITE/'torch'/v for v in ('serialization.py','nn/functional.py')]
    write(a.out,dict(schema='HYBRID_GROUP_SUM_BINDING_V1',python=str(Path(sys.executable).resolve()),
         worker_path=str(Path(__file__).resolve()),trace=str(trace_path.resolve()),queries=str(query_path.resolve()),
         layers=layers,relative_RMS_max=.01,limits=dict(seconds=300,OS_bytes=4<<30,GPU_allocated_bytes=2<<30,
                 GPU_reserved_bytes=3<<30,output_bytes=128<<20),
         runtime_binding_scope='Actual original C operands and source-informed initial projected masters, selected runtime files and pinned versions. F64 function/energy arithmetic, not exact real elementary functions or full DLL-tree certification.',
         inputs=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start=time.monotonic(); assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes()); assert b['schema']=='HYBRID_GROUP_SUM_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve() and sys.version_info[:3]==(3,12,10)
    a.directory.mkdir(exist_ok=False); complete=[]; phase='imports'
    try:
        import numpy as np
        import psutil
        import torch
        from torch.nn import functional as F
        assert (np.__version__,psutil.__version__,torch.__version__)==('2.4.6','7.2.2','2.6.0+cu124')
        proc=psutil.Process(); proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6); torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        def guard():
            v=b['limits']
            assert time.monotonic()-start<=v['seconds']-30,'worker deadline reserve'
            assert proc.memory_info().peak_wset<=v['OS_bytes'],'OS cap'
            assert torch.cuda.max_memory_allocated()<=v['GPU_allocated_bytes'],'GPU allocated cap'
            assert torch.cuda.max_memory_reserved()<=v['GPU_reserved_bytes'],'GPU reserved cap'
            assert not proc.children(recursive=True),'worker subprocess'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=v['output_bytes'],'output cap'
        dtype=np.dtype([(n,'<f4',512) for n in ('input','core_input','core_output','ff_input')]+[
              ('scores','<f4',72),('ids','<u4',8),('mass','<f4',8),('ff_output','<f4',512),('output','<f4',512)])
        assert dtype.itemsize==12640
        trace=np.fromfile(b['trace'],dtype=dtype); assert len(trace)==3132
        queries=json.loads(Path(b['queries']).read_bytes()); mask=[]; case_ids=[]
        for rec in queries:
            n=len(rec['input_ids']); mask.extend([rec['split']=='FIT']*n); case_ids.extend([rec['id']]*n)
        fit=np.array(mask,dtype=bool); assert fit.shape==(261,) and fit.sum()==170 and (~fit).sum()==91
        reports=[]; saved=[]
        with torch.no_grad():
            for site,descriptor in enumerate(b['layers']):
                phase=f'layer{site}'; guard()
                initial=torch.load(descriptor['path'],map_location='cpu',weights_only=True)
                def weight(name,shape):
                    v=initial['banks.'+name]
                    assert tuple(v.shape)==shape and v.dtype==torch.float32 and torch.isfinite(v).all().item()
                    return v.to(device='cuda',dtype=torch.float64)
                gate=weight('gate',(72,128,512)); up=weight('up',(72,128,512)); down=weight('down',(72,512,128))
                router=weight('router.weight',(72,512)); bias=weight('router.bias',(72,))
                x=torch.tensor(trace['ff_input'][site::12].copy(),device='cuda',dtype=torch.float64)
                assert torch.isfinite(x).all().item()
                g=F.linear(x,gate.reshape(9216,512)).reshape(261,72,128)
                u=F.linear(x,up.reshape(9216,512)).reshape(261,72,128)
                atoms=torch.einsum('teh,edh->ted',F.silu(g)*u,down)
                assert atoms.shape==(261,72,512) and torch.isfinite(atoms).all().item()
                scores=F.linear(x,router,bias); ids=torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:8]
                mass=torch.softmax(torch.gather(scores,1,ids),-1)
                selected=torch.gather(atoms,1,ids[:,:,None].expand(-1,-1,512))
                dense=atoms.sum(1); selected_sum=selected.sum(1); mixture=(selected*mass[:,:,None]).sum(1)
                responses=torch.stack((dense,selected_sum,mixture),dim=1).cpu().numpy()
                reference,summed,mixed=[responses[:,i] for i in range(3)]
                energy=np.square(reference).sum(-1); mixed_energy=np.square(mixed).sum(-1)
                assert (energy>0).all() and (mixed_energy>0).all() and np.isfinite(responses).all()
                assert np.isfinite(energy).all() and np.isfinite(mixed_energy).all()
                dot=(reference*mixed).sum(-1)
                assert np.isfinite(dot).all()
                scalar=float(dot[fit].sum()/mixed_energy[fit].sum())
                oracle=dot/mixed_energy
                def rms(value): return np.sqrt(np.square(value-reference).sum(-1)/energy)
                errors={ 'selected_sum':rms(summed),'normalized_mixture':rms(mixed),
                         'mixture_times72':rms(mixed*72),'FIT_scalar_mixture':rms(mixed*scalar),
                         'per_operand_oracle_scalar':rms(mixed*oracle[:,None]) }
                assert np.isfinite(scalar) and np.isfinite(oracle).all() and all(np.isfinite(v).all() for v in errors.values())
                def distribution(v,chosen):
                    z=v[chosen]
                    return dict(rows=len(z),minimum=float(z.min()),median=float(np.median(z)),
                                maximum=float(z.max()),rows_over_fixed1percent=int((z>b['relative_RMS_max']).sum()))
                # Covariance identity quantifies cancellation/shared contribution.
                covariances=[]; covariance_checks=[]; atom_energies=[]
                for chosen in (fit,~fit):
                    at=atoms[torch.tensor(chosen,device='cuda')]
                    gram=torch.einsum('ted,tfd->ef',at,at).cpu().numpy()
                    total_energy=float(energy[chosen].sum()); derived=float(gram.sum())
                    relative=abs(derived-total_energy)/total_energy
                    assert relative<=1e-10,'group covariance energy identity'
                    covariances.append(gram); covariance_checks.append(relative)
                    atom_energies.append(float(np.trace(gram)))
                path=a.directory/(f'layer_{site:02d}.responses.f64')
                with path.open('xb') as f: f.write(responses.astype('<f8',copy=False).tobytes()); f.flush(); os.fsync(f.fileno())
                covariance_path=a.directory/(f'layer_{site:02d}.covariance.f64')
                with covariance_path.open('xb') as f: f.write(np.array(covariances,dtype='<f8').tobytes()); f.flush(); os.fsync(f.fileno())
                row=dict(layer=site,FIT_scalar=scalar,errors={k:{'FIT':distribution(v,fit),'DEV':distribution(v,~fit)} for k,v in errors.items()},
                   covariance_energy_relative_error=covariance_checks,
                   dense_energy_over_atom_energy=[float(energy[m].sum()/ae) for m,ae in zip((fit,~fit),atom_energies,strict=True)],
                   response_names=['all72_sum','selected8_sum','selected8_normalized_mixture'],
                   response_shape=[261,3,512],response=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path)),
                   covariance=dict(path=str(covariance_path.resolve()),shape=[2,72,72],bytes=covariance_path.stat().st_size,sha256=sha(covariance_path)),
                   selected_ids=ids.cpu().tolist(),selected_mass=mass.cpu().tolist(),
                   cases=[dict(case=case_ids[i],row=i,split='FIT' if fit[i] else 'DEV',reference_energy=float(energy[i]),
                     mixture_energy=float(mixed_energy[i]),oracle_scalar=float(oracle[i]),**{k:float(v[i]) for k,v in errors.items()}) for i in range(261)])
                write(a.directory/(f'layer_{site:02d}.json'),row); reports.append(row); complete.append(site)
                saved.extend([row['response'],row['covariance']]); torch.cuda.synchronize()
                print(json.dumps(dict(layer=site,seconds=time.monotonic()-start,scalar=scalar,DEV_oracle=row['errors']['per_operand_oracle_scalar']['DEV'])),flush=True)
                del initial,gate,up,down,router,bias,x,g,u,atoms,scores,ids,mass,selected,dense,selected_sum,mixture,at
        guard(); assert len(reports)==12
        oracle_fail=sum(r['errors']['per_operand_oracle_scalar']['DEV']['rows_over_fixed1percent'] for r in reports)
        fitted_fail=sum(r['errors']['FIT_scalar_mixture']['DEV']['rows_over_fixed1percent'] for r in reports)
        decision='AMPLITUDE_ONLY_REPAIR_INSUFFICIENT_ON_TESTED_OPERANDS' if oracle_fail else (
            'FIXED_SCALAR_CANDIDATE_ON_TESTED_OPERANDS' if not fitted_fail else 'SCALAR_REQUIRES_INPUT_DEPENDENCE_ON_TESTED_OPERANDS')
        write(a.out,dict(schema='HYBRID_GROUP_SUM_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),decision=decision,
            layers=reports,completed_layers=12,operand_rows=3132,FIT_rows=2040,DEV_rows=1092,
            DEV_oracle_scalar_rows_over_fixed1percent=oracle_fail,DEV_FIT_scalar_rows_over_fixed1percent=fitted_fail,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
            source_calls=0,whole_student_calls=0,optimizer_updates=0,native_C_calls=0,saved_outputs=saved,
            quality_admission=False,scope='F64 evaluation of source-informed INITIAL projected FFN groups on six retained old-student C operands. Pair members evaluated at the same operand; not actual source two-block composition or source history. No ternary/core/width variable, no current-boundary286 parity, fresh quality/rate/n/family admission.'))
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(fault=repr(error),phase=phase,completed_layers=complete,elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bind',action='store_true');p.add_argument('--binding',type=Path)
    p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();bind(a) if a.bind else worker(a)
