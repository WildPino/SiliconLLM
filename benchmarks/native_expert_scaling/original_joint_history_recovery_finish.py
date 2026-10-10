"""Missing-only assessment completion of durable matched actual51 states.

The first family exhausted its completion reserve after all48 optimizer updates,
all six native DEV assessments, A tasks, and sixteen final B auxiliary cases.
Adopt these outputs. Restore B51 for eight unsaved auxiliary cases only, then
assess B51 tasks with the same native binary. Never construct an optimizer.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path.insert(0,str(B))
from original_packed_capacity import SITE,sha,extent,write,raw
from original_falcon_whole_recovery import check_inputs,memory_reader
sys.path.insert(0,str(SITE))
PARENT=ROOT/'results/native_expert_scaling/original_joint_history_recovery_20261009'


def bind(a):
    parent_binding=DOC/'original_joint_history_recovery_binding_20261009.json'
    pb=json.loads(parent_binding.read_bytes());check_inputs(pb)
    fault_path=PARENT/'first_fault.json';fault=json.loads(fault_path.read_bytes())
    failure=DOC/'original_joint_history_recovery_result_20261009.launcher_failure.json'
    fail=json.loads(failure.read_bytes())
    assert fault['stage']=='final_auxiliary_DEV' and fault['arm']=='B' and fault['counter']==fault['durable']==51
    assert not fault['optimizer_partial_possible'] and fail['exit_code']==1
    assert len(fault['completed_updates'])==24 and len(fault['completed_arms'])==1
    A=json.loads((PARENT/'A.arm_result.json').read_bytes());assert A==fault['completed_arms'][0] and A['new_updates']==24
    milestones=[json.loads((PARENT/f'B.actual{c:03d}.milestone.json').read_bytes()) for c in (33,39,51)]
    assert fault['last_snapshot']==milestones[-1]['checkpoint']
    dev=[r for r in pb['records'] if r['split']=='DEV'];adopted=[];missing=[]
    for rec in dev:
        path=PARENT/('B.final.'+rec['id']+'.aux.json')
        if path.exists():
            row=json.loads(path.read_bytes());assert row['id']==rec['id'];adopted.append(extent(path))
        else:missing.append(rec['id'])
    assert len(adopted)==16 and len(missing)==8
    sealed=[extent(p) for p in sorted(PARENT.iterdir()) if p.is_file()]
    files=[Path(__file__),DOC/'ORIGINAL_JOINT_HISTORY_RECOVERY_FINISH_PROTOCOL_20261010.md',parent_binding,
        failure,DOC/'original_joint_history_recovery_result_20261009.worker.log']
    inputs={i['path']:i for i in pb['inputs']+sealed}
    for path in files:inputs[str(path.resolve())]=extent(path)
    result=dict(schema='ORIGINAL_JOINT_HISTORY_RECOVERY_FINISH_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),parent_binding=extent(parent_binding),parent_failure=extent(failure),
        first_fault=extent(fault_path),parent_directory=str(PARENT),parent_outputs=sealed,
        adopted_B_auxiliary=adopted,missing_B_auxiliary_ids=missing,source_state=milestones[-1]['checkpoint'],
        packed=milestones[-1]['packed'],A= A,records=pb['records'],
        limits=dict(seconds=1200,OS_bytes=12<<30,GPU_allocated_bytes=5<<30,GPU_reserved_bytes=6<<30,
            output_bytes=1<<30,log_bytes=8<<20,child_seconds=100,stream_seconds=300),inputs=list(inputs.values()),
        scope='Zero optimizer/source/native DEV replay. Adopt all48 updates/six native milestones/A tasks/16 durable B auxiliary cases. '
            'Eight new B51 target-only auxiliary histories plus B51 native16 tasks/own-answer followup. '
            'The first timed-out17th auxiliary case has no durable intermediate state and is completed as an unsaved case. '
            'Original scientific gates/source capture/numerical/resource gaps unchanged. Parent completion-reserve FAIL retained.')
    write(a.out,result);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(inputs),parent_files=len(sealed))),flush=True)


def worker(a):
    import numpy as np
    import psutil
    import torch
    from original_joint_history_learner import JointHistoryLearner
    from original_joint_native_assessment import tasks
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes())
    pb=json.loads(Path(b['parent_binding']['path']).read_bytes());a.directory.mkdir(exist_ok=False)
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats();reader=memory_reader();child_peak=0;stage='restore_B51';completed=[]
    def observe(peak):
        nonlocal child_peak
        child_peak=max(child_peak,peak)
    def guard():
        lim=b['limits'];assert time.monotonic()-start<=lim['seconds'],'completion deadline'
        assert proc.memory_info().peak_wset+child_peak<=lim['OS_bytes'],'completion OS cap'
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'completion GPU cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes'],'completion output cap'
    def event(**values):
        guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**values)),flush=True)
    try:
        state=torch.load(b['source_state']['path'],map_location='cpu',weights_only=True,mmap=True)
        assert state['schema']=='ORIGINAL_JOINT_HISTORY_STATE_V1' and state['arm']=='B' and state['updates']==51
        assert state['new_updates']==24 and not state['optimizer_partial_possible']
        assert state['completed_new_FIT_ids']==pb['FIT_order'] and state['binding_sha256']==b['parent_binding']['sha256']
        model=JointHistoryLearner(device='cuda');model.load_state_dict(state['model'])
        names=list(model.named_parameters());assert len(names)==92 and sum(p.numel() for _,p in names)==721008128
        for name,p in names:
            assert torch.equal(p.detach().cpu().contiguous().view(torch.int32),state['model'][name].contiguous().view(torch.int32)),name
            assert torch.isfinite(p).all(),name
        torch.set_rng_state(state['CPU_rng']);torch.cuda.set_rng_state_all(state['CUDA_rng'])
        assert torch.equal(torch.get_rng_state(),state['CPU_rng']) and all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),state['CUDA_rng']))
        del state,p;gc.collect();torch.cuda.empty_cache();model.use_checkpoint=False
        identities={n:(id(p),p._version) for n,p in names};event(all92_restored_masters_RNG_exact=True,optimizer_constructed=False)
        rows=[json.loads(Path(i['path']).read_bytes()) for i in b['adopted_B_auxiliary']]
        records={r['id']:r for r in pb['records']};stage='missing_B_auxiliary'
        with torch.no_grad():
            for identifier in b['missing_B_auxiliary_ids']:
                rec=records[identifier];ids=torch.tensor([rec['student_input_ids']],device='cuda');targets=[]
                for item in pb['boundary_targets'][identifier]:
                    target=np.fromfile(item['path'],dtype='<f4').reshape(1,len(rec['student_input_ids']),256)
                    assert np.isfinite(target).all();targets.append(torch.from_numpy(target).cuda())
                logits,aux,terms=model(ids,torch.empty(0,dtype=torch.long,device='cuda'),targets);torch.cuda.synchronize()
                assert logits.shape==(1,0,65537) and torch.isfinite(aux)
                summaries=[]
                for term in terms:
                    item={k:(float(v.detach()) if isinstance(v,torch.Tensor) else v) for k,v in term.items()}
                    assert all(math.isfinite(item[k]) for k in ('total','mean','centered','centered_relative'))
                    assert abs(item['total']-item['mean']-item['centered'])<=1e-4*max(item['total'],1e-24);summaries.append(item)
                row=dict(id=identifier,domain=rec['domain'],history=len(rec['student_input_ids']),auxiliary=float(aux),sites=summaries)
                write(a.directory/('B.final.'+identifier+'.aux.json'),row);rows.append(row);completed.append(identifier)
                del ids,targets,target,logits,aux,terms;gc.collect();torch.cuda.empty_cache();event(id=identifier,new_cases=len(completed))
        assert {n:(id(p),p._version) for n,p in names}==identities
        rows=[next(r for r in rows if r['id']==rec['id']) for rec in pb['records'] if rec['split']=='DEV']
        def mean(values):return dict(cases=len(values),**{k:float(np.mean([np.mean([s[k] for s in r['sites']]) for r in values])) for k in ('total','mean','centered','centered_relative')})
        auxiliary=dict(records=rows,aggregate=dict(**mean(rows),domains={d:mean([r for r in rows if r['domain']==d]) for d in sorted({r['domain'] for r in rows})}),
            parameter_ids_versions_unchanged=True,heads_computed=0,seconds=time.monotonic()-start,
            adopted_cases=16,new_cases=8,scope='Durable16 adopted; eight previously unsaved histories evaluated from immutable B51. Timing excludes adopted cases.')
        write(a.directory/'B.final.auxiliary_DEV.json',auxiliary)
        del model,names;gc.collect();torch.cuda.empty_cache();stage='missing_B_tasks'
        exe=a.directory/'original_engine_chat.exe';raw(exe,Path(pb['native']['path']).read_bytes());assert sha(exe)==pb['native']['sha256']
        task_binding=dict(pb,limits=dict(pb['limits'],child_seconds=b['limits']['child_seconds'],stream_seconds=b['limits']['stream_seconds']))
        task=tasks(task_binding,a.directory,exe,b['packed']['path'],'B.final',reader,observe,guard,event)
        fault=json.loads(Path(b['first_fault']['path']).read_bytes());milestones=[json.loads((PARENT/f'B.actual{c:03d}.milestone.json').read_bytes()) for c in (33,39,51)]
        dev=milestones[-1]['native']['aggregate'];base=pb['before_native'];g=pb['gates']
        quality=dict(DEV_KL=dev['case_KL']<=g['native_DEV_KL'],DEV_disagreement=dev['case_disagreement']<=g['native_DEV_disagreement'],
            domain_KL=all(x['case_KL']<=g['domain_KL'] for x in dev['domains'].values()),domain_disagreement=all(x['case_disagreement']<=g['domain_disagreement'] for x in dev['domains'].values()),
            relative_DEV_KL=dev['case_KL']<=g['relative_DEV_KL']*base['case_KL'],relative_DEV_disagreement=dev['case_disagreement']<=g['relative_DEV_disagreement']*base['case_disagreement'])
        arm=dict(name='B',auxiliary_weight=1.,initial_counter=27,final_counter=51,durable_counter=51,new_updates=24,complete_onepass=True,stop_reason=None,
            updates=fault['completed_updates'],milestones=milestones,final_checkpoint=b['source_state'],final_packed=b['packed'],auxiliary_DEV=auxiliary,tasks=task,quality_gates=quality)
        write(a.directory/'B.arm_result.json',arm);A=b['A'];aDEV=A['milestones'][-1]['native']['aggregate'];bDEV=dev
        preference=dict(both_complete24=True,native_KL=bDEV['case_KL']<=g['B_relative_to_A_KL']*aDEV['case_KL'],
            centered_auxiliary=auxiliary['aggregate']['centered_relative']<=g['B_relative_to_A_centered']*A['auxiliary_DEV']['aggregate']['centered_relative'],
            disagreement=bDEV['case_disagreement']<=aDEV['case_disagreement']+g['B_disagreement_add'],
            domains=all(bDEV['domains'][d]['case_KL']<=g['B_domain_KL_multiple']*x['case_KL'] and bDEV['domains'][d]['case_disagreement']<=x['case_disagreement']+g['B_domain_disagreement_add'] for d,x in aDEV['domains'].items()),
            behavioral_support=task['correct']>=max(g['B_behavioral_min'],A['tasks']['correct']+2))
        stage='complete';guard()
        result=dict(schema='ORIGINAL_JOINT_HISTORY_RECOVERY_FINISH_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,parent_binding_sha256=b['parent_binding']['sha256'],
            arms=[A,arm],before_auxiliary_DEV=json.loads((PARENT/'initial27.auxiliary_DEV.json').read_bytes()),B_preference_gates=preference,B_preferred=all(preference.values()),
            decision='JOINT_HISTORY_B_SUPPORTED' if all(preference.values()) else 'JOINT_HISTORY_B_NOT_SUPPORTED',new_optimizer_updates=0,parent_optimizer_updates=48,
            source_calls=0,reserved_queries=0,GPU_source_calls=0,candidate_auxiliary_calls=8,native_tasks=16,
            adopted_B_auxiliary_cases=16,parent_completion_reserve_gate=False,parent_first_fault=b['first_fault'],parent_launcher_failure=b['parent_failure'],
            initial_92_states_RNG_exports_GPU_heads_routes_exact=True,quality_admission=False,speed_admission=False,useful_large_n_admission=False,
            historical_GPU_native_numerical_gate='FAIL retained',physical_DRAM_bytes=None,inherited_source_capture_resource_gate=False,
            children=[task['receipt']],max_direct_child_OS_peak=child_peak,worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-start,scope=b['scope'])
        write(a.out,result);event(decision=result['decision'],B_task_correct=task['correct'])
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),completed_missing_B_auxiliary=completed,
            optimizer_updates=0,elapsed_seconds=time.monotonic()-start));raise


def launch(a):
    import psutil
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['schema']=='ORIGINAL_JOINT_HISTORY_RECOVERY_FINISH_BINDING_V1' and Path(sys.executable).resolve()==Path(b['python']).resolve()
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','native','original_engine','clang','lld','packed_original')),('overlap',p.pid,name)
    log=a.out.with_suffix('.worker.log');terminal=a.out.with_suffix('.terminal.json');failure=a.out.with_suffix('.launcher_failure.json')
    assert not a.directory.exists() and not any(p.exists() for p in (a.out,log,terminal,failure))
    process=None;peak=0;reader=memory_reader();seen={};record=dict(freeze=a.freeze,binding_sha256=a.binding_sha,launcher_pid=proc.pid)
    try:
        check_inputs(b)
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command=[b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker','--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
        record['command']=command
        def guard():
            nonlocal peak
            peak=max(peak,reader(process));assert time.monotonic()-start<=b['limits']['seconds'],'family deadline'
            assert peak+proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'family OS cap';assert log.stat().st_size<=b['limits']['log_bytes'],'log cap'
        with log.open('xb') as f:
            process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,creationflags=8)
            record.update(worker_pid=process.pid,worker_creation_time=psutil.Process(process.pid).create_time());offset=0
            while process.poll() is None:
                guard()
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        try:name=child.name().lower();created=child.create_time();exe=Path(child.exe().removeprefix('\\\\?\\')).resolve()
                        except psutil.NoSuchProcess:continue
                        seen[(child.pid,created)]=child
                        assert (name=='original_engine_chat.exe' and exe==a.directory.resolve()/name) or (name=='conhost.exe' and exe==Path('C:/Windows/System32/conhost.exe').resolve()),('unexpected child',name,exe)
                except psutil.NoSuchProcess:pass
                with log.open('rb') as stream:
                    stream.seek(offset);chunk=stream.read();end=chunk.rfind(b'\n')+1
                    if end:print(chunk[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.1)
        guard();record.update(exit_code=process.returncode,worker_OS_peak_through_exit=peak)
        assert process.returncode==0,log.read_text(errors='replace')[-4000:]
        check_inputs(b);r=json.loads(a.out.read_bytes());assert r['new_optimizer_updates']==0 and r['parent_optimizer_updates']==48
        assert all(child['exit_code']==0 for child in r['children'])
        assert peak+r['max_direct_child_OS_peak']+proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()];assert sum(i['bytes'] for i in outputs)<=b['limits']['output_bytes']
        assert time.monotonic()-start<=b['limits']['seconds'],'final family deadline'
        record.update(elapsed_seconds=time.monotonic()-start,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,result_sha256=sha(a.out),output_files=outputs,decision=r['decision'],resource_gates=True)
        write(terminal,record);print(json.dumps(dict(terminal=str(terminal),exit_code=0,seconds=record['elapsed_seconds'],decision=r['decision'])),flush=True)
    except BaseException as error:
        for (pid,created),child in reversed(list(seen.items())):
            try:
                if child.is_running() and child.create_time()==created:child.kill()
            except psutil.NoSuchProcess:pass
        if process is not None:
            if process.poll() is None:process.kill();process.wait()
            peak=max(peak,reader(process));record.update(exit_code=process.returncode,worker_OS_peak_through_exit=peak)
        record.update(error=repr(error),elapsed_seconds=time.monotonic()-start,launcher_OS_peak_snapshot=proc.memory_info().peak_wset);write(failure,record);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true');m.add_argument('--launch',action='store_true');m.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.bind:bind(args)
    elif args.launch:launch(args)
    else:worker(args)
