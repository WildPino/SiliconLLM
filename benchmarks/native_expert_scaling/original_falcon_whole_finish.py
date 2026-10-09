"""Finish missing native evaluation only; adopts actual24 updates/Adam25/export."""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_falcon_whole_recovery import SITE,extent,sha,write,memory_reader,launch
sys.path.insert(0,str(SITE))


def bind(a):
    old_b=DOC/'original_falcon_whole_recovery_binding_20261009.json';b=json.loads(old_b.read_bytes())
    failure=DOC/'original_falcon_whole_recovery_result_20261009.launcher_failure.json';f=json.loads(failure.read_bytes())
    assert f['exit_code']==1 and f['binding_sha256']==sha(old_b)
    old_dir=ROOT/'results/native_expert_scaling/original_falcon_whole_recovery_20261009'
    fault=old_dir/'first_fault.json';v=json.loads(fault.read_bytes())
    assert v['error']=="KeyError('path')" and v['actual_counter']==v['durable_counter']==25
    assert len(v['updates'])==24 and len(v['before'])==48 and len(v['after'])==0 and len(v['children'])==47
    assert [u['counter'] for u in v['updates']]==list(range(2,26)) and [u['id'] for u in v['updates']]==b['FIT_order']
    assert not v['optimizer_partial_possible'] and extent(v['last_snapshot']['path'])==v['last_snapshot']
    frozen=DOC/'original_falcon_whole_recovery_frozen_20261009.py.txt';aliases=[]
    for i in b['inputs']:
        p=Path(i['path'])
        if p.resolve()==(B/'original_falcon_whole_recovery.py').resolve():
            aliases.append(dict(original=i,validation_path=str(frozen.resolve())));p=frozen
        assert p.stat().st_size==i['bytes'] and sha(p)==i['sha256'],p
    files=[Path(__file__),B/'original_falcon_whole_recovery.py',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
        DOC/'ORIGINAL_FALCON_WHOLE_FINISH_PROTOCOL_20261009.md',old_b,failure,fault,frozen,
        DOC/'original_falcon_whole_recovery_result_20261009.worker.log',Path(v['last_snapshot']['path']),
        old_dir/'candidate_13.pt',old_dir/'candidate_25.packed',old_dir/'expected_final.witness',Path(sys.executable)]
    files += [Path(i['path']) if Path(i['path']).resolve()!=(B/'original_falcon_whole_recovery.py').resolve() else frozen for i in b['inputs']]
    for row in v['before']:
        for key in ('logits','routes'):
            i=row[key];assert extent(i['path'])==i;files.append(Path(i['path']))
    files += list(old_dir.glob('before_*.metrics.json'))+list(old_dir.glob('before_*.receipt.json'))+list(old_dir.glob('update_*.json'))
    packed=extent(old_dir/'candidate_25.packed');assert packed['bytes']==507505920
    write(a.out,dict(schema='ORIGINAL_FALCON_WHOLE_RECOVERY_FINISH_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),corpus=b['corpus'],native=b['native'],old_binding=str(old_b.resolve()),
        old_fault=str(fault.resolve()),old_failure=str(failure.resolve()),old_log=str(files[9].resolve()),
        packed=packed,expected_witness=extent(old_dir/'expected_final.witness'),gates=b['gates'],aliases=aliases,
        limits=dict(seconds=900,reserve_seconds=60,OS_bytes=4<<30,GPU_allocated_bytes=0,GPU_reserved_bytes=0,output_bytes=8<<30),
        inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in files)],
        runtime_binding_scope='CPU-only missing48 native predictions;actual24 updates/Adam25/initial48/export adopted and hash-checked,old runner validated by frozen exact-byte alias;historical GPU/launcher peak gaps retained.'))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files),new_updates=0)),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='ORIGINAL_FALCON_WHOLE_RECOVERY_FINISH_BINDING_V1'
    a.directory.mkdir(exist_ok=False)
    import numpy as np
    import psutil
    assert np.__version__=='2.4.6' and psutil.__version__=='7.2.2'
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));reader=memory_reader();child_peak=0;children=[];after=[]
    fault=json.loads(Path(b['old_fault']).read_bytes());failure=json.loads(Path(b['old_failure']).read_bytes())
    before=fault['before'];updates=fault['updates'];expected=Path(b['expected_witness']['path']).read_bytes();assert len(expected)==18432*4
    records=sorted(json.loads(Path(b['corpus']).read_bytes())['records'],key=lambda r:r['id'])
    def guard():
        assert time.monotonic()-start<=b['limits']['seconds']-b['limits']['reserve_seconds'],'finish reserve'
        assert proc.memory_info().peak_wset+child_peak<=b['limits']['OS_bytes'],'finish worker/child OS cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=b['limits']['output_bytes'],'finish output cap'
    try:
        for ordinal,rec in enumerate(records):
            guard();n=len(rec['student_input_ids']);prefix=rec['id']
            query=a.directory/(prefix+'.u32')
            with query.open('xb') as f:f.write(struct.pack('<I',n)+np.asarray(rec['student_input_ids'],dtype='<u4').tobytes())
            files=[a.directory/(prefix+s) for s in ('.f32','.routes','.witness','.native.json')]
            argv=[str(p) for p in [b['native'],b['packed']['path'],query,*files]];t=time.monotonic();peak=0
            with (a.directory/(prefix+'.native.log')).open('xb') as stream:
                child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,creationflags=8);created=psutil.Process(child.pid).create_time()
                try:
                    while child.poll() is None:
                        peak=max(peak,reader(child));child_peak=max(child_peak,peak);guard();time.sleep(.05)
                    peak=max(peak,reader(child));child_peak=max(child_peak,peak)
                except BaseException:
                    if child.poll() is None:child.kill();child.wait()
                    peak=max(peak,reader(child));child_peak=max(child_peak,peak);raise
                finally:
                    receipt=dict(command=argv,pid=child.pid,creation_time=created,exit_code=child.returncode,
                        held_OS_peak=peak,seconds=time.monotonic()-t,prefix=prefix)
                    children.append(receipt);write(a.directory/(prefix+'.receipt.json'),receipt)
            assert child.returncode==0 and files[2].read_bytes()==expected
            assert files[0].stat().st_size==n*65537*4 and files[1].stat().st_size==n*6*64
            logits=np.memmap(files[0],dtype='<f4',mode='r',shape=(n,65537));labels=len(rec['positions'])
            for first in range(0,n,128):assert np.isfinite(logits[first:first+128]).all()
            teacher=np.memmap(rec['logits']['path'],dtype='<u2',mode='r',shape=(labels,65537));losses=[];uniform=[];dis=0
            for j,pos in enumerate(rec['positions']):
                qlog=(teacher[j].astype('<u4')<<16).view('<f4').astype('f8');qlog-=qlog.max();qlog-=np.log(np.exp(qlog).sum())
                q=np.exp(qlog);z=logits[pos].astype('f8');z-=z.max();z-=np.log(np.exp(z).sum())
                losses.append(float(np.sum(q*(qlog-z))));uniform.append(math.log(65537)+float(np.sum(q*qlog)))
                dis+=int(np.argmax(logits[pos])!=np.argmax(qlog))
            routes=np.fromfile(files[1],dtype=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))])).reshape(n,6)
            ids=routes['ids'];mass=routes['mass'];assert ids.min()>=0 and ids.max()<1152
            assert (np.diff(np.sort(ids,axis=-1),axis=-1)>0).all() and np.isfinite(mass).all() and (mass>=0).all()
            defect=float(np.max(np.abs(mass.astype('f8').sum(-1)-1)));assert defect<=b['gates']['mass_defect']
            measured=dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,labels=labels,KL=float(np.mean(losses)),
                KL_per_label=losses,uniform_KL=float(np.mean(uniform)),disagreement=dis,disagreement_rate=dis/labels,
                unions=[int(np.unique(ids[:,site]).size) for site in range(6)],mass_defect=defect,adopted=False,
                packed_path=b['packed']['path'],logits=extent(files[0]),routes=extent(files[1]))
            after.append(measured);write(a.directory/(prefix+'.metrics.json'),measured);del logits,teacher,routes
            print(json.dumps(dict(stage='native_finish',seconds=time.monotonic()-start,completed_cases=ordinal+1,
                case=rec['id'],KL=measured['KL'],disagreement=measured['disagreement_rate'])),flush=True)
        def aggregate(rows):
            def group(v):
                return dict(cases=len(v),labels=sum(r['labels'] for r in v),case_KL=sum(r['KL'] for r in v)/len(v),
                    label_KL=sum(sum(r['KL_per_label']) for r in v)/sum(r['labels'] for r in v),
                    case_disagreement=sum(r['disagreement_rate'] for r in v)/len(v),
                    label_disagreement=sum(r['disagreement'] for r in v)/sum(r['labels'] for r in v),
                    uniform_case_KL=sum(r['uniform_KL'] for r in v)/len(v))
            return {s:dict(**group([r for r in rows if r['split']==s]),domains={d:group([r for r in rows if r['split']==s and r['domain']==d])
                for d in sorted({r['domain'] for r in rows})}) for s in ('FIT','DEV')}
        ab=aggregate(before);aa=aggregate(after);dev=aa['DEV'];g=b['gates']
        gates=dict(absolute_DEV_KL=dev['case_KL']<=g['DEV_case_KL'],absolute_DEV_disagreement=dev['case_disagreement']<=g['DEV_case_disagreement'],
            all_domain_KL=all(v['case_KL']<=g['domain_case_KL'] for v in dev['domains'].values()),
            all_domain_disagreement=all(v['case_disagreement']<=g['domain_case_disagreement'] for v in dev['domains'].values()),
            relative_DEV_KL=dev['case_KL']<=g['relative_DEV_KL']*ab['DEV']['case_KL'],
            relative_DEV_disagreement=dev['case_disagreement']<=g['relative_DEV_disagreement']*ab['DEV']['case_disagreement'])
        decision='ORIGINAL_WHOLE_RECOVERY_DEV_DISTRIBUTION_GATES_PASS' if all(gates.values()) else 'ORIGINAL_WHOLE_RECOVERY_DEV_DISTRIBUTION_GATES_FAIL'
        snapshots=[]
        for line in Path(b['old_log']).read_text(encoding='utf8',errors='replace').splitlines():
            try:value=json.loads(line)
            except ValueError:continue
            if 'snapshot' in value:snapshots.append(value['snapshot'])
        result=dict(schema='ORIGINAL_FALCON_WHOLE_RECOVERY_FINISH_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            decision=decision,new_updates=0,inherited_updates=24,initial_counter=1,final_counter=25,updates=updates,snapshots=snapshots,
            before=before,after=after,before_aggregates=ab,after_aggregates=aa,gates=gates,packed=b['packed'],
            source_calls=0,quality_admission=False,speed_admission=False,useful_large_n_admission=False,
            historical_native_numerical_gate='FAIL retained',children=children,inherited_children=fault['children'],
            max_direct_child_OS_peak=child_peak,worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_allocated_peak=0,GPU_reserved_peak=0,GPU_scope='Only this CPU-only evaluation completion;original training peak values were not recorded on fault.',
            inherited_worker_OS_peak=failure['worker_OS_peak_through_exit'],inherited_GPU_peaks_missing=True,inherited_launcher_peak_missing=True,
            original_fault=fault['error'],original_family_seconds=failure['elapsed_seconds'],elapsed_seconds=time.monotonic()-start,
            combined_family_seconds=failure['elapsed_seconds']+time.monotonic()-start,
            excluded_quality_scopes=['old32 retention','fresh RESERVED','own-history generation','chatbot tasks','family/scale variants'])
        write(a.out,result);guard();print(json.dumps(dict(stage='complete',decision=decision,gates=gates,result_sha256=sha(a.out))),flush=True)
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(error=repr(error),after=after,children=children,elapsed_seconds=time.monotonic()-start,new_updates=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true');m.add_argument('--launch',action='store_true');m.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
