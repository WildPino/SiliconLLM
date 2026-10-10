"""Frozen stored-score diagnosis of the single source alignment gate failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
sys.path.insert(0, str(B))
from original_packed_capacity import SITE, extent, sha, write
from original_falcon_whole_recovery import memory_reader
sys.path.insert(0, str(SITE))
IDENT = 'broad_dev_everyday_conversations_036'
V = 65537


def bind(a):
    r = json.loads(a.result.read_bytes())
    audit = json.loads(a.audit.read_bytes())
    receipt = json.loads(a.audit.with_suffix('.receipt.json').read_bytes())
    assert receipt['exit_code'] == 0 and receipt['error'] is None
    assert receipt['output'] == extent(a.audit)
    assert audit['result'] == extent(a.result)
    assert audit['decision'] == r['decision'] == 'SOURCE_STATE_LABEL_ALIGNMENT_FAIL'
    rows = {arm['name']: next(row for row in arm['records'] if row['id'] == IDENT) for arm in r['arms']}
    for row in rows.values():
        assert row['labels'] == 53 and row['disagreement'] == 3
    assert rows['BF16']['teacher'] == rows['F64']['teacher']
    paths = [Path(__file__), B/'original_packed_capacity.py', B/'original_falcon_whole_recovery.py',
             Path(sys.executable), Path(sys.executable).parent/'python312.dll', a.protocol,
             a.result, a.audit, a.audit.with_suffix('.receipt.json'),
             a.result.with_suffix('.terminal.json'),
             Path(rows['BF16']['teacher']['path']),
             Path(rows['BF16']['scores']['path']), Path(rows['F64']['scores']['path'])]
    # Adopt qualified pure helpers and NumPy/psutil runtime identities.
    old = json.loads(a.source_binding.read_bytes())
    paths += [Path(x['path']) for x in old['inputs'] if any(key in x['path'].lower() for key in ('numpy','psutil')) or x['path'].lower().endswith('.py')]
    paths += [a.source_binding]
    inputs = [extent(p) for p in dict.fromkeys(paths)]
    payload = dict(schema='SOURCE_READOUT_MARGIN_BINDING_V1',inputs=inputs,rows=rows,
                   result=extent(a.result),audit=extent(a.audit),id=IDENT,
                   limits=dict(seconds=60,OS_bytes=512<<20,log_bytes=1<<20,output_bytes=256<<10),
                   metric_tolerance=1e-10,scope='Stored score diagnosis only; original FAIL retained.')
    write(a.binding,payload)
    print(json.dumps(dict(binding=str(a.binding),sha256=sha(a.binding),inputs=len(inputs),bytes=sum(x['bytes'] for x in inputs))),flush=True)


def decode(row):
    import numpy as np
    return (row.astype('<u4') << 16).view('<f4').astype('f8')


def metrics(q,p):
    import numpy as np
    q = q - q.max(); p = p - p.max()
    q = q - np.logaddexp.reduce(q); p = p - np.logaddexp.reduce(p)
    return float(np.dot(np.exp(q),q-p))


def worker(a):
    import numpy as np
    import psutil
    psutil.Process().cpu_affinity(list(range(6)))
    b=json.loads(a.binding.read_bytes());rows=b['rows'];m=53
    arrays={name:np.memmap(row['scores']['path'],mode='r',dtype='<u2' if name=='BF16' else '<f8',shape=(m,V)) for name,row in rows.items()}
    teacher=np.memmap(rows['BF16']['teacher']['path'],mode='r',dtype='<u2',shape=(m,V))
    observations=[];max_metric=0.;counts={name:0 for name in rows}
    shifted={name:{'-1':[], '1':[]} for name in rows}
    for j in range(m):
        q=decode(teacher[j]);winner=int(np.argmax(q));top=np.flatnonzero(q==q[winner])
        runner=float(np.partition(q,-2)[-2]);gap=float(q[winner]-runner)
        item=dict(label_index=j,history_position=rows['BF16']['positions'][j],teacher_argmax=winner,
                  teacher_top_ties=top.tolist(),teacher_top_two_gap=gap,readouts={})
        for name,array in arrays.items():
            p=decode(array[j]) if name=='BF16' else np.array(array[j],dtype='f8')
            pred=int(np.argmax(p));bad=pred!=winner;counts[name]+=int(bad)
            kl=metrics(q,p);max_metric=max(max_metric,abs(kl-rows[name]['KL_per_label'][j]))
            delta=p-q;lo=float(delta.min());hi=float(delta.max());eps=(hi-lo)/2
            certified=gap>2*eps
            assert not (certified and bad),'argmax margin certificate contradiction'
            candidate_gap=float(q[winner]-q[pred])
            # The selected contender must satisfy this exact pairwise perturbation identity.
            perturbation=float(delta[pred]-delta[winner])
            assert not bad or perturbation>=candidate_gap-1e-12
            item['readouts'][name]=dict(argmax=pred,disagreement=bad,KL=kl,
                teacher_gap_to_prediction=candidate_gap,prediction_in_teacher_top_ties=bool(q[pred]==q[winner]),
                predicted_top_tie_count=int(np.count_nonzero(p==p[pred])),
                predicted_advantage=float(p[pred]-p[winner]),pairwise_perturbation=perturbation,
                centered_Linf_error=eps,centering_shift=(hi+lo)/2,
                unique_argmax_certified=certified)
            for shift in (-1,1):
                if 0<=j+shift<m:shifted[name][str(shift)].append(metrics(decode(teacher[j+shift]),p))
        observations.append(item)
    assert counts=={name:rows[name]['disagreement'] for name in rows}
    assert max_metric<=b['metric_tolerance']
    mismatch=[x for x in observations if any(y['disagreement'] for y in x['readouts'].values())]
    categories={name:dict(total=counts[name],teacher_top_tie_flips=sum(x['readouts'][name]['disagreement'] and x['readouts'][name]['prediction_in_teacher_top_ties'] for x in observations),
                    positive_teacher_gap_flips=sum(x['readouts'][name]['disagreement'] and not x['readouts'][name]['prediction_in_teacher_top_ties'] for x in observations),
                    certified_labels=sum(x['readouts'][name]['unique_argmax_certified'] for x in observations)) for name in rows}
    decision='ALL_MISMATCHES_TEACHER_TOP_TIES' if all(x['positive_teacher_gap_flips']==0 for x in categories.values()) else 'POSITIVE_MARGIN_DRIFT_PRESENT'
    write(a.out,dict(schema='SOURCE_READOUT_MARGIN_RESULT_V1',binding_sha256=sha(a.binding),freeze=a.freeze,id=IDENT,labels=m,
        original_decision='SOURCE_STATE_LABEL_ALIGNMENT_FAIL',diagnosis=decision,categories=categories,
        mismatches=mismatch,rows=observations,max_absolute_metric_delta=max_metric,
        temporal_shift_control={name:{shift:dict(labels=len(v),mean_KL=float(np.mean(v))) for shift,v in groups.items()} for name,groups in shifted.items()},
        source_history_forwards=0,new_head_contractions=0,optimizer_updates=0,reserved_queries=0,
        scope='Post-result diagnosis on one DEV case; no gate change, codec fit, or causal attribution to an unobserved source history.'))
    print(json.dumps(dict(diagnosis=decision,categories=categories)),flush=True)


def launch(a):
    import psutil
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','native','original_engine','clang','lld','packed_original')),('overlap',p.pid,name)
    log=a.out.with_suffix('.worker.log');receipt=a.out.with_suffix('.receipt.json')
    assert not any(p.exists() for p in (a.out,log,receipt))
    assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());lim=b['limits']
    began=time.monotonic();reader=memory_reader();peak=0
    for item in b['inputs']:assert extent(item['path'])==item
    command=[sys.executable,'-I','-S','-B','-X','utf8',str(Path(__file__).resolve()),'--worker',
             '--binding',str(a.binding.resolve()),'--out',str(a.out.resolve()),'--freeze',a.freeze]
    record=dict(binding_sha256=a.binding_sha,freeze=a.freeze,command=command,launcher_pid=proc.pid,limits=lim)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    with log.open('xb') as output:
        child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env,creationflags=8)
        record.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time());fault=None
        try:
            while child.poll() is None:
                peak=max(peak,reader(child))
                assert time.monotonic()-began<=lim['seconds'],'deadline'
                assert peak+proc.memory_info().peak_wset<=lim['OS_bytes'],'OS cap'
                assert log.stat().st_size<=lim['log_bytes'],'log cap'
                try:assert not psutil.Process(child.pid).children(recursive=True),'unexpected child'
                except psutil.NoSuchProcess:pass
                if a.out.exists():assert a.out.stat().st_size<=lim['output_bytes'],'output cap'
                time.sleep(.1)
            assert child.returncode==0,log.read_text(errors='replace')[-4000:]
            for item in b['inputs']:assert extent(item['path'])==item
            assert time.monotonic()-began<=lim['seconds'],'sealing deadline'
            record.update(inputs_before_after_exact=True,result=extent(a.out))
        except BaseException as error:
            fault=repr(error)
            if child.poll() is None:child.kill();child.wait()
        finally:
            peak=max(peak,reader(child));record.update(exit_code=child.returncode,error=fault,seconds=time.monotonic()-began,
                worker_OS_peak_through_exit=peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,log=extent(log))
            write(receipt,record)
    assert child.returncode==0 and fault is None,fault
    print(json.dumps(dict(receipt=str(receipt),seconds=record['seconds'],result_sha256=sha(a.out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bind',action='store_true');p.add_argument('--worker',action='store_true')
    for key in ('binding','out','result','audit','protocol','source-binding'):p.add_argument('--'+key,type=Path)
    for key in ('binding-sha','freeze'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.bind:bind(a)
    elif a.worker:worker(a)
    else:launch(a)
