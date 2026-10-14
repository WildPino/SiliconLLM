"""Complete per-phase native/quality/custody audit, also on quality failures."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['OPENBLAS_NUM_THREADS']='6'
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from fixed_depth_quality import load,check_inputs,same_sequence,fit_quality,quality


def main(a):
    import numpy as np
    import psutil
    from fixed_depth_bridge import kept_sites
    from fixed_depth_native_audit import audit_native
    from original_engine_chat import ChatTokenizer
    from chatbot_falcon_usability import normalized
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));start=time.monotonic()
    r=load(a.report);b=load(r['binding']['path']);assert extent(r['binding']['path'])==r['binding']
    holder=load(a.report.with_suffix('.terminal.json'));worker=load(a.report.with_suffix('.worker_terminal.json'))
    assert holder['exit_code']==worker['exit_code']==holder['worker_exit_code']==0
    assert holder['result']==worker['result']==extent(a.report)
    assert holder['worker_pid']==worker['pid'] and abs(holder['worker_creation_time']-worker['creation_time'])<.001
    assert worker['phase']==r['phase'] and worker['depth']==r['depth']
    def guard():
        assert time.monotonic()-start<=b['limits']['audit_seconds'],'audit deadline'
        assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'OS limit'
        assert not proc.children(recursive=True),'unexpected child'
    fixed=[extent(a.report),r['binding'],extent(a.report.with_suffix('.terminal.json')),extent(a.report.with_suffix('.worker_terminal.json'))]
    def check():
        check_inputs(b,guard)
        for e in fixed+r['outputs']:guard();assert extent(e['path'])==e,e['path']
    check();active=list(kept_sites(r['depth']))
    assert len(r['records'])==24 and len(r['prefix_controls'])==16
    assert r['optimizer_updates']==r['RESERVED_queries']==r['omitted_source_operator_calls']==0
    assert r['all411_source_parameters_exact'] and r['before_after_hashes_exact'] and not r['timing_admitted']
    sequences=[q['sequence'] for q in r['records']+r['tasks']+([r['followup']] if r['followup'] else [])]
    sequences += [q[k] for q in r['prefix_controls'] for k in ('wrapper','continuation')]
    assert len({s['label'] for s in sequences})==len(sequences)
    assert sum(s['seen'] for s in sequences)==r['processed_tokens']
    for s in sequences:
        assert s['active_sites']==active and s['omitted_sites']==[i for i in range(24) if i not in active]
        assert [q['site'] for q in s['sites']]==active and s['selected_functions_per_step']==len(active)
        assert s['block_invocations']==[s['seen']]*24 and s['seen']==len(s['consumed_ids'])>0
        assert all(0<=v<65537 for v in s['consumed_ids'])
        assert len(s['attention'])==2*len(active) and [e['site'] for e in s['attention']]==[i for i in active for _ in range(2)]
        assert all(e['shape'][-2]==s['seen'] for e in s['attention'])
        assert s['native_parameter_payload_bytes']==len(active)*12952320
    directory=Path(r['outputs'][0]['path']).parent
    assert {str(p.resolve()) for p in directory.iterdir() if p.is_file()}=={e['path'] for e in r['outputs']}
    assert sum(e['bytes'] for e in r['outputs'])==r['output_bytes']
    before=load(directory/'parameter_before.json');after=load(directory/'parameter_after.json')
    assert before==after and before['canonical_coefficient_words']==1554863488 and len(before['tensors'])==411
    assert before['exact_source_BF16_values']
    for p in before['tensors']:
        spec=b['checkpoint_tensors'][p['name']]
        assert p['shape']==spec['shape'] and p['canonical_BF16_sha256']==spec['sha256']
    native=audit_native(b,sequences,np,guard)
    expected_split='FIT' if r['phase']=='fit' else 'DEV'
    originals=[q for q in b['records'] if q['split']==expected_split]
    assert [q['id'] for q in r['records']]==[q['id'] for q in originals]
    metric_rows=[]
    for rec,item in zip(originals,r['records'],strict=True):
        guard();assert item['sequence']['consumed_ids']==rec['student_input_ids']
        n=len(rec['positions']);assert item['labels']==n
        teacher=np.memmap(rec['logits']['path'],'u2','r',shape=(n,65537))
        scores=np.memmap(item['scores']['path'],'f4','r',shape=(n,65537));assert np.isfinite(scores).all()
        losses=[];wrong=0
        for i in range(n):
            q=(teacher[i].astype('u4')<<16).view('f4').astype('f8');p=scores[i].astype('f8')
            q-=np.logaddexp.reduce(q);p-=np.logaddexp.reduce(p)
            losses.append(float(np.sum(np.exp(q)*(q-p))));wrong+=int(q.argmax()!=p.argmax())
        assert np.allclose(losses,item['KL_per_label'],rtol=1e-10,atol=1e-10)
        assert abs(float(np.mean(losses))-item['KL'])<=1e-10 and wrong==item['disagreement']
        assert item['split']==rec['split'] and item['domain']==rec['domain'] and item['disagreement_rate']==wrong/n
        metric_rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],labels=n,
            KL=float(np.mean(losses)),disagreement=wrong,disagreement_rate=wrong/n))
    assert [q['id'] for q in r['prefix_controls']]==[q['id'] for q in b['cases']]
    for case,control in zip(b['cases'],r['prefix_controls'],strict=True):
        assert control['wrapper']['consumed_ids']==control['continuation']['consumed_ids']==case['input_ids']
        with np.load(control['scores']['path']) as z:exact=np.array_equal(z['wrapper'].view('u4'),z['continuation'].view('u4'))
        assert bool(exact and same_sequence(control['wrapper'],control['continuation'],np))==control['exact']==True
    parent=load(b['parent_report']['path'])
    if r['depth']==24:
        assert r['phase']=='fit' and r['parent_24_exact']
        for item in r['records']:
            old=next(q for q in parent['records'] if q['id']==item['id'])
            assert item['scores']['sha256']==old['scores']['sha256'] and same_sequence(item['sequence'],old['sequence'],np)
        for control in r['prefix_controls']:
            old=next(q for q in parent['prefix_controls'] if q['id']==control['id'])
            for key in ('wrapper','continuation'):assert same_sequence(control[key],old[key],np)
            with np.load(control['scores']['path']) as z,np.load(old['scores']['path']) as oldz:
                for key in ('wrapper','continuation'):assert np.array_equal(z[key].view('u4'),oldz[key].view('u4'))
    if r['phase']=='fit':
        fit=fit_quality(metric_rows);assert fit['gates']==r['FIT_quality']['gates']
        assert abs(fit['KL']-r['FIT_quality']['KL'])<=1e-10 and abs(fit['disagreement']-r['FIT_quality']['disagreement'])<=1e-10
        assert not r['tasks'] and r['followup'] is None and len(sequences)==56
        q=None;selection=None
    else:
        assert len(r['tasks'])==16 and r['followup'] is not None and len(sequences)==73
        assert extent(r['selection']['path'])==r['selection'];selection=load(r['selection']['path'])
        assert selection['evaluation_depth']==r['depth'] and selection['nonselected_defect_diagnostic']==r['nonselected_defect_diagnostic']
        eligible=[]
        for entry in selection['reports']:
            fitr=load(entry['report']['path']);ad=load(entry['audit']['path'])
            assert extent(entry['report']['path'])==entry['report'] and extent(entry['audit']['path'])==entry['audit']
            assert ad['decision']=='INDEPENDENT_STORED_NUMERIC_PASS' and ad['report']==entry['report']
            if entry['depth']!=24 and fit_quality(fitr['records'])['all_pass']:eligible.append(entry['depth'])
        assert selection['selected_depth']==(min(eligible) if eligible else None)
        tokenizer=ChatTokenizer(b['source'])
        for task in r['tasks']+[r['followup']]:
            ids=tokenizer.encode(task['messages']);new=task['output_ids']
            assert ids==task['input_ids'] and 1<=len(new)<=64 and task['sequence']['consumed_ids']==ids+new[:-1]
            scores=np.memmap(task['scores']['path'],'f4','r',shape=(len(new),65537))
            assert np.isfinite(scores).all() and scores.argmax(axis=1).tolist()==new
            text=tokenizer.decode(new);assert text==task['output_text'] and normalized(text)==task['normalized']
            assert task['correct']==(normalized(text)==task['expected'])
            assert task['special_leak']==any(t in set(b['special_ids'])-set(b['eos_ids']) for t in new)
            assert task['stop_policy']==(not any(t in b['eos_ids'] for t in new[:-1]) and (new[-1] in b['eos_ids'] or len(new)==64))
        assert [q['id'] for q in r['tasks']]==[q['id'] for q in b['cases']]
        for case,task in zip(b['cases'],r['tasks'],strict=True):assert task['expected']==case['expected'] and task['category']==case['category']
        original=next(q for q in b['cases'] if q['id']==b['followup']['case_id'])
        candidate=next(q for q in r['tasks'] if q['id']==b['followup']['case_id'])
        assert r['followup']['messages']==original['messages']+[dict(role='assistant',content=candidate['output_text']),dict(role='user',content=b['followup']['prompt'])]
        assert r['followup']['expected']==b['followup']['expected']
        q=quality(r['tasks'],r['followup'],metric_rows,r['prefix_controls'],b['gates']);assert q['gates']==r['quality']['gates']
        fit=None
    check();guard()
    write(a.out,dict(schema='FIXED_DEPTH_QUALITY_AUDIT_V1',report=extent(a.report),binding=r['binding'],depth=r['depth'],phase=r['phase'],
        native=native,all_numeric_pass=True,FIT_quality=fit,quality=q,selection=selection,
        decision='INDEPENDENT_STORED_NUMERIC_PASS',quality_admissible=bool(q and q['all_pass'] and not r['nonselected_defect_diagnostic']),
        before_after_hashes_exact=True,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,GPU_calls=0,
        scope='Every kept trajectory/state and independent F64/quantizer/integer/metrics/control/FIT choice checked. '
              'Source GPU attention/FFN/head are witnesses, no independent GPU arithmetic or final CPU TPS/DRAM/n admission.'))
    print(json.dumps(dict(audit=extent(a.out),depth=r['depth'],phase=r['phase'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();import psutil
    proc=psutil.Process();start=time.monotonic();t=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try:
        assert not a.out.exists();main(a);t.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:t.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        t.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),t)
