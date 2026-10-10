"""Stored first-response conditional diagnostic; no model/history/fit call."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write
sys.path.insert(0,str(SITE))

def main(a):
    import numpy as np
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([1]);b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    before=[extent(item['path']) for item in b['inputs']];assert before==b['inputs']
    r=json.loads(Path(b['native_result']['path']).read_bytes());audit=json.loads(Path(b['native_audit']['path']).read_bytes());term=json.loads(Path(b['native_audit_terminal']['path']).read_bytes())
    assert audit['result']==b['native_result'] and term['exit_code']==0 and term['error'] is None and term['result']==b['native_audit'] and term['inputs_before_after_exact']
    native_binding=json.loads(Path(b['native_binding']['path']).read_bytes());rows=[]
    for case in native_binding['cases']:
        source=json.loads(Path(case['source_record']['path']).read_bytes());task=next(t for t in r['tasks'] if t['id']==case['id']);assert source['input_ids']==task['request']['query_ids'] and len(source['output_ids']) and len(task['request']['generated_ids'])
        q=np.fromfile(source['scores']['path'],dtype='<f4',count=65537).astype('f8');p=np.fromfile(task['request']['scores']['path'],dtype='<f4',count=65537).astype('f8');assert np.isfinite(q).all() and np.isfinite(p).all()
        assert int(q.argmax())==source['output_ids'][0] and int(p.argmax())==task['request']['generated_ids'][0]
        lq=q-q.max();lq-=math.log(float(np.exp(lq).sum()));lp=p-p.max();lp-=math.log(float(np.exp(lp).sum()));loss=float(np.dot(np.exp(lq),lq-lp))
        rq=q-np.logaddexp.reduce(q);rp=p-np.logaddexp.reduce(p);reference=math.fsum(float(x)*float(y) for x,y in zip(np.exp(rq),rq-rp));assert abs(reference-loss)<=1e-10
        rows.append(dict(id=case['id'],category=case['category'],source_correct=source['correct'],source_first_id=source['output_ids'][0],native_first_id=task['request']['generated_ids'][0],first_disagreement=source['output_ids'][0]!=task['request']['generated_ids'][0],first_KL=loss,independent_KL=reference,source_first_probability=float(np.exp(lq[int(q.argmax())])),native_probability_of_source_first=float(np.exp(lp[int(q.argmax())])),native_full_task_correct=task['correct']))
    phase={}
    for split in ('FIT','DEV'):
        selected=[x for x in r['native_records'] if x['split']==split];assert len(selected)==24
        records={x['id']:x for x in native_binding['records']}
        for row in selected:assert records[row['id']]['positions'][0]==len(records[row['id']]['input_ids'])-1
        onset=sum(x['KL_per_label'][0] for x in selected)/24
        continuation=sum(sum(x['KL_per_label'][1:])/(x['labels']-1) for x in selected)/24
        weight=sum(1/x['labels'] for x in selected)/24
        phase[split]=dict(cases=24,first_response_case_KL=onset,continuation_case_KL=continuation,whole_case_KL=r['native_aggregate'][split]['case_KL'],first_response_total_weight=weight,continuation_total_weight=1-weight,first_response_case_losses={x['id']:x['KL_per_label'][0] for x in selected})
    correct_source=[row for row in rows if row['source_correct']];assert len(correct_source)==14
    disagree=sum(x['first_disagreement'] for x in correct_source);decision='FIRST_RESPONSE_CONDITIONAL_GAP' if disagree>=8 else 'ONSET_NOT_DOMINANT_BY_FROZEN_GATE'
    assert all(extent(item['path'])==item for item in b['inputs'])
    assert time.monotonic()-start<=60 and proc.memory_info().peak_wset<=256<<20 and not proc.children(recursive=True)
    result=dict(schema='CATEGORICAL_RESPONSE_ONSET_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,records=rows,phase=phase,source_correct_cases=14,first_disagreement_on_source_correct=disagree,first_disagreement_all=sum(x['first_disagreement'] for x in rows),first_case_KL=sum(x['first_KL'] for x in rows)/16,distinct_native_first_ids=sorted({x['native_first_id'] for x in rows}),decision=decision,all16_first_full_V_heads_verified=True,all48_phase_metrics_verified=True,all_input_hashes_before_after_exact=True,source_calls=0,native_calls=0,optimizer_updates=0,head_FG_calls=0,reserved_queries=0,GPU_calls=0,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,scope='Stored conditional first response before any candidate-answer feedback. Exact original source/new native prompt IDs. Does not prove internal information loss, linear-head floor or temporal reweighting sufficiency.')
    write(a.out,result);assert a.out.stat().st_size<1<<20;print(json.dumps(dict(decision=decision,disagreement=disagree,phase=phase,distinct_native_first_ids=result['distinct_native_first_ids'],seconds=result['seconds'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('binding','out'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('binding-sha','freeze'):p.add_argument('--'+key,required=True)
    main(p.parse_args())
