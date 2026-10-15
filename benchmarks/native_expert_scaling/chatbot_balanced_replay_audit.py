"""CPU audit of saved replay scores, state, exposure, resources and custody."""
import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='6')
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from fixed_depth_quality import load,check_inputs
REPORT=DOC/'chatbot_balanced_replay_result_20261015.json'
OUT=DOC/'chatbot_balanced_replay_audit_20261015.json'
PROTOCOL=DOC/'CHATBOT_BALANCED_REPLAY_AUDIT_20261015.md'
SECONDS=1800


def main(freeze,guard):
    import numpy as np
    import psutil
    import torch
    import chatbot_hybrid_target as model_code
    torch.set_num_threads(6)
    assert not torch.cuda.is_initialized() and not OUT.exists()
    for path in (Path(__file__),PROTOCOL):
        blob=subprocess.check_output(['git','show',freeze+':'+path.relative_to(ROOT).as_posix()],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==path.read_bytes().replace(b'\r\n',b'\n')
    r=load(REPORT);b=load(r['binding']['path']);assert extent(r['binding']['path'])==r['binding']
    holder=load(REPORT.with_suffix('.terminal.json'));worker=load(REPORT.with_suffix('.worker_terminal.json'))
    assert holder['exit_code']==holder['worker_exit_code']==worker['exit_code']==0
    assert holder['result']==worker['result']==extent(REPORT)
    assert holder['worker_pid']==worker['pid'] and abs(holder['worker_creation_time']-worker['creation_time'])<.001
    for receipt in (holder,worker):
        if psutil.pid_exists(receipt['pid']):assert abs(psutil.Process(receipt['pid']).create_time()-receipt['creation_time'])>.001
    pins=[extent(p) for p in (Path(__file__),PROTOCOL,REPORT,REPORT.with_suffix('.terminal.json'),REPORT.with_suffix('.worker_terminal.json'))]
    def seal():
        check_inputs(b,guard)
        for item in pins+r['outputs']:guard();assert extent(item['path'])==item,item['path']
    seal()
    assert r['updates']==b['updates']==512 and r['initial_step']==318 and r['final_step']==830
    assert r['saved_state_roundtrip_exact'] and r['inputs_before_after_exact'] and r['GPU_released_before_posthash']
    assert r['RESERVED_queries']==r['DEV_training_queries']==0 and not r['quality_admissible']
    assert r['seconds']<=b['limits']['seconds'] and r['GPU_phase_seconds']<=b['limits']['GPU_seconds']
    for key,cap in (('OS_peak','OS_bytes'),('GPU_allocated_peak','GPU_allocated_bytes'),('GPU_reserved_peak','GPU_reserved_bytes')):
        assert r[key]<=b['limits'][cap]
    directory=Path(b['family'])
    assert {str(p.resolve()):p.stat().st_size for p in directory.iterdir()}=={e['path']:e['bytes'] for e in r['outputs']}
    assert sum(e['bytes'] for e in r['outputs'])<=b['limits']['output_bytes'] and not (directory/'candidate.next.pt').exists()
    old_records,new_records=(load(path)['records'] for path in b['corpora'])
    records=old_records+new_records;by_id={q['id']:q for q in records};old_ids={q['id'] for q in old_records}
    old_fit=sorted((q for q in old_records if q['split']=='FIT'),key=lambda q:q['id'])
    prior_binding=load(DOC/'chatbot_broad_pilot_binding_repair1_20261009.json')
    new_order=prior_binding['order'];expected=[]
    for epoch in range(2):
        for i,record in enumerate(old_fit):expected.extend([record['id'],new_order[(epoch*128+i)%24]])
    assert len(old_fit)==128 and len(new_order)==24 and expected==b['order']
    assert all(by_id[name]['split']=='FIT' for name in expected)
    exposure=Counter(expected);assert len(exposure)==152 and all(exposure[q['id']]==2 for q in old_fit)
    assert set(exposure[name] for name in new_order)=={10,11}
    processed=0;compute=0.
    for step,(name,update) in enumerate(zip(expected,r['update_records'],strict=True),1):
        guard();assert update==load(directory/f'update_{step:04d}.json')
        assert update['step']==step and update['id']==name and update['optimizer_step']==318+step
        assert math.isfinite(update['KL']) and math.isfinite(update['gradient_norm']) and update['gradient_norm']>0
        assert [q['site'] for q in update['groups']]==list(range(12))
        assert all(math.isfinite(q[key]) and q[key]>0 for q in update['groups'] for key in ('core','banks','norm'))
        processed+=len(by_id[name]['student_input_ids']);assert update['trained_tokens']==processed
        assert 0<=update['mass_max']<=1e-6 and 0<update['compute_seconds']<1800
        compute+=update['compute_seconds']
    routes=np.array(r['route_counts'],dtype='i8');assert routes.shape==(12,72) and (routes>=0).all()
    assert np.all(routes.sum(1)==processed*8) and r['mass_max']<=1e-6
    checkpoint=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=True)
    assert extent(r['checkpoint']['path'])==r['checkpoint']
    assert checkpoint['candidate_schema']=='BALANCED_512_TRANSFER_CANDIDATE_V1'
    assert checkpoint['completed_balanced_updates']==512 and checkpoint['optimizer_step']==830 and checkpoint['initial_optimizer_step']==318
    assert checkpoint['binding']==r['binding'] and checkpoint['order']==expected and checkpoint['updates']==r['update_records']
    assert checkpoint['route_counts']==r['route_counts'] and len(checkpoint['model'])==len(checkpoint['optimizer']['state'])==211
    with torch.device('meta'):prototype=model_code.Target(checkpoint['config'])
    parameters=dict(prototype.named_parameters());assert list(parameters)==list(checkpoint['model'])
    assert sum(t.numel() for t in parameters.values())==254932736
    ids=sum([group['params'] for group in checkpoint['optimizer']['param_groups']],[])
    assert len(ids)==len(set(ids))==211 and set(ids)==set(checkpoint['optimizer']['state'])
    master_words=moment_words=0
    for (name,parameter),index in zip(parameters.items(),ids,strict=True):
        guard();tensor=checkpoint['model'][name];slot=checkpoint['optimizer']['state'][index]
        assert tuple(tensor.shape)==tuple(parameter.shape) and tensor.dtype==torch.float32 and tensor.device.type=='cpu'
        assert np.isfinite(tensor.numpy()).all();master_words+=tensor.numel()
        if name.endswith(('_scale',)):
            assert (tensor.numpy()>=np.float32(1e-8)).all()
        assert int(slot['step'].item())==830
        for key in ('exp_avg','exp_avg_sq'):
            value=slot[key];assert value.shape==tensor.shape and value.dtype==torch.float32 and np.isfinite(value.numpy()).all()
            if key=='exp_avg_sq':assert (value.numpy()>=0).all()
            moment_words+=value.numel()
    assert master_words==254932736 and moment_words==2*master_words
    for group in checkpoint['optimizer']['param_groups']:
        assert group['lr']==5e-5 and group['weight_decay']==0 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
    assert checkpoint['torch_CPU_rng'].dtype==torch.uint8 and checkpoint['torch_CPU_rng'].device.type=='cpu'
    assert len(checkpoint['torch_CUDA_rng'])==1 and checkpoint['torch_CUDA_rng'][0].dtype==torch.uint8
    del checkpoint,prototype,parameters,tensor,value,slot;gc.collect()
    prior=load(DOC/'chatbot_broad_pilot_result_repair1_20261009.json')
    for control in r['current_runtime_controls']:
        guard();row=next(q for q in prior['after']['cases'] if q['id']==control['id']);shape=tuple(row['logits']['shape'])
        a=np.memmap(control['scores']['path'],'f4','r',shape=shape);p=np.memmap(row['logits']['path'],'f4','r',shape=shape)
        assert np.isfinite(a).all() and np.array_equal(a.view('u4'),p.view('u4'))
        assert control['relative']==0 and a.argmax(1).tolist()==row['predicted_ids']
    wanted=[q['id'] for q in records if q['split']=='DEV' or (q['split']=='FIT' and q['id'] in set(new_order))]
    assert len(wanted)==80 and [q['id'] for q in r['evaluation']]==wanted
    rows=[];largest_label_error=0.;label_count=0
    for item in r['evaluation']:
        guard();record=by_id[item['id']];assert item==load(directory/(item['id']+'.metrics.json'))
        n=len(record['output_ids']);shape=(n,65537)
        assert item['split']==record['split'] and item['domain']==record['domain'] and item['labels']==n
        assert item['scores']['shape']==list(shape) and item['scores']['bytes']==n*65537*4
        values=np.memmap(item['scores']['path'],'f4','r',shape=shape)
        teacher=np.memmap(record['logits']['path'],'u2','r',shape=shape)
        assert np.isfinite(values).all();winners=values.argmax(1).tolist();assert winners==item['predicted_ids']
        errors=sum(a!=b for a,b in zip(winners,record['output_ids'],strict=True));assert errors==item['disagreements']
        losses=[]
        for position in range(n):
            q=(teacher[position].astype('u4')<<16).view('f4').astype('f8');p=values[position].astype('f8')
            q-=np.logaddexp.reduce(q);p-=np.logaddexp.reduce(p)
            losses.append(float(np.sum(np.exp(q)*(q-p))))
        delta=float(np.max(np.abs(np.array(losses)-item['KL_per_label'])));largest_label_error=max(largest_label_error,delta)
        assert delta<=1e-4 and abs(float(np.mean(losses))-item['KL'])<=5e-5
        rows.append(dict(id=item['id'],split=item['split'],domain=item['domain'],labels=n,KL=float(np.mean(losses)),disagreements=errors))
        label_count+=n
        print(json.dumps(dict(operation='scores_audited',cases=len(rows),labels=label_count,id=item['id'])),flush=True)
    summaries=[]
    for cohort in ('old_DEV','new_DEV','new_FIT'):
        selected=[q for q in rows if (q['split']=='FIT')==(cohort=='new_FIT') and ((q['id'] in old_ids)==(cohort=='old_DEV'))]
        assert len(selected)==(32 if cohort=='old_DEV' else 24)
        domains={}
        for domain in sorted({q['domain'] for q in selected}):
            subset=[q for q in selected if q['domain']==domain]
            domains[domain]=dict(KL=sum(q['KL'] for q in subset)/len(subset),disagreement=sum(q['disagreements'] for q in subset)/sum(q['labels'] for q in subset))
        summary=dict(cohort=cohort,KL=sum(q['KL'] for q in selected)/len(selected),
            disagreement=sum(q['disagreements'] for q in selected)/sum(q['labels'] for q in selected),domains=domains)
        summary['absolute_pass']=summary['KL']<=1 and summary['disagreement']<=.2 and all(v['KL']<=2 and v['disagreement']<=.35 for v in domains.values())
        old=next(q for q in r['summaries'] if q['cohort']==cohort)
        assert abs(summary['KL']-old['KL'])<=5e-5 and summary['disagreement']==old['disagreement'] and summary['absolute_pass']==old['absolute_pass']
        assert domains.keys()==old['domains'].keys()
        for domain in domains:
            assert abs(domains[domain]['KL']-old['domains'][domain]['KL'])<=5e-5 and domains[domain]['disagreement']==old['domains'][domain]['disagreement']
        summaries.append(summary)
    decision='KNOWN_PREFIX_GATES_PASS' if all(q['absolute_pass'] for q in summaries) else 'KNOWN_PREFIX_GATES_FAIL'
    assert decision==r['decision'];seal();guard()
    write(OUT,dict(schema='BALANCED_REPLAY_STORED_AUDIT_V1',freeze=freeze,report=extent(REPORT),binding=r['binding'],
        checkpoint=r['checkpoint'],fixed_inputs=pins,all_stored_checks_pass=True,quality_decision=decision,quality_admissible=False,
        summaries=summaries,evaluation_cases=80,label_positions=label_count,logit_words=label_count*65537,
        F64_label_KL_max_difference=largest_label_error,model_fields=211,master_words=master_words,moment_words=moment_words,
        optimizer_step=830,training_updates=512,unique_FIT_ids=152,trained_prefix_tokens=processed,
        per_site_expert_coverage=(routes>0).sum(1).tolist(),route_counts_exact=True,mass_max=r['mass_max'],
        current_runtime_controls_bit_exact=True,inputs_outputs_before_after_exact=True,
        recorded_compute_seconds=compute,producer_in_memory_roundtrip_witness=r['saved_state_roundtrip_exact'],
        independent_past_in_memory_or_gradient_replay=False,independent_final_model_forward=False,GPU_calls=0,
        seconds=time.monotonic()-START,OS_peak=psutil.Process().memory_info().peak_wset,
        scope='Full stored score/state/optimizer/RNG/exposure/resource/current-custody audit. GPU training and final forwards are producer witnesses; no native/generation/fresh-heldout/rate/DRAM/useful-n admission.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',required=True);args=parser.parse_args()
    import psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));START=time.monotonic();last_children=0.
    terminal=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),seconds_cap=SECONDS,OS_bytes_cap=16<<30,GPU_calls=0)
    def guard():
        global last_children
        assert time.monotonic()-START<=SECONDS,'CPU audit deadline'
        assert proc.memory_info().peak_wset<=16<<30,'OS cap'
        if time.monotonic()-last_children>=1:
            assert not proc.children(recursive=True);last_children=time.monotonic()
        if 'torch' in sys.modules:assert not sys.modules['torch'].cuda.is_initialized()
    try:
        assert not OUT.exists() and not OUT.with_suffix('.terminal.json').exists()
        write(OUT.with_suffix('.owned_process.json'),terminal)
        main(args.freeze,guard);terminal.update(exit_code=0,error=None,result=extent(OUT))
    except BaseException as error:terminal.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        terminal.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(OUT.with_suffix('.terminal.json'),terminal)
