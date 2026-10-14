"""Read-only recovery of complete FIT24 data; original failed receipts stay failed."""
import argparse
import json
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
from fixed_depth_quality import load,fit_quality,same_sequence

STEM=DOC/'fixed_depth_fit24_result_20261014.json'
BINDING=DOC/'fixed_depth_fit24_recovery_binding_20261014.json'
AUDIT=DOC/'fixed_depth_fit24_audit_20261014.json'
CLOSURE=DOC/'fixed_depth_fit24_recovery_closure_20261014.json'
SECONDS=7200


def guards(start,proc):
    def guard():
        assert time.monotonic()-start<=SECONDS,'CPU recovery deadline'
        assert proc.memory_info().peak_wset<=8<<30,'CPU OS cap'
        assert not proc.children(recursive=True),'unexpected child'
        if 'torch' in sys.modules:assert not sys.modules['torch'].cuda.is_initialized()
        recovery=[p for p in DOC.glob('fixed_depth_fit24_recovery*') if p.is_file()]
        assert sum(p.stat().st_size for p in recovery)<16<<20
    return guard


def check(pins,guard):
    for e in pins:guard();assert extent(e['path'])==e,e['path']


def receipts():
    wt=STEM.with_suffix('.worker_terminal.json');ht=STEM.with_suffix('.terminal.json')
    w,h=load(wt),load(ht)
    assert w['exit_code']==h['exit_code']==h['worker_exit_code']==1
    assert w['stage']=='after_hash' and w['completed_histories']==24 and w['completed_controls']==16
    assert w['error']=="AssertionError('worker deadline')" and w['phase']=='fit' and w['depth']==24
    assert h['worker_pid']==w['pid'] and abs(h['worker_creation_time']-w['creation_time'])<.001
    import psutil
    for t in (w,h):
        if psutil.pid_exists(t['pid']):assert abs(psutil.Process(t['pid']).create_time()-t['creation_time'])>.001
    return w,h,extent(wt),extent(ht)


def prepare(a,guard):
    w,h,we,he=receipts();assert not STEM.exists() and not BINDING.exists() and not AUDIT.exists()
    old=DOC/'fixed_depth_quality_binding_20261014.json';b=load(old)
    directory=Path(b['family'])/'fit24';files=sorted(p for p in directory.iterdir() if p.is_file())
    assert len(files)==2755 and sum(p.stat().st_size for p in files)==8876432536
    assert load(directory/'parameter_before.json')==load(directory/'parameter_after.json')
    frozen=[Path(__file__),B/'fixed_depth_quality_audit_recovered.py',
            DOC/'FIXED_DEPTH_POSTHASH_RECOVERY_20261014.md']
    for p in frozen:
        blob=subprocess.check_output(['git','show',a.freeze+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n')
    paths=frozen+[old,Path(we['path']),Path(he['path']),STEM.with_suffix('.worker.log'),STEM.with_suffix('.owned_process.json')]
    inputs=[]
    for p in paths:guard();inputs.append(extent(p))
    rb=dict(schema='FIXED_DEPTH_POSTHASH_RECOVERY_BINDING_V1',freeze=a.freeze,old_binding=extent(old),
        original_worker=we,original_holder=he,inputs=inputs,directory=str(directory),
        seconds=SECONDS,OS_bytes=8<<30,metadata_bytes=16<<20,original_posthash_complete=False,
        lineage='Original inference/411 parameter checks complete, original worker/holder exit1 at posthash. '
                'Current CPU reconciliation cannot recreate original posthash/exit0.',prefix_attention='ATTESTED: source/log checks, parent values; no fresh KV byte capture')
    write(BINDING,rb)


def reconcile(guard):
    import numpy as np
    rb=load(BINDING);b=load(rb['old_binding']['path']);directory=Path(rb['directory'])
    assert extent(rb['old_binding']['path'])==rb['old_binding'];receipts()
    check(rb['inputs']+b['inputs'],guard)
    outputs=[]
    for p in sorted(directory.iterdir()):guard();assert p.is_file();outputs.append(extent(p))
    assert len(outputs)==2755
    sealed={e['path']:e for e in outputs}
    records=[load(directory/(q['id']+'.metrics.json')) for q in b['records'] if q['split']=='FIT']
    assert len(records)==24
    parent=load(b['parent_report']['path']);log=STEM.with_suffix('.worker.log').read_text(encoding='utf8')
    events=[]
    for line in log.splitlines():
        if line.startswith('{'):
            try:events.append(json.loads(line))
            except json.JSONDecodeError:raise AssertionError('Invalid structured worker event')
    prefixes=[q for q in events if q.get('operation')=='prefix']
    assert [q['id'] for q in prefixes]==[q['id'] for q in b['cases']] and all(q['exact'] for q in prefixes)
    assert [q['id'] for q in events if q.get('operation')=='history_complete']==[q['id'] for q in records]
    controls=[]
    parameter_bytes=24*12952320
    for case in b['cases']:
        old=next(q for q in parent['prefix_controls'] if q['id']==case['id']);sequences={}
        for mode in ('wrapper','continuation'):
            suffix='prefix_wrapper' if mode=='wrapper' else 'prefix_steps';label=case['id']+'.'+suffix
            sites=[];n=len(case['input_ids'])
            for site in range(24):
                guard();trace=directory/f'{label}.site{site:02d}.trace.bf16';final=directory/f'{label}.site{site:02d}.state.npz'
                width=2048 if site==0 else 10800
                with np.load(final) as z:
                    assert z['seen'][0]==n and z['last'][0]==case['input_ids'][-1]
                te=sealed[str(trace.resolve())];assert te['bytes']==n*width*2
                sites.append(dict(site=site,trace=dict(**te,shape=[n,width],dtype='BF16'),final=sealed[str(final.resolve())]))
            # Explicit inference from enforced, logged parent-cache equality; no fresh KV capture.
            seq=dict(label=label,consumed_ids=case['input_ids'],seen=n,sites=sites,attention=old[mode]['attention'],
                active_sites=list(range(24)),omitted_sites=[],block_invocations=[n]*24,selected_functions_per_step=24,
                native_parameter_payload_bytes=parameter_bytes,selection_policy='Frozen source site schedule, masses1.',
                prefix_attention_provenance='Parent digest values attested by frozen executed/logged equality checks; not fresh capture')
            assert same_sequence(seq,old[mode],np);sequences[mode]=seq
        p=directory/(case['id']+'.prefix_scores.npz')
        with np.load(p) as z,np.load(old['scores']['path']) as oz:
            for k in ('wrapper','continuation'):assert np.array_equal(z[k].view('u4'),oz[k].view('u4'))
        controls.append(dict(id=case['id'],exact=True,scores=sealed[str(p.resolve())],**sequences))
    for row in records:
        old=next(q for q in parent['records'] if q['id']==row['id'])
        assert row['scores']['sha256']==old['scores']['sha256'] and same_sequence(row['sequence'],old['sequence'],np)
    before=load(directory/'parameter_before.json');after=load(directory/'parameter_after.json');assert before==after
    assert before['exact_source_BF16_values'] and before['canonical_coefficient_words']==1554863488
    sequences=[q['sequence'] for q in records]+[q[k] for q in controls for k in ('wrapper','continuation')]
    processed=sum(q['seen'] for q in sequences);assert processed==b['processed_bounds']['FIT_processed_tokens']
    # Full before seal above; full after seal is in the inline numerical auditor.
    w,h,we,he=receipts()
    r=dict(schema='FIXED_DEPTH_QUALITY_RECOVERED_CONTROL_V1',binding=rb['old_binding'],phase='fit',depth=24,
        selection=None,nonselected_defect_diagnostic=False,records=records,prefix_controls=controls,tasks=[],followup=None,
        FIT_quality=fit_quality(records),quality=None,parent_24_exact=True,processed_tokens=processed,
        all411_source_parameters_exact=True,before_after_hashes_exact=False,original_posthash_complete=False,
        reconciliation_current_inputs_outputs_exact=True,outputs=outputs,optimizer_updates=0,RESERVED_queries=0,
        omitted_source_operator_calls=0,timing_admitted=False,seconds=w['seconds'],OS_peak=w['OS_peak'],
        GPU_allocated_peak=w['GPU_allocated_peak'],GPU_reserved_peak=w['GPU_reserved_peak'],output_bytes=sum(e['bytes'] for e in outputs),
        recovery=dict(binding=extent(BINDING),original_worker=we,original_holder=he,prefix_attention=rb['prefix_attention']),
        scope=b['scope'],custody=rb['lineage'])
    write(STEM,r);print(dict(recovered=extent(STEM),FIT_quality=r['FIT_quality']),flush=True)


def close(guard):
    import psutil
    r=load(STEM);ad=load(AUDIT);rb=load(BINDING);b=load(r['binding']['path'])
    assert ad['decision']=='INDEPENDENT_STORED_NUMERIC_PASS' and ad['report']==extent(STEM) and ad['all_numeric_pass']
    assert ad['original_posthash_complete']==False and ad['recovery_binding']==extent(BINDING)
    t=load(AUDIT.with_suffix('.terminal.json'));assert t['exit_code']==0 and t['result']==extent(AUDIT)
    # The original receipts remain exit1; closure is explicitly of recovered stored evidence.
    # The inline auditor just rehashed ALL old inputs/outputs after every check.
    # Closure now checks metadata without a third redundant full data read.
    receipts();check(rb['inputs']+[r['binding'],extent(STEM),extent(AUDIT)],guard)
    write(CLOSURE,dict(schema='FIXED_DEPTH_RECOVERED_CONTROL_CLOSURE_V1',report=extent(STEM),audit=extent(AUDIT),
        recovery_binding=extent(BINDING),original_worker=rb['original_worker'],original_holder=rb['original_holder'],
        original_worker_exit_code=1,original_posthash_complete=False,prefix_attention=rb['prefix_attention'],
        current_inputs_outputs_exact=True,phase_family_files=len(r['outputs']),phase_family_bytes=r['output_bytes'],
        decision='STORED_CONTROL_QUALIFIED_WITH_EXPLICIT_ORIGINAL_POSTHASH_FAILURE',goal_status='ACTIVE_INCOMPLETE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);a=p.parse_args()
    import psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));start=time.monotonic();guard=guards(start,proc)
    out=DOC/'fixed_depth_fit24_recovery_pipeline_20261014.json'
    t=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),seconds_cap=SECONDS,OS_bytes_cap=8<<30)
    stage='prepare'
    try:
        assert not out.exists() and not STEM.exists() and not BINDING.exists() and not AUDIT.exists()
        write(out.with_suffix('.owned_process.json'),t)
        prepare(a,guard);stage='reconcile';reconcile(guard)
        # Reuse unchanged full mathematical audit, with explicit recovery-receipt branch only.
        import fixed_depth_quality_audit_recovered as auditing
        auditing.PIPELINE_GUARD=guard
        stage='full_numeric_audit';auditing.main(argparse.Namespace(report=STEM,out=AUDIT))
        audit_terminal=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),exit_code=0,
            result=extent(AUDIT),execution='Inline full recovered auditor, same owned CPU pipeline; not a separate worker')
        write(AUDIT.with_suffix('.terminal.json'),audit_terminal)
        stage='closure';close(guard);guard()
        write(out,dict(report=extent(STEM),audit=extent(AUDIT),closure=extent(CLOSURE),
            original_worker_exit_code=1,original_posthash_complete=False,decision='RECOVERED_STORED_CONTROL_PIPELINE_PASS'))
        t.update(exit_code=0,error=None,result=extent(out))
    except BaseException as error:t.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage);raise
    finally:
        t.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(out.with_suffix('.terminal.json'),t)
