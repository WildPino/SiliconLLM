"""Custody/adoption for missing-only stored audits; no model operations."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import extent,sha,write


def metadata(p):
    st=Path(p).stat()
    return dict(path=str(Path(p).resolve()),bytes=st.st_size,mtime_ns=st.st_mtime_ns)


def bind(args):
    old=DOC/'original_joint_history_recovery_stored_adjudication_20261010.receipt.json'
    receipt=json.loads(old.read_bytes());log=Path(receipt['log']['path'])
    assert receipt['exit_code']!=0 and receipt['error']=="AssertionError('stored audit deadline')"
    assert receipt['output'] is None and extent(log)==receipt['log']
    for item in receipt['inputs']:assert extent(item['path'])==item
    markers=[json.loads(line) for line in log.read_text().splitlines()]
    assert [(x['arm'],x['audited_counter']) for x in markers]==[('A',33),('A',39),('A',51)]
    assert all(x['seconds']<=receipt['seconds'] for x in markers)
    assert receipt['inputs'][0]['sha256']=='7c88c326ce83a8b479b8b2af401a2fa7c8f71a325dc77e28d6f18b0916cd4ed6'
    assert receipt['inputs'][1]['sha256']=='7dd6ab452c7958b5dce6e4e353e38b541e6424ef8fd5eaa7a40c070ab15d3b08'
    finish=DOC/'original_joint_history_recovery_finish_consumed_binding_20261010.json'
    fb=json.loads(finish.read_bytes())
    result=DOC/'original_joint_history_recovery_finish_result_20261010.json'
    terminal=result.with_suffix('.terminal.json');term=json.loads(terminal.read_bytes())
    assert term['exit_code']==0 and term['result_sha256']==sha(result)
    items={i['path']:i for i in fb['adjudication_inputs']+term['output_files']}
    assert all(metadata(i['path'])['bytes']==i['bytes'] for i in items.values())
    files=[old,log,finish,result,terminal,Path(__file__),
        B/'original_joint_history_recovery_audit.py',B/'original_joint_history_recovery_audit_launch.py',
        B/'original_joint_history_audit_finish.py',B/'original_joint_history_audit_finish_launch.py',
        B/'original_joint_history_audit_finish_build.py',DOC/'ORIGINAL_JOINT_HISTORY_AUDIT_FINISH_PROTOCOL_20261010.md']
    files += [Path(i['path']) for i in receipt['inputs']]
    out=dict(schema='ORIGINAL_JOINT_HISTORY_AUDIT_FINISH_BINDING_V1',prior_receipt=extent(old),
        prior_log=extent(log),adopted_A_markers=markers,prior_hash_phase_pass_derived_from_frozen_control_flow=True,
        original_completion_binding=extent(finish),result=extent(result),terminal=extent(terminal),
        sealed_items=list(items.values()),metadata=[metadata(p) for p in items],
        inputs=[extent(p) for p in dict.fromkeys(files)],
        limits=dict(seconds=4200,science_seconds=1800,seal_seconds=2100,OS_bytes=20<<30,log_bytes=8<<20,output_bytes=16<<20),
        scope='Adopt only A33/39/51 and initial hashes proved by immutable first audit trace; '
            'new B33/39/51, cheap assembly and complete end hashes. No learner/optimizer/native/source/GPU calls. '
            'A error bounds adopted, not invented exact deltas. First audit deadline FAIL retained.')
    write(args.out,out)
    print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out),sealed_items=len(items),
        sealed_bytes=sum(i['bytes'] for i in items.values()),adopted_A=3)),flush=True)


def check(args):
    b=json.loads(args.resume_binding.read_bytes())
    assert b['schema']=='ORIGINAL_JOINT_HISTORY_AUDIT_FINISH_BINDING_V1'
    for i in b['inputs']:assert extent(i['path'])==i
    assert all(metadata(m['path'])==m for m in b['metadata']), 'bound input metadata changed'
    receipt=json.loads(Path(b['prior_receipt']['path']).read_bytes())
    assert receipt['error']=="AssertionError('stored audit deadline')" and receipt['output'] is None
    marks=[json.loads(line) for line in Path(b['prior_log']['path']).read_text().splitlines()]
    assert marks==b['adopted_A_markers']
    assert [(m['arm'],m['audited_counter']) for m in marks]==[('A',33),('A',39),('A',51)]
    assert extent(args.result)==b['result'] and extent(args.binding)==b['original_completion_binding']
    assert not args.out.exists() and not args.directory.exists()
    args.directory.mkdir()
    return b


def adopt_A(m,records,resume):
    assert m['counter'] in (33,39,51)
    marker=next(x for x in resume['adopted_A_markers'] if x['audited_counter']==m['counter'])
    state=dict(arm='A',counter=m['counter'],new_updates=m['new_updates'],masters=92,
        coefficients=721008128,Adam_steps_all=m['counter'],all_master_moments_finite=True,
        all110_export_fields_exact=True,independent_integer_coordinates=18432,
        source_lineage_exact=True,RNG_saved=True,adopted_success_marker=marker,
        provenance=resume['prior_log'],prior_receipt=resume['prior_receipt'])
    expected=Path(m['checkpoint']['path']).with_suffix('.expected.witness').read_bytes()
    rows=m['native']['records']
    metrics=dict(labels=sum(x['labels'] for x in rows),positions=sum(x['history'] for x in rows),
        max_case_KL_delta_upper_bound=1e-10,max_label_KL_delta_upper_bound=1e-10,
        max_mass_defect=max(x['mass_defect'] for x in rows),adopted_success_marker=marker)
    assert len(expected)==18432*4 and len(rows)==24
    return state,expected,rows,metrics


def finish_seal(args,resume,result,started):
    from original_joint_history_recovery_audit import extent as frozen_extent
    assert time.monotonic()-started<=resume['limits']['science_seconds'], 'science phase deadline'
    result.update(schema='ORIGINAL_JOINT_HISTORY_RECOVERY_STORED_ADJUDICATION_FINISH_V1',
        prior_stored_audit_receipt=resume['prior_receipt'],prior_stored_audit_deadline_gate=False,
        adopted_A_milestones=3,new_B_milestones=3,resume_binding=extent(args.resume_binding),
        hash_provenance='Initial full hash traversal adopted from frozen first-audit control flow; '
            'complete end traversal independently repeated before final result.',
        science_seconds=time.monotonic()-started)
    pending=args.directory/'pending_scientific_result.json';write(pending,result)
    phase=time.monotonic();journal=args.directory/'end_hashes.jsonl'
    with journal.open('xb') as f:
        for index,item in enumerate(resume['sealed_items']):
            assert time.monotonic()-phase<=resume['limits']['seal_seconds'], 'seal phase deadline'
            assert frozen_extent(item['path'])==item,item['path']
            row=dict(index=index,extent=item,seconds=time.monotonic()-phase)
            f.write((json.dumps(row)+'\n').encode());f.flush()
            if index%100==0:print(json.dumps(dict(stage='end_hash_seal',files=index+1,seconds=time.monotonic()-phase)),flush=True)
    assert all(metadata(m['path'])==m for m in resume['metadata']), 'input metadata changed during completion'
    for item in resume['inputs']:assert extent(item['path'])==item
    result.update(end_hash_journal=extent(journal),end_hash_count=len(resume['sealed_items']),
        sealed_input_metadata_unchanged=True,seal_seconds=time.monotonic()-phase,
        seconds=time.monotonic()-started,pending_scientific_result=extent(pending))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    bind(parser.parse_args())
