"""Narrow posthoc adjudication of a completed numerical audit's isolation fault.

Never rewrites the failed holder, claims clean timing, or replays mathematics.
Storage after-check completion is proved from the bound holder control flow.
"""
import argparse,ast,hashlib,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'


def extent(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def classify(terminal,result,binding,source):
    assert terminal['schema']=='RECOVERED_CATEGORICAL_CAMPAIGN_HOLDER_V4'
    assert result['schema']=='RECOVERED_CATEGORICAL_CAMPAIGN_AUDIT_V4'
    assert terminal['exit_code']==0 and terminal['error'] is not None
    assert terminal['error']['stage']=='input_hash_after'
    error=terminal['error']['error']
    assert error.startswith('AssertionError(') and error.endswith(')')
    fault=ast.literal_eval(error[len('AssertionError('):-1])
    assert len(fault)==4 and fault[0]=='overlap' and fault[2]=='python.exe'
    argv=fault[3];assert len(argv)==2
    assert Path(argv[0]).resolve()==Path('C:/Users/giosa/AppData/Local/Programs/Python/Python311/python.exe').resolve()
    assert Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve()
    assert 'line 121, in launch' in terminal['error']['traceback']
    assert 'quiet(proc);guard()' in terminal['error']['traceback']
    proof="""        assert child.returncode==0,('worker_exit',child.returncode)
        stage='input_hash_after';check_inputs()
        assert {str(p.resolve()) for p in a.directory.rglob('*') if p.is_file()}==set(signature)
        assert all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in signature.items())
        quiet(proc);guard()
"""
    assert source.count(proof)==1
    assert source.splitlines()[120].strip()=='quiet(proc);guard()'
    assert not terminal['unexpected_child_observations']
    lim=binding['limits']
    assert 0<terminal['elapsed_seconds']<=lim['seconds']
    assert terminal['worker_OS_peak_through_exit']+terminal['launcher_OS_peak_snapshot']<=lim['OS_bytes']
    assert terminal['log']['bytes']<=lim['log_bytes']
    assert not result['original_successful_custody'] and not result['complete_original_campaign']
    assert not result['quality_admission'] and not result['speed_admission']
    assert all(result[k] for k in ('complete_input_output_hashes_at_recovery_boundary',
        'independent_integer_byte_transitions','independent_fsum_all_proposals',
        'full_original_packing_all48_trials','all24_completed_gradients_and_bridges','all48_final_forced_cases'))
    assert result['stored_counts']==dict(completed_steps=24,completed_trials=48,accepted_displacements=5,
        completed_native_prefix_receipts=115,milestone_ordinals=[6,12,18,24])
    assert len(result['audited'])==24 and all(x['completed_direction'] and x['trials']==2 for x in result['audited'])
    assert all(result[k]==0 for k in ('history_forwards','backward_calls','native_calls','GPU_calls','optimizer_updates'))
    assert all(math.isfinite(result[k]) for k in ('seconds','max_metric_delta','max_group_relative_error','max_proposal_ULP'))
    return dict(pid=fault[1],name=fault[2],command=argv)


def main(out):
    start=time.monotonic();base=DOC/'recovered_categorical_campaign_stored_adjudication_v4_20261011'
    paths=[Path(str(base)+suffix) for suffix in ('.json','.terminal.json','.worker.log')]
    report,terminal=json.loads(paths[0].read_bytes()),json.loads(paths[1].read_bytes())
    bp=Path(terminal['binding']['path']);binding=json.loads(bp.read_bytes())
    assert extent(bp)==terminal['binding']==report['binding']
    assert terminal['binding_sha256']==extent(bp)['sha256']
    assert extent(binding['holder']['path'])==binding['holder']
    assert extent(binding['worker']['path'])==binding['worker']
    assert extent(terminal['log']['path'])==terminal['log']
    assert paths[0].stat().st_size<=binding['limits']['output_bytes']
    assert paths[0].stat().st_mtime<=paths[1].stat().st_mtime
    source=Path(binding['holder']['path']).read_text(encoding='utf-8')
    external=classify(terminal,report,binding,source)
    original=json.loads(Path(binding['original_numeric_binding']['path']).read_bytes())
    assert extent(binding['original_numeric_binding']['path'])==binding['original_numeric_binding']
    assert report['max_metric_delta']<=original['numeric']['metric_absolute']
    assert report['max_group_relative_error']<=original['numeric']['group_relative']
    assert report['max_proposal_ULP']<=original['numeric']['proposal_max_F32_ULP']
    lines=[json.loads(line) for line in paths[2].read_text().splitlines()]
    assert [x['audited_case'] for x in lines if 'audited_case' in x]==original['FIT_order']
    last=json.loads((ROOT/'results/native_expert_scaling/original_categorical_campaign_20261010/step_24/step.json').read_bytes())
    assert report['final_packed']==last['selected_packed'] and report['final_parameters']==last['selected_parameters']
    rejected=[]
    for label,change in [('earlier_stage',dict(stage='worker_running')),
                         ('changed_input',dict(error="AssertionError(('changed_input','input_hash_after','x'))")),
                         ('unknown_process',dict(error="AssertionError(('overlap',1,'python.exe',['x','unrelated.py']))")),
                         ('different_traceback',dict(traceback='unknown'))]:
        changed=dict(terminal,error={**terminal['error'],**change})
        try:classify(changed,report,binding,source)
        except (AssertionError,ValueError):rejected.append(label)
        else:raise RuntimeError(('Unrecognized failure admitted',label))
    assert len(rejected)==4 and not out.exists()
    result=dict(schema='RECOVERED_CATEGORICAL_CLOSURE_ADJUDICATION_V1',
        decision='MATH_AND_STORAGE_COMPLETE_ISOLATION_FAILED',
        source=extent(__file__),binding=extent(bp),report=extent(paths[0]),terminal=extent(paths[1]),log=extent(paths[2]),
        holder_source=binding['holder'],worker_source=binding['worker'],original_freeze=terminal['original_freeze'],
        original_holder_success=False,observed_unified_holder_session_exit=1,worker_exit=0,
        input_output_hashes_before_after_proven_by_bound_control_flow=True,
        tree_membership_size_mtime_before_after_proven_by_bound_control_flow=True,
        complete_numeric_worker=True,full_hash_or_math_replay=False,
        process_isolation_failed=True,known_external_publisher=external,timing_control_admitted=False,
        original_successful_custody=False,complete_original_campaign=False,
        quality_admission=False,speed_admission=False,unknown_original_measurements=report['unknown_original_measurements'],
        rejected_failure_fixtures=rejected,seconds=time.monotonic()-start,
        scope='Posthoc classification of the ONLY late known-publisher isolation fault. '
              'Failed holder remains failed. Bound source and traceback prove full after-checks completed; '
              'numerical worker report retained, no clean timing or original capture qualification.')
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
