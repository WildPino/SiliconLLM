"""Close the failed affine representation and complete nonlinear defect audit."""
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(ROOT/'benchmarks/native_expert_scaling'),
               str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write


def main():
    import psutil
    start = time.monotonic()
    stems = ['ffn_affine_pilot_result','ffn_affine_pilot_adjudication',
             'ffn_curvature_split','ffn_curvature_split_adjudication']
    paths = [DOC/(s+'_20261013.json') for s in stems]
    r,a,d,da = [json.loads(p.read_bytes()) for p in paths]
    b = json.loads(Path(r['binding']['path']).read_bytes())
    assert a['report'] == extent(paths[0]) and a['checked_predictions'] == 432
    assert a['decision'] == da['decision'] == 'INDEPENDENT_STORED_PASS'
    assert a['all_numeric_pass'] and a['proxy_controls_pass'] and da['all_numeric_pass']
    assert all(not m['FIT_eligible'] and m['FIT_selected_count'] is None for m in r['models'])
    assert d['parent_report'] == extent(paths[0]) and d['parent_audit'] == extent(paths[1])
    assert da['report'] == extent(paths[2]) and da['checked_cases'] == 144
    commands = []
    for path in paths:
        tp = path.with_suffix('.terminal.json'); t = json.loads(tp.read_bytes())
        assert t['exit_code'] == 0 and t['error'] is None and t['result'] == extent(path)
        if psutil.pid_exists(t['pid']):
            assert abs(psutil.Process(t['pid']).create_time()-t['creation_time']) > .001
        commands.append(dict(terminal=extent(tp),pid=t['pid'],creation_time=t['creation_time'],
            exit_code=0,seconds=t['seconds'],OS_peak=t['OS_peak'],matching_process_absent=True))
    pins = b['inputs']+[r['binding']]+[x['file'] for x in r['models']+r['records']]+d['inputs']+da['inputs']
    unique = {}
    for e in pins:
        if e['path'] in unique: assert unique[e['path']] == e
        unique[e['path']] = e
    for e in unique.values(): assert extent(e['path']) == e,e['path']
    expected = {x['file']['path']:x['file']['bytes'] for x in r['models']+r['records']}
    actual = {str(p.resolve()):p.stat().st_size for p in Path(b['output_directory']).iterdir() if p.is_file()}
    assert actual == expected and len(actual) == 441
    assert sum(actual.values()) == r['output_bytes'] <= b['limits']['output_bytes']
    protected = {
        'benchmarks/phase60/engine.c':'5f948fc0dcd28b1647a2a2c73067bdc0a76a7d41ae7f34395ede8120aeaa83ce',
        'benchmarks/donor_adaptation/configs/_manifest.json':'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
        'benchmarks/donor_adaptation/density/build_document_holdout.py':'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
        'docs/research/RESEARCH_INDEX.md':'99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273',
        'benchmarks/donor_adaptation/nibble_pack.h':'915ffd79dd1db0b2d3dc35fb79bfb7ae41da44dbbc062daf2c31f3f9e0001d23'}
    preserved = []
    for name,sha in protected.items():
        e = extent(ROOT/name); assert e['sha256'] == sha,name; preserved.append(e)
    out = DOC/'ffn_affine_defect_session_closure_20261013.json'; assert not out.exists()
    result = dict(schema='FFN_AFFINE_DEFECT_CLOSURE_V1',source=extent(Path(__file__)),
        reports=[extent(p) for p in paths],binding=r['binding'],commands=commands,
        unique_input_and_output_pins_checked=len(unique),all_inputs_and_outputs_hash_verified=True,
        family_files=len(actual),family_bytes=sum(actual.values()),protected=preserved,
        decision='AFFINE_1_16_64_REJECTED_NONLINEAR_DEFECT_QUALIFIED',goal_status='ACTIVE_INCOMPLETE',
        seconds=time.monotonic()-start,
        scope='All432 affine predictions and144 diagnostic cases fully audited. Frozen '
              'local eligibility failure retained. No universal affine-cell impossibility, '
              'new whole-model candidate, useful expert scaling, final CPU or same useful50 claim.')
    write(out,result); print(json.dumps(dict(closure=extent(out),decision=result['decision'])),flush=True)


if __name__ == '__main__':
    import psutil
    start = time.monotonic(); proc = psutil.Process()
    out = DOC/'ffn_affine_defect_session_closure_20261013.json'
    receipt = dict(schema='FFN_AFFINE_DEFECT_CLOSURE_TERMINAL_V1',command=sys.orig_argv,
                   pid=proc.pid,creation_time=proc.create_time())
    try: main(); receipt.update(exit_code=0,error=None,result=extent(out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(out.with_suffix('.terminal.json'),receipt)
