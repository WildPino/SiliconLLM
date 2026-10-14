"""Close historical CPU audit with explicit, nonexecuted GPU-tool drift."""
import ast
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
B = ROOT/'benchmarks/native_expert_scaling'
sys.path[:0] = [str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write


def main():
    import psutil
    rp,ap = [DOC/(name+'_20261013.json') for name in ('ffn_sketch_router_result','ffn_sketch_router_adjudication')]
    r,a = [json.loads(p.read_bytes()) for p in (rp,ap)]
    b = json.loads(Path(r['binding']['path']).read_bytes())
    diff_path = DOC/'ffn_sketch_router_current_pin_diff_20261014.json'
    diff = json.loads(diff_path.read_bytes())
    assert diff['report'] == extent(rp) and diff['binding'] == r['binding']
    allowed = {str(Path('C:/Windows/System32')/name).lower() for name in
               ('nvidia-smi.exe','nvcuda.dll','nvml.dll','nvapi64.dll')}
    changes = {e['expected']['path'].lower():e for e in diff['changes']}
    assert set(changes) == allowed and len(diff['changes']) == 4
    for change in changes.values():
        assert extent(change['expected']['path']) == change['current'] and change['expected'] in b['inputs']
    assert a['all_numeric_pass'] and a['decision'] == 'INDEPENDENT_STORED_PASS' and a['checked_predictions'] == 576
    assert a['report'] == extent(rp) and a['literal_selected_IDs_exact']
    # These CPU numerical modules neither import subprocess nor mention the
    # executable. The shared helper only reads/hash-pins;its tensor reader
    # imports no GPU framework or executable runner.
    reviewed = []
    for name in ('ffn_sketch_router.py','ffn_sketch_router_audit.py','falcon_auxiliary_s6_transport.py'):
        path = B/name; source = path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports.update(x.name.split('.')[0] for x in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module: imports.add(node.module.split('.')[0])
        assert not {'subprocess','torch','transformers','cupy'} & imports
        assert 'nvidia-smi' not in source.lower()
        reviewed.append(dict(file=extent(path),imports=sorted(imports)))
    pins = b['inputs']+[r['binding']]+[x['file'] for x in r['models']+r['records']]
    assert diff['checked'] == len(pins)
    for e in pins:
        change = changes.get(e['path'].lower())
        expected = change['current'] if change is not None else e
        assert extent(e['path']) == expected,e['path']
    commands = []
    for p,suffix in ((Path(r['binding']['path']),'.preparation_terminal.json'),(rp,'.terminal.json'),(ap,'.terminal.json')):
        tp = p.with_suffix(suffix); t = json.loads(tp.read_bytes())
        assert t['exit_code'] == 0 and t['error'] is None and t['result'] == extent(p)
        if psutil.pid_exists(t['pid']): assert abs(psutil.Process(t['pid']).create_time()-t['creation_time']) > .001
        commands.append(dict(terminal=extent(tp),pid=t['pid'],creation_time=t['creation_time'],exit_code=0,
            seconds=t['seconds'],OS_peak=t['OS_peak'],matching_process_absent=True))
    actual = {str(p.resolve()):p.stat().st_size for p in Path(b['output_directory']).iterdir() if p.is_file()}
    assert actual == {x['file']['path']:x['file']['bytes'] for x in r['models']+r['records']}
    assert len(actual) == 579 and sum(actual.values()) == r['output_bytes'] <= b['limits']['output_bytes']
    old = json.loads(Path(b['parent_closure']['path']).read_bytes())
    for e in old['protected']: assert extent(e['path']) == e,e['path']
    duplicate = DOC/'ffn_sketch_router_duplicate_attempt_20261014.json'
    d = json.loads(duplicate.read_bytes()); assert d['exit_code'] == 1 and d['successful_terminal'] == extent(ap.with_suffix('.terminal.json'))
    assert not psutil.pid_exists(d['observed_pid'])
    failed = DOC/'ffn_sketch_router_session_closure_20261013.terminal.json'
    ft = json.loads(failed.read_bytes()); assert ft['exit_code'] == 1 and 'nvidia-smi.exe' in ft['error']
    if psutil.pid_exists(ft['pid']): assert abs(psutil.Process(ft['pid']).create_time()-ft['creation_time']) > .001
    out = DOC/'ffn_sketch_router_session_closure_environment_20261014.json'; assert not out.exists()
    result = dict(schema='FFN_SKETCH_ROUTER_ENVIRONMENT_CLOSURE_V1',source=extent(Path(__file__)),report=extent(rp),audit=extent(ap),
        binding=r['binding'],commands=commands,protected=old['protected'],pin_inventory=extent(diff_path),incidental_GPU_component_drift=diff['changes'],
        CPU_module_review=reviewed,all_historical_pins_current_exact=False,all_CPU_numerical_and_artifact_pins_current_exact=True,
        original_current_closure_failure=extent(failed),duplicate_observation=extent(duplicate),
        family_files=len(actual),family_bytes=sum(actual.values()),FIT_selections=[dict(site=m['site'],count=m['count'],rank=m['FIT_selected_rank']) for m in r['models']],
        decision='HISTORICAL_CPU_SKETCH_AUDIT_QUALIFIED_WITH_EXPLICIT_UNUSED_TOOL_DRIFT',goal_status='ACTIVE_INCOMPLETE',
        scope='Original producer/auditor before-after hashes and numeric PASS retained. Current full inventory '
              'differs only at NVIDIA CLI/driver components not used by these NumPy CPU algorithms. No unchanged-current-GPU environment or GPU replay '
              'claim;future GPU work requires fresh binding. No gate/candidate/data changed;no physical/model/TPS admission.')
    write(out,result); print(json.dumps(dict(closure=extent(out),decision=result['decision'])),flush=True)


if __name__ == '__main__':
    import psutil
    start = time.monotonic(); proc = psutil.Process(); out = DOC/'ffn_sketch_router_session_closure_environment_20261014.json'
    receipt = dict(schema='SKETCH_ENVIRONMENT_CLOSURE_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try: main(); receipt.update(exit_code=0,error=None,result=extent(out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(out.with_suffix('.terminal.json'),receipt)
