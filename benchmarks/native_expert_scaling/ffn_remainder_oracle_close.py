"""Close the full-source per-token diagnostic without admitting a router."""
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
    rp,ap = [DOC/(stem+'_20261013.json') for stem in ('ffn_remainder_oracle_result','ffn_remainder_oracle_adjudication')]
    r,a = [json.loads(p.read_bytes()) for p in (rp,ap)]
    b = json.loads(Path(r['binding']['path']).read_bytes())
    assert a['report'] == extent(rp) and a['all_numeric_pass'] and a['checked_predictions'] == 720
    assert a['decision'] == 'INDEPENDENT_STORED_PASS_DIAGNOSTIC_ONLY' and a['literal_selected_IDs_exact']
    commands = []
    for p,suffix in ((Path(r['binding']['path']),'.preparation_terminal.json'),(rp,'.terminal.json'),(ap,'.terminal.json')):
        tp = p.with_suffix(suffix); t = json.loads(tp.read_bytes())
        assert t['exit_code'] == 0 and t['error'] is None and t['result'] == extent(p)
        if psutil.pid_exists(t['pid']): assert abs(psutil.Process(t['pid']).create_time()-t['creation_time']) > .001
        commands.append(dict(terminal=extent(tp),pid=t['pid'],creation_time=t['creation_time'],exit_code=0,
                             seconds=t['seconds'],OS_peak=t['OS_peak'],matching_process_absent=True))
    for e in b['inputs']+[r['binding']]+[x['file'] for x in r['records']]: assert extent(e['path']) == e,e['path']
    actual = {str(p.resolve()):p.stat().st_size for p in Path(b['output_directory']).iterdir() if p.is_file()}
    assert actual == {x['file']['path']:x['file']['bytes'] for x in r['records']}
    assert len(actual) == 720 and sum(actual.values()) == r['output_bytes'] <= b['limits']['output_bytes']
    previous = json.loads(Path(b['parent_closure']['path']).read_bytes())
    for e in previous['protected']: assert extent(e['path']) == e,e['path']
    out = DOC/'ffn_remainder_oracle_session_closure_20261013.json'; assert not out.exists()
    result = dict(schema='FFN_REMAINDER_ORACLE_CLOSURE_V1',source=extent(Path(__file__)),report=extent(rp),audit=extent(ap),
        binding=r['binding'],commands=commands,protected=previous['protected'],all_inputs_and_outputs_hash_verified=True,
        family_files=len(actual),family_bytes=sum(actual.values()),decision='PER_TOKEN_IMPORTANCE_DIAGNOSTIC_QUALIFIED',
        candidate_admitted=False,goal_status='ACTIVE_INCOMPLETE',scope=b['scope'])
    write(out,result); print(json.dumps(dict(closure=extent(out),decision=result['decision'])),flush=True)


if __name__ == '__main__':
    import psutil
    start = time.monotonic(); proc = psutil.Process(); out = DOC/'ffn_remainder_oracle_session_closure_20261013.json'
    receipt = dict(schema='FFN_REMAINDER_ORACLE_CLOSURE_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try: main(); receipt.update(exit_code=0,error=None,result=extent(out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(out.with_suffix('.terminal.json'),receipt)
