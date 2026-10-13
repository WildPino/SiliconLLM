"""Close complete failed representation assay and necessary-rank witness."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(ROOT/'benchmarks/native_expert_scaling'),
               str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write


def main():
    import psutil
    rp = DOC/'token_context_rank_result_20261013.json'; ap = DOC/'token_context_rank_adjudication_20261013.json'
    bp = DOC/'token_context_rank_bound_20261013.json'
    r,a,bound = [json.loads(p.read_bytes()) for p in (rp,ap,bp)]
    b = json.loads(Path(r['binding']['path']).read_bytes())
    assert extent(r['binding']['path']) == r['binding'] and a['report'] == extent(rp)
    assert a['decision'] == 'INDEPENDENT_STORED_PASS' and a['all_numeric_pass'] and a['checked_candidates'] == 1440
    assert not a['compact_candidates'] and all(m['selected_rank'] is None for m in r['models'])
    assert bound['report'] == extent(rp) and bound['audit'] == extent(ap)
    assert [row['minimum_necessary_shared_rank'] for row in bound['rows']] == [1408,1857,1857]
    commands = []
    for name in ('token_context_rank_result_20261013.terminal.json',
                 'token_context_rank_adjudication_20261013.terminal.json',
                 'token_context_rank_bound_20261013.terminal.json'):
        p = DOC/name; t = json.loads(p.read_bytes())
        assert t['exit_code'] == 0 and t['result'] == extent(t['result']['path'])
        if psutil.pid_exists(t['pid']): assert abs(psutil.Process(t['pid']).create_time()-t['creation_time']) > .001
        commands.append(dict(terminal=extent(p),pid=t['pid'],creation_time=t['creation_time'],
                             exit_code=0,seconds=t['seconds'],OS_peak=t['OS_peak'],matching_process_absent=True))
    for e in b['inputs']+bound['inputs']+[m['file'] for m in r['models']]: assert extent(e['path']) == e,e['path']
    actual = {str(p.resolve()):p.stat().st_size for p in Path(b['output_directory']).iterdir() if p.is_file()}
    assert actual == {m['file']['path']:m['file']['bytes'] for m in r['models']}
    assert sum(actual.values()) == r['output_bytes'] <= b['limits']['output_bytes']
    out = DOC/'token_context_rank_session_closure_20261013.json'; assert not out.exists()
    closure = dict(schema='TOKEN_CONTEXT_RANK_CLOSURE_V1',source=extent(Path(__file__)),
        report=extent(rp),audit=extent(ap),bound=extent(bp),binding=r['binding'],commands=commands,
        family_bytes=sum(actual.values()),family_files=len(actual),all_inputs_and_outputs_hash_verified=True,
        decision='TOKEN_ONLY_PROTOTYPE_SHARED_COMPACT_LINEAR_CORRECTION_REJECTED',
        goal_status='ACTIVE_INCOMPLETE',
        scope='All1440 source/causal field curves checked;common residual rank<=256 rejected '
              'in this model family. Necessary FIT ranks1408/1857/1857 include prototype/BF '
              'rounding allowances. Does not rule out routed token-dependent operators or '
              'nonlinear/full-rank expert representations;no whole-model new-candidate/speed claim.')
    write(out,closure); print(json.dumps(dict(closure=extent(out),decision=closure['decision'])),flush=True)


if __name__ == '__main__': main()
