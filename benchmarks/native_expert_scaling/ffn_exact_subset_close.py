"""Seal subset coverage assay after complete independent audit."""
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
    rp,ap = [DOC/(stem+'_20261013.json') for stem in ('ffn_exact_subset_result','ffn_exact_subset_adjudication')]
    r,a = [json.loads(p.read_bytes()) for p in (rp,ap)]
    b = json.loads(Path(r['binding']['path']).read_bytes())
    assert a['report'] == extent(rp) and a['decision'] == 'INDEPENDENT_STORED_PASS' and a['all_numeric_pass']
    assert a['checked_predictions'] == 720
    commands = []
    for p,suffix in ((Path(r['binding']['path']),'.preparation_terminal.json'),(rp,'.terminal.json'),(ap,'.terminal.json')):
        t = json.loads(p.with_suffix(suffix).read_bytes())
        assert t['exit_code'] == 0 and t['error'] is None and t['result'] == extent(p)
        if psutil.pid_exists(t['pid']): assert abs(psutil.Process(t['pid']).create_time()-t['creation_time']) > .001
        commands.append(dict(terminal=extent(p.with_suffix(suffix)),pid=t['pid'],creation_time=t['creation_time'],
            seconds=t['seconds'],OS_peak=t['OS_peak'],exit_code=0,matching_process_absent=True))
    for e in b['inputs']+[r['binding']]+[x['file'] for x in r['models']+r['records']]: assert extent(e['path']) == e,e['path']
    actual = {str(p.resolve()):p.stat().st_size for p in Path(b['output_directory']).iterdir() if p.is_file()}
    assert actual == {x['file']['path']:x['file']['bytes'] for x in r['models']+r['records']}
    assert len(actual) == 723 and sum(actual.values()) == r['output_bytes'] <= b['limits']['output_bytes']
    previous = json.loads((DOC/'ffn_affine_defect_session_closure_20261013.json').read_bytes())
    for e in previous['protected']: assert extent(e['path']) == e,e['path']
    out = DOC/'ffn_exact_subset_session_closure_20261013.json'; assert not out.exists()
    result = dict(schema='FFN_EXACT_SUBSET_CLOSURE_V1',source=extent(Path(__file__)),report=extent(rp),audit=extent(ap),
        binding=r['binding'],commands=commands,protected=previous['protected'],all_inputs_and_outputs_hash_verified=True,
        family_files=len(actual),family_bytes=sum(actual.values()),decision='EXACT_SUBSET_COVERAGE_ASSAY_QUALIFIED',
        selections=[dict(site=m['site'],FIT_selected_count=m['FIT_selected_count']) for m in r['models']],
        goal_status='ACTIVE_INCOMPLETE',scope='Local coverage audit only;no physical ternary '
        'candidate,own-model substitution,useful-n/DRAM/final CPU or same useful50 admission.')
    write(out,result); print(json.dumps(dict(closure=extent(out),selections=result['selections'])),flush=True)


if __name__ == '__main__': main()
