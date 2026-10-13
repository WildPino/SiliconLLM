"""Seal qualified FFN operands and source/parent custody."""
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
    rp = DOC/'ffn_operand_capture_result_20261013.json'; ap = DOC/'ffn_operand_adjudication_20261013.json'
    r,a = [json.loads(p.read_bytes()) for p in (rp,ap)]
    b = json.loads(Path(r['binding']['path']).read_bytes())
    abp = DOC/'ffn_operand_audit_binding_20261013.json'; ab = json.loads(abp.read_bytes())
    assert a['decision'] == 'INDEPENDENT_STORED_PASS' and a['all_numeric_pass'] and all(a['numeric_gates'].values())
    assert a['report'] == extent(rp) and a['binding'] == r['binding'] == extent(r['binding']['path'])
    assert a['source'] == extent(a['source']['path']) and a['before_after_hashes_exact']
    assert a['counts']['case_sites'] == 1152 and a['counts']['FFN_positions'] == 541128
    assert a['counts']['field_words'] == 12190531584 and a['counts']['label_words'] == 577249896
    assert all(v == 0 for k,v in a['counts'].items() if 'mismatches' in k)
    commands = []
    for name in ('ffn_operand_capture_binding_v2_20261013.preparation_terminal.json',
        'ffn_operand_capture_result_20261013.terminal.json','ffn_operand_capture_result_20261013.worker_terminal.json',
        'ffn_operand_audit_binding_20261013.preparation_terminal.json','ffn_operand_adjudication_20261013.terminal.json',
        'ffn_operand_adjudication_20261013.holder_terminal.json'):
        p = DOC/name; t = json.loads(p.read_bytes())
        assert t['exit_code'] == 0 and t['result'] == extent(t['result']['path'])
        if psutil.pid_exists(t['pid']): assert abs(psutil.Process(t['pid']).create_time()-t['creation_time']) > .001
        commands.append(dict(terminal=extent(p),pid=t['pid'],creation_time=t['creation_time'],
                             exit_code=0,seconds=t['seconds'],matching_process_absent=True))
    for owned_name,worker_name,holder_name in (
        ('ffn_operand_capture_result_20261013.owned_process.json','ffn_operand_capture_result_20261013.worker_terminal.json',
         'ffn_operand_capture_result_20261013.terminal.json'),
        ('ffn_operand_adjudication_20261013.owned_process.json','ffn_operand_adjudication_20261013.terminal.json',
         'ffn_operand_adjudication_20261013.holder_terminal.json')):
        o,w,h = [json.loads((DOC/n).read_bytes()) for n in (owned_name,worker_name,holder_name)]
        assert (o['pid'],o['creation_time']) == (h['pid'],h['creation_time'])
        assert (o['worker_pid'],o['worker_creation_time']) == (w['pid'],w['creation_time'])
        assert h['worker_exit_code'] == 0 and h['worker_terminal'] == extent(DOC/worker_name)
    for e in b['inputs']+ab['supplemental_inputs']: assert extent(e['path']) == e,e['path']
    actual = {str(p.resolve()):p.stat().st_size for p in Path(b['output_directory']).iterdir() if p.is_file()}
    assert actual == {e['path']:e['bytes'] for e in r['outputs']}
    assert sum(actual.values()) == r['output_bytes'] <= b['limits']['output_bytes']
    prior_failure = DOC/'ffn_operand_capture_binding_20261013.preparation_terminal.json'
    assert json.loads(prior_failure.read_bytes())['exit_code'] == 1
    out = DOC/'ffn_operand_session_closure_20261013.json'; assert not out.exists()
    closure = dict(schema='FFN_OPERAND_CLOSURE_V1',source=extent(Path(__file__)),report=extent(rp),audit=extent(ap),
        binding=r['binding'],audit_binding=extent(abp),commands=commands,failed_preparation=extent(prior_failure),
        new_payload_files=len(actual),new_payload_bytes=sum(actual.values()),
        current_input_hashes_and_output_extent_set_exact=True,
        output_byte_custody='Complete full audit hashes every output before/after.',
        decision='ACTUAL_OWN_ALL24_FFN_OPERANDS_QUALIFIED',goal_status='ACTIVE_INCOMPLETE',
        scope='All24 actual pre-FFN/source nodes/known48 histories qualified;source matrices '
              'within frozen budgets,gate scalar and feature exact,parent label/state bits '
              'exact. Parent native stream/attention controls are producer witnesses. '
              'No affine experts/new model/CPU speed/useful-n/DRAM/other-family admission.')
    write(out,closure); print(json.dumps(dict(closure=extent(out),decision=closure['decision'])),flush=True)


if __name__ == '__main__': main()
