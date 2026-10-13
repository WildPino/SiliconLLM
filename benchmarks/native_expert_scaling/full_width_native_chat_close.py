"""Seal full-width chatbot producer/audit custody and the qualified scope."""
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
    rp = DOC/'full_width_native_chat_result_20261013.json'
    ap = DOC/'full_width_native_chat_adjudication_20261013.json'
    r,a = (json.loads(p.read_bytes()) for p in (rp,ap))
    b = json.loads(Path(r['binding']['path']).read_bytes())
    abp = DOC/'full_width_native_chat_audit_binding_20261013.json'
    ab = json.loads(abp.read_bytes())
    assert a['decision'] == 'INDEPENDENT_STORED_QUALITY_PASS'
    assert a['report'] == extent(rp) and a['binding'] == r['binding'] == extent(r['binding']['path'])
    assert a['source'] == extent(a['source']['path']) and a['quality_pass'] and a['all_numeric_pass']
    assert all(a['numeric_gates'].values()) and all(a['quality']['gates'].values())
    assert a['counts']['sequence_sites'] == 97*24 and a['counts']['native_positions'] == r['processed_tokens']*24
    assert a['counts']['output_words'] == a['counts']['integer_words'] == r['processed_tokens']*24*2048
    assert all(v == 0 for k,v in a['counts'].items() if 'mismatches' in k)
    assert a['quality']['gates'] == r['quality']['gates'] and a['before_after_hashes_exact']
    assert a['GPU_calls'] == a['source_model_forwards'] == 0
    commands = []
    for name in ('full_width_native_chat_binding_20261013.preparation_terminal.json',
                 'full_width_native_chat_result_20261013.terminal.json',
                 'full_width_native_chat_result_20261013.worker_terminal.json',
                 'full_width_native_chat_audit_binding_20261013.preparation_terminal.json',
                 'full_width_native_chat_adjudication_20261013.terminal.json',
                 'full_width_native_chat_adjudication_20261013.holder_terminal.json'):
        p = DOC/name; t = json.loads(p.read_bytes())
        assert t['exit_code'] == 0 and t['result'] == extent(t['result']['path'])
        if psutil.pid_exists(t['pid']):
            assert abs(psutil.Process(t['pid']).create_time()-t['creation_time']) > .001
        commands.append(dict(terminal=extent(p),pid=t['pid'],creation_time=t['creation_time'],
                             exit_code=0,seconds=t['seconds'],matching_process_absent=True))
    for owned_name,worker_name,holder_name in (
        ('full_width_native_chat_result_20261013.owned_process.json',
         'full_width_native_chat_result_20261013.worker_terminal.json',
         'full_width_native_chat_result_20261013.terminal.json'),
        ('full_width_native_chat_adjudication_20261013.owned_process.json',
         'full_width_native_chat_adjudication_20261013.terminal.json',
         'full_width_native_chat_adjudication_20261013.holder_terminal.json')):
        o,w,h = [json.loads((DOC/n).read_bytes()) for n in (owned_name,worker_name,holder_name)]
        assert (o['pid'],o['creation_time']) == (h['pid'],h['creation_time'])
        assert (o['worker_pid'],o['worker_creation_time']) == (w['pid'],w['creation_time'])
        assert h['worker_exit_code'] == 0 and h['worker_terminal'] == extent(DOC/worker_name)
    for e in b['inputs']+ab['extra_inputs']:
        assert extent(e['path']) == e,e['path']
    actual = {str(p.resolve()):p.stat().st_size for p in Path(b['output_directory']).iterdir() if p.is_file()}
    assert actual == {e['path']:e['bytes'] for e in r['outputs']}
    assert sum(actual.values()) == r['output_bytes'] <= b['limits']['output_bytes']
    # Full output bytes were hashed before/after in the qualified audit; closure
    # checks current file extents and retains the audit's sealed hash witness.
    logs = [extent(DOC/n) for n in (
        'full_width_native_chat_result_20261013.worker.log',
        'full_width_native_chat_result_20261013.holder_stdout.log',
        'full_width_native_chat_result_20261013.holder_stderr.log',
        'full_width_native_chat_adjudication_20261013.worker.log',
        'full_width_native_chat_adjudication_20261013.holder_stdout.log',
        'full_width_native_chat_adjudication_20261013.holder_stderr.log')]
    closure = dict(schema='FULL_WIDTH_NATIVE_CHAT_CLOSURE_V1',source=extent(Path(__file__)),
        report=extent(rp),adjudication=extent(ap),binding=r['binding'],audit_binding=extent(abp),
        commands=commands,logs=logs,new_payload_files=len(actual),new_payload_bytes=sum(actual.values()),
        all_declared_inputs_and_current_extra_pins_unchanged=True,
        current_output_extent_set_exact=True,output_hash_custody='All bytes verified before/after by complete audit.',
        historical_pin_limit=ab['historical_pin_limit'],
        decision='KNOWN_CORPUS_FULL_WIDTH_NATIVE_BACKBONE_QUALIFIED',
        goal_status='ACTIVE_INCOMPLETE',
        scope='Known16 chat tasks/own-answer followup/FIT24+DEV24/all24 native states and '
              'readouts qualified with source GPU attention/FFN/head identity witnesses. '
              'No RESERVED/general donor/CPU engine/conditional useful-n/DRAM/same useful50 admission.')
    out = DOC/'full_width_native_chat_session_closure_20261013.json'
    assert not out.exists(); write(out,closure)
    print(json.dumps(dict(closure=extent(out),decision=closure['decision'],commands=commands)),flush=True)


if __name__ == '__main__': main()
