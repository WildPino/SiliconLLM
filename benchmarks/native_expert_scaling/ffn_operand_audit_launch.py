"""Close FFN producer custody and launch frozen full CPU audit with watchdog."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'benchmarks/native_expert_scaling'
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write


def load(path): return json.loads(Path(path).read_bytes())


def prepare(a):
    import psutil
    begin = time.monotonic(); proc = psutil.Process(); assert not a.out.exists()
    rp = DOC/'ffn_operand_capture_result_20261013.json'
    tp,wp,op = [rp.with_suffix(s) for s in ('.terminal.json','.worker_terminal.json','.owned_process.json')]
    r,t,w,o = [load(p) for p in (rp,tp,wp,op)]
    assert t['exit_code'] == w['exit_code'] == t['worker_exit_code'] == 0
    assert t['result'] == w['result'] == extent(rp) and t['worker_terminal'] == extent(wp)
    assert (t['pid'],t['creation_time']) == (o['pid'],o['creation_time'])
    assert (w['pid'],w['creation_time']) == (o['worker_pid'],o['worker_creation_time'])
    for terminal in (t,w):
        if psutil.pid_exists(terminal['pid']):
            assert abs(psutil.Process(terminal['pid']).create_time()-terminal['creation_time']) > .001
    assert extent(r['binding']['path']) == r['binding']; b = load(r['binding']['path'])
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    for e in b['inputs']+r['outputs']:
        assert time.monotonic()-begin <= 1800 and proc.memory_info().peak_wset <= 8 << 30
        assert extent(e['path']) == e,e['path']
    actual = {str(p.resolve()):p.stat().st_size for p in Path(b['output_directory']).iterdir() if p.is_file()}
    assert actual == {e['path']:e['bytes'] for e in r['outputs']}
    assert sum(actual.values()) == r['output_bytes'] <= b['limits']['output_bytes']
    assert len(r['records']) == 48 and r['field_files'] == 6912 and r['FFN_positions'] == 541128
    assert r['fields_payload_bytes'] == b['fields_payload_bytes'] == 24381063168
    assert r['all_parent_native_traces_states_cache_and_label_bits_exact']
    assert r['all411_source_parameters_exact_before_after'] and r['before_after_hashes_exact']
    before,after = [load(Path(b['output_directory'])/name) for name in ('parameter_before.json','parameter_after.json')]
    assert before == after and before['canonical_coefficient_words'] == 1554863488
    assert len(before['tensors']) == 411
    for tensor in before['tensors']:
        spec = b['checkpoint_tensors'][tensor['name']]
        assert tensor['shape'] == spec['shape'] and tensor['canonical_BF16_sha256'] == spec['sha256']
    supplemental = [extent(p) for p in (Path(__file__),DOC/'FFN_OPERAND_AUDIT_LAUNCH_PROTOCOL_20261013.md')]
    binding = dict(schema='FFN_OPERAND_AUDIT_BINDING_V1',freeze=a.freeze,python=b['python'],
        report=extent(rp),terminal=extent(tp),worker_terminal=extent(wp),owned_process=extent(op),
        parent_binding=r['binding'],supplemental_inputs=supplemental,
        worker=str((B/'ffn_operand_capture_audit.py').resolve()),
        output=str((DOC/'ffn_operand_adjudication_20261013.json').resolve()),
        limits=dict(seconds=7500,worker_seconds=b['limits']['audit_seconds'],OS_bytes=8 << 30,log_bytes=128 << 20),
        producer_closed=dict(holder_pid=t['pid'],holder_creation=t['creation_time'],worker_pid=w['pid'],
            worker_creation=w['creation_time'],both_exit_code=0,matching_processes_absent=True,
            all9900_inputs_and8163_outputs_hash_verified=True,output_files=len(actual),output_bytes=sum(actual.values()),
            source_parameter_words=1554863488,worker_seconds=w['seconds'],holder_seconds=t['seconds'],
            OS_peak=t['worker_OS_peak'],GPU_allocated_peak=w['GPU_allocated_peak'],GPU_reserved_peak=w['GPU_reserved_peak']),
        environment=dict(OPENBLAS_NUM_THREADS='6',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES=''),
        preparation_seconds=time.monotonic()-begin,
        scope='Closed qualified-parent-exact FFN operand producer;full CPU audit pending. '
              'No expert fit/new model quality/speed/DRAM/useful-n admission.')
    write(a.out,binding); print(json.dumps(dict(binding=extent(a.out),producer_closed=binding['producer_closed'])),flush=True)


def launch(a):
    import psutil
    from original_falcon_whole_recovery import memory_reader
    begin = time.monotonic(); assert extent(a.binding)['sha256'] == a.binding_sha
    b = load(a.binding); proc = psutil.Process(); proc.cpu_affinity([11]); reader = memory_reader()
    out = Path(b['output']); log = out.with_suffix('.worker.log'); terminal = out.with_suffix('.holder_terminal.json')
    owned = out.with_suffix('.owned_process.json')
    assert not any(p.exists() for p in (out,log,terminal,owned,out.with_suffix('.terminal.json')))
    pins = b['supplemental_inputs']+[b[k] for k in ('report','terminal','worker_terminal','owned_process','parent_binding')]
    def check():
        for e in pins: assert extent(e['path']) == e,e['path']
    check(); env = dict(os.environ,**b['environment'])
    argv = [b['python'],'-I','-S','-B','-X','utf8',b['worker'],'--report',b['report']['path'],
        '--report-sha',b['report']['sha256'],'--terminal',b['terminal']['path'],
        '--terminal-sha',b['terminal']['sha256'],'--freeze',a.freeze,'--out',str(out)]
    child = None; peak = 0; offset = 0
    receipt = dict(schema='FFN_OPERAND_AUDIT_HOLDER_V1',binding=extent(a.binding),command=sys.orig_argv,
        worker_argv=argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,GPU_calls=0)
    try:
        with log.open('xb') as stream:
            child = subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time())
            write(owned,receipt)
            while child.poll() is None:
                peak = max(peak,reader(child))
                assert time.monotonic()-begin <= b['limits']['seconds'],'holder deadline'
                assert peak+proc.memory_info().peak_wset <= b['limits']['OS_bytes'],'combined OS cap'
                assert log.stat().st_size <= b['limits']['log_bytes'],'log cap'
                with log.open('rb') as f:
                    f.seek(offset); data = f.read(); end = data.rfind(b'\n')+1
                    if end: print(data[:end].decode('utf8',errors='replace'),end='',flush=True); offset += end
                time.sleep(.5)
        peak = max(peak,reader(child)); receipt['worker_exit_code'] = child.returncode
        assert child.returncode == 0,log.read_text(encoding='utf8',errors='replace')[-5000:]
        wt = load(out.with_suffix('.terminal.json'))
        assert wt['exit_code'] == 0 and wt['result'] == extent(out)
        assert (wt['pid'],wt['creation_time']) == (receipt['worker_pid'],receipt['worker_creation_time'])
        check(); receipt.update(exit_code=0,error=None,result=extent(out),worker_terminal=extent(out.with_suffix('.terminal.json')))
    except BaseException as error:
        if child is not None and child.poll() is None:
            assert abs(psutil.Process(child.pid).create_time()-receipt['worker_creation_time']) < .001
            child.terminate(); child.wait(timeout=15)
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),
                       worker_exit_code=None if child is None else child.returncode); raise
    finally:
        receipt.update(seconds=time.monotonic()-begin,worker_OS_peak=peak,OS_peak=proc.memory_info().peak_wset)
        write(terminal,receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); m = p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true'); m.add_argument('--launch',action='store_true')
    for k in ('binding','out'): p.add_argument('--'+k,type=Path)
    for k in ('binding-sha','freeze'): p.add_argument('--'+k,required=k == 'freeze')
    a = p.parse_args()
    if a.launch: launch(a)
    else:
        import psutil
        start = time.monotonic(); proc = psutil.Process()
        receipt = dict(schema='FFN_OPERAND_AUDIT_PREPARATION_TERMINAL_V1',command=sys.orig_argv,
                       pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze)
        try: prepare(a); receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as error:
            receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
        finally:
            receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
            write(a.out.with_suffix('.preparation_terminal.json'),receipt)
