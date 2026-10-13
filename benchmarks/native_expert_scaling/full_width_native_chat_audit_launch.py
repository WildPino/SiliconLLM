"""Close producer custody and launch the frozen CPU audit with an owned watchdog."""
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


def load(path):
    return json.loads(Path(path).read_bytes())


def prepare(a):
    import numpy as np
    import psutil
    import tokenizers
    import jinja2
    import markupsafe
    start = time.monotonic(); proc = psutil.Process()
    assert not a.out.exists()
    rp = DOC/'full_width_native_chat_result_20261013.json'
    tp = rp.with_suffix('.terminal.json')
    wp = rp.with_suffix('.worker_terminal.json')
    op = rp.with_suffix('.owned_process.json')
    r,t,w,o = map(load,[rp,tp,wp,op])
    assert t['exit_code'] == w['exit_code'] == t['worker_exit_code'] == 0
    assert t['result'] == w['result'] == extent(rp)
    assert t['worker_terminal'] == extent(wp)
    assert (t['pid'],t['creation_time']) == (o['pid'],o['creation_time'])
    assert (w['pid'],w['creation_time']) == (o['worker_pid'],o['worker_creation_time'])
    for terminal in (t,w):
        if psutil.pid_exists(terminal['pid']):
            assert abs(psutil.Process(terminal['pid']).create_time()-terminal['creation_time']) > .001
    assert extent(r['binding']['path']) == r['binding']
    b = load(r['binding']['path'])
    assert np.__version__ == b['versions']['numpy'] and psutil.__version__ == b['versions']['psutil']
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    for e in b['inputs']+r['outputs']:
        assert time.monotonic()-start <= 1800 and proc.memory_info().peak_wset <= 8 << 30
        assert extent(e['path']) == e, e['path']
    family = Path(b['output_directory'])
    actual = {str(p.resolve()):p.stat().st_size for p in family.iterdir() if p.is_file()}
    expected = {e['path']:e['bytes'] for e in r['outputs']}
    assert actual == expected and sum(actual.values()) == r['output_bytes'] <= b['limits']['output_bytes']
    assert len(r['tasks']) == 16 and len(r['records']) == 48 and len(r['prefix_controls']) == 16
    # Extra current pins strengthen the audit; they do NOT establish historical
    # before/after custody for omitted transitive producer imports.
    names = ['full_width_native_chat_audit.py','row_state.py','token_state.py',
        'full_channel_native.py','composed_operator_mixed.py','falcon_auxiliary_s6_transport.py',
        'full_width_native_chat.py','true_generator_input_capture_v3.py',
        'original_falcon_whole_recovery.py','chatbot_falcon_usability.py',
        'original_engine_chat.py','chatbot_hybrid_engine_chat.py']
    extra = [Path(__file__),DOC/'FULL_WIDTH_NATIVE_CHAT_AUDIT_LAUNCH_PROTOCOL_20261013.md']
    extra += [B/n for n in names]
    for module in (np,psutil,tokenizers,jinja2,markupsafe):
        base = Path(module.__file__).resolve().parent
        extra += [p for p in base.rglob('*') if p.is_file() and p.suffix in ('.py','.pyd','.dll')]
    extras = [extent(p) for p in dict.fromkeys(extra)]
    protected = [e for e in b['inputs'] if Path(e['path']).name in
        ('engine.c','_manifest.json','build_document_holdout.py','RESEARCH_INDEX.md')]
    assert len(protected) == 4
    binding = dict(schema='FULL_WIDTH_NATIVE_CHAT_AUDIT_BINDING_V1',freeze=a.freeze,
        python=b['python'],report=extent(rp),terminal=extent(tp),worker_terminal=extent(wp),
        owned_process=extent(op),parent_binding=r['binding'],
        producer_closed=dict(holder_pid=t['pid'],holder_creation=t['creation_time'],
            worker_pid=w['pid'],worker_creation=w['creation_time'],both_exit_code=0,
            matching_processes_absent=True,output_files=len(actual),output_bytes=sum(actual.values()),
            all_declared_input_and_output_hashes_verified=True,protected=protected,
            worker_seconds=w['seconds'],holder_seconds=t['seconds'],OS_peak=t['worker_OS_peak'],
            GPU_allocated_peak=w['GPU_allocated_peak'],GPU_reserved_peak=w['GPU_reserved_peak']),
        extra_inputs=extras,worker=str((B/'full_width_native_chat_audit.py').resolve()),
        output=str((DOC/'full_width_native_chat_adjudication_20261013.json').resolve()),
        limits=dict(seconds=7500,worker_seconds=b['limits']['audit_seconds'],OS_bytes=8 << 30,
                    ledger_bytes=128 << 20),
        environment=dict(OPENBLAS_NUM_THREADS='6',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
                         HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false'),
        versions=dict(numpy=np.__version__,psutil=psutil.__version__,
            tokenizers=tokenizers.__version__,jinja2=jinja2.__version__),
        historical_pin_limit='Producer pinned188 selected files,not complete transitive import closure. '
            'composed_operator_mixed.py and tokenizer/Jinja runtime bodies were not separately '
            'pinned before/after producer. Current extra pins support this audit only;do not '
            'retroactively assert exhaustive producer runtime custody. Full numeric/output replay '
            'remains required independently of this limitation.',
        preparation_seconds=time.monotonic()-start)
    write(a.out,binding)
    print(json.dumps(dict(binding=extent(a.out),producer_closed=binding['producer_closed'],
                          extra_pins=len(extras))),flush=True)


def launch(a):
    import psutil
    from original_falcon_whole_recovery import memory_reader
    start = time.monotonic()
    assert extent(a.binding)['sha256'] == a.binding_sha
    b = load(a.binding); proc = psutil.Process(); proc.cpu_affinity([11]); reader = memory_reader()
    out = Path(b['output']); log = out.with_suffix('.worker.log'); terminal = out.with_suffix('.holder_terminal.json')
    owned = out.with_suffix('.owned_process.json')
    assert not any(p.exists() for p in [out,log,terminal,owned,out.with_suffix('.terminal.json')])
    pins = b['extra_inputs']+[b[k] for k in ('report','terminal','worker_terminal','owned_process','parent_binding')]
    def check():
        for e in pins: assert extent(e['path']) == e,e['path']
    check()
    env = dict(os.environ,**b['environment'])
    argv = [b['python'],'-I','-S','-B','-X','utf8',b['worker'],
        '--report',b['report']['path'],'--report-sha',b['report']['sha256'],
        '--terminal',b['terminal']['path'],'--terminal-sha',b['terminal']['sha256'],
        '--freeze',a.freeze,'--out',str(out)]
    receipt = dict(schema='FULL_WIDTH_NATIVE_CHAT_AUDIT_HOLDER_V1',binding=extent(a.binding),
        command=sys.orig_argv,worker_argv=argv,pid=proc.pid,creation_time=proc.create_time(),
        freeze=a.freeze,GPU_calls=0)
    child = None; peak = 0; offset = 0
    try:
        with log.open('xb') as stream:
            child = subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time())
            write(owned,receipt)
            while child.poll() is None:
                peak = max(peak,reader(child))
                assert time.monotonic()-start <= b['limits']['seconds'],'holder deadline'
                assert peak+proc.memory_info().peak_wset <= b['limits']['OS_bytes'],'combined OS cap'
                assert log.stat().st_size <= b['limits']['ledger_bytes'],'log cap'
                with log.open('rb') as f:
                    f.seek(offset); data = f.read(); end = data.rfind(b'\n')+1
                    if end:
                        print(data[:end].decode('utf8',errors='replace'),end='',flush=True); offset += end
                time.sleep(.5)
        peak = max(peak,reader(child)); receipt['worker_exit_code'] = child.returncode
        assert child.returncode == 0,log.read_text(encoding='utf8',errors='replace')[-5000:]
        wt = load(out.with_suffix('.terminal.json'))
        assert wt['exit_code'] == 0 and wt['result'] == extent(out)
        assert (wt['pid'],wt['creation_time']) == (receipt['worker_pid'],receipt['worker_creation_time'])
        check()
        receipt.update(exit_code=0,error=None,result=extent(out),worker_terminal=extent(out.with_suffix('.terminal.json')))
    except BaseException as error:
        if child is not None and child.poll() is None:
            assert abs(psutil.Process(child.pid).create_time()-receipt['worker_creation_time']) < .001
            child.terminate(); child.wait(timeout=15)
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),
                       worker_exit_code=None if child is None else child.returncode); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,worker_OS_peak=peak,OS_peak=proc.memory_info().peak_wset)
        write(terminal,receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true'); m.add_argument('--launch',action='store_true')
    for k in ('binding','out'): p.add_argument('--'+k,type=Path)
    for k in ('binding-sha','freeze'): p.add_argument('--'+k,required=k == 'freeze')
    a = p.parse_args()
    if a.launch: launch(a)
    else:
        import psutil
        start = time.monotonic(); proc = psutil.Process()
        receipt = dict(schema='FULL_WIDTH_NATIVE_CHAT_AUDIT_PREPARATION_TERMINAL_V1',command=sys.orig_argv,
                       pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze)
        try:
            prepare(a); receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as error:
            receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
        finally:
            receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
            write(a.out.with_suffix('.preparation_terminal.json'),receipt)
