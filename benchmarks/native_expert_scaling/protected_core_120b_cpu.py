"""Controller for the synthetic gpt-oss-120b protected-projection lower screen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_DYNAMIC'] = 'FALSE'
ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path.insert(0, str(SITE))
SOURCE = Path(__file__).with_suffix('.c')
HEADER = ROOT / 'benchmarks/phase60/strat01_q4k_q8k.h'
BINDING_HELPER = ROOT / 'benchmarks/native_expert_scaling/meth388_switch_thread_binding.h'
M374_PROFILE = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925/meth374_switch_physical_workers_contract_result.json'
PSUTIL_PY = SITE / 'psutil/__init__.py'
PSUTIL_NATIVE = SITE / 'psutil/_psutil_windows.pyd'
DOC = ROOT / 'docs/research/PIPELINE_REASSESSMENT_20261015'
PROTOCOL = DOC / 'PROTECTED_CORE_120B_CPU_PROTOCOL_20261015.md'
META = DOC / 'large_family_metadata'
CONFIG = META / 'config120.json'
REFERENCE = META / 'reference_model.py'
INDEX = META / 'index120.json'
CONFIG_SHA = '933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566'
REFERENCE_SHA = '319ba3c6e4d827eb3115fe4838942af448376b513c02cea4b6b471bf6c21a346'
TOOLCHAIN = Path(r'C:\Users\giosa\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20251216-msvcrt-x86_64\bin\clang.exe')
LIBOMP = TOOLCHAIN.parent / 'libomp.dll'
OUT = ROOT / 'results/native_expert_scaling/protected_core_120b_cpu_20261015'
LIMIT_SECONDS = 600
LIMIT_OS_BYTES = 4 << 30
LIMIT_OUTPUT_BYTES = 128 << 20
CREATED_OUTPUT = False


def descriptor(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def frozen(path, revision):
    blob = subprocess.check_output(['git', 'show', revision + ':' + path.relative_to(ROOT).as_posix()], cwd=ROOT)
    assert blob.replace(b'\r\n', b'\n') == path.read_bytes().replace(b'\r\n', b'\n')


def guard(process, child, began):
    import psutil
    assert time.monotonic() - began < LIMIT_SECONDS, 'deadline'
    peak = process.memory_info().peak_wset
    if child and child.poll() is None:
        try:
            peak += psutil.Process(child.pid).memory_info().peak_wset
        except psutil.NoSuchProcess:
            pass
    assert peak < LIMIT_OS_BYTES, 'combined OS memory cap'
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) < LIMIT_OUTPUT_BYTES, 'output cap'
    return peak


def run(args, process, began):
    global CREATED_OUTPUT
    import psutil
    assert not OUT.exists(), 'exclusive output directory exists'
    OUT.mkdir()
    CREATED_OUTPUT = True
    process.cpu_affinity([0, 2, 4])
    for path in (Path(__file__), SOURCE, PROTOCOL, HEADER, BINDING_HELPER,
                 M374_PROFILE, CONFIG, REFERENCE):
        frozen(path, args.freeze)
    pins = [descriptor(p) for p in (Path(__file__), SOURCE, PROTOCOL, HEADER, BINDING_HELPER,
                                   M374_PROFILE, CONFIG, REFERENCE, INDEX, TOOLCHAIN,
                                   LIBOMP, PSUTIL_PY, PSUTIL_NATIVE)]
    psutil_version = psutil.__version__
    assert psutil.__file__ and Path(psutil.__file__).resolve() == PSUTIL_PY.resolve()
    assert psutil._psutil_windows.__file__ and Path(psutil._psutil_windows.__file__).resolve() == PSUTIL_NATIVE.resolve()
    assert descriptor(CONFIG)['sha256'] == CONFIG_SHA
    assert descriptor(REFERENCE)['sha256'] == REFERENCE_SHA
    config = json.loads(CONFIG.read_bytes())
    assert (config['num_hidden_layers'], config['hidden_size'], config['num_attention_heads'],
            config['num_key_value_heads'], config['head_dim'], config['num_local_experts'],
            config['experts_per_token'], config['sliding_window']) == (36, 2880, 64, 8, 64, 128, 4, 128)
    assert config['layer_types'] == ['sliding_attention' if i % 2 == 0 else 'full_attention' for i in range(36)]
    assert all(descriptor(p['path']) == p for p in pins)
    owned = dict(schema='PROTECTED_CORE_120B_OWNERSHIP_V1', command=sys.orig_argv,
                 pid=process.pid, creation_time=process.create_time(), freeze=args.freeze,
                 pins=pins, limits=dict(seconds=LIMIT_SECONDS, OS_bytes=LIMIT_OS_BYTES, output_bytes=LIMIT_OUTPUT_BYTES))
    write(OUT / 'owned_process.json', owned)
    profile = json.loads(M374_PROFILE.read_bytes())
    runtime_environment = profile['runtime_environment']
    assert runtime_environment == dict(OMP_WAIT_POLICY='ACTIVE', KMP_AFFINITY='none',
        KMP_BLOCKTIME='infinite', OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1')
    copied_dll = OUT / 'libomp.dll'
    shutil.copyfile(LIBOMP, copied_dll)
    assert descriptor(copied_dll)['sha256'] == descriptor(LIBOMP)['sha256']
    compiler_version = subprocess.check_output([str(TOOLCHAIN), '--version'], text=True, timeout=10)
    exe = OUT / 'protected_core_120b_cpu.exe'
    build = [str(TOOLCHAIN), '-O3', '-std=c11', '-mavx2', '-mfma', '-fopenmp',
             str(SOURCE), '-o', str(exe), '-lm']
    compile_log = OUT / 'compile.log'
    with compile_log.open('xb') as log:
        subprocess.run(build, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       timeout=min(120, max(1, LIMIT_SECONDS-(time.monotonic()-began))), check=True)
    assert exe.exists()
    binary = descriptor(exe)
    runs = []
    for run_index in range(3):
        threads = 3
        process.cpu_affinity([0, 2, 4])
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('OMP_', 'KMP_', 'GOMP_', 'SILICON_WORKER_BINDING_'))}
        env.update(runtime_environment)
        env['OMP_NUM_THREADS'] = '3'
        argv = [str(exe), str(threads), str(run_index)]
        log_path = OUT / f'physical_024_run_{run_index}.jsonl'
        with log_path.open('xb') as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     creationflags=0x08000000)
            birth = None
            peak = 0
            try:
                worker = psutil.Process(child.pid)
                birth = worker.create_time()
                worker.cpu_affinity([0, 2, 4])
                while child.poll() is None:
                    peak = max(peak, guard(process, child, began))
                    time.sleep(.1)
            except BaseException as error:
                if child.poll() is None:
                    if birth is not None:
                        assert abs(psutil.Process(child.pid).create_time()-birth) < .001
                    child.terminate()
                    child.wait(timeout=10)
                log.flush()
                failure = OUT / 'first_child_failure.json'
                if not failure.exists():
                    write(failure, dict(run=run_index, pid=child.pid, creation_time=birth,
                        exit_code=child.returncode, error=repr(error), log=descriptor(log_path)))
                raise
        if child.returncode != 0:
            failure = OUT / 'first_child_failure.json'
            if not failure.exists():
                write(failure, dict(run=run_index, pid=child.pid, creation_time=birth,
                    exit_code=child.returncode, error='nonzero worker exit', log=descriptor(log_path)))
            raise RuntimeError(f'worker failed: {log_path} exit={child.returncode}')
        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        controls = [r for r in rows if r['kind'] == 'control']
        timed = [r for r in rows if r['kind'] == 'timed']
        ends = [r for r in rows if r['kind'] == 'end']
        assert len(controls) == 1 and controls[0]['threads'] == threads and controls[0]['run'] == run_index
        assert len(ends) == 1 and ends[0]['run'] == run_index
        for placement in (controls[0], ends[0]):
            assert placement['worker_physical_cores'] >= 3
            assert [int(v['actual_mask']) for v in placement['worker_affinity']] == [1, 4, 16]
            assert [v['slot'] for v in placement['worker_affinity']] == [0, 1, 2]
            assert all(v['group'] == 0 for v in placement['worker_affinity'])
            assert placement['worker_binding_events']
        assert controls[0]['q4_bytes'] == 557383680 and controls[0]['router_f32_bytes'] == 53084160
        assert len(timed) == 24
        assert {(r['rep'], r['fixture']) for r in timed} == {(a, b) for a in range(3) for b in range(8)}
        assert all(r['threads'] == 3 and r['run'] == run_index and
                   0 < r['operator_seconds'] <= r['wall_seconds'] < LIMIT_SECONDS for r in timed)
        assert controls[0]['hash'] == next(r['hash'] for r in timed if r['rep'] == r['fixture'] == 0)
        fixture_medians = {str(f): 1000*statistics.median(r['operator_seconds'] for r in timed if r['fixture'] == f)
                           for f in range(8)}
        repetition_medians = {str(rep): 1000*statistics.median(r['operator_seconds'] for r in timed if r['rep'] == rep)
                              for rep in range(3)}
        runs.append(dict(run=run_index, threads=3, affinity=[0, 2, 4], argv=argv,
                         pid=child.pid, creation_time=birth,
                         exit_code=child.returncode, log=descriptor(log_path), combined_OS_peak=peak,
                         control=controls[0], end=ends[0], timed=timed,
                         fixture_median_operator_ms=fixture_medians,
                         repetition_median_operator_ms=repetition_medians,
                         median_operator_ms=1000*statistics.median(r['operator_seconds'] for r in timed),
                         median_wall_ms=1000*statistics.median(r['wall_seconds'] for r in timed),
                         p95_operator_ms=1000*sorted(r['operator_seconds'] for r in timed)[22]))
    assert all([r['hash'] for r in runs[0]['timed']] == [r['hash'] for r in other['timed']]
               for other in runs[1:]), 'thread-count output mismatch'
    assert all(descriptor(p['path']) == p for p in pins), 'input changed'
    value = statistics.median(r['median_operator_ms'] for r in runs)
    decision = ('REJECT_PROTECTED_CORE_LOWER_SCREEN' if value >= 20 else
                'NEW_CORE_TRANSFORM_REQUIRED' if value >= 14 else
                'UNRESOLVED_INCOMPLETE_SCREEN' if value >= 10 else
                'ENABLING_MARGIN_ONLY_INCOMPLETE_SCREEN')
    write(OUT / 'result.json', dict(schema='PROTECTED_CORE_120B_PROJECTION_SCREEN_V1',
        inputs=pins, compiler_version=compiler_version, build=build, binary=binary,
        psutil_version=psutil_version, copied_libomp=descriptor(copied_dll),
        runtime_environment=runtime_environment, runs=runs,
        across_run_median_operator_ms=value,
        geometry=dict(layers=36, width=2880, Q=4096, K=512, V=512, O_input=4096,
                      q4_block=256, q4_row_input_padded=3072, q4_bytes=557383680,
                      router_F32_bytes=53084160, expert_topk=4, experts=128),
        measured='Q/K/V/O projections and F32 biases in physical Q4_K blocks; Q8_K activations; F32 router+top4 softmax; RMS norms; synthetic seeded values. Operator timer excludes finite checks/output hashes; separate wall timer includes them.',
        omitted='RoPE, F16 KV cache and attention scores/softmax/value sum, sink logits, expert functions, LM head, source weights, quality and full model execution',
        zero_head_and_expert_cost=True, incomplete_favorable_lower_screen=True,
        decision=decision, admitted_full_rate=False, admitted_quality=False,
        seconds=time.monotonic()-began, OS_peak=process.memory_info().peak_wset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    import psutil
    proc = psutil.Process()
    start = time.monotonic()
    try:
        run(args, proc, start)
        print(json.dumps(dict(result=descriptor(OUT / 'result.json'))), flush=True)
    except BaseException as error:
        if CREATED_OUTPUT and not (OUT / 'terminal.json').exists():
            write(OUT / 'terminal.json', dict(exit_code=1, error=repr(error), traceback=traceback.format_exc(),
                pid=proc.pid, creation_time=proc.create_time(), seconds=time.monotonic()-start))
        raise
    else:
        write(OUT / 'terminal.json', dict(exit_code=0, error=None, result=descriptor(OUT / 'result.json'),
            pid=proc.pid, creation_time=proc.create_time(), seconds=time.monotonic()-start,
            OS_peak=proc.memory_info().peak_wset))
