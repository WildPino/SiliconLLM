"""Read-only present-boundary custody of a holder-interrupted campaign.

Does NOT manufacture a successful producer receipt or recover missing counters.
No model, numerical inference, optimization, native child or GPU execution.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
import psutil

LIMITS = dict(seconds=1800, OS_bytes=256 << 20, output_bytes=8 << 20,
              recovery_output_tree_bytes=140 << 30, input_chunk_bytes=8 << 20)


def write(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode('utf8')
    if len(raw) > LIMITS['output_bytes']:
        raise RuntimeError('Recovery metadata output cap exceeded')
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main(a):
    proc = psutil.Process()
    proc.cpu_affinity([1])
    started = time.monotonic()
    state = dict(stage='parent_receipts',files_hashed=0,bytes_hashed=0)
    def guard():
        if time.monotonic()-started >= LIMITS['seconds'] or proc.memory_info().peak_wset > LIMITS['OS_bytes']:
            raise RuntimeError('Recovery seal resource cap exceeded')
    def extent(path):
        path = Path(path).resolve()
        before = path.stat()
        h = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(LIMITS['input_chunk_bytes']):
                h.update(chunk)
                state['bytes_hashed'] += len(chunk)
                guard()
        after = path.stat()
        if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
            raise RuntimeError('Input changed while hashing: '+str(path))
        state['files_hashed'] += 1
        return dict(path=str(path),bytes=before.st_size,sha256=h.hexdigest())
    def read(path, expected_sha):
        item = extent(path)
        if item['sha256'] != expected_sha:
            raise RuntimeError('Parent hash mismatch')
        return json.loads(Path(path).read_bytes()), item
    def quiet():
        own = {proc.pid, *(p.pid for p in proc.parents())}
        for p in psutil.process_iter(['name','cmdline']):
            if p.pid in own:
                continue
            name = (p.info['name'] or '').lower()
            argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                continue
            if name.startswith(('python','original_engine','native','packed_original','clang','lld')):
                raise RuntimeError('Recovery overlaps a computational process: '+str(p.pid))
    def signature(paths):
        return {str(p.resolve()):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
    try:
        quiet()
        binding, bi = read(a.binding,a.binding_sha)
        holder, hi = read(a.holder_binding,a.holder_binding_sha)
        failed, fi = read(a.failed_terminal,a.failed_terminal_sha)
        if failed['binding_sha256'] != bi['sha256'] or failed['error'] is None or failed['exit_code']==0:
            raise RuntimeError('Expected actual failed producer boundary')
        if failed['freeze'] != a.freeze or holder['numeric_binding_sha256'] != bi['sha256']:
            raise RuntimeError('Frozen producer identity mismatch')
        for key in ('worker_pid','launcher_pid'):
            if psutil.pid_exists(failed[key]):
                raise RuntimeError('Failed producer PID must be absent before sealing: '+key)
        inputs = binding['inputs']+holder['inputs']
        state['stage']='original_bound_inputs_current_check'
        for item in inputs:
            if extent(item['path']) != item:
                raise RuntimeError('Originally bound input currently differs')
        directory = a.directory.resolve()
        paths = sorted(p for p in directory.rglob('*') if p.is_file())
        before = signature(paths)
        if sum(v[0] for v in before.values()) > LIMITS['recovery_output_tree_bytes']:
            raise RuntimeError('Original output-tree cap exceeded')
        state['stage']='present_output_boundary_hash'
        outputs=[]
        for number,path in enumerate(paths,1):
            outputs.append(extent(path))
            if number%250==0:
                print(json.dumps(dict(stage=state['stage'],files=number,total=len(paths),seconds=time.monotonic()-started)),flush=True)
        after_paths=sorted(p for p in directory.rglob('*') if p.is_file())
        if signature(after_paths) != before:
            raise RuntimeError('Recovery tree metadata changed during sealing')
        state['stage']='original_bound_inputs_current_recheck'
        for item in inputs:
            if extent(item['path']) != item:
                raise RuntimeError('Originally bound input changed during recovery')
        log = extent(failed['log']['path'])
        if log != failed['log']:
            raise RuntimeError('Original failed log changed')
        if extent(a.failed_terminal) != fi or extent(a.binding) != bi or extent(a.holder_binding) != hi:
            raise RuntimeError('Parent boundary changed during recovery')
        quiet()
        guard()
        write(a.out,dict(schema='CATEGORICAL_CAMPAIGN_RECOVERY_SEAL_V1',freeze=a.freeze,
              original_numeric_binding=bi,original_holder_binding=hi,original_failed_terminal=fi,
              original_log=log,seal_source=extent(__file__),limits=LIMITS,outputs=outputs,
              original_bound_inputs_currently_exact_before_after=True,output_tree_metadata_stable_during_seal=True,
              output_hashes_at_recovery_boundary=True,original_successful_custody=False,
              producer_result_missing=True,unrecorded_stream_execution_unknown=True,
              seconds=time.monotonic()-started,OS_peak=proc.memory_info().peak_wset,
              bytes_hashed=state['bytes_hashed'],files_hashed=state['files_hashed'],
              model_loads=0,history_forwards=0,backwards=0,native_calls=0,GPU_calls=0,optimizer_updates=0,
              scope='Current closed-process artifact boundary only. Original failed holder/worker receipts remain unchanged. '
                    'A current input match does not prove complete original in-flight custody; per-file hashes and tree stat '
                    'stability are not a second full output hash pass. Independent audit must check every recorded numerical '
                    'artifact/state transition and recheck this boundary. Missing GPU peaks/transient counters and unknown '
                    'initial stream startup are not reconstructed as measured zeros or success.'))
        print(json.dumps(dict(seal=str(a.out),seconds=time.monotonic()-started,files=len(outputs))),flush=True)
    except BaseException as error:
        write(a.out.with_suffix('.fault.json'),dict(schema='CATEGORICAL_CAMPAIGN_RECOVERY_SEAL_FAULT_V1',
              error=repr(error),state=state,seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('binding','holder-binding','failed-terminal','directory','out'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('binding-sha','holder-binding-sha','failed-terminal-sha','freeze'):
        p.add_argument('--'+key,required=True)
    args=p.parse_args()
    if args.out.exists() or args.out.with_suffix('.fault.json').exists():
        p.error('Exclusive recovery namespace already consumed')
    main(args)
