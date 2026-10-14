"""Verify ALL pins before the exhaustive preflight as well as before timing."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
import native_bf_activation_lut_v2 as implementation
from native_24_cost import load,check_pins
from falcon_auxiliary_s6_transport import extent,write


def prepare(a):
    doc=implementation.DOC;parent=doc/'native_bf_activation_lut_binding_v2_20261014.json'
    b=load(parent);tp=parent.with_suffix('.preparation_terminal.json');t=load(tp)
    assert t['exit_code']==0 and t['result']==extent(parent)
    assert not Path(b['output_directory']).exists()
    new=[Path(__file__),doc/'NATIVE_BF_ACTIVATION_LUT_EXECUTION_20261014.md']
    for p in new:
        blob=subprocess.check_output(['git','show',a.freeze+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n')
    b.update(schema='NATIVE_BF_ACTIVATION_LUT_EXECUTION_BINDING_V1',execution_freeze=a.freeze,
        exhaustive_preparation=dict(binding=extent(parent),terminal=extent(tp)),
        execution_order='ALL input hashes before exhaustive table/FP guard preflight; inherited paired/replay hashes before/after unchanged.')
    b['inputs'] += [extent(parent),extent(tp)]+[extent(p) for p in new]
    write(a.out,b);print({'binding':extent(a.out)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run','audit','close'])
    for name in ('out','binding','report','audit'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--freeze');parser.add_argument('--binding-sha');a=parser.parse_args()
    import psutil
    proc=psutil.Process();proc.cpu_affinity([0]);start=time.monotonic()
    receipt=dict(schema='NATIVE_BF_ACTIVATION_LUT_EXECUTION_TERMINAL_V1',phase=a.phase,
        command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try:
        assert not a.out.exists()
        if a.phase=='prepare':prepare(a)
        else:
            b=load(a.binding) if a.phase=='run' else load(load(a.report)['binding']['path'])
            def guard():
                assert time.monotonic()-start<=2700 and proc.memory_info().peak_wset<=2<<30
                assert not proc.children(recursive=True)
            check_pins(b,guard)
            {'run':implementation.run,'audit':implementation.audit,'close':implementation.close}[a.phase](a)
            check_pins(b,guard)
        assert time.monotonic()-start<=2700 and proc.memory_info().peak_wset<=2<<30
        receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.preparation_terminal.json' if a.phase=='prepare' else '.terminal.json'),receipt)
