"""Pre-observation wall-budget amendment; original numerical code unchanged."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),
             str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
import native_wide_readout as original
import native_wide_readout_audit as auditor
from falcon_auxiliary_s6_transport import extent,write


def prepare(a):
    doc=original.DOC;parent=doc/'native_wide_readout_binding_20261014.json'
    b=original.load(parent);receipt=parent.with_suffix('.preparation_terminal.json')
    t=original.load(receipt);assert t['exit_code']==0 and t['result']==extent(parent)
    assert t['seconds']==884.0310000000027
    assert b['limits']==dict(seconds=1800,OS_bytes=2<<30,output_bytes=32<<20)
    assert not a.out.exists() and not Path(b['output_directory']).exists()
    new_paths=[Path(__file__),doc/'NATIVE_WIDE_READOUT_RESOURCE_V2_20261014.md']
    for p in new_paths:
        blob=subprocess.check_output(['git','show',a.freeze+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n')
    for e in b['inputs']:
        if e['path'].endswith(('native_wide_readout.py','native_wide_readout_audit.py','integer_readout_wide.c',
                               'row_state_wide.dll','row_state_wide.c','compile.log')):
            assert extent(e['path'])==e
    b.update(schema='NATIVE_WIDE_READOUT_BINDING_V2',resource_freeze=a.freeze,
        original_preparation=dict(binding=extent(parent),terminal=extent(receipt)),
        resource_amendment=dict(before=b['limits'].copy(),after=dict(seconds=2700,OS_bytes=2<<30,output_bytes=32<<20),
            reason='Observed884.031s full-file preparation before numeric observations; preserve complete hashing/paired prefixes/all97 replay.'),
        limits=dict(seconds=2700,OS_bytes=2<<30,output_bytes=32<<20))
    b['inputs'] += [extent(parent),extent(receipt)]+[extent(p) for p in new_paths]
    write(a.out,b);print({'binding':extent(a.out)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run','audit','close'])
    for name in ('out','binding','report','audit'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--freeze');parser.add_argument('--binding-sha');a=parser.parse_args()
    import psutil
    proc=psutil.Process();start=time.monotonic();t=dict(schema='NATIVE_WIDE_READOUT_V2_TERMINAL',
        phase=a.phase,command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try:
        assert not a.out.exists()
        {'prepare':prepare,'run':original.run,'audit':auditor.main,'close':original.close}[a.phase](a)
        assert time.monotonic()-start<=2700 and proc.memory_info().peak_wset<=2<<30
        t.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        t.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        t.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.preparation_terminal.json' if a.phase=='prepare' else '.terminal.json'),t)
