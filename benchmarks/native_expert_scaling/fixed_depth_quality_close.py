"""Seal one completed depth phase without declaring the campaign/goal complete."""
import argparse
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from fixed_depth_quality import load,check_inputs


def main(a):
    import psutil
    r=load(a.report);b=load(r['binding']['path']);ad=load(a.audit)
    assert ad['decision']=='INDEPENDENT_STORED_NUMERIC_PASS' and ad['all_numeric_pass']
    assert ad['report']==extent(a.report) and ad['binding']==r['binding']
    assert ad['depth']==r['depth'] and ad['phase']==r['phase']
    commands=[]
    for path in (a.report.with_suffix('.terminal.json'),a.report.with_suffix('.worker_terminal.json'),a.audit.with_suffix('.terminal.json')):
        t=load(path);assert t['exit_code']==0
        target=a.audit if path==a.audit.with_suffix('.terminal.json') else a.report
        assert t['result']==extent(target)
        if psutil.pid_exists(t['pid']):assert abs(psutil.Process(t['pid']).create_time()-t['creation_time'])>.001
        commands.append(dict(receipt=extent(path),pid=t['pid'],creation_time=t['creation_time'],matching_process_absent=True))
    check_inputs(b)
    for e in r['outputs']+[r['binding']]:assert extent(e['path'])==e,e['path']
    directory=Path(r['outputs'][0]['path']).parent
    assert {str(p.resolve()):p.stat().st_size for p in directory.iterdir() if p.is_file()}=={e['path']:e['bytes'] for e in r['outputs']}
    write(a.out,dict(schema='FIXED_DEPTH_PHASE_CLOSURE_V1',report=extent(a.report),audit=extent(a.audit),binding=r['binding'],
        depth=r['depth'],phase=r['phase'],commands=commands,phase_family_files=len(r['outputs']),phase_family_bytes=r['output_bytes'],
        inputs_outputs_exact=True,decision='STORED_DEPTH_PHASE_QUALIFIED',quality_admissible=ad['quality_admissible'],
        goal_status='ACTIVE_INCOMPLETE',scope='Per-phase numerical/custody closure. Other FIT arms, selection/evaluation '
        'and final same-artifact CPU quality/useful50/DRAM/n/general-family requirements remain separate.'))
    print(dict(closure=extent(a.out)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('report','audit','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();import psutil
    proc=psutil.Process();start=time.monotonic();t=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try:
        assert not a.out.exists();main(a);t.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:t.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        t.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),t)
