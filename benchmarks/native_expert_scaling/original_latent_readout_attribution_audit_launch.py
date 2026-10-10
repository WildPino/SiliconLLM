"""Finite held audit of saved F64 attribution scores; no contractions or model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write
from original_falcon_whole_recovery import memory_reader
sys.path.insert(0,str(SITE))


def main(a):
    import psutil
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','native','original_engine','clang','lld','packed_original')),('overlap',p.pid,name)
    assert not any(p.exists() for p in (a.out,a.log,a.receipt))
    script=B/'original_latent_readout_attribution_audit.py'
    inputs=[extent(script),extent(Path(__file__)),extent(a.result),extent(a.binding)]
    began=time.monotonic();reader=memory_reader();peak=0
    command=[sys.executable,'-I','-S','-B','-X','utf8',str(script),'--result',str(a.result.resolve()),'--binding',str(a.binding.resolve()),'--out',str(a.out.resolve())]
    record=dict(command=command,inputs=inputs,launcher_pid=proc.pid,
        limits=dict(seconds=600,OS_bytes=4<<30,log_bytes=8<<20,output_bytes=2<<20))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',CUDA_VISIBLE_DEVICES='')
    with a.log.open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=8)
        record.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time());fault=None
        try:
            while child.poll() is None:
                peak=max(peak,reader(child))
                assert time.monotonic()-began<=600,'attribution audit deadline'
                assert peak+proc.memory_info().peak_wset<=4<<30,'attribution audit OS cap'
                assert a.log.stat().st_size<=8<<20,'attribution audit log cap'
                assert not psutil.Process(child.pid).children(recursive=True),'unexpected audit child'
                if a.out.exists():assert a.out.stat().st_size<=2<<20,'audit output cap'
                time.sleep(.1)
            assert child.returncode==0,a.log.read_text(errors='replace')[-5000:]
            for item in inputs:assert extent(item['path'])==item
        except BaseException as error:
            fault=repr(error)
            if child.poll() is None:child.kill();child.wait()
        finally:
            peak=max(peak,reader(child));record.update(exit_code=child.returncode,error=fault,
                worker_OS_peak_through_exit=peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,
                seconds=time.monotonic()-began,output=extent(a.out) if a.out.exists() else None,log=extent(a.log),
                model_calls=0,head_contractions=0,optimizer_updates=0)
            write(a.receipt,record)
    assert child.returncode==0 and fault is None,fault
    print(json.dumps(dict(receipt=str(a.receipt),exit_code=0,seconds=record['seconds'],audit_sha256=sha(a.out))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('result','binding','out','log','receipt'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
