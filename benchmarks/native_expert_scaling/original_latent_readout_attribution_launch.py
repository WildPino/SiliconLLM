"""Hold the two stored decoder interventions and seal their output extents."""
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
    assert not a.out.exists() and not a.directory.exists()
    terminal=a.out.with_suffix('.terminal.json');log_path=a.out.with_suffix('.worker.log')
    assert not terminal.exists() and not log_path.exists()
    assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());lim=b['limits']
    began=time.monotonic();peak=0;reader=memory_reader()
    for item in b['inputs']:assert extent(item['path'])==item,item['path']
    assert time.monotonic()-began<=lim['seconds'],'prehash deadline'
    command=[sys.executable,'-I','-S','-B','-X','utf8',str(B/'original_latent_readout_attribution.py'),
        '--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--freeze',a.freeze,
        '--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='6',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',CUDA_VISIBLE_DEVICES='')
    receipt=dict(freeze=a.freeze,binding_sha256=a.binding_sha,command=command,launcher_pid=proc.pid,limits=lim)
    with log_path.open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=8)
        receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time());fault=None
        try:
            while child.poll() is None:
                peak=max(peak,reader(child))
                assert time.monotonic()-began<=lim['seconds'],'attribution family deadline'
                assert peak+proc.memory_info().peak_wset<=lim['OS_bytes'],'attribution OS cap'
                assert log_path.stat().st_size<=lim['log_bytes'],'attribution log cap'
                assert not psutil.Process(child.pid).children(recursive=True),'unexpected child'
                time.sleep(.1)
            assert child.returncode==0,log_path.read_text(errors='replace')[-5000:]
            for item in b['inputs']:assert extent(item['path'])==item,item['path']
            outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
            assert sum(i['bytes'] for i in outputs)<=lim['output_bytes'],'attribution outputs cap'
            assert time.monotonic()-began<=lim['seconds'],'attribution sealing deadline'
            receipt.update(result=extent(a.out),outputs=outputs,inputs_before_after_exact=True)
        except BaseException as error:
            fault=repr(error)
            if child.poll() is None:child.kill();child.wait()
        finally:
            peak=max(peak,reader(child));receipt.update(exit_code=child.returncode,error=fault,
                worker_OS_peak_through_exit=peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,
                elapsed_seconds=time.monotonic()-began,log=extent(log_path),model_calls=0,optimizer_updates=0)
            write(terminal,receipt)
    assert child.returncode==0 and fault is None,fault
    print(json.dumps(dict(terminal=str(terminal),exit_code=0,seconds=receipt['elapsed_seconds'],result_sha256=receipt['result']['sha256'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('binding','directory','out'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('binding-sha','freeze'):parser.add_argument('--'+name,required=True)
    main(parser.parse_args())
