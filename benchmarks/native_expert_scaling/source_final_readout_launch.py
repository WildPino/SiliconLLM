"""Hold final source readouts through exit; no history or learner construction."""
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
    terminal=a.out.with_suffix('.terminal.json');log_path=a.out.with_suffix('.worker.log')
    assert not any(p.exists() for p in (a.out,a.directory,terminal,log_path))
    assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());lim=b['limits']
    began=time.monotonic();peak=0;reader=memory_reader()
    for item in b['inputs']:assert extent(item['path'])==item,item['path']
    assert time.monotonic()-began<=lim['seconds'],'prehash deadline'
    command=[sys.executable,'-I','-S','-B','-X','utf8',str(B/'source_final_readout.py'),
        '--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--freeze',a.freeze,
        '--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='6',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    receipt=dict(freeze=a.freeze,binding_sha256=a.binding_sha,command=command,launcher_pid=proc.pid,limits=lim)
    with log_path.open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=8)
        receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time());fault=None
        try:
            while child.poll() is None:
                peak=max(peak,reader(child))
                assert time.monotonic()-began<=lim['seconds'],'source readout family deadline'
                assert peak+proc.memory_info().peak_wset<=lim['OS_bytes'],'OS cap'
                assert log_path.stat().st_size<=lim['log_bytes'],'log cap'
                try:assert not psutil.Process(child.pid).children(recursive=True),'unexpected child'
                except psutil.NoSuchProcess:pass
                time.sleep(.1)
            assert child.returncode==0,log_path.read_text(errors='replace')[-5000:]
            r=json.loads(a.out.read_bytes())
            assert r['BF16_source_head_calls']==r['F64_head_contractions']==8808
            assert r['source_history_forwards']==r['source_generations']==r['model_instances']==r['optimizer_updates']==r['native_calls']==r['reserved_queries']==0
            assert r['GPU_allocated_peak']<=lim['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=lim['GPU_reserved_bytes']
            for item in b['inputs']:assert extent(item['path'])==item,item['path']
            outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
            assert sum(i['bytes'] for i in outputs)<=lim['output_bytes'],'output cap'
            assert time.monotonic()-began<=lim['seconds'],'sealing deadline'
            receipt.update(result=extent(a.out),outputs=outputs,inputs_before_after_exact=True,
                GPU_allocated_peak=r['GPU_allocated_peak'],GPU_reserved_peak=r['GPU_reserved_peak'])
        except BaseException as error:
            fault=repr(error)
            if child.poll() is None:child.kill();child.wait()
        finally:
            peak=max(peak,reader(child));receipt.update(exit_code=child.returncode,error=fault,
                worker_OS_peak_through_exit=peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,
                elapsed_seconds=time.monotonic()-began,log=extent(log_path),source_history_forwards=0,optimizer_updates=0)
            write(terminal,receipt)
    assert child.returncode==0 and fault is None,fault
    print(json.dumps(dict(terminal=str(terminal),exit_code=0,seconds=receipt['elapsed_seconds'],result_sha256=receipt['result']['sha256'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('binding','directory','out'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('binding-sha','freeze'):p.add_argument('--'+key,required=True)
    main(p.parse_args())
