"""Held full24-case GPU campaign and complete stored CPU audit."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write
from original_falcon_whole_recovery import memory_reader
sys.path.insert(0,str(SITE))

def main(a):
    import psutil
    lb=json.loads(a.launch_binding.read_bytes());assert sha(a.launch_binding)==a.launch_binding_sha
    assert all(extent(i['path'])==i for i in lb['inputs'])
    b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha==lb['numeric_binding_sha256']
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())};observed=[];native_children={}
    allowed={row['pid']:row for row in lb['allowed_foreign']}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        if p.pid in allowed:
            permit=allowed[p.pid];assert abs(p.create_time()-permit['creation_time'])<1e-5 and argv==permit['cmdline'];observed.append(permit);continue
        assert not name.startswith(('python','native','original_engine','clang','lld','packed_original')),('overlap',p.pid,name)
    lim=b['audit_limits' if a.audit else 'capture_limits'];terminal=a.out.with_suffix('.terminal.json');log=a.out.with_suffix('.worker.log')
    assert not any(p.exists() for p in (a.out,terminal,log));began=time.monotonic();peak=0;reader=memory_reader();native_peak=0
    inputs=b['inputs']+lb['inputs'];extra=[]
    if a.audit:extra=[extent(a.source_result),extent(a.source_result.with_suffix('.terminal.json'))];inputs+=extra
    else:
        assert not a.directory.exists()
        assert shutil.disk_usage(a.directory.parent).free>=lim['output_bytes']+lim['free_disk_bytes']
    for item in inputs:assert extent(item['path'])==item
    if a.audit:
        source_terminal=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes());assert source_terminal['exit_code']==0 and source_terminal['error'] is None
        for item in source_terminal['outputs']:assert extent(item['path'])==item
    assert time.monotonic()-began<lim['seconds']
    command=[sys.executable,'-I','-S','-B','-X','utf8',lb['numeric_script'],'--audit-worker' if a.audit else '--capture-worker','--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
    if a.audit:command+=['--source-result',str(a.source_result.resolve())]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='' if a.audit else '0',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    receipt=dict(freeze=a.freeze,binding_sha256=a.binding_sha,launcher_freeze=a.launch_freeze,launch_binding=extent(a.launch_binding),launcher_pid=proc.pid,limits=lim,command=command,allowed_foreign_observed=observed,timing_control_admitted=False)
    with log.open('xb') as output:
        child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env,creationflags=8);fault=None
        receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time());write(a.out.with_suffix('.live.json'),receipt)
        try:
            while child.poll() is None:
                peak=max(peak,reader(child))
                try:
                    descendants=psutil.Process(child.pid).children(recursive=True);assert len(descendants)<=1
                    for p in descendants:
                        argv=p.cmdline();assert not a.audit and p.ppid()==child.pid and Path(p.exe()).resolve()==Path(b['executable']['path']).resolve() and argv[1] in ('--prefix','--stream')
                        key=(p.pid,p.create_time());old=native_children.get(key,dict(pid=p.pid,creation_time=p.create_time(),command=argv,observed_OS_peak=0));assert old['command']==argv
                        old['observed_OS_peak']=max(old['observed_OS_peak'],p.memory_info().peak_wset);native_children[key]=old;native_peak=max(native_peak,old['observed_OS_peak'])
                except psutil.NoSuchProcess:pass
                assert time.monotonic()-began<=lim['seconds'] and peak+native_peak+proc.memory_info().peak_wset<=lim['OS_bytes'] and log.stat().st_size<=lim['log_bytes'];time.sleep(.1)
            assert child.returncode==0,log.read_text(errors='replace')[-6000:]
            for item in inputs+(source_terminal['outputs'] if a.audit else []):assert extent(item['path'])==item
            outputs=[] if a.audit else [extent(p) for p in sorted(a.directory.rglob('*')) if p.is_file()]
            peak=max(peak,reader(child))
            if not a.audit:
                native_peak=max(native_peak,json.loads(a.out.read_bytes())['native_OS_peak'])
            assert peak+native_peak+proc.memory_info().peak_wset<=lim['OS_bytes'] and sum(item['bytes'] for item in outputs)+a.out.stat().st_size<=lim['output_bytes'] and time.monotonic()-began<=lim['seconds']
            receipt.update(inputs_before_after_exact=True,result=extent(a.out),outputs=outputs,extra_inputs=extra)
        except BaseException as error:
            fault=repr(error)
            if child.poll() is None:
                try:
                    for p in psutil.Process(child.pid).children(recursive=True):
                        if p.ppid()==child.pid and Path(p.exe()).resolve()==Path(b['executable']['path']).resolve():p.kill()
                except psutil.NoSuchProcess:pass
                child.kill();child.wait()
        finally:
            peak=max(peak,reader(child));receipt.update(exit_code=child.returncode,error=fault,elapsed_seconds=time.monotonic()-began,worker_OS_peak_through_exit=peak,native_OS_peak=native_peak,observed_native_children=list(native_children.values()),launcher_OS_peak_snapshot=proc.memory_info().peak_wset,log=extent(log));write(terminal,receipt)
    assert child.returncode==0 and fault is None,fault
    print(json.dumps(dict(terminal=str(terminal),seconds=receipt['elapsed_seconds'],result_sha256=receipt['result']['sha256'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--source-result',type=Path)
    for key in ('binding','launch-binding','out','directory'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('binding-sha','freeze','launch-binding-sha','launch-freeze'):p.add_argument('--'+key,required=True)
    main(p.parse_args())
