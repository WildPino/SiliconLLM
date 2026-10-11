"""Bound CPU-only holder for the recovered stored campaign audit."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha
sys.path.insert(0,str(SITE))
from original_categorical_campaign_geometry import emit
from original_falcon_whole_recovery import memory_reader

LIMITS=dict(seconds=32400,OS_bytes=24<<30,output_bytes=4<<20,log_bytes=8<<20)


def quiet(proc):
    import psutil
    own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        if p.pid in own:continue
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','native','original_engine','clang','lld','packed_original')),('overlap',p.pid,name,argv)


def prepare(a):
    import psutil,torch,numpy as np
    import recovered_categorical_campaign_audit_v2
    start=time.monotonic();proc=psutil.Process();quiet(proc)
    assert not a.binding.exists()
    seal=json.loads(a.recovery_seal.read_bytes());assert sha(a.recovery_seal)==a.recovery_seal_sha
    assert seal['schema']=='CATEGORICAL_CAMPAIGN_RECOVERY_SEAL_V1'
    assert not seal['original_successful_custody'] and seal['producer_result_missing']
    original=json.loads(Path(seal['original_numeric_binding']['path']).read_bytes())
    holder=json.loads(Path(seal['original_holder_binding']['path']).read_bytes())
    failed=json.loads(Path(seal['original_failed_terminal']['path']).read_bytes())
    assert failed['exit_code']!=0 and failed['error'] is not None
    assert not Path(a.missing_original_result).exists()
    assert (torch.__version__,np.__version__,psutil.__version__)==('2.6.0+cu124','2.4.6','7.2.2')
    for key in ('worker_pid','launcher_pid'):assert not psutil.pid_exists(failed[key])
    fixture=json.loads(a.preflight.read_bytes())
    assert all(fixture['flags'].values())
    paths=[Path(i['path']) for i in original['inputs']+holder['inputs']]
    paths += [a.recovery_seal,a.protocol,a.preflight,Path(__file__),B/'recovered_categorical_campaign_audit_v2.py',
              B/'recovered_categorical_campaign_preflight_v2.py',*[Path(seal[k]['path']) for k in ('original_numeric_binding','original_holder_binding','original_failed_terminal','original_log','seal_source')]]
    for module in list(sys.modules.values()):
        for key in ('__file__','__cached__'):
            p=getattr(module,key,None)
            if isinstance(p,(str,os.PathLike)) and Path(p).is_file():paths.append(Path(p))
    paths += [Path(sys.executable),Path(sys.executable).parent/'python312.dll']
    fault_base=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/recovered_categorical_campaign_stored_adjudication_20261011'
    paths += [Path(str(fault_base)+suffix) for suffix in ('.terminal.json','.fault.json','.worker.log')]
    paths += [ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/recovered_categorical_campaign_binding_20261011.json']
    paths += sorted((SITE/'torch/lib').glob('*.dll'))+sorted((SITE/'numpy.libs').glob('*.dll'))+sorted((SITE/'psutil').glob('*.pyd'))
    inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)]
    lookup={x['path']:x for x in inputs}
    for item in original['inputs']+holder['inputs']+[seal[k] for k in ('original_numeric_binding','original_holder_binding','original_failed_terminal','original_log','seal_source')]:assert lookup[item['path']]==item
    assert lookup[str(a.recovery_seal.resolve())]['sha256']==a.recovery_seal_sha
    for key,item in fixture['sources'].items():assert lookup[item['path']]==item
    quiet(proc)
    emit(a.binding,dict(schema='RECOVERED_CATEGORICAL_CAMPAIGN_BINDING_V2',inputs=inputs,limits=LIMITS,
        recovery_seal=extent(a.recovery_seal),original_numeric_binding=seal['original_numeric_binding'],
        original_holder_binding=seal['original_holder_binding'],original_failed_terminal=seal['original_failed_terminal'],
        original_freeze=seal['freeze'],missing_original_result=str(a.missing_original_result.resolve()),
        protocol=extent(a.protocol),preflight=extent(a.preflight),worker=extent(B/'recovered_categorical_campaign_audit_v2.py'),
        holder=extent(__file__),preparation_seconds=time.monotonic()-start,
        scope='Full stored24-direction/48-proposal/48-final-forced audit at recovered boundary. No successful original receipt or transient runtime counters invented.'))
    print(json.dumps(dict(binding_sha256=sha(a.binding),inputs=len(inputs),seconds=time.monotonic()-start)),flush=True)


def launch(a):
    import psutil
    binding=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    assert binding['schema']=='RECOVERED_CATEGORICAL_CAMPAIGN_BINDING_V2' and binding['limits']==LIMITS
    assert binding['holder']==extent(__file__)
    proc=psutil.Process();proc.cpu_affinity([11]);quiet(proc)
    terminal=a.out.with_suffix('.terminal.json');log=a.out.with_suffix('.worker.log');live=a.out.with_suffix('.live.json')
    assert not any(p.exists() for p in (a.out,terminal,log,live,a.out.with_suffix('.fault.json')))
    seal=json.loads(Path(binding['recovery_seal']['path']).read_bytes())
    inputs=binding['inputs']+seal['outputs'];signature={x['path']:(Path(x['path']).stat().st_size,Path(x['path']).stat().st_mtime_ns) for x in seal['outputs']}
    assert {str(p.resolve()) for p in a.directory.rglob('*') if p.is_file()}==set(signature)
    began=time.monotonic();reader=memory_reader();peak=0;child=None;stage='input_hash_before';error=None;observations=[]
    command=[sys.executable,'-I','-S','-B','-X','utf8',binding['worker']['path'],'--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
    receipt=dict(schema='RECOVERED_CATEGORICAL_CAMPAIGN_HOLDER_V2',binding=extent(a.binding),binding_sha256=a.binding_sha,
        source_freeze=a.freeze,original_freeze=binding['original_freeze'],original_successful_custody=False,
        launcher_pid=proc.pid,limits=LIMITS,command=command,timing_control_admitted=False)
    def guard():
        assert time.monotonic()-began<=LIMITS['seconds'],('time_cap',stage)
        assert peak+proc.memory_info().peak_wset<=LIMITS['OS_bytes'],('OS_cap',stage)
        if log.exists():assert log.stat().st_size<=LIMITS['log_bytes'],('log_cap',stage)
    def check_inputs():
        for number,item in enumerate(inputs,1):
            assert extent(item['path'])==item,('changed_input',stage,item['path']);guard()
            if number%1000==0:print(json.dumps(dict(stage=stage,files=number,seconds=time.monotonic()-began)),flush=True)
    try:
        check_inputs();quiet(proc);stage='worker_running'
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with log.open('xb') as output:
            child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env,creationflags=8)
            receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time())
            emit(live,receipt)
            while child.poll() is None:
                peak=max(peak,reader(child))
                try:
                    descendants=psutil.Process(child.pid).children(recursive=True)
                    if descendants:
                        details=[]
                        for p in descendants:
                            try:details.append(dict(pid=p.pid,creation_time=p.create_time(),ppid=p.ppid(),exe=p.exe(),command=p.cmdline()))
                            except psutil.NoSuchProcess:details.append(dict(pid=p.pid,exited_during_observation=True))
                        observations.append(dict(stage=stage,seconds=time.monotonic()-began,children=details))
                        raise RuntimeError('Audit spawned forbidden child; observed details retained')
                except psutil.NoSuchProcess:pass
                guard();time.sleep(.25)
            peak=max(peak,reader(child));guard()
        assert child.returncode==0,('worker_exit',child.returncode)
        stage='input_hash_after';check_inputs()
        assert {str(p.resolve()) for p in a.directory.rglob('*') if p.is_file()}==set(signature)
        assert all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in signature.items())
        quiet(proc);guard()
        assert a.out.stat().st_size<=LIMITS['output_bytes']
        receipt.update(inputs_before_after_exact=True,recovered_output_tree_before_after_exact=True,result=extent(a.out))
    except BaseException as failure:
        error=dict(error=repr(failure),traceback=traceback.format_exc(),stage=stage)
        if child is not None and child.poll() is None:
            try:
                for p in psutil.Process(child.pid).children(recursive=True):
                    try:p.kill()
                    except psutil.NoSuchProcess:pass
            except psutil.NoSuchProcess:pass
            child.kill();child.wait()
    finally:
        if child is not None:peak=max(peak,reader(child))
        receipt.update(exit_code=child.returncode if child else None,error=error,elapsed_seconds=time.monotonic()-began,
            worker_OS_peak_through_exit=peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,
            unexpected_child_observations=observations,log=extent(log) if log.exists() else None)
        emit(terminal,receipt)
    assert error is None,error
    print(json.dumps(dict(terminal=str(terminal),seconds=receipt['elapsed_seconds'],result_sha256=receipt['result']['sha256'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    for key in ('binding','directory','out','recovery-seal','protocol','preflight','missing-original-result'):p.add_argument('--'+key,type=Path)
    for key in ('binding-sha','recovery-seal-sha','freeze'):p.add_argument('--'+key)
    a=p.parse_args();prepare(a) if a.prepare else launch(a)
