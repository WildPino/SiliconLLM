"""Read-only P1 recovery and unchanged unexecuted A/P2 continuation."""
import argparse,inspect,json,os,re,subprocess,sys,textwrap,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
import encoded_organs_wait as old
from falcon_auxiliary_s6_transport import extent,write
STEM='encoded_organs_wait_recovered_20261014'


def validator():
    source=inspect.getsource(old.pipeline)
    lo=source.index("        r=cg.load(result);assert r['profile']==label")
    hi=source.index('        results.append(',lo)
    block=textwrap.dedent(source[lo:hi])
    before='re.search(r"\\bKMP_BLOCKTIME\\s*=\\s*[\'\\\"]?"+block+r"\\b",effective)'
    after='re.search(r"^\\s*KMP_BLOCKTIME\\s*=\\s*[\'\\\"]?"+block+r"(?:ms)?[\'\\\"]?\\s*$",effective,re.MULTILINE)'
    assert block.count(before)==1
    patched=block.replace(before,after)
    assert patched.replace(after,before)==block
    namespace=dict(old.__dict__)
    body='def validate(label,policy,block,result,stderr,gb,np):\n'+textwrap.indent(patched,'    ')+'    return r\n'
    exec(compile(body,'<retained-ledger-unit-repair>','exec'),namespace)
    return namespace['validate']


def run(a,proc,start,guard):
    import numpy as np,psutil
    root=old.ROOT;doc=old.DOC;family=old.FAMILY
    frozen=[Path(__file__),doc/'ENCODED_ORGANS_WAIT_RECOVERY_20261014.md']
    for p in frozen:
        assert subprocess.check_output(['git','show',a.freeze+':'+p.relative_to(root).as_posix()],cwd=root).replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n')
    bp=doc/'encoded_organs_wait_20261014_binding.json';b=old.cg.load(bp)
    fault_path=doc/'encoded_organs_wait_20261014_pipeline.terminal.json';fault=old.cg.load(fault_path)
    assert fault['exit_code']==1 and 'KMP_BLOCKTIME' in fault['traceback']
    original=root/'benchmarks/native_expert_scaling/encoded_organs_wait.py'
    assert subprocess.check_output(['git','show',b['freeze']+':'+original.relative_to(root).as_posix()],cwd=root).replace(b'\r\n',b'\n')==original.read_bytes().replace(b'\r\n',b'\n')
    assert not (doc/'encoded_organs_wait_20261014_result.json').exists()
    for label in ('A','P2'):
        assert not any((family/(label+suffix)).exists() for suffix in ('.json','.terminal.json','.stdout','.stderr'))
    for e in (fault,old.cg.load(family/'P1.terminal.json')):
        if psutil.pid_exists(e['pid']):assert abs(psutil.Process(e['pid']).create_time()-e['creation_time'])>.001
    old.check(b['inputs'],guard)
    preserved=[extent(p) for p in (bp,fault_path,family/'P1.json',family/'P1.terminal.json',family/'P1.stdout',family/'P1.stderr')]
    preserved += [extent(p) for p in frozen]
    rbp=doc/(STEM+'_binding.json');assert not rbp.exists()
    write(rbp,dict(original_binding=extent(bp),original_fault=extent(fault_path),freeze=a.freeze,preserved=preserved,
        limits=b['limits'],original_posthash_complete=False,recovery_current_seal=True,interrupted_schedule=True))
    validate=validator();gb=old.cg.load(b['generator_binding']['path']);results=[];children=[]
    for label,policy,block in old.PROFILES:
        guard();result=family/(label+'.json');stdout=family/(label+'.stdout');stderr=family/(label+'.stderr')
        if label!='P1':
            with stdout.open('xb') as fo,stderr.open('xb') as fe:
                env=os.environ.copy();env.update(OMP_WAIT_POLICY=policy,KMP_BLOCKTIME=block,KMP_SETTINGS='1')
                command=[sys.executable,'-I','-S','-B','-X','utf8',str(original),'--worker',label,'--binding',str(bp),'--out',str(result)]
                child=subprocess.Popen(command,cwd=root,env=env,stdout=fo,stderr=fe,creationflags=0x00000008)
                observer=psutil.Process(child.pid);born=observer.create_time();begin=time.monotonic();peak=0
                try:
                    while child.poll() is None:
                        guard();assert time.monotonic()-begin<=600
                        try:peak=max(peak,observer.memory_info().peak_wset)
                        except psutil.NoSuchProcess:pass
                        assert peak<=4<<30;time.sleep(.2)
                except BaseException:
                    if child.poll() is None:child.kill();child.wait()
                    raise
            terminal=old.cg.load(result.with_suffix('.terminal.json'))
            assert child.returncode==terminal['exit_code']==0 and terminal['pid']==child.pid and abs(terminal['creation_time']-born)<.001
        else:
            terminal=old.cg.load(result.with_suffix('.terminal.json'));assert terminal['exit_code']==0
        assert terminal['result']==extent(result)
        r=validate(label,policy,block,result,stderr,gb,np)
        children.append(dict(profile=label,pid=terminal['pid'],creation_time=terminal['creation_time'],terminal=extent(result.with_suffix('.terminal.json')),
            executed_in_recovery=label!='P1',original_worker_exit_code=terminal['exit_code']))
        results.append(dict(profile=label,report=extent(result),summary=r['summary'],exact_cells=r['exact_cells'],stderr=extent(stderr),stdout=extent(stdout)))
        print(json.dumps(dict(profile=label,reused=label=='P1',ledger_pass=True)),flush=True)
    decisions={}
    for mode in ('resident','parameter_evicted'):
        decisions[mode]={}
        for fmt in (16,8,4):
            v=[x['summary'][mode][str(fmt)]['whole_ns']['median'] for x in results]
            drift=max(v[0],v[2])/min(v[0],v[2]);decisions[mode][str(fmt)]=dict(passive_drift=drift,drift_pass=drift<=1.10,
                active_median_ns=v[1],active_partial_20ms_pass=v[1]<=20_000_000,active_vs_passive_mean=v[1]/((v[0]+v[2])/2))
    first=old.cg.load(family/'P1.terminal.json');active=old.cg.load(family/'A.terminal.json')
    gap=active['creation_time']-(first['creation_time']+first['seconds']);assert gap>=0
    old.check(b['inputs']+preserved,guard)
    for x in results:
        for k in ('report','stdout','stderr'):assert extent(x[k]['path'])==x[k]
    out=doc/(STEM+'_result.json');assert not out.exists()
    write(out,dict(schema='ENCODED_ORGANS_WAIT_RECOVERED_RESULT_V1',binding=extent(rbp),profiles=results,children=children,decisions=decisions,
        all1152_groups_ledger_audited=True,all_parent_outputs_exact=True,original_parent_exit_code=1,original_posthash_complete=False,
        recovery_current_input_output_seal=True,interrupted_schedule=True,P1_to_A_gap_seconds=gap,runtime_recipe_admitted=False,
        GPU_calls=0,RESERVED_queries=0,scope=old.SCOPE+' Original parent metadata failure and schedule interruption retained.',seconds=time.monotonic()-start))
    closure=doc/(STEM+'_closure.json');assert not closure.exists()
    write(closure,dict(report=extent(out),binding=extent(rbp),original_fault=extent(fault_path),recovery_seal_exact=True,
        original_parent_exit_code=1,original_posthash_complete=False,interrupted_schedule=True,runtime_recipe_admitted=False,goal_status='ACTIVE_INCOMPLETE'))
    return dict(report=extent(out),closure=extent(closure),decisions=decisions)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',required=True);a=parser.parse_args()
    import psutil
    proc=psutil.Process();start=time.monotonic();out=old.DOC/(STEM+'_pipeline.json');terminal=out.with_suffix('.terminal.json');owned=out.with_suffix('.owned_process.json')
    assert not out.exists() and not terminal.exists() and not owned.exists()
    receipt=dict(pid=proc.pid,creation_time=proc.create_time(),command=sys.orig_argv,seconds_cap=1800);write(owned,receipt)
    def guard():
        assert time.monotonic()-start<=1800 and proc.memory_info().peak_wset<=4<<30
        assert sum(p.stat().st_size for p in old.FAMILY.rglob('*') if p.is_file())<=32<<20
    try:
        result=run(a,proc,start,guard);guard();write(out,result);receipt.update(exit_code=0,error=None,result=extent(out))
    except BaseException as error:receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset);write(terminal,receipt)
