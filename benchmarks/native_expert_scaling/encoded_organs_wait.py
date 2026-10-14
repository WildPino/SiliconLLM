"""Fresh P/A/P processes; same encoded generators and integer readouts."""
import argparse,ctypes,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import contextual_generator_cost as cg
from contextual_generator_codec import mapped,ORDERS
from physical_worker_organs import function,exact,reference,distribution,check
from native_24_cost import parameter_sets
STEM='encoded_organs_wait_20261014';FAMILY=Path('E:/SiliconLLM/results/native_expert_scaling')/STEM
PROFILES=[('P1','PASSIVE','0'),('A','ACTIVE','200'),('P2','PASSIVE','0')]
SCOPE='Same23 encoded/source generators and24 integer readouts on supplied h/r/a. Runtime waiting only; not a joined decoder, new quality, useful50, useful-n or physical DRAM.'


def summary(samples,np):
    return {mode:{str(fmt):{key:distribution([q[key] for q in samples if q['mode']==mode and q['format']==fmt],np)
        for key in ('whole_ns','generator_ns','readout_ns','flush_ns')} for fmt in (16,8,4)} for mode in ('resident','parameter_evicted')}


def worker(a):
    import numpy as np,psutil
    b=cg.load(a.binding);label,policy,block=next(x for x in PROFILES if x[0]==a.worker)
    os.environ.update(OMP_WAIT_POLICY=policy,KMP_BLOCKTIME=block,KMP_SETTINGS='1')
    proc=psutil.Process();start=time.monotonic();receipt=dict(pid=proc.pid,creation_time=proc.create_time(),command=sys.orig_argv,profile=label)
    try:
        runtime=os.add_dll_directory(str(Path(b['compile']['command'][0]).parent));dll=ctypes.CDLL(b['compile']['dll']['path'])
        assert function(dll,'ew_setup',[])()==0
        blocktime=function(dll,'ew_blocktime',[])();assert blocktime==int(block)
        binder=function(dll,'po_binding',[ctypes.c_int,ctypes.c_int],ctypes.c_uint64)
        initial=[[int(binder(i,j)) for j in range(5)] for i in range(3)]
        g=function(dll,'po_generator',[ctypes.c_void_p]*6);qg=function(dll,'qc_step',[ctypes.c_int]+[ctypes.c_void_p]*7)
        ro=function(dll,'po_readout',[ctypes.c_int]+[ctypes.c_void_p]*7)
        gb=cg.load(b['generator_binding']['path']);qb=cg.load(b['codec_binding']['path']);rb=cg.load(b['readout_binding']['path'])
        gr=cg.load(b['generator_report']['path']);qr=cg.load(b['codec_report']['path'])
        source=cg.weights(gb,np);pack={fmt:[mapped(p,np) for p in qb['packets'] if p['format']==fmt] for fmt in (8,4)}
        params=parameter_sets(rb,np);mup=np.asarray(gb['MuP_bits'],'u2');_,support=cg.native(gb)
        samples=[];cells=dict(generator=0,readout=0);cpu_start=proc.cpu_times()._asdict()
        for mi,mode in enumerate(('resident','parameter_evicted')):
            for ri,rec in enumerate(gb['records']):
                assert time.monotonic()-start<600 and proc.memory_info().peak_wset<=4<<30
                h,_=cg.record_inputs(rec,np);ref=reference(rec,mode,rb,np)
                gp=next(x['file'] for x in gr['records'] if x['id']==rec['id'] and x['mode']==mode)
                qp=next(x['file'] for x in qr['records'] if x['id']==rec['id'] and x['mode']==mode)
                with np.load(gp['path'],allow_pickle=False) as z:gold={16:{key:z[key].copy() for key in ('raw','gg','fields')}}
                with np.load(qp['path'],allow_pickle=False) as z:
                    for fmt in (8,4):gold[fmt]={key:z[f'{key}_{fmt}'].copy() for key in ('raw','gg','fields')}
                outg={key:np.empty((23,6704),dtype) for key,dtype in (('raw','f4'),('gg','u2'),('fields','u2'))}
                outr={key:np.empty((24,2048),dtype) for key,dtype in (('dots','i8'),('raw','f8'),('output','u2'))}
                def call_group(fmt,t,timed):
                    gt=[];rt=[];rcs=[];begin=time.perf_counter_ns()
                    for site in range(24):
                        if site:
                            i=site-1;tail=tuple(x.ctypes.data for x in (h[i,t],mup,outg['raw'][i],outg['gg'][i],outg['fields'][i]))
                            before=time.perf_counter_ns()
                            rc=g(source[i].ctypes.data,*tail) if fmt==16 else qg(fmt,pack[fmt][i][0].ctypes.data,pack[fmt][i][1].ctypes.data,*tail)
                            after=time.perf_counter_ns();gt.append(after-before);rcs.append(rc)
                        p=params[site][0];before=time.perf_counter_ns()
                        rc=ro(1,p['q'].ctypes.data,p['scale'].ctypes.data,ref['r'][site,t].ctypes.data,ref['a'][site,t:t+1].ctypes.data,
                            outr['dots'][site].ctypes.data,outr['raw'][site].ctypes.data,outr['output'][site].ctypes.data)
                        after=time.perf_counter_ns();rt.append(after-before);rcs.append(rc)
                    elapsed=time.perf_counter_ns()-begin;assert rcs==[0]*47
                    for key,value in outg.items():exact(value,np.ascontiguousarray(gold[fmt][key][:,t]),np)
                    for key,value in outr.items():exact(value,np.ascontiguousarray(ref[key][:,t]),np)
                    cells['generator']+=3*23*6704;cells['readout']+=3*24*2048
                    return dict(whole_ns=elapsed,generator_ns=sum(gt),readout_ns=sum(rt),generator_site_ns=gt,readout_site_ns=rt)
                for fmt in (16,8,4):call_group(fmt,0,False)
                for t in range(16):
                    order=ORDERS[(t+ri*16+mi*64)%6]
                    for fmt in order:
                        before=time.perf_counter_ns()
                        if mode=='parameter_evicted':
                            owners=source if fmt==16 else [x for pair in pack[fmt] for x in pair]
                            for x in owners:support.cost_evict(x.ctypes.data,x.nbytes)
                            support.cost_evict(mup.ctypes.data,mup.nbytes)
                            for p,_,_ in params:
                                for key in ('q','scale'):support.cost_evict(p[key].ctypes.data,p[key].nbytes)
                        flush=time.perf_counter_ns()-before if mode=='parameter_evicted' else 0
                        result=call_group(fmt,t,True)
                        samples.append(dict(mode=mode,id=rec['id'],position=rec['positions'][t],format=fmt,order=list(order),flush_ns=flush,**result))
                print(json.dumps(dict(profile=label,mode=mode,id=rec['id'],groups=len(samples))),flush=True)
        final=[[int(binder(i,j)) for j in range(5)] for i in range(3)]
        r=dict(schema='ENCODED_ORGANS_WAIT_WORKER_V1',profile=label,policy=policy,requested_blocktime=int(block),effective_blocktime=blocktime,
            initial_workers=initial,final_workers=final,samples=samples,summary=summary(samples,np),exact_cells=cells,
            cpu_start=cpu_start,cpu_end=proc.cpu_times()._asdict(),seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            all_outputs_match_parent=True,scope=SCOPE)
        assert not a.out.exists();write(a.out,r);receipt.update(exit_code=0,error=None,result=extent(a.out));runtime.close()
    except BaseException as error:receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


def pipeline(a,proc,start,guard):
    import numpy as np,psutil
    assert not FAMILY.exists()
    for peer in psutil.process_iter(['pid','name','cmdline']):
        try:
            if peer.pid!=proc.pid and (peer.info['name'] or '').lower().startswith('python'):
                assert not any('native_expert_scaling' in x for x in (peer.info['cmdline'] or [])),peer.info
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    frozen=[Path(__file__),B/'encoded_organs_wait.c',DOC/'ENCODED_ORGANS_WAIT_PROTOCOL_20261014.md']
    for p in frozen:
        assert subprocess.check_output(['git','show',a.freeze+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n')
    gbp=DOC/'contextual_generator_cost_binding_20261014.json';qbp=DOC/'contextual_generator_codec_20261014_binding.json'
    rbp=DOC/'native_bf_activation_lut_binding_v2_20261014.json';gb,qb,rb=[cg.load(p) for p in (gbp,qbp,rbp)]
    qp=DOC/'contextual_generator_codec_20261014_result.json';qr=cg.load(qp)
    gp=DOC/'contextual_generator_cost_result_20261014.json';gr=cg.load(gp)
    closure=DOC/'contextual_generator_codec_20261014_closure.json';cl=cg.load(closure)
    assert cl['report']==extent(qp) and extent(cl['audit']['path'])==cl['audit'] and cl['binding']==extent(qbp)
    compiler=Path(qb['compile']['command'][0])
    paths=frozen+[gbp,qbp,rbp,qp,gp,closure,Path(cl['audit']['path']),B/'contextual_generator_codec.c',B/'physical_worker_organs.c',
        B/'contextual_generator_cost.c',B/'integer_readout.c',B/'meth388_switch_thread_binding.h',Path(sys.executable),Path(sys.executable).parent/'python312.dll',compiler]
    paths += [Path(e['path']) for e in qb['inputs']+qb['generated_inputs']]
    paths += [Path(x['file']['path']) for x in qr['records']+gr['records']]
    paths += [Path(x['file']['path']) for x in rb['exports']]
    paths += [Path(x['file']['path']) for row in rb['reference_records'] for x in row['files']]
    pins=[]
    for p in dict.fromkeys(paths):guard();pins.append(extent(p))
    lookup={e['path']:e for e in pins}
    expected=qb['inputs']+qb['generated_inputs']+[x['file'] for x in qr['records']+gr['records']]+[x['file'] for x in rb['exports']]+[x['file'] for row in rb['reference_records'] for x in row['files']]
    for e in expected:assert all(lookup[e['path']][k]==e[k] for k in ('path','bytes','sha256'))
    FAMILY.mkdir();dll=FAMILY/'organs.dll';log=FAMILY/'compile.log'
    command=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off','-fopenmp','-shared',str(B/'encoded_organs_wait.c'),'-o',str(dll),'-lm']
    co=subprocess.run(command,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(co.stdout+co.stderr);assert co.returncode==0,log.read_text(errors='replace')
    pins += [extent(dll),extent(log)]
    b=dict(schema='ENCODED_ORGANS_WAIT_BINDING_V1',freeze=a.freeze,inputs=pins,generator_binding=extent(gbp),codec_binding=extent(qbp),
        readout_binding=extent(rbp),generator_report=extent(gp),codec_report=extent(qp),compile=dict(command=command,dll=extent(dll),log=extent(log)),
        profiles=PROFILES,scope=SCOPE,limits=dict(seconds=1800,worker_seconds=600,worker_OS_bytes=4<<30,output_bytes=32<<20))
    bp=DOC/(STEM+'_binding.json');assert not bp.exists();write(bp,b);results=[];children=[]
    for label,policy,block in PROFILES:
        guard();result=FAMILY/(label+'.json');stdout=FAMILY/(label+'.stdout');stderr=FAMILY/(label+'.stderr')
        with stdout.open('xb') as fo,stderr.open('xb') as fe:
            env=os.environ.copy();env.update(OMP_WAIT_POLICY=policy,KMP_BLOCKTIME=block,KMP_SETTINGS='1')
            cmd=[sys.executable,'-I','-S','-B','-X','utf8',str(Path(__file__)),'--worker',label,'--binding',str(bp),'--out',str(result)]
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=fo,stderr=fe,creationflags=0x00000008);watched=psutil.Process(child.pid);born=watched.create_time();begin=time.monotonic();peak=0
            try:
                while child.poll() is None:
                    guard();assert time.monotonic()-begin<=600,'worker wall cap'
                    try:peak=max(peak,watched.memory_info().peak_wset)
                    except psutil.NoSuchProcess:pass
                    assert peak<=4<<30,'worker OS cap';time.sleep(.2)
            except BaseException:
                if child.poll() is None:child.kill();child.wait()
                raise
        terminal=cg.load(result.with_suffix('.terminal.json'))
        children.append(dict(pid=child.pid,creation_time=born,exit_code=child.returncode,peak_sampled_OS=peak,terminal=extent(result.with_suffix('.terminal.json'))))
        assert child.returncode==terminal['exit_code']==0 and terminal['result']==extent(result)
        assert terminal['pid']==child.pid and abs(terminal['creation_time']-born)<.001
        r=cg.load(result);assert r['profile']==label and r['effective_blocktime']==int(block)
        assert len(r['samples'])==384 and r['summary']==summary(r['samples'],np) and r['all_outputs_match_parent']
        assert r['exact_cells']==dict(generator=188731008,readout=60162048)
        rows=gb['records'];expected_samples=[(mode,rec['id'],rec['positions'][t],fmt,list(ORDERS[(t+ri*16+mi*64)%6]))
            for mi,mode in enumerate(('resident','parameter_evicted')) for ri,rec in enumerate(rows) for t in range(16) for fmt in ORDERS[(t+ri*16+mi*64)%6]]
        assert [(x['mode'],x['id'],x['position'],x['format'],x['order']) for x in r['samples']]==expected_samples
        for x in r['samples']:
            assert len(x['generator_site_ns'])==23 and len(x['readout_site_ns'])==24
            assert x['generator_ns']==sum(x['generator_site_ns']) and x['readout_ns']==sum(x['readout_site_ns'])
            assert x['whole_ns']>=x['generator_ns']+x['readout_ns'] and x['flush_ns']>=0
            assert all(v>=0 for v in x['generator_site_ns']+x['readout_site_ns'])
            assert x['mode']!='resident' or x['flush_ns']==0
        for key in ('initial_workers','final_workers'):
            workers=r[key];assert len(workers)==3 and len({x[0] for x in workers})==3 and len({x[1] for x in workers})==3
            assert all(x[1]>0 and x[1]&(x[1]-1)==0 and x[2]==0 and x[3]>=3 for x in workers)
        settings=stderr.read_text(errors='replace');assert 'KMP_BLOCKTIME' in settings and 'OMP_WAIT_POLICY' in settings
        assert 'Effective settings:' in settings
        effective=settings.split('Effective settings:',1)[1]
        assert re.search(r"\bOMP_WAIT_POLICY\s*=\s*['\"]?"+policy+r"\b",effective)
        assert re.search(r"\bKMP_BLOCKTIME\s*=\s*['\"]?"+block+r"\b",effective)
        results.append(dict(profile=label,report=extent(result),summary=r['summary'],exact_cells=r['exact_cells'],stderr=extent(stderr),stdout=extent(stdout)))
        print(json.dumps(dict(profile=label,closed=True,whole_median_ms={mode:{fmt:values['whole_ns']['median']/1e6
            for fmt,values in table.items()} for mode,table in r['summary'].items()})),flush=True)
    decisions={}
    for mode in ('resident','parameter_evicted'):
        decisions[mode]={}
        for fmt in (16,8,4):
            values=[x['summary'][mode][str(fmt)]['whole_ns']['median'] for x in results]
            drift=max(values[0],values[2])/min(values[0],values[2]);decisions[mode][str(fmt)]=dict(passive_drift=drift,
                drift_pass=drift<=1.10,active_median_ns=values[1],active_partial_20ms_pass=values[1]<=20_000_000,
                active_vs_passive_mean=values[1]/((values[0]+values[2])/2))
    check(pins,guard);guard()
    for x in results:
        for k in ('report','stderr','stdout'):assert extent(x[k]['path'])==x[k]
    out=DOC/(STEM+'_result.json');assert not out.exists()
    r=dict(schema='ENCODED_ORGANS_WAIT_RESULT_V1',binding=extent(bp),profiles=results,children=children,decisions=decisions,
        all1152_groups_ledger_audited=True,all_parent_outputs_exact=True,GPU_calls=0,RESERVED_queries=0,scope=SCOPE,
        seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset);write(out,r)
    closure_path=DOC/(STEM+'_closure.json');assert not closure_path.exists()
    write(closure_path,dict(report=extent(out),binding=extent(bp),all_input_output_hashes_exact=True,scope=SCOPE,
        family_files=sum(1 for p in FAMILY.rglob('*') if p.is_file()),family_bytes=sum(p.stat().st_size for p in FAMILY.rglob('*') if p.is_file()),goal_status='ACTIVE_INCOMPLETE'))
    return dict(report=extent(out),closure=extent(closure_path),decisions=decisions)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze');parser.add_argument('--worker',choices=[x[0] for x in PROFILES]);parser.add_argument('--binding',type=Path);parser.add_argument('--out',type=Path);a=parser.parse_args()
    if a.worker:worker(a)
    else:
        import psutil
        proc=psutil.Process();start=time.monotonic();out=DOC/(STEM+'_pipeline.json');terminal=out.with_suffix('.terminal.json');owned=out.with_suffix('.owned_process.json')
        assert not out.exists() and not terminal.exists() and not owned.exists()
        receipt=dict(pid=proc.pid,creation_time=proc.create_time(),command=sys.orig_argv,seconds_cap=1800);write(owned,receipt)
        def guard():
            assert time.monotonic()-start<=1800,'pipeline cap';assert proc.memory_info().peak_wset<=4<<30
            if FAMILY.exists():assert sum(p.stat().st_size for p in FAMILY.rglob('*') if p.is_file())<=32<<20
        try:
            r=pipeline(a,proc,start,guard);guard();write(out,r);receipt.update(exit_code=0,error=None,result=extent(out))
        except BaseException as error:receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
        finally:receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset);write(terminal,receipt)
