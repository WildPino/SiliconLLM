"""Export source-only I8/I4 generators, charge native decode, audit all outputs."""
import argparse,ctypes,itertools,json,os,subprocess,sys,time,traceback
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0')
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import contextual_generator_cost as cg
from physical_worker_organs import function,exact,distribution,check
STEM='contextual_generator_codec_20261014';FAMILY=Path('E:/SiliconLLM/results/native_expert_scaling')/STEM
ORDERS=list(itertools.permutations((16,8,4)))
SCOPE='Source-only full23 generator codecs on supplied actual own-history h; no joined decoder, new model quality, useful50, useful-n or physical DRAM claim.'


def quantize(w,fmt,np):
    x=cg.decode(w,np);groups=x if fmt==8 else x.reshape(6704,16,128)
    peak=np.max(np.abs(groups),axis=-1);scale=np.asarray(peak/np.float32(127 if fmt==8 else 7),'f4')
    ratio=np.divide(groups.astype('f8'),scale[...,None].astype('f8'),out=np.zeros_like(groups,dtype='f8'),where=scale[...,None]!=0)
    q=np.clip(np.rint(ratio),-127 if fmt==8 else -7,127 if fmt==8 else 7).astype('i1').reshape(6704,2048)
    if fmt==8:codes=q
    else:
        unsigned=(q.astype('i2')+7).astype('u1');codes=unsigned[:,::2]|(unsigned[:,1::2]<<4)
    return np.ascontiguousarray(codes),np.ascontiguousarray(scale)


def mapped(packet,np):
    return (np.memmap(packet['codes']['path'],mode='r',dtype='i1' if packet['format']==8 else 'u1',shape=tuple(packet['codes_shape'])),
            np.memmap(packet['scales']['path'],mode='r',dtype='f4',shape=tuple(packet['scale_shape'])))


def summarize(samples,np):
    return {mode:{str(fmt):dict(whole_ns=distribution([x['whole_ns'] for x in samples if x['mode']==mode and x['format']==fmt],np),
        flush_ns=distribution([x['flush_ns'] for x in samples if x['mode']==mode and x['format']==fmt],np))
        for fmt in (16,8,4)} for mode in ('resident','parameter_evicted')}


def controls(dll,choose,np):
    probe=function(dll,'qc_probe',[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p])
    q=np.arange(-127,128,dtype='i2').astype('i1');out=np.empty_like(q)
    assert probe(8,q.size,q.ctypes.data,out.ctypes.data)==0;exact(q,out,np)
    nib=np.arange(16,dtype='u1');packed=nib[::2]|(nib[1::2]<<4);out=np.empty(16,'i1')
    assert probe(4,15,packed.ctypes.data,out.ctypes.data)==0;exact(out[:15],(nib[:15].astype('i2')-7).astype('i1'),np)
    assert probe(4,16,packed.ctypes.data,out.ctypes.data)==4
    bad=np.array([128],'u1');assert probe(8,1,bad.ctypes.data,out.ctypes.data)==4
    assert probe(5,1,bad.ctypes.data,out.ctypes.data)==1
    assert choose(3)==0
    step=function(dll,'qc_step',[ctypes.c_int]+[ctypes.c_void_p]*7)
    setter=function(dll,'qc_set_control_for_test',[ctypes.c_uint],None)
    weight=np.zeros((6704,2048),'i1');scale=np.zeros(6704,'f4');h=np.zeros(2048,'u2');mup=np.full(6704,0x3f80,'u2')
    raw=np.full(6704,123.,'f4');gg=np.full(6704,123,'u2');fields=gg.copy()
    args=(8,weight.ctypes.data,scale.ctypes.data,h.ctypes.data,mup.ctypes.data,raw.ctypes.data,gg.ctypes.data,fields.ctypes.data)
    h[0]=0x7f80;assert step(*args)==4;h[0]=0
    setter(0x9f80)
    try:assert step(*args)==3
    finally:setter(0x1f80)
    assert np.all(raw==123.) and np.all(gg==123) and np.all(fields==123)
    assert step(*args)==0 and not raw.any() and not gg.any() and not fields.any()
    return dict(I8_codes=255,I4_codes=15,reserved_codes_rejected=True,bad_format_rejected=True,
        nonfinite_h_rejected=True,changed_FP_rejected_without_output_mutation=True,zero_row_scale_exact=True)


def run(a,proc,start,guard):
    import numpy as np,psutil
    assert not FAMILY.exists()
    frozen=[Path(__file__),B/'contextual_generator_codec_audit.py',B/'contextual_generator_codec.c',DOC/'CONTEXTUAL_GENERATOR_CODEC_PROTOCOL_20261014.md']
    for p in frozen:
        blob=subprocess.check_output(['git','show',a.freeze+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n')
    for peer in psutil.process_iter(['pid','name','cmdline']):
        try:
            if peer.pid!=proc.pid and (peer.info['name'] or '').lower().startswith('python'):
                assert not any('native_expert_scaling' in x for x in (peer.info['cmdline'] or [])),peer.info
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    gbp=DOC/'contextual_generator_cost_binding_20261014.json';gb=cg.load(gbp)
    rp=DOC/'contextual_generator_cost_result_20261014.json';old=cg.load(rp)
    assert old['binding']==extent(gbp)
    closure_path=DOC/'contextual_generator_cost_session_closure_20261014.json';cl=cg.load(closure_path)
    assert cl['report']==extent(rp)
    assert extent(cl['audit']['path'])==cl['audit']
    compiler=Path(gb['compile']['command'][0])
    paths=frozen+[gbp,rp,closure_path,Path(cl['audit']['path']),Path(gb['source']['path']),
        B/'physical_worker_organs.c',B/'physical_worker_organs.py',B/'contextual_generator_cost.c',B/'contextual_generator_cost.py',
        B/'integer_readout.c',B/'meth388_switch_thread_binding.h',B/'falcon_auxiliary_s6_transport.py',
        compiler,Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(gb['support']['path']),Path(gb['compile']['dll']['path'])]
    paths += [Path(x['file']['path']) for x in old['records']]
    paths += [Path(e['trace']['path']) for r in gb['records'] for e in r['traces']]
    for module in (np,psutil):paths.extend(p for p in Path(module.__file__).parent.rglob('*') if p.is_file() and p.suffix in ('.py','.pyd','.dll'))
    paths.extend((Path(np.__file__).parent.parent/'numpy.libs').glob('*.dll'));paths.extend(compiler.parent.glob('*.dll'))
    paths.extend(Path(m.__file__) for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).is_file())
    paths += [ROOT/p for p in ('benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    pins=[]
    for p in dict.fromkeys(paths):guard();pins.append(extent(p))
    known=[gb['source'],gb['support'],gb['compile']['dll']]+[x['file'] for x in old['records']]+[e['trace'] for r in gb['records'] for e in r['traces']]
    indexed={e['path']:e for e in pins}
    for e in known:assert all(indexed[e['path']][k]==e[k] for k in ('path','bytes','sha256'))
    FAMILY.mkdir();source=cg.weights(gb,np);packets=[];export_start=time.monotonic()
    for fmt in (8,4):
        for site,w in enumerate(source,1):
            guard();assert np.isfinite(cg.decode(w,np)).all()
            codes,scales=quantize(w,fmt,np)
            cp=FAMILY/f'site{site:02d}.q{fmt}.codes';sp=FAMILY/f'site{site:02d}.q{fmt}.scales'
            codes.tofile(cp);scales.tofile(sp)
            packets.append(dict(site=site,format=fmt,codes=extent(cp),scales=extent(sp),codes_shape=list(codes.shape),scale_shape=list(scales.shape)))
    conversion_seconds=time.monotonic()-export_start
    dll_path=FAMILY/'codec.dll';log=FAMILY/'compile.log'
    command=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off','-fopenmp','-shared',str(B/'contextual_generator_codec.c'),'-o',str(dll_path),'-lm']
    co=subprocess.run(command,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(co.stdout+co.stderr)
    assert co.returncode==0,log.read_text(errors='replace')
    new_pins=[extent(dll_path),extent(log)]+[e[k] for e in packets for k in ('codes','scales')]
    b=dict(schema='CONTEXTUAL_GENERATOR_CODEC_BINDING_V1',freeze=a.freeze,inputs=pins,generated_inputs=new_pins,
        parent_binding=extent(gbp),parent_report=extent(rp),packets=packets,compile=dict(command=command,dll=extent(dll_path),log=extent(log)),
        formats=[16,8,4],orders=[list(x) for x in ORDERS],records=gb['records'],modes=['resident','parameter_evicted'],
        conversion_seconds=conversion_seconds,conversion_OS_peak=proc.memory_info().peak_wset,workers=3,
        versions=dict(numpy=np.__version__,psutil=psutil.__version__),limits=dict(seconds=3600,OS_bytes=4<<30,output_bytes=1<<30),
        gates=dict(decoded_F64_raw_relative=1e-4,historical_source_field_relative=.01),scope=SCOPE)
    bp=DOC/(STEM+'_binding.json');assert not bp.exists();write(bp,b)
    print(json.dumps(dict(phase='bound',binding=extent(bp),conversion_seconds=conversion_seconds)),flush=True)
    runtime=os.add_dll_directory(str(compiler.parent));dll=ctypes.CDLL(str(dll_path))
    assert function(dll,'po_setup',[])()==0
    choose=function(dll,'po_threads',[ctypes.c_int]);assert choose(3)==0
    binder=function(dll,'po_binding',[ctypes.c_int,ctypes.c_int],ctypes.c_uint64)
    first=[[int(binder(i,j)) for j in range(5)] for i in range(6)]
    negative=controls(dll,choose,np)
    baseline=function(dll,'po_generator',[ctypes.c_void_p]*6)
    codec=function(dll,'qc_step',[ctypes.c_int]+[ctypes.c_void_p]*7)
    _,support=cg.native(gb);mup=np.asarray(gb['MuP_bits'],'u2')
    mapped_packets={fmt:[mapped(p,np) for p in packets if p['format']==fmt] for fmt in (8,4)}
    samples=[];records=[]
    for mi,mode in enumerate(b['modes']):
        for ri,rec in enumerate(gb['records']):
            guard();h,truth=cg.record_inputs(rec,np)
            output={fmt:dict(raw=np.empty((23,16,6704),'f4'),gg=np.empty((23,16,6704),'u2'),fields=np.empty((23,16,6704),'u2')) for fmt in (16,8,4)}
            calls={}
            for fmt in (16,8,4):
                calls[fmt]=[]
                for t in range(16):
                    group=[]
                    for i in range(23):
                        tail=tuple(x.ctypes.data for x in (h[i,t],mup,output[fmt]['raw'][i,t],output[fmt]['gg'][i,t],output[fmt]['fields'][i,t]))
                        group.append((source[i].ctypes.data,*tail) if fmt==16 else (fmt,mapped_packets[fmt][i][0].ctypes.data,mapped_packets[fmt][i][1].ctypes.data,*tail))
                    calls[fmt].append(group)
                fn=baseline if fmt==16 else codec
                for call in calls[fmt][0]:assert fn(*call)==0
            for t in range(16):
                order=ORDERS[(t+ri*16+mi*64)%6]
                for fmt in order:
                    guard();fb=time.perf_counter_ns()
                    if mode=='parameter_evicted':
                        owners=source if fmt==16 else [x for pair in mapped_packets[fmt] for x in pair]
                        for x in owners:support.cost_evict(x.ctypes.data,x.nbytes)
                        support.cost_evict(mup.ctypes.data,mup.nbytes)
                    flush=time.perf_counter_ns()-fb if mode=='parameter_evicted' else 0
                    times=[];rcs=[];fn=baseline if fmt==16 else codec;begin=time.perf_counter_ns()
                    for call in calls[fmt][t]:
                        before=time.perf_counter_ns();rc=fn(*call);after=time.perf_counter_ns();times.append(after-before);rcs.append(rc)
                    elapsed=time.perf_counter_ns()-begin;assert rcs==[0]*23
                    samples.append(dict(mode=mode,id=rec['id'],position=rec['positions'][t],format=fmt,order=list(order),whole_ns=elapsed,site_ns=times,flush_ns=flush))
            original=next(x for x in old['records'] if x['id']==rec['id'] and x['mode']==mode)
            with np.load(original['file']['path'],allow_pickle=False) as z:
                for key,value in output[16].items():exact(value,z[key],np)
            errors={str(fmt):[cg.controls(output[fmt]['fields'][i],truth[i],np) for i in range(23)] for fmt in (8,4)}
            stored={f'{key}_{fmt}':v for fmt in (8,4) for key,v in output[fmt].items()}
            path=FAMILY/(mode+'.'+rec['id']+'.npz');np.savez(path,**stored)
            records.append(dict(id=rec['id'],mode=mode,file=extent(path),source_errors=errors,BF_control_exact=True))
            print(json.dumps(dict(mode=mode,id=rec['id'],samples=len(samples),seconds=time.monotonic()-start)),flush=True)
    assert choose(6)==0 and function(dll,'po_team_check',[])()==0
    last=[[int(binder(i,j)) for j in range(5)] for i in range(6)]
    r=dict(schema='CONTEXTUAL_GENERATOR_CODEC_RESULT_V1',binding=extent(bp),samples=samples,records=records,summary=summarize(samples,np),
        controls=negative,initial_workers=first,final_workers=last,GPU_calls=0,RESERVED_queries=0,
        payload_bytes={str(fmt):sum(e[k]['bytes'] for e in packets if e['format']==fmt for k in ('codes','scales')) for fmt in (8,4)},
        scope=SCOPE,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
    report=DOC/(STEM+'_result.json');assert not report.exists();write(report,r)
    from contextual_generator_codec_audit import audit
    ap=DOC/(STEM+'_audit.json');assert not ap.exists();write(ap,dict(report=extent(report),binding=extent(bp),**audit(r,b,gb,np,dll,guard)))
    check(pins+new_pins,guard)
    for e in records:assert extent(e['file']['path'])==e['file']
    assert extent(report)==cg.load(ap)['report']
    expected={e['path']:e['bytes'] for e in new_pins+[x['file'] for x in records]}
    actual={str(p.resolve()):p.stat().st_size for p in FAMILY.rglob('*') if p.is_file()};assert actual==expected
    close=DOC/(STEM+'_closure.json');assert not close.exists()
    write(close,dict(schema='CONTEXTUAL_GENERATOR_CODEC_CLOSURE_V1',report=extent(report),audit=extent(ap),binding=extent(bp),
        all_inputs_outputs_exact=True,family_files=len(actual),family_bytes=sum(actual.values()),scope=SCOPE,decision='FULL_CODEC_SCREEN_CLOSED',goal_status='ACTIVE_INCOMPLETE'))
    runtime.close();return dict(report=extent(report),audit=extent(ap),closure=extent(close),summary=r['summary'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',required=True);a=parser.parse_args()
    import psutil
    proc=psutil.Process();start=time.monotonic();out=DOC/(STEM+'_pipeline.json')
    terminal=out.with_suffix('.terminal.json');owned=out.with_suffix('.owned_process.json')
    assert not out.exists() and not terminal.exists() and not owned.exists()
    receipt=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),seconds_cap=3600,OS_bytes_cap=4<<30);write(owned,receipt)
    def guard():
        assert time.monotonic()-start<=3600,'wall limit';assert proc.memory_info().peak_wset<=4<<30,'OS limit'
        if FAMILY.exists():assert sum(p.stat().st_size for p in FAMILY.rglob('*') if p.is_file())<=1<<30,'family limit'
    try:
        result=run(a,proc,start,guard);guard();write(out,result)
        receipt.update(exit_code=0,error=None,result=extent(out));print(json.dumps(dict(pipeline=extent(out),summary=result['summary'])),flush=True)
    except BaseException as error:receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset);write(terminal,receipt)
