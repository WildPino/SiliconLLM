"""Native own-state token branch,complete history and incremental qualification."""
import argparse,ctypes,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from full_channel_native import interface as prefix_interface,buffers as prefix_buffers,ptrs
START=time.monotonic()

def interface(b):
    dll=ctypes.CDLL(b['compile']['dll']['path']);abi=dll.ts_abi;abi.argtypes=[ctypes.c_int];abi.restype=ctypes.c_int;assert [abi(i) for i in range(7)]==[65537,6704,3072,96,3584,3,2048]
    create=dll.ts_create;create.argtypes=[];create.restype=ctypes.c_void_p;destroy=dll.ts_destroy;destroy.argtypes=[ctypes.c_void_p];destroy.restype=None
    block=dll.ts_block;block.argtypes=[ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*12;block.restype=ctypes.c_int;snapshot=dll.ts_snapshot;snapshot.argtypes=[ctypes.c_void_p]*4;snapshot.restype=ctypes.c_int;return dll,create,destroy,block,snapshot

def arrays(n,np):return dict(Bp=np.empty((n,96),'f4'),Cp=np.empty((n,96),'f4'),latent=np.empty((n,3072),'f4'),fields=np.empty((n,6704),'u2'),norm=np.empty((n,3072),'u2'),output=np.empty((n,2048),'u2'),r=np.empty((n,3072),'i2'),dots=np.empty((n,2048),'i8'),a=np.empty(n,'f8'),raw=np.empty((n,2048),'f8'))

def arguments(z,lo,hi,np):
    f=ptrs([z[k][lo:hi] for k in ['Bp','Cp','latent']]);bits=ptrs([z[k][lo:hi] for k in ['fields','norm','output']]);return [f,bits]+[z[k][lo:hi].ctypes.data for k in ['r','dots','a','raw']]

def prepare(a):
    import numpy as np,psutil
    rp=DOC/'token_generator_compiler_result_20261013.json';ap=DOC/'token_generator_compiler_adjudication_20261013.json';cp=DOC/'token_generator_compiler_session_closure_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());cl=json.loads(cp.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and cl['eligible_for_native_composition'];gb=json.loads(Path(r['binding']['path']).read_bytes());trp=DOC/'true_input_temporal_result_20261013.json';tap=DOC/'true_input_temporal_adjudication_20261013.json';tr=json.loads(trp.read_bytes());ta=json.loads(tap.read_bytes());assert ta['decision']=='INDEPENDENT_STORED_PASS' and ta['report']==extent(trp) and tr['temporal_all_pass'];tb=json.loads(Path(tr['binding']['path']).read_bytes());irp=DOC/'integer_readout_result_20261013.json';iap=DOC/'integer_readout_adjudication_v3_20261013.json';ir=json.loads(irp.read_bytes());ia=json.loads(iap.read_bytes());assert ia['decision']=='INDEPENDENT_STORED_PASS' and ia['report']==extent(irp) and ir['additional_all_pass'];ib=json.loads(Path(ir['binding']['path']).read_bytes());compiler=Path(ib['compile']['command'][0]);stp=DOC/'token_state_storage_20261013.json';st=json.loads(stp.read_text(encoding='utf-8-sig'));assert not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(st['new_family_directory']).resolve();src=B/'token_state.c';params=next(q['parameters'] for q in tb['exports'] if q['site']==0);packet=next(q['file'] for q in ib['packets'] if q['site']==0)
    paths=[Path(__file__),src,B/'token_state_audit.py',B/'full_channel_native.c',B/'full_channel_native.py',B/'integer_readout.c',B/'integer_readout.py',B/'composed_operator_mixed.py',B/'falcon_auxiliary_s6_transport.py',DOC/'TOKEN_STATE_PROTOCOL_20261013.md',stp,rp,ap,cp,Path(r['binding']['path']),trp,tap,Path(tr['binding']['path']),irp,iap,Path(ir['binding']['path']),Path(params['path']),Path(packet['path']),Path(r['tables']['F0']['path']),Path(r['parameters']['path']),Path(tb['native_dll']['path']),compiler,Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)]
    nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))+list(compiler.parent.glob('*.dll'))+list(compiler.parent.glob('*lld*.exe'))
    paths+=[Path(q['file']['path']) for q in r['records']]+[Path(q['file']['path']) for q in tr['records']+ir['records'] if q['site']==0]
    for rec in gb['records']:paths.append(Path(rec['sites'][0]['fields']['output']['path']))
    paths+=[ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=[extent(p) for p in dict.fromkeys(paths)];a.directory.mkdir();dll=a.directory/'token_state.dll';log=a.directory/'compile.log';cmd=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off','-shared',str(src.resolve()),'-o',str(dll.resolve()),'-lm'];co=subprocess.run(cmd,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(co.stdout+co.stderr);assert co.returncode==0;inputs+=[extent(dll),extent(log)]
    for e in inputs:assert extent(e['path'])==e
    proc=psutil.Process();assert time.monotonic()-START<900 and proc.memory_info().peak_wset<6<<30 and not proc.children(recursive=True)
    b=dict(schema='TOKEN_STATE_BINDING_V1',freeze=a.freeze,python=gb['python'],versions=dict(numpy=np.__version__,psutil=psutil.__version__),generator_report=extent(rp),generator_adjudication=extent(ap),table=r['tables']['F0'],conv_parameters=r['parameters'],core_parameters=params,readout_packet=packet,prefix_native_dll=tb['native_dll'],records=gb['records'],generator_records=r['records'],temporal_records=[q for q in tr['records'] if q['site']==0],integer_records=[q for q in ir['records'] if q['site']==0],inputs=inputs,compile=dict(command=cmd,source=extent(src),dll=extent(dll),log=extent(log),exit_code=0),output_directory=str(a.directory/'run'),numeric=dict(field_relative=1e-6,projected_relative=1e-6,scan_relative=1e-4,state_relative=1e-4,controlled_relative=1e-4,metadata_absolute_or_relative=1e-8,exact_prefix_core_bits=True,exact_incremental_bits=True,exact_integer_and_BF16=True),criteria=dict(own_output_each_case=.01,final_state_each_case=.01,donor_mean=.1,donor_worst=.2,donor_centered=.1,donor_domain=.2),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=2<<30),scope='Initial native SSM branch consumes uint32 tokens/F0 table with persistent conv and own trained3072x96 state/full skip/own gate-RMS/new C quantizer/accepted integer readout. Full48 history versus token-step continuation and qualified prefix kernel/all stored fields/core/output. No target denominator/reset within history/source model forward/GPU/T4/RESERVED/FFN/attention/chatbot/useful-n/model rate admission.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)

def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and dict(numpy=np.__version__,psutil=psutil.__version__)==b['versions'] and not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(b['output_directory']).resolve();proc=psutil.Process();proc.cpu_affinity([0]);rows=[];stage='startup';contexts=[];receipt=dict(schema='TOKEN_STATE_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def dec(x):return (x.astype('u4')<<16).view('f4').astype('f8')
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def metrics(x,y,pos):return dict(relative_RMS=rel(x,y),centered_relative_RMS=rel(x-x.mean(0),y-y.mean(0)),label_relative_RMS=rel(x[pos],y[pos]))
    def snapshot(ctx):
        state=np.empty((3072,96),'f4');ring=np.empty((3,3584),'u2');seen=np.zeros(1,'u8');assert snap(ctx,state.ctypes.data,ring.ctypes.data,seen.ctypes.data)==0;return state,ring,int(seen[0])
    try:
        check();a.directory.mkdir();dll,create,destroy,block,snap=interface(b);last=dll.ts_last;last.argtypes=[ctypes.c_void_p];last.restype=ctypes.c_uint32;table=np.memmap(b['table']['path'],dtype='u2',mode='r',shape=(65537,6704))
        with np.load(b['core_parameters']['path'],allow_pickle=False) as z:params={k:z[k].copy() for k in z.files}
        with np.load(b['conv_parameters']['path'],allow_pickle=False) as z:conv={k:z[k].copy() for k in z.files}
        with np.load(b['readout_packet']['path'],allow_pickle=False) as z:q=z['q'].copy();scale=z['scale'].copy()
        w=ptrs([params[k] for k in ['Ws','Vs','Alog','D','gamma']]);wp=ptrs([params[k] for k in ['Ws','Vs','Alog','D','gamma','O','in_proj']]);cp=ptrs([conv[k] for k in ['cw','cb','db']]);prefix_dll,prefix=prefix_interface(dict(native_dll=b['prefix_native_dll']))
        controls=dict(invalid_ID_rejected=None,token65536_supported=None)
        for token in [65537,65536]:
            ctx=create();assert ctx;contexts.append(ctx);zz=arrays(1,np);control_ids=np.array([token],'u4');rc=block(ctx,1,control_ids.ctypes.data,table.ctypes.data,w,cp,q.ctypes.data,scale.ctypes.data,*arguments(zz,0,1,np));st,rg,count=snapshot(ctx)
            if token==65537:controls['invalid_ID_rejected']=bool(rc==1 and count==0 and not st.any() and not rg.any() and last(ctx)==4294967295)
            else:controls['token65536_supported']=bool(rc==0 and count==1 and last(ctx)==65536 and np.isfinite(zz['raw']).all())
            destroy(ctx);contexts.remove(ctx)
        for rec in b['records']:
            stage=rec['id'];guard();n=rec['history'];ids=np.array(rec['input_ids'],'u4');z=arrays(n,np);ctx=create();assert ctx;contexts.append(ctx);assert block(ctx,n,ids.ctypes.data,table.ctypes.data,w,cp,q.ctypes.data,scale.ctypes.data,*arguments(z,0,n,np))==0;final,ring,seen=snapshot(ctx);destroy(ctx);contexts.remove(ctx);assert seen==n
            # Separate one-token calls preserve SAME context;no teacher resets.
            inc=arrays(n,np);ctx=create();assert ctx;contexts.append(ctx);trace=np.empty(n,'u4')
            for t in range(n):
                if t%64==0:guard()
                assert block(ctx,1,ids[t:t+1].ctypes.data,table.ctypes.data,w,cp,q.ctypes.data,scale.ctypes.data,*arguments(inc,t,t+1,np))==0;trace[t]=last(ctx)
            fi,ri,si=snapshot(ctx);destroy(ctx);contexts.remove(ctx);incremental=all(np.array_equal(z[k].view('u1'),inc[k].view('u1')) for k in z) and np.array_equal(final.view('u4'),fi.view('u4')) and np.array_equal(ring,ri) and si==n and np.array_equal(trace,ids);del inc
            compiled=next(e for e in b['generator_records'] if e['id']==rec['id']);field_errors={};field_differences={};names=['gate','x','B','C','delta'];bounds=[0,3072,6144,6400,6656,6704]
            with np.load(compiled['file']['path'],allow_pickle=False) as ref:
                for i,k in enumerate(names):val=z['fields'][:,bounds[i]:bounds[i+1]];field_errors[k]=rel(dec(val),dec(ref[k]));field_differences[k]=int(np.count_nonzero(val!=ref[k]))
            inp=[np.ascontiguousarray(dec(z['fields'][:,bounds[names.index(k)]:bounds[names.index(k)+1]]),dtype='f4') for k in ['x','B','C','delta','gate']];pf,pb=prefix_buffers(n,np);tt=np.zeros(4,'f8');unused=np.zeros(n,'f8');assert prefix(n,1,wp,ptrs(inp),unused.ctypes.data,ptrs(pf),ptrs(pb),tt.ctypes.data,0)==0;prefix_exact=all(np.array_equal(z[k].view('u4'),v.view('u4')) for k,v in [('Bp',pf[0]),('Cp',pf[1]),('latent',pf[2])]) and np.array_equal(final.view('u4'),pf[3].view('u4')) and np.array_equal(z['norm'],pb[1])
            old=next(e for e in b['temporal_records'] if e['id']==rec['id']);ir=next(e for e in b['integer_records'] if e['id']==rec['id'])
            with np.load(old['file']['path'],allow_pickle=False) as rr:state_error=rel(final.astype('f8'),rr['final_state'].astype('f8'))
            with np.load(ir['file']['path'],allow_pickle=False) as rr:mm_base=metrics(dec(z['output']),dec(rr['output_BF16']),rec['positions'])
            se=rec['sites'][0]['fields']['output'];source=np.fromfile(se['path'],'u2').reshape(se['shape']);mm_source=metrics(dec(z['output']),dec(source),rec['positions']);ng=dict(fields=all(v<=b['numeric']['field_relative'] for v in field_errors.values()),prefix_core_bits=bool(prefix_exact),incremental_bits=bool(incremental));quality=dict(own_output=mm_base['relative_RMS']<=b['criteria']['own_output_each_case'],final_state=state_error<=b['criteria']['final_state_each_case']);path=a.directory/(rec['id']+'.npz');np.savez(path,**z,final_state=final,final_conv=ring,token_IDs=trace,implicit_masses=np.ones(n,'f4'));rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,file=extent(path),generator_reference=compiled['file'],temporal_reference=old['file'],integer_reference=ir['file'],field_reference_relative=field_errors,field_reference_bit_differences=field_differences,numeric_gates=ng,quality_gates=quality,metrics=dict(source=mm_source,baseline=mm_base),final_state_relative=state_error,seen=seen));print(json.dumps(dict(complete=len(rows),numeric_gates=ng,quality_gates=quality,seconds=time.monotonic()-START)),flush=True)
        check();guard();aggregates={};donor={}
        for split in ['FIT','DEV']:
            rr=[r for r in rows if r['split']==split];assert len(rr)==24;aggregates[split]={}
            for kind in ['source','baseline']:
                vals=[r['metrics'][kind]['relative_RMS'] for r in rr];dm={d:float(np.mean([r['metrics'][kind]['relative_RMS'] for r in rr if r['domain']==d])) for d in sorted({r['domain'] for r in rr})};aggregates[split][kind]=dict(mean=float(np.mean(vals)),worst=max(vals),centered_mean=float(np.mean([r['metrics'][kind]['centered_relative_RMS'] for r in rr])),domain_means=dm,worst_domain_mean=max(dm.values()))
            d=aggregates[split]['source'];donor[split]=dict(mean=d['mean']<=.1,worst=d['worst']<=.2,centered=d['centered_mean']<=.1,domain=d['worst_domain_mean']<=.2)
        result=dict(schema='TOKEN_STATE_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,controls=controls,aggregates=aggregates,donor_gates=donor,numeric_all_pass=all(all(q['numeric_gates'].values()) for q in rows) and all(controls.values()),quality_all_pass=all(all(q['quality_gates'].values()) for q in rows),before_after_hashes_exact=True,token_steps=22547,initial_state_zero_histories=48,source_model_forwards=0,GPU_calls=0,optimizer_updates=0,output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),numeric_all_pass=result['numeric_all_pass'],quality_all_pass=result['quality_all_pass'],donor_gates=donor)),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:
        for ctx in contexts:destroy(ctx)
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for k in ['binding','directory','out']:p.add_argument('--'+k,type=Path)
    for k in ['binding-sha','freeze']:p.add_argument('--'+k)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='TOKEN_STATE_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as er:
            import traceback
            receipt.update(exit_code=1,error=repr(er),traceback=traceback.format_exc());raise
        finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
