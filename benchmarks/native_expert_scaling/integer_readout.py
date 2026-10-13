"""Full same-p6 integer readout with separately gated source input quantization."""
import argparse,ctypes,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from ternary_readout_precision import decode,metrics,bf
from ternary_readout_native import summarize
START=time.monotonic()

def quantize(Z,np):
    a=np.max(np.abs(Z),axis=1)/32767
    ratio=np.divide(Z,a[:,None],out=np.zeros_like(Z),where=a[:,None]!=0)
    r=np.clip(np.rint(ratio),-32767,32767).astype('i2')
    nonzero=a>0;bound=float(np.max(np.abs(Z[nonzero]-r[nonzero]*a[nonzero,None])/(a[nonzero,None]*.5))) if nonzero.any() else 0.
    return r,a,dict(zero_rows=int((~nonzero).sum()),half_step_max=bound,relative_RMS=float(np.linalg.norm(Z-r*a[:,None])/max(np.linalg.norm(Z),1e-30)))

def interface(b):
    dll=ctypes.CDLL(b['compile']['dll']['path']);abi=dll.ir_abi;abi.argtypes=[ctypes.c_int];abi.restype=ctypes.c_int;assert [abi(i) for i in range(5)]==[3072,2048,128,364,32767];fn=dll.ir_history;fn.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*7;fn.restype=ctypes.c_int;return dll,fn

def prepare(a):
    import numpy as np,psutil
    rp=DOC/'ternary_readout_native_result_20261013.json';ap=DOC/'ternary_readout_native_adjudication_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and r['numeric_all_pass'] and r['additional_all_pass'] and extent(r['binding']['path'])==r['binding'];pb=json.loads(Path(r['binding']['path']).read_bytes());compiler=Path(pb['compile'][0]['command'][0]);stp=DOC/'integer_readout_storage_20261013.json';st=json.loads(stp.read_text(encoding='utf-8-sig'));assert not a.directory.exists() and not a.out.exists() and a.directory.resolve()==Path(st['new_family_directory']).resolve();src=B/'integer_readout.c'
    paths=[Path(__file__),src,B/'integer_readout_audit.py',B/'ternary_readout_native.py',B/'ternary_readout_precision.py',B/'falcon_auxiliary_s6_transport.py',DOC/'INTEGER_READOUT_PROTOCOL_20261013.md',stp,rp,ap,Path(r['binding']['path']),compiler,Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)];nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))+list(compiler.parent.glob('*.dll'))+list(compiler.parent.glob('*lld*.exe'));paths+=[Path(e['file']['path']) for e in pb['packets']]+[Path(e['native']['path']) for e in pb['records']]+[Path(e['source_output']['path']) for e in pb['records']]+[Path(e['file']['path']) for e in r['records']];paths+=[ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=[extent(p) for p in dict.fromkeys(paths)];a.directory.mkdir();dll=a.directory/'integer.dll';log=a.directory/'compile.log';cmd=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off','-shared',str(src.resolve()),'-o',str(dll.resolve()),'-lm'];co=subprocess.run(cmd,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(co.stdout+co.stderr);assert co.returncode==0;inputs+=[extent(dll),extent(log)]
    for e in inputs:assert extent(e['path'])==e
    assert time.monotonic()-START<900 and psutil.Process().memory_info().peak_wset<6<<30 and not psutil.Process().children(recursive=True)
    b=dict(schema='INTEGER_READOUT_BINDING_V1',freeze=a.freeze,python=pb['python'],versions=pb['versions'],parent_report=extent(rp),parent_adjudication=extent(ap),parent_binding=r['binding'],packets=pb['packets'],records=pb['records'],reference_records=r['records'],inputs=inputs,compile=dict(command=cmd,source=extent(src),dll=extent(dll),log=extent(log),exit_code=0),output_directory=str(a.directory/'run'),bounds=dict(weight=364,input=32767,block=128,block_abs_sum=1526680064,full_abs_sum=36640321536,exact_F64_integer_limit=2**53),numeric=dict(scale_and_score_relative=1e-12,metadata_absolute_or_relative=1e-10,activation_half_step_slack=1e-10,integer_dot_exact=True,BF16_exact=True),criteria=pb['criteria']|dict(incremental_activation_each_case=.001),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=3<<30),scope='Same p6 q/scales,explicit per-vector I16 source input approximation;one-core PMADDWD block128/int64 exact dot. Full actual histories/source gates and integer equality. No source/recurrence/GPU/T4/RESERVED/optimizer/OWN chatbot/useful-n/DRAM/model-rate admission;native cost separate after correctness.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)

def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(b['output_directory']).resolve();proc=psutil.Process();proc.cpu_affinity(list(range(6)));rows=[];controls=[];stage='startup';receipt=dict(schema='INTEGER_READOUT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def call(q,s,r,aa):
        n=len(r);dots=np.empty((n,2048),'i8');raw=np.empty((n,2048),'f8');bits=np.empty((n,2048),'u2');proc.cpu_affinity([0]);rc=fn(n,*[v.ctypes.data for v in [q,s,r,aa,dots,raw,bits]]);proc.cpu_affinity(list(range(6)));assert rc==0 and np.isfinite(raw).all();return dots,raw,bits
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    try:
        check();a.directory.mkdir();dll,fn=interface(b)
        # Fixed extrema/all lanes,zero and mutation controls BEFORE actual data.
        cs=np.ones(2048,'f8');ca=np.ones(1,'f8');cq=np.empty((2048,3072),'i2');cx=np.empty((1,3072),'i2')
        for signw,signx in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            cq.fill(signw*364);cx.fill(signx*32767);dot,_,_=call(cq,cs,cx,ca);expected=signw*signx*b['bounds']['full_abs_sum'];assert np.all(dot==expected);controls.append(dict(weight_sign=signw,input_sign=signx,exact_dot=expected,all2048=True))
        cq.fill(364);cx.fill(32767);cq[0,0]=363;dot,_,_=call(cq,cs,cx,ca);assert dot[0,0]==b['bounds']['full_abs_sum']-32767 and np.all(dot[0,1:]==b['bounds']['full_abs_sum']);controls.append(dict(single_coordinate_mutation_exact=True));cx.fill(0);dot,raw,bits=call(cq,cs,cx,ca);assert not dot.any() and not raw.any() and not bits.any();controls.append(dict(zero_input_exact=True));del cq,cx,cs,ca
        for packet in b['packets']:
            site=packet['site'];guard()
            with np.load(packet['file']['path'],allow_pickle=False) as z:q=z['q'].copy();scale=z['scale'].copy()
            assert q.dtype==np.dtype('<i2') and q.shape==(2048,3072) and np.max(np.abs(q.astype('i4')))<=364;qd=q.astype('f8')
            for e in [r for r in b['records'] if r['site']==site]:
                stage=f'{site}/{e["id"]}';guard()
                with np.load(e['native']['path'],allow_pickle=False) as z:Z=decode(z['reconstructed_denominator__normalized'],np);Y=z['reconstructed_denominator__raw_scores'].astype('f8');base=z['reconstructed_denominator__output'].copy()
                ref=next(r['file'] for r in b['reference_records'] if r['site']==site and r['id']==e['id'])
                with np.load(ref['path'],allow_pickle=False) as z:p6=z['lut_scores'].copy();p6bits=z['lut_BF16'].copy()
                qr,aa,qm=quantize(Z,np);assert np.max(np.abs(qr.astype('i4')))<=32767;dot,raw,bits=call(q,scale,qr,aa);exact=qr.astype('f8')@qd.T;assert np.array_equal(exact,np.rint(exact)) and np.max(np.abs(exact))<=b['bounds']['full_abs_sum'];expected=exact.astype('i8');mismatch=int(np.count_nonzero(dot!=expected));gold=exact*aa[:,None]*scale;f=e['source_output'];source=decode(np.fromfile(f['path'],'u2').reshape(f['shape']),np);out=decode(bits,np);mm=dict(raw_vs_native=metrics(raw,Y,e['positions'],np),BF16_vs_baseline=metrics(out,decode(base,np),e['positions'],np),BF16_vs_source=metrics(out,source,e['positions'],np));ng=dict(integer=mismatch==0,scaled_score=rel(raw,gold)<=b['numeric']['scale_and_score_relative'],BF16=bool(np.array_equal(bits,bf(gold,np))),activation_bound=qm['half_step_max']<=1+b['numeric']['activation_half_step_slack']);increment=rel(raw,p6);extra=dict(raw=mm['raw_vs_native']['relative_RMS']<=b['criteria']['additional_raw_each_case'],BF16=mm['BF16_vs_baseline']['relative_RMS']<=b['criteria']['additional_BF16_each_case'],activation=increment<=b['criteria']['incremental_activation_each_case']);p=a.directory/f'{e["id"]}.site{site:02d}.npz';np.savez(p,input_I16=qr,input_scale=aa,integer_dots=dot,scores=raw,output_BF16=bits);rows.append(dict(id=e['id'],site=site,split=e['split'],domain=e['domain'],history=e['history'],file=extent(p),metrics=mm,quantization=qm,integer_mismatches=mismatch,scaled_score_relative=rel(raw,gold),incremental_activation_relative=increment,p6_BF16_mismatches=int(np.count_nonzero(bits!=p6bits)),numeric_gates=ng,additional_gates=extra));print(json.dumps(dict(complete=len(rows),numeric_gates=ng,additional_gates=extra,seconds=time.monotonic()-START)),flush=True)
        check();guard();aa=summarize(rows,np);donor={}
        for site in [0,12,23]:
            donor[str(site)]={}
            for split in ['FIT','DEV']:
                d=aa[str(site)][split]['BF16_vs_source'];donor[str(site)][split]=dict(mean=d['relative_RMS']['mean']<=b['criteria']['donor_mean'],worst=d['relative_RMS']['worst']<=b['criteria']['donor_worst'],centered=d['centered_relative_RMS']['mean']<=b['criteria']['donor_centered'],domain=d['relative_RMS']['worst_domain_mean']<=b['criteria']['donor_domain'])
        result=dict(schema='INTEGER_READOUT_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,controls=controls,aggregates=aa,donor_gates=donor,numeric_all_pass=all(all(r['numeric_gates'].values()) for r in rows),additional_all_pass=all(all(r['additional_gates'].values()) for r in rows),before_after_hashes_exact=True,native_history_calls=144,source_forwards=0,recurrence_calls=0,optimizer_updates=0,decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),numeric_all_pass=result['numeric_all_pass'],additional_all_pass=result['additional_all_pass'])),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for k in ['binding','directory','out']:p.add_argument('--'+k,type=Path)
    for k in ['binding-sha','freeze']:p.add_argument('--'+k)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='INTEGER_READOUT_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as er:
            import traceback
            receipt.update(exit_code=1,error=repr(er),traceback=traceback.format_exc());raise
        finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
