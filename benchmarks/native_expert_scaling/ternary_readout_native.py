"""Same selected-p6 artifact through native blocked LUT,exact engine dense control."""
import argparse,ctypes,json,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from ternary_readout_precision import decode,metrics
START=time.monotonic()

def api(b):
    dll=ctypes.CDLL(b['compile'][0]['dll']['path']);f=dll.tr_history;f.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*7;f.restype=ctypes.c_int
    abi=dll.tr_abi;abi.argtypes=[ctypes.c_int];abi.restype=ctypes.c_int;assert [abi(i) for i in range(6)]==[3072,2048,615,6,8,243]
    table=dll.tr_table;table.argtypes=[ctypes.c_void_p]*2;table.restype=None
    ref=ctypes.CDLL(b['compile'][1]['dll']['path']);rf=ref.tr_scalar;rf.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*4;rf.restype=ctypes.c_int
    return dll,f,table,ref,rf

def summarize(rows,np):
    aa={}
    for site in [0,12,23]:
        aa[str(site)]={}
        for split in ['FIT','DEV']:
            rr=[q for q in rows if q['site']==site and q['split']==split];assert len(rr)==24;aa[str(site)][split]={}
            for kind in ['raw_vs_native','BF16_vs_baseline','BF16_vs_source']:
                aa[str(site)][split][kind]={}
                for field in ['relative_RMS','centered_relative_RMS','label_relative_RMS']:
                    v=[q['metrics'][kind][field] for q in rr];dm={d:float(np.mean([q['metrics'][kind][field] for q in rr if q['domain']==d])) for d in sorted({q['domain'] for q in rr})}
                    aa[str(site)][split][kind][field]=dict(mean=float(np.mean(v)),worst=max(v),domain_means=dm,worst_domain_mean=max(dm.values()))
    return aa

def prepare(a):
    import numpy as np,psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    rp=DOC/'ternary_readout_precision_result_20261013.json';ap=DOC/'ternary_readout_precision_adjudication_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and r['FIT_selected_planes']==6 and r['selected_DEV_all_pass'];assert extent(r['binding']['path'])==r['binding'];pb=json.loads(Path(r['binding']['path']).read_bytes())
    compiler=Path(json.loads((DOC/'full_channel_native_binding_20261013.json').read_bytes())['compile'][0]['command'][0]);storage=DOC/'ternary_readout_native_storage_20261013.json';st=json.loads(storage.read_text(encoding='utf-8-sig'));assert not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(st['new_family_directory']).resolve()
    source=B/'ternary_readout_native.c';ref=B/'ternary_readout_scalar.c';engine=ROOT/'benchmarks/phase60/engine.c';bodies={}
    for name in ['hsum256','dotf']:
        pat=r'static inline float '+name+r'\([^\n]*?\)\{.*?return [^;]*; \}';body=re.search(pat,engine.read_text(),re.S).group(0);assert body==re.search(pat,source.read_text(),re.S).group(0);bodies[name]=body
    packets=[dict(site=m['site'],file=next(e['file'] for e in m['arms'] if e['planes']==6)) for m in r['models']]
    paths=[Path(__file__),B/'ternary_readout_native_audit.py',B/'ternary_readout_precision.py',B/'falcon_auxiliary_s6_transport.py',source,ref,engine,storage,DOC/'TERNARY_READOUT_NATIVE_PROTOCOL_20261013.md',rp,ap,Path(r['binding']['path']),compiler,Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)]
    nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))+list(compiler.parent.glob('*.dll'))+list(compiler.parent.glob('*lld*.exe'));paths+=[Path(q['file']['path']) for q in packets]+[Path(q['parameters']['path']) for q in pb['exports']]+[Path(q['file']['path']) for q in r['records']]+[Path(q[k]['path']) for q in pb['records'] for k in ['native','source_output']];paths+=[ROOT/q for q in ['benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=[extent(p) for p in dict.fromkeys(paths)]
    a.directory.mkdir();comp=[]
    for tag,src,extra in [('native',source,[]),('scalar',ref,['-fno-vectorize','-fno-slp-vectorize'])]:
        dll=a.directory/(tag+'.dll');log=a.directory/(tag+'.compile.log');cmd=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off',*extra,'-shared',str(src),'-o',str(dll.resolve()),'-lm'];out=subprocess.run(cmd,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(out.stdout+out.stderr);assert out.returncode==0;comp.append(dict(command=cmd,exit_code=0,source=extent(src),dll=extent(dll),log=extent(log)))
    inputs+=[e for q in comp for e in [q['dll'],q['log']]]
    for e in inputs:assert extent(e['path'])==e
    assert time.monotonic()-START<900 and proc.memory_info().peak_wset<6<<30 and not proc.children(recursive=True)
    b=dict(schema='TERNARY_READOUT_NATIVE_BINDING_V1',freeze=a.freeze,python=pb['python'],versions=pb['versions'],parent_report=extent(rp),parent_adjudication=extent(ap),parent_binding=r['binding'],packets=packets,exports=pb['exports'],records=pb['records'],precision_records=r['records'],compile=comp,engine_bodies=bodies,inputs=inputs,output_directory=str(a.directory/'run'),numeric=dict(raw_relative=1e-10,BF16_smooth_relative=1e-4,scalar_relative=1e-10,metadata_absolute_or_relative=1e-10,original_F32_bit_exact=True),criteria=pb['criteria'],limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=2<<30),scope='Same selected p6/actual own-normalized BF16 inputs,one-core C blocked F64 LUT;every original AVX F32 score bit control/full outputs and source gates. No recurrence/source/GPU/T4/activation quantization/OWN chatbot/DRAM/cost or useful-n/model-rate admission.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)

def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(b['output_directory']).resolve();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='startup';rows=[];controls=[];receipt=dict(schema='TERNARY_READOUT_NATIVE_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def call(codes,scale,O,X):
        n=len(X);Y=np.empty((n,2048),'f8');dense=np.empty((n,2048),'f4');bits=np.empty((n,2048),'u2');proc.cpu_affinity([0]);rc=fn(n,*[q.ctypes.data for q in [codes,scale,O,X,Y,dense,bits]]);proc.cpu_affinity(list(range(6)));assert rc==0 and np.isfinite(Y).all() and np.isfinite(dense).all();return Y,dense,bits
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    try:
        check();a.directory.mkdir();dll,fn,table,ref,rf=api(b)
        for packet in b['packets']:
            site=packet['site'];guard()
            with np.load(packet['file']['path'],allow_pickle=False) as z:codes=z['codes'].copy();scale=z['scale'].copy();q=z['q'].astype('f8');qi=z['q'].copy()
            with np.load(next(e['parameters']['path'] for e in b['exports'] if e['site']==site),allow_pickle=False) as z:O=z['O'].copy()
            fit=min((e for e in b['records'] if e['site']==site and e['split']=='FIT'),key=lambda e:(e['history'],e['id']))
            with np.load(fit['native']['path'],allow_pickle=False) as z:X=decode(z['reconstructed_denominator__normalized'][:16],np).astype('f4')
            native,_,_=call(codes,scale,O,X);scalar=np.empty_like(native);proc.cpu_affinity([0]);assert rf(16,*[z.ctypes.data for z in [qi,scale,X,scalar]])==0;proc.cpu_affinity(list(range(6)));serr=rel(native,scalar)
            synthetic=np.zeros((1,3072),'f4');synthetic[0,:5]=[1,3,9,27,81];tt=np.empty(243,'f8');table(synthetic.ctypes.data,tt.ctypes.data);assert np.array_equal(tt,np.arange(243)-121);orig,_,_=call(codes,scale,O,synthetic);mut=codes.copy();old=int(mut[0,0,0]);mut[0,0,0]=(old+1)%243;changed,_,_=call(mut,scale,O,synthetic);delta=float(((old+1)%243-old)*scale[0]);assert changed[0,0]!=orig[0,0] and np.array_equal(changed[0,1:],orig[0,1:]);derr=float(abs((changed[0,0]-orig[0,0])-delta)/max(abs(delta),1e-30));controls.append(dict(site=site,id=fit['id'],scalar_positions=16,scalar_relative=serr,table_patterns_exact=243,mutation_old_code=old,mutation_new_code=int(mut[0,0,0]),mutation_delta_relative=derr));assert serr<=b['numeric']['scalar_relative'] and derr<=b['numeric']['scalar_relative']
            for e in [r for r in b['records'] if r['site']==site]:
                stage=f'{site}/{e["id"]}';guard()
                with np.load(e['native']['path'],allow_pickle=False) as z:X=decode(z['reconstructed_denominator__normalized'],np).astype('f4');Y=z['reconstructed_denominator__raw_scores'].copy();basebits=z['reconstructed_denominator__output'].copy()
                precision=next(r['file'] for r in b['precision_records'] if r['site']==site and r['id']==e['id'])
                with np.load(precision['path'],allow_pickle=False) as z:pb=z['p6'].copy()
                raw,dense,bits=call(codes,scale,O,X);ideal=(X.astype('f8')@q.T)*scale;f=e['source_output'];source=decode(np.fromfile(f['path'],'<u2').reshape(f['shape']),np);out=decode(bits,np);mm=dict(raw_vs_native=metrics(raw,Y.astype('f8'),e['positions'],np),BF16_vs_baseline=metrics(out,decode(basebits,np),e['positions'],np),BF16_vs_source=metrics(out,source,e['positions'],np));ne=dict(raw_relative=rel(raw,ideal),BF16_smooth_relative=rel(out,decode(pb,np)),precision_BF16_mismatches=int(np.count_nonzero(bits!=pb)),original_F32_mismatches=int(np.count_nonzero(dense.view('u4')!=Y.view('u4'))));ng=dict(raw=ne['raw_relative']<=b['numeric']['raw_relative'],BF16=ne['BF16_smooth_relative']<=b['numeric']['BF16_smooth_relative'],original_F32=ne['original_F32_mismatches']==0);extra=dict(raw=mm['raw_vs_native']['relative_RMS']<=b['criteria']['additional_raw_each_case'],BF16=mm['BF16_vs_baseline']['relative_RMS']<=b['criteria']['additional_BF16_each_case']);p=a.directory/f'{e["id"]}.site{site:02d}.npz';np.savez(p,lut_scores=raw,original_scores=dense,lut_BF16=bits);rows.append(dict(id=e['id'],site=site,split=e['split'],domain=e['domain'],history=e['history'],file=extent(p),metrics=mm,numeric_metrics=ne,numeric_gates=ng,additional_gates=extra));print(json.dumps(dict(stage=stage,complete=len(rows),numeric_gates=ng,seconds=time.monotonic()-START)),flush=True)
        check();guard();aa=summarize(rows,np);donor={}
        for site in [0,12,23]:
            donor[str(site)]={}
            for split in ['FIT','DEV']:
                d=aa[str(site)][split]['BF16_vs_source'];donor[str(site)][split]=dict(mean=d['relative_RMS']['mean']<=b['criteria']['donor_mean'],worst=d['relative_RMS']['worst']<=b['criteria']['donor_worst'],centered=d['centered_relative_RMS']['mean']<=b['criteria']['donor_centered'],domain=d['relative_RMS']['worst_domain_mean']<=b['criteria']['donor_domain'])
        result=dict(schema='TERNARY_READOUT_NATIVE_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,controls=controls,aggregates=aa,donor_gates=donor,numeric_all_pass=all(all(r['numeric_gates'].values()) for r in rows),additional_all_pass=all(all(r['additional_gates'].values()) for r in rows),before_after_hashes_exact=True,native_history_calls=144,source_forwards=0,recurrence_calls=0,activation_quantizations=0,optimizer_updates=0,decision='COMPLETE_PENDING_FULL_INDEPENDENT_AUDIT',output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),numeric_all_pass=result['numeric_all_pass'],additional_all_pass=result['additional_all_pass'])),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for k in ['binding','directory','out']:p.add_argument('--'+k,type=Path)
    for k in ['binding-sha','freeze']:p.add_argument('--'+k)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='TERNARY_READOUT_NATIVE_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as er:
            import traceback
            receipt.update(exit_code=1,error=repr(er),traceback=traceback.format_exc());raise
        finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
