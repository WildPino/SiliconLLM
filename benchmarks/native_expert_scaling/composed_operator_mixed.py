"""Build/bind fixed native composed operator,then qualify all48 supplied histories."""
import argparse,ctypes,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter()
PATHS=['mixed_decode64_out32'];MODES=['full_source_denominator','reconstructed_denominator']


def interface(b,np):
    dll=ctypes.CDLL(b['native_dll']['path']);abi=dll.composed_abi;abi.argtypes=[ctypes.c_int];abi.restype=ctypes.c_int;assert [abi(i) for i in range(8)]==[3072,384,96,48,2048,5,9,8]
    fn=dll.composed_history;fn.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int];fn.restype=ctypes.c_int
    return dll,fn


def ptrs(arrays):return (ctypes.c_void_p*len(arrays))(*[q.ctypes.data for q in arrays])


def prepare(a):
    import numpy as np,psutil
    oldpath=DOC/'recurrent_axis_composition_result_20261013.json';adpath=DOC/'recurrent_axis_composition_adjudication_20261013.json';old=json.loads(oldpath.read_bytes());ad=json.loads(adpath.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(oldpath);pb=json.loads(Path(old['binding']['path']).read_bytes());assert not a.directory.exists() and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));compiler=Path(json.loads((DOC/'native_scalar_readout_preparation_20261012.json').read_bytes())['compiler']['path']);engine=ROOT/'benchmarks/phase60/engine.c';source=B/'composed_operator_mixed.c';reference=B/'composed_operator_scalar_reference.c';engine_text=engine.read_text();src_text=source.read_text();bodies={}
    for name in ['hsum256','dotf']:
        pattern=r'static inline float '+name+r'\([^\n]*?\)\{.*?return [^;]*; \}'
        x=re.search(pattern,engine_text,re.S).group(0);y=re.search(pattern,src_text,re.S).group(0);assert x==y,name;bodies[name]=x
    paths=[Path(__file__),source,reference,B/'falcon_auxiliary_s6_transport.py',DOC/'COMPOSED_OPERATOR_MIXED_PROTOCOL_20261013.md',DOC/'composed_operator_mixed_storage_20261013.json',oldpath,adpath,Path(old['binding']['path']),engine,compiler]
    v1path=DOC/'composed_operator_native_result_20261013.json';v1adpath=DOC/'composed_operator_native_adjudication_20261013.json';v1ad=json.loads(v1adpath.read_bytes());assert v1ad['decision']=='INDEPENDENT_STORED_NUMERIC_FAILURE' and v1ad['report']==extent(v1path);assert sum(not q['passed'] for q in v1ad['numeric_flags'])==1 and v1ad['maximum']['controlled_F64_normalized_smooth']==0 and v1ad['maximum']['F32_score_bit_mismatches']==0;paths += [v1path,v1adpath]
    paths += list(compiler.parent.glob('*.dll'))+list(compiler.parent.glob('clang*.exe'))+list(compiler.parent.glob('*lld*.exe'));inputs=list({e['path']:e for e in pb['inputs']+[extent(p) for p in paths]+[q['file'] for q in old['records']]}.values())
    for e in inputs:assert extent(e['path'])==e
    a.directory.mkdir(parents=True);compile_records=[]
    for tag,src,extra in [('native',source,[]),('reference',reference,['-fno-vectorize','-fno-slp-vectorize'])]:
        dll=a.directory/(tag+'.dll');log=a.directory/(tag+'.compile.log');cmd=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off',*extra,'-shared',str(src),'-o',str(dll.resolve()),'-lm'];p=subprocess.run(cmd,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(p.stdout+p.stderr);compile_records.append(dict(command=cmd,exit_code=p.returncode,source=extent(src),dll=extent(dll) if dll.exists() else None,log=extent(log)));assert p.returncode==0
    with np.load(pb['channel_basis']['path'],allow_pickle=False) as z:
        W=np.concatenate([z[f'adaptive384_W_{h}'].T.astype('f4') for h in range(48)]);V=np.concatenate([z[f'adaptive384_V_{h}'].astype('f4').ravel() for h in range(48)])
    with np.load(pb['state_basis']['path'],allow_pickle=False) as z:Ws=z['W'].T.astype('f4').copy();Vs=z['V'].T.astype('f4').copy()
    get,stream=tensor_reader(pb['parent']['source_weights'],np);prefix='model.layers.23.mamba.';params=dict(Wc=W,Vc=V,Ws=Ws,Vs=Vs,Alog=get(prefix+'A_log').astype('f4'),D=get(prefix+'D').astype('f4'),gamma=get(prefix+'norm.weight').astype('f4'),O=get(prefix+'out_proj.weight').astype('f4'),in_proj=get(prefix+'in_proj.weight').astype('f4'));stream.close();parampath=a.directory/'parameters.npz';np.savez(parampath,**params)
    inputs += [extent(parampath)]+[e for q in compile_records for e in [q['dll'],q['log']]]
    for e in inputs:assert extent(e['path'])==e
    assert time.monotonic()-START<=900 and proc.memory_info().peak_wset<=6<<30 and not proc.children(recursive=True)
    b=dict(schema='COMPOSED_OPERATOR_MIXED_BINDING_V1',python=pb['python'],versions=pb['versions'],freeze=a.freeze,compile=compile_records,native_dll=compile_records[0]['dll'],reference_dll=compile_records[1]['dll'],parameters=extent(parampath),original_primitive_bodies=bodies,parent=pb,parent_report=extent(oldpath),parent_adjudication=extent(adpath),capture=pb['capture'],ranks=pb['ranks'],offsets=pb['offsets'],paths=PATHS,modes=MODES,output_directory=str(a.directory.parent/'run'),native_threads=1,numeric=dict(projected_relative=1e-6,scan_relative=1e-4,state_relative=1e-4,controlled_relative=1e-4,F32_smooth_relative=1e-4,old_bridge_relative=1e-3,scalar_relative=1e-8,F32_raw_score_bit_exact=True),criteria=dict(mean=.1,worst=.2,centered=.1,domain=.2),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=2<<30),inputs=inputs,preparation_seconds=time.perf_counter()-CLOCK,preparation_OS_peak=proc.memory_info().peak_wset,scope='Actual separate native384x96 CPU supplied-operand operator,F64 decode/gate/RMS before BF16 then AVX F32 output/all48 histories. Retained source generator/gate/conv/fullskip/norm/out costs;no native chatbot/own-history/n/DRAM/full-model50 admission.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=b['preparation_seconds'])),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and not a.directory.exists() and not a.out.exists() and str(a.directory.resolve())==str(Path(b['output_directory']).resolve());assert np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'];proc=psutil.Process();proc.cpu_affinity([0]);rows=[];stage='startup';receipt=dict(schema='COMPOSED_OPERATOR_MIXED_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        assert sum(p.stat().st_size for p in a.directory.parent.rglob('*') if p.is_file())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');return np.ascontiguousarray(z.reshape(e['shape']),dtype='f4')
    def decode(x):return (x.astype('<u4')<<16).view('<f4').astype('f8')
    def rel(x,y):return float(np.linalg.norm(x.astype('f8')-y.astype('f8'))/max(np.linalg.norm(y.astype('f8')),1e-30))
    def metrics(op,ref,pos):return dict(output_relative_RMS=rel(op,ref),centered_output_relative_RMS=rel(op-op.mean(0),ref-ref.mean(0)),label_output_relative_RMS=rel(op[pos],ref[pos]))
    try:
        check();a.directory.mkdir();dll,fn=interface(b,np)
        with np.load(b['parameters']['path'],allow_pickle=False) as z:params={k:z[k].copy() for k in z.files}
        weights=[params[k] for k in ['Wc','Vc','Ws','Vs','Alog','D','gamma','O']];wp=ptrs(weights);ranks=np.array(b['ranks'],'i4');offs=np.array(b['offsets'],'i4');capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];prior=json.loads(Path(b['parent_report']['path']).read_bytes());priorrows={q['id']:q for q in prior['records']}
        for rec in capture:
            stage=rec['id'];guard();n=rec['history'];f=rec['sites'][23]['fields'];x,bb,cc,dt,gate=[raw(f[k]).reshape(n,-1) for k in ['x','B','C','delta','gate']];yy=raw(f['y']).astype('f8');gg=gate.astype('f8')*np.exp(-np.logaddexp(0.,-gate.astype('f8')));inv=np.ascontiguousarray(1/np.sqrt(np.mean((yy*gg)**2,axis=1)+1e-5));floats=[np.empty((n,c),'f4') for c in [384,96,96,384]]+[np.empty((384,96),'f4')]+[np.empty((n,2048),'f4') for _ in range(4)];bits=[np.empty((n,3072),'u2') for _ in range(4)]+[np.empty((n,2048),'u2') for _ in range(4)];times=np.zeros(8,'f8');start=time.perf_counter();rc=fn(n,1,ranks.ctypes.data,offs.ctypes.data,wp,ptrs([x,bb,cc,dt,gate]),inv.ctypes.data,ptrs(floats),ptrs(bits),times.ctypes.data,0);call_seconds=time.perf_counter()-start;assert rc==0 and all(np.isfinite(q).all() for q in floats[:7]);arrays={k:q for k,q in zip(['projected_input','state96_B','state96_C','latent','final_state'],floats[:5])};mm={};bridge={};pr=priorrows[rec['id']]
            with np.load(pr['file']['path'],allow_pickle=False) as z:old={k:z[k].copy() for k in ['projected_input','state96_B','state96_C','joint96_latent','channel384_state96__full_source_denominator','channel384_state96__reconstructed_denominator']}
            comparison={key:rel(arrays[key],old[oldkey]) for key,oldkey in [('projected_input','projected_input'),('state96_B','state96_B'),('state96_C','state96_C'),('latent','joint96_latent')]};ref=raw(f['output']).astype('f8')
            for pidx,path in enumerate(PATHS):
                mm[path]={};bridge[path]={}
                for midx,mode in enumerate(MODES):
                    i=pidx*2+midx;arrays[path+'__'+mode+'__normalized']=bits[i];arrays[path+'__'+mode+'__raw_scores']=floats[5+i];arrays[path+'__'+mode+'__output']=bits[4+i];op=decode(bits[4+i]);mm[path][mode]=metrics(op,ref,rec['positions']);err=rel(op,decode(old['channel384_state96__'+mode]));bridge[path][mode]=dict(relative=err,pass_original_1e3=err<=b['numeric']['old_bridge_relative'])
            path=a.directory/(rec['id']+'.npz');np.savez(path,**arrays);rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,metrics=mm,prior_comparison=comparison,old_bridge=bridge,prior=pr['file'],file=extent(path),native_call_seconds=call_seconds));print(json.dumps(dict(stage=stage,complete=len(rows),native_call_seconds=call_seconds,seconds=time.perf_counter()-CLOCK)),flush=True)
        check();guard();assert len(rows)==48;aggregates={};gates={}
        for path in PATHS:
            aggregates[path]={};gates[path]={}
            for mode in MODES:
                aggregates[path][mode]={}
                for split in ['FIT','DEV']:
                    sr=[q for q in rows if q['split']==split];vals=[q['metrics'][path][mode]['output_relative_RMS'] for q in sr];c=float(np.mean([q['metrics'][path][mode]['centered_output_relative_RMS'] for q in sr]));dm={d:float(np.mean([q['metrics'][path][mode]['output_relative_RMS'] for q in sr if q['domain']==d])) for d in {q['domain'] for q in sr}};aa=dict(mean=float(np.mean(vals)),worst=max(vals),centered_mean=c,worst_domain_mean=max(dm.values()),domain_means=dm);aggregates[path][mode][split]=aa
                    if split=='DEV':gates[path][mode]=dict(mean=aa['mean']<=.1,worst=aa['worst']<=.2,centered=c<=.1,domain=aa['worst_domain_mean']<=.2)
        r=dict(schema='COMPOSED_OPERATOR_MIXED_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,aggregates=aggregates,gates=gates,before_after_hashes_exact=True,source_forwards=0,optimizer_updates=0,basis_fits=0,native_history_calls=48,new_native_scans=48,native_component_timing_admitted=False,whole_model_quality_admitted=False,whole_model_speed_admitted=False,decision='COMPLETE_PENDING_FULL_INDEPENDENT_AUDIT',output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),family_output_bytes=sum(p.stat().st_size for p in a.directory.parent.rglob('*') if p.is_file()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope=b['scope']);write(a.out,r);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),gates=gates)),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc())
        if a.directory.exists():write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),records=rows))
        raise
    finally:
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='COMPOSED_OPERATOR_MIXED_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as e:
            import traceback
            receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
        finally:
            receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
