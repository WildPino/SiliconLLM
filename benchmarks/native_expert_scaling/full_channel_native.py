"""Fixed source-state96 native qualification for all48 histories/sites0,12,23."""
import argparse,ctypes,json,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from composed_operator_mixed import ptrs
START=time.monotonic();CLOCK=time.perf_counter();SITES=[0,12,23];MODES=['full_source_denominator','reconstructed_denominator']


def interface(b):
    dll=ctypes.CDLL(b['native_dll']['path']);abi=dll.full_abi;abi.argtypes=[ctypes.c_int];abi.restype=ctypes.c_int;assert [abi(i) for i in range(6)]==[3072,96,48,2048,6,4];fn=dll.full_history;fn.argtypes=[ctypes.c_int,ctypes.c_int,*([ctypes.c_void_p]*6),ctypes.c_int];fn.restype=ctypes.c_int;return dll,fn


def raw(e,np):
    z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');return np.ascontiguousarray(z.reshape(e['shape']),dtype='f4')


def buffers(n,np):return [np.empty((n,96),'f4'),np.empty((n,96),'f4'),np.empty((n,3072),'f4'),np.empty((3072,96),'f4'),np.empty((n,2048),'f4'),np.empty((n,2048),'f4')],[np.empty((n,3072),'u2'),np.empty((n,3072),'u2'),np.empty((n,2048),'u2'),np.empty((n,2048),'u2')]


def prepare(a):
    import numpy as np,psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));statepath=DOC/'original_falcon_recurrent_projection_finish_result_20261009.json';statead=DOC/'original_falcon_recurrent_projection_audit_result_20261009.json';sr=json.loads(statepath.read_bytes());sa=json.loads(statead.read_bytes());assert sa['decision']=='STORED_IDENTITIES_AND_AGGREGATES_VERIFIED';basepath=DOC/'recurrent_output_bases_binding_20261012.json';pb=json.loads(basepath.read_bytes());capture=json.loads(Path(pb['capture']['path']).read_bytes())['records'];assert len(capture)==48;compiler=Path(json.loads((DOC/'composed_operator_mixed_binding_20261013.json').read_bytes())['compile'][0]['command'][0]);source=B/'full_channel_native.c';ref=B/'composed_operator_scalar_reference.c';engine=ROOT/'benchmarks/phase60/engine.c';bodies={}
    for name in ['hsum256','dotf']:
        pat=r'static inline float '+name+r'\([^\n]*?\)\{.*?return [^;]*; \}';x=re.search(pat,engine.read_text(),re.S).group(0);assert x==re.search(pat,source.read_text(),re.S).group(0);bodies[name]=x
    assert not a.directory.exists() and not a.out.exists();paths=[Path(__file__),source,ref,B/'composed_operator_mixed.py',B/'falcon_auxiliary_s6_transport.py',DOC/'FULL_CHANNEL_NATIVE_PROTOCOL_20261013.md',DOC/'full_channel_native_storage_20261013.json',statepath,statead,basepath,Path(pb['capture']['path']),Path(pb['parent']['source_weights']),Path(pb['parent']['source_config']),engine,compiler]
    for q in capture:
        for site in SITES:paths += [Path(e['path']) for e in q['sites'][site]['fields'].values()]
    packets=[q for q in sr['bases'] if q['site'] in SITES];assert {q['site'] for q in packets}==set(SITES);paths += [Path(q['basis']['path']) for q in packets];paths += list(compiler.parent.glob('*.dll'))+list(compiler.parent.glob('clang*.exe'))+list(compiler.parent.glob('*lld*.exe'));runtime=[e for e in pb['inputs'] if e['path'].startswith(str(Path(sys.executable).parent)) or e['path'].startswith(str(SITE.resolve())) or e['path'].startswith('C:\\Windows\\System32')];paths += [ROOT/p for p in ['benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=list({e['path']:e for e in runtime+[extent(p) for p in paths]}.values())
    for e in inputs:assert extent(e['path'])==e
    a.directory.mkdir(parents=True);comp=[]
    for tag,src,extra in [('native',source,[]),('reference',ref,['-fno-vectorize','-fno-slp-vectorize'])]:
        dll=a.directory/(tag+'.dll');log=a.directory/(tag+'.compile.log');cmd=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off',*extra,'-shared',str(src),'-o',str(dll.resolve()),'-lm'];p=subprocess.run(cmd,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(p.stdout+p.stderr);assert p.returncode==0;comp.append(dict(command=cmd,exit_code=0,source=extent(src),dll=extent(dll),log=extent(log)))
    get,stream=tensor_reader(pb['parent']['source_weights'],np);exports=[]
    for q in packets:
        site=q['site'];prefix=f'model.layers.{site}.mamba.'
        with np.load(q['basis']['path'],allow_pickle=False) as z:Ws=z['W'].T.astype('f4').copy();Vs=z['V'].T.astype('f4').copy()
        assert Ws.shape==Vs.shape==(96,256);params=dict(Ws=Ws,Vs=Vs,Alog=get(prefix+'A_log').astype('f4'),D=get(prefix+'D').astype('f4'),gamma=get(prefix+'norm.weight').astype('f4'),O=get(prefix+'out_proj.weight').astype('f4'),in_proj=get(prefix+'in_proj.weight').astype('f4'));path=a.directory/f'site{site:02d}.parameters.npz';np.savez(path,**params);exports.append(dict(site=site,basis=q['basis'],parameters=extent(path)))
    stream.close();inputs += [q['parameters'] for q in exports]+[e for q in comp for e in [q['dll'],q['log']]]
    for e in inputs:assert extent(e['path'])==e
    assert time.monotonic()-START<=900 and proc.memory_info().peak_wset<=6<<30 and not proc.children(recursive=True)
    b=dict(schema='FULL_CHANNEL_NATIVE_BINDING_V1',python=pb['python'],versions=pb['versions'],freeze=a.freeze,compile=comp,native_dll=comp[0]['dll'],reference_dll=comp[1]['dll'],exports=exports,original_primitive_bodies=bodies,parent=pb['parent'],capture=pb['capture'],state_report=extent(statepath),state_adjudication=extent(statead),sites=SITES,modes=MODES,output_directory=str(a.directory.parent/'run'),native_threads=1,numeric=dict(projected_relative=1e-6,scan_relative=1e-4,state_relative=1e-4,controlled_relative=1e-4,prior_metric_absolute=1e-3,scalar_relative=1e-8,F32_raw_score_bit_exact=True),criteria=dict(mean=.1,worst=.2,centered=.1,domain=.2),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=6<<30),inputs=inputs,preparation_seconds=time.perf_counter()-CLOCK,preparation_OS_peak=proc.memory_info().peak_wset,scope='All48 source-supplied histories/all3 sites,fixed FIT source-state96 with all3072 channels,actual C F32 state/projections,F64 gate/RMS/BF16/AVX F32 output. No refit/source/GPU/native chatbot/own-history/n/DRAM/full-model50 admission. Full source nonlinear/gen/conv/output costs retained.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=b['preparation_seconds'])),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and not a.directory.exists() and not a.out.exists() and str(a.directory.resolve())==str(Path(b['output_directory']).resolve());proc=psutil.Process();proc.cpu_affinity([0]);rows=[];stage='startup';receipt=dict(schema='FULL_CHANNEL_NATIVE_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True);assert sum(p.stat().st_size for p in a.directory.parent.rglob('*') if p.is_file())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def dec(q):return (q.astype('<u4')<<16).view('<f4').astype('f8')
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def metrics(op,ref,pos):return dict(output_relative_RMS=rel(op,ref),centered_output_relative_RMS=rel(op-op.mean(0),ref-ref.mean(0)),label_output_relative_RMS=rel(op[pos],ref[pos]))
    try:
        check();a.directory.mkdir();dll,fn=interface(b);capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];old=json.loads(Path(b['state_report']['path']).read_bytes());oldrows={(q['site'],q['id']):q for q in old['records']}
        for packet in b['exports']:
            site=packet['site']
            with np.load(packet['parameters']['path'],allow_pickle=False) as z:params={k:z[k].copy() for k in z.files}
            w=ptrs([params[k] for k in ['Ws','Vs','Alog','D','gamma','O','in_proj']])
            for rec in capture:
                stage=f'{site}/{rec["id"]}';guard();n=rec['history'];f=rec['sites'][site]['fields'];x,bb,cc,dt,gate=[raw(f[k],np).reshape(n,-1) for k in ['x','B','C','delta','gate']];y=raw(f['y'],np).astype('f8');g=gate.astype('f8')*np.exp(-np.logaddexp(0.,-gate.astype('f8')));inv=np.ascontiguousarray(1/np.sqrt(np.mean((y*g)**2,axis=1)+1e-5));floats,bits=buffers(n,np);times=np.zeros(4,'f8');start=time.perf_counter();rc=fn(n,0,w,ptrs([x,bb,cc,dt,gate]),inv.ctypes.data,ptrs(floats),ptrs(bits),times.ctypes.data,0);elapsed=time.perf_counter()-start;assert rc==0 and all(np.isfinite(q).all() for q in floats);arrays={k:q for k,q in zip(['state96_B','state96_C','latent','final_state'],floats[:4])};ref=raw(f['output'],np).astype('f8');mm={};oldmetric={};pr=oldrows[(site,rec['id'])]['arms']['dual96']
                for i,mode in enumerate(MODES):
                    arrays[mode+'__normalized']=bits[i];arrays[mode+'__raw_scores']=floats[4+i];arrays[mode+'__output']=bits[2+i];mm[mode]=metrics(dec(bits[2+i]),ref,rec['positions'])
                for key in mm['reconstructed_denominator']:
                    d=abs(mm['reconstructed_denominator'][key]-pr[key]);oldmetric[key]=dict(absolute=d,pass_original_1e3=d<=b['numeric']['prior_metric_absolute'])
                path=a.directory/(rec['id']+f'.site{site:02d}.npz');np.savez(path,**arrays);rows.append(dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=n,metrics=mm,prior_metric_delta=oldmetric,file=extent(path),native_call_seconds=elapsed));print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True)
        check();guard();assert len(rows)==144;aggregates={};gates={}
        for site in SITES:
            aggregates[str(site)]={};gates[str(site)]={}
            for mode in MODES:
                aggregates[str(site)][mode]={}
                for split in ['FIT','DEV']:
                    sr=[q for q in rows if q['site']==site and q['split']==split];assert len(sr)==24;vals=[q['metrics'][mode]['output_relative_RMS'] for q in sr];c=float(np.mean([q['metrics'][mode]['centered_output_relative_RMS'] for q in sr]));dm={d:float(np.mean([q['metrics'][mode]['output_relative_RMS'] for q in sr if q['domain']==d])) for d in {q['domain'] for q in sr}};aa=dict(mean=float(np.mean(vals)),worst=max(vals),centered_mean=c,worst_domain_mean=max(dm.values()),domain_means=dm);aggregates[str(site)][mode][split]=aa
                    if split=='DEV':gates[str(site)][mode]=dict(mean=aa['mean']<=.1,worst=aa['worst']<=.2,centered=c<=.1,domain=aa['worst_domain_mean']<=.2)
        r=dict(schema='FULL_CHANNEL_NATIVE_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,aggregates=aggregates,gates=gates,before_after_hashes_exact=True,source_forwards=0,optimizer_updates=0,basis_fits=0,native_history_calls=144,new_native_scans=144,state_coordinates=3072*96,state_F32_bytes=4*3072*96,native_component_timing_admitted=False,whole_model_quality_admitted=False,whole_model_speed_admitted=False,decision='COMPLETE_PENDING_FULL_INDEPENDENT_AUDIT',output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),family_output_bytes=sum(p.stat().st_size for p in a.directory.parent.rglob('*') if p.is_file()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope=b['scope']);write(a.out,r);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),gates=gates)),flush=True)
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
    for name in ['binding-sha','freeze']:p.add_argument('--'+name,required=False)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='FULL_CHANNEL_NATIVE_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as e:
            import traceback
            receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
        finally:
            receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
