"""Complete source-only F0 vocabulary compiler, literal CPU norm, full source calibration."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from gate_input_recovery import bf,decode
START=time.monotonic();FIELDS=['gate','x','B','C','delta']

def token_norm(E,gamma,cfg,np,torch):
    # Source BF16 multiply, F32 RMS, cast BEFORE gamma, BF16 SSMin.
    with torch.no_grad():
        h=torch.from_numpy(E.astype('f4')).to(torch.bfloat16)[None]*cfg['embedding_multiplier']
        v=h.float();v=v*torch.rsqrt(v.pow(2).mean(-1,keepdim=True)+cfg['rms_norm_eps'])
        h=torch.from_numpy(gamma.astype('f4')).to(torch.bfloat16)*v.to(torch.bfloat16)
        return (h*cfg['ssm_in_multiplier'])[0].view(torch.uint16).numpy().copy()

def fields(scaled,cw,cb,db,minimum,maximum,np):
    n=len(scaled);bc=scaled[:,3072:6656];conv=np.broadcast_to(cb,(n,3584)).copy()
    for tap in range(4):
        lag=3-tap
        if lag==0:conv+=bc*cw[:,tap]
        else:conv[lag:]+=bc[:-lag]*cw[:,tap]
    conv=decode(bf(conv,np),np);act=decode(bf(conv*np.exp(-np.logaddexp(0.,-conv)),np),np)
    d=decode(bf(scaled[:,6656:]+db,np),np);sp=decode(bf(np.where(d>20,d,np.logaddexp(0.,d)),np),np);sp=np.clip(sp,minimum,np.inf if maximum is None else maximum)
    return dict(gate=bf(scaled[:,:3072],np),x=bf(act[:,:3072],np),B=bf(act[:,3072:3328],np),C=bf(act[:,3328:],np),delta=bf(sp,np))

def prepare(a):
    import numpy as np,psutil,torch
    assert not torch.cuda.is_initialized();proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    rp=DOC/'true_input_function_replay_result_20261013.json';ap=DOC/'true_input_function_replay_adjudication_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and r['calibration_consistency_all_fields'];pb=json.loads(Path(r['binding']['path']).read_bytes());capture=json.loads(Path(pb['capture']['path']).read_bytes());wp=DOC/'source_token_input_witness_20261013.json';w=json.loads(wp.read_bytes());assert w['decision']=='EXACT_REPEATED_TOKEN_H0_INVARIANCE' and w['capture']==pb['capture'] and w['positions']==22547;cfg=json.loads(Path(pb['source_config']).read_bytes());assert cfg['vocab_size']==65537 and cfg['hidden_size']==2048 and cfg['rms_norm_eps']==1e-5
    stp=DOC/'token_generator_compiler_storage_20261013.json';st=json.loads(stp.read_text(encoding='utf-8-sig'));assert not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(st['new_family_directory']).resolve()
    paths=[Path(__file__),B/'token_generator_compiler_audit.py',B/'gate_input_recovery.py',B/'falcon_auxiliary_s6_transport.py',DOC/'TOKEN_GENERATOR_COMPILER_PROTOCOL_20261013.md',stp,rp,ap,wp,Path(r['binding']['path']),Path(pb['capture']['path']),Path(pb['capture_adjudication']['path']),Path(pb['source_weights']),Path(pb['source_config']),SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py',Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__),Path(torch.__file__),Path(torch._C.__file__)]
    nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;tr=Path(torch.__file__).resolve().parent
    paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))+list((tr/'lib').glob('*.dll'))
    for rec in capture['records']:paths+=[Path(rec['sites'][0]['fields'][k]['path']) for k in ['H',*FIELDS]]
    paths+=[Path(e['file']['path']) for e in r['records'] if e['site']==0]+[ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=[extent(p) for p in dict.fromkeys(paths)]
    for e in inputs:assert extent(e['path'])==e
    assert time.monotonic()-START<900 and proc.memory_info().peak_wset<6<<30 and not proc.children(recursive=True);a.directory.mkdir()
    b=dict(schema='TOKEN_GENERATOR_COMPILER_BINDING_V1',freeze=a.freeze,python=str(Path(sys.executable).resolve()),versions=dict(numpy=np.__version__,psutil=psutil.__version__,torch=torch.__version__),parent_report=extent(rp),parent_adjudication=extent(ap),parent_binding=r['binding'],witness=extent(wp),capture=pb['capture'],capture_adjudication=pb['capture_adjudication'],source_weights=pb['source_weights'],source_config=pb['source_config'],semantics=pb['semantics'],records=capture['records'],reference_records=[e for e in r['records'] if e['site']==0],inputs=inputs,vocabulary=65537,width=6704,chunk_rows=256,output_directory=str(a.directory/'run'),numeric=dict(norm_BF16_exact=True,table_BF16_exact=True,field_vector_relative=1e-6,metadata_absolute_or_relative=1e-8),criteria=dict(H0_each_case_relative=1e-4,each_case_each_source_field_relative=.01,incremental_each_field_relative=.01),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=2<<30),scope='Complete canonical source-only vocabulary F0 compiler: literal CPU Torch BF16/F32 norm, promoted F64 generator/BF16 boundaries,all48 source field calibration and actual H0 norms. New backend approximation explicit,no CUDA identity assumption. No fitted/captured-response rows,no source model forward/GPU/T4/RESERVED/native state/model rate/useful-n admission.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)

def run(a):
    import numpy as np,psutil,torch
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and dict(numpy=np.__version__,psutil=psutil.__version__,torch=torch.__version__)==b['versions'] and not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(b['output_directory']).resolve()
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);assert not torch.cuda.is_initialized();rows=[];stage='startup';stream=None;receipt=dict(schema='TOKEN_GENERATOR_COMPILER_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True) and not torch.cuda.is_initialized()
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def mm(x,y,pos):return dict(relative_RMS=rel(x,y),centered_relative_RMS=rel(x-x.mean(0),y-y.mean(0)),label_relative_RMS=rel(x[pos],y[pos]))
    try:
        check();a.directory.mkdir();get,stream=tensor_reader(b['source_weights'],np);cfg=json.loads(Path(b['source_config']).read_bytes());gamma=get('model.layers.0.input_layernorm.weight');E=get('model.embed_tokens.weight');W=get('model.layers.0.mamba.in_proj.weight');cw=get('model.layers.0.mamba.conv1d.weight')[:,0];cb=get('model.layers.0.mamba.conv1d.bias');db=get('model.layers.0.mamba.dt_bias');assert E.shape==(65537,2048) and W.shape==(6704,2048);mu=decode(np.array(b['semantics']['MuP_bits'],'u2'),np)
        # Scalar promotion control,observed CPU behavior,no alternative selected by responses.
        u=np.arange(65536,dtype='u2');v=(u.astype('u4')<<16).view('f4');finite=np.isfinite(v)&(np.abs(v)<=4);u=u[finite];v=v[finite];t=torch.from_numpy(v.copy()).to(torch.bfloat16)
        with torch.no_grad():literal=(t*cfg['embedding_multiplier']).view(torch.uint16).numpy().copy()
        float_scalar=bf(v*cfg['embedding_multiplier'],np);BF_scalar=bf(v*decode(bf(np.array(cfg['embedding_multiplier']),np),np),np);control=a.directory/'scalar_control.npz';np.savez(control,input_bits=u,literal_bits=literal,F32_scalar_bits=float_scalar,BF16_scalar_bits=BF_scalar);scalar=dict(file=extent(control),values=len(u),F32_scalar_differences=int(np.count_nonzero(literal!=float_scalar)),BF16_scalar_differences=int(np.count_nonzero(literal!=BF_scalar)))
        hp=a.directory/'H0.BF16';fp=a.directory/'F0.BF16';H=np.memmap(hp,dtype='u2',mode='w+',shape=(65537,2048));F=np.memmap(fp,dtype='u2',mode='w+',shape=(65537,6704));compiled=0
        for lo in range(0,65537,b['chunk_rows']):
            stage=f'vocabulary/{lo}';guard();hi=min(lo+b['chunk_rows'],65537);hb=token_norm(E[lo:hi],gamma,cfg,np,torch);H[lo:hi]=hb;projected=decode(bf(decode(hb,np)@W.T,np),np);F[lo:hi]=bf(projected*mu,np);compiled=hi
            if hi%8192==0 or hi==65537:print(json.dumps(dict(compiled_rows=compiled,seconds=time.monotonic()-START)),flush=True)
        H.flush();F.flush();h_extent=extent(hp);f_extent=extent(fp);del E,W;params=a.directory/'source_conv_parameters.npz';np.savez(params,cw=cw,cb=cb,db=db);assert compiled==65537
        for rec in b['records']:
            stage=rec['id'];guard();ids=np.array(rec['input_ids'],'u4');assert len(ids)==rec['history'] and np.max(ids)<=65536;hh=H[ids].copy();scaled=decode(F[ids].copy(),np);pred=fields(scaled,cw,cb,db,b['semantics']['delta_min'],b['semantics']['delta_max'],np);sf=rec['sites'][0]['fields'];actualH=np.fromfile(sf['H']['path'],'u2').reshape(sf['H']['shape']);hm=mm(decode(hh,np),decode(actualH,np),rec['positions']);hm_bits=int(np.count_nonzero(hh!=actualH));prior=next(q for q in b['reference_records'] if q['id']==rec['id']);metrics={};flags={};differences={};increment={}
            with np.load(prior['file']['path'],allow_pickle=False) as z:
                for k in FIELDS:
                    source=np.fromfile(sf[k]['path'],'u2').reshape(sf[k]['shape']);metrics[k]=mm(decode(pred[k],np),decode(source,np),rec['positions']);flags[k]=metrics[k]['relative_RMS']<=b['criteria']['each_case_each_source_field_relative'];differences[k]=int(np.count_nonzero(pred[k]!=source));increment[k]=rel(decode(pred[k],np),decode(z['predicted_'+k],np))
            path=a.directory/(rec['id']+'.fields.npz');np.savez(path,**pred);rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=rec['history'],file=extent(path),reference=prior['file'],H0_metrics=hm,H0_bit_mismatches=hm_bits,H0_gate=hm['relative_RMS']<=b['criteria']['H0_each_case_relative'],metrics=metrics,source_field_gates=flags,source_field_bit_mismatches=differences,incremental_vs_actual_H_fields=increment,incremental_field_gates={k:v<=b['criteria']['incremental_each_field_relative'] for k,v in increment.items()}));print(json.dumps(dict(complete=len(rows),H0_error=hm['relative_RMS'],source_gates=flags,seconds=time.monotonic()-START)),flush=True)
        stream.close();stream=None;check();guard();assert len(rows)==48;assert extent(hp)==h_extent and extent(fp)==f_extent;aggregates={}
        for split in ['FIT','DEV']:
            rr=[q for q in rows if q['split']==split];assert len(rr)==24;aggregates[split]={}
            for key in ['H0',*FIELDS]:
                vals=[q['H0_metrics']['relative_RMS'] if key=='H0' else q['metrics'][key]['relative_RMS'] for q in rr];aggregates[split][key]=dict(mean=float(np.mean(vals)),worst=max(vals),failed_cases=sum(not (q['H0_gate'] if key=='H0' else q['source_field_gates'][key]) for q in rr))
        result=dict(schema='TOKEN_GENERATOR_COMPILER_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,tables=dict(H0=h_extent,F0=f_extent),vocabulary_rows=65537,table_F0_words=65537*6704,table_H0_words=65537*2048,scalar_control=scalar,parameters=extent(params),records=rows,aggregates=aggregates,H0_all_pass=all(q['H0_gate'] for q in rows),source_fields_all_pass=all(all(q['source_field_gates'].values()) for q in rows),incremental_fields_all_pass=all(all(q['incremental_field_gates'].values()) for q in rows),before_after_hashes_exact=True,GPU_initialized=False,source_model_forwards=0,source_function_compiler_passes=1,native_state_calls=0,optimizer_updates=0,output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),H0_all_pass=result['H0_all_pass'],source_fields_all_pass=result['source_fields_all_pass'],incremental_fields_all_pass=result['incremental_fields_all_pass'])),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for k in ['binding','directory','out']:p.add_argument('--'+k,type=Path)
    for k in ['binding-sha','freeze']:p.add_argument('--'+k)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='TOKEN_GENERATOR_COMPILER_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as er:
            import traceback
            receipt.update(exit_code=1,error=repr(er),traceback=traceback.format_exc());raise
        finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
