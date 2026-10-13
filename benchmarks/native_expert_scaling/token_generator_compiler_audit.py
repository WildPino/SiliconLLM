"""Complete vocabulary and calibration audit; independent program, no producer import."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site';sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()

def main(a):
    import numpy as np,psutil,torch
    re,te=extent(a.report),extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==0 and t['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and dict(numpy=np.__version__,psutil=psutil.__version__,torch=torch.__version__)==b['versions'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);stream=None;stage='startup';counts=dict(H0_words=0,F0_words=0,H0_bit_mismatches=0,F0_bit_mismatches=0,scalar_control_words=0,source_H0_words=0,field_words=0,independent_field_BF16_mismatches=0);rows=[];maximum={};receipt=dict(schema='TOKEN_GENERATOR_COMPILER_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True) and not torch.cuda.is_initialized()
    def check():
        files=list(r['tables'].values())+[r['parameters'],r['scalar_control']['file']]+[q['file'] for q in r['records']]
        for e in b['inputs']+[r['binding'],re,te]+files:guard();assert extent(e['path'])==e,e['path']
    def dec(x):return (x.astype('u4')*65536).view('f4').astype('f8')
    def bf(x):
        u=x.astype('f4').view('u4');top=u>>16;low=u&65535;return (top+((low>32768)|((low==32768)&((top&1)!=0))).astype('u4')).astype('u2')
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def mm(x,y,pos):return dict(relative_RMS=rel(x,y),centered_relative_RMS=rel(x-x.mean(0),y-y.mean(0)),label_relative_RMS=rel(x[pos],y[pos]))
    def meta(x,y):
        if isinstance(y,dict):
            assert x.keys()==y.keys()
            for k in y:meta(x[k],y[k])
        elif isinstance(y,(bool,int,str)):assert x==y,(stage,x,y)
        else:
            error=abs(x-y)/max(abs(y),1.);maximum['metadata']=max(maximum.get('metadata',0.),error);assert error<=b['numeric']['metadata_absolute_or_relative']
    try:
        check();pa=json.loads(Path(b['parent_adjudication']['path']).read_bytes());assert pa['decision']=='INDEPENDENT_STORED_PASS' and pa['report']==b['parent_report'];assert r['vocabulary_rows']==65537 and r['table_H0_words']==65537*2048 and r['table_F0_words']==65537*6704 and all(r[k]==0 for k in ['source_model_forwards','native_state_calls','optimizer_updates']) and not r['GPU_initialized'];get,stream=tensor_reader(b['source_weights'],np);cfg=json.loads(Path(b['source_config']).read_bytes());E=get('model.embed_tokens.weight');G=get('model.layers.0.mamba.in_proj.weight');gamma=get('model.layers.0.input_layernorm.weight');mu=dec(np.array(b['semantics']['MuP_bits'],'u2'));assert np.array_equal(bf(np.repeat(np.array(cfg['ssm_multipliers']),[3072,3072,256,256,48])),np.array(b['semantics']['MuP_bits'],'u2'))
        H=np.memmap(r['tables']['H0']['path'],dtype='u2',mode='r',shape=(65537,2048));F=np.memmap(r['tables']['F0']['path'],dtype='u2',mode='r',shape=(65537,6704));assert r['tables']['H0']['bytes']==65537*2048*2 and r['tables']['F0']['bytes']==65537*6704*2
        # Separate program reproduces the declared CPU source primitive semantics;
        # it shares Torch kernels,not an independent implementation of those kernels.
        weight=torch.tensor(gamma.astype('f4'),dtype=torch.bfloat16)
        for lo in range(0,65537,64):
            stage=f'vocabulary/{lo}';guard();hi=min(lo+64,65537)
            with torch.no_grad():
                embedded=torch.tensor(E[lo:hi].astype('f4'),dtype=torch.bfloat16).unsqueeze(0);embedded=embedded*cfg['embedding_multiplier'];expanded=embedded.to(torch.float32);variance=torch.square(expanded).mean(dim=-1,keepdim=True);normalized=expanded*torch.rsqrt(variance+cfg['rms_norm_eps']);fresh=(normalized.to(torch.bfloat16)*weight)*cfg['ssm_in_multiplier'];expectedH=fresh[0].view(torch.uint16).numpy().copy()
            expectedF=bf(dec(bf(np.einsum('td,od->to',dec(expectedH),G,optimize=True)))*mu);counts['H0_words']+=expectedH.size;counts['F0_words']+=expectedF.size;counts['H0_bit_mismatches']+=int(np.count_nonzero(H[lo:hi]!=expectedH));counts['F0_bit_mismatches']+=int(np.count_nonzero(F[lo:hi]!=expectedF))
            if hi%8192==0 or hi==65537:print(json.dumps(dict(audited_rows=hi,seconds=time.monotonic()-START)),flush=True)
        del E,G;stage='scalar_control'
        with np.load(r['scalar_control']['file']['path'],allow_pickle=False) as z:ctl={k:z[k].copy() for k in z.files}
        all_u=np.arange(65536,dtype='u2');all_v=(all_u.astype('u4')<<16).view('f4');choose=np.isfinite(all_v)&(np.abs(all_v)<=4);assert np.array_equal(ctl['input_bits'],all_u[choose]);v=all_v[choose];counts['scalar_control_words']=len(v)
        with torch.no_grad():literal=(torch.tensor(v,dtype=torch.bfloat16)*cfg['embedding_multiplier']).view(torch.uint16).numpy().copy()
        assert np.array_equal(ctl['literal_bits'],literal) and np.array_equal(ctl['F32_scalar_bits'],bf(v*cfg['embedding_multiplier'])) and np.array_equal(ctl['BF16_scalar_bits'],bf(v*dec(bf(np.array(cfg['embedding_multiplier'])))));assert r['scalar_control']['values']==len(v) and r['scalar_control']['F32_scalar_differences']==int(np.count_nonzero(literal!=ctl['F32_scalar_bits'])) and r['scalar_control']['BF16_scalar_differences']==int(np.count_nonzero(literal!=ctl['BF16_scalar_bits']))
        cw=get('model.layers.0.mamba.conv1d.weight')[:,0];cb=get('model.layers.0.mamba.conv1d.bias');db=get('model.layers.0.mamba.dt_bias')
        with np.load(r['parameters']['path'],allow_pickle=False) as z:assert set(z.files)=={'cw','cb','db'} and all(np.array_equal(z[k],v) for k,v in [('cw',cw),('cb',cb),('db',db)])
        cases={q['id']:q for q in b['records']};assert len(r['records'])==48 and {q['id'] for q in r['records']}==set(cases)
        for item in r['records']:
            stage=item['id'];guard();rec=cases[item['id']];assert all(item[k]==rec[k] for k in ['id','history','split','domain']);sf=rec['sites'][0]['fields'];ids=np.array(rec['input_ids'],'u4');n=rec['history'];sourceH=np.fromfile(sf['H']['path'],'u2').reshape(n,2048);h=H[ids].copy();hm=mm(dec(h),dec(sourceH),rec['positions']);meta(item['H0_metrics'],hm);assert item['H0_bit_mismatches']==int(np.count_nonzero(h!=sourceH)) and item['H0_gate']==(hm['relative_RMS']<=b['criteria']['H0_each_case_relative']);counts['source_H0_words']+=h.size
            scaled=dec(F[ids].copy());bc=scaled[:,3072:6656];padded=np.pad(bc,((3,0),(0,0)));convolved=np.empty_like(bc)
            for lo in range(0,n,128):
                hi=min(lo+128,n);window=np.stack([padded[lo+tap:hi+tap] for tap in range(4)],axis=2);convolved[lo:hi]=np.einsum('tck,ck->tc',window,cw,optimize=True)+cb
            convolved=dec(bf(convolved));sig=np.empty_like(convolved);positive=convolved>=0;sig[positive]=1/(1+np.exp(-convolved[positive]));exp=np.exp(convolved[~positive]);sig[~positive]=exp/(1+exp);act=dec(bf(convolved*sig));d=dec(bf(scaled[:,6656:]+db));sp=np.maximum(d,0)+np.log1p(np.exp(-np.abs(d)));sp[d>20]=d[d>20];sp=dec(bf(sp));delta=np.maximum(sp,b['semantics']['delta_min'])
            if b['semantics']['delta_max'] is not None:delta=np.minimum(delta,b['semantics']['delta_max'])
            pred=dict(gate=bf(scaled[:,:3072]),x=bf(act[:,:3072]),B=bf(act[:,3072:3328]),C=bf(act[:,3328:]),delta=bf(delta));prior=next(q for q in b['reference_records'] if q['id']==item['id']);assert prior['file']==item['reference'];metrics={};flags={};increment={};field_numeric={}
            with np.load(item['file']['path'],allow_pickle=False) as z,np.load(prior['file']['path'],allow_pickle=False) as old:
                assert set(z.files)==set(pred)
                for k in pred:
                    actual=z[k];assert actual.dtype==np.dtype('u2') and actual.shape==tuple(sf[k]['shape']);counts['field_words']+=actual.size;counts['independent_field_BF16_mismatches']+=int(np.count_nonzero(actual!=pred[k]));err=rel(dec(actual),dec(pred[k]));maximum[k]=max(maximum.get(k,0.),err);field_numeric[k]=bool(err<=b['numeric']['field_vector_relative']);source=np.fromfile(sf[k]['path'],'u2').reshape(sf[k]['shape']);metrics[k]=mm(dec(actual),dec(source),rec['positions']);flags[k]=metrics[k]['relative_RMS']<=b['criteria']['each_case_each_source_field_relative'];assert item['source_field_bit_mismatches'][k]==int(np.count_nonzero(actual!=source));increment[k]=rel(dec(actual),dec(old['predicted_'+k]))
            meta(item['metrics'],metrics);meta(item['incremental_vs_actual_H_fields'],increment);assert item['source_field_gates']==flags and item['incremental_field_gates']=={k:v<=b['criteria']['incremental_each_field_relative'] for k,v in increment.items()};rows.append(dict(split=item['split'],H0_metrics=hm,H0_gate=item['H0_gate'],metrics=metrics,source_field_gates=flags,incremental_field_gates=item['incremental_field_gates'],numeric=field_numeric));print(json.dumps(dict(complete=len(rows),seconds=time.monotonic()-START)),flush=True)
        aggregates={}
        for split in ['FIT','DEV']:
            rr=[q for q in rows if q['split']==split];assert len(rr)==24;aggregates[split]={}
            for key in ['H0','gate','x','B','C','delta']:
                vals=[q['H0_metrics']['relative_RMS'] if key=='H0' else q['metrics'][key]['relative_RMS'] for q in rr];aggregates[split][key]=dict(mean=float(np.mean(vals)),worst=max(vals),failed_cases=sum(not(q['H0_gate'] if key=='H0' else q['source_field_gates'][key]) for q in rr))
        meta(r['aggregates'],aggregates);hg=all(q['H0_gate'] for q in rows);sg=all(all(q['source_field_gates'].values()) for q in rows);ig=all(all(q['incremental_field_gates'].values()) for q in rows);assert r['H0_all_pass']==hg and r['source_fields_all_pass']==sg and r['incremental_fields_all_pass']==ig;numeric=counts['H0_bit_mismatches']==counts['F0_bit_mismatches']==0 and all(all(q['numeric'].values()) for q in rows);stream.close();stream=None;check();guard();ad=dict(schema='TOKEN_GENERATOR_COMPILER_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],counts=counts,maximum=maximum,numeric_all_pass=numeric,H0_all_pass=hg,source_fields_all_pass=sg,incremental_fields_all_pass=ig,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS' if numeric else 'INDEPENDENT_STORED_NUMERIC_FAILURE',seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope='Every complete vocabulary H0/F0 BF16 word/extracted source conv parameter/scalar control/all48 calibration fields/actual source norms/metrics/flags/hash independently recomputed. CPU norm uses separate program with same literal Torch primitive backend,not independent Torch kernel implementation or original CUDA identity. No native state/causal chatbot/cost/DRAM/useful-n admission.');write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(adjudication=extent(a.out),decision=ad['decision'],counts=counts,maximum=maximum,H0_all_pass=hg,source_fields_all_pass=sg)),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['report','terminal','out']:p.add_argument('--'+k,type=Path,required=True)
    for k in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+k,required=True)
    main(p.parse_args())
