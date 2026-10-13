"""Independent full recovered-operand native temporal and baseline/source audit."""
import argparse,ctypes,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());terminal=json.loads(a.terminal.read_bytes());assert terminal['exit_code']==0 and terminal['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='startup';stream=None;rows=[];maximum={};flags=[];counts=dict(projected=0,scan=0,final_state=0,normalized=0,raw_scores=0,BF16_outputs=0,F32_scores_bit_verified=0)
    receipt=dict(schema='TRUE_INPUT_TEMPORAL_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);sources=[extent(Path(__file__)),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'TRUE_INPUT_TEMPORAL_AUDIT_PROTOCOL_20261013.md')]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+sources+[r['binding'],re,te]+[q['file'] for q in r['records']]:guard();assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');return z.reshape(e['shape']).astype('f8')
    def dec(z):assert z.dtype==np.dtype('<u2');return (z.astype('<u4')<<16).view('<f4').astype('f8')
    def bf(z):
        q=np.asarray(z,dtype='<f4');assert np.isfinite(q).all();u=q.view('<u4');return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2')
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def vector(x,y,tol,key):
        assert np.isfinite(x).all() and np.isfinite(y).all();err=rel(x,y);maximum[key]=max(maximum.get(key,0.),err);flags.append(dict(case=stage,control=key,relative=err,threshold=tol,passed=bool(err<=tol)))
    def scalar(x,y):
        d=float(abs(x-y)/max(abs(y),1.));maximum['scalar']=max(maximum.get('scalar',0.),d);assert d<=b['numeric']['scalar_relative'],(stage,x,y)
    def metrics(op,ref,pos):return dict(output_relative_RMS=rel(op,ref),centered_output_relative_RMS=rel(op-op.mean(0),ref-ref.mean(0)),label_output_relative_RMS=rel(op[pos],ref[pos]))
    try:
        check();get,stream=tensor_reader(b['parent']['source_weights'],np);assert json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps']==1e-5;capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];cases={q['id']:q for q in capture};old=json.loads(Path(b['state_report']['path']).read_bytes());bases={q['site']:q for q in old['bases']};base=json.loads(Path(b['native_report']['path']).read_bytes());recovery=json.loads(Path(b['generator_report']['path']).read_bytes());baserows={(q['site'],q['id']):q for q in base['records']};recoveryrows={(q['site'],q['id']):q for q in recovery['records']};assert b['sites']==[0,12,23] and len(r['records'])==144 and {(q['site'],q['id']) for q in r['records']}=={(s,i) for s in b['sites'] for i in cases};reference=ctypes.CDLL(b['reference_dll']['path']);fn=reference.scalar_output;fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];fn.restype=ctypes.c_int;ids=np.repeat(np.arange(48),64)
        for packet in b['exports']:
            site=packet['site'];stage=f'export/{site}';params=load(packet['parameters']);assert packet['basis']==bases[site]['basis'];basis=load(packet['basis']);assert np.max(np.abs(basis['W'].T@basis['V']-np.eye(96)))<=1e-8;assert np.array_equal(params['Ws'],basis['W'].T.astype('f4')) and np.array_equal(params['Vs'],basis['V'].T.astype('f4'));prefix=f'model.layers.{site}.mamba.'
            for key,name in [('Alog','A_log'),('D','D'),('gamma','norm.weight'),('O','out_proj.weight'),('in_proj','in_proj.weight')]:assert np.array_equal(params[key],get(prefix+name).astype('f4'))
            D=params['D'].astype('f8');aa=-np.exp(params['Alog'].astype('f8'));gamma=params['gamma'].astype('f8');O=params['O'].astype('f8');ws=params['Ws'].astype('f8');vs=params['Vs'].astype('f8')
            for item in [q for q in r['records'] if q['site']==site]:
                stage=f'{site}/{item["id"]}';guard();rec=cases[item['id']];n=rec['history'];f=rec['sites'][site]['fields'];be=baserows[(site,item['id'])];recoveryitem=recoveryrows[(site,item['id'])];assert item['baseline']==be['file'] and item['generator']==recoveryitem['file'];zz=load(be['file']);rr=load(recoveryitem['file']);x,bb,cc,dt,gate=[dec(rr['predicted_'+k]) for k in ['x','B','C','delta','gate']];assert x.shape==gate.shape==(n,3072) and bb.shape==cc.shape==(n,256) and dt.shape==(n,48);y=raw(f['y']);originalgate=raw(f['gate']);originalg=originalgate*np.exp(-np.logaddexp(0.,-originalgate));g=gate*np.exp(-np.logaddexp(0.,-gate));z=load(item['file']);assert len(z)==10 and (item['split'],item['domain'],n)==(rec['split'],rec['domain'],item['history']);bp=z['state96_B'].astype('f8');cp=z['state96_C'].astype('f8');lat=z['latent'].astype('f8');final=z['final_state'].astype('f8');assert bp.shape==cp.shape==(n,96) and lat.shape==(n,3072) and final.shape==(3072,96)
                for key in ['state96_B','state96_C','latent','final_state']:assert z[key].dtype==np.dtype('<f4')
                vector(bp,bb@ws.T,b['numeric']['projected_relative'],'B_projection');vector(cp,cc@vs.T,b['numeric']['projected_relative'],'C_projection');counts['projected']+=bp.size+cp.size;state=np.zeros((3072,96));direct=np.empty_like(lat)
                for t in range(n):
                    if t%32==0:guard()
                    d=dt[t,ids];state*=np.exp(d*aa[ids])[:,None];state+=(d*x[t])[:,None]*bp[t][None,:];direct[t]=np.sum(state*cp[t][None,:],axis=1)+D[ids]*x[t]
                vector(lat,direct,b['numeric']['scan_relative'],'sequential_scan');vector(final,state,b['numeric']['state_relative'],'final_state');counts['scan']+=lat.size;counts['final_state']+=final.size;ug=lat*g;invsource=1/np.sqrt(np.mean((y*originalg)**2,axis=1,keepdims=True)+1e-5);invown=1/np.sqrt(np.mean(ug**2,axis=1,keepdims=True)+1e-5);ref=raw(f['output']);mm=dict(source={},baseline={});baselinevectors={key:rel(z[key].astype('f8'),zz[key].astype('f8')) for key in ['state96_B','state96_C','latent','final_state']}
                for mode,inv in zip(b['modes'],[invsource,invown]):
                    normbits=z[mode+'__normalized'];score=z[mode+'__raw_scores'];outbits=z[mode+'__output'];assert normbits.shape==(n,3072) and score.shape==outbits.shape==(n,2048) and score.dtype==np.dtype('<f4');assert np.array_equal(bf(score),outbits);norm=dec(normbits);expectednorm=dec(bf((ug*inv)*gamma));vector(norm,expectednorm,b['numeric']['controlled_relative'],'normalized_'+mode);counts['normalized']+=norm.size;expected=np.empty_like(score)
                    for lo in range(0,n,128):guard();hi=min(lo+128,n);expected[lo:hi]=(norm[lo:hi]@O.T).astype('f4')
                    vector(score.astype('f8'),expected.astype('f8'),b['numeric']['controlled_relative'],'raw_smooth_'+mode);vector(dec(outbits),dec(bf(expected)),b['numeric']['controlled_relative'],'BF16_smooth_'+mode);scalarref=np.empty_like(score);rc=fn(n,params['O'].ctypes.data,normbits.ctypes.data,scalarref.ctypes.data);assert rc==0;errors=int(np.count_nonzero(score.view('u4')!=scalarref.view('u4')));maximum['F32_score_bit_mismatches']=max(maximum.get('F32_score_bit_mismatches',0),errors);flags.append(dict(case=stage,control='F32_scalar_bits_'+mode,mismatches=errors,passed=errors==0));counts['F32_scores_bit_verified']+=score.size;counts['raw_scores']+=score.size;counts['BF16_outputs']+=outbits.size;mm['source'][mode]=metrics(dec(outbits),ref,rec['positions']);mm['baseline'][mode]=metrics(dec(outbits),dec(zz[mode+'__output']),rec['positions']);baselinevectors[mode+'__normalized']=rel(norm,dec(zz[mode+'__normalized']));baselinevectors[mode+'__raw_scores']=rel(score.astype('f8'),zz[mode+'__raw_scores'].astype('f8'))
                    for group in ['source','baseline']:
                        for key,value in mm[group][mode].items():scalar(item['metrics'][group][mode][key],value)
                for key,value in baselinevectors.items():scalar(item['baseline_vectors'][key],value)
                temporalflags=dict(own_output=mm['baseline']['reconstructed_denominator']['output_relative_RMS']<=b['temporal_criteria']['each_case_own_output_relative'],final_state=baselinevectors['final_state']<=b['temporal_criteria']['each_case_final_state_relative']);assert item['temporal_flags']==temporalflags
                rows.append(dict(id=item['id'],site=site,split=rec['split'],domain=rec['domain'],metrics=mm,baseline_vectors=baselinevectors,temporal_flags=temporalflags));print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True)
        for site in b['sites']:
            for mode in b['modes']:
                for split in ['FIT','DEV']:
                    sr=[q for q in rows if q['site']==site and q['split']==split];assert len(sr)==24
                    for group in ['source','baseline']:
                        vals=[q['metrics'][group][mode]['output_relative_RMS'] for q in sr];c=float(np.mean([q['metrics'][group][mode]['centered_output_relative_RMS'] for q in sr]));dm={d:float(np.mean([q['metrics'][group][mode]['output_relative_RMS'] for q in sr if q['domain']==d])) for d in {q['domain'] for q in sr}};fresh=dict(mean=float(np.mean(vals)),worst=max(vals),centered_mean=c,worst_domain_mean=max(dm.values()));actual=r['aggregates'][str(site)][mode][split][group]
                        for key,value in fresh.items():scalar(actual[key],value)
                        assert set(dm)==set(actual['domain_means'])
                        for key,value in dm.items():scalar(actual['domain_means'][key],value)
                        if split=='DEV' and group=='source':assert r['gates'][str(site)][mode]==dict(mean=fresh['mean']<=.1,worst=fresh['worst']<=.2,centered=c<=.1,domain=fresh['worst_domain_mean']<=.2)
            for split in ['FIT','DEV']:
                sr=[q for q in rows if q['site']==site and q['split']==split];vals=[q['baseline_vectors']['final_state'] for q in sr];actual=r['aggregates'][str(site)]['final_state'][split];scalar(actual['mean'],float(np.mean(vals)));scalar(actual['worst'],max(vals));assert actual['failed_cases']==sum(not q['temporal_flags']['final_state'] for q in sr)
        temporalgates={key:all(q['temporal_flags'][key] for q in rows) for key in ['own_output','final_state']};assert r['temporal_gates']==temporalgates and r['temporal_all_pass']==all(temporalgates.values())
        stream.close();stream=None;assert r['native_history_calls']==144 and r['new_native_scans']==144  and all(r[k]==0 for k in ['source_forwards','optimizer_updates','basis_fits','generator_updates']);check();guard();ad=dict(schema='TRUE_INPUT_TEMPORAL_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],case_sites_verified=144,counts=counts,maximum=maximum,numeric_flags=flags,before_after_hashes_exact=True,temporal_gates=temporalgates,temporal_all_pass=all(temporalgates.values()),decision='INDEPENDENT_STORED_PASS' if all(q['passed'] for q in flags) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='All144 fixed actual-input regenerated-operand native histories/all exports/every timestep/finalstate/normalization/BF16 output/raw F32 scorebits/source and original-operand baseline metrics/temporal and source gates/hash. No causal/chatbot/speed/n admission.')
        write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(decision=ad['decision'],maximum=maximum,counts=counts,seconds=ad['seconds'])),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
