"""All48 native states/readouts,full scalar FMA score reference and source gates."""
import argparse,ctypes,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());terminal=json.loads(a.terminal.read_bytes());assert terminal['exit_code']==0 and terminal['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;stage='startup';rows=[];maximum={};numeric_flags=[];counts=dict(projected=0,scan=0,final_state=0,normalized=0,raw_scores=0,BF16_outputs=0,F32_scores_bit_verified=0)
    receipt=dict(schema='COMPOSED_OPERATOR_MIXED_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);sources=[extent(Path(__file__)),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'COMPOSED_OPERATOR_MIXED_AUDIT_PROTOCOL_20261013.md')]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[r['binding'],re,te]+sources+[q['file'] for q in r['records']]:guard();assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');return z.reshape(e['shape']).astype('f8')
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def vector(x,y,tol,key):
        assert np.isfinite(x).all() and np.isfinite(y).all();err=rel(x,y);maximum[key]=max(maximum.get(key,0.),err);numeric_flags.append(dict(case=stage,control=key,relative=err,threshold=tol,passed=bool(err<=tol)));return err
    def scalar(x,y):
        err=float(abs(x-y)/max(abs(y),1.));maximum['scalar']=max(maximum.get('scalar',0.),err);assert err<=b['numeric']['scalar_relative'],(stage,x,y,err)
    def decode(z):assert z.dtype==np.dtype('<u2');return (z.astype('<u4')<<16).view('<f4').astype('f8')
    def bf16(z):
        f=np.asarray(z,dtype='<f4');assert np.isfinite(f).all();u=f.view('<u4');return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2')
    def metrics(op,ref,pos):return dict(output_relative_RMS=rel(op,ref),centered_output_relative_RMS=rel(op-op.mean(0),ref-ref.mean(0)),label_output_relative_RMS=rel(op[pos],ref[pos]))
    try:
        check();params=load(b['parameters']);parent=b['parent'];channel=load(parent['channel_basis']);sb=load(parent['state_basis']);expectedW=np.concatenate([channel[f'adaptive384_W_{h}'].T.astype('f4') for h in range(48)]);expectedV=np.concatenate([channel[f'adaptive384_V_{h}'].astype('f4').ravel() for h in range(48)]);assert np.array_equal(params['Wc'],expectedW) and np.array_equal(params['Vc'],expectedV) and np.array_equal(params['Ws'],sb['W'].T.astype('f4')) and np.array_equal(params['Vs'],sb['V'].T.astype('f4'));get,stream=tensor_reader(parent['parent']['source_weights'],np);prefix='model.layers.23.mamba.'
        for key,name in [('Alog','A_log'),('D','D'),('gamma','norm.weight'),('O','out_proj.weight'),('in_proj','in_proj.weight')]:assert np.array_equal(params[key],get(prefix+name).astype('f4'))
        assert json.loads(Path(parent['parent']['source_config']).read_bytes())['rms_norm_eps']==1e-5;capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];captures={q['id']:q for q in capture};old=json.loads(Path(b['parent_report']['path']).read_bytes());oldrows={q['id']:q for q in old['records']};ranks=b['ranks'];offs=b['offsets'];ids=np.repeat(np.arange(48),ranks);assert sum(ranks)==384 and offs==[sum(ranks[:h]) for h in range(49)] and len(r['records'])==48 and {q['id'] for q in r['records']}==set(captures);w=[channel[f'adaptive384_W_{h}'].astype('f4').astype('f8') for h in range(48)];v=[channel[f'adaptive384_V_{h}'].astype('f4').astype('f8') for h in range(48)];D=params['D'].astype('f8');aa=-np.exp(params['Alog'].astype('f8'));gamma=params['gamma'].astype('f8');O=params['O'].astype('f8');refdll=ctypes.CDLL(b['reference_dll']['path']);fn=refdll.scalar_output;fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];fn.restype=ctypes.c_int
        for item in r['records']:
            stage=item['id'];guard();rec=captures[stage];n=rec['history'];f=rec['sites'][23]['fields'];x=raw(f['x']).reshape(n,48,64);bb=raw(f['B']);cc=raw(f['C']);dt=raw(f['delta']);gate=raw(f['gate']);y=raw(f['y']);g=gate*np.exp(-np.logaddexp(0.,-gate));skip=D[None,:,None]*x;z=load(item['file']);assert len(z)==11 and (item['split'],item['domain'],item['history'])==(rec['split'],rec['domain'],n);px=z['projected_input'].astype('f8');bp=z['state96_B'].astype('f8');cp=z['state96_C'].astype('f8');lat=z['latent'].astype('f8');final=z['final_state'].astype('f8');assert px.shape==lat.shape==(n,384) and bp.shape==cp.shape==(n,96) and final.shape==(384,96)
            for key in ['projected_input','state96_B','state96_C','latent','final_state']:assert z[key].dtype==np.dtype('<f4')
            vector(px,np.concatenate([x[:,h]@w[h] for h in range(48)],axis=1),b['numeric']['projected_relative'],'input_projection');vector(bp,bb@params['Ws'].astype('f8').T,b['numeric']['projected_relative'],'B_projection');vector(cp,cc@params['Vs'].astype('f8').T,b['numeric']['projected_relative'],'C_projection');counts['projected']+=px.size+bp.size+cp.size;state=np.zeros((384,96));direct=np.empty_like(lat)
            for t in range(n):
                if t%64==0:guard()
                d=dt[t,ids];state*=np.exp(d*aa[ids])[:,None];state+=(d*px[t])[:,None]*bp[t][None,:];direct[t]=np.sum(state*cp[t][None,:],axis=1)+D[ids]*px[t]
            vector(lat,direct,b['numeric']['scan_relative'],'sequential_scan');vector(final,state,b['numeric']['state_relative'],'final_state');counts['scan']+=lat.size;counts['final_state']+=final.size;Y=np.concatenate([(lat[:,offs[h]:offs[h+1]]-D[h]*px[:,offs[h]:offs[h+1]])@v[h].T+skip[:,h] for h in range(48)],axis=1);ug=Y*g;invsource=1/np.sqrt(np.mean((y*g)**2,axis=1,keepdims=True)+1e-5);invown=1/np.sqrt(np.mean(ug**2,axis=1,keepdims=True)+1e-5);ref=raw(f['output']);pr=oldrows[stage];assert item['prior']==pr['file'];prior=load(pr['file']);comparison={key:rel(z[key].astype('f8'),prior[oldkey].astype('f8')) for key,oldkey in [('projected_input','projected_input'),('state96_B','state96_B'),('state96_C','state96_C'),('latent','joint96_latent')]}
            for key,value in comparison.items():scalar(item['prior_comparison'][key],value)
            mm={}
            for path in b['paths']:
                mm[path]={}
                for mode,inv in zip(b['modes'],[invsource,invown]):
                    stem=path+'__'+mode;normbits=z[stem+'__normalized'];score=z[stem+'__raw_scores'];outbits=z[stem+'__output'];assert normbits.shape==(n,3072) and score.shape==outbits.shape==(n,2048) and score.dtype==np.dtype('<f4');assert np.array_equal(bf16(score),outbits);norm=decode(normbits);idealnorm=decode(bf16((ug*inv)*gamma));tol=b['numeric']['controlled_relative'];vector(norm,idealnorm,tol,path+'_normalized_smooth');counts['normalized']+=norm.size
                    if path=='controlled_F64':
                        expected=np.empty_like(score)
                        for lo in range(0,n,128):guard();hi=min(lo+128,n);expected[lo:hi]=(norm[lo:hi]@O.T).astype('f4')
                        vector(score.astype('f8'),expected.astype('f8'),b['numeric']['controlled_relative'],'controlled_raw_scores');vector(decode(outbits),decode(bf16(expected)),b['numeric']['controlled_relative'],'controlled_BF16_scores')
                    else:
                        smooth_scores=(norm@O.T).astype('f4');vector(score.astype('f8'),smooth_scores.astype('f8'),b['numeric']['controlled_relative'],'mixed_raw_smooth');vector(decode(outbits),decode(bf16(smooth_scores)),b['numeric']['controlled_relative'],'mixed_BF16_smooth')
                        expected=np.empty_like(score);start=time.perf_counter();rc=fn(n,params['O'].ctypes.data,normbits.ctypes.data,expected.ctypes.data);assert rc==0;biterrors=int(np.count_nonzero(score.view('u4')!=expected.view('u4')));maximum['F32_score_bit_mismatches']=max(maximum.get('F32_score_bit_mismatches',0),biterrors);numeric_flags.append(dict(case=stage,control='F32_scalar_raw_score_bits_'+mode,mismatches=biterrors,passed=biterrors==0));counts['F32_scores_bit_verified']+=score.size
                    counts['raw_scores']+=score.size;counts['BF16_outputs']+=outbits.size;op=decode(outbits);mm[path][mode]=metrics(op,ref,rec['positions'])
                    for key,value in mm[path][mode].items():scalar(item['metrics'][path][mode][key],value)
                    bridge=rel(op,decode(prior['channel384_state96__'+mode]));scalar(item['old_bridge'][path][mode]['relative'],bridge);assert item['old_bridge'][path][mode]['pass_original_1e3']==(bridge<=b['numeric']['old_bridge_relative'])
            rows.append(dict(id=stage,split=rec['split'],domain=rec['domain'],metrics=mm));print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True)
        for path in b['paths']:
            for mode in b['modes']:
                for split in ['FIT','DEV']:
                    sr=[q for q in rows if q['split']==split];assert len(sr)==24;vals=[q['metrics'][path][mode]['output_relative_RMS'] for q in sr];c=float(np.mean([q['metrics'][path][mode]['centered_output_relative_RMS'] for q in sr]));dm={d:float(np.mean([q['metrics'][path][mode]['output_relative_RMS'] for q in sr if q['domain']==d])) for d in {q['domain'] for q in sr}};fresh=dict(mean=float(np.mean(vals)),worst=max(vals),centered_mean=c,worst_domain_mean=max(dm.values()));actual=r['aggregates'][path][mode][split]
                    for key,value in fresh.items():scalar(actual[key],value)
                    assert set(dm)==set(actual['domain_means'])
                    for key,value in dm.items():scalar(actual['domain_means'][key],value)
                    if split=='DEV':assert r['gates'][path][mode]==dict(mean=fresh['mean']<=.1,worst=fresh['worst']<=.2,centered=c<=.1,domain=fresh['worst_domain_mean']<=.2)
        stream.close();stream=None;assert r['native_history_calls']==48 and r['new_native_scans']==48 and all(r[k]==0 for k in ['basis_fits','source_forwards','optimizer_updates']);check();guard();passed=all(q['passed'] for q in numeric_flags);ad=dict(schema='COMPOSED_OPERATOR_MIXED_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],cases_verified=48,counts=counts,maximum=maximum,numeric_flags=numeric_flags,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS' if passed else 'INDEPENDENT_STORED_NUMERIC_FAILURE',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='Full parameter export/source custody,all actual native projected inputs and explicit CPU96-state recurrence/final states,all normalized vectors/controlled outputs/every F32 scalar-reference score bit/all BF16 outputs/metrics/domains/bridges/gates. Full audit even failed control flags;quality and old precision bridges separate. No whole chatbot/speed/n admission.')
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
