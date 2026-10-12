"""All48 fixed-candidate ideal outputs and complete CPU compact recurrence."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());terminal=json.loads(a.terminal.read_bytes());assert terminal['exit_code']==0 and terminal['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;stage='startup';rows=[];maximum={};ideal_words=0;machine_words=0;scan_words=0
    receipt=dict(schema='WHITENED_CANDIDATE_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);sources=[extent(Path(__file__)),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'WHITENED_CANDIDATE_AUDIT_PROTOCOL_20261012.md')]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[r['binding'],re,te]+sources+[q['file'] for q in r['records']]:guard();assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');z=z.reshape(e['shape']).astype('f8');assert np.isfinite(z).all();return z
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def array_check(x,y,tol,key):
        assert np.isfinite(x).all() and np.isfinite(y).all();err=rel(x,y);maximum[key]=max(maximum.get(key,0.),err);assert err<=tol,(stage,key,err)
    def scalar(x,y):
        err=float(abs(x-y)/max(abs(y),1.));maximum['scalar']=max(maximum.get('scalar',0.),err);assert err<=b['numeric']['audit_scalar_relative'],(stage,x,y,err)
    def decode(z):assert z.dtype==np.dtype('<u2');return (z.astype('<u4')<<16).view('<f4').astype('f8')
    def bf16(z):
        f=np.asarray(z,dtype='<f4');assert np.isfinite(f).all();u=f.view('<u4');return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2')
    def metrics(op,ref,pos):return dict(output_relative_RMS=rel(op,ref),centered_output_relative_RMS=rel(op-op.mean(0),ref-ref.mean(0)),label_output_relative_RMS=rel(op[pos],ref[pos]))
    try:
        check();get,stream=tensor_reader(b['parent']['source_weights'],np);p='model.layers.12.mamba.';dd=get(p+'D');aa=-np.exp(get(p+'A_log'));ow=get(p+'out_proj.weight');gamma=get(p+'norm.weight');F=ow*gamma[None,:];eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps'];capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];captures={q['id']:q for q in capture};pre=json.loads(Path(b['preflight']['path']).read_bytes());assert (pre['recommended_curve_coordinate'],pre['recommended_curve_alpha'])==(b['coordinate'],b['alpha']);assert b['candidate']==pre['curves'][b['coordinate']]['best_FIT_point']['parameters'];prior=json.loads(Path(b['prior']['path']).read_bytes());priorrows={q['id']:q for q in prior['ideal']};prefit={q['id']:q for q in pre['curves'][b['coordinate']]['best_FIT_point']['FIT_cases']};points=[]
        for e in [b['baseline'],b['candidate']]:
            z=load(e);points.append(([z[f'V_{h}'] for h in range(48)],[z[f'W_{h}'] for h in range(48)]))
        V,W=points[1];ranks=b['ranks'];offs=b['offsets'];ids=np.repeat(np.arange(48),ranks);assert sum(ranks)==384 and offs==[sum(ranks[:h]) for h in range(49)] and len(r['records'])==48 and {q['id'] for q in r['records']}==set(captures)
        for h in range(48):assert V[h].shape==W[h].shape==(64,ranks[h]) and np.max(np.abs(W[h].T@V[h]-np.eye(ranks[h])))<=b['numeric']['biorthogonality']
        for item in r['records']:
            stage=item['id'];guard();rec=captures[item['id']];n=rec['history'];f=rec['sites'][12]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);g=gate*np.exp(-np.logaddexp(0.,-gate));R=y.reshape(n,48,64)-dd[None,:,None]*x;skip=dd[None,:,None]*x;su=y*g;teacher=(su/np.sqrt(np.mean(su**2,axis=1,keepdims=True)+eps))@F.T;den=float(np.sum(teacher**2));pos=rec['positions'];z=load(item['file']);ideal={};assert len(z)==6 and (item['split'],item['domain'],item['history'])==(rec['split'],rec['domain'],n)
            for name,(v,w) in zip(['baseline','candidate'],points):
                Y=np.concatenate([(R[:,h]@w[h])@v[h].T+skip[:,h] for h in range(48)],axis=1);u=Y*g;op=(u/np.sqrt(np.mean(u**2,axis=1,keepdims=True)+eps))@F.T;array_check(z[name+'_ideal_output'],op,b['numeric']['audit_vector_relative'],'ideal_'+name);ideal_words+=op.size;loss=float(np.sum((op-teacher)**2)/den);ideal[name]=dict(loss=loss,relative_RMS=loss**.5,**metrics(op,teacher,pos))
                for key,value in ideal[name].items():scalar(item['ideal'][name][key],value)
            assert abs(ideal['baseline']['relative_RMS']-priorrows[rec['id']]['selected']['relative_RMS'])<=b['numeric']['curve_scalar_absolute']
            if rec['split']=='FIT':assert abs(ideal['candidate']['loss']-prefit[rec['id']]['loss'])<=b['numeric']['curve_scalar_absolute']
            px=z['projected_input'].astype('f8');lat=z['latent'].astype('f8');assert z['projected_input'].dtype==z['latent'].dtype==np.dtype('<f4') and px.shape==lat.shape==(n,384);wv=[w.astype('f4').astype('f8') for w in W];vv=[v.astype('f4').astype('f8') for v in V];xp=np.concatenate([x[:,h]@wv[h] for h in range(48)],axis=1);array_check(px,xp,b['numeric']['projected_input_relative'],'projected_input');bb=raw(f['B']);cc=raw(f['C']);dt=raw(f['delta']);state=np.zeros((384,256));direct=np.empty_like(lat)
            for t in range(n):
                if t%64==0:guard()
                delta=dt[t,ids];state*=np.exp(delta*aa[ids])[:,None];state+=(delta*px[t])[:,None]*bb[t][None,:];direct[t]=np.sum(state*cc[t][None,:],axis=1)+dd[ids]*px[t]
            array_check(lat,direct,b['numeric']['scan_relative'],'sequential_scan');scan_words+=lat.size;ep=np.concatenate([y.reshape(n,48,64)[:,h]@wv[h] for h in range(48)],axis=1);erp=np.concatenate([R[:,h]@wv[h] for h in range(48)],axis=1);pe=rel(lat,ep);rr=rel(lat-dd[ids][None,:]*px,erp);assert pe<=b['numeric']['scan_relative'] and rr<=b['numeric']['scan_relative'];scalar(item['full_scan_relative'],pe);scalar(item['recurrent_scan_relative'],rr);Y=np.concatenate([(lat[:,offs[h]:offs[h+1]]-dd[h]*px[:,offs[h]:offs[h+1]])@vv[h].T+skip[:,h] for h in range(48)],axis=1);ug=Y*g;source_inv=1/np.sqrt(np.mean(su**2,axis=1,keepdims=True)+eps);ref=raw(f['output']);machine={}
            for mode in ['full_source_denominator','reconstructed_denominator']:
                inv=source_inv if mode=='full_source_denominator' else 1/np.sqrt(np.mean(ug**2,axis=1,keepdims=True)+eps);inp=decode(bf16(ug*inv*gamma));op=np.empty((n,2048))
                for lo in range(0,n,128):guard();hi=min(lo+128,n);op[lo:hi]=decode(bf16(inp[lo:hi]@ow.T))
                actual=decode(z[mode]);array_check(actual,op,b['numeric']['audit_BF16_relative'],'machine_'+mode);machine_words+=op.size;machine[mode]=metrics(actual,ref,pos)
                for key,value in machine[mode].items():scalar(item['machine'][mode][key],value)
            rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],ideal=ideal,machine=machine));print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True)
        for split in ['FIT','DEV']:
            sr=[q for q in rows if q['split']==split];assert len(sr)==24
            for name in ['baseline','candidate']:
                for key in sr[0]['ideal'][name]:scalar(r['ideal_aggregates'][split][name][key],float(np.mean([q['ideal'][name][key] for q in sr])))
        for mode in r['machine_aggregates']:
            sr=[q for q in rows if q['split']=='DEV'];m=float(np.mean([q['machine'][mode]['output_relative_RMS'] for q in sr]));w=max(q['machine'][mode]['output_relative_RMS'] for q in sr);c=float(np.mean([q['machine'][mode]['centered_output_relative_RMS'] for q in sr]));d=max(float(np.mean([q['machine'][mode]['output_relative_RMS'] for q in sr if q['domain']==domain])) for domain in {q['domain'] for q in sr});expected=dict(mean=m,worst=w,centered_mean=c,worst_domain_mean=d)
            for key,value in expected.items():scalar(r['machine_aggregates'][mode][key],value)
            assert r['gates'][mode]==dict(mean=m<=.1,worst=w<=.2,centered=c<=.1,domain=d<=.2)
        assert abs(r['ideal_aggregates']['FIT']['candidate']['loss']-b['expected_FIT_loss'])<=b['numeric']['curve_scalar_absolute'] and r['new_packed_scans']==48 and all(r[k]==0 for k in ['optimizer_updates','source_forwards','native_calls']);stream.close();stream=None;check();guard();ad=dict(schema='WHITENED_CANDIDATE_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],case_count=48,ideal_output_words_verified=ideal_words,machine_output_words_verified=machine_words,full_sequential_scan_words_verified=scan_words,maximum=maximum,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='Fixed FIT point parameters/no new selection,all48 ideal and actual sequential/BF16 output/metric/domain/gate/source qualification. Source injection retained;no native/model/n/50 admission.')
        write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(ad),flush=True)
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
