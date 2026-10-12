"""Full CPU covariance, weighted basis, every-coordinate sequential SSD audit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    report_extent=extent(a.report);terminal_extent=extent(a.terminal);assert report_extent['sha256']==a.report_sha and terminal_extent['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes());terminal=json.loads(a.terminal.read_bytes());assert terminal['exit_code']==0 and terminal['result']==report_extent and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes())
    assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'];assert not a.out.exists()
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;maximum=dict(covariance_relative=0.,weighted_kernel_relative=0.,weighted_tail_relative=0.,projected_input_relative=0.,sequential_scan_relative=0.,output_relative=0.,metric_delta=0.,energy_relative=0.);rows=[];words=0;sequence_words=0;input_words=0;basis_heads=0;stage='startup'
    receipt=dict(schema='RECURRENT_OUTPUT_BASES_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    sources=[extent(Path(__file__)),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'RECURRENT_OUTPUT_BASES_AUDIT_PROTOCOL_20261012.md')]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[r['binding'],report_extent,terminal_extent]+sources+[rr['output'] for rr in r['records']]+[p['basis'] for p in r['selections']]:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');z=z.reshape(e['shape']).astype('f8');assert np.isfinite(z).all();return z
    def decode(x):assert x.dtype==np.dtype('<u2');return (x.astype('<u4')<<16).view('<f4').astype('f8')
    def bf16(x):
        f=np.asarray(x,dtype='<f4');assert np.isfinite(f).all();bits=f.view('<u4');return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype('<u2')
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12))
    def close(x,y,tol,key,relative=False):
        delta=abs(x-y)/max(abs(y),1.) if relative else abs(x-y);maximum[key]=max(maximum[key],delta);assert delta<=tol,(stage,key,x,y,delta)
    def array_check(x,y,tol,key):
        delta=rel(x,y);maximum[key]=max(maximum[key],delta);assert delta<=tol,(stage,key,delta)
    def metrics(op,ref,pos):return dict(output_relative_RMS=rel(op,ref),centered_output_relative_RMS=rel(op-op.mean(axis=0),ref-ref.mean(axis=0)),label_output_relative_RMS=rel(op[pos],ref[pos]))
    def root(g):
        ev,u=np.linalg.eigh((g+g.T)/2);assert -ev.min()/max(np.max(np.abs(ev)),1e-300)<=b['numeric']['PSD_relative_negative'];return (u*np.sqrt(np.maximum(ev,0)))@u.T
    try:
        check();get,stream=tensor_reader(b['parent']['source_weights'],np);eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps'];capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];fit=[q for q in capture if q['split']=='FIT'];assert len(fit)==24
        captures={q['id']:q for q in capture};old=json.loads(Path(b['old']['path']).read_bytes());oldpackets={p['site']:p for p in old['selections']};prior=json.loads(Path(b['prior']['path']).read_bytes());priorrows={(q['site'],q['id']):q for q in prior['records']}
        for packet in r['selections']:
            site=packet['site'];prefix=f'model.layers.{site}.mamba.';dd=get(prefix+'D');aa=-np.exp(get(prefix+'A_log'));gamma=get(prefix+'norm.weight');out=get(prefix+'out_proj.weight');gy=np.zeros((48,64,64),'f8')
            for rec in fit:
                stage=f'FIT_covariance/{site}/{rec["id"]}';guard();f=rec['sites'][site]['fields'];x=raw(f['x']).reshape(-1,48,64);y=raw(f['y']).reshape(-1,48,64);yr=y-dd[None,:,None]*x
                for h in range(48):gy[h]+=np.dot(yr[:,h].T,yr[:,h])/(24*rec['history'])
            with np.load(oldpackets[site]['basis']['path'],allow_pickle=False) as z:gm=z['GM'].copy()
            with np.load(packet['basis']['path'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            assert np.array_equal(gm,arrays['GM']) and packet['GM_reused']==oldpackets[site]['basis'];array_check(arrays['GY_rec'],gy,b['numeric']['covariance_relative'],'covariance_relative');assert packet['FIT_cases']==24 and packet['FIT_positions']==sum(q['history'] for q in fit)
            sy=np.stack([root(g) for g in gy]);sm=np.stack([root(g) for g in gm]);kernels=sm@sy;uu,ss,qh=np.linalg.svd(kernels,full_matrices=False);choices=sorted([(ss[h,rank]**2,h,rank) for h in range(48) for rank in range(1,64)],key=lambda q:(-q[0],q[1],q[2]))[:336];adaptive=[1+sum(h==j for _,j,_ in choices) for h in range(48)];assert sum(adaptive)==384
            maps={}
            for name in b['parent']['layouts']:
                ranks=packet['variants'][name]['ranks'];assert ranks==([8]*48 if name=='fixed8' else adaptive);offs=packet['variants'][name]['offsets'];assert offs==[sum(ranks[:h]) for h in range(49)];vv=[];ww=[];tail=0.
                for h,rank in enumerate(ranks):
                    stage=f'basis/{site}/{name}/{h}';v=arrays[f'{name}_V_{h}'];w=arrays[f'{name}_W_{h}'];assert v.shape==w.shape==(64,rank)
                    bio=np.max(np.abs(w.T@v-np.eye(rank)));assert bio<=b['numeric']['biorthogonality'];assert ss[h,rank-1]>b['numeric']['minimum_retained_singular']
                    actual=sm[h]@v@w.T@sy[h];reference=(uu[h,:,:rank]*ss[h,:rank])@qh[h,:rank];array_check(actual,reference,b['numeric']['weighted_kernel_relative'],'weighted_kernel_relative')
                    expected_tail=float(np.sum(ss[h,rank:]**2));close(float(np.linalg.norm(kernels[h]-actual)**2),expected_tail,b['numeric']['weighted_kernel_relative'],'weighted_tail_relative',True);tail+=expected_tail;vv.append(v.astype('f4').astype('f8'));ww.append(w.astype('f4').astype('f8'));basis_heads+=1
                close(packet['variants'][name]['weighted_tail_energy'],tail,b['numeric']['weighted_kernel_relative'],'weighted_tail_relative',True);maps[name]=(vv,ww,offs,ranks)
            print(json.dumps(dict(stage='basis_verified',site=site,seconds=time.perf_counter()-CLOCK)),flush=True)
            for rr in [q for q in r['records'] if q['site']==site]:
                stage=f'evaluation/{site}/{rr["id"]}';guard();rec=captures[rr['id']];n=rec['history'];f=rec['sites'][site]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);ref=raw(f['output']);bb=raw(f['B']);cc=raw(f['C']);dt=raw(f['delta']);sg=gate*np.exp(-np.logaddexp(0.,-gate));source_inv=1/np.sqrt(np.mean((y*sg)**2,axis=1,keepdims=True)+eps)
                assert rr['split']==rec['split'] and rr['domain']==rec['domain'] and rr['history']==n;fresh=dict(id=rr['id'],site=site,split=rr['split'],domain=rr['domain'],history=n,arms={})
                with np.load(rr['output']['path'],allow_pickle=False) as saved:
                    assert len(saved.files)==12
                    for name in b['parent']['layouts']:
                        v,w,offs,ranks=maps[name];latbits=saved[name+'__latent'];pxbits=saved[name+'__projected_input'];assert latbits.dtype==pxbits.dtype==np.dtype('<f4') and latbits.shape==pxbits.shape==(n,384);lat=latbits.astype('f8');px=pxbits.astype('f8');assert np.isfinite(lat).all() and np.isfinite(px).all()
                        expected_px=np.concatenate([np.dot(x[:,h],w[h]) for h in range(48)],axis=1);array_check(px,expected_px,b['numeric']['projected_input_relative'],'projected_input_relative');input_words+=px.size
                        ids=np.repeat(np.arange(48),ranks);state=np.zeros((384,256),'f8');direct=np.empty_like(lat)
                        for t in range(n):
                            if t%64==0:guard()
                            delta=dt[t,ids];state*=np.exp(delta*aa[ids])[:,None];state+=(delta*px[t])[:,None]*bb[t][None,:];direct[t]=np.sum(state*cc[t][None,:],axis=1)+dd[ids]*px[t]
                        array_check(lat,direct,b['numeric']['scalar_sequence_relative'],'sequential_scan_relative');sequence_words+=lat.size
                        ep=np.concatenate([np.dot(y.reshape(n,48,64)[:,h],w[h]) for h in range(48)],axis=1);erp=np.concatenate([np.dot(y.reshape(n,48,64)[:,h]-dd[h]*x[:,h],w[h]) for h in range(48)],axis=1);skip=dd[ids][None,:]*px
                        pe=rel(lat,ep);re=rel(lat-skip,erp);assert pe<=b['numeric']['projected_scan_relative'] and re<=b['numeric']['projected_scan_relative'];close(rr['scan_checks'][name]['full_relative'],pe,1e-10,'metric_delta');close(rr['scan_checks'][name]['recurrent_relative'],re,1e-10,'metric_delta')
                        yo=np.empty((n,48,64),'f8');yn=np.empty_like(yo);es=np.empty_like(yo)
                        for h in range(48):sl=slice(offs[h],offs[h+1]);yo[:,h]=np.dot(lat[:,sl],v[h].T);es[:,h]=dd[h]*(np.dot(px[:,sl],v[h].T)-x[:,h]);yn[:,h]=yo[:,h]-es[:,h]
                        yo=yo.reshape(n,3072);yn=yn.reshape(n,3072);es=es.reshape(n,3072);eo=yo-y;er=yn-y;d=rr['decomposition'][name];energies=dict(old_error_energy=np.sum(eo**2),recurrent_error_energy=np.sum(er**2),skip_error_energy=np.sum(es**2),recurrent_skip_inner_product=np.sum(er*es),source_y_energy=np.sum(y**2))
                        for k,value in energies.items():close(d[k],float(value),b['numeric']['audit_energy_relative'],'energy_relative',True)
                        close(energies['old_error_energy'],energies['recurrent_error_energy']+energies['skip_error_energy']+2*energies['recurrent_skip_inner_product'],b['numeric']['audit_energy_relative'],'energy_relative',True);assert d['identity_relative']<=b['numeric']['algebra_relative'];close(d['new_y_relative_RMS'],rel(yn,y),b['numeric']['audit_metric_absolute'],'metric_delta');close(d['old_y_relative_RMS'],rel(yo,y),b['numeric']['audit_metric_absolute'],'metric_delta');fresh['arms'][name]={}
                        for arm,yr in [('projected_skip',yo),('full_skip',yn)]:
                            fresh['arms'][name][arm]={};ug=yr*sg
                            for mode in b['parent']['modes']:
                                inv=source_inv if mode=='full_source_denominator' else 1/np.sqrt(np.mean(ug**2,axis=1,keepdims=True)+eps);inp=decode(bf16(ug*inv*gamma));key=f'{name}__{arm}__{mode}';bits=saved[key];assert bits.shape==(n,2048);actual=decode(bits);expected=np.empty_like(actual)
                                for lo in range(0,n,128):guard();hi=min(lo+128,n);expected[lo:hi]=decode(bf16(np.dot(inp[lo:hi],out.T)))
                                array_check(actual,expected,b['numeric']['audit_output_relative'],'output_relative');mm=metrics(actual,ref,rec['positions']);fresh['arms'][name][arm][mode]=mm
                                for k,value in mm.items():close(rr['arms'][name][arm][mode][k],value,b['numeric']['audit_metric_absolute'],'metric_delta')
                                words+=actual.size;control=rr['controls'][name][arm][mode];pm=priorrows[(site,rec['id'])]['arms'][name][arm][mode];delta=max(abs(control['metrics'][k]-pm[k]) for k in pm);assert abs(delta-control['max_prior_delta'])<=1e-12 and delta<=b['numeric']['old_metric_absolute']
                rows.append(fresh);print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True)
        stream.close();stream=None;assert len(rows)==144 and basis_heads==288;pass_any=False
        for name in b['parent']['layouts']:
            for arm in ['projected_skip','full_skip']:
                for mode in b['parent']['modes']:
                    site_pass=[]
                    for site in b['parent']['sites']:
                        sk=str(site)
                        for split in ['FIT','DEV']:
                            sr=[q for q in rows if q['site']==site and q['split']==split];assert len(sr)==24
                            for k in sr[0]['arms'][name][arm][mode]:
                                vals=[q['arms'][name][arm][mode][k] for q in sr];aa=r['aggregates'][sk][split][name][arm][mode][k];close(aa['mean'],float(np.mean(vals)),1e-10,'metric_delta');close(aa['worst'],max(vals),1e-10,'metric_delta')
                                for domain in sorted({q['domain'] for q in sr}):
                                    vals=[q['arms'][name][arm][mode][k] for q in sr if q['domain']==domain];assert len(vals)==2;dr=r['domains'][sk][split][name][arm][mode][domain][k];close(dr['mean'],float(np.mean(vals)),1e-10,'metric_delta');close(dr['worst'],max(vals),1e-10,'metric_delta')
                        sr=[q for q in rows if q['site']==site and q['split']=='DEV'];vals=[q['arms'][name][arm][mode]['output_relative_RMS'] for q in sr];c=b['criteria'];domainmax=max(float(np.mean([q['arms'][name][arm][mode]['output_relative_RMS'] for q in sr if q['domain']==d])) for d in {q['domain'] for q in sr})
                        flags=dict(mean=float(np.mean(vals))<=c['DEV_output_mean'],case_worst=max(vals)<=c['DEV_case_worst'],domain_mean_worst=domainmax<=c['DEV_domain_mean_worst'],centered_mean=float(np.mean([q['arms'][name][arm][mode]['centered_output_relative_RMS'] for q in sr]))<=c['DEV_centered_mean']);assert flags==r['gates'][sk][name][arm][mode];site_pass.append(all(flags.values()))
                    if arm=='full_skip' and all(site_pass):pass_any=True
        assert r['decision']==('RECURRENT_BASES_LOCAL_BUDGET_PASS_PENDING_AUDIT' if pass_any else 'RECURRENT_BASES_LOCAL_BUDGET_FAIL_PENDING_AUDIT') and r['new_scans']==288 and r['basis_head_fits']==144 and all(r[k]==0 for k in ['source_forwards','optimizer_updates','native_calls']) and not r['old_precision_bridge_admitted']
        check();guard();ad=dict(schema='RECURRENT_OUTPUT_BASES_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=report_extent,terminal=terminal_extent,binding=r['binding'],records=len(rows),basis_head_layouts_verified=basis_heads,all_output_words_verified=words,all_projected_input_words_verified=input_words,all_sequential_scan_words_verified=sequence_words,maximum=maximum,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='Full FIT recurrent covariance,unchanged stored GM,all balanced bases/ranks,all384-coordinate state256 sequential scans,all controlled BF16 outputs/energies/metrics/aggregates/gates verified. Prior-control deltas are bound observations versus independently audited prior outputs,not exact CUDA replay. No native/model/n/50 admission.')
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
