"""Independent CPU contractions/BF16 conversion/full skip-output audit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes());terminal=json.loads(a.terminal.read_bytes());assert terminal['exit_code']==0 and terminal['result']==extent(a.report)
    assert extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil']
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;maximum=dict(output_relative=0.,metric_delta=0.,energy_relative=0.);rows=[];words=0;stage='startup'
    receipt=dict(schema='RECURRENT_ONLY_CHANNEL_SKIP_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    sources=[extent(Path(__file__)),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'RECURRENT_ONLY_CHANNEL_SKIP_AUDIT_PROTOCOL_20261012.md')]
    failure_files=[extent(DOC/'recurrent_only_channel_skip_v2_result_20261012.terminal.json'),extent(ROOT/'results/native_expert_scaling/recurrent_only_channel_skip_v2_20261012/first_fault.json')]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[r['binding'],extent(a.report),extent(a.terminal)]+sources+failure_files+[rr['output'] for rr in r['records']]:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4')
        z=z.reshape(e['shape']).astype('f8');assert np.isfinite(z).all();return z
    def decode(x):assert x.dtype==np.dtype('<u2');return (x.astype('<u4')<<16).view('<f4').astype('f8')
    def bf16(x):
        f=np.asarray(x,dtype='<f4');assert np.isfinite(f).all();bits=f.view('<u4');return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype('<u2')
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12))
    def close(x,y,tol,key,relative=False):
        delta=abs(x-y)/max(abs(y),1.) if relative else abs(x-y);maximum[key]=max(maximum[key],delta);assert delta<=tol,(stage,key,x,y,delta)
    def metrics(op,ref,pos):return dict(output_relative_RMS=rel(op,ref),centered_output_relative_RMS=rel(op-op.mean(axis=0),ref-ref.mean(axis=0)),label_output_relative_RMS=rel(op[pos],ref[pos]))
    try:
        check();get,stream=tensor_reader(b['parent']['source_weights'],np);eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps']
        captures={q['id']:q for q in b['capture']['records']};oldrows={(q['site'],q['id']):q for q in b['old']['records']};packets={p['site']:p for p in b['old']['selections']}
        for site in b['parent']['sites']:
            prefix=f'model.layers.{site}.mamba.';dd=get(prefix+'D');gamma=get(prefix+'norm.weight');out=get(prefix+'out_proj.weight');packet=packets[site]
            with np.load(packet['basis']['path'],allow_pickle=False) as z:maps={name:([z[f'{name}_V_{h}'].astype('f4').astype('f8') for h in range(48)],[z[f'{name}_W_{h}'].astype('f4').astype('f8') for h in range(48)]) for name in b['parent']['layouts']}
            for rr in [q for q in r['records'] if q['site']==site]:
                stage=f'{site}/{rr["id"]}';guard();rec=captures[rr['id']];n=rec['history'];f=rec['sites'][site]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);ref=raw(f['output']);sg=gate*np.exp(-np.logaddexp(0.,-gate));source_inv=1/np.sqrt(np.mean((y*sg)**2,axis=1,keepdims=True)+eps)
                assert rr['split']==rec['split'] and rr['domain']==rec['domain'] and rr['history']==n
                fresh=dict(id=rr['id'],site=site,split=rr['split'],domain=rr['domain'],history=n,arms={})
                with np.load(rr['output']['path'],allow_pickle=False) as saved:
                    assert len(saved.files)==8
                    for name in b['parent']['layouts']:
                        v,w=maps[name];oldrec=oldrows[(site,rec['id'])];lat=raw(oldrec['measurements'][name]['latent']);offs=packet['variants'][name]['offsets'];yo=np.empty((n,48,64),'f8');yn=np.empty_like(yo);es=np.empty_like(yo)
                        for h in range(48):
                            sl=slice(offs[h],offs[h+1]);xm=np.dot(x[:,h],w[h]);skip=dd[h]*x[:,h];yo[:,h]=np.dot(lat[:,sl],v[h].T)
                            # Independent additive correction, rather than producer's subtract-then-decode.
                            es[:,h]=dd[h]*(np.dot(xm,v[h].T)-x[:,h]);yn[:,h]=yo[:,h]-es[:,h]
                        yo=yo.reshape(n,3072);yn=yn.reshape(n,3072);es=es.reshape(n,3072);eo=yo-y;er=yn-y;d=rr['decomposition'][name]
                        energies=dict(old_error_energy=np.sum(eo**2),recurrent_error_energy=np.sum(er**2),skip_error_energy=np.sum(es**2),recurrent_skip_inner_product=np.sum(er*es),source_y_energy=np.sum(y**2))
                        for k,value in energies.items():close(d[k],float(value),b['numeric']['audit_energy_relative'],'energy_relative',True)
                        close(energies['old_error_energy'],energies['recurrent_error_energy']+energies['skip_error_energy']+2*energies['recurrent_skip_inner_product'],b['numeric']['audit_energy_relative'],'energy_relative',True)
                        assert d['identity_relative']<=b['numeric']['algebra_relative']
                        close(d['new_y_relative_RMS'],rel(yn,y),b['numeric']['audit_metric_absolute'],'metric_delta');close(d['old_y_relative_RMS'],rel(yo,y),b['numeric']['audit_metric_absolute'],'metric_delta')
                        fresh['arms'][name]={}
                        for arm,yr in [('projected_skip',yo),('full_skip',yn)]:
                            fresh['arms'][name][arm]={};ug=yr*sg
                            for mode in b['parent']['modes']:
                                inv=source_inv if mode=='full_source_denominator' else 1/np.sqrt(np.mean(ug**2,axis=1,keepdims=True)+eps)
                                inp=decode(bf16(ug*inv*gamma));key=f'{name}__{arm}__{mode}';bits=saved[key];assert bits.shape==(n,2048) and bits.dtype==np.dtype('<u2');actual=decode(bits);expected=np.empty((n,2048),'f8')
                                for lo in range(0,n,128):guard();hi=min(lo+128,n);expected[lo:hi]=decode(bf16(np.dot(inp[lo:hi],out.T)))
                                delta=rel(actual,expected);maximum['output_relative']=max(maximum['output_relative'],delta);assert delta<=b['numeric']['audit_output_relative'],(stage,key,delta)
                                mm=metrics(actual,ref,rec['positions']);fresh['arms'][name][arm][mode]=mm
                                for k,value in mm.items():close(rr['arms'][name][arm][mode][k],value,b['numeric']['audit_metric_absolute'],'metric_delta')
                                words+=actual.size
                        for mode,c in rr['controls'][name].items():
                            prior=oldrec['arms'][name][mode];delta=max(abs(c['original_F32_metrics'][k]-prior[k]) for k in prior)
                            assert abs(delta-c['original_metric_max_delta'])<=1e-12 and delta<=b['numeric']['old_control_metric_absolute']
                            assert c['original_precision_bridge_gate']==(c['F64_BF16_precision_bridge_relative']<=b['numeric']['precision_bridge_relative'])
                rows.append(fresh);print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True)
        stream.close();stream=None;assert len(rows)==144
        pass_any=False
        for name in b['parent']['layouts']:
            for arm in ['projected_skip','full_skip']:
                for mode in b['parent']['modes']:
                    site_pass=[]
                    for site in b['parent']['sites']:
                        sk=str(site)
                        for split in ['FIT','DEV']:
                            sr=[q for q in rows if q['site']==site and q['split']==split];assert len(sr)==24
                            for k in sr[0]['arms'][name][arm][mode]:
                                values=[q['arms'][name][arm][mode][k] for q in sr];aa=r['aggregates'][sk][split][name][arm][mode][k]
                                close(aa['mean'],float(np.mean(values)),1e-10,'metric_delta');close(aa['worst'],max(values),1e-10,'metric_delta')
                                for domain in sorted({q['domain'] for q in sr}):
                                    vals=[q['arms'][name][arm][mode][k] for q in sr if q['domain']==domain];assert len(vals)==2;dr=r['domains'][sk][split][name][arm][mode][domain][k]
                                    close(dr['mean'],float(np.mean(vals)),1e-10,'metric_delta');close(dr['worst'],max(vals),1e-10,'metric_delta')
                        sr=[q for q in rows if q['site']==site and q['split']=='DEV'];values=[q['arms'][name][arm][mode]['output_relative_RMS'] for q in sr];c=b['criteria']
                        domainmax=max(float(np.mean([q['arms'][name][arm][mode]['output_relative_RMS'] for q in sr if q['domain']==d])) for d in {q['domain'] for q in sr})
                        flags=dict(mean=float(np.mean(values))<=c['DEV_output_mean'],case_worst=max(values)<=c['DEV_case_worst'],domain_mean_worst=domainmax<=c['DEV_domain_mean_worst'],centered_mean=float(np.mean([q['arms'][name][arm][mode]['centered_output_relative_RMS'] for q in sr]))<=c['DEV_centered_mean'])
                        assert flags==r['gates'][sk][name][arm][mode];site_pass.append(all(flags.values()))
                    if arm=='full_skip' and all(site_pass):pass_any=True
        assert r['decision']==('LOCAL_FULL_SKIP_BUDGET_PASS_PENDING_AUDIT' if pass_any else 'LOCAL_FULL_SKIP_BUDGET_FAIL_PENDING_AUDIT')
        assert r['original_precision_bridge_all_pass']==all(c['original_precision_bridge_gate'] for rr in r['records'] for cs in rr['controls'].values() for c in cs.values())
        assert all(r[k]==0 for k in ['source_forwards','new_scans','optimizer_updates','native_calls'])
        check();guard();ad=dict(schema='RECURRENT_ONLY_CHANNEL_SKIP_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],first_fault_custody=failure_files,records=len(rows),full_output_words_verified=words,maximum=maximum,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',precision_bridge_admitted=r['original_precision_bridge_all_pass'],seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='All stored F64-controlled BF16 outputs/decomposition/metrics/aggregates/gates independently verified. Original GPU F32 precision bridge magnitudes are bound observations, not exact CUDA replay. First V2 failure has no saved output arrays. No native/chatbot/n/50 admission.')
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
