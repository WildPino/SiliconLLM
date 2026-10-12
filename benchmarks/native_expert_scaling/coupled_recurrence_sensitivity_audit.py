"""Independent full CPU radial/tangent and explicit head-vector coupling audit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());terminal=json.loads(a.terminal.read_bytes());assert terminal['exit_code']==0 and terminal['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes())
    assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;stage='startup';rows=[];proxies={};vectors=0;stats_words=0;heads_words=0;control_count=0;maxima=dict(vector_relative=0.,scalar_relative=0.,Gram_relative=0.,control_scalar_absolute=0.)
    receipt=dict(schema='COUPLED_RECURRENCE_SENSITIVITY_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);sources=[extent(Path(__file__)),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'COUPLED_RECURRENCE_SENSITIVITY_AUDIT_PROTOCOL_20261012.md')]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[r['binding'],re,te]+sources+[q['output'] for q in r['records']]+[p['file'] for p in r['FIT_Grams']]:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');z=z.reshape(e['shape']).astype('f8');assert np.isfinite(z).all();return z
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def array_check(x,y,tol,key):
        assert np.isfinite(x).all() and np.isfinite(y).all();d=rel(x,y);maxima[key]=max(maxima[key],d);assert d<=tol,(stage,key,d)
    def scalar_check(x,y):
        d=abs(x-y)/max(abs(y),1.);maxima['scalar_relative']=max(maxima['scalar_relative'],d);assert d<=b['numeric']['audit_scalar_relative'],(stage,x,y,d)
    def control_scalar(x,y):
        d=abs(x-y);maxima['control_scalar_absolute']=max(maxima['control_scalar_absolute'],d);assert d<=1e-8,(stage,x,y,d)
    def radial_tangent(u,du,r,eps):
        axis=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-300);radial=axis*np.sum(axis*du,axis=1,keepdims=True);return (du-radial)/r+radial*eps/r**3
    try:
        check();get,stream=tensor_reader(b['parent']['source_weights'],np);eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps'];capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];captures={q['id']:q for q in capture};prior=json.loads(Path(b['prior']['path']).read_bytes());priorrows={(q['site'],q['id']):q for q in prior['records']};packets={p['site']:p for p in prior['selections']}
        assert len(r['records'])==len({(q['site'],q['id']) for q in r['records']})==144 and {(q['site'],q['id']) for q in r['records']}=={(s,q['id']) for s in b['parent']['sites'] for q in capture};assert {p['site'] for p in r['FIT_Grams']}=={0,12,23}
        shortest=min((q for q in capture if q['split']=='FIT'),key=lambda q:(q['history'],q['id']));assert b['control_case']==shortest['id'] and b['control_positions']==[0,shortest['history']//2,shortest['history']-1]
        keys=['source_energy','finite_energy','J_energy','no_radial_energy','remainder_energy','radial_energy','head_J_sum','head_no_radial_sum','scan_difference_y_energy','ideal_y_energy','cross_head_energy'];assert b['stat_keys']==keys
        for gpkt in r['FIT_Grams']:
            site=gpkt['site'];packet=packets[site];prefix=f'model.layers.{site}.mamba.';dd=get(prefix+'D');F=get(prefix+'out_proj.weight')*get(prefix+'norm.weight')[None,:];G=F.T@F;gm0=np.zeros((48,64,64),'f8');gmc=np.zeros_like(gm0)
            with np.load(packet['basis']['path'],allow_pickle=False) as z:gy=z['GY_rec'].copy();oldgm=z['GM'].copy();maps={name:([z[f'{name}_V_{h}'].astype('f4').astype('f8') for h in range(48)],[z[f'{name}_W_{h}'].astype('f4').astype('f8') for h in range(48)]) for name in b['parent']['layouts']}
            for rr in [q for q in r['records'] if q['site']==site]:
                stage=f'{site}/{rr["id"]}';guard();rec=captures[rr['id']];n=rec['history'];f=rec['sites'][site]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);g=gate*np.exp(-np.logaddexp(0.,-gate));u=y*g;rad=np.sqrt(np.mean(u*u,axis=1,keepdims=True)+eps);o=np.dot(u/rad,F.T);o2=np.sum(o*o,axis=1);vnorm=np.dot(o,F);a0=(g/rad).reshape(n,48,64);b0=(u*g/(3072*rad**2)).reshape(n,48,64);c0=a0*vnorm.reshape(n,48,64)
                assert (rr['split'],rr['domain'],rr['history'])==(rec['split'],rec['domain'],n)
                if rec['split']=='FIT':
                    for h in range(48):
                        sl=slice(h*64,(h+1)*64);q=a0[:,h];p=b0[:,h];c=c0[:,h];base=G[sl,sl]*(q.T@q)/(24*n);cross=c.T@p/(24*n);gm0[h]+=base;gmc[h]+=base-cross-cross.T+(p.T@(p*o2[:,None]))/(24*n)
                if rec['id']==b['control_case']:
                    for t in b['control_positions']:
                        for h in range(48):
                            sl=slice(h*64,(h+1)*64);jh=F[:,sl]*a0[t,h][None,:]-o[t,:,None]*b0[t,h][None,:];direct=jh.T@jh;formula=G[sl,sl]*np.outer(a0[t,h],a0[t,h])-np.outer(c0[t,h],b0[t,h])-np.outer(b0[t,h],c0[t,h])+o2[t]*np.outer(b0[t,h],b0[t,h]);err=rel(formula,direct);assert err<=b['numeric']['Gram_relative'];control=next(c for c in r['controls'] if c['kind']=='explicit_head_Gram' and c['site']==site and c['position']==t and c['head']==h);control_scalar(control['relative'],err);control_count+=1
                metrics={}
                with np.load(rr['output']['path'],allow_pickle=False) as saved,np.load(priorrows[(site,rec['id'])]['output']['path'],allow_pickle=False) as oldsaved:
                    assert len(saved.files)==12
                    for name in b['parent']['layouts']:
                        vv,ww=maps[name];yr=y.reshape(n,48,64)-dd[None,:,None]*x;e=np.empty_like(yr);ae=np.empty_like(yr);lat=oldsaved[name+'__latent'].astype('f8');px=oldsaved[name+'__projected_input'].astype('f8');offs=packet['variants'][name]['offsets']
                        for h in range(48):sl=slice(offs[h],offs[h+1]);e[:,h]=(yr[:,h]@ww[h])@vv[h].T-yr[:,h];ae[:,h]=(lat[:,sl]-dd[h]*px[:,sl])@vv[h].T-yr[:,h]
                        e=e.reshape(n,3072);ae=ae.reshape(n,3072);du=g*e;alpha=du/rad;rhoh=np.sum((u*du).reshape(n,48,64),axis=2)/(3072*rad**2);rho=np.sum(rhoh,axis=1,keepdims=True);nr=alpha@F.T;j=radial_tangent(u,du,rad,eps)@F.T;un=g*(y+e);rn=np.sqrt(np.mean(un*un,axis=1,keepdims=True)+eps);delta=(un/rn)@F.T-o;hj=np.empty((n,48),'f8');hn=np.empty_like(hj)
                        for h in range(48):
                            guard();sl=slice(h*64,(h+1)*64);head_n=alpha[:,sl]@F[:,sl].T;head_j=head_n-o*rhoh[:,h,None];hn[:,h]=np.sum(head_n**2,axis=1);hj[:,h]=np.sum(head_j**2,axis=1)
                        JE=np.sum(j*j,axis=1);headsum=np.sum(hj,axis=1);named=dict(source_energy=o2,finite_energy=np.sum(delta**2,axis=1),J_energy=JE,no_radial_energy=np.sum(nr**2,axis=1),remainder_energy=np.sum((delta-j)**2,axis=1),radial_energy=o2*rho[:,0]**2,head_J_sum=headsum,head_no_radial_sum=np.sum(hn,axis=1),scan_difference_y_energy=np.sum((ae-e)**2,axis=1),ideal_y_energy=np.sum(e**2,axis=1),cross_head_energy=JE-headsum);stats=np.column_stack([named[k] for k in keys])
                        for key,value in [('finite_delta',delta),('J_total',j)]:
                            actual=saved[name+'__'+key];assert actual.dtype==np.dtype('<f8') and actual.shape==(n,2048);array_check(actual,value,b['numeric']['audit_vector_relative'],'vector_relative');vectors+=actual.size
                        for key,value in [('stats',stats),('head_J_energy',hj),('head_no_radial_energy',hn),('head_rho',rhoh)]:
                            actual=saved[name+'__'+key];assert actual.dtype==np.dtype('<f8') and actual.shape==value.shape;array_check(actual,value,b['numeric']['audit_scalar_relative'],'scalar_relative');stats_words+=actual.size if key=='stats' else 0;heads_words+=actual.size if key!='stats' else 0
                        metrics[name]=dict(finite_relative_RMS=float(np.sqrt(np.sum(named['finite_energy'])/max(np.sum(o2),1e-30))),remainder_relative=float(np.sqrt(np.sum(named['remainder_energy'])/max(np.sum(named['finite_energy']),1e-30))),radial_relative=float(np.sqrt(np.sum(named['radial_energy'])/max(np.sum(named['no_radial_energy']),1e-30))),paired_head_J_energy=float(np.mean(headsum)),paired_no_radial_energy=float(np.mean(named['head_no_radial_sum'])),full_J_energy=float(np.mean(JE)),cross_head_fraction=float(np.sum(named['cross_head_energy'])/max(np.sum(headsum),1e-30)),scan_difference_relative=float(np.sqrt(np.sum(named['scan_difference_y_energy'])/max(np.sum(named['ideal_y_energy']),1e-30))))
                        for k,value in metrics[name].items():scalar_check(rr['metrics'][name][k],value)
                        if rec['id']==b['control_case']:
                            pos=b['control_positions'];yp=y[pos];gp=g[pos];ep=e[pos];dp=ep*np.linalg.norm(yp,axis=1,keepdims=True)/np.maximum(np.linalg.norm(ep,axis=1,keepdims=True),1e-30);rp=rad[pos];up=u[pos];jd=radial_tangent(up,gp*dp,rp,eps)@F.T
                            for step,tol in zip(b['FD_steps'],b['FD_relative_tolerances']):
                                plus=(yp+step*dp)*gp;minus=(yp-step*dp)*gp;fd=((plus/np.sqrt(np.mean(plus**2,axis=1,keepdims=True)+eps))@F.T-(minus/np.sqrt(np.mean(minus**2,axis=1,keepdims=True)+eps))@F.T)/(2*step);err=rel(fd,jd);assert err<=tol;control=next(c for c in r['controls'] if c['kind']=='finite_difference' and c['site']==site and c['layout']==name and c['step']==step);assert control['passed'] and control['relative']<=tol;control_scalar(control['relative'],err);control_count+=1
                rows.append(dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=n,metrics=metrics));print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True)
            with np.load(gpkt['file']['path'],allow_pickle=False) as saved:
                array_check(saved['GM_no_radial'],gm0,b['numeric']['audit_Gram_relative'],'Gram_relative');array_check(saved['GM_corrected'],gmc,b['numeric']['audit_Gram_relative'],'Gram_relative')
            eigen=np.linalg.eigvalsh((gmc+gmc.transpose(0,2,1))/2);psd=float(np.max(-np.min(eigen,axis=1)/np.maximum(np.max(np.abs(eigen),axis=1),1e-30)));assert psd<=b['numeric']['PSD_relative'];scalar_check(gpkt['PSD_relative_negative'],psd);scalar_check(gpkt['old_GM_relative'],rel(gm0,oldgm));proxies[str(site)]={}
            for name in b['parent']['layouts']:
                vv,ww=maps[name];values=dict(no_radial_trace=0.,corrected_trace=0.,old_trace=0.)
                for h in range(48):
                    R=np.eye(64)-vv[h]@ww[h].T;cov=R@gy[h]@R.T
                    values['no_radial_trace']+=float(np.trace(cov@gm0[h]));values['corrected_trace']+=float(np.trace(cov@gmc[h]));values['old_trace']+=float(np.trace(cov@oldgm[h]))
                for key,value in values.items():scalar_check(r['proxies'][str(site)][name][key],value)
                proxies[str(site)][name]=values
        stream.close();stream=None;assert len(rows)==144 and control_count==len(r['controls'])==450
        for site in b['parent']['sites']:
            sk=str(site)
            for split in ['FIT','DEV']:
                sr=[q for q in rows if q['site']==site and q['split']==split];assert len(sr)==24
                for name in b['parent']['layouts']:
                    for key in sr[0]['metrics'][name]:scalar_check(r['aggregates'][sk][split][name][key],float(np.mean([q['metrics'][name][key] for q in sr])))
            for name in b['parent']['layouts']:
                sr=[q for q in rows if q['site']==site and q['split']=='FIT'];g={key:float(np.mean([q['metrics'][name][key] for q in sr])) for key in sr[0]['metrics'][name]};p=proxies[sk][name];pair=g['paired_no_radial_energy']/max(p['no_radial_trace'],1e-30);radial=abs(g['paired_head_J_energy']-g['paired_no_radial_energy'])/max(g['paired_no_radial_energy'],1e-30)
                flags=dict(finite_remainder=g['remainder_relative']>b['decisions']['finite_remainder_relative'],time_dependence=(pair<1/b['decisions']['time_pair_ratio'] or pair>b['decisions']['time_pair_ratio']),cross_head=abs(g['cross_head_fraction'])>b['decisions']['cross_head_fraction'],radial=radial>b['decisions']['radial_fraction']);selected='COUPLED_FINITE_OBJECTIVE_NEXT' if any(flags[k] for k in ['finite_remainder','time_dependence','cross_head']) else ('CORRECTED_GM_FIT_NEXT' if flags['radial'] else 'OTHER_GEOMETRY_COST_STUDY');actual=r['decisions'][sk][name]
                assert flags==actual['flags'] and selected==actual['selected'];scalar_check(actual['paired_to_independent_time_ratio'],pair);scalar_check(actual['mean_radial_head_energy_effect'],radial)
        assert all(r[k]==0 for k in ['new_scans','source_forwards','optimizer_updates','native_calls']);check();guard();ad=dict(schema='COUPLED_RECURRENCE_SENSITIVITY_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],records=len(rows),full_vector_words_verified=vectors,raw_stat_words_verified=stats_words,head_energy_rho_words_verified=heads_words,derivative_Gram_controls_verified=control_count,FIT_only_Grams_and_proxies_verified=True,maximum=maxima,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='Full unrounded F64 finite outputs and independent radial/tangent Jacobian actions;explicit per-head vectors,all statistics/mean Grams/proxies/controls/decisions. No BF16 derivative/native/model/n/50 admission.')
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
