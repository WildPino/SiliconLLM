"""Full finite/Jacobian/head/time-coupling diagnosis; no new model or fit."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from native_head_gpu_preflight import gpu_metadata
START=time.monotonic();CLOCK=time.perf_counter()
STAT_KEYS=['source_energy','finite_energy','J_energy','no_radial_energy','remainder_energy','radial_energy','head_J_sum','head_no_radial_sum','scan_difference_y_energy','ideal_y_energy','cross_head_energy']


def prepare(a):
    rpath=DOC/'recurrent_output_bases_result_20261012.json';apath=DOC/'recurrent_output_bases_adjudication_20261012.json';r=json.loads(rpath.read_bytes());ad=json.loads(apath.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rpath)
    parent=json.loads(Path(r['binding']['path']).read_bytes());capture=json.loads(Path(parent['capture']['path']).read_bytes())['records'];fit=[q for q in capture if q['split']=='FIT'];control=min(fit,key=lambda q:(q['history'],q['id']))
    storage_path=DOC/'coupled_recurrence_sensitivity_storage_20261012.json';storage=json.loads(storage_path.read_text(encoding='utf-8-sig'))
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',B/'native_head_gpu_preflight.py',DOC/'COUPLED_RECURRENCE_SENSITIVITY_PROTOCOL_20261012.md',storage_path,rpath,apath,Path(r['binding']['path']),Path(parent['capture']['path']),Path(parent['parent']['source_weights']),Path(parent['parent']['source_config'])]
    for q in capture:
        for site in parent['parent']['sites']:paths += [Path(q['sites'][site]['fields'][k]['path']) for k in ['x','y','gate']]
    paths += [Path(p['basis']['path']) for p in r['selections']]+[Path(q['output']['path']) for q in r['records']]
    runtime=[e for e in parent['inputs'] if e['path'].startswith(str(Path(sys.executable).parent)) or e['path'].startswith(str(SITE.resolve())) or e['path'].startswith('C:\\Windows\\System32')]
    paths += [ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']]
    b=dict(schema='COUPLED_RECURRENCE_SENSITIVITY_BINDING_V1',python=parent['python'],gpu=parent['gpu'],versions=parent['versions'],parent=parent['parent'],capture=parent['capture'],prior=extent(rpath),prior_adjudication=extent(apath),output_directory=storage['new_output_directory'],control_case=control['id'],control_positions=[0,control['history']//2,control['history']-1],FD_steps=[1e-3,1e-4,1e-5],FD_relative_tolerances=[1e-4,1e-5,1e-6],
           numeric=dict(Gram_relative=1e-9,PSD_relative=1e-10,energy_identity_relative=1e-8,audit_vector_relative=1e-9,audit_scalar_relative=1e-8,audit_Gram_relative=1e-9),decisions=dict(radial_fraction=.25,time_pair_ratio=1.25,cross_head_fraction=.25,finite_remainder_relative=.25),stat_keys=STAT_KEYS,
           limits=dict(seconds=1800,audit_seconds=1800,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,GPU_reserved_bytes=5<<30,output_bytes=6<<30),inputs=list({e['path']:e for e in runtime+[extent(p) for p in paths]}.values()),scope='All48 source histories/sites0,12,23/two frozen recurrent384 layouts;unrounded F64 finite output and full RMS Jacobian,head/time coupling and FIT-only corrected Grams. No fitting/scan/model/native/chatbot/n/50 admission;BF16 outputs have a distinct discontinuous operator.')
    assert gpu_metadata()==b['gpu'];write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.perf_counter()-CLOCK)),flush=True)


def classify(rows,proxies,b):
    aggregates={};decisions={}
    for site in b['parent']['sites']:
        sk=str(site);aggregates[sk]={}
        for split in ['FIT','DEV']:
            sr=[q for q in rows if q['site']==site and q['split']==split];assert len(sr)==24;aggregates[sk][split]={}
            for name in b['parent']['layouts']:aggregates[sk][split][name]={k:sum(q['metrics'][name][k] for q in sr)/24 for k in sr[0]['metrics'][name]}
        decisions[sk]={}
        for name in b['parent']['layouts']:
            g=aggregates[sk]['FIT'][name];p=proxies[sk][name];pair=g['paired_no_radial_energy']/max(p['no_radial_trace'],1e-30);radial=abs(g['paired_head_J_energy']-g['paired_no_radial_energy'])/max(g['paired_no_radial_energy'],1e-30)
            flags=dict(finite_remainder=g['remainder_relative']>b['decisions']['finite_remainder_relative'],time_dependence=max(pair,1/max(pair,1e-30))>b['decisions']['time_pair_ratio'],cross_head=abs(g['cross_head_fraction'])>b['decisions']['cross_head_fraction'],radial=radial>b['decisions']['radial_fraction'])
            selected='COUPLED_FINITE_OBJECTIVE_NEXT' if any(flags[k] for k in ['finite_remainder','time_dependence','cross_head']) else ('CORRECTED_GM_FIT_NEXT' if flags['radial'] else 'OTHER_GEOMETRY_COST_STUDY')
            decisions[sk][name]=dict(paired_to_independent_time_ratio=pair,mean_radial_head_energy_effect=radial,flags=flags,selected=selected)
    return aggregates,decisions


def run(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np,psutil,torch
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and gpu_metadata()==b['gpu'];assert str(a.directory.resolve())==str(Path(b['output_directory']).resolve()) and not a.directory.exists() and not a.out.exists()
    assert dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0';assert all(os.environ.get(k)=='6' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();torch.cuda.reset_peak_memory_stats();stream=None;rows=[];packets=[];controls=[];proxies={};stage='startup'
    receipt=dict(schema='COUPLED_RECURRENCE_SENSITIVITY_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        l=b['limits'];assert time.monotonic()-START<=l['seconds'] and proc.memory_info().peak_wset<=l['OS_bytes'] and not proc.children(recursive=True);assert torch.cuda.max_memory_allocated()<=l['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=l['GPU_reserved_bytes']
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=l['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');z=z.reshape(e['shape']).astype('f8');assert np.isfinite(z).all();return torch.from_numpy(z).cuda()
    def rel(x,y):return float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp_min(1e-30))
    try:
        check();a.directory.mkdir();capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];prior=json.loads(Path(b['prior']['path']).read_bytes());priorrows={(q['site'],q['id']):q for q in prior['records']};get,stream=tensor_reader(b['parent']['source_weights'],np);eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps']
        with torch.inference_mode():
            for packet in prior['selections']:
                site=packet['site'];prefix=f'model.layers.{site}.mamba.';dd=torch.from_numpy(get(prefix+'D')).cuda();gamma=torch.from_numpy(get(prefix+'norm.weight')).cuda();F=torch.from_numpy(get(prefix+'out_proj.weight')).cuda()*gamma[None,:];G=F.T@F;ghh=torch.stack([G[h*64:(h+1)*64,h*64:(h+1)*64] for h in range(48)]);gm0=torch.zeros_like(ghh);gmc=torch.zeros_like(ghh)
                with np.load(packet['basis']['path'],allow_pickle=False) as z:gy=torch.from_numpy(z['GY_rec'].copy()).cuda();oldgm=torch.from_numpy(z['GM'].copy()).cuda();maps={name:([torch.from_numpy(z[f'{name}_V_{h}'].astype('f4').astype('f8')).cuda() for h in range(48)],[torch.from_numpy(z[f'{name}_W_{h}'].astype('f4').astype('f8')).cuda() for h in range(48)]) for name in b['parent']['layouts']}
                for rec in capture:
                    stage=f'{site}/{rec["id"]}';guard();n=rec['history'];fields=rec['sites'][site]['fields'];x=raw(fields['x']).reshape(n,48,64);y=raw(fields['y']);gate=raw(fields['gate']);g=gate*torch.sigmoid(gate);u=y*g;r=torch.sqrt(u.square().mean(-1,keepdim=True)+eps);o=(u/r)@F.T;o2=o.square().sum(-1);vnorm=o@F;a0=(g/r).reshape(n,48,64);b0=(u*g/(3072*r.square())).reshape(n,48,64);c0=a0*vnorm.reshape(n,48,64)
                    if rec['split']=='FIT':
                        ah=a0.permute(1,0,2);bh=b0.permute(1,0,2);ch=c0.permute(1,0,2);base=ghh*(ah.transpose(-2,-1)@ah)/(24*n);cross=(ch.transpose(-2,-1)@bh)/(24*n);last=((bh*o2[None,:,None]).transpose(-2,-1)@bh)/(24*n);gm0+=base;gmc+=base-cross-cross.transpose(-2,-1)+last
                    if rec['id']==b['control_case']:
                        for t in b['control_positions']:
                            for h in range(48):
                                sl=slice(h*64,(h+1)*64);jh=F[:,sl]*a0[t,h][None,:]-o[t,:,None]*b0[t,h][None,:];direct=jh.T@jh;formula=ghh[h]*torch.outer(a0[t,h],a0[t,h])-torch.outer(c0[t,h],b0[t,h])-torch.outer(b0[t,h],c0[t,h])+o2[t]*torch.outer(b0[t,h],b0[t,h]);err=rel(formula,direct);assert err<=b['numeric']['Gram_relative'];controls.append(dict(kind='explicit_head_Gram',site=site,id=rec['id'],position=t,head=h,relative=err))
                    arrays={};metrics={}
                    with np.load(priorrows[(site,rec['id'])]['output']['path'],allow_pickle=False) as saved:
                        for name in b['parent']['layouts']:
                            vv,ww=maps[name];yr=y.reshape(n,48,64)-dd[None,:,None]*x;e=torch.empty_like(yr);ae=torch.empty_like(e);lat=torch.from_numpy(saved[name+'__latent'].astype('f8')).cuda();px=torch.from_numpy(saved[name+'__projected_input'].astype('f8')).cuda();offs=packet['variants'][name]['offsets']
                            for h in range(48):sl=slice(offs[h],offs[h+1]);e[:,h]=(yr[:,h]@ww[h])@vv[h].T-yr[:,h];ae[:,h]=(lat[:,sl]-dd[h]*px[:,sl])@vv[h].T-yr[:,h]
                            e=e.reshape(n,3072);ae=ae.reshape(n,3072);du=g*e;alpha=du/r;rhoh=(u*du).reshape(n,48,64).sum(-1)/(3072*r.square());rho=rhoh.sum(-1,keepdim=True);nr=alpha@F.T;j=nr-o*rho;un=g*(y+e);rn=torch.sqrt(un.square().mean(-1,keepdim=True)+eps);delta=(un/rn)@F.T-o
                            alphah=alpha.reshape(n,48,64);hn=torch.einsum('nhc,hcd,nhd->nh',alphah,ghh,alphah);ho=(alphah*vnorm.reshape(n,48,64)).sum(-1);hj=hn-2*rhoh*ho+o2[:,None]*rhoh.square();assert float(hj.min())>=-1e-10*max(float(hn.abs().max()),1.);JE=j.square().sum(-1);headsum=hj.sum(-1);stats=torch.stack([o2,delta.square().sum(-1),JE,nr.square().sum(-1),(delta-j).square().sum(-1),o2*rho[:,0].square(),headsum,hn.sum(-1),(ae-e).square().sum(-1),e.square().sum(-1),JE-headsum],dim=1)
                            metrics[name]=dict(finite_relative_RMS=float(torch.sqrt(stats[:,1].sum()/stats[:,0].sum().clamp_min(1e-30))),remainder_relative=float(torch.sqrt(stats[:,4].sum()/stats[:,1].sum().clamp_min(1e-30))),radial_relative=float(torch.sqrt(stats[:,5].sum()/stats[:,3].sum().clamp_min(1e-30))),paired_head_J_energy=float(stats[:,6].mean()),paired_no_radial_energy=float(stats[:,7].mean()),full_J_energy=float(stats[:,2].mean()),cross_head_fraction=float(stats[:,10].sum()/stats[:,6].sum().clamp_min(1e-30)),scan_difference_relative=float(torch.sqrt(stats[:,8].sum()/stats[:,9].sum().clamp_min(1e-30))))
                            for key,val in [('finite_delta',delta),('J_total',j),('stats',stats),('head_J_energy',hj),('head_no_radial_energy',hn),('head_rho',rhoh)]:arrays[name+'__'+key]=val.cpu().numpy()
                            if rec['id']==b['control_case']:
                                pos=torch.tensor(b['control_positions'],device='cuda');yp=y[pos];gp=g[pos];ep=e[pos];dp=ep*torch.linalg.vector_norm(yp,dim=1,keepdim=True)/torch.linalg.vector_norm(ep,dim=1,keepdim=True).clamp_min(1e-30);rp=r[pos];up=u[pos];op=o[pos];jd=(gp*dp/rp)@F.T-op*((up*gp*dp).sum(-1,keepdim=True)/(3072*rp.square()))
                                for step,tol in zip(b['FD_steps'],b['FD_relative_tolerances']):
                                    plus=(yp+step*dp)*gp;minus=(yp-step*dp)*gp;fd=((plus/torch.sqrt(plus.square().mean(-1,keepdim=True)+eps))@F.T-(minus/torch.sqrt(minus.square().mean(-1,keepdim=True)+eps))@F.T)/(2*step);err=rel(fd,jd);controls.append(dict(kind='finite_difference',site=site,id=rec['id'],layout=name,step=step,relative=err,passed=err<=tol));assert err<=tol,(stage,name,step,err,tol)
                            del lat,px,e,ae,du,alpha,nr,j,un,rn,delta,stats,hj,hn,ho,rhoh;torch.cuda.empty_cache()
                    path=a.directory/f'{rec["id"]}.site{site:02d}.npz';np.savez(path,**arrays);rows.append(dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=n,metrics=metrics,output=extent(path)));print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True);del x,y,gate,g,u,r,o,o2,vnorm,a0,b0,c0,arrays;torch.cuda.empty_cache()
                eigen=torch.linalg.eigvalsh((gmc+gmc.transpose(-2,-1))/2);psd=float((-eigen.amin(-1)/eigen.abs().amax(-1).clamp_min(1e-30)).max());assert psd<=b['numeric']['PSD_relative'];proxies[str(site)]={}
                for name in b['parent']['layouts']:
                    vv,ww=maps[name];values=dict(no_radial_trace=0.,corrected_trace=0.,old_trace=0.)
                    for h in range(48):P=vv[h]@ww[h].T;R=torch.eye(64,device='cuda',dtype=torch.float64)-P;E=R@gy[h]@R.T;values['no_radial_trace']+=float((E*gm0[h].T).sum());values['corrected_trace']+=float((E*gmc[h].T).sum());values['old_trace']+=float((E*oldgm[h].T).sum())
                    proxies[str(site)][name]=values
                path=a.directory/f'site{site:02d}.FIT_Grams.npz';np.savez(path,GM_no_radial=gm0.cpu().numpy(),GM_corrected=gmc.cpu().numpy());packets.append(dict(site=site,file=extent(path),PSD_relative_negative=psd,old_GM_relative=rel(gm0,oldgm)));del F,G,ghh,gm0,gmc,gy,oldgm,maps;torch.cuda.empty_cache()
        stream.close();stream=None;check();guard();assert len(rows)==144 and len(packets)==3 and len(controls)==450;aggregates,decisions=classify(rows,proxies,b)
        result=dict(schema='COUPLED_RECURRENCE_SENSITIVITY_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,records=rows,FIT_Grams=packets,controls=controls,proxies=proxies,aggregates=aggregates,decisions=decisions,decision='DIAGNOSTIC_COMPLETE_PENDING_AUDIT',before_after_hashes_exact=True,new_scans=0,source_forwards=0,optimizer_updates=0,native_calls=0,quality_admission=False,speed_admission=False,output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),decisions=decisions)),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());
        if a.directory.exists():write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),completed_records=rows,controls=controls,FIT_Grams=packets))
        raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved());write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args();prepare(a) if a.prepare else run(a)
