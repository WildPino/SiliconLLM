"""FIT recurrent-only covariance, retained GM, actual384 scans/full source skip."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from native_head_gpu_preflight import gpu_metadata
from recurrent_only_channel_skip_v3 import aggregate
START=time.monotonic();CLOCK=time.perf_counter()


def prepare(a):
    import numpy as np
    parent_path=DOC/'original_falcon_balanced_heads_binding_20261009.json';parent=json.loads(parent_path.read_bytes())
    old_path=DOC/'original_falcon_balanced_heads_result_20261009.json';old=json.loads(old_path.read_bytes());capture=json.loads(Path(parent['capture']).read_bytes())
    prior_path=DOC/'recurrent_only_channel_skip_v3_result_20261012.json';ad_path=DOC/'recurrent_only_channel_skip_adjudication_20261012.json';ad=json.loads(ad_path.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(prior_path)
    gb=json.loads((DOC/'native_head_gpu_preflight_binding_20261012.json').read_bytes())
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',B/'native_head_gpu_preflight.py',B/'recurrent_only_channel_skip_v3.py',
           B/'original_falcon_mixed_history.py',B/'original_falcon_channel_transport.py',B/'original_falcon_whole_recovery.py',B/'chatbot_falcon_usability_launch.py',B/'chatbot_falcon_usability.py',
           SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py',DOC/'RECURRENT_OUTPUT_BASES_PROTOCOL_20261012.md',parent_path,old_path,prior_path,ad_path,
           Path(parent['capture']),Path(parent['source_weights']),Path(parent['source_config'])]
    for rec in capture['records']:
        for site in parent['sites']:paths += [Path(f['path']) for f in rec['sites'][site]['fields'].values()]
    paths += [Path(s['basis']['path']) for s in old['selections']]
    paths += [Path(m['latent']['path']) for r in old['records'] for m in r['measurements'].values()]
    runtime=[e for e in gb['inputs'] if e['path'].startswith(str(Path(sys.executable).parent)) or e['path'].startswith(str(SITE.resolve())) or e['path'].startswith('C:\\Windows\\System32')]
    for path in [ROOT/'benchmarks/phase60/engine.c',ROOT/'benchmarks/donor_adaptation/configs/_manifest.json',ROOT/'benchmarks/donor_adaptation/density/build_document_holdout.py',ROOT/'docs/research/RESEARCH_INDEX.md']:paths.append(path)
    inputs=list({e['path']:e for e in runtime+[extent(p) for p in paths]}.values())
    b=dict(schema='RECURRENT_OUTPUT_BASES_BINDING_V1',python=gb['python'],gpu=gb['gpu'],versions=gb['versions'],parent=dict(source_weights=parent['source_weights'],source_config=parent['source_config'],sites=parent['sites'],layouts=parent['layouts'],modes=parent['modes']),
           capture=extent(parent['capture']),old=extent(old_path),prior=extent(prior_path),prior_adjudication=extent(ad_path),
           numeric=dict(PSD_relative_negative=1e-10,minimum_retained_singular=1e-12,biorthogonality=1e-8,weighted_kernel_relative=1e-8,covariance_relative=1e-10,projected_scan_relative=1e-4,scalar_sequence_relative=1e-4,projected_input_relative=1e-6,old_metric_absolute=1e-4,audit_output_relative=1e-4,audit_metric_absolute=1e-4,audit_energy_relative=1e-8,algebra_relative=1e-10),
           criteria={k:parent['criteria'][k] for k in ['DEV_output_mean','DEV_case_worst','DEV_domain_mean_worst','DEV_centered_mean']},
           limits=dict(seconds=1800,audit_seconds=1800,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,GPU_reserved_bytes=5<<30,output_bytes=3<<30),inputs=inputs,
           scope='FIT-only recurrent covariance y-D*x,unchanged frozen GM,two384 per-head layouts,all48 source histories/three sites,actual new F32 SSD scans. Full source generators/gates/conv/state256 retained;controlled F64/BF16 output. No native/chatbot/n/50 or old-F32 precision admission.')
    assert gpu_metadata()==b['gpu'];write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.perf_counter()-CLOCK)),flush=True)


def run(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np,psutil,torch
    from original_falcon_mixed_history import scan
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and gpu_metadata()==b['gpu']
    assert not a.out.exists() and not a.directory.exists();assert dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='6' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    receipt=dict(schema='RECURRENT_OUTPUT_BASES_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);stream=None;rows=[];packets=[];stage='startup'
    def guard():
        l=b['limits'];assert time.monotonic()-START<=l['seconds'] and proc.memory_info().peak_wset<=l['OS_bytes'] and not proc.children(recursive=True)
        assert torch.cuda.max_memory_allocated()<=l['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=l['GPU_reserved_bytes']
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=l['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');z=z.reshape(e['shape']);assert np.isfinite(z).all();return torch.from_numpy(z).cuda()
    def error(x,y):return float(torch.linalg.vector_norm(x.double()-y.double())/torch.linalg.vector_norm(y.double()).clamp_min(1e-12))
    def metrics(op,ref,pos):return dict(output_relative_RMS=error(op,ref),centered_output_relative_RMS=error(op.double()-op.double().mean(0),ref.double()-ref.double().mean(0)),label_output_relative_RMS=error(op[pos],ref[pos]))
    def root(g):
        ev,u=torch.linalg.eigh((g+g.transpose(-2,-1))/2);assert float((-ev.amin(-1)/ev.abs().amax(-1).clamp_min(1e-300)).max())<=b['numeric']['PSD_relative_negative']
        return (u*ev.clamp_min(0).sqrt()[:,None,:])@u.transpose(-2,-1),ev
    try:
        check();a.directory.mkdir();capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];fit=[r for r in capture if r['split']=='FIT'];assert len(fit)==24
        old=json.loads(Path(b['old']['path']).read_bytes());prior=json.loads(Path(b['prior']['path']).read_bytes());oldrows={(r['site'],r['id']):r for r in old['records']};priorrows={(r['site'],r['id']):r for r in prior['records']}
        get,stream=tensor_reader(b['parent']['source_weights'],np);eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps']
        with torch.inference_mode():
            for oldpacket in old['selections']:
                site=oldpacket['site'];prefix=f'model.layers.{site}.mamba.';dd=torch.from_numpy(get(prefix+'D')).float().cuda();alog=torch.from_numpy(get(prefix+'A_log')).float().cuda();gamma=torch.from_numpy(get(prefix+'norm.weight')).cuda();ow=torch.from_numpy(get(prefix+'out_proj.weight')).cuda()
                with np.load(oldpacket['basis']['path'],allow_pickle=False) as z:gm=torch.from_numpy(z['GM'].copy());oldmaps={name:([torch.from_numpy(z[f'{name}_V_{h}'].astype('<f4')).cuda() for h in range(48)],[torch.from_numpy(z[f'{name}_W_{h}'].astype('<f4')).cuda() for h in range(48)]) for name in b['parent']['layouts']}
                gy=torch.zeros((48,64,64),dtype=torch.float64,device='cuda')
                for rec in fit:
                    stage=f'FIT_covariance/{site}/{rec["id"]}';guard();f=rec['sites'][site]['fields'];x=raw(f['x']).double().reshape(-1,48,64);y=raw(f['y']).double().reshape(-1,48,64);yr=y-dd.double()[None,:,None]*x;z=yr.permute(1,0,2);gy+=(z.transpose(-2,-1)@z)/(24*rec['history'])
                gy=gy.cpu();sy,ey=root(gy);sm,em=root(gm);uu,ss,qh=torch.linalg.svd(sm@sy,full_matrices=False);qq=qh.transpose(-2,-1);scores=ss.numpy()**2;choices=sorted([(scores[h,r],h,r) for h in range(48) for r in range(1,64)],key=lambda z:(-z[0],z[1],z[2]))[:336]
                adaptive=[1+sum(h==j for _,j,_ in choices) for h in range(48)];assert sum(adaptive)==384 and all({r for _,j,r in choices if j==h}==set(range(1,adaptive[h])) for h in range(48))
                arrays=dict(GY_rec=gy.numpy(),GM=gm.numpy(),sqrt_GY=sy.numpy(),sqrt_GM=sm.numpy(),GY_eigenvalues=ey.numpy(),GM_eigenvalues=em.numpy(),U=uu.numpy(),Q=qq.numpy(),singular_values=ss.numpy());variants={};maps={}
                for name,ranks in [('fixed8',[8]*48),('adaptive384',adaptive)]:
                    vv=[];ww=[];offs=[0];bios=[]
                    for h,rank in enumerate(ranks):
                        sigma=ss[h,:rank];assert float(sigma.min())>b['numeric']['minimum_retained_singular'];v=(sy[h]@qq[h,:,:rank])/sigma.sqrt();w=(sm[h]@uu[h,:,:rank])/sigma.sqrt();bio=float((w.T@v-torch.eye(rank,dtype=torch.float64)).abs().max());assert bio<=b['numeric']['biorthogonality'];bios.append(bio)
                        arrays[f'{name}_V_{h}']=v.numpy();arrays[f'{name}_W_{h}']=w.numpy();vv.append(v.float().cuda());ww.append(w.float().cuda());offs.append(offs[-1]+rank)
                    groups={rank:[h for h in range(48) if ranks[h]==rank] for rank in sorted(set(ranks))};maps[name]=dict(V=vv,W=ww,ranks=ranks,offsets=offs,groups=groups);variants[name]=dict(ranks=ranks,offsets=offs,groups=groups,biorthogonality=bios,weighted_tail_energy=float(sum((ss[h,rank:]**2).sum() for h,rank in enumerate(ranks))))
                bp=a.directory/f'site{site:02d}.recurrent_bases.npz';np.savez(bp,**arrays);packet=dict(site=site,basis=extent(bp),variants=variants,FIT_cases=24,FIT_positions=sum(r['history'] for r in fit),GM_reused=oldpacket['basis']);packets.append(packet)
                print(json.dumps(dict(stage='basis_frozen',site=site,adaptive_ranks=adaptive,seconds=time.perf_counter()-CLOCK)),flush=True)
                del gy,sy,sm,em,ey,uu,ss,qq,qh,arrays;torch.cuda.empty_cache()
                for rec in capture:
                    stage=f'evaluation/{site}/{rec["id"]}';guard();f=rec['sites'][site]['fields'];n=rec['history'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);ref=raw(f['output']);bb=raw(f['B']);cc=raw(f['C']);dt=raw(f['delta']).bfloat16();pos=torch.tensor(rec['positions'],device='cuda');sg=gate.double()*torch.sigmoid(gate.double());source_inv=torch.rsqrt((y.double()*sg).square().mean(-1,keepdim=True)+eps)
                    arms={};decomp={};controls={};outputs={};scanchecks={}
                    for name in b['parent']['layouts']:
                        m=maps[name];latent=torch.empty((n,384),device='cuda');px=torch.empty_like(latent)
                        for rank,heads in m['groups'].items():
                            w=torch.stack([m['W'][h] for h in heads]);xm=torch.einsum('thc,hcr->thr',x[:,heads],w);ym=scan(torch,code,xm,bb,cc,dt[:,heads],alog[heads],dd[heads])
                            for j,h in enumerate(heads):sl=slice(m['offsets'][h],m['offsets'][h+1]);latent[:,sl]=ym[:,j];px[:,sl]=xm[:,j]
                            del w,xm,ym;torch.cuda.empty_cache()
                        expected=torch.empty_like(latent,dtype=torch.float64);rec_expected=torch.empty_like(expected);skip=torch.empty_like(expected);yo=torch.empty((n,48,64),device='cuda',dtype=torch.float64);yn=torch.empty_like(yo);es=torch.empty_like(yo);oldyo=torch.empty_like(yo);oldyn=torch.empty_like(yo);oldlat=raw(oldrows[(site,rec['id'])]['measurements'][name]['latent']);oldv,oldw=oldmaps[name];oldoffs=oldpacket['variants'][name]['offsets']
                        for h in range(48):
                            sl=slice(m['offsets'][h],m['offsets'][h+1]);v=m['V'][h].double();w=m['W'][h].double();expected[:,sl]=y.double().reshape(n,48,64)[:,h]@w;rec_expected[:,sl]=(y.double().reshape(n,48,64)[:,h]-dd[h].double()*x[:,h].double())@w;skip[:,sl]=dd[h].double()*px[:,sl].double();yo[:,h]=latent[:,sl].double()@v.T;yn[:,h]=(latent[:,sl].double()-skip[:,sl])@v.T+dd[h].double()*x[:,h].double();es[:,h]=dd[h].double()*(px[:,sl].double()@v.T-x[:,h].double())
                            osl=slice(oldoffs[h],oldoffs[h+1]);ox=x[:,h].double()@oldw[h].double();oldyo[:,h]=oldlat[:,osl].double()@oldv[h].double().T;oldyn[:,h]=(oldlat[:,osl].double()-dd[h].double()*ox)@oldv[h].double().T+dd[h].double()*x[:,h].double()
                        pe=error(latent,expected);re=error(latent.double()-skip,rec_expected);assert pe<=b['numeric']['projected_scan_relative'] and re<=b['numeric']['projected_scan_relative'],(stage,name,pe,re);scanchecks[name]=dict(full_relative=pe,recurrent_relative=re)
                        outputs[name+'__latent']=latent.cpu().numpy().astype('<f4');outputs[name+'__projected_input']=px.cpu().numpy().astype('<f4')
                        yo=yo.reshape(n,3072);yn=yn.reshape(n,3072);es=es.reshape(n,3072);eo=yo-y.double();er=yn-y.double();identity=error(eo,er+es);assert identity<=b['numeric']['algebra_relative'];decomp[name]=dict(old_error_energy=float(eo.square().sum()),recurrent_error_energy=float(er.square().sum()),skip_error_energy=float(es.square().sum()),recurrent_skip_inner_product=float((er*es).sum()),source_y_energy=float(y.double().square().sum()),identity_relative=identity,new_y_relative_RMS=error(yn,y),old_y_relative_RMS=error(yo,y));arms[name]={};controls[name]={}
                        for arm,yr,oy in [('projected_skip',yo,oldyo),('full_skip',yn,oldyn)]:
                            arms[name][arm]={};controls[name][arm]={};ug=yr*sg;ou=oy.reshape(n,3072)*sg
                            for mode in b['parent']['modes']:
                                inv=source_inv if mode=='full_source_denominator' else torch.rsqrt(ug.square().mean(-1,keepdim=True)+eps);op=(((ug*inv*gamma).bfloat16().double())@ow.T).bfloat16();arms[name][arm][mode]=metrics(op,ref,pos);outputs[f'{name}__{arm}__{mode}']=op.cpu().view(torch.uint16).numpy().astype('<u2')
                                oi=source_inv if mode=='full_source_denominator' else torch.rsqrt(ou.square().mean(-1,keepdim=True)+eps);oldop=(((ou*oi*gamma).bfloat16().double())@ow.T).bfloat16();mm=metrics(oldop,ref,pos);prior_mm=priorrows[(site,rec['id'])]['arms'][name][arm][mode];delta=max(abs(mm[k]-prior_mm[k]) for k in mm);assert delta<=b['numeric']['old_metric_absolute'],(stage,name,arm,mode,delta);controls[name][arm][mode]=dict(metrics=mm,max_prior_delta=delta)
                        del latent,px,expected,rec_expected,skip,yo,yn,es,eo,er,oldyo,oldyn,oldlat,ug,ou,op,oldop;torch.cuda.empty_cache()
                    path=a.directory/f'{rec["id"]}.site{site:02d}.npz';np.savez(path,**outputs);rows.append(dict(id=rec['id'],site=site,split=rec['split'],domain=rec['domain'],history=n,arms=arms,decomposition=decomp,controls=controls,scan_checks=scanchecks,output=extent(path)))
                    print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True);del x,y,gate,ref,bb,cc,dt,sg,source_inv,outputs;torch.cuda.empty_cache()
        stream.close();stream=None;check();guard();assert len(rows)==144 and len(packets)==3;aggregates,domains,gates=aggregate(rows,b);pass_any=any(all(all(gates[str(s)][name]['full_skip'][mode].values()) for s in b['parent']['sites']) for name in b['parent']['layouts'] for mode in b['parent']['modes'])
        r=dict(schema='RECURRENT_OUTPUT_BASES_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,selections=packets,records=rows,aggregates=aggregates,domains=domains,gates=gates,decision='RECURRENT_BASES_LOCAL_BUDGET_PASS_PENDING_AUDIT' if pass_any else 'RECURRENT_BASES_LOCAL_BUDGET_FAIL_PENDING_AUDIT',before_after_hashes_exact=True,FIT_covariance_cases_per_site=24,basis_head_fits=144,new_scans=288,source_forwards=0,optimizer_updates=0,native_calls=0,quality_admission=False,speed_admission=False,old_precision_bridge_admitted=False,output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),scope=b['scope']);write(a.out,r);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(decision=r['decision'],result=extent(a.out))),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());
        if a.directory.exists():write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),selections=packets,completed_records=rows))
        raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved());write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args();prepare(a) if a.prepare else run(a)
