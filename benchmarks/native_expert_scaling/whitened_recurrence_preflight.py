"""Exact recurrent whitening/chain rule and matched output-direction curves."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from native_head_gpu_preflight import gpu_metadata
from finite_recurrence_gradient_v2 import retract,tangent
START=time.monotonic();CLOCK=time.perf_counter()


def prepare(a):
    rp=DOC/'finite_recurrence_gradient_v3_result_20261012.json';ap=DOC/'finite_recurrence_gradient_adjudication_20261012.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp);parent=json.loads(Path(r['binding']['path']).read_bytes());capture=json.loads(Path(parent['capture']['path']).read_bytes())['records'];sp=DOC/'whitened_recurrence_storage_20261012.json';storage=json.loads(sp.read_text(encoding='utf-8-sig'))
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',B/'native_head_gpu_preflight.py',B/'finite_recurrence_gradient_v2.py',DOC/'WHITENED_RECURRENCE_PREFLIGHT_PROTOCOL_20261012.md',sp,rp,ap,Path(r['binding']['path']),Path(parent['capture']['path']),Path(parent['parent']['source_weights']),Path(parent['parent']['source_config']),Path(parent['packet']['basis']['path']),Path(r['selected']['path'])]
    for q in capture:paths += [Path(q['sites'][12]['fields'][k]['path']) for k in ['x','y','gate']]
    runtime=[e for e in parent['inputs'] if e['path'].startswith(str(Path(sys.executable).parent)) or e['path'].startswith(str(SITE.resolve())) or e['path'].startswith('C:\\Windows\\System32')]
    paths += [ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']]
    b=dict(schema='WHITENED_RECURRENCE_BINDING_V1',python=parent['python'],gpu=parent['gpu'],versions=parent['versions'],parent=parent['parent'],capture=parent['capture'],prior=extent(rp),prior_adjudication=extent(ap),selected=r['selected'],basis=parent['packet']['basis'],site=12,layout='adaptive384',output_directory=storage['new_output_directory'],eigen_floor_relative=1e-6,alphas=[0.,1e-4,3e-4,1e-3,3e-3,.01,.03,.1,.3,1.],FD_steps=[1e-5,1e-6,1e-7],FD_relative_tolerance=1e-6,
           numeric=dict(PSD_relative=1e-10,inverse_relative=1e-9,projector_relative=1e-9,output_relative=1e-9,gradient_relative=1e-8,tangent_relative=1e-8,scalar_relative=1e-8,M_condition=1e8,V_rank_ratio=1e-12,biorthogonality=1e-8),selection=dict(Armijo=1e-4,minimum_relative_improvement=1e-8,rule='Report minimum FIT loss per coordinate among feasible Armijo candidates;reference then smaller alpha breaks exact global ties;no installed update or DEV curve selection'),limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,GPU_reserved_bytes=5<<30,output_bytes=2<<30),inputs=list({e['path']:e for e in runtime+[extent(p) for p in paths]}.values()),scope='Exact FIT recurrent whitening at previously qualified selected physical point,full physical/whitened outputs and gradients,matched first-order output RMS directions/full FIT curves. No installed update/new scan/model/native/n/50 admission.')
    assert gpu_metadata()==b['gpu'];write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.perf_counter()-CLOCK)),flush=True)


def run(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np,psutil,torch
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and gpu_metadata()==b['gpu'];assert str(a.directory.resolve())==str(Path(b['output_directory']).resolve()) and not a.directory.exists() and not a.out.exists();assert dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='6' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']);proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();torch.cuda.reset_peak_memory_stats();stream=None;rows=[];curves={};stage='startup'
    receipt=dict(schema='WHITENED_RECURRENCE_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        l=b['limits'];assert time.monotonic()-START<=l['seconds'] and proc.memory_info().peak_wset<=l['OS_bytes'] and not proc.children(recursive=True);assert torch.cuda.max_memory_allocated()<=l['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=l['GPU_reserved_bytes']
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=l['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');return torch.from_numpy(z.reshape(e['shape']).astype('f8')).cuda()
    def rel(x,y):return float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp_min(1e-30))
    def save_params(name,V,W,**extra):
        p=a.directory/(name+'.npz');np.savez(p,**{f'V_{h}':v for h,v in enumerate(V)},**{f'W_{h}':w for h,w in enumerate(W)},**extra);return extent(p)
    try:
        check();a.directory.mkdir();capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];fit=[q for q in capture if q['split']=='FIT'];assert len(fit)==24;prior=json.loads(Path(b['prior']['path']).read_bytes());priorrows={q['id']:q for q in prior['ideal']};get,stream=tensor_reader(b['parent']['source_weights'],np);prefix='model.layers.12.mamba.';dd=torch.from_numpy(get(prefix+'D')).cuda();F=torch.from_numpy(get(prefix+'out_proj.weight')).cuda()*torch.from_numpy(get(prefix+'norm.weight')).cuda()[None,:];eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps']
        with np.load(b['selected']['path'],allow_pickle=False) as z:V=[z[f'V_{h}'].copy() for h in range(48)];W=[z[f'W_{h}'].copy() for h in range(48)]
        with np.load(b['basis']['path'],allow_pickle=False) as z:cov=z['GY_rec'].copy()
        S=[];Si=[];eigen=[];floor=[];matrix_checks=[]
        for h in range(48):
            ev,U=np.linalg.eigh((cov[h]+cov[h].T)/2);assert -ev.min()/max(abs(ev).max(),1e-30)<=b['numeric']['PSD_relative'];cut=b['eigen_floor_relative']*ev.max();clipped=np.maximum(ev,cut);s=(U*np.sqrt(clipped))@U.T;si=(U*(1/np.sqrt(clipped)))@U.T;err=float(np.linalg.norm(s@si-np.eye(64))/8);assert err<=b['numeric']['inverse_relative'];S.append(s);Si.append(si);eigen.append(ev);floor.append(cut);matrix_checks.append(dict(head=h,clipped_eigenvalues=int(np.sum(ev<cut)),regularized_condition=float(clipped.max()/clipped.min()),inverse_relative=err))
        barV=[si@v for si,v in zip(Si,V)];barW=[s@w for s,w in zip(S,W)];projector_error=max(float(np.linalg.norm(s@bv@bw.T@si-v@w.T)/max(np.linalg.norm(v@w.T),1e-30)) for s,si,bv,bw,v,w in zip(S,Si,barV,barW,V,W));assert projector_error<=b['numeric']['projector_relative']
        matrices=a.directory/'whitening.npz';np.savez(matrices,covariance=cov,S=np.stack(S),inverse=np.stack(Si),eigenvalues=np.stack(eigen),floors=np.asarray(floor));initial=save_params('initial',V,W,**{f'barV_{h}':v for h,v in enumerate(barV)},**{f'barW_{h}':w for h,w in enumerate(barW)})
        def case(rec,v,w,whiten=False,backward=False,direction=None):
            guard();n=rec['history'];f=rec['sites'][12]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);g=gate*torch.sigmoid(gate);R=y.reshape(n,48,64)-dd[None,:,None]*x;skip=dd[None,:,None]*x;vv=[torch.from_numpy(q).cuda() for q in v];ww=[torch.from_numpy(q).cuda() for q in w];ss=[torch.from_numpy(q).cuda() for q in S];ii=[torch.from_numpy(q).cuda() for q in Si];inputs=[R[:,h]@ii[h] if whiten else R[:,h] for h in range(48)];Z=[inputs[h]@ww[h] for h in range(48)];Y=torch.stack([(Z[h]@vv[h].T)@ss[h]+skip[:,h] if whiten else Z[h]@vv[h].T+skip[:,h] for h in range(48)],dim=1).reshape(n,3072);u=Y*g;r=torch.sqrt(u.square().mean(-1,keepdim=True)+eps);op=(u/r)@F.T;su=y*g;teacher=(su/torch.sqrt(su.square().mean(-1,keepdim=True)+eps))@F.T;den=teacher.square().sum();loss=float((op-teacher).square().sum()/den);m=dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,loss=loss,relative_RMS=loss**.5,teacher_energy=float(den))
            if direction is not None:
                dv,dw=direction;dv=[torch.from_numpy(q).cuda() for q in dv];dw=[torch.from_numpy(q).cuda() for q in dw];dy=torch.stack([(R[:,h]@dw[h])@vv[h].T+Z[h]@dv[h].T for h in range(48)],dim=1).reshape(n,3072);du=g*dy;j=(du/r-u*(u*du).sum(-1,keepdim=True)/(3072*r**3))@F.T;return m,j.cpu().numpy(),float(j.square().sum()/den)
            if backward:
                go=2*(op-teacher)/(24*den);gu=go@F;gy=g*(gu/r-u*(u*gu).sum(-1,keepdim=True)/(3072*r**3));gh=gy.reshape(n,48,64);gv=[];gw=[]
                for h in range(48):qh=gh[:,h]@ss[h].T if whiten else gh[:,h];gv.append((qh.T@Z[h]).cpu().numpy());gw.append((inputs[h].T@(qh@vv[h])).cpu().numpy())
                return m,op.cpu().numpy(),gy.cpu().numpy(),gv,gw
            return m,op.cpu().numpy()
        with torch.inference_mode():
            GV=[np.zeros_like(q) for q in V];GW=[np.zeros_like(q) for q in W];GBV=[np.zeros_like(q) for q in barV];GBW=[np.zeros_like(q) for q in barW]
            for rec in capture:
                stage='coordinate_invariance/'+rec['id'];backward=rec['split']=='FIT';p=case(rec,V,W,False,backward);q=case(rec,barV,barW,True,backward);err=rel(torch.from_numpy(q[1]),torch.from_numpy(p[1]));assert err<=b['numeric']['output_relative'];prior_delta=abs(p[0]['relative_RMS']-priorrows[rec['id']]['selected']['relative_RMS']);assert prior_delta<=b['numeric']['scalar_relative'];arr=dict(physical_output=p[1],whitened_output=q[1]);chain_errors=[]
                if backward:
                    arr['GY']=p[2];assert rel(torch.from_numpy(p[2]),torch.from_numpy(q[2]))<=b['numeric']['gradient_relative']
                    for h in range(48):
                        cv=S[h].T@p[3][h];cw=Si[h].T@p[4][h];ev=float(np.linalg.norm(cv-q[3][h])/max(np.linalg.norm(cv),1e-30));ew=float(np.linalg.norm(cw-q[4][h])/max(np.linalg.norm(cw),1e-30));assert max(ev,ew)<=b['numeric']['gradient_relative'];chain_errors.append(max(ev,ew));GV[h]+=p[3][h];GW[h]+=p[4][h];GBV[h]+=q[3][h];GBW[h]+=q[4][h];arr[f'GV_{h}']=p[3][h];arr[f'GW_{h}']=p[4][h];arr[f'barGV_{h}']=q[3][h];arr[f'barGW_{h}']=q[4][h]
                path=a.directory/(rec['id']+'.invariance.npz');np.savez(path,**arr);rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=rec['history'],physical=p[0],whitened=q[0],output_relative=err,prior_scalar_delta=prior_delta,chain_errors=chain_errors,file=extent(path)));print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.perf_counter()-CLOCK)),flush=True)
            drV,drW,tr=tangent(np,V,W,GV,GW);dbV,dbW,tb=tangent(np,barV,barW,GBV,GBW);dwV=[s@d for s,d in zip(S,dbV)];dwW=[si@d for si,d in zip(Si,dbW)];assert max(tr['tangent_relative_residuals']+tb['tangent_relative_residuals'])<=b['numeric']['tangent_relative'];directions={};norm_packets=[]
            for name,dv,dw in [('reference',drV,drW),('whitened',dwV,dwW)]:
                scores=[]
                for rec in fit:
                    m,j,score=case(rec,V,W,direction=(dv,dw));scores.append(score);path=a.directory/(rec['id']+'.'+name+'.direction.npz');np.savez(path,J_direction=j);norm_packets.append(dict(id=rec['id'],coordinate=name,normalized_output_energy=score,file=extent(path)))
                rms=float(np.sqrt(sum(scores)/24));dv=[d/rms for d in dv];dw=[d/rms for d in dw];slope=float(sum(np.sum(gv*d1)+np.sum(gw*d2) for gv,gw,d1,d2 in zip(GV,GW,dv,dw)));assert slope<0;path=save_params(name+'_direction',dv,dw);directions[name]=dict(parameters=path,raw_output_RMS=rms,normalized_slope=slope,tangent=tr if name=='reference' else tb);curves[name]=dict(FD=[],points=[])
                for step in b['FD_steps']:
                    vp,wp,cp,okp=retract(np,V,W,dv,dw,step,b['numeric']);vm,wm,cm,okm=retract(np,V,W,dv,dw,-step,b['numeric']);assert okp and okm;plus=[case(rec,vp,wp)[0] for rec in fit];minus=[case(rec,vm,wm)[0] for rec in fit];lp=sum(q['loss'] for q in plus)/24;lm=sum(q['loss'] for q in minus)/24;fd=(lp-lm)/(2*step);err=abs(fd-slope)/abs(slope);assert err<=b['FD_relative_tolerance'];curves[name]['FD'].append(dict(step=step,plus_loss=lp,minus_loss=lm,derivative=fd,relative=err,plus_cases=plus,minus_cases=minus,plus_parameters=save_params(name+'_FD_plus_'+str(step),vp,wp),minus_parameters=save_params(name+'_FD_minus_'+str(step),vm,wm)))
                L0=sum(q['physical']['loss'] for q in rows if q['split']=='FIT')/24
                for alpha in b['alphas']:
                    vp,wp,checks,ok=retract(np,V,W,dv,dw,alpha,b['numeric']);p=dict(alpha=alpha,feasible=ok,checks=checks,parameters=save_params(name+'_point_'+str(alpha),vp,wp))
                    if ok:
                        cases=[case(rec,vp,wp)[0] for rec in fit];loss=sum(q['loss'] for q in cases)/24;p.update(FIT_cases=cases,FIT_loss=loss,Armijo=bool(alpha>0 and loss<=L0+b['selection']['Armijo']*alpha*slope),minimum_improvement=bool(loss<L0*(1-b['selection']['minimum_relative_improvement'])))
                    curves[name]['points'].append(p);print(json.dumps(dict(stage='curve',coordinate=name,alpha=alpha,FIT_loss=p.get('FIT_loss'),seconds=time.perf_counter()-CLOCK)),flush=True)
                accepted=[q for q in curves[name]['points'] if q.get('Armijo') and q.get('minimum_improvement')];curves[name]['best_FIT_point']=min(accepted,key=lambda q:(q['FIT_loss'],q['alpha'])) if accepted else curves[name]['points'][0]
            gradient=save_params('physical_gradient',GV,GW,**{f'barGV_{h}':q for h,q in enumerate(GBV)},**{f'barGW_{h}':q for h,q in enumerate(GBW)})
        stream.close();stream=None;check();guard();assert len(rows)==48;best=min([(q['best_FIT_point']['FIT_loss'],0 if name=='reference' else 1,q['best_FIT_point']['alpha'],name) for name,q in curves.items()]);result=dict(schema='WHITENED_RECURRENCE_PREFLIGHT_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,initial=initial,whitening=extent(matrices),matrix_checks=matrix_checks,projector_relative=projector_error,gradient=gradient,records=rows,direction_norm_packets=norm_packets,directions=directions,curves=curves,recommended_curve_coordinate=best[3],recommended_curve_alpha=best[2],FIT_baseline=L0,FIT_recommended_loss=best[0],decision='COORDINATE_AND_MATCHED_DIRECTION_PREFLIGHT_COMPLETE_PENDING_AUDIT',before_after_hashes_exact=True,optimizer_updates=0,new_scans=0,source_forwards=0,native_calls=0,quality_admission=False,speed_admission=False,output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(FIT_baseline=L0,best=[(name,q['best_FIT_point']['alpha'],q['best_FIT_point']['FIT_loss']) for name,q in curves.items()],result=extent(a.out))),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());
        if a.directory.exists():write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),records=rows,curves=curves))
        raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved());write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args();prepare(a) if a.prepare else run(a)
