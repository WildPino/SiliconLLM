"""Independent full coordinate/autograd/tangent/output-direction/curve audit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from finite_recurrence_gradient_v2 import retract
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil,torch
    re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());terminal=json.loads(a.terminal.read_bytes());assert terminal['exit_code']==0 and terminal['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);stream=None;stage='startup';maximum={};outputs=0;gradients=0;direction_words=0;parameter_words=0
    receipt=dict(schema='WHITENED_RECURRENCE_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);sources=[extent(Path(__file__)),extent(B/'falcon_auxiliary_s6_transport.py'),extent(B/'finite_recurrence_gradient_v2.py'),extent(DOC/'WHITENED_RECURRENCE_AUDIT_PROTOCOL_20261012.md')]
    files=[r['initial'],r['whitening'],r['gradient']]+[q['file'] for q in r['records']+r['direction_norm_packets']]+[q['parameters'] for q in r['directions'].values()]+[q['parameters'] for curve in r['curves'].values() for q in curve['points']]+[q[k] for curve in r['curves'].values() for q in curve['FD'] for k in ['plus_parameters','minus_parameters']]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[r['binding'],re,te]+sources+files:guard();assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');return z.reshape(e['shape']).astype('f8')
    def params(z):return [z[f'V_{h}'] for h in range(48)],[z[f'W_{h}'] for h in range(48)]
    def array_check(x,y,tol,key):
        assert np.isfinite(x).all() and np.isfinite(y).all();err=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30));maximum[key]=max(maximum.get(key,0.),err);assert err<=tol,(stage,key,err)
    def scalar(x,y):
        err=float(abs(x-y)/max(abs(y),1.));maximum['scalar']=max(maximum.get('scalar',0.),err);assert err<=b['numeric']['scalar_relative'],(stage,x,y,err)
    def bit(x,y):assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),stage
    try:
        check();get,stream=tensor_reader(b['parent']['source_weights'],np);prefix='model.layers.12.mamba.';dd=get(prefix+'D');F=get(prefix+'out_proj.weight')*get(prefix+'norm.weight')[None,:];ft=torch.from_numpy(F);eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps'];capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];captures={q['id']:q for q in capture};fit=[q for q in capture if q['split']=='FIT'];assert len(fit)==24 and len(r['records'])==48 and {q['id'] for q in r['records']}==set(captures);initial=load(r['initial']);V,W=params(initial);barV=[initial[f'barV_{h}'] for h in range(48)];barW=[initial[f'barW_{h}'] for h in range(48)];matrix=load(r['whitening']);S=matrix['S'];Si=matrix['inverse'];prior_basis=load(b['basis']);assert np.array_equal(matrix['covariance'],prior_basis['GY_rec'])
        for h in range(48):
            cov=matrix['covariance'][h];ev,U=np.linalg.eigh((cov+cov.T)/2);assert -ev.min()/max(abs(ev).max(),1e-30)<=b['numeric']['PSD_relative'];floor=b['eigen_floor_relative']*ev.max();clipped=np.maximum(ev,floor);s=(U*np.sqrt(clipped))@U.T;si=(U/np.sqrt(clipped))@U.T;array_check(S[h],s,b['numeric']['inverse_relative'],'S');array_check(Si[h],si,b['numeric']['inverse_relative'],'inverse');array_check(S[h]@Si[h],np.eye(64),b['numeric']['inverse_relative'],'inverse_identity');array_check(barV[h],Si[h]@V[h],b['numeric']['projector_relative'],'barV');array_check(barW[h],S[h]@W[h],b['numeric']['projector_relative'],'barW');array_check(S[h]@barV[h]@barW[h].T@Si[h],V[h]@W[h].T,b['numeric']['projector_relative'],'physical_projector');assert np.max(np.abs(barW[h].T@barV[h]-np.eye(barV[h].shape[1])))<=b['numeric']['biorthogonality']
        def forward(rec,v,w,white=False,backward=False):
            guard();n=rec['history'];f=rec['sites'][12]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);g=gate*np.exp(-np.logaddexp(0.,-gate));R=y.reshape(n,48,64)-dd[None,:,None]*x;skip=dd[None,:,None]*x;vv=[torch.from_numpy(q.copy()).requires_grad_(backward) for q in v];ww=[torch.from_numpy(q.copy()).requires_grad_(backward) for q in w];pieces=[]
            for h in range(48):
                rr=torch.from_numpy(R[:,h]);rr=rr@torch.from_numpy(Si[h]) if white else rr;yh=(rr@ww[h])@vv[h].T;yh=yh@torch.from_numpy(S[h]) if white else yh;pieces.append(yh+torch.from_numpy(skip[:,h]))
            Y=torch.cat(pieces,dim=1)
            if backward:Y.retain_grad()
            gt=torch.from_numpy(g);u=Y*gt;rad=(u.square().sum(1,keepdim=True)/3072+eps).sqrt();op=(u/rad)@ft.T;su=torch.from_numpy(y)*gt;teacher=(su/(su.square().sum(1,keepdim=True)/3072+eps).sqrt())@ft.T;den=teacher.square().sum();loss=(op-teacher).square().sum()/den;m=dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,loss=float(loss.detach()),relative_RMS=float(loss.detach().sqrt()),teacher_energy=float(den));out=op.detach().numpy().copy()
            if backward:
                (loss/24).backward();return m,out,Y.grad.numpy().copy(),[q.grad.numpy().copy() for q in vv],[q.grad.numpy().copy() for q in ww]
            return m,out
        GPV=[np.zeros_like(v) for v in V];GPW=[np.zeros_like(w) for w in W];GBV=[np.zeros_like(v) for v in barV];GBW=[np.zeros_like(w) for w in barW];records=[]
        for row in r['records']:
            stage='coordinate_autograd/'+row['id'];rec=captures[row['id']];back=rec['split']=='FIT';p=forward(rec,V,W,False,back);q=forward(rec,barV,barW,True,back);z=load(row['file']);array_check(z['physical_output'],p[1],b['numeric']['output_relative'],'physical_output');array_check(z['whitened_output'],q[1],b['numeric']['output_relative'],'white_output');array_check(q[1],p[1],b['numeric']['output_relative'],'output_invariance');outputs+=p[1].size+q[1].size
            for name,m in [('physical',p[0]),('whitened',q[0])]:
                for key in ['loss','relative_RMS','teacher_energy']:scalar(row[name][key],m[key])
            if back:
                array_check(z['GY'],p[2],b['numeric']['gradient_relative'],'GY');array_check(q[2],p[2],b['numeric']['gradient_relative'],'GY_invariance');gradients+=p[2].size
                for h in range(48):
                    for key,arr in [('GV',p[3][h]),('GW',p[4][h]),('barGV',q[3][h]),('barGW',q[4][h])]:array_check(z[f'{key}_{h}'],arr,b['numeric']['gradient_relative'],key)
                    array_check(q[3][h],S[h].T@p[3][h],b['numeric']['gradient_relative'],'chainV');array_check(q[4][h],Si[h].T@p[4][h],b['numeric']['gradient_relative'],'chainW');GPV[h]+=p[3][h];GPW[h]+=p[4][h];GBV[h]+=q[3][h];GBW[h]+=q[4][h]
            records.append(p[0]);print(json.dumps(dict(stage=stage,complete=len(records),seconds=time.perf_counter()-CLOCK)),flush=True)
        totals=load(r['gradient'])
        for h in range(48):
            for key,arr in [('V',GPV[h]),('W',GPW[h]),('barGV',GBV[h]),('barGW',GBW[h])]:array_check(totals[f'{key}_{h}'],arr,b['numeric']['gradient_relative'],'total_'+key)
        def independent_tangent(v,w,gv,gw):
            pv=[];pw=[]
            for vh,wh,g1,g2 in zip(v,w,gv,gw):
                A=wh.T@wh;Bv=vh.T@vh;C=g2.T@vh+wh.T@g1;rank=A.shape[0];K=np.kron(np.eye(rank),A)+np.kron(Bv.T,np.eye(rank));lam=np.linalg.solve(K,C.reshape(-1,order='F')).reshape(C.shape,order='F');v1=g1-wh@lam;w1=g2-vh@lam.T;assert np.linalg.norm(w1.T@vh+wh.T@v1)/max(np.linalg.norm(C),1e-30)<=b['numeric']['tangent_relative'];pv.append(v1);pw.append(w1)
            norm=np.sqrt(sum(np.sum(x*x) for x in pv+pw));theta=np.sqrt(sum(np.sum(x*x) for x in v+w));return [-theta/norm*x for x in pv],[-theta/norm*x for x in pw]
        ref=independent_tangent(V,W,GPV,GPW);bar=independent_tangent(barV,barW,GBV,GBW);raw_directions=dict(reference=ref,whitened=([s@d for s,d in zip(S,bar[0])],[si@d for si,d in zip(Si,bar[1])]))
        packets={(p['coordinate'],p['id']):p for p in r['direction_norm_packets']};L0=sum(q['loss'] for q in records if q['split']=='FIT')/24;scalar(r['FIT_baseline'],L0);best=[]
        for name,(dv,dw) in raw_directions.items():
            energies=[]
            for rec in fit:
                stage='direction_norm/'+name+'/'+rec['id'];guard();n=rec['history'];f=rec['sites'][12]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);g=gate*np.exp(-np.logaddexp(0.,-gate));R=y.reshape(n,48,64)-dd[None,:,None]*x;Y=np.empty_like(R);dY=np.empty_like(R)
                for h in range(48):Y[:,h]=(R[:,h]@W[h])@V[h].T+dd[h]*x[:,h];dY[:,h]=(R[:,h]@dw[h])@V[h].T+(R[:,h]@W[h])@dv[h].T
                u=Y.reshape(n,3072)*g;du=dY.reshape(n,3072)*g;rad=np.sqrt(np.mean(u*u,axis=1,keepdims=True)+eps);axis=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-300);radial=axis*np.sum(axis*du,axis=1,keepdims=True);j=((du-radial)/rad+radial*eps/rad**3)@F.T;su=y*g;teacher=(su/np.sqrt(np.mean(su*su,axis=1,keepdims=True)+eps))@F.T;score=float(np.sum(j*j)/np.sum(teacher*teacher));p=packets[(name,rec['id'])];z=load(p['file']);array_check(z['J_direction'],j,b['numeric']['output_relative'],'raw_J_direction');scalar(p['normalized_output_energy'],score);energies.append(score);direction_words+=j.size
            rms=float(np.sqrt(np.mean(energies)));dV=[q/rms for q in dv];dW=[q/rms for q in dw];direction=load(r['directions'][name]['parameters']);storedDV,storedDW=params(direction)
            for h in range(48):array_check(storedDV[h],dV[h],b['numeric']['gradient_relative'],'normalized_directionV');array_check(storedDW[h],dW[h],b['numeric']['gradient_relative'],'normalized_directionW')
            slope=float(sum(np.sum(g1*d1)+np.sum(g2*d2) for g1,g2,d1,d2 in zip(GPV,GPW,storedDV,storedDW)));assert slope<0;scalar(r['directions'][name]['raw_output_RMS'],rms);scalar(r['directions'][name]['normalized_slope'],slope)
            def replay(e,alpha):
                nonlocal parameter_words
                z=load(e);v,w=params(z);vr,wr,checks,ok=retract(np,V,W,storedDV,storedDW,alpha,b['numeric'])
                for h in range(48):bit(v[h],vr[h]);bit(w[h],wr[h]);parameter_words+=v[h].size+w[h].size
                return v,w,ok
            for control in r['curves'][name]['FD']:
                stage='FD/'+name+'/'+str(control['step']);vp,wp,okp=replay(control['plus_parameters'],control['step']);vm,wm,okm=replay(control['minus_parameters'],-control['step']);assert okp and okm
                with torch.inference_mode():plus=[forward(rec,vp,wp)[0] for rec in fit];minus=[forward(rec,vm,wm)[0] for rec in fit]
                lp=sum(q['loss'] for q in plus)/24;lm=sum(q['loss'] for q in minus)/24;fd=(lp-lm)/(2*control['step']);assert abs(fd-slope)/abs(slope)<=b['FD_relative_tolerance'];scalar(control['plus_loss'],lp);scalar(control['minus_loss'],lm);scalar(control['derivative'],fd)
                for actual,expected in zip(control['plus_cases']+control['minus_cases'],plus+minus):
                    assert actual['id']==expected['id']
                    for key in ['loss','relative_RMS','teacher_energy']:scalar(actual[key],expected[key])
            accepted=[]
            for point in r['curves'][name]['points']:
                stage='curve/'+name+'/'+str(point['alpha']);v,w,ok=replay(point['parameters'],point['alpha']);assert ok==point['feasible']
                if ok:
                    with torch.inference_mode():cases=[forward(rec,v,w)[0] for rec in fit]
                    loss=sum(q['loss'] for q in cases)/24;scalar(point['FIT_loss'],loss);armijo=bool(point['alpha']>0 and loss<=L0+b['selection']['Armijo']*point['alpha']*slope);improved=bool(loss<L0*(1-b['selection']['minimum_relative_improvement']));assert armijo==point['Armijo'] and improved==point['minimum_improvement']
                    for actual,expected in zip(point['FIT_cases'],cases):
                        assert actual['id']==expected['id']
                        for key in ['loss','relative_RMS','teacher_energy']:scalar(actual[key],expected[key])
                    if armijo and improved:accepted.append((point['FIT_loss'],point['alpha']))
            value,alpha=min(accepted) if accepted else (r['curves'][name]['points'][0]['FIT_loss'],0.);chosen=r['curves'][name]['best_FIT_point'];assert value==chosen['FIT_loss'] and alpha==chosen['alpha'];best.append((value,0 if name=='reference' else 1,alpha,name));print(json.dumps(dict(stage='coordinate_complete',coordinate=name,best_loss=value,best_alpha=alpha,seconds=time.perf_counter()-CLOCK)),flush=True)
        recommended=min(best);assert recommended[3]==r['recommended_curve_coordinate'] and recommended[2]==r['recommended_curve_alpha'];scalar(r['FIT_recommended_loss'],recommended[0]);assert all(r[k]==0 for k in ['optimizer_updates','new_scans','source_forwards','native_calls']);check();guard();ad=dict(schema='WHITENED_RECURRENCE_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],coordinate_case_count=48,dual_autograd_FIT_cases=24,output_words_verified=outputs,GY_words_verified=gradients,raw_J_direction_words_verified=direction_words,parameter_replay_words_verified=parameter_words,maximum=maximum,before_after_hashes_exact=True,recommended_coordinate=recommended[3],recommended_alpha=recommended[2],decision='INDEPENDENT_STORED_PASS',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='Full physical/whitened autograd,SPD/inverse/projector/output invariance,independent tangent/radial direction normalization/full FIT curves/replay/recommendation. No installed update,new scan,model/native/n/50 admission.')
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
