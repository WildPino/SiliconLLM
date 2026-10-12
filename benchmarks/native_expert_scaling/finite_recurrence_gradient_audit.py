"""Full CPU autograd/closed-form/Sylvester/selection/actual scan qualification."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from finite_recurrence_gradient_v2 import retract
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil,torch
    re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());terminal=json.loads(a.terminal.read_bytes());assert terminal['exit_code']==0 and terminal['result']==re and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions']
    assert not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);stream=None;stage='startup';maxima={};gradient_words=0;autograd_cases=0;output_words=0;scan_words=0;parameter_words=0
    receipt=dict(schema='FINITE_RECURRENCE_GRADIENT_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);sources=[extent(Path(__file__)),extent(B/'finite_recurrence_gradient_v2.py'),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'FINITE_RECURRENCE_GRADIENT_AUDIT_PROTOCOL_20261012.md')]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    files=[r['initial'],r['gradient'],r['selected']]+[q['file'] for q in r['gradient_cases']+r['ideal']+r['machine_records']]+[q['parameters'] for q in r['points']]+[q[k] for q in r['FD']+r.get('coarse_FD',[]) for k in ['plus_parameters','minus_parameters']]
    def check():
        for e in b['inputs']+[r['binding'],re,te]+sources+files:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');z=z.reshape(e['shape']).astype('f8');assert np.isfinite(z).all();return z
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def params(z):return [z[f'V_{h}'] for h in range(48)],[z[f'W_{h}'] for h in range(48)]
    def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def array_check(x,y,tol,key):
        assert np.isfinite(x).all() and np.isfinite(y).all();err=rel(x,y);maxima[key]=max(maxima.get(key,0.),err);assert err<=tol,(stage,key,err)
    def scalar(x,y):
        err=abs(x-y)/max(abs(y),1.);maxima['scalar']=max(maxima.get('scalar',0.),err);assert err<=b['numeric']['audit_scalar_relative'],(stage,x,y,err)
    def bits(x,y):assert x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes(),('parameter bits',stage)
    def decode(z):assert z.dtype==np.dtype('<u2');return (z.astype('<u4')<<16).view('<f4').astype('f8')
    def bf16(z):
        f=np.asarray(z,dtype='<f4');assert np.isfinite(f).all();u=f.view('<u4');return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2')
    try:
        check();get,stream=tensor_reader(b['parent']['source_weights'],np);prefix='model.layers.12.mamba.';dd=get(prefix+'D');aa=-np.exp(get(prefix+'A_log'));gamma=get(prefix+'norm.weight');ow=get(prefix+'out_proj.weight');F=ow*gamma[None,:];ft=torch.from_numpy(F);eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps'];capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];captures={q['id']:q for q in capture};fit=[q for q in capture if q['split']=='FIT'];assert len(fit)==24
        initial=load(r['initial']);V,W=params(initial);rawV=[initial[f'raw_V_{h}'] for h in range(48)];rawW=[initial[f'raw_W_{h}'] for h in range(48)];vref,wref,checks,ok=retract(np,rawV,rawW,[np.zeros_like(v) for v in rawV],[np.zeros_like(w) for w in rawW],0.,b['numeric']);assert ok
        for h in range(48):bits(V[h],vref[h]);bits(W[h],wref[h]);assert np.max(np.abs(W[h].T@V[h]-np.eye(V[h].shape[1])))<=b['numeric']['biorthogonality'];array_check(V[h]@W[h].T,rawV[h]@rawW[h].T,b['numeric']['gauge_projector_relative'],'gauge_projector')
        fault_path=Path('E:/SiliconLLM/results/native_expert_scaling/finite_recurrence_gradient_20261012/first_fault.json');fault=json.loads(fault_path.read_bytes());old_initial=load(extent(fault_path.parent/'initial_gauge.npz'))
        for key in initial:bits(initial[key],old_initial[key])
        oldrows={q['id']:q for q in fault['gradient_cases']};gradrows={q['id']:q for q in r['gradient_cases']};assert set(oldrows)==set(gradrows)=={q['id'] for q in fit}
        def forward(rec,v,w,backward=False):
            guard();n=rec['history'];f=rec['sites'][12]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);g=gate*np.exp(-np.logaddexp(0.,-gate));R=y.reshape(n,48,64)-dd[None,:,None]*x;skip=dd[None,:,None]*x;rt=torch.from_numpy(R);gt=torch.from_numpy(g);yt=torch.from_numpy(y);st=torch.from_numpy(skip);vv=[torch.from_numpy(vh.copy()).requires_grad_(backward) for vh in v];ww=[torch.from_numpy(wh.copy()).requires_grad_(backward) for wh in w];Y=torch.cat([(rt[:,h]@ww[h])@vv[h].T+st[:,h] for h in range(48)],dim=1)
            if backward:Y.retain_grad()
            U=Y*gt;rad=(U.square().sum(1,keepdim=True)/3072+eps).sqrt();op=(U/rad)@ft.T;us=yt*gt;teacher=(us/(us.square().sum(1,keepdim=True)/3072+eps).sqrt())@ft.T;den=teacher.square().sum();lc=(op-teacher).square().sum()/den;loss=float(lc.detach());result=dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,loss=loss,relative_RMS=float(np.sqrt(loss)),teacher_energy=float(den),centered_relative_RMS=float(torch.linalg.vector_norm((op-op.mean(0))-(teacher-teacher.mean(0))).detach()/torch.linalg.vector_norm(teacher-teacher.mean(0))))
            out=op.detach().numpy().copy()
            if backward:
                (lc/24).backward();gy=Y.grad.numpy().copy();gv=[q.grad.numpy().copy() for q in vv];gw=[q.grad.numpy().copy() for q in ww];gu=(2*(out-teacher.numpy())/(24*float(den)))@F;uv=U.detach().numpy();rv=rad.detach().numpy();axis=uv/np.maximum(np.linalg.norm(uv,axis=1,keepdims=True),1e-300);radial=axis*np.sum(axis*gu,axis=1,keepdims=True);closed=g*((gu-radial)/rv+radial*eps/rv**3);array_check(gy,closed,b['numeric']['audit_gradient_relative'],'RMS_autograd_vs_radial');return result,out,gy,gv,gw
            return result,out
        GV=[np.zeros_like(v) for v in V];GW=[np.zeros_like(w) for w in W];baseline=[]
        for rec in fit:
            stage='full_backward/'+rec['id'];m,op,gy,gv,gw=forward(rec,V,W,True);baseline.append(m);z=load(gradrows[rec['id']]['file']);old=load(oldrows[rec['id']]['file']);assert set(z)==set(old)
            for key in z:bits(z[key],old[key])
            array_check(z['output'],op,b['numeric']['audit_vector_relative'],'baseline_output');array_check(z['GY'],gy,b['numeric']['audit_gradient_relative'],'full_GY');gradient_words+=gy.size;autograd_cases+=1
            for h in range(48):array_check(z[f'GV_{h}'],gv[h],b['numeric']['audit_gradient_relative'],'GV');array_check(z[f'GW_{h}'],gw[h],b['numeric']['audit_gradient_relative'],'GW');GV[h]+=gv[h];GW[h]+=gw[h]
            for key in ['loss','relative_RMS','teacher_energy','centered_relative_RMS']:scalar(gradrows[rec['id']][key],m[key])
            print(json.dumps(dict(stage=stage,complete=autograd_cases,seconds=time.perf_counter()-CLOCK)),flush=True)
        grad=load(r['gradient']);storedGV,storedGW=params(grad);dV=[grad[f'dV_{h}'] for h in range(48)];dW=[grad[f'dW_{h}'] for h in range(48)];norm2=0.;projectedV=[];projectedW=[]
        for h in range(48):
            array_check(storedGV[h],GV[h],b['numeric']['audit_gradient_relative'],'total_GV');array_check(storedGW[h],GW[h],b['numeric']['audit_gradient_relative'],'total_GW');A=W[h].T@W[h];Bv=V[h].T@V[h];C=storedGW[h].T@V[h]+W[h].T@storedGV[h];rank=A.shape[0];K=np.kron(np.eye(rank),A)+np.kron(Bv.T,np.eye(rank));lam=np.linalg.solve(K,C.reshape(-1,order='F')).reshape(C.shape,order='F');pv=storedGV[h]-W[h]@lam;pw=storedGW[h]-V[h]@lam.T;projectedV.append(pv);projectedW.append(pw);norm2+=np.sum(pv*pv)+np.sum(pw*pw);assert np.linalg.norm(pw.T@V[h]+W[h].T@pv)/max(np.linalg.norm(C),1e-30)<=b['numeric']['tangent_relative']
        theta=np.sqrt(sum(np.sum(q*q) for q in V+W));scale=theta/np.sqrt(norm2);slope=sum(np.sum(gv*dv)+np.sum(gw*dw) for gv,gw,dv,dw in zip(storedGV,storedGW,dV,dW));assert slope<0;scalar(r['tangent']['slope'],slope);scalar(slope,-scale*norm2)
        for h in range(48):array_check(dV[h],-scale*projectedV[h],b['numeric']['audit_gradient_relative'],'independent_tangent_V');array_check(dW[h],-scale*projectedW[h],b['numeric']['audit_gradient_relative'],'independent_tangent_W')
        def replay(e,alpha):
            nonlocal parameter_words
            z=load(e);v,w=params(z);vr,wr,checks,ok=retract(np,V,W,dV,dW,alpha,b['numeric'])
            for h in range(48):bits(v[h],vr[h]);bits(w[h],wr[h]);parameter_words+=v[h].size+w[h].size
            if ok:
                for h in range(48):assert np.max(np.abs(w[h].T@v[h]-np.eye(v[h].shape[1])))<=b['numeric']['biorthogonality'];assert np.linalg.matrix_rank(v[h])==v[h].shape[1]
            return v,w,ok
        L0=sum(q['loss'] for q in baseline)/24;scalar(r['FIT_initial_loss'],L0)
        for control in r.get('coarse_FD',[]):
            stage='original_coarse_FD/'+str(control['step']);vp,wp,okp=replay(control['plus_parameters'],control['step']);vm,wm,okm=replay(control['minus_parameters'],-control['step']);assert okp and okm
            with torch.inference_mode():plus=[forward(rec,vp,wp)[0] for rec in fit];minus=[forward(rec,vm,wm)[0] for rec in fit]
            lp=sum(q['loss'] for q in plus)/24;lm=sum(q['loss'] for q in minus)/24;fd=(lp-lm)/(2*control['step']);error=abs(fd-slope)/abs(slope);assert bool(error<=control['tolerance'])==control['original_gate_passed'];scalar(control['plus_loss'],lp);scalar(control['minus_loss'],lm);scalar(control['derivative'],fd)
            for actual,expected in zip(control['plus_cases']+control['minus_cases'],plus+minus):
                assert actual['id']==expected['id']
                for key in ['loss','relative_RMS','teacher_energy','centered_relative_RMS']:scalar(actual[key],expected[key])
        for control,tol in zip(r['FD'],b['FD_tolerances']):
            stage='FD/'+str(control['step']);vp,wp,okp=replay(control['plus_parameters'],control['step']);vm,wm,okm=replay(control['minus_parameters'],-control['step']);assert okp and okm;plus=[];minus=[]
            with torch.inference_mode():
                for rec in fit:plus.append(forward(rec,vp,wp)[0]);minus.append(forward(rec,vm,wm)[0])
            lp=sum(q['loss'] for q in plus)/24;lm=sum(q['loss'] for q in minus)/24;fd=(lp-lm)/(2*control['step']);error=abs(fd-slope)/abs(slope);assert error<=tol;scalar(control['plus_loss'],lp);scalar(control['minus_loss'],lm);scalar(control['derivative'],fd)
            for recorded,expected in zip(control['plus_cases']+control['minus_cases'],plus+minus):
                assert recorded['id']==expected['id']
                for key in ['loss','relative_RMS','teacher_energy','centered_relative_RMS']:scalar(recorded[key],expected[key])
        candidates=[]
        for p in r['points']:
            stage='candidate/'+str(p['alpha']);v,w,ok=replay(p['parameters'],p['alpha']);assert ok==p['feasible']
            if ok:
                with torch.inference_mode():cases=[forward(rec,v,w)[0] for rec in fit]
                loss=sum(q['loss'] for q in cases)/24;scalar(p['FIT_loss'],loss)
                for actual,expected in zip(p['FIT_cases'],cases):
                    assert actual['id']==expected['id']
                    for key in ['loss','relative_RMS','teacher_energy','centered_relative_RMS']:scalar(actual[key],expected[key])
                armijo=bool(p['alpha']>0 and loss<=L0+b['selection']['Armijo']*p['alpha']*slope);improvement=bool(loss<L0*(1-b['selection']['minimum_relative_improvement']));assert armijo==p['Armijo'] and improvement==p['minimum_improvement']
                if armijo and improvement:candidates.append((p['FIT_loss'],p['alpha']))
        selected_alpha=min(candidates)[1] if candidates else 0.;assert selected_alpha==r['selected_alpha'];vs,ws,ok=replay(r['selected'],selected_alpha);assert ok
        assert {q['id'] for q in r['ideal']}==set(captures) and len(r['ideal'])==48
        with torch.inference_mode():
            for item in r['ideal']:
                stage='ideal/'+item['id'];rec=captures[item['id']];m0,o0=forward(rec,V,W);ms,os=forward(rec,vs,ws);z=load(item['file']);array_check(z['baseline_output'],o0,b['numeric']['audit_vector_relative'],'ideal_baseline');array_check(z['selected_output'],os,b['numeric']['audit_vector_relative'],'ideal_selected');output_words+=o0.size+os.size
                for name,m in [('baseline',m0),('selected',ms)]:
                    for key in ['loss','relative_RMS','teacher_energy','centered_relative_RMS']:scalar(item[name][key],m[key])
        machine=[];ranks=b['packet']['variants']['adaptive384']['ranks'];offs=b['packet']['variants']['adaptive384']['offsets'];ids=np.repeat(np.arange(48),ranks)
        for item in r['machine_records']:
            stage='actual_scan/'+item['id'];guard();rec=captures[item['id']];n=rec['history'];f=rec['sites'][12]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);g=gate*np.exp(-np.logaddexp(0.,-gate));bb=raw(f['B']);cc=raw(f['C']);dt=raw(f['delta']);ref=raw(f['output']);z=load(item['file']);px=z['projected_input'].astype('f8');lat=z['latent'].astype('f8');wv=[w.astype('f4').astype('f8') for w in ws];vv=[v.astype('f4').astype('f8') for v in vs];expected_px=np.concatenate([x[:,h]@wv[h] for h in range(48)],axis=1);array_check(px,expected_px,b['numeric']['projected_input_relative'],'projected_input');state=np.zeros((384,256));direct=np.empty_like(lat)
            for t in range(n):
                if t%64==0:guard()
                delta=dt[t,ids];state*=np.exp(delta*aa[ids])[:,None];state+=(delta*px[t])[:,None]*bb[t][None,:];direct[t]=np.sum(state*cc[t][None,:],axis=1)+dd[ids]*px[t]
            array_check(lat,direct,b['numeric']['scan_relative'],'full_sequential_scan');scan_words+=lat.size;Y=np.empty((n,48,64))
            ep=np.concatenate([y.reshape(n,48,64)[:,h]@wv[h] for h in range(48)],axis=1);erp=np.concatenate([(y.reshape(n,48,64)[:,h]-dd[h]*x[:,h])@wv[h] for h in range(48)],axis=1);pe=rel(lat,ep);rr=rel(lat-dd[ids][None,:]*px,erp);assert pe<=b['numeric']['scan_relative'] and rr<=b['numeric']['scan_relative'];scalar(item['full_scan_relative'],pe);scalar(item['recurrent_scan_relative'],rr)
            for h in range(48):sl=slice(offs[h],offs[h+1]);Y[:,h]=(lat[:,sl]-dd[h]*px[:,sl])@vv[h].T+dd[h]*x[:,h]
            ug=Y.reshape(n,3072)*g;source_inv=1/np.sqrt(np.mean((y*g)**2,axis=1,keepdims=True)+eps);mm={}
            for mode in ['full_source_denominator','reconstructed_denominator']:
                inv=source_inv if mode=='full_source_denominator' else 1/np.sqrt(np.mean(ug**2,axis=1,keepdims=True)+eps);inp=decode(bf16(ug*inv*gamma));expected=np.empty((n,2048))
                for lo in range(0,n,128):guard();hi=min(lo+128,n);expected[lo:hi]=decode(bf16(inp[lo:hi]@ow.T))
                actual=decode(z[mode]);array_check(actual,expected,b['numeric']['audit_BF16_relative'],'machine_output');pos=rec['positions'];mm[mode]=dict(output_relative_RMS=rel(actual,ref),centered_output_relative_RMS=rel(actual-actual.mean(0),ref-ref.mean(0)),label_output_relative_RMS=rel(actual[pos],ref[pos]));output_words+=actual.size
                for key,value in mm[mode].items():scalar(item['metrics'][mode][key],value)
            machine.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],metrics=mm));print(json.dumps(dict(stage=stage,complete=len(machine),seconds=time.perf_counter()-CLOCK)),flush=True)
        assert len(machine)==(48 if selected_alpha>0 else 0)
        for mode in r['aggregates']:
            sr=[q for q in machine if q['split']=='DEV'];values=[q['metrics'][mode]['output_relative_RMS'] for q in sr];m=float(np.mean(values));w=max(values);c=float(np.mean([q['metrics'][mode]['centered_output_relative_RMS'] for q in sr]));dm=max(float(np.mean([q['metrics'][mode]['output_relative_RMS'] for q in sr if q['domain']==d])) for d in {q['domain'] for q in sr});expected=dict(mean=m,worst=w,centered_mean=c,worst_domain_mean=dm)
            for key,value in expected.items():scalar(r['aggregates'][mode][key],value)
            assert r['gates'][mode]==dict(mean=m<=.1,worst=w<=.2,centered=c<=.1,domain=dm<=.2)
        assert all(r[k]==0 for k in ['source_forwards','native_calls']) and r['optimizer_updates']==int(selected_alpha>0) and r['new_packed_scans']==len(machine);check();guard();ad=dict(schema='FINITE_RECURRENCE_GRADIENT_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],full_autograd_FIT_cases=autograd_cases,full_GY_words_verified=gradient_words,V1_V2_prefix_arrays_bit_exact=True,independent_Kronecker_tangent_verified=True,parameter_replay_words=parameter_words,all_output_words_verified=output_words,full_sequential_scan_words_verified=scan_words,selected_alpha=selected_alpha,maximum=maxima,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='All backward/constraint/FD/FIT candidate selection/parameter replay/ideal and actual compact scan/BF16 output qualification. Parameter replay uses frozen helper and stored qualified direction;independent math checks are distinct. No native/model/n/50 admission.')
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
