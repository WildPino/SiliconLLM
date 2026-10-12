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
    re=extent(a.report);te=extent(a.terminal);assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha;r=json.loads(a.report.read_bytes());terminal=json.loads(a.terminal.read_bytes());assert terminal['exit_code']==1 and extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions']
    assert not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);stream=None;stage='startup';maxima={};gradient_words=0;autograd_cases=0;output_words=0;scan_words=0;parameter_words=0
    receipt=dict(schema='FINITE_RECURRENCE_GRADIENT_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False);sources=[extent(DOC/'finite_recurrence_gradient_transpose_check_20261012.json'),extent(Path(__file__)),extent(B/'finite_recurrence_gradient_v2.py'),extent(B/'falcon_auxiliary_s6_transport.py'),extent(DOC/'FINITE_RECURRENCE_GRADIENT_PARTIAL_AUDIT_PROTOCOL_20261012.md')]
    def guard():assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    files=[r['initial'],r['gradient'],r['selected']]+[q['file'] for q in r['gradient_cases']+r['ideal']+r['machine_records']]+[q['parameters'] for q in r['points']]+[q[k] for q in r['FD'] for k in ['plus_parameters','minus_parameters']]
    def check():
        for e in b['inputs']+[r['binding'],re,te]+sources+files+[r['first_fault']]:guard();assert extent(e['path'])==e,e['path']
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
        theta=np.sqrt(sum(np.sum(q*q) for q in V+W));scale=theta/np.sqrt(norm2);slope=sum(np.sum(gv*dv)+np.sum(gw*dw) for gv,gw,dv,dw in zip(storedGV,storedGW,dV,dW));assert slope<0;prior_check=json.loads((DOC/'finite_recurrence_gradient_transpose_check_20261012.json').read_bytes());scalar(prior_check['tangent']['slope'],slope);scalar(slope,-scale*norm2)
        for h in range(48):array_check(dV[h],-scale*projectedV[h],b['numeric']['audit_gradient_relative'],'independent_tangent_V');array_check(dW[h],-scale*projectedW[h],b['numeric']['audit_gradient_relative'],'independent_tangent_W')
        controls=[]
        for step,tol in zip([1e-3,1e-4,1e-5,1e-6],[1e-4,1e-5,1e-6,1e-6]):
            stage='full_FIT_FD/'+str(step);vp,wp,_,okp=retract(np,V,W,dV,dW,step,b['numeric']);vm,wm,_,okm=retract(np,V,W,dV,dW,-step,b['numeric']);assert okp and okm;plus=[];minus=[]
            with torch.inference_mode():
                for rec in fit:plus.append(forward(rec,vp,wp)[0]);minus.append(forward(rec,vm,wm)[0])
            lp=sum(q['loss'] for q in plus)/24;lm=sum(q['loss'] for q in minus)/24;fd=(lp-lm)/(2*step);err=abs(fd-slope)/max(abs(slope),1e-30);controls.append(dict(step=step,tolerance=tol,plus_loss=lp,minus_loss=lm,derivative=fd,relative=err,original_gate_passed=bool(err<=tol),plus_cases=plus,minus_cases=minus));print(json.dumps(dict(stage=stage,error=err,passed=bool(err<=tol),seconds=time.perf_counter()-CLOCK)),flush=True)
        stream.close();stream=None;check();guard();ad=dict(schema='FINITE_RECURRENCE_GRADIENT_PARTIAL_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,report=re,terminal=te,binding=r['binding'],first_fault=r['first_fault'],full_autograd_FIT_cases=autograd_cases,full_GY_words_verified=gradient_words,V1_V2_prefix_arrays_bit_exact=True,independent_Kronecker_tangent_verified=True,slope=slope,FIT_loss=sum(q['loss'] for q in baseline)/24,FD=controls,maximum=maxima,before_after_hashes_exact=True,decision='FULL_SAVED_BACKWARD_AND_TANGENT_PASS_FD_FLAGS_RETAINED',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='All saved V1/V2 baseline output/GY/per-case and total gradients;CPU autograd plus radial/tangent,Kron Sylvester,full FIT FD-scale diagnosis. Every original false FD flag retained;no candidate update/model/native admission.')
        write(a.out,ad);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),slope=slope,FD=[{k:q[k] for k in ['step','relative','original_gate_passed']} for q in controls],maximum=maxima)),flush=True)
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
