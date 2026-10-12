"""Full finite SSM output loss/backward, tangent projection and one FIT update."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from native_head_gpu_preflight import gpu_metadata
START=time.monotonic();CLOCK=time.perf_counter()


def retract(np,V,W,dV,dW,alpha,limits):
    vv=[];ww=[];checks=[];feasible=True
    for v,w,dv,dw in zip(V,W,dV,dW):
        vr=v+alpha*dv;wr=w+alpha*dw;q,R=np.linalg.qr(vr,mode='reduced');sign=np.where(np.diag(R)<0,-1.,1.);q=q*sign[None,:];R=sign[:,None]*R;M=wr.T@q;sv=np.linalg.svd(M,compute_uv=False);vsv=np.linalg.svd(vr,compute_uv=False);cond=float(sv[0]/max(sv[-1],1e-300));rankratio=float(vsv[-1]/max(vsv[0],1e-300));ok=cond<=limits['M_condition'] and rankratio>=limits['V_rank_ratio']
        if ok:
            wn=np.linalg.solve(M,wr.T).T;bio=float(np.max(np.abs(wn.T@q-np.eye(q.shape[1]))));ok=bio<=limits['biorthogonality'] and np.isfinite(wn).all()
        else:wn=wr;bio=None
        checks.append(dict(M_condition=cond,V_rank_ratio=rankratio,biorthogonality=bio,passed=bool(ok)));feasible=feasible and bool(ok);vv.append(q);ww.append(wn)
    return vv,ww,checks,feasible


def tangent(np,V,W,GV,GW):
    projectedV=[];projectedW=[];residuals=[]
    for v,w,gv,gw in zip(V,W,GV,GW):
        A=w.T@w;B=v.T@v;C=gw.T@v+w.T@gv;ea,qa=np.linalg.eigh(A);eb,qb=np.linalg.eigh(B);lam=qa@((qa.T@C@qb)/(ea[:,None]+eb[None,:]))@qb.T;pv=gv-w@lam;pw=gw-v@lam.T
        projectedV.append(pv);projectedW.append(pw);residuals.append(float(np.linalg.norm(pw.T@v+w.T@pv)/max(np.linalg.norm(C),1e-30)))
    norm=float(np.sqrt(sum(np.sum(x*x) for x in projectedV+projectedW)));theta=float(np.sqrt(sum(np.sum(x*x) for x in V+W)));scale=theta/max(norm,1e-30);dV=[-scale*x for x in projectedV];dW=[-scale*x for x in projectedW];slope=float(sum(np.sum(gv*dv)+np.sum(gw*dw) for gv,gw,dv,dw in zip(GV,GW,dV,dW)))
    return dV,dW,dict(projected_gradient_norm=norm,parameter_norm=theta,scale=scale,slope=slope,tangent_relative_residuals=residuals)


def prepare(a):
    prior_path=DOC/'recurrent_output_bases_result_20261012.json';prior=json.loads(prior_path.read_bytes());parent=json.loads(Path(prior['binding']['path']).read_bytes());capture=json.loads(Path(parent['capture']['path']).read_bytes())['records']
    diag_path=DOC/'coupled_recurrence_sensitivity_result_20261012.json';ad_path=DOC/'coupled_recurrence_sensitivity_adjudication_20261012.json';ad=json.loads(ad_path.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(diag_path)
    storage_path=DOC/'finite_recurrence_gradient_storage_v3_20261012.json';storage=json.loads(storage_path.read_text(encoding='utf-8-sig'));packet=next(p for p in prior['selections'] if p['site']==12)
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',B/'native_head_gpu_preflight.py',B/'original_falcon_mixed_history.py',B/'original_falcon_channel_transport.py',B/'original_falcon_whole_recovery.py',B/'chatbot_falcon_usability_launch.py',B/'chatbot_falcon_usability.py',SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py',DOC/'FINITE_RECURRENCE_GRADIENT_PROTOCOL_20261012.md',DOC/'FINITE_RECURRENCE_GRADIENT_CONTROL_SCALE_20261012.md',DOC/'finite_recurrence_gradient_partial_adjudication_v3_20261012.json',storage_path,prior_path,diag_path,ad_path,Path(prior['binding']['path']),Path(parent['capture']['path']),Path(parent['parent']['source_weights']),Path(parent['parent']['source_config']),Path(packet['basis']['path'])]
    for q in capture:paths += [Path(e['path']) for e in q['sites'][12]['fields'].values()]
    fault_path=Path('E:/SiliconLLM/results/native_expert_scaling/finite_recurrence_gradient_20261012/first_fault.json');fault=json.loads(fault_path.read_bytes());paths += [DOC/'FINITE_RECURRENCE_GRADIENT_TRANSPOSE_FAULT_20261012.md',DOC/'finite_recurrence_gradient_result_20261012.terminal.json',fault_path,fault_path.parent/'initial_gauge.npz']+[Path(q['file']['path']) for q in fault['gradient_cases']]
    runtime=[e for e in parent['inputs'] if e['path'].startswith(str(Path(sys.executable).parent)) or e['path'].startswith(str(SITE.resolve())) or e['path'].startswith('C:\\Windows\\System32')]
    paths += [ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']]
    b=dict(schema='FINITE_RECURRENCE_GRADIENT_BINDING_V1',python=parent['python'],gpu=parent['gpu'],versions=parent['versions'],parent=parent['parent'],capture=parent['capture'],prior=extent(prior_path),diagnosis=extent(diag_path),diagnosis_adjudication=extent(ad_path),packet=packet,site=12,layout='adaptive384',output_directory=storage['new_output_directory'],alphas=[0.,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,.1,.3,1.],FD_steps=[1e-6,1e-7,1e-8],FD_tolerances=[1e-6,1e-6,1e-6],
           numeric=dict(biorthogonality=1e-8,M_condition=1e8,V_rank_ratio=1e-12,gauge_projector_relative=1e-9,tangent_relative=1e-8,slope_identity_relative=1e-8,prior_F32_metric_absolute=1e-5,audit_gradient_relative=1e-8,audit_scalar_relative=1e-8,audit_vector_relative=1e-9,scan_relative=1e-4,projected_input_relative=1e-6,audit_BF16_relative=1e-4),selection=dict(Armijo=1e-4,minimum_relative_improvement=1e-8,rule='minimum FIT loss among feasible positive-alpha Armijo candidates;smaller alpha breaks exact tie;DEV excluded'),
           criteria=dict(DEV_output_mean=.1,DEV_case_worst=.2,DEV_domain_mean_worst=.2,DEV_centered_mean=.1),limits=dict(seconds=1800,audit_seconds=1800,OS_bytes=6<<30,GPU_allocated_bytes=4<<30,GPU_reserved_bytes=5<<30,output_bytes=4<<30),inputs=list({e['path']:e for e in runtime+[extent(p) for p in paths]}.values()),scope='Site12 adaptive384 fixed ranks,FIT-only exact finite source-output backward/tangent/one line-search update. New actual compact scans if accepted;full source generators/gates/conv/state256/skip retained,no complete native/chatbot/n/50 admission.')
    assert gpu_metadata()==b['gpu'];write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.perf_counter()-CLOCK)),flush=True)


def run(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np,psutil,torch
    from original_falcon_mixed_history import scan
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and gpu_metadata()==b['gpu'] and str(a.directory.resolve())==str(Path(b['output_directory']).resolve()) and not a.directory.exists() and not a.out.exists()
    assert dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='6' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();torch.cuda.reset_peak_memory_stats();stream=None;stage='startup';rows=[];points=[];gradient_rows=[]
    receipt=dict(schema='FINITE_RECURRENCE_GRADIENT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        l=b['limits'];assert time.monotonic()-START<=l['seconds'] and proc.memory_info().peak_wset<=l['OS_bytes'] and not proc.children(recursive=True);assert torch.cuda.max_memory_allocated()<=l['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=l['GPU_reserved_bytes']
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=l['output_bytes']
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');z=z.reshape(e['shape']).astype('f8');assert np.isfinite(z).all();return torch.from_numpy(z).cuda()
    def save_params(name,V,W,**extra):
        fields={**{f'V_{h}':v for h,v in enumerate(V)},**{f'W_{h}':w for h,w in enumerate(W)},**extra};p=a.directory/(name+'.npz');np.savez(p,**fields);return extent(p)
    try:
        check();a.directory.mkdir();capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];fit=[q for q in capture if q['split']=='FIT'];dev=[q for q in capture if q['split']=='DEV'];assert len(fit)==len(dev)==24;diag=json.loads(Path(b['diagnosis']['path']).read_bytes());diagrows={q['id']:q for q in diag['records'] if q['site']==12}
        get,stream=tensor_reader(b['parent']['source_weights'],np);prefix='model.layers.12.mamba.';dd=torch.from_numpy(get(prefix+'D')).cuda();F=torch.from_numpy(get(prefix+'out_proj.weight')).cuda()*torch.from_numpy(get(prefix+'norm.weight')).cuda()[None,:];ow=torch.from_numpy(get(prefix+'out_proj.weight')).cuda();gamma=torch.from_numpy(get(prefix+'norm.weight')).cuda();alog=torch.from_numpy(get(prefix+'A_log')).float().cuda();eps=json.loads(Path(b['parent']['source_config']).read_bytes())['rms_norm_eps'];ranks=b['packet']['variants']['adaptive384']['ranks'];offs=b['packet']['variants']['adaptive384']['offsets']
        with np.load(b['packet']['basis']['path'],allow_pickle=False) as z:rawV=[z[f'adaptive384_V_{h}'].copy() for h in range(48)];rawW=[z[f'adaptive384_W_{h}'].copy() for h in range(48)]
        zeroV=[np.zeros_like(v) for v in rawV];zeroW=[np.zeros_like(w) for w in rawW];V0,W0,gauge_checks,gauge_ok=retract(np,rawV,rawW,zeroV,zeroW,0.,b['numeric']);assert gauge_ok;gauge_err=max(float(np.linalg.norm(v@w.T-rv@rw.T)/max(np.linalg.norm(rv@rw.T),1e-30)) for v,w,rv,rw in zip(V0,W0,rawV,rawW));assert gauge_err<=b['numeric']['gauge_projector_relative'];initial=save_params('initial_gauge',V0,W0,**{f'raw_V_{h}':v for h,v in enumerate(rawV)},**{f'raw_W_{h}':w for h,w in enumerate(rawW)})
        def case(rec,V,W,backward=False):
            guard();n=rec['history'];f=rec['sites'][12]['fields'];x=raw(f['x']).reshape(n,48,64);y=raw(f['y']);gate=raw(f['gate']);g=gate*torch.sigmoid(gate);R=y.reshape(n,48,64)-dd[None,:,None]*x;skip=dd[None,:,None]*x;source_u=y*g;teacher=(source_u/torch.sqrt(source_u.square().mean(-1,keepdim=True)+eps))@F.T;den=teacher.square().sum();vv=[torch.from_numpy(v).cuda() for v in V];ww=[torch.from_numpy(w).cuda() for w in W];Z=[R[:,h]@ww[h] for h in range(48)];Y=torch.stack([Z[h]@vv[h].T+skip[:,h] for h in range(48)],dim=1).reshape(n,3072);U=Y*g;radius=torch.sqrt(U.square().mean(-1,keepdim=True)+eps);op=(U/radius)@F.T;diff=op-teacher;loss=float(diff.square().sum()/den);relative=float(torch.sqrt(diff.square().sum()/den));result=dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,loss=loss,relative_RMS=relative,teacher_energy=float(den),centered_relative_RMS=float(torch.linalg.vector_norm((op-op.mean(0))-(teacher-teacher.mean(0)))/torch.linalg.vector_norm(teacher-teacher.mean(0)).clamp_min(1e-30)))
            if backward:
                go=2*diff/(24*den);gx=go@F;gy=g*(gx/radius-U*(U*gx).sum(-1,keepdim=True)/(3072*radius.pow(3)));gh=gy.reshape(n,48,64);GV=[(gh[:,h].T@Z[h]).cpu().numpy() for h in range(48)];GW=[(R[:,h].T@(gh[:,h]@vv[h])).cpu().numpy() for h in range(48)];return result,op.cpu().numpy(),gy.cpu().numpy(),GV,GW
            return result,op.cpu().numpy()
        with torch.inference_mode():
            GV=[np.zeros_like(v) for v in V0];GW=[np.zeros_like(w) for w in W0];baseline=[]
            for rec in fit:
                stage='baseline_gradient/'+rec['id'];m,op,gy,cv,cw=case(rec,V0,W0,True);baseline.append(m)
                for h in range(48):GV[h]+=cv[h];GW[h]+=cw[h]
                p=a.directory/(rec['id']+'.gradient.npz');np.savez(p,output=op,GY=gy,**{f'GV_{h}':v for h,v in enumerate(cv)},**{f'GW_{h}':w for h,w in enumerate(cw)});gradient_rows.append(dict(**m,file=extent(p)));print(json.dumps(dict(stage=stage,seconds=time.perf_counter()-CLOCK)),flush=True)
            L0=sum(m['loss'] for m in baseline)/24;dV,dW,tangent_record=tangent(np,V0,W0,GV,GW);assert max(tangent_record['tangent_relative_residuals'])<=b['numeric']['tangent_relative'] and tangent_record['slope']<0;identity=abs(tangent_record['slope']+tangent_record['scale']*tangent_record['projected_gradient_norm']**2)/max(abs(tangent_record['slope']),1e-30);assert identity<=b['numeric']['slope_identity_relative'];gradient=save_params('full_gradient_direction',GV,GW,**{f'dV_{h}':v for h,v in enumerate(dV)},**{f'dW_{h}':w for h,w in enumerate(dW)})
            def evaluate(V,W,records):return [case(rec,V,W)[0] for rec in records]
            coarse_FD=[]
            for step,tol in zip([1e-3,1e-4,1e-5],[1e-4,1e-5,1e-6]):
                vp,wp,cp,okp=retract(np,V0,W0,dV,dW,step,b['numeric']);vm,wm,cm,okm=retract(np,V0,W0,dV,dW,-step,b['numeric']);assert okp and okm;plus=evaluate(vp,wp,fit);minus=evaluate(vm,wm,fit);lp=sum(q['loss'] for q in plus)/24;lm=sum(q['loss'] for q in minus)/24;fd=(lp-lm)/(2*step);error=abs(fd-tangent_record['slope'])/max(abs(tangent_record['slope']),1e-30);coarse_FD.append(dict(step=step,tolerance=tol,plus_loss=lp,minus_loss=lm,derivative=fd,relative=error,original_gate_passed=bool(error<=tol),plus_cases=plus,minus_cases=minus,plus_parameters=save_params('coarse_plus_'+str(step),vp,wp),minus_parameters=save_params('coarse_minus_'+str(step),vm,wm)))
            FD=[]
            for step,tol in zip(b['FD_steps'],b['FD_tolerances']):
                vp,wp,cp,okp=retract(np,V0,W0,dV,dW,step,b['numeric']);vm,wm,cm,okm=retract(np,V0,W0,dV,dW,-step,b['numeric']);assert okp and okm;plus=evaluate(vp,wp,fit);minus=evaluate(vm,wm,fit);lp=sum(q['loss'] for q in plus)/24;lm=sum(q['loss'] for q in minus)/24;fd=(lp-lm)/(2*step);error=abs(fd-tangent_record['slope'])/max(abs(tangent_record['slope']),1e-30);assert error<=tol;FD.append(dict(step=step,plus_loss=lp,minus_loss=lm,derivative=fd,relative=error,plus_cases=plus,minus_cases=minus,plus_parameters=save_params('FD_plus_'+str(step),vp,wp),minus_parameters=save_params('FD_minus_'+str(step),vm,wm)))
            for alpha in b['alphas']:
                stage='candidate/'+str(alpha);vp,wp,checks,feasible=retract(np,V0,W0,dV,dW,alpha,b['numeric']);point=dict(alpha=alpha,feasible=feasible,checks=checks,parameters=save_params('point_'+str(alpha),vp,wp))
                if feasible:
                    cases=baseline if alpha==0 else evaluate(vp,wp,fit);loss=sum(q['loss'] for q in cases)/24;point.update(FIT_cases=cases,FIT_loss=loss,Armijo=bool(alpha>0 and loss<=L0+b['selection']['Armijo']*alpha*tangent_record['slope']),minimum_improvement=bool(loss<L0*(1-b['selection']['minimum_relative_improvement'])))
                points.append(point);print(json.dumps(dict(stage=stage,feasible=feasible,FIT_loss=point.get('FIT_loss'),seconds=time.perf_counter()-CLOCK)),flush=True)
            accepted=[p for p in points if p.get('Armijo') and p.get('minimum_improvement')];chosen=min(accepted,key=lambda p:(p['FIT_loss'],p['alpha'])) if accepted else points[0];alpha=chosen['alpha'];Vs,Ws,_,ok=retract(np,V0,W0,dV,dW,alpha,b['numeric']);assert ok;selected=save_params('selected_parameters',Vs,Ws);ideal=[];baseline_controls=[]
            for rec in capture:
                stage='full_evaluation/'+rec['id'];m0,o0=case(rec,V0,W0);ms,osel=case(rec,Vs,Ws);control=abs(m0['relative_RMS']-diagrows[rec['id']]['metrics']['adaptive384']['finite_relative_RMS']);assert control<=b['numeric']['prior_F32_metric_absolute'];baseline_controls.append(dict(id=rec['id'],promoted_F32_metric_delta=control));p=a.directory/(rec['id']+'.ideal.npz');np.savez(p,baseline_output=o0,selected_output=osel);ideal.append(dict(id=rec['id'],baseline=m0,selected=ms,file=extent(p)))
                if alpha>0:
                    n=rec['history'];f=rec['sites'][12]['fields'];x=raw(f['x']).float().reshape(n,48,64);y=raw(f['y']);gateraw=raw(f['gate']);g=gateraw*torch.sigmoid(gateraw);bb=raw(f['B']).float();cc=raw(f['C']).float();dt=raw(f['delta']).bfloat16();lat=torch.empty((n,384),device='cuda');px=torch.empty_like(lat)
                    for rank in sorted(set(ranks)):
                        heads=[h for h in range(48) if ranks[h]==rank];w=torch.stack([torch.from_numpy(Ws[h].astype('f4')).cuda() for h in heads]);xm=torch.einsum('thc,hcr->thr',x[:,heads],w);ym=scan(torch,code,xm,bb,cc,dt[:,heads],alog[heads],dd[heads].float())
                        for j,h in enumerate(heads):sl=slice(offs[h],offs[h+1]);lat[:,sl]=ym[:,j];px[:,sl]=xm[:,j]
                    expected=torch.empty_like(lat,dtype=torch.float64);rec_expected=torch.empty_like(expected);Y=torch.empty((n,48,64),device='cuda',dtype=torch.float64);skip=torch.empty_like(expected)
                    for h in range(48):sl=slice(offs[h],offs[h+1]);w=torch.from_numpy(Ws[h].astype('f4').astype('f8')).cuda();v=torch.from_numpy(Vs[h].astype('f4').astype('f8')).cuda();expected[:,sl]=y.reshape(n,48,64)[:,h]@w;rec_expected[:,sl]=(y.reshape(n,48,64)[:,h]-dd[h]*x[:,h].double())@w;skip[:,sl]=dd[h]*px[:,sl].double();Y[:,h]=(lat[:,sl].double()-skip[:,sl])@v.T+dd[h]*x[:,h].double()
                    pe=float(torch.linalg.vector_norm(lat.double()-expected)/torch.linalg.vector_norm(expected));re=float(torch.linalg.vector_norm(lat.double()-skip-rec_expected)/torch.linalg.vector_norm(rec_expected));assert pe<=b['numeric']['scan_relative'] and re<=b['numeric']['scan_relative'];ug=Y.reshape(n,3072)*g;source_inv=torch.rsqrt((y*g).square().mean(-1,keepdim=True)+eps);machine={};metrics={};reference=raw(f['output']);pos=torch.tensor(rec['positions'],device='cuda')
                    for mode,inv in [('full_source_denominator',source_inv),('reconstructed_denominator',torch.rsqrt(ug.square().mean(-1,keepdim=True)+eps))]:
                        output=(((ug*inv*gamma).bfloat16().double())@ow.T).bfloat16();of=output.double();metrics[mode]=dict(output_relative_RMS=float(torch.linalg.vector_norm(of-reference)/torch.linalg.vector_norm(reference)),centered_output_relative_RMS=float(torch.linalg.vector_norm((of-of.mean(0))-(reference-reference.mean(0)))/torch.linalg.vector_norm(reference-reference.mean(0))),label_output_relative_RMS=float(torch.linalg.vector_norm(of[pos]-reference[pos])/torch.linalg.vector_norm(reference[pos])));machine[mode]=output.cpu().view(torch.uint16).numpy().astype('<u2')
                    p=a.directory/(rec['id']+'.scan.npz');np.savez(p,latent=lat.cpu().numpy().astype('f4'),projected_input=px.cpu().numpy().astype('f4'),**machine);rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,full_scan_relative=pe,recurrent_scan_relative=re,metrics=metrics,file=extent(p)))
                print(json.dumps(dict(stage=stage,complete=len(ideal),seconds=time.perf_counter()-CLOCK)),flush=True);torch.cuda.empty_cache()
        stream.close();stream=None;check();guard();aggregates={};gates={}
        for mode in ['full_source_denominator','reconstructed_denominator']:
            rr=[r for r in rows if r['split']=='DEV']
            if rr:
                mean=sum(r['metrics'][mode]['output_relative_RMS'] for r in rr)/24;worst=max(r['metrics'][mode]['output_relative_RMS'] for r in rr);center=sum(r['metrics'][mode]['centered_output_relative_RMS'] for r in rr)/24;domain=max(sum(r['metrics'][mode]['output_relative_RMS'] for r in rr if r['domain']==d)/2 for d in {r['domain'] for r in rr});aggregates[mode]=dict(mean=mean,worst=worst,centered_mean=center,worst_domain_mean=domain);gates[mode]=dict(mean=mean<=.1,worst=worst<=.2,centered=center<=.1,domain=domain<=.2)
        result=dict(schema='FINITE_RECURRENCE_GRADIENT_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,initial=initial,gauge_checks=gauge_checks,gauge_projector_relative=gauge_err,gradient=gradient,gradient_cases=gradient_rows,tangent=tangent_record,slope_identity_relative=identity,coarse_FD=coarse_FD,FD=FD,points=points,selected_alpha=alpha,selected=selected,FIT_initial_loss=L0,FIT_selected_loss=chosen.get('FIT_loss',L0),ideal=ideal,baseline_controls=baseline_controls,machine_records=rows,aggregates=aggregates,gates=gates,decision='FIT_UPDATE_ACCEPTED_PENDING_AUDIT' if alpha>0 else 'NO_FIT_UPDATE_ACCEPTED_PENDING_AUDIT',before_after_hashes_exact=True,source_forwards=0,optimizer_updates=int(alpha>0),new_packed_scans=len(rows),native_calls=0,quality_admission=False,speed_admission=False,output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(selected_alpha=alpha,FIT_initial=L0,FIT_selected=chosen.get('FIT_loss'),aggregates=aggregates,result=extent(a.out))),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());
        if a.directory.exists():write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),points=points,gradient_cases=gradient_rows,machine_records=rows))
        raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved());write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args();prepare(a) if a.prepare else run(a)
