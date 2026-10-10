"""Fixed paired-head KL projection: warm codes, valid bounds, no head refit."""
import argparse
import ctypes
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];BROOT=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(BROOT))
from original_packed_capacity import SITE,extent,sha,write
from original_falcon_whole_recovery import memory_reader
from source_cached_fit_capture import decode,save
sys.path.insert(0,str(SITE))
V,K,D,N=65537,255,256,72


def bind(a):
    assert not a.binding.exists()
    result_path=DOC/'paired_output_codec_result_20261010.json';r=json.loads(result_path.read_bytes())
    audit_path=DOC/'paired_output_codec_stored_adjudication_20261010.json';audit=json.loads(audit_path.read_bytes())
    receipt_path=audit_path.with_suffix('.terminal.json');receipt=json.loads(receipt_path.read_bytes())
    parent_path=DOC/'paired_output_codec_binding_20261010.json';parent=json.loads(parent_path.read_bytes())
    source_meta_path=DOC/'source_cached_fit_binding_20261010.json';source_meta=json.loads(source_meta_path.read_bytes())
    assert r['decision']==audit['decision']=='PAIRED_CODEC_FIT_QUALITY_FAIL' and all(r['numeric_flags'].values())
    assert audit['result']==extent(result_path) and receipt['exit_code']==0 and receipt['error'] is None and receipt['result']==extent(audit_path)
    assert audit['binding']==extent(parent_path) and r['binding_sha256']==sha(parent_path)
    by_id={rec['id']:rec for rec in source_meta['records']};selected=[]
    files=[Path(__file__),a.protocol,parent_path,result_path,result_path.with_suffix('.terminal.json'),audit_path,receipt_path,source_meta_path,
        BROOT/'original_packed_capacity.py',BROOT/'original_falcon_whole_recovery.py',BROOT/'source_cached_fit_capture.py',BROOT/'paired_output_codec.py',
        BROOT/'original_latent_readout_attribution.py',ROOT/'benchmarks/phase60/engine.c']
    files += [Path(item['path']) for item in parent['inputs'] if str(SITE.resolve()).lower() in item['path'].lower() or item['path'].lower().endswith(('.exe','.dll','.pyd'))]
    for row in r['records']:
        m=row['labels'];rec=by_id[row['id']];assert row['split']==rec['split']=='FIT' and m==len(rec['output_ids'])>=3
        indices=[0,m//2,m-1];assert len(set(indices))==3
        selected.append(dict(id=row['id'],domain=row['domain'],case_labels=m,label_indices=indices,positions=[rec['positions'][j] for j in indices],
            teacher=row['source_logits'],phi=row['phi'],initial_KL=[row['real']['KL_per_label'][j] for j in indices],
            initial_disagreement=[row['real']['disagreement_per_label'][j] for j in indices]))
        files += [Path(row['source_logits']['path']),Path(row['phi']['path'])]
    assert len(selected)==24 and sum(rec['case_labels'] for rec in selected)==4422 and len({rec['domain'] for rec in selected})==12
    for key in ('decoder','head','final_norm'):files.append(Path(r['artifacts'][key]['path']))
    for key in ('dll','source','engine'):files.append(Path(parent['native'][key]['path']))
    inputs=[extent(path) for path in dict.fromkeys(files)];by_path={item['path']:item for item in inputs}
    for item in parent['inputs']:
        if item['path'] in by_path:assert by_path[item['path']]==item
    for rec in selected:
        for key in ('teacher','phi'):assert by_path[rec[key]['path']]=={name:rec[key][name] for name in ('path','bytes','sha256')}
    for key in ('decoder','head','final_norm'):
        item=r['artifacts'][key];assert by_path[item['path']]=={name:item[name] for name in ('path','bytes','sha256')}
    write(a.binding,dict(schema='PAIRED_HEAD_IMAGE_BINDING_V1',inputs=inputs,selected=selected,artifacts={key:r['artifacts'][key] for key in ('decoder','head','final_norm')},
        native=parent['native'],gain=r['calibration']['gain'],rho=1.,epsilon=1e-5,radius=16.,labels=N,head_dimension=K,target_dimension=D,vocab=V,
        solver=dict(max_iterations=512,max_backtracks=30,initial_L=1.,L_floor=1e-12,gap_tolerance=1e-5,certificate_slack=1e-8,majorant_slack=1e-12,
            checkpoint_every=16,initialization='Stored paired phi/gain; no zero start, scale arm, head fit or restart.'),
        decisions=dict(encoder_upper_to_initial=.25,cohort_KL=.01,case_KL=.05,domain_KL=.05,sampled_mean_KL=.01,sampled_max_KL=.05,sampled_disagreement=.01),
        numeric=dict(metric_absolute=1e-9,gradient_absolute=1e-8,bounds_absolute=1e-8,initial_loss_absolute=1e-10,radius_slack=1e-10,
            native_score_absolute=1e-4,native_mean_KL_to_real=1e-6,native_disagreement_to_real=.01,native_norm_relative=1e-5),
        capture_limits=dict(seconds=300,reserve_seconds=30,OS_bytes=4<<30,GPU_allocated_bytes=2<<30,GPU_reserved_bytes=3<<30,output_bytes=256<<20,log_bytes=8<<20),
        audit_limits=dict(seconds=300,OS_bytes=2<<30,output_bytes=2<<20,log_bytes=8<<20),
        scope='New fitted K/gain/domain, warm72 FIT codes: bounded convex KL information probe. No old-head replay, source/history/training/DEV/T4.'))
    print(json.dumps(dict(binding=str(a.binding),sha256=sha(a.binding),inputs=len(inputs),bytes=sum(item['bytes'] for item in inputs))),flush=True)


def branch(b,labels,initial,upper,lower,wrong,numeric_flags):
    groups=[]
    for rec in b['selected']:
        ix=[j for j,lab in enumerate(labels) if lab['id']==rec['id']];assert len(ix)==3
        groups.append(dict(id=rec['id'],domain=rec['domain'],case_labels=rec['case_labels'],selected_labels=3,
            whole_case_lower=sum(lower[j] for j in ix)/rec['case_labels']))
    cohort=sum(row['whole_case_lower'] for row in groups)/24
    domains={name:sum(row['whole_case_lower'] for row in groups if row['domain']==name)/2 for name in sorted({row['domain'] for row in groups})}
    dc=b['decisions'];room=sum(upper)<=dc['encoder_upper_to_initial']*sum(initial)
    insufficient=cohort>dc['cohort_KL'] or any(row['whole_case_lower']>dc['case_KL'] for row in groups) or any(value>dc['domain_KL'] for value in domains.values())
    sampled=sum(upper)/N<=dc['sampled_mean_KL'] and max(upper)<=dc['sampled_max_KL'] and wrong/N<=dc['sampled_disagreement']
    decision='MIXED_ENCODER_ROOM_AND_HEAD_DOMAIN_FLOOR' if room and insufficient else (
        'NONLINEAR_ENCODER_ROOM' if room else ('HEAD_DOMAIN_FLOOR_PRESENT' if insufficient else 'UNRESOLVED_HEAD_IMAGE'))
    return dict(decision=decision,encoder_room=room,head_domain_floor_present=insufficient,sampled_quality_pass=sampled,
        whole_cohort_lower=cohort,whole_case_lowers=groups,whole_domain_lowers=domains,native_precision_pass=all(numeric_flags.values()),
        scope='72 sampled upper bounds; whole4422-label lowers use selected losses/actual case sizes and zero lower for omitted labels.')


def capture(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import psutil
    import torch
    assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and psutil.__version__=='7.2.2'
    b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1)
    torch.manual_seed(0);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.directory.mkdir(exist_ok=False);stage='load';calls=iteration=0;labels=[];trace=[]
    def guard():
        lim=b['capture_limits'];assert time.monotonic()-start<lim['seconds']-lim['reserve_seconds'],'worker reserve'
        assert proc.memory_info().peak_wset<=lim['OS_bytes'] and not proc.children(recursive=True)
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        assert sum(path.stat().st_size for path in a.directory.iterdir() if path.is_file())<=lim['output_bytes']
    def event(**values):guard();print(json.dumps(dict(seconds=time.monotonic()-start,**values)),flush=True)
    try:
        decoder=np.fromfile(b['artifacts']['decoder']['path'],dtype='<f8').reshape(V,K)
        rawA=decoder*b['gain'];column_mean=rawA.mean(0);matrix=torch.from_numpy(rawA-column_mean).to('cuda');initial_x=[];teacher=[]
        for rec in b['selected']:
            m=rec['case_labels'];phi=np.memmap(rec['phi']['path'],mode='r',dtype='<f8',shape=(m,K));qbits=np.memmap(rec['teacher']['path'],mode='r',dtype='<u2',shape=(m,V))
            for j,pos,oldloss,oldwrong in zip(rec['label_indices'],rec['positions'],rec['initial_KL'],rec['initial_disagreement']):
                teacher.append(decode(qbits[j]));initial_x.append(phi[j]/b['gain']);labels.append(dict(id=rec['id'],domain=rec['domain'],label_index=j,position=pos,initial_KL=oldloss,initial_disagreement=oldwrong))
        assert len(labels)==N;xx=np.stack(initial_x);qvalues=np.stack(teacher);assert np.isfinite(xx).all() and np.isfinite(qvalues).all()
        qlog=torch.log_softmax(torch.from_numpy(qvalues).to('cuda'),dim=-1);q=qlog.exp();negH=(q*qlog).sum(-1);target=q@matrix
        x=torch.from_numpy(xx).to('cuda');assert float(x.norm(dim=-1).max())<=8+b['numeric']['radius_slack']
        initial_file=save(a.directory,'initial_x.f64',xx);teacher_file=save(a.directory,'selected_teacher.f64',qvalues)
        calls=0;R=b['radius'];solver=b['solver']
        def fg(value):
            nonlocal calls
            calls+=1;guard();z=value@matrix.T;lp=torch.log_softmax(z,dim=-1)
            loss=negH-(q*lp).sum(-1);gradient=lp.exp()@matrix-target
            assert torch.isfinite(loss).all() and torch.isfinite(gradient).all()
            return loss,gradient
        def project(value):return value*(R/value.norm(dim=-1,keepdim=True).clamp_min(R))
        stage='solve';loss,gradient=fg(x);initial_loss=loss.clone();upper=loss.clone();lower=torch.zeros_like(loss);best=x.clone();lower_point=x.clone()
        assert float((initial_loss-torch.tensor([row['initial_KL'] for row in labels],device='cuda',dtype=torch.float64)).abs().max())<=b['numeric']['initial_loss_absolute']
        # Explicit F64 construction avoids accidental default-F32 baseline reference.
        steps=torch.full_like(loss,solver['initial_L']);converged=False;checkpoints=[]
        with torch.inference_mode():
            for iteration in range(solver['max_iterations']+1):
                bound=loss-(gradient*x).sum(-1)-R*gradient.norm(dim=-1);improved_lower=bound>lower
                lower_point[improved_lower]=x[improved_lower];lower=torch.maximum(lower,bound)
                improved=loss<upper;best[improved]=x[improved];upper=torch.minimum(upper,loss)
                assert float((lower-upper).max())<=solver['certificate_slack'],'inconsistent bounds'
                gap=(upper-lower).clamp_min(0);converged=bool((gap<=solver['gap_tolerance']).all())
                if iteration%solver['checkpoint_every']==0 or converged or iteration==solver['max_iterations']:
                    checkpoint=a.directory/('checkpoint_%04d.npz'%iteration)
                    with checkpoint.open('xb') as f:
                        np.savez(f,current_x=x.cpu().numpy(),current_loss=loss.cpu().numpy(),current_gradient=gradient.cpu().numpy(),
                            upper_x=best.cpu().numpy(),lower_x=lower_point.cpu().numpy(),upper=upper.cpu().numpy(),lower=lower.cpu().numpy(),steps=steps.cpu().numpy())
                        f.flush();os.fsync(f.fileno())
                    checkpoints.append(extent(checkpoint));row=dict(iteration=iteration,objective_gradient_calls=calls,upper_mean=float(upper.mean()),lower_mean=float(lower.mean()),
                        max_gap=float(gap.max()),mean_gap=float(gap.mean()),converged_labels=int((gap<=solver['gap_tolerance']).sum()))
                    trace.append(row);event(stage='solve_checkpoint',**row)
                if converged or iteration==solver['max_iterations']:break
                steps=(steps*.5).clamp_min(solver['L_floor'])
                for backtrack in range(solver['max_backtracks']):
                    candidate=project(x-gradient/steps[:,None]);delta=candidate-x;next_loss,next_gradient=fg(candidate)
                    majorant=loss+(gradient*delta).sum(-1)+.5*steps*(delta*delta).sum(-1);accepted=next_loss<=majorant+solver['majorant_slack']
                    if bool(accepted.all()):break
                    steps[~accepted]*=2
                else:raise RuntimeError('backtracking cap')
                assert float((next_loss-loss).max())<=1e-9,'projected descent violation';x,loss,gradient=candidate,next_loss,next_gradient
            stage='final_certificates';final_loss,final_grad=fg(best)
            assert float((final_loss-upper).abs().max())<=b['numeric']['metric_absolute']
            bound=final_loss-(final_grad*best).sum(-1)-R*final_grad.norm(dim=-1);improved_lower=bound>lower
            lower_point[improved_lower]=best[improved_lower];lower=torch.maximum(lower,bound)
            lower_loss,lower_grad=fg(lower_point);explicit_lower=(lower_loss-(lower_grad*lower_point).sum(-1)-R*lower_grad.norm(dim=-1)).clamp_min(0)
            assert float((explicit_lower-lower).abs().max())<=b['numeric']['bounds_absolute']
            assert float((lower-upper).max())<=solver['certificate_slack'] and float(best.norm(dim=-1).max())<=R+b['numeric']['radius_slack']
            gap=(upper-lower).clamp_min(0);best_numpy=best.cpu().numpy();lower_numpy=lower_point.cpu().numpy()
            points=dict(initial_x=initial_file,teacher=teacher_file,upper_x=save(a.directory,'upper_x.f64',best_numpy),lower_x=save(a.directory,'lower_x.f64',lower_numpy),
                upper_gradient=save(a.directory,'upper_gradient.f64',final_grad.cpu().numpy()),lower_gradient=save(a.directory,'lower_gradient.f64',lower_grad.cpu().numpy()))
            for name,value in (('initial_scores',torch.from_numpy(xx).to('cuda')@matrix.T),('upper_scores',best@matrix.T),('lower_scores',lower_point@matrix.T)):
                points[name]=save(a.directory,name+'.f64',value.cpu().numpy());guard()
            pbest=best@matrix.T;initial_scores=torch.from_numpy(xx).to('cuda')@matrix.T
            assert (initial_scores.argmax(-1)!=q.argmax(-1)).cpu().tolist()==[bool(row['initial_disagreement']) for row in labels]
            norms=np.sqrt(np.sum(best_numpy*best_numpy,axis=1));radicand=D-np.sum(best_numpy*best_numpy,axis=1)
            assert float(radicand.min())>=-b['numeric']['radius_slack']
            # Explicitly declared F64 roundoff correction, never clip a true domain violation.
            u=np.column_stack((best_numpy,np.sqrt(np.maximum(radicand,0)))).astype('<f4')
            head32=np.fromfile(b['artifacts']['head']['path'],dtype='<f4').reshape(V,D);assert np.isfinite(head32).all()
            dll=ctypes.CDLL(b['native']['dll']['path']);assert dll.codec_abi(0)==D and dll.codec_abi(1)==V
            pointer=ctypes.POINTER(ctypes.c_float);dll.codec_readout.argtypes=[pointer,pointer,ctypes.c_int,pointer,pointer];dll.codec_readout.restype=ctypes.c_int
            norm32=np.empty_like(u);native_scores=np.empty((N,V),'<f4');stage='original_F32_readout'
            assert dll.codec_readout(head32.ctypes.data_as(pointer),u.ctypes.data_as(pointer),N,norm32.ctypes.data_as(pointer),native_scores.ctypes.data_as(pointer))==0
            points['native_state']=save(a.directory,'native_state.f32',u);points['native_normalized']=save(a.directory,'native_normalized.f32',norm32)
            points['native_scores']=save(a.directory,'native_scores.f32',native_scores)
            real_raw=best_numpy@rawA.T;real_centered=pbest.cpu().numpy();assert np.isfinite(real_raw).all()
            max_native_delta=float(np.max(abs(native_scores.astype('f8')-real_raw)))
            ideal_norm=u.astype('f8')/np.sqrt(np.mean(u.astype('f8')**2,axis=1,keepdims=True)+b['epsilon'])
            norm_relative=math.sqrt(float(np.sum((norm32-ideal_norm)**2))/float(np.sum(ideal_norm**2)))
            from original_latent_readout_attribution import row_metrics
            native_metrics=[row_metrics(qrow,prow) for qrow,prow in zip(qvalues,native_scores)]
            precision=[row_metrics(qrow,prow) for qrow,prow in zip(real_centered,native_scores)]
            native_losses=[row[0] for row in native_metrics];native_wrong=sum(row[1] for row in native_metrics)
            precision_KL=sum(row[0] for row in precision)/N;precision_wrong=sum(row[1] for row in precision)
            numeric_flags=dict(norm=norm_relative<=b['numeric']['native_norm_relative'],scores=max_native_delta<=b['numeric']['native_score_absolute'],
                probability=precision_KL<=b['numeric']['native_mean_KL_to_real'],argmax=precision_wrong/N<=b['numeric']['native_disagreement_to_real'])
            uppers=upper.cpu().tolist();lowers=lower.cpu().tolist();initials=initial_loss.cpu().tolist();wrong=int((pbest.argmax(-1)!=q.argmax(-1)).sum())
            conclusion=branch(b,labels,initials,uppers,lowers,wrong,numeric_flags)
            result=dict(schema='PAIRED_HEAD_IMAGE_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,labels=labels,points=points,checkpoints=checkpoints,
                decision=conclusion['decision'],conclusion=conclusion,numeric_flags=numeric_flags,initial_mean_KL=sum(initials)/N,
                initial_per_label=initials,per_label_upper=uppers,per_label_lower=lowers,per_label_gap=gap.cpu().tolist(),
                lower_point_loss=lower_loss.cpu().tolist(),upper_mean=sum(uppers)/N,lower_mean=sum(lowers)/N,max_gap=float(gap.max()),mean_gap=float(gap.mean()),
                converged=bool((gap<=solver['gap_tolerance']).all()),converged_labels=int((gap<=solver['gap_tolerance']).sum()),
                optimistic_argmax_disagreement=wrong,native_teacher_KL=native_losses,native_teacher_disagreement=native_wrong,
                native_precision_mean_KL=precision_KL,native_precision_disagreement=precision_wrong,
                max_native_raw_real_score_absolute_delta=max_native_delta,native_norm_relative=norm_relative,
                minimum_carrier_radicand_F64=float(radicand.min()),rounded_negative_carrier_coordinates=int(np.count_nonzero(radicand<0)),
                feature_norms=norms.tolist(),iterations=iteration,latent_descent_updates=iteration,latent_row_updates=iteration*N,
                objective_gradient_calls=calls,objective_gradient_label_evaluations=calls*N,extra_stored_score_label_evaluations=N*6,
                native_final_readout_labels=N,native_final_readout_DLL_calls=1,native_full_engine_calls=0,
                source_generations=0,source_history_forwards=0,source_LM_head_calls=0,head_pairs_fit=0,model_parameter_updates=0,DEV_queries=0,reserved_queries=0,
                trace=trace,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),worker_OS_peak_snapshot=proc.memory_info().peak_wset,
                seconds=time.monotonic()-start,quality_admission=False,speed_admission=False,
                scope='One fixed newly paired head/domain; bounded latent KL optimization and native72 final readouts, no causal model or generality claim.')
            write(a.out,result);event(stage='complete',decision=result['decision'],upper_mean=result['upper_mean'],lower_mean=result['lower_mean'],max_gap=result['max_gap'])
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),iteration=iteration,objective_gradient_calls=calls,
            last_checkpoint=trace[-1] if trace else None,seconds=time.monotonic()-start));raise


def audit(a):
    import hashlib
    import re
    import numpy as np
    import psutil
    from original_latent_readout_attribution import row_metrics
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes())
    assert sha(a.binding)==a.binding_sha==r['binding_sha256']==t['binding_sha256'] and a.freeze==r['freeze']==t['freeze']
    assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(a.source_result)
    def guard():
        assert time.monotonic()-start<b['audit_limits']['seconds'] and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes']
        assert not proc.children(recursive=True)
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item;guard()
    output_map={item['path']:item for item in t['outputs']};g=b['numeric'];R=b['radius']
    def array(item,shape,dtype='<f8'):
        assert output_map[item['path']]=={key:item[key] for key in ('path','bytes','sha256')}
        value=np.fromfile(item['path'],dtype=dtype).reshape(shape);assert np.isfinite(value).all();return value
    points=r['points'];teacher=array(points['teacher'],(N,V));initial=array(points['initial_x'],(N,K))
    upper_x=array(points['upper_x'],(N,K));lower_x=array(points['lower_x'],(N,K))
    ug=array(points['upper_gradient'],(N,K));lg=array(points['lower_gradient'],(N,K))
    expected_labels=[];expected_x=[];expected_q=[]
    for rec in b['selected']:
        m=rec['case_labels'];phi=np.memmap(rec['phi']['path'],mode='r',dtype='<f8',shape=(m,K))
        qb=np.memmap(rec['teacher']['path'],mode='r',dtype='<u2',shape=(m,V))
        for j,pos,oldloss,oldwrong in zip(rec['label_indices'],rec['positions'],rec['initial_KL'],rec['initial_disagreement']):
            expected_labels.append(dict(id=rec['id'],domain=rec['domain'],label_index=j,position=pos,initial_KL=oldloss,initial_disagreement=oldwrong))
            expected_x.append(phi[j]/b['gain']);expected_q.append(decode(qb[j]))
    assert r['labels']==expected_labels and len(expected_labels)==N
    assert np.array_equal(initial,np.stack(expected_x)) and np.array_equal(teacher,np.stack(expected_q))
    rawA=np.fromfile(b['artifacts']['decoder']['path'],dtype='<f8').reshape(V,K)*b['gain'];A=rawA-rawA.mean(0)
    qlog=teacher-teacher.max(1,keepdims=True);qlog-=np.logaddexp.reduce(qlog,axis=1)[:,None];q=np.exp(qlog)
    negH=np.sum(q*qlog,axis=1);target=q@A;fg_calls=score_rows=0;maxloss=maxgrad=maxscores=0.
    def fg(x):
        nonlocal fg_calls,score_rows
        assert np.isfinite(x).all() and float(np.linalg.norm(x,axis=1).max())<=R+g['radius_slack']
        z=x@A.T;lp=z-z.max(1,keepdims=True);lp-=np.logaddexp.reduce(lp,axis=1)[:,None]
        loss=negH-np.sum(q*lp,axis=1);gradient=np.exp(lp)@A-target
        fg_calls+=1;score_rows+=N;guard();return z,loss,gradient
    observed=[]
    for key,x in (('initial_scores',initial),('upper_scores',upper_x),('lower_scores',lower_x)):
        z,loss,grad=fg(x);stored=array(points[key],(N,V));delta=float(np.max(abs(z-stored)));maxscores=max(maxscores,delta)
        assert delta<=g['metric_absolute'];observed.append((z,loss,grad))
    initial_scores,initial_loss,initial_grad=observed[0];upper_scores,upper_loss,upper_grad=observed[1];lower_scores,lower_loss,lower_grad=observed[2]
    def check_metric(actual,stored,tolerance):
        nonlocal maxloss
        delta=float(np.max(abs(np.asarray(actual)-np.asarray(stored))));maxloss=max(maxloss,delta);assert delta<=tolerance;return delta
    check_metric(initial_loss,[row['initial_KL'] for row in expected_labels],g['initial_loss_absolute'])
    check_metric(initial_loss,r['initial_per_label'],g['metric_absolute']);check_metric(upper_loss,r['per_label_upper'],g['metric_absolute'])
    check_metric(lower_loss,r['lower_point_loss'],g['metric_absolute'])
    maxgrad=max(float(np.max(abs(upper_grad-ug))),float(np.max(abs(lower_grad-lg))));assert maxgrad<=g['gradient_absolute']
    lower=np.maximum(0,lower_loss-np.sum(lower_grad*lower_x,axis=1)-R*np.linalg.norm(lower_grad,axis=1))
    check_metric(lower,r['per_label_lower'],g['bounds_absolute']);assert np.max(lower-upper_loss)<=b['solver']['certificate_slack']
    gaps=np.maximum(0,upper_loss-lower);check_metric(gaps,r['per_label_gap'],g['bounds_absolute'])
    assert (initial_scores.argmax(1)!=teacher.argmax(1)).tolist()==[bool(row['initial_disagreement']) for row in expected_labels]
    # Audit every saved current objective/gradient. No descent update or solver replay.
    previous_upper=initial_loss.copy();previous_lower=np.zeros(N);checkpoint_rows=0
    assert len(r['trace'])==len(r['checkpoints']) and r['trace'][0]['iteration']==0 and r['trace'][-1]['iteration']==r['iterations']
    for item,row in zip(r['checkpoints'],r['trace']):
        assert output_map[item['path']]==item
        with np.load(item['path'],allow_pickle=False) as cp:
            assert set(cp.files)=={'current_x','current_loss','current_gradient','upper_x','lower_x','upper','lower','steps'}
            for key in cp.files:
                assert np.isfinite(cp[key]).all() and cp[key].shape==((N,K) if key.endswith('_x') or key=='current_gradient' else (N,))
            _,cl,cg=fg(cp['current_x']);checkpoint_rows+=N
            check_metric(cl,cp['current_loss'],g['metric_absolute']);delta=float(np.max(abs(cg-cp['current_gradient'])));maxgrad=max(maxgrad,delta);assert delta<=g['gradient_absolute']
            for key in ('upper_x','lower_x'):assert np.linalg.norm(cp[key],axis=1).max()<=R+g['radius_slack']
            assert np.all(cp['steps']>=b['solver']['L_floor']) and np.all(cp['lower']>=0)
            assert np.max(cp['upper']-previous_upper)<=g['metric_absolute'] and np.max(previous_lower-cp['lower'])<=g['bounds_absolute']
            assert np.max(cp['lower']-cp['upper'])<=b['solver']['certificate_slack']
            gap=np.maximum(0,cp['upper']-cp['lower'])
            for key,value in (('upper_mean',cp['upper'].mean()),('lower_mean',cp['lower'].mean()),('max_gap',gap.max()),('mean_gap',gap.mean())):
                check_metric(value,row[key],g['metric_absolute'])
            assert int(np.count_nonzero(gap<=b['solver']['gap_tolerance']))==row['converged_labels']
            previous_upper=cp['upper'].copy();previous_lower=cp['lower'].copy()
        print(json.dumps(dict(stage='checkpoint_audited',iteration=row['iteration'],seconds=time.monotonic()-start)),flush=True)
    check_metric(previous_upper,upper_loss,g['metric_absolute']);assert np.max(previous_lower-lower)<=g['bounds_absolute']
    normsq=np.sum(upper_x*upper_x,axis=1);radicand=D-normsq;assert radicand.min()>=-g['radius_slack']
    expected_u=np.column_stack((upper_x,np.sqrt(np.maximum(radicand,0)))).astype('<f4')
    u=array(points['native_state'],(N,D),'<f4');nf=array(points['native_normalized'],(N,D),'<f4');ns=array(points['native_scores'],(N,V),'<f4')
    assert np.array_equal(u,expected_u)
    ideal=u.astype('f8')/np.sqrt(np.mean(u.astype('f8')**2,axis=1,keepdims=True)+b['epsilon'])
    norm_relative=math.sqrt(float(np.sum((nf-ideal)**2))/float(np.sum(ideal**2)))
    head=np.fromfile(b['artifacts']['head']['path'],dtype='<f4').reshape(V,D).astype('f8')
    native_reference=nf.astype('f8')@head.T;score_rows+=N
    native_linear_delta=float(np.max(abs(native_reference-ns)));assert native_linear_delta<=g['native_score_absolute']
    real_raw=upper_x@rawA.T;score_rows+=N;native_delta=float(np.max(abs(ns.astype('f8')-real_raw)))
    native=[row_metrics(qr,pr) for qr,pr in zip(teacher,ns)];precision=[row_metrics(qr,pr) for qr,pr in zip(upper_scores,ns)]
    native_losses=[row[0] for row in native];native_wrong=sum(row[1] for row in native);precision_KL=sum(row[0] for row in precision)/N;precision_wrong=sum(row[1] for row in precision)
    wrong=int(np.count_nonzero(upper_scores.argmax(1)!=teacher.argmax(1)))
    flags=dict(norm=norm_relative<=g['native_norm_relative'],scores=native_delta<=g['native_score_absolute'],probability=precision_KL<=g['native_mean_KL_to_real'],argmax=precision_wrong/N<=g['native_disagreement_to_real'])
    assert flags==r['numeric_flags'] and wrong==r['optimistic_argmax_disagreement'] and native_wrong==r['native_teacher_disagreement'] and precision_wrong==r['native_precision_disagreement']
    for actual,stored,tol in ((native_losses,r['native_teacher_KL'],g['metric_absolute']),(precision_KL,r['native_precision_mean_KL'],g['metric_absolute']),
        (native_delta,r['max_native_raw_real_score_absolute_delta'],g['metric_absolute']),(norm_relative,r['native_norm_relative'],g['metric_absolute']),
        (radicand.min(),r['minimum_carrier_radicand_F64'],g['metric_absolute']),(np.sqrt(normsq),r['feature_norms'],g['metric_absolute']),
        (initial_loss.mean(),r['initial_mean_KL'],g['metric_absolute']),(upper_loss.mean(),r['upper_mean'],g['metric_absolute']),
        (lower.mean(),r['lower_mean'],g['bounds_absolute']),(gaps.max(),r['max_gap'],g['bounds_absolute']),(gaps.mean(),r['mean_gap'],g['bounds_absolute'])):check_metric(actual,stored,tol)
    assert int(np.count_nonzero(radicand<0))==r['rounded_negative_carrier_coordinates']
    assert r['converged_labels']==int(np.count_nonzero(gaps<=b['solver']['gap_tolerance'])) and r['converged']==bool(np.all(gaps<=b['solver']['gap_tolerance']))
    # Reproduce arithmetic grouping exactly from stored F64 scalars, after independent checks.
    conclusion=branch(b,r['labels'],r['initial_per_label'],r['per_label_upper'],r['per_label_lower'],wrong,flags)
    assert conclusion==r['conclusion'] and conclusion['decision']==r['decision']
    assert r['latent_descent_updates']==r['iterations']<=b['solver']['max_iterations'] and r['latent_row_updates']==N*r['iterations']
    assert r['objective_gradient_calls']==r['trace'][-1]['objective_gradient_calls']+2 and r['objective_gradient_label_evaluations']==N*r['objective_gradient_calls']
    assert r['extra_stored_score_label_evaluations']==6*N and r['native_final_readout_labels']==N and r['native_final_readout_DLL_calls']==1
    for key in ('native_full_engine_calls','source_generations','source_history_forwards','source_LM_head_calls','head_pairs_fit','model_parameter_updates','DEV_queries','reserved_queries'):assert r[key]==0
    assert not r['quality_admission'] and not r['speed_admission']
    assert t['elapsed_seconds']<=b['capture_limits']['seconds'] and t['worker_OS_peak_through_exit']+t['launcher_OS_peak_snapshot']<=b['capture_limits']['OS_bytes']
    for key in ('GPU_allocated','GPU_reserved'):assert r[key+'_peak']<=b['capture_limits'][key+'_bytes']
    assert sum(item['bytes'] for item in t['outputs'])+t['result']['bytes']<=b['capture_limits']['output_bytes']
    for item in b['native']['functions']:
        for path in (b['native']['source']['path'],b['native']['engine']['path']):
            text=Path(path).read_text();match=re.search(r'^static[^\n]*\b'+item['name']+r'\(',text,re.M);assert match
            pos=text.index('{',match.start());depth=1;end=pos+1
            while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
            assert hashlib.sha256(text[match.start():end].encode()).hexdigest()==item['original_body_sha256']
    guard()
    write(a.out,dict(schema='PAIRED_HEAD_IMAGE_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision=r['decision'],conclusion=conclusion,numeric_flags=flags,
        complete_input_output_hashes=True,all72_final_upper_and_lower_points_verified=True,all_saved_current_objectives_gradients_verified=True,
        original_four_native_bodies_exact=True,lower_bound_kind='F64 convex tangent-ball bound; no directed-rounding interval certification',
        max_score_reconstruction_absolute_delta=maxscores,max_gradient_absolute_delta=maxgrad,max_metric_absolute_delta=maxloss,
        native_scores_F64_linear_reference_absolute_delta=native_linear_delta,selected_labels=N,full_cohort_labels=4422,
        stored_objective_gradient_calls=fg_calls,checkpoint_labels_verified=checkpoint_rows,stored_linear_head_labels_reconstructed=score_rows,
        source_generations=0,source_history_forwards=0,native_binary_calls=0,optimizer_updates=0,new_pairs_fit=0,
        seconds=time.monotonic()-start,scope='Complete hash/custody/checkpoint/final-bound/native arithmetic/metrics/decision/resource audit; no descent or native replay.'))
    print(json.dumps(dict(audit=str(a.out),decision=r['decision'],seconds=time.monotonic()-start)),flush=True)


def launch(a):
    import psutil
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','native','original_engine','clang','lld','packed_original')),('overlap',p.pid,name)
    terminal=a.out.with_suffix('.terminal.json');log=a.out.with_suffix('.worker.log')
    assert not any(path.exists() for path in (a.out,terminal,log)) and (a.audit or not a.directory.exists())
    assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());lim=b['audit_limits' if a.audit else 'capture_limits']
    began=time.monotonic();reader=memory_reader();peak=0
    for item in b['inputs']:assert extent(item['path'])==item
    assert time.monotonic()-began<lim['seconds'],'prehash deadline'
    extra=[]
    if a.audit:extra=[extent(a.source_result),extent(a.source_result.with_suffix('.terminal.json'))]
    command=[sys.executable,'-I','-S','-B','-X','utf8',str(Path(__file__).resolve()),'--audit-worker' if a.audit else '--capture-worker',
        '--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--freeze',a.freeze,'--out',str(a.out.resolve())]
    command += ['--source-result',str(a.source_result.resolve())] if a.audit else ['--directory',str(a.directory.resolve())]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1' if a.audit else '6',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
             CUDA_VISIBLE_DEVICES='' if a.audit else '0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    receipt=dict(freeze=a.freeze,binding_sha256=a.binding_sha,command=command,launcher_pid=proc.pid,limits=lim)
    with log.open('xb') as output:
        child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env,creationflags=8)
        receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time());fault=None
        try:
            while child.poll() is None:
                peak=max(peak,reader(child))
                assert time.monotonic()-began<=lim['seconds'],'deadline'
                assert peak+proc.memory_info().peak_wset<=lim['OS_bytes'],'OS cap'
                assert log.stat().st_size<=lim['log_bytes'],'log cap'
                try:assert not psutil.Process(child.pid).children(recursive=True),'unexpected child'
                except psutil.NoSuchProcess:pass
                time.sleep(.1)
            assert child.returncode==0,log.read_text(errors='replace')[-5000:]
            for item in b['inputs']+extra:assert extent(item['path'])==item
            outputs=[] if a.audit else [extent(path) for path in sorted(a.directory.iterdir()) if path.is_file()]
            assert sum(item['bytes'] for item in outputs)+a.out.stat().st_size<=lim['output_bytes']
            if not a.audit:
                r=json.loads(a.out.read_bytes());assert r['GPU_allocated_peak']<=lim['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=lim['GPU_reserved_bytes']
            assert time.monotonic()-began<=lim['seconds'],'seal deadline'
            receipt.update(inputs_before_after_exact=True,result=extent(a.out),outputs=outputs,extra_inputs=extra)
        except BaseException as error:
            fault=repr(error)
            if child.poll() is None:child.kill();child.wait()
        finally:
            peak=max(peak,reader(child));receipt.update(exit_code=child.returncode,error=fault,elapsed_seconds=time.monotonic()-began,
                worker_OS_peak_through_exit=peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,log=extent(log))
            write(terminal,receipt)
    assert child.returncode==0 and fault is None,fault
    print(json.dumps(dict(terminal=str(terminal),seconds=receipt['elapsed_seconds'],result_sha256=receipt['result']['sha256'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('bind','audit','capture-worker','audit-worker'):p.add_argument('--'+key,action='store_true')
    for key in ('binding','out','directory','source-result','protocol'):p.add_argument('--'+key,type=Path)
    for key in ('binding-sha','freeze'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.bind:bind(a)
    elif a.capture_worker:capture(a)
    elif a.audit_worker:audit(a)
    else:launch(a)
