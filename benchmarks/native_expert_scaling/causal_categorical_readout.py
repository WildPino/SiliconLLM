"""FIT-only convex categorical adapter, fused into unchanged original C head."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];BROOT=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(BROOT))
from original_packed_capacity import SITE,extent,sha,write
from original_falcon_whole_recovery import memory_reader
from original_joint_history_recovery_audit import packed_fields
from source_cached_fit_capture import decode,save
from paired_output_codec import quality,relative
sys.path.insert(0,str(SITE))
V,K,D,N=65537,255,256,4422


def bind(a):
    import numpy as np
    import torch
    assert not a.binding.exists()
    paths={name:DOC/name for name in ('causal_coordinate_dot_bound_result_20261010.json','causal_coordinate_dot_bound_stored_adjudication_20261010.json',
        'causal_coordinate_dot_bound_binding_20261010.json','causal_readout_coordinates_result_20261010.json','causal_readout_coordinates_binding_20261010.json',
        'paired_output_codec_result_20261010.json','paired_output_codec_stored_adjudication_20261010.json','paired_output_codec_binding_20261010.json')}
    objs={name:json.loads(path.read_bytes()) for name,path in paths.items()}
    cr=objs['causal_coordinate_dot_bound_result_20261010.json'];ca=objs['causal_coordinate_dot_bound_stored_adjudication_20261010.json']
    ct=json.loads(paths['causal_coordinate_dot_bound_stored_adjudication_20261010.json'].with_suffix('.terminal.json').read_bytes())
    assert cr['decision']==ca['decision']=='CAUSAL_COORDINATES_QUALIFIED_APPROXIMATE' and all(cr['numeric_flags'].values())
    assert ca['result']==extent(paths['causal_coordinate_dot_bound_result_20261010.json']) and ct['exit_code']==0 and ct['error'] is None and ct['result']==extent(paths['causal_coordinate_dot_bound_stored_adjudication_20261010.json'])
    p=objs['paired_output_codec_result_20261010.json'];pa=objs['paired_output_codec_stored_adjudication_20261010.json'];pb=objs['paired_output_codec_binding_20261010.json']
    assert all(p['numeric_flags'].values()) and pa['result']==extent(paths['paired_output_codec_result_20261010.json'])
    cb=objs['causal_readout_coordinates_binding_20261010.json'];raw=objs['causal_readout_coordinates_result_20261010.json'];assert cr['parent_result']==extent(paths['causal_readout_coordinates_result_20261010.json'])
    source_meta_path=DOC/'source_cached_fit_binding_20261010.json';source_meta=json.loads(source_meta_path.read_bytes());source_meta_by={rec['id']:rec for rec in source_meta['records']}
    pair_by={rec['id']:rec for rec in p['records']};source_by={rec['id']:rec for rec in cb['records']};records=[]
    files=[Path(__file__),a.protocol,source_meta_path,*paths.values()]
    files += [path.with_suffix('.terminal.json') for name,path in paths.items() if 'result_' in name or 'adjudication_' in name]
    files += [BROOT/name for name in ('original_packed_capacity.py','original_falcon_whole_recovery.py','original_joint_history_recovery_audit.py',
        'source_cached_fit_capture.py','paired_output_codec.py','original_latent_readout_attribution.py','chatbot_falcon_usability.py')]
    for rec in cr['records']:
        if rec['split']!='FIT':continue
        pair=pair_by[rec['id']];source=source_by[rec['id']];qmeta=source_meta_by[rec['id']];assert pair['split']==source['split']==qmeta['split']=='FIT'
        assert pair['labels']==rec['labels']==len(rec['positions']) and pair['domain']==rec['domain']
        assert rec['positions']==source['positions']==qmeta['positions'] and rec['labels']==len(qmeta['output_ids'])
        records.append(dict(id=rec['id'],split='FIT',domain=rec['domain'],labels=rec['labels'],positions=rec['positions'],features=rec['features'],
            output_ids=qmeta['output_ids'],feature_error=rec['feature_error_L2_upper_per_label'],phi=pair['phi'],teacher=pair['source_logits']))
        files += [Path(rec['features']['path']),Path(pair['phi']['path']),Path(pair['source_logits']['path'])]
    assert len(records)==24 and sum(rec['labels'] for rec in records)==N and len({rec['domain'] for rec in records})==12
    packed=cb['packed'];assert packed_fields(packed['path'])=={name:{**field,'shape':tuple(field['shape'])} for name,field in cb['fields'].items()}
    files += [Path(packed['path']),Path(p['artifacts']['decoder']['path']),Path(sys.executable),Path(sys.executable).parent/'python312.dll']
    # Freeze actual imported Python runtime plus native array/CUDA libraries, not source weights.
    files += [Path(module.__file__) for module in list(sys.modules.values()) if getattr(module,'__file__',None) and str(SITE.resolve()).lower() in str(Path(module.__file__).resolve()).lower()]
    files += sorted((SITE/'torch/lib').glob('*.dll'))+sorted((SITE/'numpy.libs').glob('*.dll'))+sorted((SITE/'psutil').glob('*.pyd'))+[SITE/'psutil/__init__.py']
    inputs=[extent(path) for path in dict.fromkeys(files)];lookup={item['path']:item for item in inputs}
    for item in cb['inputs']+pb['inputs']:
        if item['path'] in lookup:assert lookup[item['path']]==item
    for rec in records:
        for key in ('features','phi','teacher'):assert lookup[rec[key]['path']]=={name:rec[key][name] for name in ('path','bytes','sha256')}
    write(a.binding,dict(schema='CAUSAL_CATEGORICAL_READOUT_BINDING_V1',inputs=inputs,records=records,decoder=p['artifacts']['decoder'],gain=p['calibration']['gain'],
        packed=packed,fields=cb['fields'],native_rounding=cr['calibration'],quality_gates=pb['source_label_gate'],labels=N,dimension=D,code_dimension=K,vocab=V,
        solver=dict(max_updates=32,radius=16.,max_backtracks=30,initial_L=1.,gap_tolerance=1e-5,majorant_slack=1e-10,bounds_slack=1e-8,checkpoint_every=8,
            covariance_relative_cutoff=1e-12,discarded_positive_trace_fraction=1e-8,initial_norm_bound=8.,L_rule='Nondecreasing accepted L, double on rejection',
            extrapolation='FISTA t_next=(1+sqrt(1+4t^2))/2; Y=accepted+(t-1)/t_next*(accepted-old)'),
        numeric=dict(factor_relative=1e-10,orthogonal_absolute=1e-10,whitening_relative=1e-8,initializer_absolute=1e-8,moment_absolute=1e-9,
            metric_absolute=1e-9,gradient_absolute=1e-8,bound_absolute=1e-8,fusion_absolute=1e-8,radius_slack=1e-10),
        label_chunk=64,capture_limits=dict(seconds=1200,reserve_seconds=120,OS_bytes=4<<30,GPU_allocated_bytes=4<<30,GPU_reserved_bytes=5<<30,output_bytes=1<<30,log_bytes=8<<20),
        audit_limits=dict(seconds=600,OS_bytes=4<<30,output_bytes=2<<20,log_bytes=8<<20),
        scope='New shared FIT-only convex Theta adapter and offline original packed head; no core/expert/route/source/history/native/DEV/T4 call. Approximate native features retained.'))
    print(json.dumps(dict(binding=str(a.binding),sha256=sha(a.binding),inputs=len(inputs),bytes=sum(item['bytes'] for item in inputs))),flush=True)


def case_rows(b,losses,predicted,qids,constants):
    import numpy as np
    rows=[];cursor=0
    for rec in b['records']:
        m=rec['labels'];sl=slice(cursor,cursor+m);kl=np.asarray(losses[sl]).tolist();wrong=(np.asarray(predicted[sl])!=qids[sl]).astype('i4').tolist()
        entropy=(-constants[sl]).tolist();uniform=(math.log(V)+constants[sl]).tolist()
        rows.append(dict(id=rec['id'],domain=rec['domain'],split='FIT',labels=m,KL=sum(kl)/m,disagreement=sum(wrong),disagreement_rate=sum(wrong)/m,
            teacher_entropy=sum(entropy)/m,uniform_KL=sum(uniform)/m,KL_per_label=kl,disagreement_per_label=wrong,entropy_per_label=entropy,uniform_KL_per_label=uniform))
        cursor+=m
    assert cursor==N;return rows


def range_sha(path,offset,size):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        f.seek(offset)
        while size:
            part=f.read(min(size,1<<20));assert part;digest.update(part);size-=len(part)
    return digest.hexdigest()


def export_check(b,path):
    fields=packed_fields(path);assert fields=={name:{**field,'shape':tuple(field['shape'])} for name,field in b['fields'].items()}
    head=fields['head'];size=Path(path).stat().st_size;assert size==b['packed']['bytes'];checks={}
    for name,offset,n in (('prefix',0,head['offset']),('suffix',head['offset']+head['size'],size-head['offset']-head['size'])):
        original=range_sha(b['packed']['path'],offset,n);observed=range_sha(path,offset,n);assert original==observed
        checks[name]=dict(offset=offset,bytes=n,sha256=observed)
    return dict(fields=fields,all_bytes_outside_head_exact=True,checks=checks)


def capture(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import psutil
    import torch
    assert (np.__version__,torch.__version__,psutil.__version__)==('2.4.6','2.6.0+cu124','7.2.2')
    b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1)
    torch.manual_seed(0);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.directory.mkdir(exist_ok=False);artifacts={};trace=[];checkpoints=[];stage='FIT_geometry';calls=label_evals=updates=moment_labels=0;last_checkpoint=None
    def guard():
        lim=b['capture_limits'];assert time.monotonic()-start<lim['seconds']-lim['reserve_seconds'],'worker reserve'
        assert proc.memory_info().peak_wset<=lim['OS_bytes'] and not proc.children(recursive=True)
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes']
    def event(**values):guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**values)),flush=True)
    def persist(key,x):artifacts[key]=save(a.directory,key+'.'+('f32' if x.dtype==np.float32 else 'i32' if x.dtype==np.int32 else 'f64'),np.ascontiguousarray(x));return artifacts[key]
    try:
        features=np.concatenate([np.fromfile(rec['features']['path'],dtype='<f8').reshape(rec['labels'],D) for rec in b['records']])
        warm=np.concatenate([np.fromfile(rec['phi']['path'],dtype='<f8').reshape(rec['labels'],K)/b['gain'] for rec in b['records']])
        weights=np.concatenate([np.full(rec['labels'],1/(24*rec['labels']),'f8') for rec in b['records']]);assert features.shape==(N,D) and warm.shape==(N,K)
        Z=np.sqrt(weights)[:,None]*features;U,s,VT=np.linalg.svd(Z,full_matrices=False);eigenvalues=s*s;solver=b['solver'];keep=eigenvalues>=solver['covariance_relative_cutoff']*eigenvalues[0]
        rank=int(keep.sum());assert rank>0;W=VT[keep]/s[keep,None];T=features@W.T;covariance=Z.T@Z
        discarded=float(eigenvalues[~keep].sum()/eigenvalues.sum());whitening=relative(T.T@(weights[:,None]*T),np.eye(rank))
        assert discarded<=solver['discarded_positive_trace_fraction'] and whitening<=b['numeric']['whitening_relative']
        theta0=warm.T@(weights[:,None]*T);assert np.linalg.norm(theta0)<=solver['initial_norm_bound']+b['numeric']['radius_slack']
        for key,value in (('features',features),('warm_codes',warm),('weights',weights),('covariance',covariance),('U',U),('singular_values',s),('VT',VT),('whitener',W),('whitened_features',T),('initial_theta',theta0)):persist(key,value)
        geometry=dict(rank=rank,condition_retained=float(eigenvalues[0]/eigenvalues[keep][-1]),discarded_trace_fraction=discarded,whitening_relative=whitening,
            initial_theta_norm=float(np.linalg.norm(theta0)),whitener_spectral_norm=float(1/s[keep][-1]),method='One weighted-feature thin SVD, support criterion on singular values squared')
        geometry_path=a.directory/'geometry.json';write(geometry_path,dict(geometry=geometry,artifacts=artifacts));geometry_file=extent(geometry_path)
        rawA=np.fromfile(b['decoder']['path'],dtype='<f8').reshape(V,K)*b['gain'];Ac=rawA-rawA.mean(0);A=torch.from_numpy(Ac).to('cuda')
        stage='teacher_moments';moments=np.empty((N,K),'f8');constants=np.empty(N,'f8');qids=np.empty(N,'i4');cursor=0
        with torch.inference_mode():
            for rec in b['records']:
                qs=np.memmap(rec['teacher']['path'],mode='r',dtype='<u2',shape=(rec['labels'],V))
                for first in range(0,rec['labels'],b['label_chunk']):
                    last=min(first+b['label_chunk'],rec['labels']);qvalues=torch.from_numpy(decode(qs[first:last])).to('cuda');lp=torch.log_softmax(qvalues,dim=-1);q=lp.exp()
                    sl=slice(cursor+first,cursor+last);moments[sl]=(q@A).cpu().numpy();constants[sl]=(q*lp).sum(-1).cpu().numpy();qids[sl]=qvalues.argmax(-1).cpu().numpy()
                    moment_labels+=last-first;guard()
                cursor+=rec['labels'];del qs
        assert cursor==moment_labels==N
        assert qids.tolist()==[identifier for rec in b['records'] for identifier in rec['output_ids']]
        for key,value in (('teacher_moments',moments),('teacher_constants',constants),('teacher_argmax',qids)):persist(key,value)
        tt=torch.from_numpy(T).to('cuda');mm=torch.from_numpy(moments).to('cuda');cc=torch.from_numpy(constants).to('cuda');ww=torch.from_numpy(weights).to('cuda')
        def fg(theta):
            nonlocal calls,label_evals
            calls+=1;total=torch.zeros((),device='cuda',dtype=torch.float64);gradient=torch.zeros_like(theta);losses=[];predictions=[]
            for first in range(0,N,b['label_chunk']):
                last=min(first+b['label_chunk'],N);sl=slice(first,last);x=tt[sl]@theta.T;z=x@A.T;lp=torch.log_softmax(z,dim=-1)
                loss=cc[sl]+torch.logsumexp(z,dim=-1)-(mm[sl]*x).sum(-1);dx=lp.exp()@A-mm[sl]
                total+=(ww[sl]*loss).sum();gradient+=(ww[sl,None]*dx).T@tt[sl];losses.append(loss.cpu().numpy());predictions.append(z.argmax(-1).cpu().numpy())
                label_evals+=last-first;guard()
            assert torch.isfinite(total) and torch.isfinite(gradient).all();return float(total),gradient,np.concatenate(losses),np.concatenate(predictions)
        stage='shared_descent';theta=torch.from_numpy(theta0).to('cuda');Y=theta.clone();best=theta.clone();lower_point=theta.clone();lower=0.;upper=math.inf;L=solver['initial_L'];accel=1.
        best_gradient=lower_gradient=None;best_losses=best_ids=None;lower_value=None;initial_result=None
        def observe(point,value,gradient,losses,ids):
            nonlocal lower,upper,best,best_gradient,best_losses,best_ids,lower_point,lower_gradient,lower_value
            if float(point.norm())<=solver['radius']+b['numeric']['radius_slack'] and value<upper:
                upper=value;best=point.clone();best_gradient=gradient.clone();best_losses=losses.copy();best_ids=ids.copy()
            candidate=value-float((gradient*point).sum())-solver['radius']*float(gradient.norm())
            if lower_gradient is None or candidate>lower:
                lower_point=point.clone();lower_gradient=gradient.clone();lower_value=value
            lower=max(lower,candidate,0.);assert lower<=upper+solver['bounds_slack']
        with torch.inference_mode():
            for updates in range(solver['max_updates']+1):
                FY,gY,lossY,idY=fg(Y);observe(Y,FY,gY,lossY,idY)
                if updates==0:initial_result=dict(value=FY,losses=lossY.tolist(),predictions=idY.tolist())
                if updates%solver['checkpoint_every']==0 or upper-lower<=solver['gap_tolerance'] or updates==solver['max_updates']:
                    checkpoint=a.directory/('checkpoint_%04d.npz'%updates)
                    with checkpoint.open('xb') as f:
                        np.savez(f,theta=theta.cpu().numpy(),Y=Y.cpu().numpy(),FY=FY,gY=gY.cpu().numpy(),best_theta=best.cpu().numpy(),upper=upper,best_gradient=best_gradient.cpu().numpy(),
                            lower_theta=lower_point.cpu().numpy(),lower=lower,lower_value=lower_value,lower_gradient=lower_gradient.cpu().numpy(),L=L,acceleration=accel)
                        f.flush();os.fsync(f.fileno())
                    last_checkpoint=extent(checkpoint);checkpoints.append(last_checkpoint);row=dict(updates=updates,objective_gradient_calls=calls,upper=upper,lower=lower,gap=max(0.,upper-lower),L=L)
                    trace.append(row);event(**row)
                if upper-lower<=solver['gap_tolerance'] or updates==solver['max_updates']:break
                for backtrack in range(solver['max_backtracks']):
                    candidate=Y-gY/L;candidate*=min(1.,solver['radius']/max(float(candidate.norm()),1e-300));delta=candidate-Y
                    FC,gC,lossC,idC=fg(candidate);observe(candidate,FC,gC,lossC,idC)
                    majorant=FY+float((gY*delta).sum())+.5*L*float((delta*delta).sum())
                    if FC<=majorant+solver['majorant_slack']:break
                    L*=2
                else:raise RuntimeError('backtracking cap')
                next_accel=(1+math.sqrt(1+4*accel*accel))/2;Y=candidate+(accel-1)/next_accel*(candidate-theta);theta=candidate;accel=next_accel
            stage='final_certificates';FU,gU,lossU,idU=fg(best);FL,gL,lossL,idL=fg(lower_point)
            assert abs(FU-upper)<=b['numeric']['metric_absolute'] and abs(max(0.,FL-float((gL*lower_point).sum())-solver['radius']*float(gL.norm()))-lower)<=b['numeric']['bound_absolute']
            for key,value in (('best_theta',best.cpu().numpy()),('best_gradient',gU.cpu().numpy()),('lower_theta',lower_point.cpu().numpy()),('lower_gradient',gL.cpu().numpy())):persist(key,value)
        rows=case_rows(b,lossU,idU,qids,constants);initial_rows=case_rows(b,np.asarray(initial_result['losses']),np.asarray(initial_result['predictions']),qids,constants)
        aggregate,flags=quality(rows,b['quality_gates']);initial_aggregate,initial_flags=quality(initial_rows,b['quality_gates'])
        stage='offline_fusion';J=best.cpu().numpy()@W;head64=rawA@J;head32=head64.astype('<f4');assert np.isfinite(head32).all()
        for key,value in (('adapter',J),('head_real',head64),('head_native',head32)):persist(key,value)
        packed_path=a.directory/'candidate_categorical.packed'
        with Path(b['packed']['path']).open('rb') as source,packed_path.open('xb') as dest:shutil.copyfileobj(source,dest,1<<20);dest.flush();os.fsync(dest.fileno())
        with packed_path.open('r+b') as dest:dest.seek(b['fields']['head']['offset']);dest.write(head32.tobytes());dest.flush();os.fsync(dest.fileno())
        exported=export_check(b,packed_path);artifacts['packed']=extent(packed_path)
        feature_error=np.concatenate([np.asarray(rec['feature_error']) for rec in b['records']]);feature_norm=np.linalg.norm(features,axis=1)
        row_norm64=float(np.linalg.norm(head64,axis=1).max());row_norm32=float(np.linalg.norm(head32.astype('f8'),axis=1).max());row_quant_error=float(np.linalg.norm(head32.astype('f8')-head64,axis=1).max())
        sensitivity=feature_error*row_norm64+(feature_norm+feature_error)*(row_quant_error+b['native_rounding']['F32_gamma_dot']*row_norm32)
        result=dict(schema='CAUSAL_CATEGORICAL_READOUT_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,geometry=geometry,geometry_file=geometry_file,artifacts=artifacts,
            initial=initial_result,initial_records=initial_rows,initial_aggregate=initial_aggregate,initial_quality_flags=initial_flags,
            records=rows,aggregate=aggregate,quality_flags=flags,decision='SHARED_HEAD_FIT_PASS_PENDING_NATIVE_DEV' if all(flags.values()) else 'SHARED_HEAD_FIT_QUALITY_FAIL',
            upper=FU,lower=lower,lower_point_value=FL,gap=max(0.,FU-lower),theta_norm=float(best.norm()),lower_point_norm=float(lower_point.norm()),
            updates=updates,objective_gradient_calls=calls,objective_gradient_label_evaluations=label_evals,teacher_moment_labels=moment_labels,
            shared_theta_coefficients=K*rank,head_pairs_fit=1,weighted_feature_SVD_factorizations=1,native_export=exported,
            head_row_norm_F64_max=row_norm64,head_row_norm_F32_max=row_norm32,head_quantization_row_L2_max=row_quant_error,
            forced_native_score_uncertainty_per_label=sensitivity.tolist(),forced_native_KL_uncertainty_per_label=(2*sensitivity).tolist(),
            forced_uncertainty_scope='Same teacher-forced core/final norm; old feature bound plus new head cast and original39-path dot. F64 arithmetic model, no interval or own-history claim.',
            trace=trace,checkpoints=checkpoints,native_calls=0,source_generations=0,source_history_forwards=0,core_parameter_updates=0,expert_parameter_updates=0,DEV_queries=0,reserved_queries=0,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            seconds=time.monotonic()-start,quality_admission=False,speed_admission=False,
            scope='FIT-only shared bounded categorical map and actual packed head artifact, original109 other fields exact; no native evaluation or causal/source/GPU core training.')
        guard();write(a.out,result);event(decision=result['decision'],upper=FU,lower=lower,case_KL=aggregate['case_KL'])
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),accepted_updates=updates,objective_gradient_calls=calls,objective_gradient_label_evaluations=label_evals,
            teacher_moment_labels=moment_labels,last_checkpoint=last_checkpoint,artifacts=artifacts,seconds=time.monotonic()-start));raise


def audit(a):
    import numpy as np
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes())
    assert sha(a.binding)==a.binding_sha==r['binding_sha256']==t['binding_sha256'] and a.freeze==r['freeze']==t['freeze']
    assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(a.source_result)
    def guard():
        assert time.monotonic()-start<b['audit_limits']['seconds'] and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes'] and not proc.children(recursive=True)
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item;guard()
    outputs={item['path']:item for item in t['outputs']};g=b['numeric'];art=r['artifacts'];max_metric=max_grad=max_moment=max_entropy=0.;score_rows=fg_calls=0
    def array(item,shape,dtype='<f8'):
        assert outputs[item['path']]=={key:item[key] for key in ('path','bytes','sha256')} and item['shape']==list(shape)
        x=np.fromfile(item['path'],dtype=dtype).reshape(shape);assert np.isfinite(x).all();return x
    def close(x,y,tolerance):
        nonlocal max_metric
        delta=float(np.max(abs(np.asarray(x)-np.asarray(y)),initial=0));max_metric=max(max_metric,delta);assert delta<=tolerance;return delta
    f=array(art['features'],(N,D));warm=array(art['warm_codes'],(N,K));weights=array(art['weights'],(N,))
    expected_f=np.concatenate([np.fromfile(rec['features']['path'],dtype='<f8').reshape(rec['labels'],D) for rec in b['records']]);assert np.array_equal(f,expected_f)
    assert np.array_equal(warm,np.concatenate([np.fromfile(rec['phi']['path'],dtype='<f8').reshape(rec['labels'],K)/b['gain'] for rec in b['records']]))
    assert np.array_equal(weights,np.concatenate([np.full(rec['labels'],1/(24*rec['labels']),'f8') for rec in b['records']]))
    U=array(art['U'],(N,D));s=array(art['singular_values'],(D,));VT=array(art['VT'],(D,D));C=array(art['covariance'],(D,D))
    assert np.all(s>0) and np.all(np.diff(s)<=0);Z=np.sqrt(weights)[:,None]*f
    factor_relative=relative((U*s)@VT,Z);assert factor_relative<=g['factor_relative']
    orth=max(float(np.linalg.norm(U.T@U-np.eye(D))),float(np.linalg.norm(VT@VT.T-np.eye(D))));assert orth<=g['orthogonal_absolute']
    assert relative(C,Z.T@Z)<=g['factor_relative'];keep=s*s>=b['solver']['covariance_relative_cutoff']*s[0]*s[0];rank=int(keep.sum());assert rank==r['geometry']['rank']
    W=array(art['whitener'],(rank,D));T=array(art['whitened_features'],(N,rank));theta0=array(art['initial_theta'],(K,rank))
    assert relative(W,VT[keep]/s[keep,None])<=g['factor_relative'] and relative(T,f@W.T)<=g['factor_relative']
    whitening=relative(T.T@(weights[:,None]*T),np.eye(rank));discarded=float((s*s)[~keep].sum()/(s*s).sum())
    assert whitening<=g['whitening_relative'] and discarded<=b['solver']['discarded_positive_trace_fraction']
    close(theta0,warm.T@(weights[:,None]*T),g['initializer_absolute']);assert np.linalg.norm(theta0)<=b['solver']['initial_norm_bound']+g['radius_slack']
    for key,value in (('condition_retained',float(s[0]*s[0]/(s*s)[keep][-1])),('discarded_trace_fraction',discarded),('whitening_relative',whitening),
        ('initial_theta_norm',float(np.linalg.norm(theta0))),('whitener_spectral_norm',float(1/s[keep][-1]))):close(value,r['geometry'][key],g['initializer_absolute'])
    geometry_json=json.loads(Path(r['geometry_file']['path']).read_bytes());assert outputs[r['geometry_file']['path']]==r['geometry_file'] and geometry_json['geometry']==r['geometry']
    for key,value in geometry_json['artifacts'].items():assert value==art[key]
    rawA=np.fromfile(b['decoder']['path'],dtype='<f8').reshape(V,K)*b['gain'];A=rawA-rawA.mean(0)
    moments=array(art['teacher_moments'],(N,K));constants=array(art['teacher_constants'],(N,));qids=array(art['teacher_argmax'],(N,),'<i4');cursor=0
    # Independent all-V teacher normalization/moment/entropy checks; no teacher model call.
    for rec in b['records']:
        qb=np.memmap(rec['teacher']['path'],mode='r',dtype='<u2',shape=(rec['labels'],V))
        for first in range(0,rec['labels'],b['label_chunk']):
            last=min(first+b['label_chunk'],rec['labels']);q=decode(qb[first:last]);lp=q-q.max(1,keepdims=True);lp-=np.logaddexp.reduce(lp,axis=1)[:,None];qp=np.exp(lp);sl=slice(cursor+first,cursor+last)
            max_moment=max(max_moment,float(np.max(abs(qp@A-moments[sl]))));max_entropy=max(max_entropy,float(np.max(abs(np.sum(qp*lp,axis=1)-constants[sl]))))
            assert np.array_equal(q.argmax(1),qids[sl]) and q.argmax(1).tolist()==rec['output_ids'][first:last];guard()
        cursor+=rec['labels'];del qb,q,lp,qp
    assert cursor==N and max_moment<=g['moment_absolute'] and max_entropy<=g['metric_absolute']
    def fg(theta):
        nonlocal fg_calls,score_rows
        value=0.;gradient=np.zeros_like(theta);losses=[];predicted=[];dxall=[]
        for first in range(0,N,b['label_chunk']):
            last=min(first+b['label_chunk'],N);sl=slice(first,last);x=T[sl]@theta.T;z=x@A.T;zmax=z.max(1,keepdims=True)
            shifted=z-zmax;psum=np.exp(shifted).sum(1,keepdims=True);p=np.exp(shifted)/psum;loss=constants[sl]+zmax[:,0]+np.log(psum[:,0])-np.sum(moments[sl]*x,axis=1)
            dx=p@A-moments[sl];value+=float(np.sum(weights[sl]*loss));gradient+=(weights[sl,None]*dx).T@T[sl]
            losses.extend(loss.tolist());predicted.extend(z.argmax(1).tolist());dxall.append(dx);score_rows+=last-first;guard()
        fg_calls+=1;return value,gradient,np.asarray(losses),np.asarray(predicted),np.concatenate(dxall)
    initial_value,initial_gradient,initial_loss,initial_id,initial_dx=fg(theta0)
    close(initial_value,r['initial']['value'],g['metric_absolute']);close(initial_loss,r['initial']['losses'],g['metric_absolute']);assert initial_id.tolist()==r['initial']['predictions']
    gradient_witness_delta=0.;gradient_witnesses=0
    for k,j in ((0,0),(17,rank//2),(127,rank-1),(254,rank//3)):
        witness=math.fsum(float(weights[i])*float(initial_dx[i,k])*float(T[i,j]) for i in range(N))
        gradient_witness_delta=max(gradient_witness_delta,abs(witness-float(initial_gradient[k,j])));gradient_witnesses+=1
    assert gradient_witness_delta<=g['gradient_absolute'];del initial_dx
    best=array(art['best_theta'],(K,rank));lower_point=array(art['lower_theta'],(K,rank));best_gradient=array(art['best_gradient'],(K,rank));lower_gradient=array(art['lower_gradient'],(K,rank))
    FV,gV,lossV,idV,_=fg(best);FL,gL,lossL,idL,_=fg(lower_point)
    max_grad=max(float(np.max(abs(gV-best_gradient))),float(np.max(abs(gL-lower_gradient))));assert max_grad<=g['gradient_absolute']
    bound=max(0.,FL-float(np.sum(gL*lower_point))-b['solver']['radius']*float(np.linalg.norm(gL)))
    close(FV,r['upper'],g['metric_absolute']);close(FL,r['lower_point_value'],g['metric_absolute']);close(bound,r['lower'],g['bound_absolute']);close(max(0.,FV-bound),r['gap'],g['bound_absolute'])
    close(np.linalg.norm(best),r['theta_norm'],g['metric_absolute']);close(np.linalg.norm(lower_point),r['lower_point_norm'],g['metric_absolute']);assert np.linalg.norm(best)<=b['solver']['radius']+g['radius_slack']
    # Each checkpoint is fully hashed and its saved Y objective/gradient recomputed.
    previous_upper=initial_value;previous_lower=0.;previous_L=0.;assert len(r['trace'])==len(r['checkpoints']) and r['trace'][0]['updates']==0 and r['trace'][-1]['updates']==r['updates']
    for item,row in zip(r['checkpoints'],r['trace']):
        assert outputs[item['path']]==item
        with np.load(item['path'],allow_pickle=False) as cp:
            assert set(cp.files)=={'theta','Y','FY','gY','best_theta','upper','best_gradient','lower_theta','lower','lower_value','lower_gradient','L','acceleration'}
            for key in ('theta','Y','gY','best_theta','best_gradient','lower_theta','lower_gradient'):assert cp[key].shape==(K,rank) and np.isfinite(cp[key]).all()
            FY,gY,_,_,_=fg(cp['Y']);close(FY,float(cp['FY']),g['metric_absolute']);delta=float(np.max(abs(gY-cp['gY'])));max_grad=max(max_grad,delta);assert delta<=g['gradient_absolute']
            assert np.linalg.norm(cp['theta'])<=b['solver']['radius']+g['radius_slack'] and np.linalg.norm(cp['best_theta'])<=b['solver']['radius']+g['radius_slack']
            assert float(cp['upper'])<=previous_upper+g['metric_absolute'] and float(cp['lower'])>=previous_lower-g['bound_absolute'] and float(cp['L'])>=previous_L
            assert float(cp['lower'])<=float(cp['upper'])+b['solver']['bounds_slack'] and float(cp['acceleration'])>=1
            for key in ('upper','lower','L'):close(float(cp[key]),row[key],g['bound_absolute'])
            close(max(0.,float(cp['upper'])-float(cp['lower'])),row['gap'],g['bound_absolute']);previous_upper=float(cp['upper']);previous_lower=float(cp['lower']);previous_L=float(cp['L'])
        print(json.dumps(dict(stage='checkpoint_audited',updates=row['updates'],seconds=time.monotonic()-start)),flush=True)
    close(previous_upper,FV,g['metric_absolute']);close(previous_lower,bound,g['bound_absolute'])
    actual_rows=case_rows(b,lossV,idV,qids,constants);actual_initial=case_rows(b,initial_loss,initial_id,qids,constants)
    for actual,saved in zip(actual_rows,r['records']):
        assert (actual['id'],actual['domain'],actual['labels'],actual['disagreement'],actual['disagreement_per_label'])==(saved['id'],saved['domain'],saved['labels'],saved['disagreement'],saved['disagreement_per_label'])
        for key in ('KL','KL_per_label','teacher_entropy','uniform_KL','entropy_per_label','uniform_KL_per_label'):close(actual[key],saved[key],g['metric_absolute'])
    for actual,saved in zip(actual_initial,r['initial_records']):close(actual['KL_per_label'],saved['KL_per_label'],g['metric_absolute']);assert actual['disagreement_per_label']==saved['disagreement_per_label']
    saved_aggregate,saved_flags=quality(r['records'],b['quality_gates']);initial_agg,initial_flags=quality(r['initial_records'],b['quality_gates'])
    assert saved_aggregate==r['aggregate'] and saved_flags==r['quality_flags'] and initial_agg==r['initial_aggregate'] and initial_flags==r['initial_quality_flags']
    actual_agg,_=quality(actual_rows,b['quality_gates']);close(actual_agg['case_KL'],r['aggregate']['case_KL'],g['metric_absolute'])
    outcome='SHARED_HEAD_FIT_PASS_PENDING_NATIVE_DEV' if all(saved_flags.values()) else 'SHARED_HEAD_FIT_QUALITY_FAIL';assert outcome==r['decision']
    J=array(art['adapter'],(K,D));head64=array(art['head_real'],(V,D));head32=array(art['head_native'],(V,D),'<f4')
    close(J,best@W,g['fusion_absolute']);fusion_delta=float(np.max(abs(head64-rawA@J)));assert fusion_delta<=g['fusion_absolute'] and np.array_equal(head32,head64.astype('<f4'))
    assert outputs[art['packed']['path']]==art['packed'];exported=export_check(b,art['packed']['path'])
    # JSON converts tuples to lists; compare canonical decoded field metadata.
    assert json.loads(json.dumps(exported))==r['native_export']
    with Path(art['packed']['path']).open('rb') as pf:pf.seek(b['fields']['head']['offset']);packed_head=np.frombuffer(pf.read(b['fields']['head']['size']),dtype='<f4').reshape(V,D)
    assert np.array_equal(packed_head,head32)
    feature_error=np.concatenate([np.asarray(rec['feature_error']) for rec in b['records']]);feature_norm=np.linalg.norm(f,axis=1)
    h64=float(np.linalg.norm(head64,axis=1).max());h32=float(np.linalg.norm(head32.astype('f8'),axis=1).max());quant=float(np.linalg.norm(head32.astype('f8')-head64,axis=1).max())
    sensitivity=feature_error*h64+(feature_norm+feature_error)*(quant+b['native_rounding']['F32_gamma_dot']*h32)
    for actual,saved in ((h64,r['head_row_norm_F64_max']),(h32,r['head_row_norm_F32_max']),(quant,r['head_quantization_row_L2_max'])):close(actual,saved,g['metric_absolute'])
    close(sensitivity,r['forced_native_score_uncertainty_per_label'],g['bound_absolute']);close(2*sensitivity,r['forced_native_KL_uncertainty_per_label'],g['bound_absolute'])
    assert r['objective_gradient_calls']==r['trace'][-1]['objective_gradient_calls']+2 and r['objective_gradient_label_evaluations']==N*r['objective_gradient_calls']
    assert r['shared_theta_coefficients']==K*rank and r['head_pairs_fit']==r['weighted_feature_SVD_factorizations']==1 and r['teacher_moment_labels']==N
    assert r['updates']<=b['solver']['max_updates']
    for key in ('native_calls','source_generations','source_history_forwards','core_parameter_updates','expert_parameter_updates','DEV_queries','reserved_queries'):assert r[key]==0
    assert not r['quality_admission'] and not r['speed_admission']
    assert t['elapsed_seconds']<=b['capture_limits']['seconds'] and t['worker_OS_peak_through_exit']+t['launcher_OS_peak_snapshot']<=b['capture_limits']['OS_bytes']
    assert r['GPU_allocated_peak']<=b['capture_limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=b['capture_limits']['GPU_reserved_bytes']
    assert sum(item['bytes'] for item in t['outputs'])+t['result']['bytes']<=b['capture_limits']['output_bytes'];guard()
    write(a.out,dict(schema='CAUSAL_CATEGORICAL_READOUT_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision=outcome,quality_flags=saved_flags,
        complete_input_output_hashes=True,all_FIT_teacher_moments_entropy_argmax_verified=True,full_weighted_SVD_whitening_initializer_verified=True,
        factor_relative=factor_relative,orthogonal_Frobenius=orth,max_teacher_moment_absolute_delta=max_moment,max_teacher_entropy_absolute_delta=max_entropy,
        max_metric_absolute_delta=max_metric,max_gradient_absolute_delta=max_grad,scalar_gradient_witnesses=gradient_witnesses,scalar_gradient_absolute_delta=gradient_witness_delta,
        all_checkpoint_Y_objectives_gradients_verified=True,final_feasible_upper_and_tangent_lower_verified=True,full_fusion_absolute_delta=fusion_delta,
        packed_all_bytes_except_head_exact=True,forced_native_uncertainty_bounds_verified=True,
        stored_objective_gradient_evaluations=fg_calls,stored_linear_head_labels_reconstructed=score_rows,
        source_calls=0,native_calls=0,optimizer_updates=0,SVD_factorizations=0,seconds=time.monotonic()-start,
        scope='Complete CPU stored audit, no descent/SVD/model/history replay; proxy FIT quality never native/DEV admission.'))
    print(json.dumps(dict(audit=str(a.out),decision=outcome,seconds=time.monotonic()-start)),flush=True)


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
