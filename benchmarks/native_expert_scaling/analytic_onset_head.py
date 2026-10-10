"""One closed-form first-code correction, offline original head, all stored labels."""
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write
from categorical_onset_control import save,load_array,norm,relative
from causal_categorical_readout import export_check
sys.path.insert(0,str(SITE))
V,D,K=65537,256,255

def bind(a):
    import numpy as np
    import psutil
    assert not a.binding.exists()
    names=['categorical_onset_control_result_20261010.json','categorical_onset_control_stored_adjudication_20261010.json','categorical_onset_control_runtime_binding_20261010.json','causal_categorical_readout_binding_20261010.json','causal_categorical_readout_result_20261010.json','causal_coordinate_dot_bound_result_20261010.json','categorical_native_assessment_binding_20261010.json']
    paths=[DOC/name for name in names];r,ar,rb,cb,shared,cr,nb=[json.loads(p.read_bytes()) for p in paths]
    assert r['decision']==ar['decision']=='FIRST_CODE_MATCH_ROBUSTLY_OUTSIDE_RADIUS' and all(r['numeric_flags'].values()) and ar['complete_input_output_hashes'] and ar['minimum_maps_projection_dual_verified']
    assert ar['result']==extent(paths[0]) and ar['binding']==extent(paths[2]);assert cr['decision']=='CAUSAL_COORDINATES_QUALIFIED_APPROXIMATE'
    terminals=[paths[0].with_suffix('.terminal.json'),paths[1].with_suffix('.terminal.json')]
    for p,source in zip(terminals,[paths[0],paths[1]]):
        t=json.loads(p.read_bytes());assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(source)
    coords={rec['id']:rec for rec in cr['records']};records=[]
    for rec in nb['records']:
        c=coords[rec['id']];assert c['positions']==rec['positions'] and c['labels']==len(rec['positions']) and c['split']==rec['split'];records.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],labels=c['labels'],positions=rec['positions'],features=c['features'],teacher=rec['logits'],feature_error=c['feature_error_L2_upper_per_label']))
    assert len(records)==48 and sum(x['labels'] for x in records)==8808
    files=[Path(__file__),a.protocol,*paths,*terminals,B/'categorical_onset_control.py',B/'categorical_onset_control_launch.py',B/'causal_categorical_readout.py',B/'original_packed_capacity.py',B/'original_falcon_whole_recovery.py',B/'original_joint_history_recovery_audit.py',B/'source_cached_fit_capture.py',B/'paired_output_codec.py',B/'original_latent_readout_attribution.py',B/'chatbot_falcon_usability.py',Path(shared['artifacts']['packed']['path']),Path(r['artifacts']['preserved_theta']['path']),Path(cb['artifacts']['whitener']['path']) if 'artifacts' in cb else Path(shared['artifacts']['whitener']['path']),Path(cb['decoder']['path'])]
    files += [Path(rec[key]['path']) for rec in records for key in ['features','teacher']]
    files += [Path(item['path']) for item in rb['inputs'] if any(part in item['path'].lower() for part in ['site-packages','python312.dll','python.exe'])]
    inputs=[extent(p) for p in dict.fromkeys(files)];lookup={item['path']:item for item in inputs}
    for item in rb['inputs']+nb['inputs']:
        if item['path'] in lookup:assert lookup[item['path']]==item
    for rec in records:
        for key in ['features','teacher']:assert lookup[rec[key]['path']]=={k:rec[key][k] for k in ('path','bytes','sha256')}
    write(a.binding,dict(schema='ANALYTIC_ONSET_HEAD_BINDING_V1',inputs=inputs,records=records,theta=r['artifacts']['preserved_theta'],W=shared['artifacts']['whitener'],decoder=cb['decoder'],gain=cb['gain'],packed=shared['artifacts']['packed'],fields=cb['fields'],native_rounding=cr['calibration'],strict_gates=cb['quality_gates'],native_gates=nb['gates'],selected=r['selected'],reference=r['metrics'],
        capture_limits=dict(seconds=300,reserve_seconds=30,OS_bytes=2<<30,output_bytes=1<<30,log_bytes=4<<20),audit_limits=dict(seconds=300,OS_bytes=2<<30,output_bytes=1<<20,log_bytes=4<<20),chunk=16,
        numeric=dict(fusion_absolute=1e-8,metric_absolute=1e-9,bound_absolute=1e-8),scope='One closed-form radius-unconstrained code correction, ordinary original F32 head export and all8808 stored forced-label probability products. No optimizer/history/source/native/GPU/DEV fitting/T4. DEV only diagnostic; no chatbot/speed admission.'))
    print(json.dumps(dict(binding_sha256=sha(a.binding),inputs=len(inputs),bytes=sum(item['bytes'] for item in inputs))),flush=True)

def gates(rows,b):
    result={}
    for split in ['FIT','DEV']:
        chosen=[r for r in rows if r['split']==split];mkl=sum(r['KL'] for r in chosen)/24;dis=sum(r['disagreement_rate'] for r in chosen)/24;g=b['strict_gates']
        domain={d:dict(KL=sum(r['KL'] for r in chosen if r['domain']==d)/2,disagreement=sum(r['disagreement_rate'] for r in chosen if r['domain']==d)/2) for d in sorted({r['domain'] for r in chosen})}
        result[split]=dict(case_KL=mkl,case_disagreement=dis,domains=domain,first_case_KL=sum(r['KL_per_label'][0] for r in chosen)/24,first_disagreement=sum(r['disagreement_per_label'][0] for r in chosen),continuation_case_KL=sum(sum(r['KL_per_label'][1:])/(r['labels']-1) for r in chosen)/24,
            strict_flags=dict(mean_KL=mkl<=g['mean_KL'],mean_disagreement=dis<=g['mean_disagreement'],every_case_KL=all(r['KL']<=g['case_KL'] for r in chosen),every_case_disagreement=all(r['disagreement_rate']<=g['case_disagreement'] for r in chosen),domains=all(r['KL']<=g['case_KL'] and r['disagreement']<=g['case_disagreement'] for r in domain.values())),
            original_flags=dict(mean_KL=mkl<=1.,mean_disagreement=dis<=.20,domain_KL=all(r['KL']<=2 for r in domain.values()),domain_disagreement=all(r['disagreement']<=.35 for r in domain.values())))
    return result

def rows_for(b,H,norms64,h32,quant,independent,guard):
    import numpy as np
    rows=[];count=0
    for rec in b['records']:
        f=load_array(rec['features']);qbits=np.memmap(rec['teacher']['path'],mode='r',dtype='<u2',shape=(rec['labels'],V));kl=[];wrong=[];entropy=[];uncertainty=[];predicted=[]
        for first in range(0,rec['labels'],b['chunk']):
            last=min(first+b['chunk'],rec['labels']);z=f[first:last]@H.T;q=(qbits[first:last].astype('<u4')<<16).view('<f4').astype('f8');assert np.isfinite(z).all() and np.isfinite(q).all()
            if independent:lq=q-np.logaddexp.reduce(q,axis=1)[:,None];lp=z-np.logaddexp.reduce(z,axis=1)[:,None]
            else:
                lq=q-q.max(1,keepdims=True);lp=z-z.max(1,keepdims=True);lq-=np.log(np.exp(lq).sum(1,keepdims=True));lp-=np.log(np.exp(lp).sum(1,keepdims=True))
            pq=np.exp(lq);loss=np.sum(pq*(lq-lp),axis=1);ent=-np.sum(pq*lq,axis=1);ids=z.argmax(1);dis=(ids!=q.argmax(1)).astype('i4');e=np.asarray(rec['feature_error'][first:last]);epsilon=e*norms64+(np.linalg.norm(f[first:last],axis=1)+e)*(quant+b['native_rounding']['F32_gamma_dot']*h32)
            kl.extend(loss.tolist());wrong.extend(dis.tolist());entropy.extend(ent.tolist());predicted.extend(ids.tolist());uncertainty.extend(epsilon.tolist());count+=last-first;guard()
        row=dict(id=rec['id'],split=rec['split'],domain=rec['domain'],labels=rec['labels'],KL=sum(kl)/rec['labels'],disagreement=sum(wrong),disagreement_rate=sum(wrong)/rec['labels'],KL_per_label=kl,disagreement_per_label=wrong,predicted_ids=predicted,entropy_per_label=entropy,score_uncertainty_per_label=uncertainty)
        rows.append(row);print(json.dumps(dict(stage='stored_probability_case',id=rec['id'],KL=row['KL'],first_KL=kl[0],count=count)),flush=True)
    assert count==8808;return rows,count

def capture(a):
    import numpy as np
    import psutil
    import shutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([1]);b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha;a.directory.mkdir(exist_ok=False);stage='fusion';art={}
    def guard():
        assert time.monotonic()-start<b['capture_limits']['seconds']-b['capture_limits']['reserve_seconds'] and proc.memory_info().peak_wset<=b['capture_limits']['OS_bytes'] and not proc.children(recursive=True)
    try:
        theta=load_array(b['theta']);W=load_array(b['W']);rawA=load_array(b['decoder'])*b['gain'];J=theta@W;H=rawA@J;H32=H.astype('<f4');assert np.isfinite(H32).all()
        for key,x in [('theta',theta),('adapter',J),('head_real',H)]:art[key]=save(a.directory,key,x)
        path=a.directory/'head_native.f32'
        with path.open('xb') as f:f.write(H32.tobytes());f.flush();os.fsync(f.fileno())
        art['head_native']=dict(**extent(path),shape=[V,D],dtype='float32')
        packed=a.directory/'candidate_analytic_onset.packed'
        with Path(b['packed']['path']).open('rb') as source,packed.open('xb') as dest:shutil.copyfileobj(source,dest,1<<20);dest.flush();os.fsync(dest.fileno())
        with packed.open('r+b') as f:f.seek(b['fields']['head']['offset']);f.write(H32.tobytes());f.flush();os.fsync(f.fileno())
        export=export_check(b,packed);art['packed']=extent(packed);h64=float(np.linalg.norm(H,axis=1).max());h32=float(np.linalg.norm(H32.astype('f8'),axis=1).max());quant=float(np.linalg.norm(H32.astype('f8')-H,axis=1).max());guard()
        stage='all_stored_labels';rows,count=rows_for(b,H,h64,h32,quant,False,guard);aggregates=gates(rows,b)
        first_reference={rec['id']:rec for rec in b['selected']};first_delta=0.
        for row in rows:
            if row['split']=='FIT':first_delta=max(first_delta,abs(row['KL_per_label'][0]-first_reference[row['id']]['reference_KL']))
        assert first_delta<=b['numeric']['metric_absolute'] and aggregates['FIT']['first_disagreement']==0
        decision='ANALYTIC_FIRST_CORRECTION_PROXY_PASS_PENDING_NATIVE' if all(flag for split in aggregates.values() for flag in split['strict_flags'].values()) else 'ANALYTIC_FIRST_CORRECTION_PROXY_QUALITY_FAIL'
        result=dict(schema='ANALYTIC_ONSET_HEAD_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,artifacts=art,records=rows,aggregate=aggregates,decision=decision,export=export,theta_norm=norm(theta),parent_fit_radius=16.,radius_constraint_removed=True,head_row_norm_F64_max=h64,head_row_norm_F32_max=h32,head_cast_row_error_max=quant,first_reference_max_KL_delta=first_delta,stored_head_label_products=count,analytic_head_pairs=1,source_calls=0,native_calls=0,history_forwards=0,optimizer_updates=0,head_FG_calls=0,GPU_calls=0,reserved_queries=0,SVD_calls=0,quality_admission=False,speed_admission=False,OS_peak=proc.memory_info().peak_wset,seconds=time.monotonic()-start,scope=b['scope'])
        guard();write(a.out,result);assert sum(p.stat().st_size for p in a.directory.iterdir())+a.out.stat().st_size<=b['capture_limits']['output_bytes'];print(json.dumps(dict(decision=decision,aggregate=aggregates,seconds=result['seconds'])),flush=True)
    except BaseException as error:write(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),artifacts=art,seconds=time.monotonic()-start));raise

def audit(a):
    import numpy as np
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([1]);b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes());assert sha(a.binding)==a.binding_sha==r['binding_sha256']==t['binding_sha256'] and a.freeze==r['freeze']==t['freeze'];assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(a.source_result)
    def guard():assert time.monotonic()-start<b['audit_limits']['seconds'] and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes'] and not proc.children(recursive=True)
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item;guard()
    art=r['artifacts'];by={item['path']:item for item in t['outputs']}
    for key,item in art.items():assert by[item['path']]=={k:item[k] for k in ('path','bytes','sha256')}
    theta=load_array(art['theta']);J=load_array(art['adapter']);H=load_array(art['head_real']);H32=np.fromfile(art['head_native']['path'],dtype='<f4').reshape(V,D);assert np.array_equal(theta,load_array(b['theta']))
    deltaJ=float(np.max(abs(J-theta@load_array(b['W']))));deltaH=float(np.max(abs(H-load_array(b['decoder'])*b['gain']@J)));assert max(deltaJ,deltaH)<=b['numeric']['fusion_absolute'] and np.array_equal(H32,H.astype('<f4'))
    exported=export_check(b,art['packed']['path']);assert json.loads(json.dumps(exported))==r['export']
    with Path(art['packed']['path']).open('rb') as f:f.seek(b['fields']['head']['offset']);packed_H=np.frombuffer(f.read(b['fields']['head']['size']),dtype='<f4').reshape(V,D)
    assert np.array_equal(packed_H,H32) and r['radius_constraint_removed'] and r['parent_fit_radius']==16
    assert abs(norm(theta)-r['theta_norm'])<=b['numeric']['fusion_absolute']
    h64=float(np.linalg.norm(H,axis=1).max());h32=float(np.linalg.norm(H32.astype('f8'),axis=1).max());quant=float(np.linalg.norm(H32.astype('f8')-H,axis=1).max())
    for value,key in [(h64,'head_row_norm_F64_max'),(h32,'head_row_norm_F32_max'),(quant,'head_cast_row_error_max')]:assert abs(value-r[key])<=b['numeric']['fusion_absolute']
    rows,count=rows_for(b,H,h64,h32,quant,True,guard);maxKL=maxE=0.
    for actual,saved in zip(rows,r['records']):
        assert abs(actual['KL']-saved['KL'])<=b['numeric']['metric_absolute'] and actual['disagreement_rate']==saved['disagreement_rate']
        assert actual['id']==saved['id'] and actual['predicted_ids']==saved['predicted_ids'] and actual['disagreement_per_label']==saved['disagreement_per_label'] and actual['disagreement']==saved['disagreement']
        maxKL=max(maxKL,float(np.max(abs(np.asarray(actual['KL_per_label'])-saved['KL_per_label']))));maxE=max(maxE,float(np.max(abs(np.asarray(actual['score_uncertainty_per_label'])-saved['score_uncertainty_per_label']))));assert float(np.max(abs(np.asarray(actual['entropy_per_label'])-saved['entropy_per_label'])))<=b['numeric']['metric_absolute']
    assert maxKL<=b['numeric']['metric_absolute'] and maxE<=b['numeric']['bound_absolute'] and gates(r['records'],b)==r['aggregate']
    agg=gates(rows,b)
    for split in ['FIT','DEV']:
        assert agg[split]['strict_flags']==r['aggregate'][split]['strict_flags'] and agg[split]['original_flags']==r['aggregate'][split]['original_flags'] and abs(agg[split]['case_KL']-r['aggregate'][split]['case_KL'])<=b['numeric']['metric_absolute']
    selected={rec['id']:rec for rec in b['selected']};firstdelta=max(abs(row['KL_per_label'][0]-selected[row['id']]['reference_KL']) for row in rows if row['split']=='FIT');assert firstdelta<=b['numeric']['metric_absolute'] and agg['FIT']['first_disagreement']==0
    decision='ANALYTIC_FIRST_CORRECTION_PROXY_PASS_PENDING_NATIVE' if all(flag for split in agg.values() for flag in split['strict_flags'].values()) else 'ANALYTIC_FIRST_CORRECTION_PROXY_QUALITY_FAIL';assert decision==r['decision']
    for key in ['source_calls','native_calls','history_forwards','optimizer_updates','head_FG_calls','GPU_calls','reserved_queries','SVD_calls']:assert r[key]==0
    assert r['analytic_head_pairs']==1 and r['stored_head_label_products']==count==8808 and not r['quality_admission'] and not r['speed_admission']
    assert t['elapsed_seconds']<=b['capture_limits']['seconds'] and t['worker_OS_peak_through_exit']+t['launcher_OS_peak_snapshot']<=b['capture_limits']['OS_bytes'] and sum(item['bytes'] for item in t['outputs'])+t['result']['bytes']<=b['capture_limits']['output_bytes'];guard()
    write(a.out,dict(schema='ANALYTIC_ONSET_HEAD_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision=decision,complete_input_output_hashes=True,all8808_independent_probabilities_KL_argmax_verified=True,all48_case_domain_flags_verified=True,all24_reference_first_labels_verified=True,all_forced_uncertainty_verified=True,original_packed_bytes_except_head_exact=True,full_fusion_verified=True,max_adapter_absolute_delta=deltaJ,max_fusion_absolute_delta=deltaH,max_KL_absolute_delta=maxKL,max_uncertainty_absolute_delta=maxE,first_reference_max_KL_delta=firstdelta,stored_head_labels_verified=count,source_calls=0,native_calls=0,optimizer_updates=0,GPU_calls=0,SVD_calls=0,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,scope='Full stored-only independent audit, no core/source/history/optimizer/SVD replay. Proxy probabilities not actual new-head C/chatbot admission.'))
    print(json.dumps(dict(decision=decision,seconds=time.monotonic()-start,maxKL=maxKL)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('bind','capture-worker','audit-worker'):p.add_argument('--'+key,action='store_true')
    for key in ('binding','out','directory','source-result','protocol'):p.add_argument('--'+key,type=Path)
    for key in ('binding-sha','freeze'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.bind:bind(a)
    elif a.capture_worker:capture(a)
    elif a.audit_worker:audit(a)
    else:raise RuntimeError('Use separate qualified held launcher')
