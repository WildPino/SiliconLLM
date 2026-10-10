"""Stored native causal coordinates via qualified head inversion; no history replay."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];BROOT=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(BROOT))
from original_packed_capacity import SITE,extent,sha,write
from original_falcon_whole_recovery import memory_reader
from original_joint_history_recovery_audit import packed_fields
sys.path.insert(0,str(SITE))
V,D=65537,256


def bind(a):
    import numpy as np
    import numpy._core._multiarray_umath as core
    import numpy.linalg._umath_linalg as linalg
    assert not a.binding.exists()
    rp=DOC/'original_engine_baseline_result_20261009.json';r=json.loads(rp.read_bytes());tp=rp.with_suffix('.terminal.json');t=json.loads(tp.read_bytes())
    ap=DOC/'original_engine_baseline_stored_adjudication_20261009.json';au=json.loads(ap.read_bytes())
    bp=DOC/'original_engine_baseline_binding_20261009.json';parent=json.loads(bp.read_bytes())
    assert t['exit_code']==0 and t['result_sha256']==sha(rp)==au['result']['sha256'] and au['binding']==extent(bp)
    assert au['all_prefix_heads_routes_bit_exact'] and au['all_integer_witnesses_exact'] and r['packed']==parent['packed']
    assert extent(r['packed']['path'])==r['packed']
    fields=packed_fields(r['packed']['path']);assert fields['head']['shape']==(V,D) and fields['final_norm']['shape']==(D,)
    assert fields['head']['dtype']==fields['final_norm']['dtype']==1
    records={rec['id']:rec for rec in parent['records']};selected=[]
    files=[Path(__file__),a.protocol,rp,tp,ap,bp,Path(r['packed']['path']),Path(r['executable']['path']),
        Path(r['original_bodies']['source']['path']),Path(r['original_bodies']['header']['path']),
        Path(au['audit_parent']['path']),Path(au['audit_completion']['path']),
        BROOT/'original_packed_capacity.py',BROOT/'original_falcon_whole_recovery.py',BROOT/'original_joint_history_recovery_audit.py',
        Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(core.__file__),Path(linalg.__file__),SITE/'numpy/__init__.py',SITE/'psutil/__init__.py']
    files+=sorted((SITE/'numpy.libs').glob('*.dll'));files+=sorted((SITE/'psutil').glob('*.pyd'))
    for row in r['native_records']:
        rec=records[row['id']];assert row['split']==rec['split'] and row['domain']==rec['domain']
        assert row['history']==len(rec['student_input_ids']) and row['labels']==len(rec['positions'])
        total=row['logits']['bytes']//(V*4);offset=row['row_offset'];assert row['logits']['bytes']==total*V*4
        assert min(rec['positions'])>=0 and max(rec['positions'])<row['history'] and offset+row['history']<=total
        selected.append(dict(id=row['id'],split=row['split'],domain=row['domain'],history=row['history'],labels=row['labels'],
            row_offset=offset,positions=rec['positions'],logits=row['logits'],adopted=row['adopted']))
        files.append(Path(row['logits']['path']))
    assert len(selected)==48 and sum(rec['labels'] for rec in selected)==8808
    assert sum(rec['labels'] for rec in selected if rec['split']=='FIT')==4422 and sum(rec['split']=='FIT' for rec in selected)==24
    inputs=[extent(p) for p in dict.fromkeys(files)];lookup={item['path']:item for item in inputs}
    for row in selected:assert lookup[row['logits']['path']]==row['logits']
    for item in parent['inputs']+t['output_files']:
        if item['path'] in lookup:assert lookup[item['path']]==item
    write(a.binding,dict(schema='CAUSAL_READOUT_COORDINATES_BINDING_V1',inputs=inputs,records=selected,packed=r['packed'],fields=fields,
        native_bodies=r['original_bodies'],vocab=V,dimension=D,chunk_labels=32,labels=8808,
        numeric=dict(rank_relative=1e-8,QR_relative=1e-10,orthogonal_absolute=1e-10,SVD_relative=1e-10,
            scores_absolute=1e-4,feature_bound_absolute=.05,feature_bound_relative=.01,recovered_norm_slack=.05,
            audit_scalar_absolute=1e-10,audit_relative=1e-10,normal_equation_absolute=1e-9),
        rounding=dict(unit=2.**-24,dot_operations=256,RMS_operations=512,RMS_final_roundings=4,epsilon=1e-5,
            kind='Conservative unexceptional F32/FMA rounding model, F64 evaluated; not directed-rounding intervals'),
        capture_limits=dict(seconds=600,reserve_seconds=60,OS_bytes=3<<30,GPU_allocated_bytes=0,GPU_reserved_bytes=0,output_bytes=256<<20,log_bytes=8<<20),
        audit_limits=dict(seconds=600,OS_bytes=3<<30,output_bytes=2<<20,log_bytes=8<<20),
        scope='One stored actual27 C normalized-state recovery from all8808 selected FIT/DEV labels. No teacher scores used, fit, GPU, native or source call.'))
    print(json.dumps(dict(binding=str(a.binding),sha256=sha(a.binding),inputs=len(inputs),bytes=sum(item['bytes'] for item in inputs))),flush=True)


def save(directory,name,value):
    import numpy as np
    x=np.ascontiguousarray(value,dtype='<f8');path=directory/name
    with path.open('xb') as f:f.write(x.tobytes());f.flush();os.fsync(f.fileno())
    return dict(**extent(path),shape=list(x.shape),dtype='float64')


def source_head(b):
    import numpy as np
    with Path(b['packed']['path']).open('rb') as f:
        f.seek(b['fields']['head']['offset']);H=np.frombuffer(f.read(b['fields']['head']['size']),dtype='<f4').reshape(V,D).astype('f8')
        f.seek(b['fields']['final_norm']['offset']);gamma=np.frombuffer(f.read(b['fields']['final_norm']['size']),dtype='<f4').astype('f8')
    assert np.isfinite(H).all() and np.isfinite(gamma).all();return H,gamma


def calibration(b,H,gamma,Q,R,U,s,VT):
    import numpy as np
    def relative(error,reference):return float(np.linalg.norm(error))/max(float(np.linalg.norm(reference)),1e-300)
    qr_error=float(np.linalg.norm(H-Q@R));q_orth=float(np.linalg.norm(Q.T@Q-np.eye(D)))
    svd_error=float(np.linalg.norm(R-(U*s)@VT));u_orth=float(np.linalg.norm(U.T@U-np.eye(D)));v_orth=float(np.linalg.norm(VT@VT.T-np.eye(D)))
    sigma_R_lower=math.sqrt(max(0.,1-u_orth))*math.sqrt(max(0.,1-v_orth))*float(s[-1])-svd_error
    sigma_H_lower=math.sqrt(max(0.,1-q_orth))*sigma_R_lower-qr_error
    unit=b['rounding']['unit'];gdot=b['rounding']['dot_operations']*unit/(1-b['rounding']['dot_operations']*unit)
    grms=b['rounding']['RMS_operations']*unit/(1-b['rounding']['RMS_operations']*unit)
    C=math.sqrt(D)*float(np.max(abs(gamma)))*(1+unit)**b['rounding']['RMS_final_roundings']/math.sqrt(1-grms)
    Hnorm=float(np.linalg.norm(H));rounding_l2=gdot*C*Hnorm
    effective_rank=int(np.count_nonzero(s>=b['numeric']['rank_relative']*s[0]))
    return dict(head_Frobenius=Hnorm,head_singular_max=float(s[0]),head_singular_min=float(s[-1]),head_condition=float(s[0]/s[-1]) if s[-1]>0 else None,
        effective_rank=effective_rank,QR_error_Frobenius=qr_error,QR_relative=qr_error/Hnorm,Q_orthogonal_Frobenius=q_orth,
        SVD_error_Frobenius=svd_error,SVD_relative=relative(R-(U*s)@VT,R),U_orthogonal_Frobenius=u_orth,V_orthogonal_Frobenius=v_orth,
        sigma_R_lower=sigma_R_lower,sigma_H_lower=sigma_H_lower,F32_gamma_dot=gdot,F32_gamma_RMS=grms,
        native_normalized_norm_upper=C,native_dot_rounding_L2_upper=rounding_l2,
        bound_kind=b['rounding']['kind'])


def decision(b,cal,rows):
    g=b['numeric'];invertible=cal['effective_rank']==D and cal['sigma_H_lower']>0
    flags=dict(numerically_full_rank=invertible,QR=cal['QR_relative']<=g['QR_relative'],SVD=cal['SVD_relative']<=g['SVD_relative'],
        orthogonal=max(cal[key] for key in ('Q_orthogonal_Frobenius','U_orthogonal_Frobenius','V_orthogonal_Frobenius'))<=g['orthogonal_absolute'])
    if invertible:
        assert len(rows)==48
        flags.update(scores=max(row['max_score_residual_absolute'] for row in rows)<=g['scores_absolute'],
            feature_absolute=max(row['max_feature_error_L2_upper'] for row in rows)<=g['feature_bound_absolute'],
            feature_relative=max(row['max_feature_error_relative_upper'] for row in rows)<=g['feature_bound_relative'],
            norms=max(row['max_feature_norm'] for row in rows)<=cal['native_normalized_norm_upper']+g['recovered_norm_slack'])
    else:assert not rows
    return ('CAUSAL_COORDINATES_QUALIFIED_APPROXIMATE' if all(flags.values()) else 'CAUSAL_COORDINATES_NOT_QUALIFIED'),flags


def row_statistics(b,cal,features,scores,H):
    import numpy as np
    residual=features@H.T-scores;norm=np.linalg.norm(features,axis=1);residual_l2=np.linalg.norm(residual,axis=1)
    bounds=(residual_l2+cal['native_dot_rounding_L2_upper'])/cal['sigma_H_lower']
    denominator=norm-bounds;relative=np.where(denominator>0,bounds/denominator,np.inf)
    return dict(score_residual_absolute_per_label=np.max(abs(residual),axis=1).tolist(),score_residual_L2_per_label=residual_l2.tolist(),
        feature_norm_per_label=norm.tolist(),feature_error_L2_upper_per_label=bounds.tolist(),feature_error_relative_upper_per_label=relative.tolist())


def capture(a):
    import numpy as np
    import psutil
    assert np.__version__=='2.4.6' and psutil.__version__=='7.2.2'
    b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)));a.directory.mkdir(exist_ok=False);stage='head';rows=[]
    def guard():
        lim=b['capture_limits'];assert time.monotonic()-start<lim['seconds']-lim['reserve_seconds']
        assert proc.memory_info().peak_wset<=lim['OS_bytes'] and not proc.children(recursive=True)
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes']
    try:
        H,gamma=source_head(b);Q,R=np.linalg.qr(H,mode='reduced');U,s,VT=np.linalg.svd(R,full_matrices=False);guard()
        factors={key:save(a.directory,key+'.f64',value) for key,value in (('Q',Q),('R',R),('U',U),('singular_values',s),('VT',VT))}
        cal=calibration(b,H,gamma,Q,R,U,s,VT);stage='recover'
        calibration_path=a.directory/'calibration.json';write(calibration_path,dict(calibration=cal,factors=factors));calibration_file=extent(calibration_path)
        print(json.dumps(dict(stage=stage,calibration=cal,seconds=time.monotonic()-start)),flush=True)
        if cal['effective_rank']==D and cal['sigma_H_lower']>0:
            for rec in b['records']:
                total=rec['logits']['bytes']//(V*4);stored=np.memmap(rec['logits']['path'],mode='r',dtype='<f4',shape=(total,V))
                features=[];stats={key:[] for key in ('score_residual_absolute_per_label','score_residual_L2_per_label','feature_norm_per_label','feature_error_L2_upper_per_label','feature_error_relative_upper_per_label')}
                for first in range(0,rec['labels'],b['chunk_labels']):
                    indices=np.asarray(rec['positions'][first:first+b['chunk_labels']])+rec['row_offset'];z=stored[indices].astype('f8');assert np.isfinite(z).all()
                    f=np.linalg.solve(R,(z@Q).T).T;assert np.isfinite(f).all();features.append(f)
                    current=row_statistics(b,cal,f,z,H)
                    for key in stats:stats[key]+=current[key]
                    guard()
                values=np.concatenate(features);assert values.shape==(rec['labels'],D)
                row=dict(id=rec['id'],split=rec['split'],domain=rec['domain'],labels=rec['labels'],positions=rec['positions'],row_offset=rec['row_offset'],
                    features=save(a.directory,rec['id']+'.features.f64',values),**stats,
                    max_score_residual_absolute=max(stats['score_residual_absolute_per_label']),max_feature_error_L2_upper=max(stats['feature_error_L2_upper_per_label']),
                    max_feature_error_relative_upper=max(stats['feature_error_relative_upper_per_label']),max_feature_norm=max(stats['feature_norm_per_label']))
                case_path=a.directory/(rec['id']+'.case.json');write(case_path,row);row['case_record']=extent(case_path)
                rows.append(row);del stored,z,f,values,features
                print(json.dumps(dict(stage=stage,id=rec['id'],completed=len(rows),max_residual=row['max_score_residual_absolute'],seconds=time.monotonic()-start)),flush=True)
        outcome,flags=decision(b,cal,rows);guard()
        write(a.out,dict(schema='CAUSAL_READOUT_COORDINATES_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,calibration=cal,calibration_file=calibration_file,factors=factors,records=rows,
            decision=outcome,numeric_flags=flags,recovered_labels=sum(row['labels'] for row in rows),QR_factorizations=1,SVD_factorizations=1,
            stored_full_head_reconstructions=sum(row['labels'] for row in rows),source_generations=0,source_history_forwards=0,native_binary_calls=0,
            optimizer_updates=0,teacher_distribution_queries=0,DEV_quality_queries=0,GPU_calls=0,GPU_allocated_peak=0,GPU_reserved_peak=0,
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,seconds=time.monotonic()-start,quality_admission=False,speed_admission=False,
            scope='Approximate normalized actual27 C states with conservative numeric error bounds; teacher-free stored observable, no causal/history or optimizer replay.'))
        print(json.dumps(dict(stage='complete',decision=outcome,labels=sum(row['labels'] for row in rows),seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),completed_records=rows,calibration=locals().get('cal'),seconds=time.monotonic()-start));raise


def audit(a):
    import numpy as np
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)));b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes())
    assert sha(a.binding)==a.binding_sha==r['binding_sha256']==t['binding_sha256'] and a.freeze==r['freeze']==t['freeze']
    assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(a.source_result)
    def guard():
        assert time.monotonic()-start<b['audit_limits']['seconds'] and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes'] and not proc.children(recursive=True)
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item;guard()
    output_map={item['path']:item for item in t['outputs']};g=b['numeric'];delta_max=normal_max=0.
    def array(item,shape):
        assert output_map[item['path']]=={key:item[key] for key in ('path','bytes','sha256')} and item['shape']==list(shape) and item['dtype']=='float64'
        value=np.fromfile(item['path'],dtype='<f8').reshape(shape);assert np.isfinite(value).all();return value
    def close(actual,stored):
        nonlocal delta_max
        actual=np.asarray(actual);stored=np.asarray(stored);assert actual.shape==stored.shape
        if np.isinf(stored).any():assert np.array_equal(np.isinf(actual),np.isinf(stored));actual=actual[np.isfinite(actual)];stored=stored[np.isfinite(stored)]
        delta=float(np.max(abs(actual-stored),initial=0));delta_max=max(delta_max,delta)
        assert delta<=g['audit_scalar_absolute']+g['audit_relative']*float(np.max(abs(stored),initial=0))
    H,gamma=source_head(b);factors=r['factors'];Q=array(factors['Q'],(V,D));R=array(factors['R'],(D,D));U=array(factors['U'],(D,D));s=array(factors['singular_values'],(D,));VT=array(factors['VT'],(D,D))
    assert np.all(s>=0) and np.all(np.diff(s)<=0) and np.count_nonzero(np.tril(R,-1))==0
    cal=calibration(b,H,gamma,Q,R,U,s,VT)
    assert output_map[r['calibration_file']['path']]==r['calibration_file']
    assert json.loads(Path(r['calibration_file']['path']).read_bytes())==dict(calibration=r['calibration'],factors=factors)
    for key,value in cal.items():
        if isinstance(value,(float,int)):close(value,r['calibration'][key])
        else:assert value==r['calibration'][key]
    counts=0;rows=[];scalar_witnesses=0;max_scalar_arithmetic_delta=0.
    if cal['effective_rank']==D and cal['sigma_H_lower']>0:
        assert len(r['records'])==len(b['records'])==48
        for row,rec in zip(r['records'],b['records']):
            assert output_map[row['case_record']['path']]==row['case_record']
            assert json.loads(Path(row['case_record']['path']).read_bytes())=={key:value for key,value in row.items() if key!='case_record'}
            for key in ('id','split','domain','labels','positions','row_offset'):assert row[key]==rec[key]
            features=array(row['features'],(rec['labels'],D));total=rec['logits']['bytes']//(V*4);stored=np.memmap(rec['logits']['path'],mode='r',dtype='<f4',shape=(total,V))
            stats={key:[] for key in ('score_residual_absolute_per_label','score_residual_L2_per_label','feature_norm_per_label','feature_error_L2_upper_per_label','feature_error_relative_upper_per_label')}
            for first in range(0,rec['labels'],b['chunk_labels']):
                last=min(first+b['chunk_labels'],rec['labels']);indices=np.asarray(rec['positions'][first:last])+rec['row_offset'];z=stored[indices].astype('f8');f=features[first:last]
                ne=float(np.max(abs(z@Q-f@R.T)));normal_max=max(normal_max,ne);assert ne<=g['normal_equation_absolute']
                if first==0:
                    for v in (0,1,17,257,4096,16384,32768,V-1):
                        scalar=math.fsum(float(h)*float(c) for h,c in zip(H[v],f[0]));delta=abs(scalar-float(np.dot(H[v],f[0])))
                        max_scalar_arithmetic_delta=max(max_scalar_arithmetic_delta,delta);scalar_witnesses+=1
                        assert delta<=g['audit_scalar_absolute']
                current=row_statistics(b,cal,f,z,H)
                for key in stats:stats[key]+=current[key]
                counts+=last-first;guard()
            for key in stats:close(stats[key],row[key])
            for key,value in (('max_score_residual_absolute',max(stats['score_residual_absolute_per_label'])),('max_feature_error_L2_upper',max(stats['feature_error_L2_upper_per_label'])),
                ('max_feature_error_relative_upper',max(stats['feature_error_relative_upper_per_label'])),('max_feature_norm',max(stats['feature_norm_per_label']))):close(value,row[key])
            rows.append(row);del stored,features,z,f
            print(json.dumps(dict(stage='stored_case_audited',id=row['id'],completed=len(rows),seconds=time.monotonic()-start)),flush=True)
    else:assert not r['records']
    outcome,flags=decision(b,cal,rows);assert outcome==r['decision'] and flags==r['numeric_flags'] and counts==r['recovered_labels']==r['stored_full_head_reconstructions']
    assert r['QR_factorizations']==r['SVD_factorizations']==1
    for key in ('source_generations','source_history_forwards','native_binary_calls','optimizer_updates','teacher_distribution_queries','DEV_quality_queries','GPU_calls','GPU_allocated_peak','GPU_reserved_peak'):assert r[key]==0
    assert t['elapsed_seconds']<=b['capture_limits']['seconds'] and t['worker_OS_peak_through_exit']+t['launcher_OS_peak_snapshot']<=b['capture_limits']['OS_bytes']
    assert sum(item['bytes'] for item in t['outputs'])+t['result']['bytes']<=b['capture_limits']['output_bytes']
    for item in b['native_bodies']['functions']:
        if item['name'] not in ('hsum256','dotf','matvec','rmsnorm'):continue
        for path in (b['native_bodies']['source']['path'],b['native_bodies']['header']['path']):
            text=Path(path).read_text();match=re.search(r'^static[^\n]*\b'+item['name']+r'\(',text,re.M);assert match
            pos=text.index('{',match.start());depth=1;end=pos+1
            while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
            assert hashlib.sha256(text[match.start():end].encode()).hexdigest()==item['original_body_sha256']==item['compiled_body_sha256']
    guard();write(a.out,dict(schema='CAUSAL_READOUT_COORDINATES_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision=outcome,numeric_flags=flags,
        complete_input_output_hashes=True,original_four_native_bodies_exact=True,full_QR_SVD_identities_and_orthogonality=True,all_stored_feature_normal_equations_and_residuals=True,
        max_scalar_comparison_absolute_delta=delta_max,max_normal_equation_absolute_delta=normal_max,calibration=cal,
        scalar_head_witnesses=scalar_witnesses,max_scalar_arithmetic_absolute_delta=max_scalar_arithmetic_delta,
        stored_full_head_reconstructions=counts,QR_factorizations=0,SVD_factorizations=0,native_binary_calls=0,optimizer_updates=0,source_calls=0,GPU_calls=0,
        seconds=time.monotonic()-start,scope='Full stored feature/factor/rounding-bound/metrics/decision/custody/resource audit; no QR/SVD or history replay.'))
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
             CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
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
