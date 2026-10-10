"""Refine audited coordinate error bounds from the exact native dot path only."""
import argparse
from decimal import Decimal,localcontext
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
from causal_readout_coordinates import decision
sys.path.insert(0,str(SITE))


def proof(parent):
    evidence=[]
    for path in (parent['native_bodies']['source']['path'],parent['native_bodies']['header']['path']):
        text=Path(path).read_text();functions={}
        for item in parent['native_bodies']['functions']:
            if item['name'] not in ('dotf','hsum256'):continue
            match=re.search(r'^static[^\n]*\b'+item['name']+r'\(',text,re.M);assert match
            pos=text.index('{',match.start());depth=1;end=pos+1
            while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
            body=text[match.start():end];digest=hashlib.sha256(body.encode()).hexdigest();assert digest==item['original_body_sha256']==item['compiled_body_sha256']
            functions[item['name']]=dict(sha256=digest,compact=re.sub(r'\s+','',body))
        dot=functions['dotf']['compact'];hsum=functions['hsum256']['compact'];D=parent['dimension']
        assert D==256 and D%8==0 and dot.count('_mm256_fmadd_ps(')==1
        assert 'for(;i<=n-8;i+=8)s=_mm256_fmadd_ps(' in dot and 'floatr=hsum256(s);for(;i<n;i++)r+=a[i]*b[i];returnr;' in dot
        assert 'return'+'+'.join('o[%d]'%j for j in range(8))+';' in hsum
        evidence.append(dict(path=path,functions={name:item['sha256'] for name,item in functions.items()}))
    return dict(dimension=256,lanes=8,FMA_steps_per_lane=32,scalar_reduction_additions=7,tail_products=0,
        maximum_rounding_path=39,evidence=evidence,kind='Conservative FMA rounding path, not a new readout or an interval proof.')


def bind(a):
    assert not a.binding.exists()
    rp=DOC/'causal_readout_coordinates_result_20261010.json';r=json.loads(rp.read_bytes());tp=rp.with_suffix('.terminal.json');t=json.loads(tp.read_bytes())
    ap=DOC/'causal_readout_coordinates_stored_adjudication_20261010.json';au=json.loads(ap.read_bytes());atp=ap.with_suffix('.terminal.json');at=json.loads(atp.read_bytes())
    bp=DOC/'causal_readout_coordinates_binding_20261010.json';parent=json.loads(bp.read_bytes())
    assert t['exit_code']==at['exit_code']==0 and t['error'] is at['error'] is None and t['result']==au['result']==extent(rp)
    assert au['binding']==extent(bp) and at['result']==extent(ap) and au['complete_input_output_hashes']
    assert au['full_QR_SVD_identities_and_orthogonality'] and au['all_stored_feature_normal_equations_and_residuals'] and au['original_four_native_bodies_exact']
    assert r['recovered_labels']==8808 and len(r['records'])==48 and r['decision']=='CAUSAL_COORDINATES_NOT_QUALIFIED'
    assert r['numeric_flags']['feature_absolute'] is False and all(value for key,value in r['numeric_flags'].items() if key!='feature_absolute')
    assert parent['rounding']['dot_operations']==256 and parent['rounding']['unit']==2.**-24
    derivation=proof(parent)
    files=[Path(__file__),a.protocol,BROOT/'causal_readout_coordinates.py',BROOT/'original_joint_history_recovery_audit.py',BROOT/'original_packed_capacity.py',BROOT/'original_falcon_whole_recovery.py',
        rp,tp,ap,atp,bp,Path(sys.executable),Path(sys.executable).parent/'python312.dll',SITE/'psutil/__init__.py']
    files += [Path(item['path']) for item in derivation['evidence']];files += sorted((SITE/'psutil').glob('*.pyd'))
    write(a.binding,dict(schema='CAUSAL_COORDINATE_DOT_BOUND_BINDING_V1',inputs=[extent(path) for path in dict.fromkeys(files)],parent_result=extent(rp),parent_binding=extent(bp),
        parent_audit=extent(ap),parent_audit_terminal=extent(atp),proof=derivation,
        capture_limits=dict(seconds=60,reserve_seconds=5,OS_bytes=512<<20,GPU_allocated_bytes=0,GPU_reserved_bytes=0,output_bytes=8<<20,log_bytes=2<<20),
        audit_limits=dict(seconds=60,OS_bytes=512<<20,output_bytes=2<<20,log_bytes=2<<20),
        scope='Refine only F32 dot error operation count from audited exact32-FMA/7-add native body; unchanged features, factors, residuals and admission thresholds. No inversion/history replay.'))
    print(json.dumps(dict(binding=str(a.binding),sha256=sha(a.binding))),flush=True)


def refine(parent,r):
    cal=dict(r['calibration']);unit=parent['rounding']['unit'];cal['F32_gamma_dot']=39*unit/(1-39*unit)
    cal['native_dot_rounding_L2_upper']=cal['F32_gamma_dot']*cal['native_normalized_norm_upper']*cal['head_Frobenius']
    rows=[]
    for old in r['records']:
        row=dict(old);row['inherited_case_record']=row.pop('case_record')
        bounds=[(residual+cal['native_dot_rounding_L2_upper'])/cal['sigma_H_lower'] for residual in row['score_residual_L2_per_label']]
        relative=[bound/(norm-bound) if norm>bound else math.inf for norm,bound in zip(row['feature_norm_per_label'],bounds)]
        row.update(feature_error_L2_upper_per_label=bounds,feature_error_relative_upper_per_label=relative,
            max_feature_error_L2_upper=max(bounds),max_feature_error_relative_upper=max(relative));rows.append(row)
    outcome,flags=decision(parent,cal,rows);return cal,rows,outcome,flags


def capture(a):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([0]);b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    a.directory.mkdir(exist_ok=False);parent=json.loads(Path(b['parent_binding']['path']).read_bytes());r=json.loads(Path(b['parent_result']['path']).read_bytes())
    derivation=proof(parent);assert derivation==b['proof'];proof_path=a.directory/'native_dot_path_proof.json';write(proof_path,derivation)
    cal,rows,outcome,flags=refine(parent,r)
    assert time.monotonic()-start<b['capture_limits']['seconds']-b['capture_limits']['reserve_seconds'] and proc.memory_info().peak_wset<=b['capture_limits']['OS_bytes'] and not proc.children(recursive=True)
    write(a.out,dict(schema='CAUSAL_COORDINATE_DOT_BOUND_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,parent_result=b['parent_result'],parent_audit=b['parent_audit'],
        parent_decision=r['decision'],decision=outcome,numeric_flags=flags,calibration=cal,records=rows,proof=derivation,proof_file=extent(proof_path),
        bound_labels=8808,new_coordinates_recovered=0,QR_factorizations=0,SVD_factorizations=0,stored_full_head_reconstructions=0,
        source_calls=0,native_binary_calls=0,optimizer_updates=0,teacher_distribution_queries=0,DEV_quality_queries=0,GPU_calls=0,GPU_allocated_peak=0,GPU_reserved_peak=0,
        worker_OS_peak_snapshot=proc.memory_info().peak_wset,seconds=time.monotonic()-start,quality_admission=False,speed_admission=False,
        scope='Bound-only refinement, old NOT_QUALIFIED record intact; approximate native coordinates only, all quality/engine/generality gates remain open.'))
    print(json.dumps(dict(decision=outcome,max_bound=max(row['max_feature_error_L2_upper'] for row in rows),seconds=time.monotonic()-start)),flush=True)


def audit(a):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([0]);b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes())
    assert sha(a.binding)==a.binding_sha==r['binding_sha256']==t['binding_sha256'] and a.freeze==r['freeze']==t['freeze']
    assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(a.source_result)
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item
    old=json.loads(Path(b['parent_result']['path']).read_bytes());parent=json.loads(Path(b['parent_binding']['path']).read_bytes());derivation=proof(parent)
    assert derivation==r['proof']==b['proof'] and extent(r['proof_file']['path'])==r['proof_file'] and json.loads(Path(r['proof_file']['path']).read_bytes())==derivation
    max_delta=0.;count=0
    def close(x,y):
        nonlocal max_delta
        delta=abs(float(x)-y);max_delta=max(max_delta,delta);assert delta<=parent['numeric']['audit_scalar_absolute']+parent['numeric']['audit_relative']*abs(y)
    with localcontext() as ctx:
        ctx.prec=50;unit=Decimal(2)**-24;gamma=39*unit/(1-39*unit)
        cal=r['calibration'];close(gamma,cal['F32_gamma_dot'])
        dot=gamma*Decimal.from_float(old['calibration']['native_normalized_norm_upper'])*Decimal.from_float(old['calibration']['head_Frobenius']);close(dot,cal['native_dot_rounding_L2_upper'])
        for key,value in old['calibration'].items():
            if key not in ('F32_gamma_dot','native_dot_rounding_L2_upper'):assert value==cal[key]
        assert len(old['records'])==len(r['records'])==48
        changed={'case_record','inherited_case_record','feature_error_L2_upper_per_label','feature_error_relative_upper_per_label','max_feature_error_L2_upper','max_feature_error_relative_upper'}
        for source,row in zip(old['records'],r['records']):
            assert row['inherited_case_record']==source['case_record']
            assert {k:v for k,v in row.items() if k not in changed}=={k:v for k,v in source.items() if k not in changed}
            for j,(residual,norm) in enumerate(zip(source['score_residual_L2_per_label'],source['feature_norm_per_label'])):
                bound=(Decimal.from_float(residual)+dot)/Decimal.from_float(cal['sigma_H_lower']);relative=bound/(Decimal.from_float(norm)-bound)
                close(bound,row['feature_error_L2_upper_per_label'][j]);close(relative,row['feature_error_relative_upper_per_label'][j]);count+=1
            assert row['max_feature_error_L2_upper']==max(row['feature_error_L2_upper_per_label']) and row['max_feature_error_relative_upper']==max(row['feature_error_relative_upper_per_label'])
    outcome,flags=decision(parent,r['calibration'],r['records']);assert outcome==r['decision'] and flags==r['numeric_flags'] and r['parent_decision']==old['decision']
    assert r['parent_result']==b['parent_result'] and r['parent_audit']==b['parent_audit'] and count==r['bound_labels']==8808
    for key in ('new_coordinates_recovered','QR_factorizations','SVD_factorizations','stored_full_head_reconstructions','source_calls','native_binary_calls','optimizer_updates','teacher_distribution_queries','DEV_quality_queries','GPU_calls','GPU_allocated_peak','GPU_reserved_peak'):assert r[key]==0
    assert not r['quality_admission'] and not r['speed_admission']
    assert t['elapsed_seconds']<=b['capture_limits']['seconds'] and t['worker_OS_peak_through_exit']+t['launcher_OS_peak_snapshot']<=b['capture_limits']['OS_bytes']
    assert time.monotonic()-start<b['audit_limits']['seconds'] and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes'] and not proc.children(recursive=True)
    write(a.out,dict(schema='CAUSAL_COORDINATE_DOT_BOUND_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision=outcome,numeric_flags=flags,
        complete_input_output_hashes=True,exact_native_dot_path_verified=True,parent_full_audit_adopted=b['parent_audit'],
        all8808_bounds_independently_checked_decimal50=True,max_decimal_comparison_absolute_delta=max_delta,
        unchanged_features_factors_residuals_and_thresholds=True,bound_labels=count,new_coordinates_recovered=0,QR_factorizations=0,SVD_factorizations=0,
        native_calls=0,source_calls=0,optimizer_updates=0,GPU_calls=0,seconds=time.monotonic()-start))
    print(json.dumps(dict(decision=outcome,seconds=time.monotonic()-start)),flush=True)


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
