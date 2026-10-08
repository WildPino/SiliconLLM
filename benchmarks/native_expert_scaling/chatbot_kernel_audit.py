"""First explicit-feature Gram and saved-factor audit; no model or new factor."""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_joint_retained_audit import routes


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_FULL_FIT_KERNEL_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_or_shared_responses=0,new_J_or_coefficient_solutions=0,new_Cholesky_evaluations=0)
    def guard():assert time.monotonic()-start<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='kernel_audit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        raw=json.loads(Path(b['kernel_path']).read_bytes());assert raw['decision']=='ELIGIBLE_FOR_ONE_FULL_FIT_CONVEX_COMPILER'
        def array(name,shape):
            item=raw['outputs'][name];p=Path(item['path']);a=np.load(p,mmap_mode='r',allow_pickle=False)
            assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
            assert a.dtype==np.dtype('<f8') and a.shape==shape and a.flags.c_contiguous and np.isfinite(a).all()
            assert a.offset+a.nbytes==p.stat().st_size;return a
        x=array('FIT_x',(3845,896));a=array('FIT_weights',(3845,));w=array('FIT_W',(3845,16));u=array('FIT_U',(3845,897));mu=array('mu',(896,))
        gram=array('G',(3845,3845));factor=array('L',(3845,3845))
        journal=json.loads(Path(raw['outputs']['G']['path']).with_name('FIT_journal.json').read_bytes())
        adoption=json.loads(Path(b['adoption_path']).read_bytes());keys=[];cursor=0
        for case_index,case in enumerate(adoption['cases']):
            if case['split']!='fit':continue
            with Path(case['binary_path']).open('rb') as f:
                assert struct.unpack('<8s4I',f.read(24))==(b'QWCAP001',24,896,2,0)
                for local in range(case['captured_rows']):
                    offset=24+local*86016+12*3584;f.seek(offset);key=f.read(1792);record=journal[cursor]
                    assert (record['split_row_index'],record['case_index'],record['local_input_row'],record['x_byte_offset'])==(cursor,case_index,local,offset)
                    assert hashlib.sha256(key).hexdigest()==record['x_SHA256']
                    values=struct.unpack('<896f',struct.pack('<896I',*(v<<16 for v in struct.unpack('<896H',key))))
                    assert all(float(p)==v for p,v in zip(x[cursor],values));keys.append(key);cursor+=1
            guard()
        assert cursor==len(journal)==3845;counts=Counter(keys);assert len(counts)==3166
        expected_a=[1/counts[key] for key in keys];assert np.array_equal(a,expected_a)
        sizes={split:sum(c['captured_rows'] for c in adoption['cases'] if c['split']==split) for split in ('fit','development')}
        cached=routes(Path(b['saved_E16_routes']),sizes)['fit'];expected_w=np.zeros((3845,16))
        for i,(ids,mass) in enumerate(zip(cached['ids'],cached['mass'])):expected_w[i,list(ids)]=mass
        assert np.array_equal(w,expected_w)
        total=math.fsum(expected_a)
        expected_mu=np.array([math.fsum(float(x[i,j])*expected_a[i] for i in range(3845))/total for j in range(896)])
        expected_r=math.sqrt(math.fsum(expected_a[i]*math.fsum((float(x[i,j])-expected_mu[j])**2 for j in range(896)) for i in range(3845))/(896*total));guard()
        mu_gap=float(np.max(np.abs(mu-expected_mu)));radius_gap=abs(expected_r-raw['kernel']['radius'])
        assert mu_gap<=1e-12 and radius_gap<=1e-12
        expected_u=np.column_stack((np.ones(3845),(x-expected_mu)/expected_r))
        u_gap=float(np.linalg.norm(expected_u-u)/np.linalg.norm(u));assert u_gap<=1e-12
        # Independent full explicit feature matrix: NOT U-inner times W-inner.
        features=np.empty((3845,14352))
        for e in range(16):features[:,897*e:897*(e+1)]=expected_w[:,e,None]*expected_u
        features*=np.sqrt(np.array(expected_a))[:,None]
        explicit=features@features.T;guard()
        gram_gap=float(np.linalg.norm(explicit-gram)/np.linalg.norm(gram));assert gram_gap<=1e-12
        trace=math.fsum(float(gram[i,i]) for i in range(3845));alpha=trace/999999
        assert abs(trace-raw['kernel']['trace'])<=1e-12*trace and abs(alpha-raw['kernel']['alpha'])<=1e-12*alpha
        del features,explicit
        regularized=np.array(gram);regularized.flat[::3846]+=raw['kernel']['alpha']
        assert (np.diag(factor)>0).all() and not np.any(np.triu(factor,1))
        reconstruction=float(np.linalg.norm(factor@factor.T-regularized)/np.linalg.norm(regularized));assert reconstruction<=1e-10
        symmetry=float(np.linalg.norm(gram-gram.T)/np.linalg.norm(gram));assert symmetry<=1e-12
        r['independent_checks']=dict(original_rows=3845,exact_x_classes=3166,weight_sum=total,mu_max_absolute_gap=mu_gap,
            radius_absolute_gap=radius_gap,U_relative_gap=u_gap,ALL_explicit_feature_Gram_relative_gap=gram_gap,
            Cholesky_relative_reconstruction=reconstruction,symmetry_relative=symmetry,alpha=alpha,
            condition_scope='PSD real-Gram deduction reused; no new SVD, numerical-rank or exact condition proof')
        assert all(raw['eligibility_gates'].values())
        r['procedure_gates'].update(ALL_original_FIT_x_rows_and_weights=True,original_mass_bytes=True,
            independent_scalar_mean_radius=True,ALL_Gram_entries_by_explicit_feature_assembly=True,
            saved_Cholesky_certificate_and_all_eligibility=True,no_model_labels_or_new_factor_replay=True)
        r.update(decision='SAVED_FULL_FIT_KERNEL_INDEPENDENTLY_VERIFIED',elapsed_before_final_serialization=time.monotonic()-start,
            OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
