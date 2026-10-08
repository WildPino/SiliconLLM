"""First full-FIT affine-feature kernel; no response, donor/core or fit calls."""
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


def fit_rows(adoption):
    """Read only original FIT x, never y or development x."""
    keys=[];journal=[]
    for case_index,case in enumerate(adoption['cases']):
        if case['split']!='fit':continue
        with Path(case['binary_path']).open('rb') as f:
            assert struct.unpack('<8s4I',f.read(24))==(b'QWCAP001',24,896,2,0)
            for local in range(case['captured_rows']):
                offset=24+local*86016+12*3584;f.seek(offset);key=f.read(1792);assert len(key)==1792
                keys.append(key);journal.append(dict(split_row_index=len(journal),case_index=case_index,
                    local_input_row=local,x_byte_offset=offset,x_SHA256=hashlib.sha256(key).hexdigest()))
    return keys,journal


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_FULL_FIT_KERNEL_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_or_shared_responses=0,new_source_or_shared_J=0,
        new_coefficient_solutions=0,outputs={})
    def guard():
        assert time.monotonic()-start<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
        if args.directory.exists():assert sum(v.stat().st_size for v in args.directory.iterdir() if v.is_file())<=512<<20
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='kernel'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        budget=json.loads(Path(b['cost_path']).read_bytes())['budget']
        assert (budget['matrix_MAC'],budget['logical_bytes'],budget['stored_bytes'])==(290975744,583842816,1047295232)
        assert budget['complete_matrix_gate'] and budget['complete_logical_gate'];r['reused_complete_budget']=budget
        adoption=json.loads(Path(b['adoption_path']).read_bytes());keys,journal=fit_rows(adoption)
        assert len(keys)==3845;counts=Counter(keys);assert len(counts)==3166
        bits=np.stack([np.frombuffer(key,dtype='<u2') for key in keys]);assert ((bits&0x7fff)<0x7f80).all()
        x=(bits.astype('<u4')<<16).view('<f4').astype(np.float64)
        a=np.array([1/counts[key] for key in keys],dtype=np.float64);weight_sum=math.fsum(float(v) for v in a)
        assert abs(weight_sum-3166)<=1e-11
        sizes={split:sum(c['captured_rows'] for c in adoption['cases'] if c['split']==split) for split in ('fit','development')}
        cached=routes(Path(b['saved_E16_routes']),sizes)['fit'];w=np.zeros((3845,16))
        for i,(ids,mass) in enumerate(zip(cached['ids'],cached['mass'])):
            assert len(set(ids))==4 and all(0<=e<16 for e in ids) and all(math.isfinite(v) and 0<=v<=1 for v in mass)
            assert abs(math.fsum(mass)-1)<=2e-6;w[i,list(ids)]=mass
        mu=(a[:,None]*x).sum(axis=0)/weight_sum
        centered=x-mu;radius=math.sqrt(float(np.sum(a[:,None]*centered*centered))/(896*weight_sum))
        r['eligibility_gates']=dict(positive_finite_radius=math.isfinite(radius) and radius>1e-12)
        def save(name,value):
            value=np.ascontiguousarray(value,dtype='<f8');assert np.isfinite(value).all()
            p=args.directory/(name+'.npy')
            with p.open('xb') as f:np.save(f,value,allow_pickle=False)
            r['outputs'][name]=dict(path=str(p.resolve()),shape=list(value.shape),dtype='<f8',C_order=True,bytes=p.stat().st_size,sha256=sha(p));guard()
        if r['eligibility_gates']['positive_finite_radius']:
            u=np.column_stack((np.ones(3845),centered/radius));root_a=np.sqrt(a)
            gram=(u@u.T)*(w@w.T);gram*=root_a[:,None];gram*=root_a[None,:];guard()
            trace=float(np.trace(gram));alpha=trace/(10**6-1)
            symmetry=float(np.linalg.norm(gram-gram.T)/np.linalg.norm(gram))
            r['kernel']=dict(rows=3845,exact_x_classes=3166,weighted_occurrences=weight_sum,feature_columns=14352,
                radius=radius,trace=trace,alpha=alpha,alpha_formula='trace(G)/(1000000-1)',
                nominal_real_feature_Gram_condition_upper=1+trace/alpha,symmetry_relative=symmetry,
                condition_scope='Real feature-Gram algebra, not a measured/certified condition2 of rounded saved G')
            r['eligibility_gates'].update(positive_finite_trace_alpha=math.isfinite(trace) and math.isfinite(alpha) and trace>0 and alpha>0,
                symmetry_at_most_1e_minus12=symmetry<=1e-12)
            for name,value in (('FIT_x',x),('FIT_weights',a),('FIT_W',w),('FIT_U',u),('mu',mu),('G',gram)):
                save(name,value)
            write_once(args.directory/'FIT_journal.json',journal);guard()
            if all(r['eligibility_gates'].values()):
                regularized=gram.copy();regularized.flat[::3846]+=alpha
                try:
                    factor=np.linalg.cholesky(regularized);guard()
                    reconstruction=float(np.linalg.norm(factor@factor.T-regularized)/np.linalg.norm(regularized))
                    r['kernel']['Cholesky_relative_reconstruction']=reconstruction
                    r['eligibility_gates']['Cholesky_positive_and_reconstruction_at_most_1e_minus10']=bool((np.diag(factor)>0).all() and reconstruction<=1e-10)
                    save('L',factor)
                except np.linalg.LinAlgError as error:
                    r['eligibility_gates']['Cholesky_positive_and_reconstruction_at_most_1e_minus10']=False
                    r['factorization_fault']=repr(error)
            r['procedure_gates']['finite_C_F64_original_FIT_kernel_arrays']=True
        r['decision']='ELIGIBLE_FOR_ONE_FULL_FIT_CONVEX_COMPILER' if all(r['eligibility_gates'].values()) else 'CLOSE_FULL_FIT_KERNEL_PREREQUISITE'
        r['procedure_gates'].update(exact_bound_used_inputs=True,ALL3845_original_FIT_x_only=True,
            inverse_exact_x_multiplicity_no_silent_deduplication=True,reused_original_choice_and_mass_bytes=True,
            no_source_core_response_J_coefficient_or_old_control_replay=True,no_Torch_GPU_runtime=True)
        r.update(elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='New kernel/factor only; no fitted bank, error, preserved capacity, physical DRAM, whole chat or rate')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],kernel=r.get('kernel'),seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
