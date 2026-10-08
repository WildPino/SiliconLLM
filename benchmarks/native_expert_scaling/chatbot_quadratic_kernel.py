"""FIRST changed quadratic features/covariance; old linear Gram reused only."""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def main(args):
    import psutil
    started=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_QUADRATIC_KERNEL_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},eligibility_gates={},new_original_BF16_full_forwards=0,new_source_core_H_J_responses=0,
        new_old_linear_geometry_Gram_factor_or_feature_replays=0,new_response_coefficients_or_optimizer_calls=0,outputs={})
    def guard():
        assert time.monotonic()-started<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
        if args.directory.exists():assert sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=512<<20
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='quadratic_kernel'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules and os.environ['OPENBLAS_NUM_THREADS']=='1';args.directory.mkdir()
        old=json.loads(Path(b['old_kernel_path']).read_bytes());factors=json.loads(Path(b['factor_path']).read_bytes())
        def load(item,shape,dtype='<f8'):
            p=Path(item['path']);assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
            a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.shape==shape and a.dtype==np.dtype(dtype) and a.flags.c_contiguous and np.isfinite(a).all()
            assert a.offset+a.nbytes==p.stat().st_size;return a
        def save(name,value):
            a=np.ascontiguousarray(value);assert np.isfinite(a).all();p=args.directory/(name+'.npy')
            with p.open('xb') as stream:np.save(stream,a,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
            r['outputs'][name]=dict(path=str(p.resolve()),shape=list(a.shape),dtype=a.dtype.str,C_order=True,bytes=p.stat().st_size,sha256=sha(p));guard()
        x=load(old['outputs']['FIT_x'],(3845,896));a=load(old['outputs']['FIT_weights'],(3845,));w=load(old['outputs']['FIT_W'],(3845,16))
        mu=load(old['outputs']['mu'],(896,));linear=load(old['outputs']['G'],(3845,3845));weight_sum=math.fsum(float(t) for t in a)
        assert (a>0).all() and abs(weight_sum-3166)<=1e-11 and (w>=0).all()
        radius=old['kernel']['radius'];assert radius==.7645380726589079
        mu32=np.asarray(mu,dtype='<f4');radius32=np.array([radius],dtype='<f4');assert np.isfinite(mu32).all() and radius32[0]>0
        save('mu_F32',mu32);save('radius_F32',radius32)
        z=(x-mu32.astype(np.float64))/float(radius32[0]);qraw=np.empty((3845,256))
        for e in range(16):
            rb=load(factors['outputs']['R_BF16'],(16,16,896),'<u2')[e];tb=load(factors['outputs']['T_BF16'],(16,16,896),'<u2')[e]
            assert ((rb&0x7fff)<0x7f80).all() and ((tb&0x7fff)<0x7f80).all()
            rr=(rb.astype('<u4')<<16).view('<f4').astype('<f8');tt=(tb.astype('<u4')<<16).view('<f4').astype('<f8')
            qraw[:,16*e:16*(e+1)]=w[:,e,None]*((z@rr.T)*(z@tt.T));guard()
            print(json.dumps(dict(parent=e,seconds=time.monotonic()-started)),flush=True)
        save('Q_raw',qraw)
        sigma=np.sqrt(np.sum(a[:,None]*qraw*qraw,axis=0)/weight_sum);save('sigma',sigma)
        r['scales']=dict(weight_sum=weight_sum,old_linear_radius_F64=radius,new_quadratic_radius_F32=float(radius32[0]),sigma_min=float(np.min(sigma)),sigma_max=float(np.max(sigma)),
            center_scope='Old linear G/mu/r unchanged; NEW quadratic z uses promoted F32 center/radius constants')
        gates=r['eligibility_gates'];gates['ALL256_positive_finite_sigma_above_1e_minus12']=bool(np.isfinite(sigma).all() and (sigma>1e-12).all())
        if gates['ALL256_positive_finite_sigma_above_1e_minus12']:
            bq=np.sqrt(a)[:,None]*(qraw/sigma);save('B_Q',bq)
            gram=linear+bq@bq.T;save('G',gram);trace=float(np.trace(gram));alpha=trace/(10**6-1)
            quadtrace=float(np.sum(bq*bq));lineartrace=old['kernel']['trace'];sym=float(np.linalg.norm(gram-gram.T)/np.linalg.norm(gram))
            closure=abs(trace-lineartrace-quadtrace)/trace;nominal_gap=abs(quadtrace-256*weight_sum)/max(quadtrace,1.)
            r['kernel']=dict(rows=3845,old_linear_columns=14352,new_quadratic_columns=256,total_columns=14608,
                reused_linear_trace=lineartrace,quadratic_trace=quadtrace,trace=trace,alpha=alpha,alpha_formula='trace(G_new)/(1000000-1)',
                trace_closure_relative=closure,unit_column_trace_relative_gap=nominal_gap,symmetry_relative=sym,
                nominal_real_condition_upper=1+trace/alpha,condition_scope='Real feature-Gram deduction only, not measured/certified rounded condition2')
            gates.update(positive_finite_trace_alpha=math.isfinite(trace) and math.isfinite(alpha) and trace>0 and alpha>0,
                symmetry_at_most_1e_minus12=sym<=1e-12,trace_closure_at_most_1e_minus12=closure<=1e-12,unit_column_trace_gap_at_most_1e_minus12=nominal_gap<=1e-12)
            if all(gates.values()):
                regularized=gram.copy();regularized.flat[::3846]+=alpha
                try:
                    factor=np.linalg.cholesky(regularized);guard();recon=float(np.linalg.norm(factor@factor.T-regularized)/np.linalg.norm(regularized))
                    r['kernel']['Cholesky_relative_reconstruction']=recon
                    gates['positive_factor_and_reconstruction_at_most_1e_minus10']=bool((np.diag(factor)>0).all() and recon<=1e-10);save('L',factor)
                except np.linalg.LinAlgError as error:
                    gates['positive_factor_and_reconstruction_at_most_1e_minus10']=False;r['factor_fault']=repr(error)
        r['procedure_gates']=dict(bound_actual_inputs_and_source_factors=True,FIRST_quadratic_features_FIT_only=True,
            old_linear_Gram_reused_no_geometry_factor_replay=True,F32_center_scale_fixed_before_features=True,
            fixed256_columns_no_drop_rank_scale_grid=True,no_source_response_labels_coefficients_or_optimizer=True,CPU_only_no_Torch=True)
        r.update(decision=('ELIGIBLE_FOR_ONE_QUADRATIC_CONVEX_COMPILER' if all(gates.values()) else 'CLOSE_QUADRATIC_KERNEL_PREREQUISITE'),
            complete_budget=dict(matrix_MAC=295104512,logical_bytes=592186464,stored_payload_bytes=1080411488,
                complete_matrix_gate=5*295104512<=3*493961216,complete_logical_gate=5*592186464<=3*988067328,physical_DRAM_and_rate='UNMEASURED'),
            elapsed_before_final_serialization=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],scales=r['scales'],kernel=r.get('kernel'))),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
