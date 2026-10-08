"""FIRST changed-feature/scalar/block-covariance audit, no old linear replay."""
import argparse
import datetime as dt
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


def main(args):
    import psutil
    started=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_QUADRATIC_KERNEL_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_core_H_J_responses=0,new_old_linear_geometry_Gram_factor_replays=0,
        new_response_coefficient_or_optimizer_calls=0)
    def guard():
        assert time.monotonic()-started<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='quadratic_kernel_audit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        raw=json.loads(Path(b['kernel_path']).read_bytes());old=json.loads(Path(b['old_kernel_path']).read_bytes());factor=json.loads(Path(b['factor_path']).read_bytes())
        def load(item,shape,dtype='<f8'):
            p=Path(item['path']);assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
            a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.shape==shape and a.dtype==np.dtype(dtype) and a.flags.c_contiguous and np.isfinite(a).all()
            assert a.offset+a.nbytes==p.stat().st_size;return a
        def saved(name,shape,dtype='<f8'):return load(raw['outputs'][name],shape,dtype)
        x=load(old['outputs']['FIT_x'],(3845,896));a=load(old['outputs']['FIT_weights'],(3845,));w=load(old['outputs']['FIT_W'],(3845,16));mu=load(old['outputs']['mu'],(896,))
        linear=load(old['outputs']['G'],(3845,3845));rf=load(factor['outputs']['R_BF16'],(16,16,896),'<u2');tf=load(factor['outputs']['T_BF16'],(16,16,896),'<u2')
        mu32=saved('mu_F32',(896,),'<f4');radius32=saved('radius_F32',(1,),'<f4');qraw=saved('Q_raw',(3845,256));sigma=saved('sigma',(256,));bq=saved('B_Q',(3845,256))
        gram=saved('G',(3845,3845));l=saved('L',(3845,3845));weight_sum=math.fsum(float(z) for z in a)
        expected_mu=b''.join(struct.pack('<f',float(z)) for z in mu);expected_radius=struct.pack('<f',old['kernel']['radius'])
        assert mu32.tobytes()==expected_mu and radius32.tobytes()==expected_radius and (sigma>1e-12).all()
        def promote(bits):
            exp=((bits>>7)&255).astype(np.int32);mant=(bits&127).astype(np.float64);assert (exp<255).all()
            mant+=np.where(exp==0,0.,128.);value=np.ldexp(mant,np.where(exp==0,-133,exp-134));return np.where(bits&0x8000,-value,value)
        def gap(actual,expected,tol=1e-11):
            delta=float(np.linalg.norm(actual-expected));scale=max(float(np.linalg.norm(actual)),float(np.linalg.norm(expected)),1e-20)
            assert delta<=1e-11+tol*scale,(delta,scale);return delta/scale
        z=(x-mu32.astype(np.float64))/float(radius32[0]);feature_gap=0.;scale_gap=0.;weighted_gap=0.
        accum=np.zeros((3845,3845));expected_sigmas=[]
        for e in range(16):
            assert np.array_equal(rf[e,:8],rf[e,8:]) and np.array_equal(rf[e,:8],tf[e,:8])
            gates=promote(rf[e,:8])@z.T;ups=promote(tf[e,8:])@z.T
            expected=w[:,e,None]*np.column_stack((gates.T*gates.T,gates.T*ups.T))
            observed=qraw[:,16*e:16*(e+1)];feature_gap=max(feature_gap,gap(observed,expected))
            scales=np.array([math.sqrt(math.fsum(float(aa)*float(qq)**2 for aa,qq in zip(a,col))/weight_sum) for col in observed.T])
            expected_sigmas.extend(scales);saved_sigma=sigma[16*e:16*(e+1)];scale_gap=max(scale_gap,gap(saved_sigma,scales))
            block=np.column_stack([np.array([math.sqrt(float(aa))*float(qq)/float(ss) for aa,qq in zip(a,col)]) for col,ss in zip(observed.T,scales)])
            weighted_gap=max(weighted_gap,gap(bq[:,16*e:16*(e+1)],block))
            # Separate sixteen feature blocks, unlike producer single256-column Gram.
            accum+=block@block.T;guard()
        expected_gram=linear+accum;gram_gap=gap(gram,expected_gram)
        linear_trace=old['kernel']['trace'];qtrace=math.fsum(float(z)**2 for z in bq.flat);trace=math.fsum(float(z) for z in np.diag(gram));alpha=trace/(10**6-1)
        for key,val in (('reused_linear_trace',linear_trace),('quadratic_trace',qtrace),('trace',trace),('alpha',alpha)):
            assert math.isclose(raw['kernel'][key],val,rel_tol=2e-12,abs_tol=1e-12),key
        assert (np.diag(l)>0).all() and np.array_equal(l,np.tril(l))
        expected_gram.flat[::3846]+=raw['kernel']['alpha'];recon=gap(l@l.T,expected_gram,1e-10)
        assert abs(qtrace-256*weight_sum)/qtrace<=1e-12 and abs(trace-linear_trace-qtrace)/trace<=1e-12
        assert all(raw['eligibility_gates'].values()) and raw['decision']=='ELIGIBLE_FOR_ONE_QUADRATIC_CONVEX_COMPILER'
        r.update(decision='SAVED_QUADRATIC_KERNEL_INDEPENDENTLY_VERIFIED',maximum_quadratic_feature_relative_gap=feature_gap,
            maximum_scalar_sigma_relative_gap=scale_gap,maximum_weighted_feature_relative_gap=weighted_gap,
            ALL_augmented_Gram_relative_gap=gram_gap,positive_factor_relative_reconstruction=recon,trace=trace,alpha=alpha,
            scope='FIRST new quadratic-feature/block-Gram/scalar audit; old linear Gram reused, no old factor or response acquisition')
        r['procedure_gates']=dict(bound_actual_inputs_and_source_factor_bytes=True,independent_BF16_and_F32_constants=True,
            ALL_new_FIT_quadratic_features_scalar_scales=True,ALL_block_Gram_and_positive_factor_reconstruction=True,
            no_original_source_old_linear_Gram_or_response_replay=True,CPU_only_no_Torch=True)
        r.update(elapsed_before_final_serialization=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],feature_gap=feature_gap,Gram_gap=gram_gap)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
