"""FIRST saved augmented-minimum/codec/error audit; no field evaluation or solve."""
import argparse
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


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_QUADRATIC_CONVEX_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_core_or_field_responses=0,new_features_factors_solves_or_optimizers=0)
    def guard():assert time.monotonic()-start<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    try:
        assert sha(args.binding)==args.binding_sha and sys.version_info[:3]==(3,12,10)
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='quadratic_fit_audit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        raw=json.loads(Path(b['fit_path']).read_bytes());kernel=json.loads(Path(b['kernel_path']).read_bytes());old=json.loads(Path(b['old_kernel_path']).read_bytes())
        assert raw['numerical_gate'] and raw['decision'] in ('CLOSE_QUADRATIC_CONVEX_NOVEL_FIDELITY','ELIGIBLE_FOR_SEPARATE_FULL_NOVEL_SCREEN')
        def load(path,shape,dtype='<f8'):
            v=np.load(path,mmap_mode='r',allow_pickle=False)
            assert v.shape==shape and v.dtype==np.dtype(dtype) and v.flags.c_contiguous and np.isfinite(v).all()
            assert v.offset+v.nbytes==Path(path).stat().st_size;return v
        def array(report,name,shape,dtype='<f8'):return load(report['outputs'][name]['path'],shape,dtype)
        a=array(old,'FIT_weights',(3845,));w=array(old,'FIT_W',(3845,16));u=array(old,'FIT_U',(3845,897));mu=array(old,'mu',(896,));root=np.sqrt(a)
        g=array(kernel,'G',(3845,3845));bq=array(kernel,'B_Q',(3845,256));sigma=array(kernel,'sigma',(256,));alpha=kernel['kernel']['alpha'];radius=old['kernel']['radius']
        ybits=load(b['FIT_source_y_path'],(3845,896),'<u2');exponent=(ybits>>7)&255;fraction=ybits&127
        assert (exponent!=255).all()
        y=np.ldexp(np.where(exponent==0,fraction,128+fraction).astype(np.float64),np.where(exponent==0,-133,exponent.astype(np.int32)-134))
        y*=np.where((ybits&32768)!=0,-1.,1.)
        shared=load(b['FIT_shared_F64_path'],(3845,896));rhs=load(b['RHS_path'],(3845,896));assert np.array_equal(rhs,root[:,None]*(y-shared))
        dual=array(raw,'dual_V',(3845,896));matrices=array(raw,'affine_A_F64',(16,896,896));bias=array(raw,'affine_bias_F64',(16,896));c=array(raw,'quadratic_C_F64',(16,896,16))
        pred=array(raw,'FIT_prediction_F64',(3845,896));pred32=array(raw,'FIT_prediction_F32',(3845,896),'<f4')
        theta=np.empty((16,897,896));theta[:,1:]=matrices.transpose(0,2,1)*radius;theta[:,0]=bias+matrices@mu
        theta_q=np.ascontiguousarray(c.transpose(0,2,1).reshape(256,896)*sigma[:,None])
        private=g@dual;linear=float(np.linalg.norm(private+alpha*dual-rhs)/np.linalg.norm(rhs))
        fold=float(np.linalg.norm(pred-shared-private/root[:,None])/max(1.,np.linalg.norm(private/root[:,None])))
        recovery=0.;normal_energy=0.;target_energy=0.;weighted=root[:,None]*(pred-y)
        for e in range(16):
            coefficient=u.T@((root*w[:,e])[:,None]*dual)
            recovery=max(recovery,float(np.linalg.norm(coefficient-theta[e])/max(1.,np.linalg.norm(coefficient))))
            normal=u.T@((a*w[:,e])[:,None]*(pred-y))+alpha*theta[e]
            target=u.T@((root*w[:,e])[:,None]*rhs)
            normal_energy+=float(np.sum(normal*normal));target_energy+=float(np.sum(target*target));guard()
        cq=bq.T@dual;recovery=max(recovery,float(np.linalg.norm(cq-theta_q)/max(1.,np.linalg.norm(cq))))
        nq=bq.T@weighted+alpha*theta_q;tq=bq.T@rhs
        normal_energy+=float(np.sum(nq*nq));target_energy+=float(np.sum(tq*tq))
        norm2=math.fsum(float(v)**2 for v in theta.flat)+math.fsum(float(v)**2 for v in theta_q.flat);guard()
        normal=math.sqrt(normal_energy)/max(1.,math.sqrt(target_energy)+alpha*math.sqrt(norm2))
        assert max(linear,fold,recovery,normal)<=1e-10
        # Independent scalar summation of saved prediction errors, not field re-evaluation.
        sse=math.fsum(float(a[i])*math.fsum((float(p)-float(v))**2 for p,v in zip(pred[i],y[i])) for i in range(3845));guard()
        sse32=math.fsum(float(a[i])*math.fsum((float(p)-float(v))**2 for p,v in zip(pred32[i],y[i])) for i in range(3845));guard()
        codec=math.fsum(float(a[i])*math.fsum((float(p)-float(v))**2 for p,v in zip(pred32[i],pred[i])) for i in range(3845));guard()
        original=json.loads(Path(b['original_response_audit']).read_bytes());den=original['response_audits']['F32']['fit']['metrics']['unique_x_weighted']['source_energy']
        def close(actual,want):assert abs(actual-want)<=1e-10*max(1.,abs(want)),(actual,want)
        close(sse,raw['convex_certificate']['weighted_response_SSE']);close(norm2,raw['convex_certificate']['coefficient_norm_squared']);close(sse+alpha*norm2,raw['convex_certificate']['objective'])
        for actual,name in ((sse,'F64_error_energy'),(sse32,'encoded_F32_error_energy'),(math.sqrt(sse/den),'F64_relative_RMS'),
            (math.sqrt(sse32/den),'encoded_F32_relative_RMS'),(math.sqrt(codec/den),'encoded_difference_relative_RMS')):close(actual,raw['FIT_metrics'][name])
        assert raw['FIT_metrics']['reused_source_energy']==den
        for name,value,shape in (('affine_A_BF16',matrices,(16,896,896)),('quadratic_C_BF16',c,(16,896,16))):
            bits=np.asarray(value,dtype='<f4').view('<u4');hi=bits>>16;lo=bits&65535
            expected=(hi+((lo>32768)|((lo==32768)&((hi&1)!=0)))).astype('<u2')
            assert np.array_equal(expected,array(raw,name,shape,'<u2'));guard()
        bias32=array(raw,'affine_bias_F32',(16,896),'<f4');assert np.array_equal(np.asarray(bias,dtype='<f4'),bias32)
        rb=load(b['R_path'],(16,16,896),'<u2');tb=load(b['T_path'],(16,16,896),'<u2')
        ae=array(raw,'affine_A_BF16',(16,896,896),'<u2');ce=array(raw,'quadratic_C_BF16',(16,896,16),'<u2')
        mu32=array(kernel,'mu_F32',(896,),'<f4');radius32=array(kernel,'radius_F32',(1,),'<f4')
        hashes=[hashlib.sha256(ae[e].tobytes()+bias32[e].tobytes()+rb[e].tobytes()+tb[e].tobytes()+ce[e].tobytes()+mu32.tobytes()+radius32.tobytes()).hexdigest() for e in range(16)]
        assert hashes==raw['leaf_BF16_F32_hashes'] and len(set(hashes))==16
        prefix=array(raw,'novel_prefix64_F64',(64,896));prefix32=array(raw,'novel_prefix64_F32',(64,896),'<f4');journal=json.loads(Path(b['prefix_journal_path']).read_bytes());assert len(journal)==64
        def units(value,power):
            n,d=float(value).as_integer_ratio();shift=d.bit_length()-1;assert d==1<<shift and shift<=power;return n<<(power-shift)
        n32=0;n64=0;energies=[];errors32=[];errors64=[]
        for i,item in enumerate(journal):
            with Path(item['binary_path']).open('rb') as f:f.seek(item['x_byte_offset']+1792);data=f.read(1792)
            assert hashlib.sha256(data).hexdigest()==item['y_SHA256']
            target=struct.unpack('<896f',struct.pack('<896I',*(v<<16 for v in struct.unpack('<896H',data))));mult=item['original_multiplicity'];assert mult in (1,2)
            n32+=sum((units(p,149)-units(v,149))**2 for p,v in zip(prefix32[i],target))*(2//mult)
            n64+=sum((units(p,1074)-units(v,1074))**2 for p,v in zip(prefix[i],target))*(2//mult)
            energies.append(math.fsum(v*v for v in target)/mult)
            errors32.append(math.fsum((float(p)-v)**2 for p,v in zip(prefix32[i],target))/mult)
            errors64.append(math.fsum((float(p)-v)**2 for p,v in zip(prefix[i],target))/mult);guard()
        full=int(original['response_audits']['F32']['development']['exact_necessary_failure']['source_energy_integer']);proof=raw['novel_prefix']
        fail32=10000*n32>full;fail64=10000*n64>(full<<1850)
        assert str(n32)==proof['exact_encoded_error_integer'] and str(n64)==proof['exact_unrounded_error_integer'] and str(full)==proof['reused_FULL_source_integer']
        assert fail32==proof['encoded_full_1pct_RMS_necessarily_fails'] and fail64==proof['unrounded_full_1pct_RMS_necessarily_fails']
        assert raw['decision']==('CLOSE_QUADRATIC_CONVEX_NOVEL_FIDELITY' if fail32 else 'ELIGIBLE_FOR_SEPARATE_FULL_NOVEL_SCREEN')
        rms32=math.sqrt(math.fsum(errors32)/math.fsum(energies));rms64=math.sqrt(math.fsum(errors64)/math.fsum(energies))
        close(rms32,proof['encoded_F32_relative_RMS']);close(rms64,proof['F64_relative_RMS'])
        r['independent_certificate']=dict(dual_relative=linear,folded_vs_covariance_relative=fold,coefficient_from_dual_max_relative=recovery,
            ALL_linear_quadratic_normal_gradient_relative=normal,weighted_value_SSE=sse,coefficient_norm_squared=norm2,objective=sse+alpha*norm2,
            FIT_F64_relative_RMS=math.sqrt(sse/den),FIT_encoded_relative_RMS=math.sqrt(sse32/den),novel64_F64_relative_RMS=rms64,novel64_encoded_relative_RMS=rms32)
        r['exact_proofs']=dict(encoded_prefix_error=str(n32),unrounded_prefix_error=str(n64),reused_FULL_source_integer=str(full),
            encoded_full_1pct_necessarily_fails=fail32,unrounded_full_1pct_necessarily_fails=fail64,source_FULL_denominators_recomputed=False)
        r['procedure_gates'].update(cached_RHS_and_independent_BF16_source_promotion=True,ALL_augmented_coefficient_primal_dual_fold_certificates=True,
            independent_scalar_SSE_objective_and_codec_bytes=True,independent_integer_ratio_necessary_decision=True,no_source_core_field_feature_factor_solve_replay=True,CPU_only_no_Torch=True)
        r.update(decision='SAVED_QUADRATIC_CONVEX_RESULT_INDEPENDENTLY_VERIFIED',elapsed_before_final_serialization=time.monotonic()-start,
            OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='Saved minimum/encoding/errors certified; no independent F32 field execution/native parity/fresh quality')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],certificate=r['independent_certificate'],necessary_failure=fail32)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
