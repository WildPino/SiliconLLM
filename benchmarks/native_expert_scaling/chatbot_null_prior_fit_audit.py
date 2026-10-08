"""First saved source-null convex bank audit; no response, factor or solve replay."""
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
    r=dict(schema='QWEN_NULL_PRIOR_CONVEX_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_core_or_field_responses=0,new_J_factors_or_solves=0)
    def guard():assert time.monotonic()-start<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    try:
        assert sha(args.binding)==args.binding_sha and sys.version_info[:3]==(3,12,10)
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='null_prior_fit_audit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        raw=json.loads(Path(b['fit_path']).read_bytes());cov=json.loads(Path(b['kernel_path']).read_bytes());old=json.loads(Path(b['old_kernel_path']).read_bytes())
        assert raw['numerical_gate'] and raw['decision']=='CLOSE_NULL_PRIOR_CONVEX_NOVEL_FIDELITY'
        def load(path,shape,dtype='<f8'):
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            assert value.shape==shape and value.dtype==np.dtype(dtype) and value.flags.c_contiguous and np.isfinite(value).all()
            assert value.offset+value.nbytes==Path(path).stat().st_size;return value
        def array(report,name,shape,dtype='<f8'):
            item=report['outputs'][name];assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
            return load(item['path'],shape,dtype)
        u=array(old,'FIT_U',(3845,897));w=array(old,'FIT_W',(3845,16));a=array(old,'FIT_weights',(3845,));mu=array(old,'mu',(896,));root=np.sqrt(a)
        k=array(cov,'K_H',(3845,3845));pn=array(cov,'Pi_N',(896,896));h=array(cov,'H_leaf',(16,16))
        matrices=array(raw,'affine_A_F64',(16,896,896));bias=array(raw,'affine_bias_F64',(16,896))
        theta0=array(raw,'prior_Theta0_F64',(16,897,896));targets=array(raw,'null_targets_F64',(16,896,896))
        rhs=array(raw,'dual_RHS',(3845,896));dual=array(raw,'dual_V',(3845,896));pred=array(raw,'FIT_prediction_F64',(3845,896));pred32=array(raw,'FIT_prediction_F32',(3845,896),'<f4')
        ybits=load(b['FIT_source_y_path'],(3845,896),'<u2');y=(ybits.astype('<u4')<<16).view('<f4').astype(np.float64)
        shared=load(b['FIT_shared_F64_path'],(3845,896));wa=load(b['anchor_mass_path'],(16,16))
        alpha=cov['scales']['alpha'];lam=cov['scales']['lambda_value'];beta=cov['scales']['beta'];radius=cov['scales']['radius']
        source=np.load(b['source_J_path'],mmap_mode='r',allow_pickle=False);assert source.shape==(32,896,896)
        core=load(b['core_J_path'],(16,896,896));difference=source[:16]-core
        target_null=float(np.linalg.norm(np.matmul(targets,pn)-targets)/np.linalg.norm(targets))
        target_projection=float(np.linalg.norm(np.matmul(difference-targets,pn))/np.linalg.norm(difference))
        assert max(target_null,target_projection)<=1e-10;del difference
        target_T=targets.transpose(0,2,1).reshape(16,-1);g=(lam/radius)*(wa.T@target_T).reshape(16,896,896)
        prior_eq=float(np.linalg.norm(h@theta0[:,1:].reshape(16,-1)-g.reshape(16,-1))/np.linalg.norm(g))
        prior_null=float(np.linalg.norm(np.matmul(pn,theta0[:,1:])-theta0[:,1:])/np.linalg.norm(theta0))
        assert not np.any(theta0[:,0]) and max(prior_eq,prior_null)<=1e-10
        # Independent dense linear-design RHS certificate for the saved prior mean.
        expected_rhs=np.array(y-shared)
        for e in range(16):expected_rhs-=w[:,e,None]*(u@theta0[e]);guard()
        expected_rhs*=root[:,None]
        rhs_gap=float(np.linalg.norm(expected_rhs-rhs)/np.linalg.norm(rhs));assert rhs_gap<=1e-10
        del expected_rhs
        theta=np.empty((16,897,896));theta[:,1:]=matrices.transpose(0,2,1)*radius;theta[:,0]=bias+matrices@mu
        null_theta=np.matmul(pn,theta[:,1:]);delta=theta-theta0;null_delta=np.matmul(pn,delta[:,1:])
        precision_delta=(beta*((wa.T@wa)@null_delta.reshape(16,-1))).reshape(16,896,896)
        prior_gradient=(beta*((wa.T@wa)@null_theta.reshape(16,-1))).reshape(16,896,896)
        gradient_energy=0.;target_energy=0.;recovery_error=0.;recovery_energy=0.
        for e in range(16):
            normal=u.T@((a*w[:,e])[:,None]*(pred-y))+alpha*theta[e];normal[1:]+=prior_gradient[e]-g[e]
            normal_rhs=u.T@((a*w[:,e])[:,None]*(y-shared));normal_rhs[1:]+=g[e]
            gradient_energy+=float(np.sum(normal*normal));target_energy+=float(np.sum(normal_rhs*normal_rhs))
            recovered=u.T@((root*w[:,e])[:,None]*dual)
            equation=alpha*delta[e]-recovered;equation[1:]+=precision_delta[e]
            recovery_error+=float(np.sum(equation*equation));recovery_energy+=float(np.sum(recovered*recovered));guard()
        gradient=math.sqrt(gradient_energy)/max(1.,math.sqrt(target_energy))
        recovery=math.sqrt(recovery_error)/max(1.,math.sqrt(recovery_energy))
        linear=float(np.linalg.norm(k@dual+dual-rhs)/np.linalg.norm(rhs))
        private_dual=(k@dual)/root[:,None]
        # Audit equation, not another evaluation of the folded bank's response operator.
        implied_prior=y-shared-rhs/root[:,None]
        folded=float(np.linalg.norm(pred-shared-implied_prior-private_dual)/max(1.,np.linalg.norm(pred-shared)))
        assert max(linear,gradient,recovery)<=1e-8 and folded<=1e-10
        sse=math.fsum(math.fsum((float(p)-float(v))**2 for p,v in zip(pred[i],y[i]))*float(a[i]) for i in range(3845));guard()
        sse32=math.fsum(math.fsum((float(p)-float(v))**2 for p,v in zip(pred32[i],y[i]))*float(a[i]) for i in range(3845));guard()
        prior_error=(wa@null_theta.reshape(16,-1))/radius-target_T
        prior_sse=math.fsum(float(v)*float(v) for v in prior_error.flat);norm2=math.fsum(float(v)*float(v) for v in theta.flat);guard()
        objective=sse+lam*prior_sse+alpha*norm2
        cert=raw['convex_certificate']
        for actual,name in ((sse,'weighted_value_SSE'),(prior_sse,'source_null_prior_SSE'),(norm2,'coefficient_norm_squared'),(objective,'objective')):
            assert abs(actual-cert[name])<=1e-10*max(1.,abs(actual))
        original=json.loads(Path(b['original_response_audit']).read_bytes());denominator=original['response_audits']['F32']['fit']['metrics']['unique_x_weighted']['source_energy']
        fit64=math.sqrt(sse/denominator);fit32=math.sqrt(sse32/denominator)
        assert abs(fit64-raw['FIT_metrics']['F64_relative_RMS'])<=1e-12 and abs(fit32-raw['FIT_metrics']['encoded_F32_relative_RMS'])<=1e-12
        physical=array(raw,'affine_A_BF16',(16,896,896),'<u2');bias32=array(raw,'affine_bias_F32',(16,896),'<f4')
        bits=np.asarray(matrices,dtype='<f4').view('<u4');high=bits>>16;low=bits&65535
        expected=(high+((low>32768)|((low==32768)&((high&1)!=0)))).astype('<u2')
        assert np.array_equal(expected,physical) and np.array_equal(np.asarray(bias,dtype='<f4'),bias32)
        hashes=[hashlib.sha256(physical[e].tobytes()+bias32[e].tobytes()).hexdigest() for e in range(16)]
        assert hashes==raw['leaf_BF16_F32_hashes'] and len(set(hashes))==16
        prefix=array(raw,'novel_prefix64_F64',(64,896));prefix32=array(raw,'novel_prefix64_F32',(64,896),'<f4')
        journal=json.loads(Path(b['prefix_journal_path']).read_bytes());assert len(journal)==64
        def units(value,power):
            numerator,den=float(value).as_integer_ratio();shift=den.bit_length()-1
            assert den==1<<shift and shift<=power;return numerator<<(power-shift)
        numerator32=0;numerator64=0;energies=[];errors32=[];errors64=[]
        for i,item in enumerate(journal):
            with Path(item['binary_path']).open('rb') as stream:stream.seek(item['x_byte_offset']+1792);data=stream.read(1792)
            assert hashlib.sha256(data).hexdigest()==item['y_SHA256']
            target=struct.unpack('<896f',struct.pack('<896I',*(v<<16 for v in struct.unpack('<896H',data))));mult=item['original_multiplicity'];assert mult in (1,2)
            numerator32+=sum((units(p,149)-units(v,149))**2 for p,v in zip(prefix32[i],target))*(2//mult)
            numerator64+=sum((units(p,1074)-units(v,1074))**2 for p,v in zip(prefix[i],target))*(2//mult)
            energies.append(math.fsum(v*v for v in target)/mult)
            errors32.append(math.fsum((float(p)-v)**2 for p,v in zip(prefix32[i],target))/mult)
            errors64.append(math.fsum((float(p)-v)**2 for p,v in zip(prefix[i],target))/mult)
        full=int(original['response_audits']['F32']['development']['exact_necessary_failure']['source_energy_integer'])
        proof=raw['novel_prefix'];assert str(numerator32)==proof['exact_encoded_error_integer'] and str(numerator64)==proof['exact_unrounded_error_integer']
        assert str(full)==proof['reused_FULL_source_integer'] and 10000*numerator32>full and 10000*numerator64>(full<<1850)
        novel32=math.sqrt(math.fsum(errors32)/math.fsum(energies));novel64=math.sqrt(math.fsum(errors64)/math.fsum(energies))
        assert abs(novel32-proof['encoded_F32_relative_RMS'])<=1e-12 and abs(novel64-proof['F64_relative_RMS'])<=1e-12
        r['independent_certificate']=dict(null_target_and_projection_relative=max(target_null,target_projection),prior_mean_relative=prior_eq,
            prior_mean_null_relative=prior_null,prior_mean_design_RHS_relative=rhs_gap,dual_relative=linear,recovered_coefficient_precision_relative=recovery,normal_gradient_relative=gradient,
            folded_vs_covariance_relative=folded,weighted_value_SSE=sse,source_null_prior_SSE=prior_sse,objective=objective,
            FIT_F64_relative_RMS=fit64,FIT_encoded_relative_RMS=fit32,novel64_F64_relative_RMS=novel64,novel64_encoded_relative_RMS=novel32)
        r['exact_proofs']=dict(encoded_prefix_error=str(numerator32),unrounded_prefix_error=str(numerator64),
            reused_FULL_source_integer=str(full),encoded_full_1pct_necessarily_fails=True,unrounded_full_1pct_necessarily_fails=True,source_FULL_denominators_recomputed=False)
        r['procedure_gates'].update(source_null_target_and_prior_mean_verified=True,ALL_folded_coefficient_dual_and_primal_certificates=True,
            independent_scalar_FIT_and_prefix_errors=True,independent_nearest_even_codec_bytes=True,independent_integer_ratio_necessary_failures=True,
            no_response_J_factor_solve_or_optimizer_replay=True)
        r.update(decision='SAVED_NULL_PRIOR_CONVEX_FAILURE_INDEPENDENTLY_VERIFIED',elapsed_before_final_serialization=time.monotonic()-start,
            OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],certificate=r['independent_certificate'])),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--binding-sha',required=True)
    parser.add_argument('--freeze',required=True);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
