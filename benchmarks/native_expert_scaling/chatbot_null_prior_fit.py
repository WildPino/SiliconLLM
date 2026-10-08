"""One ALL-FIT-value/source-null-J convex compiler and exact novel prefix screen."""
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
from chatbot_joint_retained_audit import integer32,routes
from chatbot_coupled_probe import integer64
from chatbot_kernel_fit import triangular
from chatbot_null_prior_kernel import small_solve,saved_array


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_NULL_PRIOR_CONVEX_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_or_core_responses=0,new_source_or_core_J=0,new_optimizer_steps=0,outputs={})
    def guard():
        assert time.monotonic()-start<=300 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
        if args.directory.exists():assert sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=512<<20
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='null_prior_fit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        old=json.loads(Path(b['old_kernel_path']).read_bytes());cov=json.loads(Path(b['kernel_path']).read_bytes())
        x=saved_array(old,'FIT_x',(3845,896));u=saved_array(old,'FIT_U',(3845,897));w=saved_array(old,'FIT_W',(3845,16))
        a=saved_array(old,'FIT_weights',(3845,));mu=saved_array(old,'mu',(896,));root=np.sqrt(a)
        kernel=saved_array(cov,'K_H',(3845,3845));factor=saved_array(cov,'L',(3845,3845));pn=saved_array(cov,'Pi_N',(896,896))
        h=saved_array(cov,'H_leaf',(16,16));lh=saved_array(cov,'L_H_leaf',(16,16))
        alpha=cov['scales']['alpha'];radius=cov['scales']['radius'];lam=cov['scales']['lambda_value'];beta=cov['scales']['beta']
        wa=np.load(b['anchor_mass_path'],mmap_mode='r',allow_pickle=False);assert wa.shape==(16,16)
        def load(path,shape,dtype):
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            assert value.shape==shape and value.dtype==np.dtype(dtype) and value.flags.c_contiguous and np.isfinite(value).all()
            assert value.offset+value.nbytes==Path(path).stat().st_size;return value
        bits=load(b['FIT_source_y_path'],(3845,896),'<u2');assert ((bits&0x7fff)<0x7f80).all()
        y=(bits.astype('<u4')<<16).view('<f4').astype(np.float64)
        shared=load(b['FIT_shared_F64_path'],(3845,896),'<f8');shared32=load(b['FIT_shared_F32_path'],(3845,896),'<f4')
        source=np.load(b['source_J_path'],mmap_mode='r',allow_pickle=False)
        core=load(b['core_J_path'],(16,896,896),'<f8')
        assert source.shape==(32,896,896) and source.dtype==np.dtype('<f8') and source.flags.c_contiguous and np.isfinite(source[:16]).all()
        targets=np.matmul(source[:16]-core,pn);guard()
        target_T=targets.transpose(0,2,1).reshape(16,-1)
        g=(lam/radius)*(wa.T@target_T).reshape(16,896,896)
        theta0=np.zeros((16,897,896));theta0[:,1:]=small_solve(lh,g.reshape(16,-1)).reshape(16,896,896)
        prior_residual=float(np.linalg.norm(h@theta0[:,1:].reshape(16,-1)-g.reshape(16,-1))/np.linalg.norm(g))
        prior_null_gap=float(np.linalg.norm(theta0[:,1:]-np.matmul(pn,theta0[:,1:]))/np.linalg.norm(theta0))
        prior_values=np.zeros((3845,896))
        for e in range(16):
            indices=np.flatnonzero(w[:,e])
            prior_values[indices]+=w[indices,e,None]*(u[indices]@theta0[e]);guard()
        rhs=root[:,None]*(y-shared-prior_values)
        dual=triangular(np,factor,triangular(np,factor,rhs),True);guard()
        dual_residual=float(np.linalg.norm(kernel@dual+dual-rhs)/np.linalg.norm(rhs))
        d=np.empty((16,897,896))
        for e in range(16):d[e]=u.T@((root*w[:,e])[:,None]*dual);guard()
        dnull=np.matmul(pn,d[:,1:]);theta=theta0.copy();theta[:,0]+=d[:,0]/alpha
        theta[:,1:]+=(d[:,1:]-dnull)/alpha+small_solve(lh,dnull.reshape(16,-1)).reshape(16,896,896)
        matrices=theta[:,1:].transpose(0,2,1)/radius;bias=theta[:,0]-matrices@mu
        def save(name,value,dtype='<f8'):
            value=np.ascontiguousarray(value,dtype=dtype);assert np.isfinite(value).all()
            p=args.directory/(name+'.npy')
            with p.open('xb') as stream:np.save(stream,value,allow_pickle=False)
            r['outputs'][name]=dict(path=str(p.resolve()),shape=list(value.shape),dtype=value.dtype.str,C_order=True,bytes=p.stat().st_size,sha256=sha(p));guard()
        save('null_targets_F64',targets);save('prior_Theta0_F64',theta0);save('dual_RHS',rhs);save('dual_V',dual)
        save('affine_A_F64',matrices);save('affine_bias_F64',bias)
        del d,dnull
        f32=np.asarray(matrices,dtype='<f4');bits32=f32.view('<u4')
        encoded=((bits32+0x7fff+((bits32>>16)&1))>>16).astype('<u2');assert ((encoded&0x7fff)<0x7f80).all()
        bias32=np.asarray(bias,dtype='<f4');matrix32=(encoded.astype('<u4')<<16).view('<f4')
        save('affine_A_BF16',encoded,'<u2');save('affine_bias_F32',bias32,'<f4')
        r['leaf_BF16_F32_hashes']=[hashlib.sha256(encoded[e].tobytes()+bias32[e].tobytes()).hexdigest() for e in range(16)]
        assert len(set(r['leaf_BF16_F32_hashes']))==16
        adoption=json.loads(Path(b['adoption_path']).read_bytes())
        sizes={s:sum(c['captured_rows'] for c in adoption['cases'] if c['split']==s) for s in ('fit','development')}
        cached=routes(Path(b['saved_E16_routes']),sizes);ids=np.asarray(cached['fit']['ids']);masses=np.asarray(cached['fit']['mass'],dtype=np.float32)
        x32=np.asarray(x,dtype=np.float32);prediction=np.array(shared,copy=True);prediction32=np.array(shared32,copy=True)
        for slot in range(4):
            for e in range(16):
                indices=np.flatnonzero(ids[:,slot]==e)
                if len(indices):
                    prediction[indices]+=w[indices,e,None]*(x[indices]@matrices[e].T+bias[e])
                    prediction32[indices]+=masses[indices,slot,None]*(x32[indices]@matrix32[e].T+bias32[e])
            guard()
        fold_gap=float(np.linalg.norm(prediction-shared-prior_values-(kernel@dual)/root[:,None])/max(1.,np.linalg.norm(prediction-shared)))
        error=prediction-y;null_theta=np.matmul(pn,theta[:,1:])
        prior_error=(wa@null_theta.reshape(16,-1))/radius-target_T;prior_SSE=float(np.sum(prior_error*prior_error))
        prior_gradient=(beta*((wa.T@wa)@null_theta.reshape(16,-1))).reshape(16,896,896)
        gradient_energy=0.;normal_target_energy=0.
        for e in range(16):
            normal=u.T@((a*w[:,e])[:,None]*error)+alpha*theta[e];normal[1:]+=prior_gradient[e]-g[e]
            gradient_energy+=float(np.sum(normal*normal))
            normal_rhs=u.T@((a*w[:,e])[:,None]*(y-shared));normal_rhs[1:]+=g[e]
            normal_target_energy+=float(np.sum(normal_rhs*normal_rhs));guard()
        gradient_relative=math.sqrt(gradient_energy)/max(1.,math.sqrt(normal_target_energy))
        sse=float(np.sum(a[:,None]*error*error));sse32=float(np.sum(a[:,None]*(prediction32.astype(np.float64)-y)**2))
        norm2=float(np.sum(theta*theta));reference=json.loads(Path(b['original_response_audit']).read_bytes())
        denominator=reference['response_audits']['F32']['fit']['metrics']['unique_x_weighted']['source_energy']
        r['convex_certificate']=dict(alpha=alpha,lambda_value=lam,beta=beta,prior_mean_relative_residual=prior_residual,
            prior_mean_null_relative=prior_null_gap,dual_relative_residual=dual_residual,folded_vs_covariance_prediction_relative=fold_gap,
            normal_gradient_relative=gradient_relative,weighted_value_SSE=sse,source_null_prior_SSE=prior_SSE,coefficient_norm_squared=norm2,
            objective=sse+lam*prior_SSE+alpha*norm2,objective_at_zero_private=float(np.sum(a[:,None]*(shared-y)**2))+lam*float(np.sum(targets*targets)),
            qualification='Numerical minimizer of ONE fixed positive quadratic value/null-J-prior objective; not encoded/native/global optimum')
        r['FIT_metrics']=dict(rows=3845,exact_x_classes=3166,reused_source_energy=denominator,F64_relative_RMS=math.sqrt(sse/denominator),
            encoded_F32_relative_RMS=math.sqrt(sse32/denominator),encoded_F32_error_energy=sse32,
            source_null_prior_RMS_against_original_energy_scale=math.sqrt(prior_SSE/cov['scales']['reused_old_Q_FIT_source_null_J_energy']))
        save('FIT_prediction_F64',prediction);save('FIT_prediction_F32',prediction32,'<f4')
        r['numerical_gate']=max(prior_residual,prior_null_gap)<=1e-10 and dual_residual<=1e-8 and fold_gap<=1e-10 and gradient_relative<=1e-8
        if not r['numerical_gate']:r['decision']='CLOSE_NULL_PRIOR_NUMERICAL_COMPILER'
        else:
            journal=json.loads(Path(b['prefix_journal_path']).read_bytes());assert len(journal)==64
            prefix=load(b['prefix_shared_F64_path'],(64,896),'<f8').copy();prefix32=load(b['prefix_shared_F32_path'],(64,896),'<f4').copy()
            ys=[]
            for i,item in enumerate(journal):
                with Path(item['binary_path']).open('rb') as stream:stream.seek(item['x_byte_offset']);data=stream.read(3584)
                assert len(data)==3584 and hashlib.sha256(data[:1792]).hexdigest()==item['x_SHA256'] and hashlib.sha256(data[1792:]).hexdigest()==item['y_SHA256']
                bx=np.frombuffer(data[:1792],dtype='<u2');ys.append(np.frombuffer(data[1792:],dtype='<u2'))
                row=(bx.astype('<u4')<<16).view('<f4');choices=item['selected_ids']
                mass=np.asarray(struct.unpack('<4f',bytes.fromhex(item['selected_mass_F32_HEX'])),dtype=np.float32)
                prefix[i]+=mass.astype(np.float64)@(matrices[choices]@row.astype(np.float64)+bias[choices])
                prefix32[i]+=mass@(matrix32[choices]@row+bias32[choices]);guard()
            save('novel_prefix64_F64',prefix);save('novel_prefix64_F32',prefix32,'<f4')
            error32=0;error64=0;float_errors32=[];float_errors64=[];float_source=[]
            for i,item in enumerate(journal):
                mult=item['original_multiplicity'];assert mult in (1,2)
                error32+=sum((integer32(p)-integer32(int(yv)<<16))**2 for p,yv in zip(struct.unpack('<896I',prefix32[i].tobytes()),ys[i]))*(2//mult)
                error64+=sum((integer64(p)-(integer32(int(yv)<<16)<<925))**2 for p,yv in zip(struct.unpack('<896Q',prefix[i].tobytes()),ys[i]))*(2//mult)
                target=(ys[i].astype('<u4')<<16).view('<f4')
                float_errors32.append(math.fsum((float(p)-float(v))**2 for p,v in zip(prefix32[i],target))/mult)
                float_errors64.append(math.fsum((float(p)-float(v))**2 for p,v in zip(prefix[i],target))/mult)
                float_source.append(math.fsum(float(v)**2 for v in target)/mult)
            full=int(reference['response_audits']['F32']['development']['exact_necessary_failure']['source_energy_integer'])
            failed=10000*error32>full
            r['novel_prefix']=dict(rows=64,total_novel_rows=1495,F64_relative_RMS=math.sqrt(math.fsum(float_errors64)/math.fsum(float_source)),
                encoded_F32_relative_RMS=math.sqrt(math.fsum(float_errors32)/math.fsum(float_source)),exact_encoded_error_integer=str(error32),
                exact_unrounded_error_integer=str(error64),reused_FULL_source_integer=str(full),
                encoded_full_1pct_RMS_necessarily_fails=failed,unrounded_full_1pct_RMS_necessarily_fails=10000*error64>(full<<1850),
                encoded_prefix_SSE_over_FULL_source_energy=error32/full,unrounded_prefix_SSE_over_FULL_source_energy=error64/(full<<1850),
                common_squared_units_encoded='2^-298',common_squared_units_unrounded='2^-2148',weight_LCM=2,source_full_sum_recomputed=False)
            r['decision']='CLOSE_NULL_PRIOR_CONVEX_NOVEL_FIDELITY' if failed else 'NOVEL_PREFIX_INCONCLUSIVE_NO_FIDELITY_ADMISSION'
        r['complete_budget']=cov['reused_complete_budget']
        r['procedure_gates'].update(ALL_original_FIT_y_and_shared_values_reused=True,only_original16FIT_source_core_J_labels_used=True,
            original_F32_value_mass_and_F64_smooth_prior_mass_distinct=True,actual_saved_factor_triangular_solves_no_inverse=True,
            finite_C_arrays_and_distinct16_encoded_hashes=True,original_full_energy_scales_reused=True,
            no_donor_core_old_bank_optimizer_or_factor_replay=True,no_Torch_GPU=True)
        r.update(elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='One new source-null prior convex bank; local NumPy, no full/native/fresh chatbot/useful n/DRAM/rate admission')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],certificate=r['convex_certificate'],FIT=r['FIT_metrics'],prefix=r.get('novel_prefix'))),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--binding-sha',required=True)
    parser.add_argument('--freeze',required=True);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
