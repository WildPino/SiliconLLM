"""ONE changed quadratic response compiler; immutable source/core values reused."""
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
from chatbot_kernel_fit import triangular
from chatbot_joint_retained_audit import routes,integer32
from chatbot_coupled_probe import integer64


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_QUADRATIC_CONVEX_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},outputs={},new_original_BF16_full_forwards=0,new_source_core_H_J_responses=0,
        new_optimizer_steps=0,new_old_kernel_or_response_replays=0)
    def guard():
        assert time.monotonic()-start<=300 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
        if args.directory.exists():assert sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=512<<20
    try:
        assert sha(args.binding)==args.binding_sha and sys.version_info[:3]==(3,12,10)
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='quadratic_fit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        def load(path,shape,dtype='<f8'):
            v=np.load(path,mmap_mode='r',allow_pickle=False)
            assert v.shape==shape and v.dtype==np.dtype(dtype) and v.flags.c_contiguous and np.isfinite(v).all()
            assert v.offset+v.nbytes==Path(path).stat().st_size;return v
        def array(raw,name,shape,dtype='<f8'):return load(raw['outputs'][name]['path'],shape,dtype)
        def save(name,value,dtype='<f8'):
            v=np.ascontiguousarray(value,dtype=dtype);assert np.isfinite(v).all();p=args.directory/(name+'.npy')
            with p.open('xb') as f:np.save(f,v,allow_pickle=False)
            r['outputs'][name]=dict(path=str(p.resolve()),shape=list(v.shape),dtype=v.dtype.str,C_order=True,bytes=p.stat().st_size,sha256=sha(p));guard()
        old=json.loads(Path(b['old_kernel_path']).read_bytes());kernel=json.loads(Path(b['kernel_path']).read_bytes())
        x=array(old,'FIT_x',(3845,896));a=array(old,'FIT_weights',(3845,));w=array(old,'FIT_W',(3845,16));u=array(old,'FIT_U',(3845,897));mu=array(old,'mu',(896,))
        radius=old['kernel']['radius'];alpha=kernel['kernel']['alpha'];root=np.sqrt(a)
        gram=array(kernel,'G',(3845,3845));factor=array(kernel,'L',(3845,3845));bq=array(kernel,'B_Q',(3845,256));qraw=array(kernel,'Q_raw',(3845,256));sigma=array(kernel,'sigma',(256,))
        mu32=array(kernel,'mu_F32',(896,),'<f4');radius32=array(kernel,'radius_F32',(1,),'<f4')[0]
        ybits=load(b['FIT_source_y_path'],(3845,896),'<u2');assert ((ybits&0x7fff)<0x7f80).all()
        y=(ybits.astype('<u4')<<16).view('<f4').astype(np.float64)
        shared=load(b['FIT_shared_F64_path'],(3845,896));shared32=load(b['FIT_shared_F32_path'],(3845,896),'<f4');rhs=load(b['RHS_path'],(3845,896))
        assert np.array_equal(rhs,root[:,None]*(y-shared))
        intermediate=triangular(np,factor,rhs);dual=triangular(np,factor,intermediate,True);del intermediate;guard()
        private_weighted=gram@dual;dual_res=float(np.linalg.norm(private_weighted+alpha*dual-rhs)/np.linalg.norm(rhs))
        theta=np.empty((16,897,896))
        for e in range(16):
            theta[e]=u.T@((root*w[:,e])[:,None]*dual);guard()
        theta_q=bq.T@dual;matrices=np.ascontiguousarray(theta[:,1:].transpose(0,2,1)/radius);bias=theta[:,0]-matrices@mu
        c=np.ascontiguousarray((theta_q/sigma[:,None]).reshape(16,16,896).transpose(0,2,1))
        save('dual_V',dual);save('affine_A_F64',matrices);save('affine_bias_F64',bias);save('quadratic_C_F64',c)
        def bf16(value):
            f32=np.asarray(value,dtype='<f4');assert np.isfinite(f32).all();bits=f32.view('<u4')
            encoded=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2');assert ((encoded&0x7fff)<0x7f80).all();return encoded
        ae=bf16(matrices);ce=bf16(c);bias32=np.asarray(bias,dtype='<f4')
        save('affine_A_BF16',ae,'<u2');save('affine_bias_F32',bias32,'<f4');save('quadratic_C_BF16',ce,'<u2')
        a32=(ae.astype('<u4')<<16).view('<f4');c32=(ce.astype('<u4')<<16).view('<f4')
        rb=load(b['R_path'],(16,16,896),'<u2');tb=load(b['T_path'],(16,16,896),'<u2')
        r32=(rb.astype('<u4')<<16).view('<f4');t32=(tb.astype('<u4')<<16).view('<f4');r64=r32.astype(np.float64);t64=t32.astype(np.float64)
        r['leaf_BF16_F32_hashes']=[hashlib.sha256(ae[e].tobytes()+bias32[e].tobytes()+rb[e].tobytes()+tb[e].tobytes()+ce[e].tobytes()+mu32.tobytes()+np.asarray([radius32],dtype='<f4').tobytes()).hexdigest() for e in range(16)]
        assert len(set(r['leaf_BF16_F32_hashes']))==16
        adoption=json.loads(Path(b['adoption_path']).read_bytes());sizes={s:sum(v['captured_rows'] for v in adoption['cases'] if v['split']==s) for s in ('fit','development')}
        saved=routes(Path(b['saved_E16_routes']),sizes);ids=np.asarray(saved['fit']['ids']);mass=np.asarray(saved['fit']['mass'],dtype='<f4')
        recovered_w=np.zeros((3845,16))
        for slot in range(4):recovered_w[np.arange(3845),ids[:,slot]]+=mass[:,slot]
        assert np.array_equal(recovered_w,w)
        x32=np.asarray(x,dtype='<f4');z32=(x32-mu32)/radius32
        pred64=np.array(shared,copy=True);pred32=np.array(shared32,copy=True)
        for slot in range(4):
            for e in range(16):
                indices=np.flatnonzero(ids[:,slot]==e)
                if len(indices):
                    # Q_raw already contains the immutable occurrence-specific W.
                    part64=w[indices,e,None]*(x[indices]@matrices[e].T+bias[e])+qraw[indices,16*e:16*(e+1)]@c[e].T
                    q32=(z32[indices]@r32[e].T)*(z32[indices]@t32[e].T)
                    leaf32=(x32[indices]@a32[e].T+bias32[e])+q32@c32[e].T
                    pred64[indices]+=part64;pred32[indices]+=mass[indices,slot,None]*leaf32
            guard()
        fold=float(np.linalg.norm(pred64-shared-private_weighted/root[:,None])/max(1.,np.linalg.norm(private_weighted/root[:,None])))
        error=pred64-y;weighted=root[:,None]*error;grad_energy=0.;target_energy=0.
        for e in range(16):
            normal=u.T@((a*w[:,e])[:,None]*error)+alpha*theta[e]
            target=u.T@((root*w[:,e])[:,None]*rhs)
            grad_energy+=float(np.sum(normal*normal));target_energy+=float(np.sum(target*target));guard()
        normal_q=bq.T@weighted+alpha*theta_q;target_q=bq.T@rhs
        grad_energy+=float(np.sum(normal_q*normal_q));target_energy+=float(np.sum(target_q*target_q))
        norm2=float(np.sum(theta*theta)+np.sum(theta_q*theta_q))
        normal_rel=math.sqrt(grad_energy)/max(1.,math.sqrt(target_energy)+alpha*math.sqrt(norm2))
        original=json.loads(Path(b['original_response_audit']).read_bytes());den=original['response_audits']['F32']['fit']['metrics']['unique_x_weighted']['source_energy']
        sse=float(np.sum(a[:,None]*error*error));sse32=float(np.sum(a[:,None]*(pred32.astype(np.float64)-y)**2))
        r['convex_certificate']=dict(alpha=alpha,dual_relative_residual=dual_res,folded_vs_kernel_relative=fold,normal_gradient_relative=normal_rel,
            weighted_response_SSE=sse,coefficient_norm_squared=norm2,objective=sse+alpha*norm2,
            objective_at_zero_private=float(np.sum(a[:,None]*(shared-y)**2)),qualification='Unique minimizer of ONE fixed augmented ridge objective, subject to numerical certificates; no exact mathematical minimizer claim')
        r['numerical_gate']=max(dual_res,fold,normal_rel)<=1e-10
        r['FIT_metrics']=dict(rows=3845,exact_x_classes=3166,reused_source_energy=den,F64_relative_RMS=math.sqrt(sse/den),
            encoded_F32_relative_RMS=math.sqrt(sse32/den),F64_error_energy=sse,encoded_F32_error_energy=sse32,
            encoded_difference_relative_RMS=float(np.linalg.norm(root[:,None]*(pred32.astype(np.float64)-pred64))/math.sqrt(den)))
        save('FIT_prediction_F64',pred64);save('FIT_prediction_F32',pred32,'<f4')
        print(json.dumps(dict(stage='FIT',numerical_gate=r['numerical_gate'],metrics=r['FIT_metrics'],seconds=time.monotonic()-start)),flush=True)
        if not r['numerical_gate']:r['decision']='CLOSE_QUADRATIC_NUMERICAL_COMPILER'
        else:
            journal=json.loads(Path(b['prefix_journal_path']).read_bytes());assert len(journal)==64
            px=[];py=[]
            for item in journal:
                with Path(item['binary_path']).open('rb') as f:f.seek(item['x_byte_offset']);data=f.read(3584)
                assert len(data)==3584 and hashlib.sha256(data[:1792]).hexdigest()==item['x_SHA256'] and hashlib.sha256(data[1792:]).hexdigest()==item['y_SHA256']
                bx=np.frombuffer(data[:1792],dtype='<u2');px.append((bx.astype('<u4')<<16).view('<f4'));py.append(np.frombuffer(data[1792:],dtype='<u2'))
            px=np.stack(px);py=np.stack(py);pz32=(px-mu32)/radius32;pz64=(px.astype(np.float64)-mu32.astype(np.float64))/float(radius32)
            prefix64=np.array(load(b['prefix_shared_F64_path'],(64,896)),copy=True);prefix32=np.array(load(b['prefix_shared_F32_path'],(64,896),'<f4'),copy=True)
            for i,item in enumerate(journal):
                choices=item['selected_ids'];m=np.asarray(struct.unpack('<4f',bytes.fromhex(item['selected_mass_F32_HEX'])),dtype='<f4')
                # Fixed order: four selected calls, slot0..3, every operation in its declared dtype.
                for slot,e in enumerate(choices):
                    q64=(r64[e]@pz64[i])*(t64[e]@pz64[i]);q32=(r32[e]@pz32[i])*(t32[e]@pz32[i])
                    prefix64[i]+=float(m[slot])*((matrices[e]@px[i].astype(np.float64)+bias[e])+c[e]@q64)
                    prefix32[i]+=m[slot]*((a32[e]@px[i]+bias32[e])+c32[e]@q32)
            save('novel_prefix64_F64',prefix64);save('novel_prefix64_F32',prefix32,'<f4')
            n32=0;n64=0;energies=[];errors32=[];errors64=[]
            for i,item in enumerate(journal):
                mult=item['original_multiplicity'];assert mult in (1,2)
                bits32=struct.unpack('<896I',prefix32[i].tobytes());bits64=struct.unpack('<896Q',prefix64[i].tobytes())
                n32+=sum((integer32(p)-integer32(int(v)<<16))**2 for p,v in zip(bits32,py[i]))*(2//mult)
                n64+=sum((integer64(p)-(integer32(int(v)<<16)<<925))**2 for p,v in zip(bits64,py[i]))*(2//mult)
                target=(py[i].astype('<u4')<<16).view('<f4');energies.append(math.fsum(float(v)**2 for v in target)/mult)
                errors32.append(math.fsum((float(p)-float(v))**2 for p,v in zip(prefix32[i],target))/mult)
                errors64.append(math.fsum((float(p)-float(v))**2 for p,v in zip(prefix64[i],target))/mult)
            full=int(original['response_audits']['F32']['development']['exact_necessary_failure']['source_energy_integer'])
            fail32=10000*n32>full;fail64=10000*n64>(full<<1850)
            r['novel_prefix']=dict(rows=64,total_novel_rows=1495,category_scope='Original first64 novel occurrences; not representative full dialogue quality',
                F64_relative_RMS=math.sqrt(math.fsum(errors64)/math.fsum(energies)),encoded_F32_relative_RMS=math.sqrt(math.fsum(errors32)/math.fsum(energies)),
                exact_encoded_error_integer=str(n32),exact_unrounded_error_integer=str(n64),reused_FULL_source_integer=str(full),
                encoded_full_1pct_RMS_necessarily_fails=fail32,unrounded_full_1pct_RMS_necessarily_fails=fail64,
                encoded_prefix_SSE_over_FULL_source_energy=n32/full,unrounded_prefix_SSE_over_FULL_source_energy=n64/(full<<1850),
                common_squared_units_encoded='2^-298',common_squared_units_unrounded='2^-2148',weight_LCM=2,source_full_sum_recomputed=False)
            r['decision']='CLOSE_QUADRATIC_CONVEX_NOVEL_FIDELITY' if fail32 else 'ELIGIBLE_FOR_SEPARATE_FULL_NOVEL_SCREEN'
            r['new_novel_field_responses']=dict(F64=64,encoded_F32=64)
        r['complete_budget']=kernel['complete_budget']
        r['procedure_gates'].update(bound_inputs_and_FIRST_kernel_audit=True,all_cached_FIT_y_shared_and_RHS_reused=True,
            saved_NEW_factor_triangular_solve_no_inverse_or_factor_replay=True,only_NEW_augmented_objective=True,
            fixed_BF16_A_C_R_T_F32_bias_center_radius=True,original_occurrence_routes_and_mass_reused=True,
            original_FULL_source_denominator_reused=True,no_source_core_H_J_old_response_optimizer_replay=True,CPU_only_no_Torch=True)
        r.update(elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='Layer12 local conversion only; no full novel/fresh chatbot/ALL24/native/LUT/physical DRAM/rate admission')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],FIT=r['FIT_metrics'],prefix=r.get('novel_prefix'),certificate=r['convex_certificate'])),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
