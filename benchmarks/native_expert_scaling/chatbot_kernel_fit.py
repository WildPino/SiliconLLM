"""One all-response convex affine compiler with necessary encoded novel screen."""
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
from chatbot_directional_plan import rows


def triangular(np,factor,right,transpose=False):
    """Blocked forward/back substitution, fixed128 panel, no inverse/LU refit."""
    n=factor.shape[0];result=np.array(right,dtype=np.float64,order='C',copy=True)
    starts=list(range(0,n,128))
    for start in reversed(starts) if transpose else starts:
        end=min(n,start+128)
        if transpose:
            if end<n:result[start:end]-=factor[end:,start:end].T@result[end:]
            for i in range(end-1,start-1,-1):
                if i+1<end:result[i]-=factor[i+1:end,i]@result[i+1:end]
                result[i]/=factor[i,i]
        else:
            if start:result[start:end]-=factor[start:end,:start]@result[:start]
            for i in range(start,end):
                if i>start:result[i]-=factor[i,start:i]@result[start:i]
                result[i]/=factor[i,i]
    return result


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_FULL_FIT_CONVEX_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_J=0,new_optimizer_steps=0,outputs={})
    def guard():
        assert time.monotonic()-start<=300 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
        if args.directory.exists():assert sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=512<<20
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='kernel_fit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        kernel=json.loads(Path(b['kernel_path']).read_bytes());alpha=kernel['kernel']['alpha'];radius=kernel['kernel']['radius']
        def kernel_array(name):return np.load(kernel['outputs'][name]['path'],mmap_mode='r',allow_pickle=False)
        x=kernel_array('FIT_x');a=kernel_array('FIT_weights');w=kernel_array('FIT_W');u=kernel_array('FIT_U');mu=kernel_array('mu')
        gram=kernel_array('G');factor=kernel_array('L');root_a=np.sqrt(a)
        r['complete_budget']=kernel['reused_complete_budget']
        journal=json.loads(Path(kernel['outputs']['G']['path']).with_name('FIT_journal.json').read_bytes())
        adoption=json.loads(Path(b['adoption_path']).read_bytes());y_bits=np.empty((3845,896),dtype='<u2')
        for i,item in enumerate(journal):
            case=adoption['cases'][item['case_index']]
            with Path(case['binary_path']).open('rb') as f:f.seek(item['x_byte_offset']+1792);data=f.read(1792)
            assert len(data)==1792;y_bits[i]=np.frombuffer(data,dtype='<u2')
        assert ((y_bits&0x7fff)<0x7f80).all();y=(y_bits.astype('<u4')<<16).view('<f4').astype(np.float64)
        def save(name,value,dtype='<f8'):
            value=np.ascontiguousarray(value,dtype=dtype);assert np.isfinite(value).all()
            p=args.directory/(name+'.npy')
            with p.open('xb') as f:np.save(f,value,allow_pickle=False)
            r['outputs'][name]=dict(path=str(p.resolve()),shape=list(value.shape),dtype=value.dtype.str,C_order=True,bytes=p.stat().st_size,sha256=sha(p));guard()
        core_bits=[np.load(b[n],mmap_mode='r',allow_pickle=False) for n in ('core_g_path','core_u_path','core_b_path')]
        core32=[(v.astype('<u4')<<16).view('<f4') for v in core_bits];core64=[v.astype(np.float64) for v in core32]
        def core_value(batch,coeff):
            gate,up,down=coeff;g=batch@gate.T;v=batch@up.T;t=np.exp(-np.abs(g));sig=np.where(g>=0,1/(1+t),t/(1+t))
            return (g*sig*v)@down.T
        plan=json.loads(Path(b['plan_path']).read_bytes());anchors=[v for v in plan['anchors'] if v['split']=='fit']
        old_core=np.load(b['old_FIT_core_path'],mmap_mode='r',allow_pickle=False)
        reused={bytes.fromhex(item['x_BF16_HEX']):np.array(old_core[i]) for i,item in enumerate(anchors)};assert len(reused)==16
        keys=[(np.asarray(row,dtype='<f4').view('<u4')>>16).astype('<u2').tobytes() for row in x]
        missing=list(dict.fromkeys(key for key in keys if key not in reused));assert len(missing)==3150
        index_by_key={key:i for i,key in enumerate(keys)}
        new_core=core_value(np.array([x[index_by_key[key]] for key in missing]),core64);guard()
        cached=dict(reused);cached.update(zip(missing,new_core))
        shared=np.array([cached[key] for key in keys]);assert len(cached)==3166
        r.update(reused_distinct_FIT_shared_F64_values=16,new_distinct_FIT_shared_F64_values=3150,
            shared_F64_contract='Old16 row-F64 values reused; NEW3150 distinct batch-F64 values; duplicates reuse same value')
        save('FIT_source_y_BF16',y_bits,'<u2');save('FIT_shared_F64',shared)
        rhs=root_a[:,None]*(y-shared)
        intermediate=triangular(np,factor,rhs);dual=triangular(np,factor,intermediate,True);guard()
        dual_res=float(np.linalg.norm(gram@dual+alpha*dual-rhs)/np.linalg.norm(rhs))
        theta=np.empty((16,897,896))
        for e in range(16):theta[e]=u.T@((root_a*w[:,e])[:,None]*dual);guard()
        matrices=theta[:,1:].transpose(0,2,1)/radius;bias=theta[:,0]-matrices@mu
        save('dual_RHS',rhs);save('dual_V',dual);save('Theta',theta);save('affine_A_F64',matrices);save('affine_bias_F64',bias)
        def bf16(value):
            f32=np.asarray(value,dtype='<f4');assert np.isfinite(f32).all();bits=f32.view('<u4')
            result=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2');assert ((result&0x7fff)<0x7f80).all();return result
        encoded=bf16(matrices);bias32=np.asarray(bias,dtype='<f4');matrix32=(encoded.astype('<u4')<<16).view('<f4')
        save('affine_A_BF16',encoded,'<u2');save('affine_bias_F32',bias32,'<f4')
        r['leaf_BF16_F32_hashes']=[hashlib.sha256(encoded[e].tobytes()+bias32[e].tobytes()).hexdigest() for e in range(16)]
        assert len(set(r['leaf_BF16_F32_hashes']))==16
        sizes={s:sum(c['captured_rows'] for c in adoption['cases'] if c['split']==s) for s in ('fit','development')}
        saved=routes(Path(b['saved_E16_routes']),sizes);ids=np.asarray(saved['fit']['ids']);masses32=np.asarray(saved['fit']['mass'],dtype=np.float32)
        x32=np.asarray(x,dtype=np.float32);shared32=core_value(x32,core32);guard()
        prediction64=shared.copy();prediction32=np.array(shared32,copy=True)
        for slot in range(4):
            for e in range(16):
                indices=np.flatnonzero(ids[:,slot]==e)
                if len(indices):
                    prediction64[indices]+=w[indices,e,None]*(x[indices]@matrices[e].T+bias[e])
                    prediction32[indices]+=masses32[indices,slot,None]*(x32[indices]@matrix32[e].T+bias32[e])
            guard()
        private_dual=(gram@dual)/root_a[:,None]
        fold_gap=float(np.linalg.norm(prediction64-shared-private_dual)/max(1.,np.linalg.norm(private_dual)))
        error=prediction64-y;weighted_error=root_a[:,None]*error
        gradient_energy=0.;normal_target_energy=0.
        for e in range(16):
            normal=u.T@((a*w[:,e])[:,None]*error)+alpha*theta[e]
            gradient_energy+=float(np.sum(normal*normal))
            target=u.T@((root_a*w[:,e])[:,None]*rhs);normal_target_energy+=float(np.sum(target*target));guard()
        gradient_rel=math.sqrt(gradient_energy)/max(1.,math.sqrt(normal_target_energy)+alpha*float(np.linalg.norm(theta)))
        fit_denominator=json.loads(Path(b['original_response_audit']).read_bytes())['response_audits']['F32']['fit']['metrics']['unique_x_weighted']['source_energy']
        sse64=math.fsum(float(v) for v in np.sum(a[:,None]*error*error,axis=0))
        sse32=float(np.sum(a[:,None]*(prediction32.astype(np.float64)-y)**2))
        coefficient_norm2=float(np.sum(theta*theta))
        r['convex_certificate']=dict(alpha=alpha,dual_relative_residual=dual_res,folded_vs_kernel_prediction_relative=fold_gap,
            normal_gradient_relative=gradient_rel,weighted_response_SSE=sse64,coefficient_norm_squared=coefficient_norm2,
            objective=sse64+alpha*coefficient_norm2,objective_at_zero_private=float(np.sum(a[:,None]*(shared-y)**2)),
            qualification='Unique minimizer of this fixed ridge objective within fixed affine class, subject to reported numerical residuals')
        r['FIT_metrics']=dict(rows=3845,exact_x_classes=3166,reused_source_energy=fit_denominator,
            F64_error_energy=sse64,F64_relative_RMS=math.sqrt(sse64/fit_denominator),encoded_F32_error_energy=sse32,encoded_F32_relative_RMS=math.sqrt(sse32/fit_denominator),
            encoded_difference_relative_RMS=float(np.linalg.norm(root_a[:,None]*(prediction32.astype(np.float64)-prediction64))/math.sqrt(fit_denominator)))
        save('FIT_shared_batched_F32',shared32,'<f4');save('FIT_prediction_F64',prediction64);save('FIT_prediction_F32',prediction32,'<f4')
        r['numerical_gate']=dual_res<=1e-8 and fold_gap<=1e-10 and gradient_rel<=1e-8
        if not r['numerical_gate']:
            r['decision']='CLOSE_FULL_FIT_NUMERICAL_COMPILER'
        else:
            prefix_journal=json.loads(Path(b['prefix_journal_path']).read_bytes());assert len(prefix_journal)==64
            old_prefix_core=np.load(b['old_prefix_core_path'],mmap_mode='r',allow_pickle=False)
            px=[];py=[]
            for item in prefix_journal:
                with Path(item['binary_path']).open('rb') as f:f.seek(item['x_byte_offset']);data=f.read(3584)
                assert hashlib.sha256(data[:1792]).hexdigest()==item['x_SHA256'] and hashlib.sha256(data[1792:]).hexdigest()==item['y_SHA256']
                bx=np.frombuffer(data[:1792],dtype='<u2');by=np.frombuffer(data[1792:],dtype='<u2')
                px.append((bx.astype('<u4')<<16).view('<f4'));py.append(by)
            px=np.stack(px);py=np.stack(py);prefix_shared32=core_value(px,core32);guard()
            prefix64=np.array(old_prefix_core,copy=True);prefix32=np.array(prefix_shared32,copy=True)
            for i,item in enumerate(prefix_journal):
                choices=item['selected_ids'];mass=np.asarray(struct.unpack('<4f',bytes.fromhex(item['selected_mass_F32_HEX'])),dtype=np.float32)
                prefix64[i]+=mass.astype(np.float64)@(matrices[choices]@px[i].astype(np.float64)+bias[choices])
                prefix32[i]+=mass@(matrix32[choices]@px[i]+bias32[choices])
            save('novel_prefix64_F64',prefix64);save('novel_prefix64_F32',prefix32,'<f4');save('novel_prefix64_shared_batched_F32',prefix_shared32,'<f4')
            old_proof=json.loads(Path(b['original_response_audit']).read_bytes())['response_audits']['F32']['development']['exact_necessary_failure']
            numerator32=0;numerator64=0;prefix_source=[];error32=[];error64=[]
            for i,item in enumerate(prefix_journal):
                mult=item['original_multiplicity'];bits32=struct.unpack('<896I',prefix32[i].tobytes());bits64=struct.unpack('<896Q',prefix64[i].tobytes())
                numerator32+=sum((integer32(p)-integer32(int(v)<<16))**2 for p,v in zip(bits32,py[i]))*(2//mult)
                numerator64+=sum((integer64(p)-(integer32(int(v)<<16)<<925))**2 for p,v in zip(bits64,py[i]))*(2//mult)
                target=(py[i].astype('<u4')<<16).view('<f4')
                prefix_source.append(math.fsum(float(v)**2 for v in target)/mult)
                error32.append(math.fsum((float(p)-float(v))**2 for p,v in zip(prefix32[i],target))/mult)
                error64.append(math.fsum((float(p)-float(v))**2 for p,v in zip(prefix64[i],target))/mult)
            denominator32=int(old_proof['source_energy_integer']);denominator64=denominator32<<1850
            failed=10000*numerator32>denominator32
            r['novel_prefix']=dict(rows=64,total_novel_rows=1495,reused_shared_F64_values=64,new_shared_F64_values=0,
                F64_relative_RMS=math.sqrt(math.fsum(error64)/math.fsum(prefix_source)),encoded_F32_relative_RMS=math.sqrt(math.fsum(error32)/math.fsum(prefix_source)),
                exact_encoded_error_integer=str(numerator32),reused_FULL_source_integer=str(denominator32),
                exact_unrounded_error_integer=str(numerator64),reused_FULL_source_integer_shifted1850=str(denominator64),
                encoded_full_1pct_RMS_necessarily_fails=failed,unrounded_full_1pct_RMS_necessarily_fails=10000*numerator64>denominator64,
                encoded_prefix_SSE_over_FULL_source_energy=numerator32/denominator32,unrounded_prefix_SSE_over_FULL_source_energy=numerator64/denominator64,
                common_squared_units_encoded='2^-298',common_squared_units_unrounded='2^-2148',weight_LCM=2,source_full_sum_recomputed=False)
            r['new_novel_field_responses']=dict(F64=64,encoded_F32=64)
            if failed:
                r['decision']='CLOSE_FULL_FIT_CONVEX_NOVEL_FIDELITY'
            else:
                # Existing64 new predictions and old core values are reused.
                groups=rows(adoption);fit_keys=set(keys);dev=groups['development'];indices=[i for i,item in enumerate(dev) if item['key'] not in fit_keys]
                assert len(indices)==1495 and indices[:64]==[item['split_row_index'] for item in prefix_journal]
                from collections import Counter
                counts=Counter(item['key'] for item in dev)
                predictions=list(prefix32);records=list(prefix_journal)
                for index in indices[64:]:
                    item=dev[index];bx=np.frombuffer(item['key'],dtype='<u2');row=(bx.astype('<u4')<<16).view('<f4')
                    choices=saved['development']['ids'][index];mass=np.asarray(saved['development']['mass'][index],dtype=np.float32)
                    prediction=core_value(row[None,:],core32)[0]+mass@(matrix32[list(choices)]@row+bias32[list(choices)])
                    predictions.append(prediction);records.append(dict(case_index=item['case_index'],x_byte_offset=item['x_byte_offset'],binary_path=item['binary_path'],
                        original_multiplicity=counts[item['key']],category=adoption['cases'][item['case_index']]['category'],split_row_index=index));guard()
                categories={};errors=[]
                for ordinal,(prediction,item) in enumerate(zip(predictions,records)):
                    with Path(item['binary_path']).open('rb') as f:f.seek(item['x_byte_offset']+1792);target_bits=np.frombuffer(f.read(1792),dtype='<u2')
                    target=(target_bits.astype('<u4')<<16).view('<f4');mult=item['original_multiplicity'];cat=item['category']
                    err=math.fsum((float(p)-float(v))**2 for p,v in zip(prediction,target))/mult;energy=math.fsum(float(v)**2 for v in target)/mult
                    errors.append(err);categories.setdefault(cat,dict(errors=[],energy=[]));categories[cat]['errors'].append(err);categories[cat]['energy'].append(energy)
                source_energy=math.ldexp(float(denominator32),-298)/2
                full_rms=math.sqrt(math.fsum(errors)/source_energy)
                cat_rms={cat:math.sqrt(math.fsum(v['errors'])/math.fsum(v['energy'])) for cat,v in categories.items()}
                save('novel_ALL_encoded_F32',np.stack(predictions),'<f4');write_once(args.directory/'novel_ALL_journal.json',records)
                r['novel_ALL']=dict(encoded_relative_RMS=full_rms,category_relative_RMS=cat_rms,new_encoded_responses=1495)
                r['decision']='ELIGIBLE_FOR_SEPARATE_NATIVE_COMPOSITION' if full_rms<=.01 and len(cat_rms)==16 and all(v<=.03 for v in cat_rms.values()) else 'CLOSE_FULL_FIT_CONVEX_NOVEL_FIDELITY'
        r['procedure_gates'].update(all_bound_original_FIT_y_and_routes_reused=True,only3150_new_distinct_FIT_shared_F64_values=True,
            actual_block_triangular_solves_no_inverse_or_factor_replay=True,finite_C_order_arrays_and_distinct16_encoded_hashes=True,
            original_full_source_energies_reused=True,no_donor_J_old_candidate_or_optimizer_replay=True,no_Torch_GPU_runtime=True)
        r.update(elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='One fixed all-response ridge compiler; NumPy local arithmetic, NOT native parity/fresh chatbot/useful n/DRAM/rate')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],certificate=r['convex_certificate'],FIT=r['FIT_metrics'],prefix=r.get('novel_prefix'),seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
