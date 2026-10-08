"""First saved convex optimum/encoding/error audit; no source or core evaluation."""
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
    r=dict(schema='QWEN_FULL_FIT_CONVEX_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_core_or_field_response_evaluations=0,new_J_optimizer_or_solves=0)
    def guard():assert time.monotonic()-start<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='kernel_fit_audit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        raw=json.loads(Path(b['fit_path']).read_bytes());kernel=json.loads(Path(b['kernel_path']).read_bytes())
        def array(container,name):
            item=container['outputs'][name];p=Path(item['path']);value=np.load(p,mmap_mode='r',allow_pickle=False)
            assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
            assert list(value.shape)==item['shape'] and value.dtype.str==item['dtype'] and value.flags.c_contiguous and np.isfinite(value).all()
            assert value.offset+value.nbytes==p.stat().st_size;return value
        weights=array(kernel,'FIT_weights');w=array(kernel,'FIT_W');u=array(kernel,'FIT_U');mu=array(kernel,'mu');g=array(kernel,'G')
        radius=kernel['kernel']['radius'];alpha=kernel['kernel']['alpha'];root=np.sqrt(weights)
        source_bits=array(raw,'FIT_source_y_BF16');source=(source_bits.astype('<u4')<<16).view('<f4').astype(np.float64)
        shared=array(raw,'FIT_shared_F64');rhs=array(raw,'dual_RHS');dual=array(raw,'dual_V');theta=array(raw,'Theta')
        matrices=array(raw,'affine_A_F64');bias=array(raw,'affine_bias_F64');pred64=array(raw,'FIT_prediction_F64');pred32=array(raw,'FIT_prediction_F32')
        journal=json.loads(Path(kernel['outputs']['G']['path']).with_name('FIT_journal.json').read_bytes())
        adoption=json.loads(Path(b['adoption_path']).read_bytes())
        # Every original y byte, read by grouped case rather than producer per row.
        cursor=0
        for case_index,case in enumerate(adoption['cases']):
            if case['split']!='fit':continue
            with Path(case['binary_path']).open('rb') as f:
                for local in range(case['captured_rows']):
                    f.seek(24+local*86016+12*3584+1792);data=f.read(1792)
                    assert data==source_bits[cursor].tobytes() and journal[cursor]['case_index']==case_index and journal[cursor]['local_input_row']==local
                    cursor+=1
            guard()
        assert cursor==3845
        old_core=np.load(b['old_FIT_core_path'],mmap_mode='r',allow_pickle=False);plan=json.loads(Path(b['plan_path']).read_bytes())
        anchors={v['x_SHA256']:i for i,v in enumerate(a for a in plan['anchors'] if a['split']=='fit')}
        seen={};reused=set()
        for i,item in enumerate(journal):
            key=item['x_SHA256']
            if key in seen:assert shared[i].tobytes()==shared[seen[key]].tobytes()
            else:seen[key]=i
            if key in anchors:assert shared[i].tobytes()==old_core[anchors[key]].tobytes();reused.add(key)
        assert len(seen)==3166 and len(reused)==16 and raw['new_distinct_FIT_shared_F64_values']==3150
        expected_rhs=root[:,None]*(source-shared);assert np.array_equal(rhs,expected_rhs)
        recovered=np.empty_like(theta);recovered[:,1:]=matrices.transpose(0,2,1)*radius;recovered[:,0]=bias+matrices@mu
        theta_gap=float(np.linalg.norm(recovered-theta)/np.linalg.norm(theta));assert theta_gap<=1e-12
        dual_res=float(np.linalg.norm(g@dual+alpha*dual-rhs)/np.linalg.norm(rhs));assert dual_res<=1e-8
        predicted_private=(g@dual)/root[:,None]
        fold_gap=float(np.linalg.norm(pred64-shared-predicted_private)/max(1.,np.linalg.norm(predicted_private)));assert fold_gap<=1e-10
        grad_energy=0.;target_energy=0.;theta_from_dual_gap=0.
        for e in range(16):
            coefficient=u.T@((root*w[:,e])[:,None]*dual)
            theta_from_dual_gap=max(theta_from_dual_gap,float(np.linalg.norm(coefficient-recovered[e])/max(1.,np.linalg.norm(coefficient))))
            normal=u.T@((weights*w[:,e])[:,None]*(pred64-source))+alpha*recovered[e]
            target=u.T@((root*w[:,e])[:,None]*rhs)
            grad_energy+=float(np.sum(normal*normal));target_energy+=float(np.sum(target*target));guard()
        normal_rel=math.sqrt(grad_energy)/max(1.,math.sqrt(target_energy)+alpha*float(np.linalg.norm(recovered)))
        assert normal_rel<=1e-8 and theta_from_dual_gap<=1e-12
        # Independent scalar fsum SSE, original full source denominator reused.
        error64=[];error32=[];codec=[]
        for i in range(3845):
            target=source[i];weight=float(weights[i])
            error64.append(weight*math.fsum((float(p)-float(y))**2 for p,y in zip(pred64[i],target)))
            error32.append(weight*math.fsum((float(p)-float(y))**2 for p,y in zip(pred32[i],target)))
            codec.append(weight*math.fsum((float(p)-float(q))**2 for p,q in zip(pred64[i],pred32[i])))
        energy=raw['FIT_metrics']['reused_source_energy'];e64=math.fsum(error64);e32=math.fsum(error32)
        original=json.loads(Path(b['original_response_audit']).read_bytes())
        assert energy==original['response_audits']['F32']['fit']['metrics']['unique_x_weighted']['source_energy']
        norm2=float(np.sum(recovered*recovered));objective=e64+alpha*norm2
        def close(actual,expected):assert abs(actual-expected)<=max(1e-10,1e-10*abs(expected)),(actual,expected)
        for actual,name in ((e64,'F64_error_energy'),(e32,'encoded_F32_error_energy'),(math.sqrt(e64/energy),'F64_relative_RMS'),(math.sqrt(e32/energy),'encoded_F32_relative_RMS'),(math.sqrt(math.fsum(codec)/energy),'encoded_difference_relative_RMS')):
            close(actual,raw['FIT_metrics'][name])
        close(objective,raw['convex_certificate']['objective']);close(norm2,raw['convex_certificate']['coefficient_norm_squared'])
        bits=np.asarray(matrices,dtype='<f4').view('<u4');hi=bits>>16;lo=bits&0xffff
        rounded=(hi+((lo>0x8000)|((lo==0x8000)&((hi&1)!=0)))).astype('<u2')
        encoded=array(raw,'affine_A_BF16');bias32=array(raw,'affine_bias_F32')
        assert np.array_equal(rounded,encoded) and np.array_equal(np.asarray(bias,dtype='<f4'),bias32)
        hashes=[hashlib.sha256(encoded[e].tobytes()+bias32[e].tobytes()).hexdigest() for e in range(16)]
        assert hashes==raw['leaf_BF16_F32_hashes'] and len(set(hashes))==16
        del bits,hi,lo,rounded
        prefix_journal=json.loads(Path(b['prefix_journal_path']).read_bytes());p64=array(raw,'novel_prefix64_F64');p32=array(raw,'novel_prefix64_F32')
        old_proof=original['response_audits']['F32']['development']['exact_necessary_failure'];den32=int(old_proof['source_energy_integer']);den64=den32<<1850
        n32=0;n64=0;errs32=[];errs64=[];prefix_energies=[]
        for i,item in enumerate(prefix_journal):
            with Path(item['binary_path']).open('rb') as f:f.seek(item['x_byte_offset']+1792);raw_y=f.read(1792)
            assert hashlib.sha256(raw_y).hexdigest()==item['y_SHA256']
            ys=struct.unpack('<896f',struct.pack('<896I',*(v<<16 for v in struct.unpack('<896H',raw_y))))
            mult=item['original_multiplicity'];ints32=[];ints64=[]
            for f32,f64,y in zip(p32[i],p64[i],ys):
                pn,pd=float(f32).as_integer_ratio();qn,qd=float(f64).as_integer_ratio();yn,yd=y.as_integer_ratio()
                assert (1<<149)%pd==0 and (1<<149)%yd==0 and (1<<1074)%qd==0
                ints32.append((pn*((1<<149)//pd)-yn*((1<<149)//yd))**2)
                ints64.append((qn*((1<<1074)//qd)-yn*((1<<1074)//yd))**2)
            n32+=sum(ints32)*(2//mult);n64+=sum(ints64)*(2//mult)
            errs32.append(math.fsum((float(p)-y)**2 for p,y in zip(p32[i],ys))/mult)
            errs64.append(math.fsum((float(p)-y)**2 for p,y in zip(p64[i],ys))/mult);prefix_energies.append(math.fsum(y*y for y in ys)/mult);guard()
        proof=raw['novel_prefix'];assert str(n32)==proof['exact_encoded_error_integer'] and str(n64)==proof['exact_unrounded_error_integer']
        assert str(den32)==proof['reused_FULL_source_integer'] and str(den64)==proof['reused_FULL_source_integer_shifted1850']
        assert 10000*n32>den32 and 10000*n64>den64 and proof['encoded_full_1pct_RMS_necessarily_fails'] and proof['unrounded_full_1pct_RMS_necessarily_fails']
        prefix32=math.sqrt(math.fsum(errs32)/math.fsum(prefix_energies));prefix64=math.sqrt(math.fsum(errs64)/math.fsum(prefix_energies))
        close(prefix32,proof['encoded_F32_relative_RMS']);close(prefix64,proof['F64_relative_RMS'])
        assert raw['decision']=='CLOSE_FULL_FIT_CONVEX_NOVEL_FIDELITY' and raw['numerical_gate']
        r['independent_certificate']=dict(dual_relative_residual=dual_res,folded_vs_kernel_relative=fold_gap,theta_from_folded_relative=theta_gap,
            theta_from_dual_max_relative=theta_from_dual_gap,normal_gradient_relative=normal_rel,weighted_SSE=e64,objective=objective,
            FIT_F64_relative_RMS=math.sqrt(e64/energy),FIT_encoded_relative_RMS=math.sqrt(e32/energy),novel64_F64_relative_RMS=prefix64,novel64_encoded_relative_RMS=prefix32)
        r['exact_proofs']=dict(encoded_prefix_error=str(n32),unrounded_prefix_error=str(n64),encoded_full_1pct_necessarily_fails=True,
            unrounded_full_1pct_necessarily_fails=True,source_FULL_denominators_recomputed=False)
        r['procedure_gates'].update(ALL_original_y_bytes_and_shared_reuse=True,convex_primal_dual_and_fold_certificates=True,
            independent_scalar_FIT_and_prefix_errors=True,independent_nearest_even_leaf_encoding=True,independent_exact_ratio_prefix_failures=True,
            no_donor_core_or_prediction_operator_replay=True)
        r.update(decision='SAVED_FULL_FIT_CONVEX_FAILURE_INDEPENDENTLY_VERIFIED',elapsed_before_final_serialization=time.monotonic()-start,
            OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
