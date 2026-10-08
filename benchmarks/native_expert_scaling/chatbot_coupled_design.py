"""First affine-jet budget/design screen; no model, value, J or solver calls."""
import argparse
import datetime as dt
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0, str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha, write_once
from chatbot_directional_plan import geometry


def price(spec, ledger):
    """Changed full affine fields, with the unchanged whole source ledger reused."""
    assert (spec['hidden_size'], spec['layers'], spec['vocabulary_size'],
            spec['source_ffn_width'], spec['shared_width'], spec['query_width'],
            spec['parents'], spec['selected_parents']) == (896,24,151936,4864,512,32,16,4)
    assert spec['children_per_parent'] == [1,10]
    assert spec['maximum_matrix_ratio'] == spec['maximum_logical_coefficient_ratio'] == [3,5]
    assert ledger['source_matrix_MAC'] == 493961216 and ledger['source_FFN_MAC'] == 313786368
    assert ledger['source_payload_bytes'] == 988065536 and ledger['source_logical_coefficient_bytes'] == 988067328
    d,l,h,q,p,k = 896,24,512,32,16,4
    fixed_mac = ledger['source_matrix_MAC']-ledger['source_FFN_MAC']
    fixed_payload = ledger['source_payload_bytes']-2*ledger['source_FFN_MAC']
    fixed_logical = ledger['source_logical_coefficient_bytes']-2*ledger['source_matrix_MAC']
    shared = l*3*d*h
    active_affine = l*k*d*d
    bias_reads = l*k*d*4
    arms = []
    for c in (1,10):
        router_mac = l*(d*q+p*q+(k*c*q if c>1 else 0))
        router_matrix_stored = l*(d*q+p*q+(p*c*q if c>1 else 0))
        router_norm_stored = l*(p+(p*c if c>1 else 0))
        router_norm_reads = l*(p+(k*c if c>1 else 0))*4
        private_stored = l*p*c*d*d
        bias_stored = l*p*c*d*4
        matrix = fixed_mac+shared+active_affine+router_mac
        logical = 2*matrix+fixed_logical+router_norm_reads+bias_reads
        payload = fixed_payload+2*(shared+private_stored+router_matrix_stored)+4*router_norm_stored+bias_stored
        arms.append(dict(children=c,functions_per_layer=p*c,stored_function_slots=l*p*c,
            active_affine_matrix_MAC=active_affine,shared_matrix_MAC=shared,
            router_matrix_MAC=router_mac,matrix_MAC_per_token=matrix,
            logical_coefficient_bytes_per_token=logical,proposed_payload_bytes=payload,
            active_F32_bias_read_bytes=bias_reads,stored_F32_bias_bytes=bias_stored,
            matrix_ratio_exact=str(Fraction(matrix,ledger['source_matrix_MAC'])),
            logical_ratio_exact=str(Fraction(logical,ledger['source_logical_coefficient_bytes'])),
            matrix_cost_gate=5*matrix<=3*ledger['source_matrix_MAC'],
            logical_cost_gate=5*logical<=3*ledger['source_logical_coefficient_bytes'],
            support='E16 design only' if c==1 else 'STORAGE/COST PROJECTION ONLY; old uniform E160 support FAIL retained'))
    return dict(scope='NEW_COMPLETE_DIMENSION_DEDUCTION_NOT_PHYSICAL_DRAM_RATE_OR_CAPACITY',
        reused_source_matrix_MAC=ledger['source_matrix_MAC'],reused_source_logical_bytes=ledger['source_logical_coefficient_bytes'],
        unchanged_head_MAC=ledger['arms'][0]['head_MAC'],unchanged_attention_projection_MAC=ledger['arms'][0]['attention_projection_MAC'],
        attention_MAC_per_context_token=ledger['source_attention_MAC_per_context_token'],
        F32_cache_bytes_at_capacity=ledger['native_cache_bytes_at_capacity'],arms=arms,
        excluded_from_matrix_count='Scalar bias adds, RMS, SiLU/products, mass/topK, accumulation, cache/control remain actual work',
        coefficient_encoding='Proposed BF16 matrices and F32 biases/centroid norms; no export performed')


def main(args):
    import psutil
    started = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(11)))
    r = dict(schema='QWEN_COUPLED_DESIGN_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),procedure_gates={},
        new_original_BF16_full_forwards=0,new_source_or_shared_responses=0,new_source_or_shared_jacobians=0,
        new_coefficient_solutions=0,new_affine_function_evaluations=0,output_arrays={})
    def guard():
        assert time.monotonic()-started<=60 and proc.memory_info().peak_wset<=512<<20
        assert not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['schema']=='QWEN_DIRECTIONAL_BINDING_V1' and b['job']['name']=='coupled_design'
        for item in b['inputs']:
            path=Path(item['path'])
            assert str(path.resolve())==item['resolved_path'] and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules
        args.directory.mkdir()
        spec=json.loads(Path(b['spec_path']).read_bytes());ledger=json.loads(Path(b['ledger_path']).read_bytes())
        r['budget']=price(spec,ledger)
        r['eligibility_gates']={f'E{arm["functions_per_layer"]}_{name}':arm[name]
            for arm in r['budget']['arms'] for name in ('matrix_cost_gate','logical_cost_gate')}
        if not all(r['eligibility_gates'].values()):
            r['decision']='CLOSE_COUPLED_AFFINE_COMPLETE_COST'
        else:
            plan=json.loads(Path(b['plan_path']).read_bytes());anchors=[a for a in plan['anchors'] if a['split']=='fit']
            assert len(anchors)==16 and [a['parent'] for a in anchors]==list(range(16))
            xs=[]
            for anchor in anchors:
                data=bytes.fromhex(anchor['x_BF16_HEX']);assert hashlib.sha256(data).hexdigest()==anchor['x_SHA256']
                with Path(anchor['binary_path']).open('rb') as stream:
                    assert struct.unpack('<8s4I',stream.read(24))==(b'QWCAP001',24,896,2,0)
                    stream.seek(anchor['x_byte_offset']);assert stream.read(1792)==data
                bits=np.frombuffer(data,dtype='<u2');assert ((bits&0x7fff)<0x7f80).all()
                xs.append((bits.astype('<u4')<<16).view('<f4').astype('<f8'))
            x=np.stack(xs);mu=x.mean(axis=0)
            q=np.load(b['Q_path'],mmap_mode='r',allow_pickle=False)
            assert q.shape==(896,32) and q.dtype==np.dtype('<f8') and q.flags.c_contiguous
            assert q.offset+q.nbytes==Path(b['Q_path']).stat().st_size
            _,pv,cv,_=geometry(Path(b['saved_geometry']))
            projection=np.asarray(pv,dtype=np.float64);centers=np.asarray(cv,dtype=np.float64)
            chart=(x-mu)@q;scale=np.sqrt(np.mean(chart*chart,axis=0))
            r['eligibility_gates']['ALL_chart_scales_above_1e_minus12']=bool(np.isfinite(scale).all() and (scale>1e-12).all())
            r['anchor_contract']=dict(split='FIT_ONLY',count=16,parents=list(range(16)),
                x_SHA256=[a['x_SHA256'] for a in anchors],split_row_indices=[a['split_row_index'] for a in anchors],
                selected_ids=[a['selected_parent_ids'] for a in anchors],
                mass='REUSED original F64 smooth shadow mass, not new choice/softmax',
                minimum_scale=float(scale.min()),maximum_scale=float(scale.max()))
            if not r['eligibility_gates']['ALL_chart_scales_above_1e_minus12']:
                r['decision']='CLOSE_COUPLED_JET_DESIGN'
            else:
                z=chart/scale;w=np.zeros((16,16));gradient=np.empty((16,4,896));gamma=np.empty((16,4,32))
                for i,a in enumerate(anchors):
                    ids=a['selected_parent_ids'];mass=np.asarray(a['selected_shadow_mass_F64'],dtype=np.float64)
                    assert len(set(ids))==4 and np.isfinite(mass).all() and (mass>0).all() and abs(float(mass.sum())-1)<1e-14
                    w[i,ids]=mass;mean=mass@centers[ids]
                    gradient[i]=2*mass[:,None]*((centers[ids]-mean)@projection)
                    gamma[i]=(gradient[i]@q)*scale
                block=33;k=np.zeros((16*block,16*block))
                for i,a in enumerate(anchors):
                    basis=np.r_[1.,z[i]]
                    for selected,e in enumerate(a['selected_parent_ids']):
                        target=k[i*block:(i+1)*block,e*block:(e+1)*block]
                        target[0]=w[i,e]*basis
                        target[1:]=gamma[i,selected,:,None]*basis[None,:]
                        target[1:,1:]+=w[i,e]*np.eye(32)
                def save(name,array):
                    value=np.ascontiguousarray(array,dtype='<f8');assert np.isfinite(value).all()
                    path=args.directory/(name+'.npy')
                    with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
                    r['output_arrays'][name]=dict(path=str(path.resolve()),shape=list(value.shape),dtype='<f8',
                        C_order=True,bytes=path.stat().st_size,sha256=sha(path));guard()
                for name,value in (('mu',mu),('scale',scale),('z',z),('W',w),('selected_gradient_x',gradient),('selected_gamma',gamma),('K',k)):
                    save(name,value)
                r['systems']={}
                for name,matrix in (('W',w),('K',k)):
                    u,s,vt=np.linalg.svd(matrix,full_matrices=False);guard()
                    tolerance=float(np.finfo(np.float64).eps*max(matrix.shape)*s[0])
                    rank=int(np.count_nonzero(s>tolerance));condition=float(s[0]/s[-1]) if s[-1]>0 else None
                    residual=float(np.linalg.norm(matrix-(u*s)@vt)/np.linalg.norm(matrix))
                    orth_u=float(np.linalg.norm(u.T@u-np.eye(len(s))))
                    orth_v=float(np.linalg.norm(vt@vt.T-np.eye(len(s))))
                    r['systems'][name]=dict(shape=list(matrix.shape),numeric_rank=rank,
                        tolerance=tolerance,sigma_max=float(s[0]),sigma_min=float(s[-1]),condition2=condition,
                        relative_SVD_reconstruction=residual,U_orthogonal_error_Frobenius=orth_u,V_orthogonal_error_Frobenius=orth_v,
                        rank_scope='F64 SVD threshold, not a formal real/exact rank proof')
                    save(name+'_U',u);save(name+'_singular_values',s);save(name+'_Vt',vt)
                    r['eligibility_gates'][name+'_full_numerical_rank']=rank==matrix.shape[0]
                    r['eligibility_gates'][name+'_condition2_at_most_1e6']=condition is not None and condition<=1e6
                    assert residual<=1e-12 and orth_u<=1e-10 and orth_v<=1e-10
                r['decision']='ELIGIBLE_FOR_ONE_SEPARATELY_FROZEN_COUPLED_COMPILER' if all(r['eligibility_gates'].values()) else 'CLOSE_COUPLED_JET_DESIGN'
                r['procedure_gates']['finite_C_F64_arrays_and_SVD_certificates']=True
            r['procedure_gates']['only16_existing_FIT_anchor_bytes_and_selected_mass']=True
        r['procedure_gates'].update(bound_used_inputs=True,new_complete_ledger_includes_biases_head_attention_cache=True,
            no_model_value_J_solver_or_old_control_evaluation=True,no_Torch_or_GPU_runtime=True)
        assert 'torch' not in sys.modules
        r.update(elapsed_before_final_serialization=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            decision_scope='This fixed budget/design only; no fitted bank, fidelity, physical DRAM, useful count, whole chat or rate')
        guard();write_once(args.out,r)
        print(json.dumps(dict(decision=r['decision'],eligibility=r['eligibility_gates'],systems=r.get('systems'),
            elapsed=time.monotonic()-started,OS_peak=proc.memory_info().peak_wset)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
