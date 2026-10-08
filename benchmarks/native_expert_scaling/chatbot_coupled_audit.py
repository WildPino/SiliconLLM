"""Independent scalar assembly and saved SVD certificate audit, no new SVD/model."""
import argparse
import datetime as dt
from fractions import Fraction
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
from chatbot_directional_plan import geometry


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_COUPLED_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_model_or_J_or_solve_evaluations=0,new_SVD_evaluations=0)
    def guard():assert time.monotonic()-start<=60 and proc.memory_info().peak_wset<=512<<20 and not proc.children(recursive=True)
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='coupled_audit'
        for item in b['inputs']:
            path=Path(item['path']);assert str(path.resolve())==item['resolved_path'] and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        raw=json.loads(Path(b['design_path']).read_bytes());plan=json.loads(Path(b['plan_path']).read_bytes())
        anchors=[a for a in plan['anchors'] if a['split']=='fit'];assert len(anchors)==16
        def load(name,shape):
            item=raw['output_arrays'][name];path=Path(item['path'])
            assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes']
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            assert a.shape==shape and a.dtype==np.dtype('<f8') and a.flags.c_contiguous and np.isfinite(a).all()
            assert a.offset+a.nbytes==path.stat().st_size;return a
        q=np.load(b['Q_path'],mmap_mode='r',allow_pickle=False)
        mu=load('mu',(896,));scale=load('scale',(32,));z=load('z',(16,32));w=load('W',(16,16));k=load('K',(528,528))
        gradient=load('selected_gradient_x',(16,4,896));gamma=load('selected_gamma',(16,4,32))
        # Scalar fsum construction, distinct from the producer's BLAS/reduction path.
        x=[]
        for a in anchors:
            data=bytes.fromhex(a['x_BF16_HEX']);assert hashlib.sha256(data).hexdigest()==a['x_SHA256']
            with Path(a['binary_path']).open('rb') as stream:stream.seek(a['x_byte_offset']);assert stream.read(1792)==data
            bits=struct.unpack('<896H',data);x.append(struct.unpack('<896f',struct.pack('<896I',*(v<<16 for v in bits))))
        expected_mu=[math.fsum(row[j] for row in x)/16 for j in range(896)]
        chart=[[math.fsum((row[j]-expected_mu[j])*float(q[j,a]) for j in range(896)) for a in range(32)] for row in x]
        expected_s=[math.sqrt(math.fsum(row[a]**2 for row in chart)/16) for a in range(32)]
        expected_z=np.array([[row[a]/expected_s[a] for a in range(32)] for row in chart])
        def gap(a,c):
            value=float(np.max(np.abs(np.asarray(a)-np.asarray(c))))
            assert value<=1e-10*(1+float(np.max(np.abs(c))))
            return value
        r['assembly_gaps']=dict(mu=gap(mu,expected_mu),scale=gap(scale,expected_s),z=gap(z,expected_z))
        _,projection,centers,_=geometry(Path(b['saved_geometry']))
        expected_w=np.zeros((16,16));expected_g=np.zeros((16,4,896));expected_gamma=np.zeros((16,4,32))
        for i,a in enumerate(anchors):
            ids=a['selected_parent_ids'];mass=a['selected_shadow_mass_F64'];expected_w[i,ids]=mass
            average=[math.fsum(weight*centers[e][j] for weight,e in zip(mass,ids)) for j in range(32)]
            for selected,(e,weight) in enumerate(zip(ids,mass)):
                for j in range(896):
                    expected_g[i,selected,j]=2*weight*math.fsum((centers[e][t]-average[t])*projection[t][j] for t in range(32))
                for t in range(32):
                    expected_gamma[i,selected,t]=expected_s[t]*math.fsum(float(expected_g[i,selected,j])*float(q[j,t]) for j in range(896))
            guard()
        assert np.array_equal(w,expected_w)
        r['assembly_gaps'].update(selected_gradient=gap(gradient,expected_g),selected_gamma=gap(gamma,expected_gamma))
        # Different row/column indexing: directly apply the linear functional to
        # every constant/chart basis coefficient; no producer block assignment.
        expected_k=np.zeros((528,528))
        for i,a in enumerate(anchors):
            for selected,e in enumerate(a['selected_parent_ids']):
                weight=expected_w[i,e]
                for coefficient in range(33):
                    basis=1. if coefficient==0 else expected_z[i,coefficient-1]
                    expected_k[33*i,33*e+coefficient]=weight*basis
                    for direction in range(32):
                        expected_k[33*i+direction+1,33*e+coefficient]=expected_gamma[i,selected,direction]*basis+(weight if coefficient==direction+1 else 0.)
            guard()
        r['assembly_gaps']['K']=gap(k,expected_k)
        r['systems']={};eligibility={}
        for name,matrix,n in (('W',w,16),('K',k,528)):
            u=load(name+'_U',(n,n));s=load(name+'_singular_values',(n,));vt=load(name+'_Vt',(n,n))
            assert (s>0).all() and (s[:-1]>=s[1:]).all()
            residual=float(np.linalg.norm((u*s)@vt-matrix)/np.linalg.norm(matrix))
            orth_u=float(np.linalg.norm(u.T@u-np.eye(n)));orth_v=float(np.linalg.norm(vt@vt.T-np.eye(n)))
            assert residual<=1e-12 and orth_u<=1e-10 and orth_v<=1e-10
            tolerance=math.ulp(1.)*n*float(s[0]);rank=sum(float(v)>tolerance for v in s);condition=float(s[0])/float(s[-1])
            expected=raw['systems'][name]
            assert rank==expected['numeric_rank'] and condition==expected['condition2'] and tolerance==expected['tolerance']
            r['systems'][name]=dict(numeric_rank=rank,condition2=condition,relative_SVD_reconstruction=residual,
                U_orthogonal_error=orth_u,V_orthogonal_error=orth_v,rank_scope='Saved F64 SVD certificate, not exact rank')
            eligibility[name+'_full_numerical_rank']=rank==n;eligibility[name+'_condition2_at_most_1e6']=condition<=1e6;guard()
        # Independent complete dimensions, no price() import or old-price rerun.
        d,l,v,h,p,sel,qdim=896,24,151936,512,16,4,32
        source_mac=l*(3*d*4864+2*d*d+2*d*128)+d*v
        norms_bias=2*(l*2*d+d+l*(d+2*128)+d)
        base_storage=2*(l*(2*d*d+2*d*128+2*d+d+2*128)+d*v+d)
        r['budget']={}
        for arm in raw['budget']['arms']:
            c=arm['children'];router=l*(d*qdim+p*qdim+(sel*c*qdim if c>1 else 0))
            total=l*(3*d*h+sel*d*d+2*d*d+2*d*128)+d*v+router
            logical=2*total+norms_bias+l*(p+(sel*c if c>1 else 0))*4+l*sel*d*4
            storage=base_storage+2*l*(3*d*h+p*c*d*d+d*qdim+p*qdim+(p*c*qdim if c>1 else 0))+4*l*(p+(p*c if c>1 else 0)+p*c*d)
            assert (total,logical,storage)==(arm['matrix_MAC_per_token'],arm['logical_coefficient_bytes_per_token'],arm['proposed_payload_bytes'])
            assert str(Fraction(total,source_mac))==arm['matrix_ratio_exact'] and str(Fraction(logical,2*source_mac+norms_bias))==arm['logical_ratio_exact']
            assert 24*2*14*64==raw['budget']['attention_MAC_per_context_token'] and 4096*24*2*128*4==raw['budget']['F32_cache_bytes_at_capacity']
            for name,value in (('matrix_cost_gate',5*total<=3*source_mac),('logical_cost_gate',5*logical<=3*(2*source_mac+norms_bias))):
                assert value==arm[name];eligibility[f'E{p*c}_{name}']=value
            r['budget'][str(p*c)]=dict(matrix_MAC=total,logical_bytes=logical,stored_bytes=storage)
        eligibility['ALL_chart_scales_above_1e_minus12']=all(s>1e-12 for s in expected_s)
        assert eligibility==raw['eligibility_gates']
        assert all(eligibility.values()) and raw['decision']=='ELIGIBLE_FOR_ONE_SEPARATELY_FROZEN_COUPLED_COMPILER'
        r['procedure_gates'].update(independent_complete_cost=True,independent_FIT_only_scalar_assembly=True,
            saved_full_SVD_certificates=True,all_numerical_eligibility_decisions_match=True,all_bound_arrays_and_original_anchor_bytes=True)
        r.update(decision='SAVED_COUPLED_DESIGN_INDEPENDENTLY_VERIFIED',elapsed_before_final_serialization=time.monotonic()-start,
            OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
