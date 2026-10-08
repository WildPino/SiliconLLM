"""First explicit whitened-feature covariance audit, no response or new factor."""
import argparse
import datetime as dt
import io
import json
import math
from pathlib import Path
import struct
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_joint_retained_audit import Descriptors


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_NULL_PRIOR_COVARIANCE_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_or_core_responses=0,new_source_or_core_J=0,
        new_field_coefficient_solutions=0,new_Cholesky_evaluations=0)
    def guard():assert time.monotonic()-start<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='null_prior_kernel_audit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        raw=json.loads(Path(b['kernel_path']).read_bytes());old=json.loads(Path(b['old_kernel_path']).read_bytes())
        assert raw['decision']=='ELIGIBLE_FOR_ONE_NULL_PRIOR_CONVEX_COMPILER' and all(raw['eligibility_gates'].values())
        def array(report,name,shape):
            item=report['outputs'][name];p=Path(item['path']);value=np.load(p,mmap_mode='r',allow_pickle=False)
            assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'] and value.offset+value.nbytes==p.stat().st_size
            assert value.dtype==np.dtype('<f8') and value.shape==shape and value.flags.c_contiguous and np.isfinite(value).all()
            return value
        p=array(raw,'P',(32,896));s=array(raw,'S',(32,32));h=array(raw,'H_leaf',(16,16))
        ls=array(raw,'L_S',(32,32));lh=array(raw,'L_H_leaf',(16,16));pn=array(raw,'Pi_N',(896,896))
        kernel=array(raw,'K_H',(3845,3845));factor=array(raw,'L',(3845,3845))
        u=array(old,'FIT_U',(3845,897));w=array(old,'FIT_W',(3845,16));a=array(old,'FIT_weights',(3845,))
        wa=np.load(b['anchor_mass_path'],mmap_mode='r',allow_pickle=False);assert wa.shape==(16,16) and wa.dtype==np.dtype('<f8')
        # Independent struct decoder of the original stored selector.
        with zipfile.ZipFile(b['saved_geometry']) as archive:
            prefix=next(n[:-8] for n in archive.namelist() if n.endswith('/data.pkl'))
            item=Descriptors(io.BytesIO(archive.read(prefix+'data.pkl'))).load()['projection']
            assert item['shape']==(32,896) and item['stride']==(896,1) and item['offset']==0
            storage=item['storage'];assert storage['dtype']=='FloatStorage' and storage['count']==28672
            binary=archive.read(prefix+'data/'+storage['key']);assert len(binary)==114688
            expected_p=np.array(struct.unpack('<28672f',binary),dtype=np.float64).reshape(32,896)
        assert np.array_equal(expected_p,p)
        energy_y=json.loads(Path(b['original_response_audit']).read_bytes())['response_audits']['F32']['fit']['metrics']['unique_x_weighted']['source_energy']
        energy_j=json.loads(Path(b['original_directional_report']).read_bytes())['pooled']['fit']['source_energy']['null']
        alpha=old['kernel']['alpha'];lam=energy_y/energy_j;radius=old['kernel']['radius'];beta=lam/(radius*radius)
        assert (alpha,lam,beta)==(raw['scales']['alpha'],raw['scales']['lambda_value'],raw['scales']['beta'])
        expected_s=np.array([[math.fsum(float(p[i,k])*float(p[j,k]) for k in range(896)) for j in range(32)] for i in range(32)])
        expected_h=np.array([[alpha*(i==j)+beta*math.fsum(float(wa[k,i])*float(wa[k,j]) for k in range(16)) for j in range(16)] for i in range(16)])
        relative=lambda x,y:float(np.linalg.norm(x-y)/np.linalg.norm(y))
        sgap=relative(expected_s,s);hgap=relative(expected_h,h);assert max(sgap,hgap)<=1e-12
        for value,l in ((s,ls),(h,lh)):
            assert (np.diag(l)>0).all() and not np.any(np.triu(l,1)) and relative(l@l.T,value)<=1e-10
        # Independent column/scalar forward substitution, not producer's two-pass solve.
        def lower(l,rhs):
            answer=np.empty(rhs.shape,dtype=np.float64)
            for j in range(rhs.shape[1]):
                for i in range(len(l)):
                    answer[i,j]=(float(rhs[i,j])-math.fsum(float(l[i,k])*float(answer[k,j]) for k in range(i)))/float(l[i,i])
                if j%128==0:guard()
            return answer
        q=lower(ls,p);expected_pn=np.eye(896)-q.T@q
        projection_gap=relative(expected_pn,pn);assert projection_gap<=1e-12
        project_sym=relative(pn.T,pn);project_idem=relative(pn@pn,pn)
        null_gap=float(np.linalg.norm(p@pn)/np.linalg.norm(p));assert max(project_sym,project_idem,null_gap)<=1e-10
        z=u[:,1:];chart=z@q.T;null_z=z-chart@q
        null_mass=lower(lh,w.T).T;guard()
        # Explicit redundant coordinates:16 bias +512 chart +14336 null features.
        # Not the producer's two Hadamard products or a coefficient-normal inverse.
        features=np.empty((3845,14864));features[:,:16]=w/math.sqrt(alpha)
        for e in range(16):
            features[:,16+32*e:16+32*(e+1)]=w[:,e,None]*chart/math.sqrt(alpha)
            features[:,528+896*e:528+896*(e+1)]=null_mass[:,e,None]*null_z
        features*=np.sqrt(a)[:,None];explicit=features@features.T;guard()
        covariance_gap=relative(explicit,kernel);assert covariance_gap<=1e-11
        del features,explicit
        regularized=np.array(kernel);regularized.flat[::3846]+=1
        assert (np.diag(factor)>0).all() and not np.any(np.triu(factor,1))
        reconstruction=relative(factor@factor.T,regularized);symmetry=relative(kernel.T,kernel)
        assert reconstruction<=1e-10 and symmetry<=1e-12
        r['independent_checks']=dict(original_selector_bytes=True,small_S_scalar_relative_gap=sgap,small_H_scalar_relative_gap=hgap,
            independent_whitened_projector_relative_gap=projection_gap,projector_symmetry=project_sym,projector_idempotence=project_idem,
            selector_null_relative=null_gap,explicit_feature_columns=14864,ALL_explicit_whitened_feature_covariance_relative_gap=covariance_gap,
            saved_Cholesky_relative_reconstruction=reconstruction,kernel_symmetry=symmetry,
            condition_scope='Real covariance identity and positive-alpha deduction; no new factor/rank/eigenvalue/condition2 assertion')
        r['procedure_gates'].update(original_P_and_FIT_geometry_reused=True,original_scales_not_recomputed=True,
            independent_scalar_small_operators_and_whitened_projector=True,ALL_kernel_entries_by_explicit_whitened_features=True,
            saved_factor_and_original_eligibility_verified=True,no_y_J_core_field_control_or_new_factor_replay=True)
        r.update(decision='SAVED_NULL_PRIOR_COVARIANCE_INDEPENDENTLY_VERIFIED',elapsed_before_final_serialization=time.monotonic()-start,
            OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],checks=r['independent_checks'],seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--binding-sha',required=True)
    parser.add_argument('--freeze',required=True);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
