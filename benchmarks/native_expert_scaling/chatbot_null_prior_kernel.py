"""Source-null prior covariance prerequisite; no y/J/core/field evaluation."""
import argparse
import datetime as dt
import io
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_joint_retained_audit import Descriptors


def selector_bytes(path):
    """Decode only original projection storage; never recompute its basis."""
    with zipfile.ZipFile(path) as archive:
        prefix=next(n[:-8] for n in archive.namelist() if n.endswith('/data.pkl'))
        assert archive.read(prefix+'byteorder')==b'little'
        item=Descriptors(io.BytesIO(archive.read(prefix+'data.pkl'))).load()['projection']
        assert item['offset']==0 and item['shape']==(32,896) and item['stride']==(896,1)
        storage=item['storage'];assert storage['dtype']=='FloatStorage' and storage['count']==32*896
        raw=archive.read(prefix+'data/'+storage['key']);assert len(raw)==32*896*4
        return raw


def small_solve(factor,rhs):
    """Actual triangular substitutions on a saved small Cholesky, no inverse."""
    import numpy as np
    value=np.array(rhs,dtype=np.float64,copy=True,order='C')
    for i in range(len(factor)):
        value[i]=(value[i]-factor[i,:i]@value[:i])/factor[i,i]
    for i in range(len(factor)-1,-1,-1):
        value[i]=(value[i]-factor[i+1:,i]@value[i+1:])/factor[i,i]
    return value


def saved_array(report,name,shape):
    import numpy as np
    item=report['outputs'][name];p=Path(item['path']);value=np.load(p,mmap_mode='r',allow_pickle=False)
    assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    assert value.shape==shape and value.dtype==np.dtype('<f8') and value.flags.c_contiguous
    assert value.offset+value.nbytes==p.stat().st_size and np.isfinite(value).all()
    return value


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_NULL_PRIOR_COVARIANCE_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},eligibility_gates={},new_original_BF16_full_forwards=0,new_source_or_core_responses=0,
        new_source_or_core_J=0,new_field_coefficient_solutions=0,outputs={})
    def guard():
        assert time.monotonic()-start<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
        if args.directory.exists():assert sum(v.stat().st_size for v in args.directory.iterdir() if v.is_file())<=512<<20
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='null_prior_kernel'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        old=json.loads(Path(b['old_kernel_path']).read_bytes())
        u=saved_array(old,'FIT_U',(3845,897));w=saved_array(old,'FIT_W',(3845,16));a=saved_array(old,'FIT_weights',(3845,))
        z=u[:,1:];assert np.array_equal(u[:,0],np.ones(3845)) and (a>0).all()
        radius=old['kernel']['radius'];alpha=old['kernel']['alpha']
        energy_y=json.loads(Path(b['original_response_audit']).read_bytes())['response_audits']['F32']['fit']['metrics']['unique_x_weighted']['source_energy']
        energy_j=json.loads(Path(b['original_directional_report']).read_bytes())['pooled']['fit']['source_energy']['null']
        assert (energy_y,energy_j)==(78343.61382048983,836.9166991571583)
        lam=energy_y/energy_j;beta=lam/(radius*radius)
        assert all(math.isfinite(v) and v>0 for v in (radius,alpha,lam,beta))
        p=np.frombuffer(selector_bytes(b['saved_geometry']),dtype='<f4').reshape(32,896).astype(np.float64)
        wa=np.load(b['anchor_mass_path'],mmap_mode='r',allow_pickle=False)
        assert p.shape==(32,896) and np.isfinite(p).all()
        assert wa.shape==(16,16) and wa.dtype==np.dtype('<f8') and wa.flags.c_contiguous and np.isfinite(wa).all()
        assert (wa>=0).all() and np.max(np.abs(wa.sum(axis=1)-1))<=1e-12
        r['scales']=dict(radius=radius,alpha=alpha,lambda_value=lam,beta=beta,reused_FIT_source_value_energy=energy_y,
            reused_old_Q_FIT_source_null_J_energy=energy_j,normalizer_scope='Original old-Q energy is a frozen scale, not recomputed for new P projector')
        r['reused_complete_budget']=old['reused_complete_budget']
        def save(name,value):
            value=np.ascontiguousarray(value,dtype='<f8');assert np.isfinite(value).all()
            path=args.directory/(name+'.npy')
            with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
            r['outputs'][name]=dict(path=str(path.resolve()),shape=list(value.shape),dtype='<f8',C_order=True,bytes=path.stat().st_size,sha256=sha(path));guard()
        def symmetry(value):return float(np.linalg.norm(value-value.T)/np.linalg.norm(value))
        def reconstruction(value,factor):return float(np.linalg.norm(factor@factor.T-value)/np.linalg.norm(value))
        s=p@p.T;h=alpha*np.eye(16)+beta*(wa.T@wa)
        save('P',p);save('S',s);save('H_leaf',h)
        r['operator']=dict(selector_S_symmetry=symmetry(s),leaf_H_symmetry=symmetry(h))
        r['eligibility_gates']['small_symmetry_at_most_1e_minus12']=all(v<=1e-12 for v in r['operator'].values())
        r['eligibility_gates']['positive_finite_scales']=True
        try:
            ls=np.linalg.cholesky(s);lh=np.linalg.cholesky(h);guard()
            sr=reconstruction(s,ls);hr=reconstruction(h,lh)
            pi_p=p.T@small_solve(ls,p);pi_n=np.eye(896)-pi_p;guard()
            project_sym=symmetry(pi_p)
            project_idem=float(np.linalg.norm(pi_p@pi_p-pi_p)/np.linalg.norm(pi_p))
            null_gap=float(np.linalg.norm(p@pi_n)/np.linalg.norm(p))
            r['operator'].update(selector_S_reconstruction=sr,leaf_H_reconstruction=hr,projector_symmetry=project_sym,
                projector_idempotence=project_idem,selector_null_relative=null_gap,
                real_algebra_scope='Exact real projector for original finite F32 P; these are rounded F64 residuals, not exact matrix identities')
            r['eligibility_gates'].update(small_positive_and_reconstruction_at_most_1e_minus10=bool((np.diag(ls)>0).all() and (np.diag(lh)>0).all() and max(sr,hr)<=1e-10),
                projector_residuals_at_most_1e_minus10=max(project_sym,project_idem,null_gap)<=1e-10)
            save('L_S',ls);save('L_H_leaf',lh);save('Pi_N',pi_n)
            if all(r['eligibility_gates'].values()):
                c=z@p.T;qinner=c@small_solve(ls,c.T);ninner=z@z.T-qinner
                kernel=(w@w.T)*((1+qinner)/alpha)
                kernel+=(w@small_solve(lh,w.T))*ninner
                roots=np.sqrt(a);kernel*=roots[:,None];kernel*=roots[None,:];guard()
                del qinner,ninner,c
                sym=symmetry(kernel);save('K_H',kernel)
                regularized=kernel.copy();regularized.flat[::3846]+=1
                factor=np.linalg.cholesky(regularized);guard()
                rec=reconstruction(regularized,factor);save('L',factor)
                r['kernel']=dict(rows=3845,feature_columns=14352,ALL896_output_RHS_shared_factor=True,symmetry_relative=sym,
                    Cholesky_relative_reconstruction=rec,nominal_real_condition_upper=1+old['kernel']['trace']/alpha,
                    condition_scope='H>=alpha I implies K_H<=B B^T/alpha in real algebra; no measured/rigorous condition2 of rounded arrays')
                r['eligibility_gates'].update(kernel_symmetry_at_most_1e_minus12=sym<=1e-12,
                    kernel_positive_and_reconstruction_at_most_1e_minus10=bool((np.diag(factor)>0).all() and rec<=1e-10))
        except np.linalg.LinAlgError as error:
            r['factorization_fault']=repr(error);r['eligibility_gates']['positive_factorization']=False
        r['procedure_gates'].update(actual_bound_used_inputs_and_C_F64=True,original_FIT_U_W_weights_only=True,
            original_selector_and_smooth_anchor_mass_reused=True,original_energy_scales_not_recomputed=True,
            no_y_J_prior_mean_field_coefficients_or_errors=True,no_PCA_QR_source_core_control_or_old_kernel_replay=True,no_Torch_GPU=True)
        r['decision']='ELIGIBLE_FOR_ONE_NULL_PRIOR_CONVEX_COMPILER' if all(r['eligibility_gates'].values()) else 'CLOSE_NULL_PRIOR_COVARIANCE_PREREQUISITE'
        r.update(elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],scales=r['scales'],operator=r['operator'],kernel=r.get('kernel'))),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--binding-sha',required=True)
    parser.add_argument('--freeze',required=True);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
