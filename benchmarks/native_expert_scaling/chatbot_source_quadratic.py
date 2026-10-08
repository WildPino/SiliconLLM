"""FIRST FIT-only source/core atom pursuit and exact BF16 quadratic factors."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_null_curvature import source_slices


def main(args):
    import psutil
    started=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_SOURCE_QUADRATIC_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},qualification_gates={},new_original_BF16_full_forwards=0,new_source_or_core_H_J_responses=0,
        new_old_feature_experiments=0,new_response_fit_or_optimizer_calls=0,parents=[],outputs={})
    def guard():
        assert time.monotonic()-started<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
        if args.directory.exists():assert sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=64<<20
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='source_quadratic'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules and os.environ['OPENBLAS_NUM_THREADS']=='1';args.directory.mkdir()
        curvature=json.loads(Path(b['curvature_path']).read_bytes());plan=json.loads(Path(b['plan_path']).read_bytes())
        def saved(name,shape):
            item=curvature['outputs'][name];p=Path(item['path']);assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
            a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.shape==shape and a.dtype==np.dtype('<f8') and a.flags.c_contiguous and np.isfinite(a).all();return a
        def bits(path,shape):
            a=np.load(path,mmap_mode='r',allow_pickle=False);assert a.shape==shape and a.dtype==np.dtype('<u2') and a.flags.c_contiguous;return a
        def decode(value):
            assert ((value&0x7fff)<0x7f80).all();return (value.astype('<u4')<<16).view('<f4').astype('<f8')
        source_bits=[np.frombuffer(v,dtype='<u2').reshape(s) for v,s in zip(source_slices(b),((4864,896),(4864,896),(896,4864)))]
        core_bits=[bits(b['core_paths'][n],s) for n,s in zip(('g','u','b'),((512,896),(512,896),(896,512)))]
        gates=np.concatenate((source_bits[0],core_bits[0]));ups=np.concatenate((source_bits[1],core_bits[1]))
        down=np.concatenate((decode(source_bits[2]),decode(core_bits[2])),axis=1);d2=np.einsum('ij,ij->j',down,down)
        gx=np.concatenate((saved('source_gx',(32,4864))[:16],saved('core_gx',(32,512))[:16]),axis=1)
        ux=np.concatenate((saved('source_ux',(32,4864))[:16],saved('core_ux',(32,512))[:16]),axis=1)
        gv=np.concatenate((saved('source_Gv',(4,4864)),saved('core_Gv',(4,512))),axis=1)
        uv=np.concatenate((saved('source_Uv',(4,4864)),saved('core_Uv',(4,512))),axis=1)
        hs=saved('source_H',(32,4,896));hc=saved('core_H',(32,4,896))
        sigexp=np.exp(-np.abs(gx));sig=np.where(gx>=0,1/(1+sigexp),sigexp/(1+sigexp))
        a1=sig+gx*sig*(1-sig);a2=sig*(1-sig)*(2+gx*(1-2*sig))
        kappas=np.empty((16,5376,4));norms=np.empty((16,5376));scores=np.zeros((16,8,5376));residuals=np.zeros((16,9,4,896))
        selected=np.zeros((16,8),dtype='<i8');atoms=np.zeros((16,8,4,896));coefficients=np.zeros((16,8,8));singulars=np.zeros((16,8,8))
        rf=np.empty((16,16,896),dtype='<u2');tf=np.empty_like(rf)
        def save(name,value):
            a=np.ascontiguousarray(value);p=args.directory/(name+'.npy')
            with p.open('xb') as stream:np.save(stream,a,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
            r['outputs'][name]=dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p),shape=list(a.shape),dtype=a.dtype.str,C_order=True);guard()
        journal=(args.directory/'parents.jsonl').open('xb')
        try:
            for parent in range(16):
                assert plan['anchors'][parent]['split']=='fit' and plan['anchors'][parent]['parent']==parent;guard()
                target=np.asarray(hs[parent]-hc[parent]).reshape(-1);te=float(target@target);assert te>1e-12
                kappa=(a2[parent]*ux[parent])[None,:]*gv*gv+2*a1[parent][None,:]*gv*uv
                kappa[:,4864:]*=-1;kappa=kappa.T.copy();kappas[parent]=kappa
                dictionary=(kappa[:,:,None]*down.T[:,None,:]).reshape(5376,3584)
                nu=np.linalg.norm(dictionary,axis=1);norms[parent]=nu
                cutoff=1e-12*float(np.max(nu));eligible=np.isfinite(nu)&(nu>cutoff);assert np.count_nonzero(eligible)>=8
                dictionary[eligible]/=nu[eligible,None];dictionary[~eligible]=0
                residual=target.copy();residuals[parent,0]=residual.reshape(4,896);ids=[];steps=[]
                for step in range(8):
                    corr=dictionary@residual;scores[parent,step]=corr;allowed=eligible.copy();allowed[ids]=False
                    ranking=np.where(allowed,np.abs(corr),-np.inf);chosen=int(np.argmax(ranking))
                    assert ranking[chosen]>1e-12*float(np.linalg.norm(residual))
                    ids.append(chosen);basis=dictionary[ids].T
                    coef,_,rank,svals=np.linalg.lstsq(basis,target,rcond=None)
                    assert rank==len(ids) and svals[-1]>0 and svals[0]/svals[-1]<=1e8
                    fresh=target-basis@coef;normal=float(np.linalg.norm(basis.T@fresh))/math.sqrt(te)
                    recon=float(np.linalg.norm(target-basis@coef-fresh))/math.sqrt(te)
                    assert normal<=1e-10 and recon<=1e-10
                    old_energy=float(residual@residual);new_energy=float(fresh@fresh)
                    assert new_energy<=old_energy+1e-12*te
                    other=ranking.copy();other[chosen]=-np.inf;runner=int(np.argmax(other))
                    coefficients[parent,step,:step+1]=coef;singulars[parent,step,:step+1]=svals
                    residuals[parent,step+1]=fresh.reshape(4,896);residual=fresh
                    steps.append(dict(step=step,candidate_id=chosen,source_or_core=('source' if chosen<4864 else 'negative_core'),
                        atom_index=(chosen if chosen<4864 else chosen-4864),score=float(ranking[chosen]),runner_up_id=runner,runner_up_score=float(ranking[runner]),
                    score_margin=float(ranking[chosen]-ranking[runner]),condition=float(svals[0]/svals[-1]),normal_relative=normal,
                        projection_reconstruction_relative=recon,residual_energy=new_energy))
                    guard()
                selected[parent]=ids;atoms[parent]=dictionary[ids].reshape(8,4,896)
                rf[parent,:8]=gates[ids];rf[parent,8:]=gates[ids];tf[parent,:8]=gates[ids];tf[parent,8:]=ups[ids]
                record=dict(parent=parent,anchor_index=parent,x_SHA256=plan['anchors'][parent]['x_SHA256'],candidate_nonzero_count=int(np.count_nonzero(eligible)),
                    dictionary_cutoff=cutoff,target_curvature_energy=te,final_residual_curvature_energy=float(residual@residual),
                    source_atom_curvature_projection_RMS=math.sqrt(float(residual@residual)/te),selected_candidate_ids=ids,steps=steps,
                    factors_BF16_SHA256=hashlib.sha256(rf[parent].tobytes()+tf[parent].tobytes()).hexdigest())
                journal.write((json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode());journal.flush();os.fsync(journal.fileno())
                r['parents'].append(record)
                # Every completed parent is retained before later apparatus failure.
                for label,value in (('selected',selected[parent]),('R_BF16',rf[parent]),('T_BF16',tf[parent]),('kappa',kappas[parent]),
                    ('norms',norms[parent]),('scores',scores[parent]),('residuals',residuals[parent]),('atoms',atoms[parent]),
                    ('coefficients',coefficients[parent]),('singulars',singulars[parent])):save('parent%02d_'%parent+label,value)
                print(json.dumps(dict(parent=parent,projection_RMS=record['source_atom_curvature_projection_RMS'],seconds=time.monotonic()-started)),flush=True)
                del dictionary,basis
        finally:journal.close()
        for name,value in (('kappa',kappas),('dictionary_norms',norms),('scores',scores),('residuals',residuals),('selected',selected),
            ('selected_normalized_atoms',atoms),('coefficients',coefficients),('singular_values',singulars),('R_BF16',rf),('T_BF16',tf)):save(name,value)
        assert len(r['parents'])==16 and np.isfinite(kappas).all() and np.isfinite(scores).all()
        source_slices(b);r['distinct_factor_hashes']=len(set(a['factors_BF16_SHA256'] for a in r['parents']))
        r['new_atom_dictionary_entries']=16*5376;r['new_greedy_selection_steps']=128
        r['qualification_gates']=dict(ALL16_complete_eight_independent_atoms=True,ALL128_projection_normal_residuals=True,ALL_BF16_factors_exact_source_rows=True)
        r['procedure_gates']=dict(bound_inputs_and_source_extents_before_after=True,FIT_only_no_novel_selection_labels=True,
            fixed_eight_atoms_sixteen_forms_no_grid=True,completed_parents_retained=True,no_original_H_J_response_feature_or_optimizer_replay=True,CPU_only_no_Torch=True)
        r.update(decision='SOURCE_QUADRATIC_FACTORS_REQUIRE_FIRST_INDEPENDENT_AUDIT',
            scope='Atom-curvature selection/projection and source-factor bytes only; C response coefficients and whole model not acquired',
            elapsed_before_final_serialization=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
