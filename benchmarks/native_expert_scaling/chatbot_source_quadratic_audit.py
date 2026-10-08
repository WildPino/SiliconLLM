"""FIRST factorized-score, saved projection and source-byte audit, no pursuit replay."""
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
    started=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_SOURCE_QUADRATIC_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_or_core_H_J_responses=0,new_pursuit_or_LS_replays=0,new_response_fit_calls=0)
    def guard():
        assert time.monotonic()-started<=180 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='source_quadratic_audit'
        for item in b['inputs']:
            p=Path(item['path']);assert str(p.resolve())==item['resolved_path'] and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];guard()
        import numpy as np
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules;args.directory.mkdir()
        raw=json.loads(Path(b['factor_path']).read_bytes());curv=json.loads(Path(b['curvature_path']).read_bytes());plan=json.loads(Path(b['plan_path']).read_bytes())
        def array(item,shape,dtype='<f8'):
            p=Path(item['path']);assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
            a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.shape==shape and a.dtype==np.dtype(dtype) and a.flags.c_contiguous
            assert a.offset+a.nbytes==p.stat().st_size and np.isfinite(a).all();return a
        def saved(name,shape,dtype='<f8'):return array(raw['outputs'][name],shape,dtype)
        def old(name,shape):return array(curv['outputs'][name],shape)
        spec=b['source_slices'];source_path=Path(spec['path']);assert source_path.stat().st_size==spec['file_bytes'] and str(source_path.resolve())==spec['resolved_path']
        source_bits=[]
        with source_path.open('rb') as stream:
            header=stream.read(spec['header_bytes']);assert hashlib.sha256(header).hexdigest()==spec['header_sha256'] and struct.unpack('<Q',header[:8])[0]+8==len(header)
            fields=json.loads(header[8:])
            for item in spec['tensors']:
                f=fields[item['name']];assert f['dtype']=='BF16' and f['shape']==item['shape'] and len(header)+f['data_offsets'][0]==item['offset']
                stream.seek(item['offset']);data=stream.read(item['bytes']);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
                source_bits.append(np.frombuffer(data,dtype='<u2').reshape(item['shape']))
        core_bits=[np.load(b['core_paths'][n],mmap_mode='r',allow_pickle=False) for n in ('g','u','b')]
        def promote(bits):
            # Independent exponent/mantissa BF16 conversion, exact powers of two.
            bits=np.asarray(bits);exp=((bits>>7)&255).astype(np.int32);mant=(bits&127).astype(np.float64)
            assert (exp<255).all();mant+=np.where(exp==0,0.,128.)
            value=np.ldexp(mant,np.where(exp==0,-133,exp-134));return np.where(bits&0x8000,-value,value)
        down=np.concatenate((promote(source_bits[2]),promote(core_bits[2])),axis=1)
        # Scalar fsum column norms qualify the norm denominator independently.
        dn=np.array([math.fsum(float(z)**2 for z in col) for col in down.T])
        gates=np.concatenate((source_bits[0],core_bits[0]));ups=np.concatenate((source_bits[1],core_bits[1]))
        gx=np.concatenate((old('source_gx',(32,4864))[:16],old('core_gx',(32,512))[:16]),axis=1)
        ux=np.concatenate((old('source_ux',(32,4864))[:16],old('core_ux',(32,512))[:16]),axis=1)
        gv=np.concatenate((old('source_Gv',(4,4864)),old('core_Gv',(4,512))),axis=1)
        uv=np.concatenate((old('source_Uv',(4,4864)),old('core_Uv',(4,512))),axis=1)
        sh=old('source_H',(32,4,896));ch=old('core_H',(32,4,896))
        kappa=saved('kappa',(16,5376,4));norms=saved('dictionary_norms',(16,5376));scores=saved('scores',(16,8,5376))
        residuals=saved('residuals',(16,9,4,896));ids=saved('selected',(16,8),'<i8');atoms=saved('selected_normalized_atoms',(16,8,4,896))
        coefs=saved('coefficients',(16,8,8));svals=saved('singular_values',(16,8,8))
        rf=saved('R_BF16',(16,16,896),'<u2');tf=saved('T_BF16',(16,16,896),'<u2')
        maxima={};normal_max=0.;parent_reports=[]
        def check(name,a,z,tol=1e-10):
            gap=float(np.linalg.norm(a-z));scale=max(float(np.linalg.norm(a)),float(np.linalg.norm(z)))
            assert gap<=1e-11+tol*scale,(name,gap,scale);maxima[name]=max(maxima.get(name,0.),gap/max(scale,1e-20))
        def energy(a):return math.fsum(float(z)**2 for z in np.asarray(a).flat)
        for parent in range(16):
            assert plan['anchors'][parent]['split']=='fit' and plan['anchors'][parent]['parent']==parent
            expected=np.empty((5376,4))
            for i in range(5376):
                g=float(gx[parent,i]);e=math.exp(-abs(g));s=1/(1+e) if g>=0 else e/(1+e)
                a1=s*(1+g*(1-s));a2=2*s*(1-s)+g*s*(1-s)*(1-2*s)
                for j in range(4):
                    a=float(gv[j,i]);z=float(uv[j,i]);expected[i,j]=(a2*float(ux[parent,i])*a*a+2*a1*a*z)*(1 if i<4864 else -1)
            check('kappa',kappa[parent],expected)
            nu=np.sqrt(dn*np.array([math.fsum(float(z)**2 for z in row) for row in expected]));check('norms',norms[parent],nu)
            cutoff=1e-12*float(np.max(norms[parent]));eligible=norms[parent]>cutoff;chosen=[]
            target=np.asarray(sh[parent]-ch[parent]);te=energy(target);check('target',residuals[parent,0],target)
            record=raw['parents'][parent];assert record['parent']==parent and record['x_SHA256']==plan['anchors'][parent]['x_SHA256'] and record['candidate_nonzero_count']==int(np.count_nonzero(eligible))
            assert math.isclose(record['dictionary_cutoff'],cutoff,rel_tol=2e-12,abs_tol=1e-14)
            for step in range(8):
                # Full factorized score contraction, no explicit dictionary or pursuit.
                contractions=down.T@residuals[parent,step].T
                corr=np.zeros(5376);corr[eligible]=np.sum(expected[eligible]*contractions[eligible],axis=1)/norms[parent,eligible]
                check('ALL_scores',scores[parent,step],corr)
                allowed=eligible.copy();allowed[chosen]=False;ranking=np.where(allowed,np.abs(corr),-np.inf)
                selected_id=int(ids[parent,step]);assert selected_id==int(np.argmax(ranking))
                assert ranking[selected_id]>1e-12*math.sqrt(energy(residuals[parent,step]));chosen.append(selected_id)
                wanted=(expected[selected_id,:,None]*down[:,selected_id][None,:])/norms[parent,selected_id]
                check('selected_atoms',atoms[parent,step],wanted)
                basis=atoms[parent,:step+1].reshape(step+1,3584)
                coef=coefs[parent,step,:step+1];assert np.array_equal(coefs[parent,step,step+1:],np.zeros(7-step))
                check('orthogonal_refit_reconstruction',residuals[parent,step+1].reshape(-1),target.reshape(-1)-coef@basis)
                normal=float(np.linalg.norm(basis@residuals[parent,step+1].reshape(-1)))/math.sqrt(te);assert normal<=1e-10;normal_max=max(normal_max,normal)
                # New FIRST small-Gram eigen qualification, not a producer LS replay.
                eig=np.linalg.eigvalsh(basis@basis.T);assert eig[0]>0 and eig[-1]/eig[0]<=1e16
                singular=np.sqrt(eig[::-1]);assert np.allclose(singular,svals[parent,step,:step+1],rtol=1e-9,atol=1e-12)
                assert np.array_equal(svals[parent,step,step+1:],np.zeros(7-step))
                before=energy(residuals[parent,step]);after=energy(residuals[parent,step+1]);assert after<=before+1e-12*te
                st=record['steps'][step];assert st['candidate_id']==selected_id and st['step']==step
                assert st['source_or_core']==('source' if selected_id<4864 else 'negative_core') and st['atom_index']==(selected_id if selected_id<4864 else selected_id-4864)
                for key,val in (('score',abs(scores[parent,step,selected_id])),('residual_energy',after)):
                    assert math.isclose(st[key],val,rel_tol=2e-12,abs_tol=1e-14)
                other=np.where(allowed,np.abs(scores[parent,step]),-np.inf);other[selected_id]=-np.inf;runner=int(np.argmax(other))
                assert st['runner_up_id']==runner and math.isclose(st['runner_up_score'],other[runner],rel_tol=2e-12,abs_tol=1e-14)
                assert math.isclose(st['score_margin'],st['score']-st['runner_up_score'],rel_tol=2e-12,abs_tol=1e-14)
                assert math.isclose(st['condition'],svals[parent,step,0]/svals[parent,step,step],rel_tol=2e-12,abs_tol=1e-14)
                guard()
            assert record['selected_candidate_ids']==chosen and len(set(chosen))==8
            assert np.array_equal(rf[parent,:8],gates[chosen]) and np.array_equal(rf[parent,8:],gates[chosen])
            assert np.array_equal(tf[parent,:8],gates[chosen]) and np.array_equal(tf[parent,8:],ups[chosen])
            assert hashlib.sha256(rf[parent].tobytes()+tf[parent].tobytes()).hexdigest()==record['factors_BF16_SHA256']
            final=energy(residuals[parent,8]);ratio=math.sqrt(final/te)
            for key,val in (('target_curvature_energy',te),('final_residual_curvature_energy',final),('source_atom_curvature_projection_RMS',ratio)):
                assert math.isclose(record[key],val,rel_tol=2e-12,abs_tol=1e-14)
            parent_reports.append(dict(parent=parent,projection_RMS=ratio,source_atoms=sum(i<4864 for i in chosen),core_atoms=sum(i>=4864 for i in chosen)))
        assert raw['distinct_factor_hashes']==len(set(z['factors_BF16_SHA256'] for z in raw['parents']))
        with source_path.open('rb') as stream:
            assert hashlib.sha256(stream.read(spec['header_bytes'])).hexdigest()==spec['header_sha256']
            for item in spec['tensors']:
                stream.seek(item['offset']);assert hashlib.sha256(stream.read(item['bytes'])).hexdigest()==item['sha256']
        r.update(decision='SOURCE_QUADRATIC_FACTORS_INDEPENDENTLY_VERIFIED',maximum_formula_relative_gaps=maxima,maximum_projection_normal_relative=normal_max,
            parents=parent_reports,distinct_factor_hashes=raw['distinct_factor_hashes'],scope='FIRST saved factor/score/projection audit; not response or full-null fidelity')
        r['procedure_gates']=dict(bound_inputs_and_source_extents_before_after=True,independent_BF16_scalar_derivatives_factorized_scores=True,
            ALL688128_scores_ALL128_choices_and_refits=True,ALL128_atoms_and_factor_bytes_verified=True,FIT_only_no_original_operator_or_pursuit_replay=True)
        r.update(elapsed_before_final_serialization=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],maximum_normal=normal_max)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-started,OS_peak_snapshot=proc.memory_info().peak_wset,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
