"""Retained first rare responses: correct buffer view; first exact gate metrics."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import ctypes
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import meth528_prefix_io_repair4 as IO
from meth528_prefix_bound_repair4 import energy,limbs,lower_factor


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha',required=True);ap.add_argument('--freeze',required=True);args=ap.parse_args();ctx=None
    try:
        IO.MANIFEST=IO.DOC/'meth528_rare_metadata_manifest_20261007.json'
        ctx=IO.PrefixContext('rare_metadata_proof',args.manifest_sha,args.freeze);np=ctx.numpy()
        import meth528_contract as S
        prior=IO.ROOT/'results/native_expert_scaling/meth528_rare_gate_repair1'
        failed=json.loads((IO.DOC/'meth528_rare_gate_repair1_result.failure.json').read_bytes())
        assert failed['new_C_response_calls_entered']==1 and len(failed['partial_outputs'])==8 and failed['gates']['fresh_frozen_used_inputs_runtime_compiler_partial_and_foreign_SHA']
        m,occ,dev,counts=S.metadata(ctx)
        ids=np.flatnonzero((~dev)&(counts[m[:,3]]>=1)&(counts[m[:,3]]<=4))
        assert len(ids)==3 and np.all(m[ids,3]==15)
        main_arrays=[np.load(prior/('e015_main_'+name+'.npy'),allow_pickle=False) for name in ('physical','integer','codes','alpha')]
        c_arrays=[np.load(prior/('e015_C_'+name+'.npy'),allow_pickle=False) for name in ('physical','integer','codes','alpha')]
        assert main_arrays[2].flags.f_contiguous and c_arrays[2].flags.f_contiguous and main_arrays[2].shape==(3,516)
        # Original C writes flat ROW-MAJOR through pointer into F-contiguous array.
        corrected=c_arrays[2].ravel(order='F').reshape((3,516),order='C')
        for i in (0,1,3):assert main_arrays[i].tobytes()==c_arrays[i].tobytes()
        assert main_arrays[2].tobytes()==corrected.tobytes()
        IO.save(np,ctx.out,'corrected_C_codes',corrected)
        ctx.r.update(prefix_parents=[],reused_rare_parent=15,new_C_response_calls=0,new_numpy_candidate_calls=0,retained_first_C_response_calls=1)
        ctx.r['gates']['retained_physical_I64_alpha_and_correctly_reinterpreted_C_hidden_codes_BYTE']=True
        auditor=IO.ROOT/'results/native_expert_scaling/meth528_prefix_energy_audit_repair4'
        reducer=ctypes.CDLL(str(auditor/'independent_integer.dll'))
        reducer.prefix_energy.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*3;reducer.prefix_energy.restype=ctypes.c_int
        reducer.prefix_weighted.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*3;reducer.prefix_weighted.restype=ctypes.c_int
        out=main_arrays[0];p=m[ids,10].copy().view('<f4');candidates=[out,np.multiply(out,p[:,None],dtype=np.float32)]
        weighted=np.empty((3,768),'<f4')
        assert reducer.prefix_weighted(3,768,IO.pointer(out),IO.pointer(p),IO.pointer(weighted))==0 and weighted.tobytes()==candidates[1].tobytes()
        sources=[S.records(ctx,'unweighted',ids),S.records(ctx,'targets',ids)]
        known=np.load(IO.ROOT/'results/native_expert_scaling/meth528_prefix_bound_repair4/source_U640.npy',allow_pickle=False)
        errors=np.empty((3,2,10),'<u8')
        for arm in range(2):
            truth=np.ascontiguousarray(sources[arm]).view('<u4');guess=np.ascontiguousarray(candidates[arm]).view('<u4')
            for j in range(3):errors[j,arm]=limbs(np,energy(truth[j].tolist(),guess[j].tolist()))
            independently=np.empty((3,10),'<u8')
            assert reducer.prefix_energy(3,768,IO.pointer(truth),IO.pointer(guess),IO.pointer(independently))==0
            assert errors[:,arm].tobytes()==independently.tobytes()
        IO.save(np,ctx.out,'error_U640',errors)
        ctx.r['gates']['FIRST_4608_error_scalar_terms_independently_C_integer_BYTE_and_F32_weighting']=True
        rows=m.tolist();dc=[0]*128
        for row in rows:
            if row[4]&5:dc[row[3]]+=1
        independent_ids=[row[0] for row in rows if row[4]&2 and 1<=dc[row[3]]<=4]
        assert independent_ids==ids.tolist()
        factor=lower_factor(3);q=1<<53;low=Fraction(q-1,q)
        independent_factor=low**6/Fraction(q+1,q)*Fraction(q-1534,q)*Fraction(q-4,q)
        assert factor==independent_factor
        threshold=Fraction(5764607523034235,576460752303423488);view=dict(kind='rare',development_class='1..4',split='consumed_validation',count=3,UIDs=ids.tolist())
        for arm,name in enumerate(('unweighted','weighted')):
            D=sum(int.from_bytes(known[uid,arm].tobytes(),'little') for uid in ids)
            N=sum(int.from_bytes(errors[j,arm].tobytes(),'little') for j in range(3))
            left=N*factor.numerator*threshold.denominator**2;right=D*factor.denominator*threshold.numerator**2
            reject=bool(D==0 or left>right)
            assert reject==(D==0 or Fraction(N,D)*independent_factor>threshold**2)
            view[name]=dict(full_source_denominator_integer=str(D),full_error_numerator_integer=str(N),guarded_cross_left=str(left),guarded_cross_right=str(right),strict_original_gate_failure=reject,full_domain_real_RMS_display=math.sqrt(N/D) if D else None)
        decision='NECESSARY_FIXED_DOMAIN_FIDELITY_FAILURE_CERTIFIED' if any(view[n]['strict_original_gate_failure'] for n in ('unweighted','weighted')) else 'INCONCLUSIVE_FIRST_ORIGINAL_RARE_DOMAIN'
        ctx.r['gates']['original_FULL_three_UID_rare_domain_and_two_guarded_decisions_independently_verified']=True
        ctx.finish(dict(decision=decision,view=view,new_C_error_energy_calls=2,new_C_weighting_calls=1,repeated_source_energy_reductions=0,independent_algorithms_same_process=True,original528_resource_gate=False,other_rare_and_full_metrics='UNKNOWN',scope='One unchanged original complete scarce domain, not a subset criterion; only C buffer representation corrected, no response/quantizer/source/model/native engine replay; IEEE prerequisites as frozen proof.'))
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()
