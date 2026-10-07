"""Complete two ORIGINAL scarce domains first; reuse all qualified apparatus."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import ctypes
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import meth528_prefix_io_repair4 as IO
from meth528_prefix_bound_repair4 import energy,limbs,lower_factor


class Context(IO.PrefixContext):
    def guard(self):
        super().guard()
        if time.monotonic()-getattr(self,'rare_output_check',-1.0)<1.0:return
        self.rare_output_check=time.monotonic()
        folders=list((IO.ROOT/'results/native_expert_scaling').glob('meth528_prefix*'))
        folders.append(IO.ROOT/'results/native_expert_scaling/meth528_rare_gate')
        folders.append(IO.ROOT/'results/native_expert_scaling/meth528_rare_gate_repair1')
        extra=sum(p.stat().st_size for folder in folders if folder.is_dir() for p in folder.iterdir() if p.is_file())
        from meth528_operations import output_bytes
        assert extra<=465<<20 and output_bytes()+extra<=2<<30


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha',required=True);ap.add_argument('--freeze',required=True);args=ap.parse_args();ctx=None
    try:
        IO.MANIFEST=IO.DOC/'meth528_rare_manifest_repair1_20261007.json'
        ctx=Context('rare_gate_repair1',args.manifest_sha,args.freeze);np=ctx.numpy()
        ctx.r.update(prefix_parents=[],new_candidate_parents=[15,36,69,115,124])
        import meth528_contract as S
        import meth528_atom_math as M
        upstream=IO.ROOT/'results/native_expert_scaling/meth528_prefix_bound_repair4'
        auditor=IO.ROOT/'results/native_expert_scaling/meth528_prefix_energy_audit_repair4'
        for path,sha,ngates in ((IO.DOC/'meth528_prefix_bound_repair4_result.json','81d5454195ce1ac9dae9bf791b31bd3edd13dc5fa4c77d6e1679e7d645a06f03',4),(IO.DOC/'meth528_prefix_energy_audit_repair4_result.json','8872d2df4c0d2b9d28804f586cd8857723cac5cc1cdf74166c0bf6642cb2ed6b',5)):
            assert ctx.digest(path)==sha
            raw=json.loads(path.read_bytes());assert len(raw['gates'])==ngates and all(raw['gates'].values()) and raw['decision']=='INCONCLUSIVE_PREFIX_BOUND'
        lib=ctypes.CDLL(str(upstream/'first_prefix_scalar.dll'));lib.m528_eval.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*11;lib.m528_eval.restype=ctypes.c_int
        reducer=ctypes.CDLL(str(auditor/'independent_integer.dll'));reducer.prefix_energy.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*3;reducer.prefix_energy.restype=ctypes.c_int
        reducer.prefix_weighted.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*3;reducer.prefix_weighted.restype=ctypes.c_int
        m,occ,dev,counts=S.metadata(ctx)
        ix=np.flatnonzero((~dev)&(counts[m[:,3]]>=1)&(counts[m[:,3]]<=15))
        assert len(ix)==22 and sorted(set(int(v) for v in m[ix,3]))==[15,36,69,115,124]
        assert not np.any((m[ix,3]>=1)&(m[ix,3]<=13))
        position={int(uid):i for i,uid in enumerate(ix)};physical=np.empty((22,768),'<f4')
        freeze=json.loads((IO.OLD/'development_mask_freeze.json').read_bytes());cases=[]
        for e in (15,36,69,115,124):
            ids=ix[m[ix,3]==e];inp=S.records(ctx,'inputs',ids)
            assert np.all(inp['accept']==1) and np.all(inp['e']==e) and inp['p'].tobytes()==m[ids,10].tobytes() and inp['alpha'].tobytes()==m[ids,12].tobytes()
            wi,si,wo,so=S.weights(ctx,e);u=S.old(ctx,'old_hinges',e).astype(int);signs=S.old(ctx,'old_signs',e);anchor=int(S.old(ctx,'old_anchors',e))
            assert dev[anchor] and m[anchor,3]==e
            selected=np.array(sorted(set(int(v) for v in u)|{j for j in range(3072) if signs[j]}),'<u2')
            saved=S.unpack(IO.OLD/f'e{e:03d}_a0.bin',e,0)
            assert all(a.tobytes(order='C')==b.tobytes(order='C') for a,b in zip(saved,(selected,si[selected],so,wi[selected],wo[:,selected])))
            assert freeze['cases'][e-1]['width']==len(selected)
            dots=S.old(ctx,'old_signed_dots',ids);assert np.max(np.abs(dots))<=768*128*32767
            out,integer,codes,alpha,_,_,_=M.function(dots[:,selected],inp['alpha'],saved[1],saved[4],saved[2],False)
            for name,value in (('main_physical',out),('main_integer',integer),('main_codes',codes),('main_alpha',alpha)):
                IO.save(np,ctx.out,f'e{e:03d}_'+name,value)
            inputs=[np.ascontiguousarray(a,dtype) for a,dtype in zip((dots[:,selected],inp['alpha'],saved[1],saved[4],saved[2]),('<i8','<f4','<f4','i1','<f4'))]
            co=np.empty_like(out);ci=np.empty_like(integer);cq=np.empty_like(codes);ca=np.empty_like(alpha)
            ctx.r['new_C_response_calls_entered']=len(cases)+1
            assert lib.m528_eval(len(ids),len(selected),*[IO.pointer(v) for v in [*inputs,cq,ca,ci,co,None,None]])==0
            for name,value in (('C_physical',co),('C_integer',ci),('C_codes',cq),('C_alpha',ca)):
                IO.save(np,ctx.out,f'e{e:03d}_'+name,value)
            for a,b in ((out,co),(integer,ci),(codes,cq),(alpha,ca)):assert a.tobytes()==b.tobytes()
            physical[[position[int(uid)] for uid in ids]]=out
            cases.append(dict(parent=e,UIDs=ids.tolist(),width=len(selected),BYTE=True))
            IO.write(ctx.out/f'e{e:03d}_checkpoint.json',cases[-1]);ctx.guard()
        ctx.r['gates']['ALL22_FIRST_numpy_candidate_responses_and_FIRST_independent_scalar_C_BYTE']=True
        source=[S.records(ctx,'unweighted',ix),S.records(ctx,'targets',ix)]
        p=m[ix,10].copy().view('<f4');candidate=[physical,np.multiply(physical,p[:,None],dtype=np.float32)]
        weighted=np.empty_like(physical)
        assert reducer.prefix_weighted(22,768,IO.pointer(physical),IO.pointer(p),IO.pointer(weighted))==0 and weighted.tobytes()==candidate[1].tobytes()
        known=np.load(upstream/'source_U640.npy',allow_pickle=False)
        errors=np.empty((22,2,10),'<u8')
        for arm in range(2):
            truth=source[arm].view('<u4');guess=candidate[arm].view('<u4')
            for j in range(22):errors[j,arm]=limbs(np,energy(truth[j].tolist(),guess[j].tolist()))
            ce=np.empty((22,10),'<u8')
            assert reducer.prefix_energy(22,768,IO.pointer(truth),IO.pointer(guess),IO.pointer(ce))==0
            assert ce.tobytes()==errors[:,arm].tobytes()
        IO.save(np,ctx.out,'UIDs',ix.astype('<u4'));IO.save(np,ctx.out,'physical',np.stack(candidate,axis=1));IO.save(np,ctx.out,'error_U640',errors)
        ctx.r['gates']['ALL_new_error_integer_energies_and_F32_weighting_independently_C_BYTE']=True
        rows=m.tolist();dc=[0]*128
        for row in rows:
            if row[4]&5:dc[row[3]]+=1
        threshold=Fraction(5764607523034235,576460752303423488);views=[]
        for label,lo,hi in (('1..4',1,4),('5..15',5,15)):
            ids=np.flatnonzero((~dev)&(counts[m[:,3]]>=lo)&(counts[m[:,3]]<=hi)).tolist()
            independently=[row[0] for row in rows if row[4]&2 and lo<=dc[row[3]]<=hi]
            assert ids==independently and all(uid in position for uid in ids)
            f=lower_factor(len(ids));q=1<<53;low=Fraction(q-1,q);up=Fraction(q+1,q)
            independent_factor=low**3/up
            for n in (767,len(ids)-1):independent_factor*=Fraction(q-2*n,q)
            independent_factor*=low**3
            assert independent_factor==f
            view=dict(development_class=label,count=len(ids),UIDs=ids)
            for arm,name in enumerate(('unweighted','weighted')):
                D=sum(int.from_bytes(known[uid,arm].tobytes(),'little') for uid in ids)
                N=sum(int.from_bytes(errors[position[uid],arm].tobytes(),'little') for uid in ids)
                left=N*f.numerator*threshold.denominator**2;right=D*f.denominator*threshold.numerator**2
                reject=bool(ids and (D==0 or left>right))
                assert reject==bool(independently and (D==0 or Fraction(N,D)*independent_factor>threshold**2))
                view[name]=dict(denominator_integer=str(D),error_integer=str(N),guarded_cross_left=str(left),guarded_cross_right=str(right),strict_original_gate_failure=reject,full_domain_real_RMS_display=math.sqrt(N/D) if D else None)
            views.append(view)
        assert [v['count'] for v in views]==[3,19]
        decision='NECESSARY_FIXED_DOMAIN_FIDELITY_FAILURE_CERTIFIED' if any(v[a]['strict_original_gate_failure'] for v in views for a in ('unweighted','weighted')) else 'INCONCLUSIVE_ORIGINAL_RARE_GATES'
        ctx.r['gates']['ALL_four_original_FULL_rare_criteria_joins_and_guarded_integer_decisions_independently_verified']=True
        ctx.finish(dict(decision=decision,cases=cases,views=views,new_numpy_candidate_calls=5,new_C_response_calls=5,new_candidate_vectors=22,new_C_error_energy_calls=2,new_C_weighting_calls=1,repeated_C_quant_or_wide_controls=0,repeated_source_energy_reductions=0,independent_algorithms_same_process=True,source_response_function_calls=0,model_calls=0,original528_resource_gate=False,full528_remaining_metrics='UNKNOWN',scope='Complete original two scarce validation domains only, reused independently qualified F32 full source energies/controls/DLLs; exact unchanged necessary full-gate failure, not complete528 or chatbot qualification.'))
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()
