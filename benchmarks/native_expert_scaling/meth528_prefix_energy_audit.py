"""Independent C limbs plus independent Python joins/proof recurrence."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import ctypes
from fractions import Fraction
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth528_prefix_io import PrefixContext, ROOT, DOC, OLD, build, pointer, save, write


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest-sha', required=True); ap.add_argument('--freeze', required=True); ap.add_argument('--main-sha', required=True)
    args=ap.parse_args();ctx=None
    try:
        ctx=PrefixContext('prefix_energy_audit',args.manifest_sha,args.freeze)
        main_path=DOC/'meth528_prefix_bound_result.json'
        assert ctx.digest(main_path)==args.main_sha
        main_raw=json.loads(main_path.read_bytes())
        assert main_raw['source_freeze']==args.freeze and len(main_raw['gates'])==4 and all(main_raw['gates'].values())
        folder=ROOT/'results/native_expert_scaling/meth528_prefix_bound'
        assert json.loads((folder/'terminal_resource.json').read_bytes())['result_sha256']==args.main_sha
        for v in main_raw['output_inventory']:assert Path(v['path']).stat().st_size==v['bytes'] and ctx.digest(v['path'])==v['sha256']
        np=ctx.numpy();import meth528_contract as S
        dll,dll_sha=build(ctx,ROOT/'benchmarks/native_expert_scaling/meth528_prefix_energy.c','independent_integer')
        lib=ctypes.CDLL(str(dll));lib.prefix_energy.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*3;lib.prefix_energy.restype=ctypes.c_int
        lib.prefix_weighted.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*3;lib.prefix_weighted.restype=ctypes.c_int
        calls=dict(energy=0,weighting=0,negative_controls=0)

        def reduce_energy(truth,guess=None):
            truth=np.ascontiguousarray(truth,'<u4');guess=None if guess is None else np.ascontiguousarray(guess,'<u4')
            n,d=truth.shape;out=np.empty((n,10),'<u8');calls['energy']+=1
            assert lib.prefix_energy(n,d,pointer(truth),pointer(guess),pointer(out))==0
            return out

        def weighted(x,p):
            x=np.ascontiguousarray(x,'<f4');p=np.ascontiguousarray(p,'<f4');out=np.empty_like(x);calls['weighting']+=1
            assert lib.prefix_weighted(*x.shape,pointer(x),pointer(p),pointer(out))==0
            return out

        # Independent control oracle via exponent power; no imported main decoder.
        control_bits=np.array([[0,0x80000000,1,0x7fffff,0x800000,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,0x3f800001]],'<u4')
        def oracle(bit):
            sign=-1 if bit & (1<<31) else 1
            exponent=bit//(1<<23)%256;fraction=bit%(1<<23)
            return sign*(fraction if exponent==0 else (fraction+(1<<23))*2**(exponent-1))
        controls=[]
        for guess in (None,control_bits[:,::-1].copy()):
            result=reduce_energy(control_bits,guess)
            expected=sum((oracle(int(v))-(0 if guess is None else oracle(int(guess[0,j]))))**2 for j,v in enumerate(control_bits[0]))
            assert int.from_bytes(result.tobytes(),'little')==expected
            controls.append(dict(guess_reversed=guess is not None,expected_integer=str(expected)))
        scratch=np.empty((1,10),'<u8')
        for bit in (0x7f800000,0xff800000,0x7fc00001):
            invalid=np.array([[bit]],'<u4');calls['negative_controls']+=1
            assert lib.prefix_energy(1,1,pointer(invalid),None,pointer(scratch))==-2
        calls['negative_controls']+=1
        assert lib.prefix_energy(0,1,pointer(control_bits),None,pointer(scratch))==-1
        ctx.r['gates']['independent_integer_extreme_subnormal_signed_zero_cancellation_and_nonfinite_controls']=True
        m=S.records(ctx,'uid',slice(None))['m'];occ=S.records(ctx,'occurrences',slice(None))
        rows=m.tolist();occurrences=occ.tolist();counts=[0]*128
        assert [row[0] for row in rows]==list(range(17540)) and [row[0] for row in occurrences]==list(range(19962))
        for row in rows:
            assert row[2]==11 and row[6]==2 and 0<=row[3]<128
            assert bool(row[4]&5)!=bool(row[4]&2)
            if row[4]&5:counts[row[3]]+=1
        assert counts[0]==0 and all(v>0 for v in counts[1:])
        seen=[0]*17540
        for row in occurrences:
            uid=row[1];assert 0<=uid<17540 and row[3]==128 and row[8]==11 and row[12]==1
            assert all(row[a]==rows[uid][b] for a,b in ((14,1),(11,3),(15,12),(16,10)))
            seen[uid]+=1
        assert seen==[row[7] for row in rows]
        prefix=np.load(folder/'prefix_UIDs.npy',allow_pickle=False)
        assert prefix.tolist()==[row[0] for row in rows if row[3] in range(1,13)]
        physical=np.load(folder/'verified_physical.npy',allow_pickle=False)
        assert physical.dtype==np.dtype('<f4') and physical.shape==(len(prefix),2,768)
        assert physical[:,0].tobytes()==S.get(OLD,'physical',prefix)[:,0].tobytes()
        p=m[:,10].copy().view('<f4')
        assert weighted(physical[:,0],p[prefix]).tobytes()==physical[:,1].tobytes()
        ref=S.records(ctx,'unweighted',slice(None));target=S.records(ctx,'targets',slice(None))
        assert weighted(ref,p).tobytes()==target.tobytes()
        den=np.empty((17540,2,10),'<u8');error=np.zeros_like(den)
        for arm,truth in enumerate((ref,target)):
            den[:,arm]=reduce_energy(truth.view('<u4'))
            error[prefix,arm]=reduce_energy(truth[prefix].view('<u4'),physical[:,arm].view('<u4'))
            ctx.guard()
        for name,actual in (('source_U640',den),('prefix_error_U640',error)):
            recorded=np.load(folder/(name+'.npy'),allow_pickle=False)
            assert recorded.dtype==np.dtype('<u8') and recorded.shape==actual.shape and recorded.tobytes()==actual.tobytes()
            save(np,ctx.out,name,actual)
        ctx.r['gates']['ALL35080_full_source_and_ALL_prefix_exact_U640_energies_BYTE_and_C_F32_weighting']=True
        # Build only original gated domains by independent record traversal.
        domains=[]
        for label,lo,hi in (('1..4',1,4),('5..15',5,15)):
            ids=[row[0] for row in rows if row[4]&2 and lo<=counts[row[3]]<=hi]
            domains.append((dict(kind='rare',development_class=label,split='consumed_validation'),ids))
        for role in range(3):
            for mode in range(2):
                domains.append((dict(kind='role',role=role,mode=mode),[row[1] for row in occurrences if row[7]==role and row[6]==mode]))
        bound=json.loads((folder/'exact_bound.json').read_bytes());assert bound['views']==main_raw['views']
        threshold=Fraction(5764607523034235,576460752303423488)
        assert (str(threshold.numerator),str(threshold.denominator))==(bound['original_binary_threshold_numerator'],bound['original_binary_threshold_denominator'])
        confirmations=[];members=set(int(uid) for uid in prefix)
        for (label,ids),view in zip(domains,bound['views']):
            assert all(view[k]==v for k,v in label.items()) and view['count']==len(ids)
            assert view['prefix_occurrences']==sum(uid in members for uid in ids)
            # Independent staged error-factor recurrence; sqrt factor applied last.
            q=1<<53;lower=Fraction(q-1,q);upper=Fraction(q+1,q)
            numerator=lower*lower*lower
            denominator=upper
            for steps in (768-1,max(0,len(ids)-1)):
                numerator*=Fraction(q-2*steps,q-steps)
                denominator*=Fraction(q,q-steps)
            ratio_lower=numerator/denominator*lower
            squared_sqrt_lower=ratio_lower*lower*lower
            assert (str(squared_sqrt_lower.numerator),str(squared_sqrt_lower.denominator))==(view['rounding_lower_squared_factor_numerator'],view['rounding_lower_squared_factor_denominator'])
            certified=[]
            for arm,name in enumerate(('unweighted','weighted')):
                D=sum(int.from_bytes(den[uid,arm].tobytes(),'little') for uid in ids)
                L=sum(int.from_bytes(error[uid,arm].tobytes(),'little') for uid in ids)
                original=view[name]
                assert (str(D),str(L))==(original['full_denominator_integer'],original['prefix_error_integer'])
                left=L*squared_sqrt_lower.numerator*threshold.denominator**2
                right=D*squared_sqrt_lower.denominator*threshold.numerator**2
                assert (str(left),str(right))==(original['guarded_cross_left'],original['guarded_cross_right'])
                # Compare rationals directly, independently of main integer crossing.
                reject=bool(ids and (D==0 or Fraction(L,D)*squared_sqrt_lower>threshold*threshold))
                assert reject==original['strict_original_gate_failure']
                if reject:certified.append(name)
            confirmations.append(dict(**label,certified_failure_arms=certified))
        assert len(domains)==len(bound['views'])==8
        decision='NECESSARY_FIXED_DOMAIN_FIDELITY_FAILURE_CERTIFIED' if any(v['certified_failure_arms'] for v in confirmations) else 'INCONCLUSIVE_PREFIX_BOUND'
        assert decision==bound['decision']==main_raw['decision']
        ctx.r['gates']['ALL16_original_full_domain_guarded_rejections_independently_joined_and_certified']=True
        ctx.r['gates']['zero_repeated_C_response_calls_and_original528_failure_preserved']=True
        write(ctx.out/'independent_proof.json',dict(decision=decision,confirmations=confirmations,controls=controls,C_calls=calls,energy_verifier_sha256=dll_sha))
        ctx.finish(dict(decision=decision,independently_energy_audited=True,main_sha256=args.main_sha,energy_verifier_sha256=dll_sha,new_C_energy_weight_control_calls=calls,new_C_response_calls=0,confirmations=confirmations,prefix_UID_count=len(prefix),full528_eligibility='UNKNOWN; necessary criteria rejected where certified, original main resource FALSE',no_completed_source_model_native_replay=True))
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()
