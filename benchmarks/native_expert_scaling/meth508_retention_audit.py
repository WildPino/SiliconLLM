"""Independent extent/format/state/head/winner and exact pair decomposition audit."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import meth508_operations as O

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binding-sha',required=True);parser.add_argument('--raw-sha',required=True)
    args=parser.parse_args();ctx=O.Context('audit')
    try:
        b=ctx.admit(args.binding_sha);mainpath=O.DOC/'meth508_main_result.json';assert ctx.digest(mainpath)==args.raw_sha
        mainrecord=json.loads(mainpath.read_bytes());prior=json.loads(Path(b['raw507']['path']).read_bytes());original=json.loads(Path(b['raw506']['path']).read_bytes())
        assert mainrecord['binding_sha256']==args.binding_sha and len(mainrecord['cases'])==len(prior['cases'])==len(original['cases'])==96
        # Independent sequential stream parser, not O.manifest or primal numerical code.
        with Path(b['source_head']['manifest']['path']).open('rb') as f:
            assert f.read(8)==b'SWI8C001';cfg=struct.unpack('<13If',f.read(56));nf,nt=struct.unpack('<2I',f.read(8));assert nf==1 and nt==3320
            def text():
                n=struct.unpack('<I',f.read(4))[0];return f.read(n).decode('utf8')
            files=[text() for _ in range(nf)];allrows={}
            for _ in range(nt):
                name=text();assert name not in allrows;allrows[name]=struct.unpack('<5I3Q',f.read(44))
            assert not f.read(1)
        row=allrows['shared.weight'];assert row[:5]==(0,2,32128,768,0) and row[5:]==(b['source_head']['offset'],0,24674304)
        assert str(Path(files[0]).resolve())==b['source_head']['path']
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        import numpy as np
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.modules()
        with Path(files[0]).open('rb') as f:f.seek(row[5]);data=f.read(4*row[7])
        assert hashlib.sha256(data).hexdigest()==b['source_head']['sha256'];W=np.frombuffer(data,dtype='<f4').reshape(32128,768).astype('f8');del data
        assert np.isfinite(W).all();scale=np.float32(1/math.sqrt(768));assert float(scale)==mainrecord['F32_tied_scale']

        def q(x):return Fraction(*float(x).as_integer_ratio())
        def encoded(x):return {'numerator':x.numerator,'denominator':x.denominator,'float':float(x)}
        def close(a,e):
            if isinstance(e,dict):
                assert set(a)==set(e)
                for k in e:close(a[k],e[k])
            elif isinstance(e,list):
                assert len(a)==len(e)
                for x,y in zip(a,e):close(x,y)
            elif isinstance(e,float):assert abs(a-e)<=3e-11*max(1,abs(e)),(a,e)
            else:assert a==e,(a,e)

        totals={'positions':0,'original_differences':0,'recovered':0,'introduced':0,'remaining_differences':0,'original_state_bridge_bad_positions':0}
        flips={'cases':0,'recovered':0,'unchanged':0,'changed_to_third_ID':0};maximum_bridge=0.
        for ordinal,(bound,p,saved,c) in enumerate(zip(b['selected'],prior['cases'],mainrecord['cases'],original['cases'])):
            assert (bound['book'],bound['index'])==(p['book'],p['index'])==(saved['book'],saved['index'])==(c['book'],c['index'])==divmod(ordinal,4)
            assert bound['wire']==c['native']['candidate']['generation'][0]['wire'] and bound['donor']==c['donor_reference']
            assert saved['source_id']==p['source_id']==c['source_id'] and saved['wire_sha256']==bound['wire']['sha256'] and saved['donor_npz_sha256']==bound['donor']['sha256']
            n=p['comparable_steps'];assert saved['comparable_steps']==n and len(saved['positions'])==n
            with Path(bound['wire']['path']).open('rb') as f:
                assert f.read(8)==b'SWR32O01';s,t,d,e,de,v,r=struct.unpack('<7I',f.read(28));assert (s,d,e,de,v,r)==(29,768,12,12,32128,174+6*t) and n<=t<=64
                f.seek(36+4*14*29*768);native=np.frombuffer(f.read(4*t*14*768),dtype='<f4').reshape(t,14,768)[:n,13]
                candidate=np.frombuffer(f.read(4*t*32128),dtype='<f4').reshape(t,32128)[:n]
                assert len(f.read())==12*r
            with np.load(bound['donor']['path'],allow_pickle=False) as ref:
                donor=ref['generation_decoder'][:n,13];expected=ref['generation_logits'][:n];official=ref['official_generation_logits'][:n]
                for a in (native,candidate,donor,expected,official):assert np.isfinite(a).all()
                ci=candidate.argmax(1).tolist();di=expected.argmax(1).tolist()
                assert ci==p['candidate_ids'][:n]==c['candidate_task']['generated_ids'][:n] and di==p['donor_ids'][:n]==c['donor_task']['generated_ids'][:n]
                assert official.argmax(1).tolist()==di and [0]+ci[:-1]==[0]+di[:-1]==p['common_decoder_input_ids']
                xc=np.multiply(native,scale,dtype='f4').astype('f8');xd=np.multiply(donor,scale,dtype='f4').astype('f8')
                # Separate spelling/control flow, same declared F64 dot; no import of primal science.
                Hc=np.einsum('ij,kj->ik',xc,W,optimize=True);Hd=np.einsum('ij,kj->ik',xd,W,optimize=True)
                assert Hc.shape==Hd.shape==(n,32128) and np.isfinite(Hc).all() and np.isfinite(Hd).all()
                hid=Hc.argmax(1).tolist();did=Hd.argmax(1).tolist();decomposition=None
                for j,entry in enumerate(saved['positions']):
                    err=Hd[j]-expected[j].astype('f8');norm=float(np.sqrt(np.add.reduce(err*err)/np.add.reduce(expected[j].astype('f8')**2)))
                    original_bad=ci[j]!=di[j];counter_bad=hid[j]!=di[j]
                    desired={'step':j,'donor_id':di[j],'candidate_id':ci[j],'source_head_candidate_state_id':hid[j],'source_head_donor_state_id':did[j],
                             'original_disagreement':original_bad,'counterfactual_disagreement':counter_bad,
                             'recovered':original_bad and not counter_bad,'introduced':not original_bad and counter_bad,
                             'donor_bridge_relative_l2':norm,'donor_bridge_winner_exact':did[j]==di[j],
                             'source_head_candidate_state_maximum_ties':int((Hc[j]==Hc[j,hid[j]]).sum()),
                             'source_head_donor_state_maximum_ties':int((Hd[j]==Hd[j,did[j]]).sum())}
                    assert hid[j]==int(np.flatnonzero(Hc[j]==Hc[j,hid[j]])[0]) and did[j]==int(np.flatnonzero(Hd[j]==Hd[j,did[j]])[0]);close(entry,desired)
                    maximum_bridge=max(maximum_bridge,norm)
                    totals['positions']+=1;totals['original_differences']+=int(original_bad);totals['recovered']+=int(desired['recovered']);totals['introduced']+=int(desired['introduced']);totals['remaining_differences']+=int(counter_bad)
                    totals['original_state_bridge_bad_positions']+=int(norm>1e-6 or did[j]!=di[j])
                    if original_bad:
                        assert j==n-1==p['common_prefix_length'] and p['divergent']
                        d,cid=di[j],ci[j]
                        pairc=q(candidate[j,cid])-q(candidate[j,d]);paird=q(expected[j,cid])-q(expected[j,d])
                        # Primal stores computed F64 cells. Audit recomputation can differ
                        # by last bits under einsum; validate cells numerically then verify
                        # THEIR exact rational bookkeeping, not equality of GEMM reductions.
                        held=saved['first_divergence_decomposition']
                        close(held['source_head_pair_candidate_state'],[float(Hc[j,d]),float(Hc[j,cid])]);close(held['source_head_pair_donor_state'],[float(Hd[j,d]),float(Hd[j,cid])])
                        heldpc=q(held['source_head_pair_candidate_state'][1])-q(held['source_head_pair_candidate_state'][0]);heldpd=q(held['source_head_pair_donor_state'][1])-q(held['source_head_pair_donor_state'][0])
                        decomposition={'step':j,'donor_id':d,'candidate_id':cid,'donor_margin':encoded(-paird),
                                       'total_pair_perturbation':encoded(pairc-paird),'readout_residual':encoded(pairc-heldpc),
                                       'upstream_scaled_state_effect':encoded(heldpc-heldpd),'donor_arithmetic_bridge_residual':encoded(heldpd-paird),
                                       'actual_pair_logits_candidate':[float(candidate[j,d]),float(candidate[j,cid])],
                                       'actual_pair_logits_donor':[float(expected[j,d]),float(expected[j,cid])],
                                       'source_head_pair_candidate_state':held['source_head_pair_candidate_state'],
                                       'source_head_pair_donor_state':held['source_head_pair_donor_state'],
                                       'source_head_candidate_state_winner':hid[j],'source_head_recovers_donor_winner':hid[j]==d}
                        assert (pairc-heldpc)+(heldpc-heldpd)+(heldpd-paird)==pairc-paird and pairc-paird>=-paird
                        flips['cases']+=1;flips['recovered']+=int(hid[j]==d);flips['unchanged']+=int(hid[j]==cid);flips['changed_to_third_ID']+=int(hid[j] not in (d,cid))
                close(saved['first_divergence_decomposition'],decomposition);del Hc,Hd
            ctx.guard()
            if ordinal%4==3:ctx.log(audited_book=p['book'])
        assert totals['positions']==996 and totals['original_differences']==flips['cases']==31
        assert totals['remaining_differences']==31-totals['recovered']+totals['introduced']
        bridge=totals['original_state_bridge_bad_positions']==0
        assert bridge, 'independent_source_state_bridge_fault'
        summary={'all_common_positions':totals,'first_divergences':flips,'source_state_bridge_maximum_relative_l2':maximum_bridge,
                 'source_state_bridge_qualified':bridge,'eligible_fixed_state_interpretation':bridge,
                 'scope':'Consumed fixed states, F64 source dots after F32 tied scale; no new generation/quality/rate/DRAM/useful-n claim.'}
        close(mainrecord['summary'],summary);assert mainrecord['source_state_bridge_gate_pass']==bridge and mainrecord['source_linear_vector_evaluations']==1992
        ctx.r['gates']['independent_source_extent_manifest_full_vocabulary_finite']=True
        ctx.r['gates']['independent_ALL96_996_histories_and_source_state_bridge']=True
        ctx.r['gates']['independent_ALL_winners_ties_recoveries_AND_introductions']=True
        ctx.r['gates']['independent_ALL31_source_dot_cells_and_exact_recorded_rational_decompositions']=True
        ctx.r['gates']['independent_ALL24_books_and_terminal_summary']=True
        ctx.finish({'source_head':b['source_head'],'main_sha256':args.raw_sha,'summary':summary,'source_linear_vector_evaluations':1992,'goal_complete':False})
    except BaseException:ctx.fail();raise
    finally:ctx.close()

if __name__=='__main__':main()
