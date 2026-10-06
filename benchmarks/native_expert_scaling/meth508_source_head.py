"""One full-vocabulary source readout counterfactual on ALL507 common histories."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct
import meth508_operations as O

def exact(x):return Fraction.from_float(float(x))
def record(x):return {'numerator':x.numerator,'denominator':x.denominator,'float':float(x)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binding-sha',required=True);args=parser.parse_args();ctx=O.Context('main')
    try:
        b=ctx.admit(args.binding_sha)
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        import numpy as np
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.modules()
        cfg,files,tensors=O.manifest(b['source_head']['manifest']['path']);row=tensors['shared.weight']
        assert row[:5]==(0,2,32128,768,0) and row[5]==b['source_head']['offset'] and row[6]==0 and row[7]==24674304
        assert Path(files[0]).resolve()==Path(b['source_head']['path']).resolve()
        with Path(b['source_head']['path']).open('rb') as stream:
            stream.seek(row[5]);headbytes=stream.read(b['source_head']['bytes'])
        assert hashlib.sha256(headbytes).hexdigest()==b['source_head']['sha256']
        W=np.frombuffer(headbytes,dtype='<f4').reshape(32128,768).astype('f8');assert np.isfinite(W).all();del headbytes
        scale=np.float32(1./np.sqrt(np.float64(768)))
        raw=json.loads(Path(b['raw506']['path']).read_bytes());prior=json.loads(Path(b['raw507']['path']).read_bytes())
        cases=[];total={'positions':0,'original_differences':0,'recovered':0,'introduced':0,'remaining_differences':0,'original_state_bridge_bad_positions':0}
        flip_total={'cases':0,'recovered':0,'unchanged':0,'changed_to_third_ID':0};maximum_bridge=0.
        for ordinal,(selected,p,c) in enumerate(zip(b['selected'],prior['cases'],raw['cases'])):
            assert (selected['book'],selected['index'])==(p['book'],p['index'])==(c['book'],c['index'])==divmod(ordinal,4)
            assert selected['wire']==c['native']['candidate']['generation'][0]['wire'] and selected['donor']==c['donor_reference']
            n=p['comparable_steps'];wire=Path(selected['wire']['path']).read_bytes();assert wire[:8]==b'SWR32O01'
            s,t,d,e,f,v,r=struct.unpack_from('<7I',wire,8);assert (s,d,e,f,v,r)==(29,768,12,12,32128,174+6*t) and n<=t<=64
            enc=14*29*768;dec=t*14*768;end=36+4*(enc+dec+t*32128);assert len(wire)==end+12*r
            native_states=np.frombuffer(wire,dtype='<f4',offset=36+4*enc,count=dec).reshape(t,14,768)[:n,-1]
            candidate_logits=np.frombuffer(wire,dtype='<f4',offset=36+4*(enc+dec),count=t*32128).reshape(t,32128)[:n]
            with np.load(selected['donor']['path'],allow_pickle=False) as ref:
                donor_states=ref['generation_decoder'][:n,-1];donor_logits=ref['generation_logits'][:n];official=ref['official_generation_logits'][:n]
                for a in (native_states,candidate_logits,donor_states,donor_logits,official):assert np.isfinite(a).all()
                ci=candidate_logits.argmax(1).tolist();di=donor_logits.argmax(1).tolist()
                assert ci==p['candidate_ids'][:n] and di==p['donor_ids'][:n] and official.argmax(1).tolist()==di
                assert [0]+ci[:-1]==[0]+di[:-1]==p['common_decoder_input_ids']
                # Tied embedding scale is rounded/multiplied in F32 before the F64 dot.
                xc=(native_states*scale).astype('f4');xd=(donor_states*scale).astype('f4')
                Hc=xc.astype('f8')@W.T;Hd=xd.astype('f8')@W.T
                assert Hc.shape==Hd.shape==(n,32128) and np.isfinite(Hc).all() and np.isfinite(Hd).all()
                norms=np.sqrt(np.sum((Hd-donor_logits.astype('f8'))**2,axis=1)/np.sum(donor_logits.astype('f8')**2,axis=1))
                assert np.isfinite(norms).all();hc_ids=Hc.argmax(1).tolist();hd_ids=Hd.argmax(1).tolist()
                assert (norms<=1e-6).all() and hd_ids==di, ('source_state_bridge_fault',ordinal,float(norms.max()),hd_ids,di)
                positions=[];decomposition=None
                for j in range(n):
                    original_bad=ci[j]!=di[j];counter_bad=hc_ids[j]!=di[j]
                    entry={'step':j,'donor_id':di[j],'candidate_id':ci[j],'source_head_candidate_state_id':hc_ids[j],
                           'source_head_donor_state_id':hd_ids[j],'original_disagreement':original_bad,'counterfactual_disagreement':counter_bad,
                           'recovered':original_bad and not counter_bad,'introduced':not original_bad and counter_bad,
                           'donor_bridge_relative_l2':float(norms[j]),'donor_bridge_winner_exact':hd_ids[j]==di[j],
                           'source_head_candidate_state_maximum_ties':int(np.count_nonzero(Hc[j]==Hc[j,hc_ids[j]])),
                           'source_head_donor_state_maximum_ties':int(np.count_nonzero(Hd[j]==Hd[j,hd_ids[j]]))}
                    maximum_bridge=max(maximum_bridge,float(norms[j]));total['positions']+=1;total['original_differences']+=int(original_bad)
                    total['recovered']+=int(entry['recovered']);total['introduced']+=int(entry['introduced']);total['remaining_differences']+=int(counter_bad)
                    total['original_state_bridge_bad_positions']+=int(norms[j]>1e-6 or hd_ids[j]!=di[j])
                    positions.append(entry)
                    if original_bad:
                        assert p['divergent'] and j==p['common_prefix_length']==n-1
                        winner,other=di[j],ci[j]
                        pl=exact(candidate_logits[j,other])-exact(candidate_logits[j,winner]);pd=exact(donor_logits[j,other])-exact(donor_logits[j,winner])
                        pc=exact(Hc[j,other])-exact(Hc[j,winner]);phd=exact(Hd[j,other])-exact(Hd[j,winner])
                        readout=pl-pc;upstream=pc-phd;arithmetic=phd-pd;perturbation=pl-pd
                        assert readout+upstream+arithmetic==perturbation and perturbation>=-pd
                        decomposition={'step':j,'donor_id':winner,'candidate_id':other,'donor_margin':record(-pd),
                                       'total_pair_perturbation':record(perturbation),'readout_residual':record(readout),
                                       'upstream_scaled_state_effect':record(upstream),'donor_arithmetic_bridge_residual':record(arithmetic),
                                       'actual_pair_logits_candidate':[float(candidate_logits[j,winner]),float(candidate_logits[j,other])],
                                       'actual_pair_logits_donor':[float(donor_logits[j,winner]),float(donor_logits[j,other])],
                                       'source_head_pair_candidate_state':[float(Hc[j,winner]),float(Hc[j,other])],
                                       'source_head_pair_donor_state':[float(Hd[j,winner]),float(Hd[j,other])],
                                       'source_head_candidate_state_winner':hc_ids[j],'source_head_recovers_donor_winner':hc_ids[j]==winner}
                        flip_total['cases']+=1;flip_total['recovered']+=int(hc_ids[j]==winner);flip_total['unchanged']+=int(hc_ids[j]==other)
                        flip_total['changed_to_third_ID']+=int(hc_ids[j] not in (winner,other))
                cases.append({'book':p['book'],'index':p['index'],'source_id':p['source_id'],'comparable_steps':n,
                              'wire_sha256':selected['wire']['sha256'],'donor_npz_sha256':selected['donor']['sha256'],
                              'positions':positions,'first_divergence_decomposition':decomposition})
                del Hc,Hd
            del wire;ctx.guard()
            if ordinal%4==3:ctx.log(book=p['book'],cases=ordinal+1)
        assert len(cases)==96 and total['positions']==996 and total['original_differences']==flip_total['cases']==31
        assert total['remaining_differences']==31-total['recovered']+total['introduced']
        bridge_pass=total['original_state_bridge_bad_positions']==0
        summary={'all_common_positions':total,'first_divergences':flip_total,'source_state_bridge_maximum_relative_l2':maximum_bridge,
                 'source_state_bridge_qualified':bridge_pass,'eligible_fixed_state_interpretation':bridge_pass,
                 'scope':'Consumed fixed states, F64 source dots after F32 tied scale; no new generation/quality/rate/DRAM/useful-n claim.'}
        ctx.r['gates']['actual_source_shared_tied_head_extent_SHA_and_full_vocabulary_finite']=True
        ctx.r['gates']['ALL96_996_common_history_positions_and_original31_differences']=True
        ctx.r['gates']['ALL31_exact_computed_rational_pair_decompositions']=True
        ctx.r['gates']['ALL965_matching_positions_checked_for_new_disagreements']=True
        ctx.finish({'source_head':b['source_head'],'F32_tied_scale':float(scale),'cases':cases,'summary':summary,
                    'source_linear_vector_evaluations':2*total['positions'],'source_state_bridge_gate_pass':bridge_pass,'goal_complete':False})
    except BaseException:ctx.fail();raise
    finally:ctx.close()

if __name__=='__main__':main()
