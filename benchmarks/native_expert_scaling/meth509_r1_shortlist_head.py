"""ONE top8 conditional F32-row head, full hybrid tail kept and checked."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import meth509_r1_operations as O

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);args=ap.parse_args();ctx=O.Context('main')
    try:
        b=ctx.admit(args.binding_sha);os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        import numpy as np
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.modules()
        with Path(b['source_head']['path']).open('rb') as f:f.seek(b['source_head']['offset']);head=f.read(b['source_head']['bytes'])
        assert hashlib.sha256(head).hexdigest()==b['source_head']['sha256'];W=np.frombuffer(head,dtype='<f4').reshape(32128,768);assert np.isfinite(W).all()
        oracle=json.loads(Path(b['oracle508']['path']).read_bytes());scale=np.float32(1./np.sqrt(np.float64(768)));assert float(scale)==oracle['F32_tied_scale']
        total={'positions':0,'source_winner_not_in_top8':0,'hybrid_differs_full_source_winner':0,'selected_only_differs_full_source_winner':0,
               'unrefined_tail_wins':0,'original_differences':0,'recovered':0,'introduced':0,'remaining_differences':0,'maximum_source_winner_I8_rank':0}
        histogram={};cases=[];ids=np.arange(32128)
        for i,(row,c) in enumerate(zip(b['selected'],oracle['cases'])):
            assert (row['book'],row['index'])==(c['book'],c['index'])==divmod(i,4);assert row['wire']['sha256']==c['wire_sha256']
            data=Path(row['wire']['path']).read_bytes();assert data[:8]==b'SWR32O01';s,t,d,e,de,v,nr=struct.unpack_from('<7I',data,8)
            assert (s,d,e,de,v,nr)==(29,768,12,12,32128,174+6*t)
            enc=14*29*768;dec=t*14*768;assert len(data)==36+4*(enc+dec+t*32128)+12*nr
            states=np.frombuffer(data,dtype='<f4',offset=36+4*enc,count=dec).reshape(t,14,768)
            logits=np.frombuffer(data,dtype='<f4',offset=36+4*(enc+dec),count=t*32128).reshape(t,32128)
            n=c['comparable_steps'];assert n<=t<=64 and n==len(c['positions']);assert np.isfinite(states).all() and np.isfinite(logits).all()
            out=[]
            for j,p in enumerate(c['positions']):
                old=logits[j];oldwinner=int(old.argmax());assert oldwinner==p['candidate_id']
                sourcewinner=p['source_head_candidate_state_id'];donor=p['donor_id']
                chosen=np.lexsort((ids,-old.astype('f8')))[:8];assert len(set(chosen.tolist()))==8
                rank=1+int(np.count_nonzero(old>old[sourcewinner]))+int(np.count_nonzero((old==old[sourcewinner])&(ids<sourcewinner)))
                x=(states[j,13]*scale).astype('f4')
                selected64=W[chosen].astype('f8')@x.astype('f8');selected=selected64.astype('f4');assert np.isfinite(selected).all()
                rerank=int(np.min(chosen[selected==selected.max()]))
                hybrid=old.copy();hybrid[chosen]=selected;winner=int(hybrid.argmax());ties=int(np.count_nonzero(hybrid==hybrid[winner]))
                inlist=sourcewinner in chosen;oldbad=oldwinner!=donor;newbad=winner!=donor;tail=winner not in chosen
                result={'step':j,'old_I8_winner':oldwinner,'donor_winner':donor,'full_source_candidate_state_winner':sourcewinner,
                        'full_source_winner_I8_rank':rank,'chosen_original_IDs':chosen.tolist(),
                        'chosen_old_logits':[float(old[k]) for k in chosen],'chosen_source_F64_logits':selected64.tolist(),'chosen_refined_F32_logits':selected.tolist(),
                        'selected_only_winner':rerank,'hybrid_winner':winner,'hybrid_maximum_ties':ties,
                        'source_winner_in_top8':bool(inlist),'hybrid_equals_full_source_winner':winner==sourcewinner,'unrefined_tail_wins':bool(tail),
                        'original_disagreement':oldbad,'hybrid_disagreement':newbad,'recovered':oldbad and not newbad,'introduced':not oldbad and newbad}
                out.append(result);histogram[str(rank)]=histogram.get(str(rank),0)+1
                total['positions']+=1;total['source_winner_not_in_top8']+=int(not inlist);total['hybrid_differs_full_source_winner']+=int(winner!=sourcewinner)
                total['selected_only_differs_full_source_winner']+=int(rerank!=sourcewinner);total['unrefined_tail_wins']+=int(tail)
                total['original_differences']+=int(oldbad);total['recovered']+=int(result['recovered']);total['introduced']+=int(result['introduced']);total['remaining_differences']+=int(newbad)
                total['maximum_source_winner_I8_rank']=max(total['maximum_source_winner_I8_rank'],rank)
            cases.append({'book':row['book'],'index':row['index'],'source_id':c['source_id'],'wire_sha256':row['wire']['sha256'],'positions':out});ctx.guard()
            if i%4==3:ctx.log(book=row['book'],cases=i+1)
        assert len(cases)==96 and total['positions']==996 and total['original_differences']==31
        assert total['remaining_differences']==31-total['recovered']+total['introduced']
        gates={'ALL996_source_winner_in_top8':total['source_winner_not_in_top8']==0,
               'ALL996_hybrid_winner_equals_qualified_source_head':total['hybrid_differs_full_source_winner']==0}
        summary={'counts':total,'source_winner_rank_histogram':histogram,'local_eligibility_gates':gates,'local_recipe_pass':all(gates.values()),
                 'full_source508_counts':oracle['summary']['all_common_positions'],'scope':'ONE consumed top8 hybrid; no unseen guarantee, new generation/quality/speed/DRAM/useful-n claim.'}
        ctx.r['gates']['ALL96_996_original_wires_and_qualified508_oracle_join']=True
        ctx.r['gates']['ALL996_deterministic_top8_source_rows_F32_scale_and_refinement']=True
        ctx.r['gates']['ALL996_full_hybrid_tail_argmax_AND_recoveries_introductions_retained']=True
        ctx.finish({'source_head':b['source_head'],'K':8,'F32_tied_scale':float(scale),'cases':cases,'summary':summary,'selected_source_row_evaluations':7968,'goal_complete':False})
    except BaseException:ctx.fail();raise
    finally:ctx.close()

if __name__=='__main__':main()
