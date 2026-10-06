"""Independent top8 ranks/row dots and full hybrid winner/tail arithmetic audit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import meth509_operations as O

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--raw-sha',required=True);args=ap.parse_args();ctx=O.Context('audit')
    try:
        b=ctx.admit(args.binding_sha);rp=O.DOC/'meth509_main_result.json';assert ctx.digest(rp)==args.raw_sha
        raw=json.loads(rp.read_bytes());oracle=json.loads(Path(b['oracle508']['path']).read_bytes());assert raw['binding_sha256']==args.binding_sha and raw['K']==8
        with Path(b['source_head']['manifest']['path']).open('rb') as f:
            assert f.read(8)==b'SWI8C001';cfg=struct.unpack('<13If',f.read(56));nf,nt=struct.unpack('<2I',f.read(8));assert (nf,nt)==(1,3320)
            def text():
                n=struct.unpack('<I',f.read(4))[0];return f.read(n).decode('utf8')
            filename=text();rows={}
            for _ in range(nt):
                name=text();assert name not in rows;rows[name]=struct.unpack('<5I3Q',f.read(44))
            assert not f.read(1)
        assert rows['shared.weight']==(0,2,32128,768,0,b['source_head']['offset'],0,24674304) and str(Path(filename).resolve())==b['source_head']['path']
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        import numpy as np
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.modules()
        with Path(filename).open('rb') as f:f.seek(b['source_head']['offset']);data=f.read(98697216)
        assert hashlib.sha256(data).hexdigest()==b['source_head']['sha256'];W=np.frombuffer(data,dtype='<f4').reshape(32128,768);assert np.isfinite(W).all()
        scale=np.float32(1/math.sqrt(768));assert float(scale)==raw['F32_tied_scale']==oracle['F32_tied_scale']
        def close(a,e):
            if isinstance(e,dict):
                assert set(a)==set(e)
                for k in e:close(a[k],e[k])
            elif isinstance(e,list):
                assert len(a)==len(e)
                for x,y in zip(a,e):close(x,y)
            elif isinstance(e,float):assert abs(a-e)<=3e-11*max(1,abs(e)),(a,e)
            else:assert a==e,(a,e)
        counts={key:0 for key in ('positions','source_winner_not_in_top8','hybrid_differs_full_source_winner','selected_only_differs_full_source_winner',
                                  'unrefined_tail_wins','original_differences','recovered','introduced','remaining_differences','maximum_source_winner_I8_rank')}
        hist={};books=[]
        assert len(raw['cases'])==len(oracle['cases'])==len(b['selected'])==96
        for i,(row,saved,ref) in enumerate(zip(b['selected'],raw['cases'],oracle['cases'])):
            assert (row['book'],row['index'])==(saved['book'],saved['index'])==(ref['book'],ref['index'])==divmod(i,4)
            assert saved['wire_sha256']==row['wire']['sha256']==ref['wire_sha256'] and saved['source_id']==ref['source_id']
            with Path(row['wire']['path']).open('rb') as f:
                assert f.read(8)==b'SWR32O01';s,t,d,e,de,v,nr=struct.unpack('<7I',f.read(28));assert (s,d,e,de,v,nr)==(29,768,12,12,32128,174+6*t)
                f.seek(36+4*14*29*768);states=np.frombuffer(f.read(4*t*14*768),dtype='<f4').reshape(t,14,768)
                logits=np.frombuffer(f.read(4*t*32128),dtype='<f4').reshape(t,32128);assert len(f.read())==12*nr
            assert np.isfinite(states).all() and np.isfinite(logits).all() and len(saved['positions'])==len(ref['positions'])==ref['comparable_steps']<=t<=64
            for j,(held,p) in enumerate(zip(saved['positions'],ref['positions'])):
                old=logits[j];oldwinner=int(np.flatnonzero(old==old.max())[0]);assert oldwinner==p['candidate_id']
                # Independent repeated maximum selection, explicitly lowest-ID ties.
                remaining=old.astype('f8').copy();chosen=[]
                for _ in range(8):
                    best=int(np.flatnonzero(remaining==remaining.max())[0]);chosen.append(best);remaining[best]=-np.inf
                source=p['source_head_candidate_state_id'];rank=1+sum(int(x>old[source] or (x==old[source] and k<source)) for k,x in enumerate(old))
                x=np.multiply(states[j,13],scale,dtype='f4').astype('f8')
                source64=[float(np.add.reduce(W[k].astype('f8')*x)) for k in chosen]
                close(held['chosen_source_F64_logits'],source64)
                # Check recomputed F32 cells EXACT; under declared reduction any
                # rounding disagreement is an arithmetic fault, not a tolerance change.
                refined=np.asarray(source64,dtype='f4');assert refined.tolist()==held['chosen_refined_F32_logits']
                selectedwinner=min(k for k,z in zip(chosen,refined) if z==max(refined));hybrid=old.copy();hybrid[chosen]=refined
                winner=int(np.flatnonzero(hybrid==hybrid.max())[0]);donor=p['donor_id'];original_bad=oldwinner!=donor;newbad=winner!=donor
                wanted={'step':j,'old_I8_winner':oldwinner,'donor_winner':donor,'full_source_candidate_state_winner':source,'full_source_winner_I8_rank':rank,
                        'chosen_original_IDs':chosen,'chosen_old_logits':[float(old[k]) for k in chosen],'chosen_source_F64_logits':held['chosen_source_F64_logits'],
                        'chosen_refined_F32_logits':refined.tolist(),'selected_only_winner':selectedwinner,'hybrid_winner':winner,'hybrid_maximum_ties':int((hybrid==hybrid[winner]).sum()),
                        'source_winner_in_top8':source in chosen,'hybrid_equals_full_source_winner':winner==source,'unrefined_tail_wins':winner not in chosen,
                        'original_disagreement':original_bad,'hybrid_disagreement':newbad,'recovered':original_bad and not newbad,'introduced':not original_bad and newbad}
                close(held,wanted);hist[str(rank)]=hist.get(str(rank),0)+1
                counts['positions']+=1;counts['source_winner_not_in_top8']+=int(source not in chosen);counts['hybrid_differs_full_source_winner']+=int(winner!=source)
                counts['selected_only_differs_full_source_winner']+=int(selectedwinner!=source);counts['unrefined_tail_wins']+=int(winner not in chosen)
                counts['original_differences']+=int(original_bad);counts['recovered']+=int(wanted['recovered']);counts['introduced']+=int(wanted['introduced']);counts['remaining_differences']+=int(newbad)
                counts['maximum_source_winner_I8_rank']=max(counts['maximum_source_winner_I8_rank'],rank)
            ctx.guard()
            if i%4==3:books.append(row['book']);ctx.log(audited_book=row['book'])
        assert counts['positions']==996 and counts['original_differences']==31 and counts['remaining_differences']==31-counts['recovered']+counts['introduced']
        gates={'ALL996_source_winner_in_top8':counts['source_winner_not_in_top8']==0,'ALL996_hybrid_winner_equals_qualified_source_head':counts['hybrid_differs_full_source_winner']==0}
        summary={'counts':counts,'source_winner_rank_histogram':hist,'local_eligibility_gates':gates,'local_recipe_pass':all(gates.values()),
                 'full_source508_counts':oracle['summary']['all_common_positions'],'scope':'ONE consumed top8 hybrid; no unseen guarantee, new generation/quality/speed/DRAM/useful-n claim.'}
        close(raw['summary'],summary);assert raw['selected_source_row_evaluations']==7968
        ctx.r['gates']['independent_ALL_actual_manifest_source_extent_and_used_wires']=True
        ctx.r['gates']['independent_ALL996_exact_ranks_and_lowest_ID_top8']=True
        ctx.r['gates']['independent_ALL7968_selected_source_rows_F64_and_exact_F32_cells']=True
        ctx.r['gates']['independent_ALL996_full_hybrid_tail_winners_ties_and_new_changes']=True
        ctx.r['gates']['independent_ALL24_books_and_frozen_local_decisions']=True
        ctx.finish({'source_head':b['source_head'],'main_sha256':args.raw_sha,'summary':summary,'audited_books':books,'selected_source_row_evaluations':7968,'goal_complete':False})
    except BaseException:ctx.fail();raise
    finally:ctx.close()

if __name__=='__main__':main()
