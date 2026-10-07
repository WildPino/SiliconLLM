"""NEW variable-width original-source atom functions; reuse admitted signed dots."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth528_operations import Context,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);args=ap.parse_args();ctx=None
    try:
        ctx=Context('main',args.binding_sha);np=ctx.numpy();import meth528_contract as S;import meth528_atom_math as M
        write(ctx.out/'controls.json',M.controls());ctx.r['gates']['NEW_piecewise_atom_fold_P_M_and_own_A16_tie_zero_wide_WO_controls']=True
        m,occ,dev,counts=S.metadata(ctx);ctx.r['gates']['ALL_original_UID_parent_mass_occurrence_and_consumed_diagnostic_domains']=True
        cases=[];prices=np.zeros(17540,'<i8');S.create(ctx.out)
        # ALL masks and byte-defined banks freeze before NEW function outputs.
        for e in range(1,128):
            wi,si,wo,so=S.weights(ctx,e);u=S.old(ctx,'old_hinges',e).astype(int);signs=S.old(ctx,'old_signs',e).astype(bool)
            anchor=int(S.old(ctx,'old_anchors',e));assert dev[anchor] and m[anchor,3]==e
            v=np.union1d(u,np.flatnonzero(signs));rotation=np.sort((v+1536)%3072)
            assert len(v)==len(rotation) and np.array_equal(u,np.unique(u)) and len(u)==512
            for arm,ids in enumerate((v,rotation)):S.pack(ctx.out/f'e{e:03d}_a{arm}.bin',e,arm,ids,wi,si,wo,so)
            bytes_=32+6*len(v)+3072+1536*len(v);prices[m[:,3]==e]=bytes_
            cases.append(dict(expert=e,anchor_UID=anchor,development=int(counts[e]),width=len(v),extra_anchor_positive_rows=len(v)-512,
                              candidate_bank_bytes=bytes_,active_MACs=1536*len(v),rotation=1536))
            ctx.guard()
        write(ctx.out/'development_mask_freeze.json',dict(cases=cases,rule='original523 U512 UNION all anchor-positive IDs; matched half-bank cyclic rotation; no fallback'))
        ctx.r['gates']['ALL127_dev_ONLY_variable_masks_original_I8_scales_packed_banks_frozen_BEFORE_new_outputs']=True
        exact=0;preserved_max=0;seen=np.zeros(17540,bool);integer_terms=0;identity_max=0.
        for case in cases:
            e=case['expert'];ix=np.flatnonzero(m[:,3]==e);inp=S.records(ctx,'inputs',ix)
            assert np.all(inp['accept']==1) and np.array_equal(inp['e'],m[ix,3]) and inp['p'].tobytes()==m[ix,10].tobytes()
            assert inp['alpha'].tobytes()==m[ix,12].tobytes() and np.all(inp['alpha']>0)
            dots=S.old(ctx,'old_signed_dots',ix);assert dots.dtype==np.dtype('<i8') and dots.shape==(len(ix),3072) and np.max(np.abs(dots))<=768*128*32767
            old=S.old(ctx,'old_predictions',ix);source_h=S.old(ctx,'hidden',ix);ref=S.records(ctx,'unweighted',ix);target=S.records(ctx,'targets',ix)
            assert np.multiply(ref,inp['p'][:,None],dtype=np.float32).tobytes()==target.tobytes()
            wi,si,wo,so=S.weights(ctx,e);u=S.old(ctx,'old_hinges',e).astype(int);signs=S.old(ctx,'old_signs',e).astype(bool)
            f=np.flatnonzero(signs&~np.isin(np.arange(3072),u));bank=[];physical=np.empty((len(ix),2,768),'<f4');ints=np.empty((len(ix),2,768),'<i8');scales=np.empty((len(ix),2),'<f4')
            for arm in range(2):
                ids,sii,soo,wii,wop=S.unpack(ctx.out/f'e{e:03d}_a{arm}.bin',e,arm)
                answer=M.function(dots[:,ids],inp['alpha'],sii,wop,soo,arm==0)
                out,dd,qh,ah,cont,h,absolute=answer
                assert h.astype('<f4').tobytes()==source_h[:,ids].tobytes()
                physical[:,arm]=out;ints[:,arm]=dd;scales[:,arm]=ah;bank.append((ids,qh,ah,cont,absolute))
                with (ctx.out/f'e{e:03d}_a{arm}_hidden_codes.npy').open('xb') as file:np.save(file,qh,allow_pickle=False)
                integer_terms+=len(ix)*768*len(ids)
            v,qh,ah,continuous,abs_atoms=bank[0];outside=np.ones(3072,bool);outside[v]=False
            zero=np.all(source_h[:,outside]==0,axis=1);original_a=S.old(ctx,'scales',ix);original_q=S.old(ctx,'codes',ix)
            assert qh[zero].tobytes()==original_q[zero][:,v].tobytes() and ah[zero].tobytes()==original_a[zero].tobytes()
            assert physical[zero,0].tobytes()==ref[zero].tobytes();exact+=int(zero.sum());preserved_max+=int(np.sum(ah.view('<u4')==original_a.view('<u4')))
            zf=(-dots[:,f].astype('<f8')*si[f].astype('<f8'))*inp['alpha'].astype('<f8')[:,None];hf=np.maximum(zf,0)
            of=wo[:,f].astype('<f8')*so.astype('<f8')[:,None];negative=hf@of.T;abs_m=np.abs(hf)@np.abs(of).T
            bound=M.identity_bound(inp,source_h,wi,si,wo,so,u,f,S.old(ctx,'old_fold64',e),abs_atoms,abs_m)
            stats=M.metrics(continuous,negative,physical,old,ref,target,inp['p'],bound);identity_max=max(identity_max,float(np.max(stats[:,11])))
            for name,value in (('physical',physical),('integer_WO',ints),('continuous',continuous),('negative_fold',negative),('hidden_alpha',scales),('metrics',stats)):S.put(ctx.out,name,ix,value)
            case.update(omitted_zero_BYTE_equal_UIDs=int(zero.sum()),global_hidden_max_preserved_UIDs=int(np.sum(ah.view('<u4')==original_a.view('<u4'))));seen[ix]=True
            ctx.guard()
            if e%16==0:print('new_atom_parent_complete='+str(e),flush=True)
        assert seen.all();ctx.r['gates']['ALL17540_NEW_candidate_rotation_own_quantizers_exact_WO_and_structural_negative_fold_bounds']=True
        data=S.get(ctx.out,'metrics',slice(None));views=[M.summarize(label,ids,data,prices) for label,ids in S.domain_ids(m,occ,counts)]
        assert len(views)==656;ctx.r['gates']['ALL656_original_report_domains_P_M_cross_codec_and_active_byte_statistics']=True
        eligible=M.eligibility(views);decision='VARIABLE_SOURCE_ATOMS_ELIGIBLE_FOR_NEW_MULTIREGION_TEST_NOT_WHOLE_PROMOTION' if all(eligible.values()) else 'VARIABLE_SOURCE_ATOM_SINGLE_REFERENCE_RECIPE_CLOSED'
        ctx.r['gates']['frozen_no_fallback_ALL_six_rare_cost_rotation_decisions_and_zero_source_replay']=True
        ctx.finish(dict(cases=cases,views=views,eligibility=eligible,decision=decision,
            candidate_functions=127,control_functions=127,candidate_bank_bytes=sum(c['candidate_bank_bytes'] for c in cases),
            new_candidate_physical_vectors=2*17540,new_continuous_atom_vectors=17540,new_negative_fold_vectors=17540,
            new_exact_selected_WO_product_terms=integer_terms,omitted_zero_BYTE_equal_UIDs=exact,global_hidden_max_preserved_UIDs=preserved_max,
            maximum_structural_identity_bound_ratio=identity_max,source_response_function_calls=0,full_source_WI_projection_rows=0,
            old_native_source_WO_replays=0,model_calls=0,native_calls=0,readout_fits=0,consumed_rows_for_selection=0,
            upstream_original523_main_completed=False,physical_DRAM_verified=False,
            scope='New packed original-source atom functions and matched rotation on already-consumed cached signed inputs; local diagnostic fidelity/logical bytes only, not fresh/whole quality, useful larger n or inference/DRAM speed.'))
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()
