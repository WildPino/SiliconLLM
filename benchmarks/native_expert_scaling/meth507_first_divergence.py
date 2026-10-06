"""Retained common histories, exact pairwise rank flips and state/route geometry."""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import struct
import meth507_operations as O

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binding-sha',required=True)
    args=parser.parse_args();ctx=O.Context('main')
    try:
        binding=ctx.admit(args.binding_sha)
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        import numpy as np
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
        ctx.modules()
        raw=json.loads(Path(binding['raw506']['path']).read_bytes())

        def wire(path):
            data=Path(path).read_bytes();assert data[:8]==b'SWR32O01'
            s,t,d,e,f,v,n=struct.unpack_from('<7I',data,8)
            assert (s,d,e,f,v,n)==(29,768,12,12,32128,6*(29+t)) and 0<t<=64
            enc_count=14*29*768;dec_count=t*14*768;log_count=t*32128
            stop=36+4*(enc_count+dec_count+log_count);assert len(data)==stop+12*n
            numbers=np.frombuffer(data,dtype='<f4',count=enc_count+dec_count+log_count,offset=36)
            assert np.isfinite(numbers).all()
            routes=np.array([struct.unpack_from('<iif',data,stop+12*j) for j in range(n)],dtype='f8')
            return (numbers[:enc_count].reshape(14,29,768),
                    numbers[enc_count:enc_count+dec_count].reshape(t,14,768),
                    numbers[enc_count+dec_count:].reshape(t,32128),routes)

        def geometry(candidate, donor):
            assert candidate.shape==donor.shape and candidate.shape[-1]==768
            c=candidate.astype('f8');d=donor.astype('f8');difference=c-d
            ds=np.sum(d*d,axis=-1);cs=np.sum(c*c,axis=-1);es=np.sum(difference*difference,axis=-1)
            assert (ds>0).all() and (cs>0).all()
            cosine=np.sum(c*d,axis=-1)/np.sqrt(cs*ds)
            return {'donor_l2':np.sqrt(ds).tolist(),'candidate_l2':np.sqrt(cs).tolist(),
                    'error_l2':np.sqrt(es).tolist(),'relative_l2':np.sqrt(es/ds).tolist(),
                    'cosine':cosine.tolist(),'max_abs_error':np.max(np.abs(difference),axis=-1).tolist()}

        def routes(candidate, donor):
            assert candidate.shape==donor.shape and candidate.shape[-1]==3
            for a in (candidate,donor):
                assert np.isfinite(a).all() and np.array_equal(a[...,:2],np.floor(a[...,:2]))
                assert ((a[...,0]>=0)&(a[...,0]<128)).all() and np.isin(a[...,1],[0,1]).all()
                assert ((a[...,2]>0)&(a[...,2]<=1)).all()
            return {'candidate':candidate.tolist(),'donor':donor.tolist(),
                    'choice_changes':int(np.count_nonzero(candidate[...,0]!=donor[...,0])),
                    'acceptance_changes':int(np.count_nonzero(candidate[...,1]!=donor[...,1])),
                    'selected_mass_max_abs_error':float(np.max(np.abs(candidate[...,2]-donor[...,2]))),
                    'selected_mass_relative_l2':float(np.sqrt(np.sum((candidate[...,2]-donor[...,2])**2)/np.sum(donor[...,2]**2)))}

        def exact(value):return Fraction.from_float(float(value))
        def rational(value):return {'numerator':value.numerator,'denominator':value.denominator,'float':float(value)}
        cases=[]
        for ordinal,case in enumerate(raw['cases']):
            row=binding['selected'][ordinal];assert (case['book'],case['index'])==(row['book'],row['index'])==divmod(ordinal,4)
            assert row['wire']==case['native']['candidate']['generation'][0]['wire'] and row['donor']==case['donor_reference']
            ce,cd,cl,cr=wire(row['wire']['path'])
            with np.load(row['donor']['path'],allow_pickle=False) as ref:
                de=ref['generation_encoder'];dd=ref['generation_decoder'];dl=ref['generation_logits'];dr=ref['generation_routes'];official=ref['official_generation_logits']
                assert de.dtype==dd.dtype==dl.dtype==np.dtype('f4') and dr.dtype==np.dtype('f8')
                assert de.shape==(14,29,768) and dd.shape==(len(dl),14,768) and dl.shape==(len(dd),32128) and dr.shape==(6*(29+len(dl)),3)
                for a in (de,dd,dl,dr,official):assert np.isfinite(a).all()
                assert np.array_equal(official.argmax(1),dl.argmax(1)) and np.array_equal(de,ref['teacher_encoder'])
                bridge=float(np.sqrt(np.sum((official.astype('f8')-dl.astype('f8'))**2)/np.sum(dl.astype('f8')**2)))
                assert bridge<=1e-6 and abs(bridge-case['official_generation_relative_l2'])<=1e-10
                ci=cl.argmax(1).tolist();di=dl.argmax(1).tolist()
                assert ci==case['candidate_task']['generated_ids'] and di==case['donor_task']['generated_ids']
                for ids in (ci,di):assert ids[-1] in (1,32095) or len(ids)==64
                prefix=0
                while prefix<min(len(ci),len(di)) and ci[prefix]==di[prefix]:prefix+=1
                divergent=prefix<min(len(ci),len(di))
                if not divergent:assert ci==di,'same_prefix_but_incompatible_stopping_rules'
                comparable=prefix+int(divergent)
                assert [0]+ci[:comparable-1]==[0]+di[:comparable-1]
                detail={'book':case['book'],'index':case['index'],'source_id':case['source_id'],
                        'wire_sha256':row['wire']['sha256'],'donor_npz_sha256':row['donor']['sha256'],
                        'candidate_ids':ci,'donor_ids':di,'common_prefix_length':prefix,
                        'common_decoder_input_ids':[0]+ci[:comparable-1],
                        'comparable_steps':comparable,'divergent':divergent,'normalized_prose_edit':case['normalized_prose_edit'],
                        'encoder_geometry':geometry(ce,de),'decoder_geometry':geometry(cd[:comparable],dd[:comparable]),
                        'encoder_routes':routes(cr[:174].reshape(6,29,3),dr[:174].reshape(6,29,3)),
                        'decoder_routes':routes(cr[174:174+6*comparable].reshape(comparable,6,3),dr[174:174+6*comparable].reshape(comparable,6,3))}
                flips=[]
                for step in range(comparable):
                    c,d=ci[step],di[step];cx,dx=cl[step],dl[step]
                    record={'step':step,'candidate_id':c,'donor_id':d,
                            'candidate_maximum_ties':int(np.count_nonzero(cx==cx[c])),
                            'donor_maximum_ties':int(np.count_nonzero(dx==dx[d]))}
                    if c!=d:
                        assert step==prefix and divergent
                        dm=exact(dx[d])-exact(dx[c]);pert=(exact(cx[c])-exact(dx[c]))-(exact(cx[d])-exact(dx[d]));cm=exact(cx[c])-exact(cx[d])
                        assert dm>=0 and cm>=0 and pert>=dm and pert-dm==cm
                        if cm==0:assert c<d
                        runnerup=float(np.max(np.concatenate((dx[:d],dx[d+1:]))))
                        record.update(donor_logits_pair=[float(dx[d]),float(dx[c])],candidate_logits_pair=[float(cx[d]),float(cx[c])],
                                      donor_margin=rational(dm),pair_perturbation=rational(pert),candidate_margin=rational(cm),
                                      donor_runnerup_margin=rational(exact(dx[d])-exact(runnerup)))
                    flips.append(record)
                detail['aligned_choices']=flips
                cases.append(detail)
            ctx.guard()
            if ordinal%4==3:ctx.log(book=case['book'],cases=ordinal+1)
        divergent=[c for c in cases if c['divergent']]
        hist={}
        for c in divergent:hist[str(c['common_prefix_length'])]=hist.get(str(c['common_prefix_length']),0)+1
        summary={'cases':96,'books':24,'divergent_cases':len(divergent),'identical_generated_trajectories':96-len(divergent),
                 'aligned_steps_including_first_divergence':sum(c['comparable_steps'] for c in cases),
                 'first_differing_step_histogram':hist,
                 'all_encoder_route_choice_changes':sum(c['encoder_routes']['choice_changes'] for c in cases),
                 'all_encoder_acceptance_changes':sum(c['encoder_routes']['acceptance_changes'] for c in cases),
                 'aligned_decoder_route_choice_changes':sum(c['decoder_routes']['choice_changes'] for c in cases),
                 'aligned_decoder_acceptance_changes':sum(c['decoder_routes']['acceptance_changes'] for c in cases),
                 'divergent_cases_without_any_prior_or_current_route_choice_change':sum(c['encoder_routes']['choice_changes']==0 and c['decoder_routes']['choice_changes']==0 for c in divergent),
                 'first_flip_donor_margin_min_median_max':None if not divergent else [float(f(v)) for f in (np.min,np.median,np.max) for v in [[c['aligned_choices'][-1]['donor_margin']['float'] for c in divergent]]],
                 'scope':'Consumed retained states; no causal attribution, new source calls, fresh quality, timing, DRAM or goal promotion.'}
        ctx.r['gates']['all96_prefixes_only_equal_histories_and_lowest_ID_choices']=True
        ctx.r['gates']['all_first_rank_flip_identities_exact_binary_rationals']=True
        ctx.r['gates']['all_recorded_states_and_actual_choice_acceptance_selected_mass_aligned']=True
        ctx.finish({'cases':cases,'summary':summary,'raw506_sha256':binding['raw506']['sha256'],'goal_complete':False})
    except BaseException:ctx.fail();raise
    finally:ctx.close()

if __name__=='__main__':main()
