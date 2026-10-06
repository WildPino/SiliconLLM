"""Independent retained-data rederivation; does not import the primal science."""
import argparse
import fractions
import json
import os
from pathlib import Path
import struct
import meth507_operations as O

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binding-sha',required=True);parser.add_argument('--raw-sha',required=True)
    args=parser.parse_args();ctx=O.Context('audit')
    try:
        b=ctx.admit(args.binding_sha);mainpath=O.DOC/'meth507_main_result.json'
        assert ctx.digest(mainpath)==args.raw_sha
        primal=json.loads(mainpath.read_bytes());source=json.loads(Path(b['raw506']['path']).read_bytes())
        assert primal['binding_sha256']==args.binding_sha and primal['raw506_sha256']==b['raw506']['sha256']
        assert len(primal['cases'])==len(source['cases'])==len(b['selected'])==96
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        import numpy as np
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.modules()

        def equal(a,e):
            if isinstance(e,dict):
                assert set(a)==set(e)
                for k in e:equal(a[k],e[k])
            elif isinstance(e,list):
                assert len(a)==len(e)
                for x,y in zip(a,e):equal(x,y)
            elif isinstance(e,float):assert abs(a-e)<=2e-12*max(1,abs(e)),(a,e)
            else:assert a==e,(a,e)

        def read_native(filename):
            with Path(filename).open('rb') as f:
                assert f.read(8)==b'SWR32O01'
                s,t,d,enc,dec,v,n=struct.unpack('<7I',f.read(28))
                assert (s,d,enc,dec,v,n)==(29,768,12,12,32128,174+6*t) and 1<=t<=64
                arrays=[]
                for shape in ((14,29,768),(t,14,768),(t,32128)):
                    count=int(np.prod(shape));a=np.frombuffer(f.read(4*count),dtype='<f4').reshape(shape)
                    assert np.isfinite(a).all();arrays.append(a)
                route=np.frombuffer(f.read(12*n),dtype=[('i','<i4'),('a','<i4'),('p','<f4')])
                assert not f.read(1)
                r=np.column_stack((route['i'],route['a'],route['p'])).astype('f8')
                return *arrays,r

        def stats(a,d):
            dimensions=a.shape[:-1];aa=a.reshape(-1,768).astype('f8');dd=d.reshape(-1,768).astype('f8')
            result={k:[] for k in ('donor_l2','candidate_l2','error_l2','relative_l2','cosine','max_abs_error')}
            for x,y in zip(aa,dd):
                yn=float(np.sqrt(np.add.reduce(y*y)));xn=float(np.sqrt(np.add.reduce(x*x)));z=x-y
                en=float(np.sqrt(np.add.reduce(z*z)));assert yn>0 and xn>0
                values=(yn,xn,en,en/yn,float(np.add.reduce(x*y)/(xn*yn)),float(np.max(np.abs(z))))
                for k,v in zip(result,values):result[k].append(v)
            return {k:np.array(v).reshape(dimensions).tolist() for k,v in result.items()}

        def route_stats(a,d):
            for x in (a,d):
                assert np.isfinite(x).all() and np.array_equal(x[...,:2],x[...,:2].astype('i4'))
                assert np.isin(x[...,1],(0,1)).all() and ((x[...,0]>=0)&(x[...,0]<128)).all()
                assert ((x[...,2]>0)&(x[...,2]<=1)).all()
            p=a[...,2].flatten();q=d[...,2].flatten();dif=p-q
            return {'candidate':a.tolist(),'donor':d.tolist(),
                    'choice_changes':sum(int(x!=y) for x,y in zip(a[...,0].flatten(),d[...,0].flatten())),
                    'acceptance_changes':sum(int(x!=y) for x,y in zip(a[...,1].flatten(),d[...,1].flatten())),
                    'selected_mass_max_abs_error':float(max(abs(dif))),
                    'selected_mass_relative_l2':float(np.sqrt(np.add.reduce(dif*dif)/np.add.reduce(q*q)))}

        def q(x):
            n,d=float(x).as_integer_ratio();return fractions.Fraction(n,d)
        def recorded(x):return {'numerator':x.numerator,'denominator':x.denominator,'float':float(x)}
        mismatches=aligned=encoder_changes=decoder_changes=encoder_accept=decoder_accept=no_changed_routes=0
        histogram={};margins=[];books=[]
        for ordinal,(row,saved,case) in enumerate(zip(b['selected'],primal['cases'],source['cases'])):
            bi,ci=divmod(ordinal,4);assert (row['book'],row['index'])==(saved['book'],saved['index'])==(case['book'],case['index'])==(bi,ci)
            assert row['wire']==case['native']['candidate']['generation'][0]['wire'] and row['donor']==case['donor_reference']
            assert saved['source_id']==case['source_id'] and saved['wire_sha256']==row['wire']['sha256'] and saved['donor_npz_sha256']==row['donor']['sha256']
            equal(saved['normalized_prose_edit'],case['normalized_prose_edit'])
            ce,cd,cl,cr=read_native(row['wire']['path'])
            with np.load(row['donor']['path'],allow_pickle=False) as ref:
                de=ref['generation_encoder'];dd=ref['generation_decoder'];dl=ref['generation_logits'];dr=ref['generation_routes'];official=ref['official_generation_logits']
                assert de.shape==(14,29,768) and dd.shape==(len(dl),14,768) and dl.shape==(len(dd),32128) and dr.shape==(174+6*len(dd),3)
                assert de.dtype==dd.dtype==dl.dtype==np.dtype('f4') and dr.dtype==np.dtype('f8')
                for a in (de,dd,dl,dr,official):assert np.isfinite(a).all()
                assert np.array_equal(ref['teacher_encoder'],de) and np.array_equal(dl.argmax(1),official.argmax(1))
                diff=official.astype('f8')-dl.astype('f8');bridge=float(np.sqrt(np.add.reduce((diff*diff).flatten())/np.add.reduce((dl.astype('f8')**2).flatten())))
                assert bridge<=1e-6;equal(bridge,case['official_generation_relative_l2'])
                cids=cl.argmax(1).tolist();dids=dl.argmax(1).tolist()
                assert cids==saved['candidate_ids']==case['candidate_task']['generated_ids'] and dids==saved['donor_ids']==case['donor_task']['generated_ids']
                for ids in (cids,dids):assert ids[-1] in (1,32095) or len(ids)==64
                differences=[j for j,(x,y) in enumerate(zip(cids,dids)) if x!=y]
                isdifferent=bool(differences);prefix=differences[0] if isdifferent else len(dids)
                if not isdifferent:assert cids==dids
                count=prefix+int(isdifferent)
                assert saved['common_prefix_length']==prefix and saved['comparable_steps']==count and saved['divergent']==isdifferent
                assert saved['common_decoder_input_ids']==[0]+cids[:count-1]==[0]+dids[:count-1]
                equal(saved['encoder_geometry'],stats(ce,de));equal(saved['decoder_geometry'],stats(cd[:count],dd[:count]))
                er=route_stats(cr[:174].reshape(6,29,3),dr[:174].reshape(6,29,3));rr=route_stats(cr[174:174+6*count].reshape(count,6,3),dr[174:174+6*count].reshape(count,6,3))
                equal(saved['encoder_routes'],er);equal(saved['decoder_routes'],rr)
                assert len(saved['aligned_choices'])==count
                for j,r in enumerate(saved['aligned_choices']):
                    c,d=cids[j],dids[j];cx,dx=cl[j],dl[j]
                    expected={'step':j,'candidate_id':c,'donor_id':d,
                              'candidate_maximum_ties':int((cx==cx[c]).sum()),'donor_maximum_ties':int((dx==dx[d]).sum())}
                    assert c==int(np.flatnonzero(cx==cx[c])[0]) and d==int(np.flatnonzero(dx==dx[d])[0])
                    if c!=d:
                        assert j==prefix
                        m=q(dx[d])-q(dx[c]);cm=q(cx[c])-q(cx[d]);pert=(q(cx[c])-q(dx[c]))-(q(cx[d])-q(dx[d]))
                        assert m>=0 and cm>=0 and pert-m==cm and pert>=m
                        if cm==0:assert c<d
                        rank=dx.astype('f8').copy();rank[d]=-np.inf;runner=rank.max()
                        expected.update(donor_logits_pair=[float(dx[d]),float(dx[c])],candidate_logits_pair=[float(cx[d]),float(cx[c])],
                                        donor_margin=recorded(m),pair_perturbation=recorded(pert),candidate_margin=recorded(cm),
                                        donor_runnerup_margin=recorded(q(dx[d])-q(runner)))
                        margins.append(float(m))
                    equal(r,expected)
                mismatches+=int(isdifferent);aligned+=count
                encoder_changes+=er['choice_changes'];decoder_changes+=rr['choice_changes'];encoder_accept+=er['acceptance_changes'];decoder_accept+=rr['acceptance_changes']
                if isdifferent:
                    histogram[str(prefix)]=histogram.get(str(prefix),0)+1
                    no_changed_routes+=int(er['choice_changes']==rr['choice_changes']==0)
            ctx.guard()
            if ordinal%4==3:books.append(bi);ctx.log(audited_book=bi)
        expected={'cases':96,'books':24,'divergent_cases':mismatches,'identical_generated_trajectories':96-mismatches,
                  'aligned_steps_including_first_divergence':aligned,'first_differing_step_histogram':histogram,
                  'all_encoder_route_choice_changes':encoder_changes,'all_encoder_acceptance_changes':encoder_accept,
                  'aligned_decoder_route_choice_changes':decoder_changes,'aligned_decoder_acceptance_changes':decoder_accept,
                  'divergent_cases_without_any_prior_or_current_route_choice_change':no_changed_routes,
                  'first_flip_donor_margin_min_median_max':None if not margins else [float(np.min(margins)),float(np.median(margins)),float(np.max(margins))],
                  'scope':'Consumed retained states; no causal attribution, new source calls, fresh quality, timing, DRAM or goal promotion.'}
        equal(primal['summary'],expected)
        ctx.r['gates']['independent_ALL96_native_NPZ_format_IDs_official_bridges_and_join']=True
        ctx.r['gates']['independent_ALL96_equal_histories_and_lowest_ID_argmax']=True
        ctx.r['gates']['independent_ALL_first_flips_exact_rational_pair_identity']=True
        ctx.r['gates']['independent_ALL_encoder_aligned_decoder_vectors_and_choice_acceptance_mass']=True
        ctx.r['gates']['independent_ALL24_books_and_terminal_summary']=True
        ctx.finish({'main_sha256':args.raw_sha,'summary':expected,'audited_books':books,'goal_complete':False})
    except BaseException:ctx.fail();raise
    finally:ctx.close()

if __name__=='__main__':main()
