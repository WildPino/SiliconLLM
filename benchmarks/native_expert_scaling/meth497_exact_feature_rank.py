"""One fixed exact feature-rank certificate; no target/readout fit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import struct
import traceback
from meth497_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);args=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth497_exact_feature_rank',Path(args.out).resolve())
    try:
        b=ctx.admit(args.binding_sha);ctx.binding=b
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1);assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        ctx.r['numerical_imports']=True;import meth497_math as M
        write(ctx.out/'controls.json',M.controls());ctx.r['gates']['NEW497_prime_dyadic_pivot_skip_duplicate_and_I64_bounds_controls']=True
        def read(key,magic,width,reserved,count,dtype):
            with Path(b['data'][key]['path']).open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
                x=np.fromfile(f,dtype,count=count);assert len(x)==count and not f.read(1);return x
        uid=read('uid',b'M493U001',180,0,17540,M.UID);m=uid['m']
        occ=read('occurrences',b'M493O001',68,0,19962,np.dtype(('<u4',(17,))))
        feat=read('features',b'M495FEA1',6148,512,17540,M.FEATURE)
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(17540)) and np.all(m[:,2]==11) and np.all(m[:,3]<128) and np.all(m[:,6]==2)
        assert (int(dev.sum()),int(val.sum()))==(11721,5819) and not np.any(dev&val) and np.all(dev|val)
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,1]<17540) and np.all(occ[:,3]==128) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1)
        for oi,mi in ((14,1),(11,3),(15,12),(16,10)):assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        assert np.array_equal(np.bincount(occ[:,1],minlength=17540),m[:,7])
        ctx.r['gates']['ALL17540_UID_19962_occurrence_original_domains_and_feature_wire_joins']=True
        cases=[];ctx.phase='ALL128_development_and_ALL_exact_field_rank_certificates'
        for e in range(128):
            di=np.flatnonzero(dev&(m[:,3]==e));vi=np.flatnonzero(val&(m[:,3]==e));ai=np.concatenate((di,vi));assert len(di)<=308 and len(ai)<=498
            row={'expert':e,'development_UIDs':len(di),'consumed_validation_UIDs':len(vi)}
            for name,ids in (('development',di),('ALL',ai)):
                h=M.design(feat['q'][ids],feat['alpha'][ids].view('<u4'));row[name]=M.certificate(h,ids,ctx.guard);del h
            row.update(M.annotate(row['development'],row['ALL'],len(vi)));cases.append(row);ctx.guard()
            if e%16==0:ctx.log(experts_certified=e+1);print(json.dumps({'experts_certified':e+1}),flush=True)
        assert sum(v['development_UIDs'] for v in cases)==11721 and sum(v['consumed_validation_UIDs'] for v in cases)==5819
        write(ctx.out/'certificates.json',{'prime':M.P,'cases':cases});ctx.r['gates']['ALL256_original_cases_exact_minor_certificates_and_rank_novelty_bounds']=True
        reports=M.reports(m,occ,cases);source=json.loads((DOC/'meth496_readout_error_result.json').read_bytes())['reports']
        for label in ('uid_roles','cells','rare','views','role_mode'):
            assert len(reports[label])==len(source[label])
            for current,old in zip(reports[label],source[label]):
                assert current['count']==old['count']
                for key in ('split','expert','development_exposure','development_class','book','role','mode','accepted'):
                    if key in current:assert current[key]==old[key]
        assert reports['exposures']==[{k:v[k] for k in ('role','expert','source')} for v in source['exposures']]
        assert sum(len(reports[k]) for k in ('uid_roles','cells','rare','views','role_mode'))==1040
        assert sum(v['count'] for v in reports['views'])==sum(v['count'] for v in reports['role_mode'])==19962
        ctx.r['gates']['ALL1040_original_groups_384_source_exposures_and_certificate_coverage']=True
        summary={'development_full_nonempty':sum(v['development']['rows']>0 and v['development']['real_rank_exact'] is not None for v in cases),
            'ALL_full_nonempty':sum(v['ALL']['rows']>0 and v['ALL']['real_rank_exact'] is not None for v in cases),
            'empty_experts':[v['expert'] for v in cases if not v['ALL']['rows']],
            'development_nullity_lower_sum':sum(v['development']['real_nullity_lower_bound'] for v in cases),
            'development_nullity_upper_sum':sum(v['development']['real_nullity_upper_bound'] for v in cases),
            'ALL_nullity_lower_sum':sum(v['ALL']['real_nullity_lower_bound'] for v in cases),
            'ALL_nullity_upper_sum':sum(v['ALL']['real_nullity_upper_bound'] for v in cases),
            'consumed_dimension_gain_lower_sum':sum(v['consumed_dimension_gain_lower_bound'] for v in cases),
            'consumed_dimension_gain_upper_sum':sum(v['consumed_dimension_gain_upper_bound'] for v in cases)}
        result=ctx.finish({'prime':M.P,'cases':cases,'summary':summary,'reports':reports,'UIDs':17540,'occurrences':19962,'experts':128,'rank_cases':256,
            'decision':M.decision(cases),'source_FFN_calls':0,'model_calls':0,'native_calls':0,'optimizer_updates':0,'readout_solves':0,
            'scope':'Exact finite feature-rank/information certificates only; no fitted candidate, generalization, quality, speed, useful n or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'summary':summary,'decision':result['decision'],'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc);print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True);raise

if __name__=='__main__':main()
