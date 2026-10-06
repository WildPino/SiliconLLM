"""Independent exact rational mapping, determinant certificates and count audit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import traceback
from meth497_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);ap.add_argument('--raw-sha',required=True);args=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth497_retention',Path(args.out).resolve())
    try:
        b=ctx.admit(args.binding_sha);ctx.binding=b;rp=DOC/'meth497_exact_feature_rank_result.json';completed=rp.exists()
        if not completed:rp=rp.with_suffix('.failure.json')
        assert ctx.digest(rp)==args.raw_sha;raw=json.loads(rp.read_bytes())
        eventpath=ROOT/'results/native_expert_scaling/meth497_windows_terminal.json';event=json.loads(eventpath.read_bytes())
        assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events'] and event['instances'][0]['pid']==raw['process_instance']['pid']
        folder=ROOT/'results/native_expert_scaling/meth497_exact_feature_rank'
        inventory=[{'path':str(p),'bytes':p.stat().st_size,'sha256':ctx.digest(p)} for p in sorted(folder.iterdir()) if p.is_file()]
        if not completed:
            assert raw['traceback'] and raw['partial_outputs'];ctx.r['gates']['sole_first_fault_and_ALL_partial_bytes_retained']=True
            result=ctx.finish({'main_completed':False,'raw':{'path':str(rp),'sha256':args.raw_sha},'main_windows_sha256':ctx.digest(eventpath),
                'retained_main_inventory':inventory,'decision':'FIRST_EXACT_RANK_FAULT_RETAINED_NO_CERTIFICATE_ELIGIBILITY',
                'source_FFN_calls':0,'model_calls':0,'native_calls':0,'scope':'First fault retained; no rank or goal promotion.'})
            print(json.dumps({'gates':result['gates'],'resource':ctx.terminal_resources['resource']}));return
        assert all(raw['gates'].values()) and raw['binding_sha256']==args.binding_sha
        lookup={Path(v['path']).name:v for v in inventory}
        for v in raw['output_inventory']:assert lookup[Path(v['path']).name]==v
        terminal=json.loads(Path(raw['terminal_resource_path']).read_bytes());assert terminal['raw_sha256']==args.raw_sha
        ctx.r['gates']['main_complete_and_ALL_current_certificate_output_and_terminal_SHA']=True
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(1);assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info());ctx.r['numerical_imports']=True
        prime=2147483647;assert raw['prime']==prime
        assert all(prime%v for v in range(2,math.isqrt(prime)+1))
        def mapped(bits):
            value=struct.unpack('<f',struct.pack('<I',int(bits)))[0];assert math.isfinite(value) and value>0
            f=Fraction.from_float(value);assert f.denominator&(f.denominator-1)==0 and f.denominator%prime
            answer=f.numerator*pow(f.denominator,-1,prime)%prime;assert answer;return answer
        def determinant(field):
            assert field.shape[0]==field.shape[1];a=field.copy();d=1
            for col in range(len(a)):
                nonzero=np.flatnonzero(a[col:,col])
                if not len(nonzero):return 0
                row=col+int(nonzero[-1])
                if row!=col:a[[col,row]]=a[[row,col]];d=-d%prime
                pivot=int(a[col,col]);d=d*pivot%prime
                if col+1<len(a):
                    factors=(a[col+1:,col]*pow(pivot,-1,prime))%prime
                    a[col+1:,col+1:]=(a[col+1:,col+1:]-factors[:,None]*a[col,None,col+1:])%prime;a[col+1:,col]=0
                if col%8==0:ctx.guard()
            return d
        def rational_det(matrix):
            a=[row[:] for row in matrix];d=Fraction(1)
            for col in range(len(a)):
                pivot=next((i for i in range(col,len(a)) if a[i][col]),None)
                if pivot is None:return Fraction(0)
                if pivot!=col:a[col],a[pivot]=a[pivot],a[col];d=-d
                v=a[col][col];d*=v
                for i in range(col+1,len(a)):
                    factor=a[i][col]/v
                    for j in range(col,len(a)):a[i][j]-=factor*a[col][j]
            return d
        control=json.loads((folder/'controls.json').read_bytes())
        assert control['prime']==prime and control['prime_trial_divisor_limit']==math.isqrt(prime) and control['inverse2']*2%prime==1
        for bits,value in control['alpha_bits_and_field_values']:assert mapped(bits)==value
        assert [v[0] for v in control['alpha_bits_and_field_values']]==[0x3f000000,0x40000000,0x3f800000,0x00000001,0x00800000,0x7f7fffff]
        exact=rational_det([[Fraction(1),Fraction(-1,2),Fraction(1)],[Fraction(0),Fraction(2),Fraction(1)],[Fraction(1),Fraction(0),Fraction(1)]])
        assert exact==Fraction(-1,2);expected=exact.numerator*pow(exact.denominator,-1,prime)%prime
        assert control['full_fixture']['minor_columns']==[0,1,512] and control['full_fixture']['rank_mod_prime']==3 and control['full_fixture']['minor_determinant_mod_prime']==expected
        assert control['duplicate_fixture']['rank_mod_prime']==1 and control['duplicate_fixture']['status']=='FULL_REAL_ROW_RANK_INCONCLUSIVE'
        assert control['I64_max_field_product']==(prime-1)**2<2**62 and control['I64_signed_code_product_bound']==32768*(prime-1)<2**46
        write(ctx.out/'independent_controls.json',{'prime':prime,'trial_all_divisor_limit':math.isqrt(prime),'minor_exact_rational':str(exact),'minor_mod_prime':expected,'alpha_fraction_maps':control['alpha_bits_and_field_values']})
        ctx.r['gates']['NEW497_independent_trial_prime_Fraction_mapping_and_rational_minor_controls']=True
        def read(key,magic,width,reserved,count,dtype):
            with Path(b['data'][key]['path']).open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
                a=np.fromfile(f,dtype,count=count);assert len(a)==count and not f.read(1);return a
        uid=read('uid',b'M493U001',180,0,17540,np.dtype([('m','<u4',(13,)),('hash','u1',(128,))]));m=uid['m']
        occ=read('occurrences',b'M493O001',68,0,19962,np.dtype(('<u4',(17,))))
        feat=read('features',b'M495FEA1',6148,512,17540,np.dtype([('phi','<f4',(512,)),('q','<i2',(512,)),('alpha','<f4'),('l','<f4',(768,))]))
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(17540)) and np.all(m[:,2]==11) and np.all(m[:,3]<128) and np.all(m[:,6]==2)
        assert (int(dev.sum()),int(val.sum()))==(11721,5819) and not np.any(dev&val) and np.all(dev|val)
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,1]<17540) and np.all(occ[:,3]==128) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1)
        for oi,mi in ((14,1),(11,3),(15,12),(16,10)):assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        assert np.array_equal(np.bincount(occ[:,1],minlength=17540),m[:,7])
        certificates=json.loads((folder/'certificates.json').read_bytes());assert certificates['prime']==prime and certificates['cases']==raw['cases'] and len(raw['cases'])==128
        ctx.phase='ALL256_independent_dyadic_maps_and_nonzero_minor_determinants'
        for e,case in enumerate(raw['cases']):
            di=np.flatnonzero(dev&(m[:,3]==e));vi=np.flatnonzero(val&(m[:,3]==e));ai=np.concatenate((di,vi))
            assert case['expert']==e and case['development_UIDs']==len(di)<=308 and case['consumed_validation_UIDs']==len(vi) and len(ai)<=498
            for name,ids in (('development',di),('ALL',ai)):
                c=case[name];assert c['UID_row_order']==ids.tolist() and c['rows']==len(ids) and c['columns']==513
                h=np.ones((len(ids),513),'<i8');bits=feat['alpha'][ids].view('<u4')
                for j,(uid_id,alpha_bits) in enumerate(zip(ids,bits)):h[j,:512]=(feat['q'][uid_id].astype('<i8')*mapped(alpha_bits))%prime
                assert hashlib.sha256(h.tobytes()).hexdigest()==c['field_matrix_I64_SHA256']
                r=c['rank_mod_prime'];rows=c['minor_rows_local_in_pivot_order'];cols=c['minor_columns']
                assert 0<=r<=len(ids) and len(rows)==len(cols)==len(c['raw_pivot_values_mod_prime'])==r and len(set(rows))==len(set(cols))==r
                assert all(0<=j<len(ids) for j in rows) and cols==sorted(cols) and all(0<=j<513 for j in cols)
                assert c['minor_UIDs_in_pivot_order']==[int(ids[j]) for j in rows]
                d=determinant(h[rows][:,cols]);assert d!=0 and d==c['minor_determinant_mod_prime']
                product=1
                for v in c['raw_pivot_values_mod_prime']:assert 0<v<prime;product=product*v%prime
                assert product==d
                if r<len(ids):
                    a=h[:,::-1].copy();rank=0
                    for col in range(513):
                        if rank==len(a):break
                        candidates=np.flatnonzero(a[rank:,col])
                        if not len(candidates):continue
                        pivot=rank+int(candidates[-1]);a[[rank,pivot]]=a[[pivot,rank]];v=int(a[rank,col])
                        a[rank,col:]=(a[rank,col:]*pow(v,-1,prime))%prime
                        if rank+1<len(a):a[rank+1:,col:]=(a[rank+1:,col:]-a[rank+1:,col,None]*a[rank,None,col:])%prime
                        rank+=1
                        if rank%8==0:ctx.guard()
                    assert rank==r and not np.any(a[rank:])
                assert c['status']==('EMPTY_NO_EXPOSURE' if not len(ids) else 'FULL_REAL_ROW_RANK_CERTIFIED' if r==len(ids) else 'FULL_REAL_ROW_RANK_INCONCLUSIVE')
                assert c['real_rank_lower_bound']==r and c['real_rank_upper_bound']==len(ids) and c['real_rank_exact']==(r if r==len(ids) else None)
                assert c['real_nullity_lower_bound']==513-len(ids) and c['real_nullity_upper_bound']==513-r and c['real_nullity_exact']==(513-r if r==len(ids) else None)
                ctx.guard()
            df=case['development']['real_rank_exact'] is not None;af=case['ALL']['real_rank_exact'] is not None
            lower=len(vi) if df and af else max(0,case['ALL']['rank_mod_prime']-len(di)) if df else 0
            assert case['consumed_dimension_gain_lower_bound']==lower and case['consumed_dimension_gain_upper_bound']==len(vi) and case['consumed_dimension_gain_exact']==(len(vi) if df and af else None)
            if e%16==0:ctx.log(experts_audited=e+1);print(json.dumps({'experts_audited':e+1}),flush=True)
        ctx.r['gates']['ALL256_exact_dyadic_minor_certificates_real_rank_nullity_and_consumed_gain']=True
        def metric(ix):
            ds=alls=novel=0
            for uid_id in ix:
                case=raw['cases'][int(m[uid_id,3])];df=case['development']['real_rank_exact'] is not None;af=case['ALL']['real_rank_exact'] is not None
                ds+=int(df);alls+=int(af);novel+=int(bool(val[uid_id]) and df and af)
            return {'count':len(ix),'unique_UID_count':len(set(int(v) for v in ix)),'development_origin_count':sum(bool(dev[v]) for v in ix),
                'consumed_validation_origin_count':sum(bool(val[v]) for v in ix),'development_rank_certificate_covered_count':ds,'ALL_rank_certificate_covered_count':alls,'consumed_novelty_certificate_covered_count':novel}
        counts=np.bincount(m[dev,3],minlength=128);reports={k:[] for k in ('uid_roles','cells','rare','views','role_mode','exposures')}
        for split,mask in (('development',dev),('consumed_validation',val)):reports['uid_roles'].append({'split':split,**metric(np.flatnonzero(mask))})
        for e in range(128):
            for split,mask in (('development',dev),('consumed_validation',val)):reports['cells'].append({'expert':e,'development_exposure':int(counts[e]),'split':split,**metric(np.flatnonzero(mask&(m[:,3]==e)))})
        for name,lo,hi in (('0',0,0),('1..4',1,4),('5..15',5,15),('>=16',16,17540)):
            category=(counts>=lo)&(counts<=hi)
            for split,mask in (('development',dev),('consumed_validation',val)):reports['rare'].append({'development_class':name,'split':split,**metric(np.flatnonzero(mask&category[m[:,3]]))})
        for book in range(192):
            for mode in range(2):
                for accepted in range(2):reports['views'].append({'book':book,'role':book//64,'mode':mode,'accepted':accepted,**metric(occ[(occ[:,4]==book)&(occ[:,6]==mode)&(occ[:,12]==accepted),1])})
        for role in range(3):
            for mode in range(2):reports['role_mode'].append({'role':role,'mode':mode,**metric(occ[(occ[:,7]==role)&(occ[:,6]==mode),1])})
            ix=occ[occ[:,7]==role,1];source=np.bincount(m[ix,3],minlength=128)
            reports['exposures'].extend({'role':role,'expert':e,'source':int(source[e])} for e in range(128))
        assert reports==raw['reports']
        summary={'development_full_nonempty':sum(v['development']['rows']>0 and v['development']['real_rank_exact'] is not None for v in raw['cases']),
            'ALL_full_nonempty':sum(v['ALL']['rows']>0 and v['ALL']['real_rank_exact'] is not None for v in raw['cases']),
            'empty_experts':[v['expert'] for v in raw['cases'] if not v['ALL']['rows']]}
        for name in ('development','ALL'):
            for bound in ('lower','upper'):summary[name+'_nullity_'+bound+'_sum']=sum(v[name]['real_nullity_'+bound+'_bound'] for v in raw['cases'])
        for bound in ('lower','upper'):summary['consumed_dimension_gain_'+bound+'_sum']=sum(v['consumed_dimension_gain_'+bound+'_bound'] for v in raw['cases'])
        assert summary==raw['summary']
        supported=[v for v in raw['cases'] if v['ALL']['rows']];df=all(v['development']['real_rank_exact'] is not None for v in supported);af=all(v['ALL']['real_rank_exact'] is not None for v in supported)
        decision='FINITE_DEVELOPMENT_INTERPOLATION_AND_ALL_CONSUMED_FEATURE_NOVELTY_CERTIFIED' if df and af else 'DEVELOPMENT_INTERPOLATION_CERTIFIED_CONSUMED_NOVELTY_PARTIAL_OR_INCONCLUSIVE' if df else 'FINITE_FEATURE_ROW_RANK_PARTLY_CERTIFIED_REMAINING_INCONCLUSIVE'
        assert decision==raw['decision']
        for key in ('source_FFN_calls','model_calls','native_calls','optimizer_updates','readout_solves'):assert raw[key]==0
        ctx.r['gates']['ALL1040_original_reports_384_source_exposures_summary_and_fixed_decision']=True
        result=ctx.finish({'main_completed':True,'raw':{'path':str(rp),'sha256':args.raw_sha},'main_windows_sha256':ctx.digest(eventpath),'retained_main_inventory':inventory,
            'UIDs_audited':17540,'occurrences_audited':19962,'experts_audited':128,'rank_cases_audited':256,'metric_groups_audited':1040,'exposure_groups_audited':384,
            'prime':prime,'summary':summary,'decision':decision,'source_FFN_calls':0,'model_calls':0,'native_calls':0,
            'scope':'Independent exact finite rank/information certificates only; no candidate, generalization, quality, speed, useful n or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'summary':summary,'decision':decision,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as exc:
        ctx.fail(exc);print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True);raise

if __name__=='__main__':main()
