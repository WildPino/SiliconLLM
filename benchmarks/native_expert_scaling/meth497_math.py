"""Exact dyadic field mapping, fixed row-rank certificates, count coverage."""
import hashlib
import math
import numpy as np
P=2147483647
UID=np.dtype([('m','<u4',(13,)),('hash','u1',(128,))])
FEATURE=np.dtype([('phi','<f4',(512,)),('q','<i2',(512,)),('alpha','<f4'),('l','<f4',(768,))])

def alpha_mod(bits):
    b=int(bits);exp=(b>>23)&255;frac=b&0x7fffff
    assert b>>31==0 and exp!=255 and (exp!=0 or frac!=0)
    mantissa=frac if exp==0 else frac+(1<<23);power=-149 if exp==0 else exp-150
    value=mantissa*(pow(2,power,P) if power>=0 else pow((P+1)//2,-power,P))%P
    assert value;return value

def design(q,alpha_bits):
    assert q.dtype==np.dtype('<i2') and q.shape==(len(alpha_bits),512)
    a=np.array([alpha_mod(v) for v in alpha_bits],'<i8');h=np.ones((len(q),513),'<i8')
    h[:,:512]=(q.astype('<i8')*a[:,None])%P;return h

def certificate(h,uids,guard):
    assert h.dtype==np.dtype('<i8') and h.shape==(len(uids),513) and np.all((h>=0)&(h<P))
    digest=hashlib.sha256(h.tobytes()).hexdigest();a=h.copy();original=np.arange(len(a));rows=[];cols=[];pivots=[];det=1;rank=0
    for col in range(513):
        if rank==len(a):break
        at=np.flatnonzero(a[rank:,col]);
        if not len(at):continue
        pivot=rank+int(at[0])
        if pivot!=rank:a[[rank,pivot]]=a[[pivot,rank]];original[[rank,pivot]]=original[[pivot,rank]]
        value=int(a[rank,col]);rows.append(int(original[rank]));cols.append(col);pivots.append(value);det=det*value%P
        a[rank,col:]=(a[rank,col:]*pow(value,-1,P))%P
        if rank+1<len(a):a[rank+1:,col:]=(a[rank+1:,col:]-a[rank+1:,col,None]*a[rank,None,col:])%P
        rank+=1
        if rank%8==0:guard()
    assert not np.any(a[rank:]) and len(set(rows))==len(set(cols))==rank and det!=0
    return {'UID_row_order':[int(v) for v in uids],'field_matrix_I64_SHA256':digest,'rows':len(a),'columns':513,'rank_mod_prime':rank,
        'minor_rows_local_in_pivot_order':rows,'minor_UIDs_in_pivot_order':[int(uids[v]) for v in rows],
        'minor_columns':cols,'raw_pivot_values_mod_prime':pivots,'minor_determinant_mod_prime':det,
        'status':'EMPTY_NO_EXPOSURE' if not len(a) else 'FULL_REAL_ROW_RANK_CERTIFIED' if rank==len(a) else 'FULL_REAL_ROW_RANK_INCONCLUSIVE',
        'real_rank_lower_bound':rank,'real_rank_upper_bound':len(a),'real_rank_exact':rank if rank==len(a) else None,
        'real_nullity_lower_bound':513-len(a),'real_nullity_upper_bound':513-rank,'real_nullity_exact':513-rank if rank==len(a) else None}

def controls():
    assert P%2 and all(P%d for d in range(3,math.isqrt(P)+1,2)) and ((P+1)//2)*2%P==1
    q=np.zeros((3,512),'<i2');q[:,:3]=[[2,-1,0],[0,1,2],[1,0,1]]
    bits=np.array([0x3f000000,0x40000000,0x3f800000],'<u4');h=design(q,bits);c=certificate(h,np.arange(3),lambda:None)
    assert c['minor_columns']==[0,1,512] and c['rank_mod_prime']==3 and c['minor_determinant_mod_prime']==P-(P+1)//2
    duplicate=certificate(np.repeat(h[:1],2,axis=0),np.arange(2),lambda:None);assert duplicate['rank_mod_prime']==1
    fixtures=[0x3f000000,0x40000000,0x3f800000,0x00000001,0x00800000,0x7f7fffff]
    assert (P-1)**2<2**62 and 32768*(P-1)<2**46
    return {'prime':P,'prime_trial_divisor_limit':math.isqrt(P),'inverse2':(P+1)//2,'full_fixture':c,'duplicate_fixture':duplicate,
        'alpha_bits_and_field_values':[[v,alpha_mod(v)] for v in fixtures],'I64_max_field_product':(P-1)**2,'I64_signed_code_product_bound':32768*(P-1)}

def annotate(dev,allrows,valcount):
    df=dev['real_rank_exact'] is not None;af=allrows['real_rank_exact'] is not None
    return {'consumed_dimension_gain_lower_bound':valcount if df and af else max(0,allrows['rank_mod_prime']-dev['rows']) if df else 0,
        'consumed_dimension_gain_upper_bound':valcount,'consumed_dimension_gain_exact':valcount if df and af else None}

def reports(meta,occ,cases):
    dev=(meta[:,4]&5)!=0;counts=np.bincount(meta[dev,3],minlength=128)
    full_dev=np.array([v['development']['real_rank_exact'] is not None for v in cases]);full_all=np.array([v['ALL']['real_rank_exact'] is not None for v in cases])
    def metric(ix):
        e=meta[ix,3]
        return {'count':len(ix),'unique_UID_count':len(np.unique(ix)),'development_origin_count':int(np.sum(dev[ix])),
            'consumed_validation_origin_count':int(np.sum(~dev[ix])),'development_rank_certificate_covered_count':int(np.sum(full_dev[e])),
            'ALL_rank_certificate_covered_count':int(np.sum(full_all[e])),
            'consumed_novelty_certificate_covered_count':int(np.sum((~dev[ix])&full_dev[e]&full_all[e]))}
    roles=[{'split':s,**metric(np.flatnonzero(mask))} for s,mask in (('development',dev),('consumed_validation',~dev))]
    cells=[{'expert':e,'development_exposure':int(counts[e]),'split':s,**metric(np.flatnonzero(mask&(meta[:,3]==e)))} for e in range(128) for s,mask in (('development',dev),('consumed_validation',~dev))]
    rare=[{'development_class':name,'split':s,**metric(np.flatnonzero(mask&category[meta[:,3]]))}
        for name,category in (('0',counts==0),('1..4',(counts>=1)&(counts<=4)),('5..15',(counts>=5)&(counts<=15)),('>=16',counts>=16))
        for s,mask in (('development',dev),('consumed_validation',~dev))]
    views=[{'book':book,'role':book//64,'mode':mode,'accepted':accepted,**metric(occ[(occ[:,4]==book)&(occ[:,6]==mode)&(occ[:,12]==accepted),1])} for book in range(192) for mode in range(2) for accepted in range(2)]
    rm=[{'role':r,'mode':mo,**metric(occ[(occ[:,7]==r)&(occ[:,6]==mo),1])} for r in range(3) for mo in range(2)]
    exposures=[]
    for role in range(3):
        rows=occ[occ[:,7]==role,1];source=np.bincount(meta[rows,3],minlength=128)
        exposures.extend({'role':role,'expert':e,'source':int(source[e])} for e in range(128))
    return {'uid_roles':roles,'cells':cells,'rare':rare,'views':views,'role_mode':rm,'exposures':exposures}

def decision(cases):
    exposed=[v for v in cases if v['ALL']['rows']]
    dev=all(v['development']['real_rank_exact'] is not None for v in exposed);allrows=all(v['ALL']['real_rank_exact'] is not None for v in exposed)
    return 'FINITE_DEVELOPMENT_INTERPOLATION_AND_ALL_CONSUMED_FEATURE_NOVELTY_CERTIFIED' if dev and allrows else 'DEVELOPMENT_INTERPOLATION_CERTIFIED_CONSUMED_NOVELTY_PARTIAL_OR_INCONCLUSIVE' if dev else 'FINITE_FEATURE_ROW_RANK_PARTLY_CERTIFIED_REMAINING_INCONCLUSIVE'
