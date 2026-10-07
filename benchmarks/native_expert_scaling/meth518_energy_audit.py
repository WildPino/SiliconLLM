"""Independent exact cross-product ordering and all ideal capacity boundaries."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
from fractions import Fraction
from functools import cmp_to_key
import json
from math import isqrt
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth518_operations import Context,DOC,ROOT,wire
N,H=17540,3072


def tiny_controls():
    tested=0
    for b in [((4,0),(0,4)),((4,1),(0,3)),((3,-1),(1,3)),((3,1),(1,4))]:
        g=[[sum(b[k][i]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        r=max(sum(abs(16*int(i==j)-g[i][j]) for j in range(2)) for i in range(2));delta=16-r
        assert delta>0
        for wx in range(-2,3):
            for wy in range(-2,3):
                for qx in range(-2,3):
                    for qy in range(-2,3):
                        w,q=(wx,wy),(qx,qy);W=sum(t*t for t in w);Q=sum(t*t for t in q)
                        u=[sum(w[k]*b[k][j] for k in range(2)) for j in range(2)]
                        v=[sum(q[k]*b[k][j] for k in range(2)) for j in range(2)]
                        pw,pq=sum(x*x for x in u),sum(x*x for x in v)
                        RW,RQ=256*W-delta*pw,256*Q-delta*pq
                        assert RW>=0 and RQ>=0
                        ceil=lambda n:0 if not n else 1+isqrt(n-1)
                        endpoint=sum(x*y for x,y in zip(u,v))*16+r*ceil(pw*pq)+ceil(RW*RQ)
                        if W and Q:
                            alpha,beta=Fraction(pw,16*W),Fraction(pq,16*Q)
                            necessary=Fraction(delta,16)*(alpha+beta)>=1
                            assert necessary==(delta*W*pq>=RW*Q)
                            if endpoint<=0:assert necessary
                        elif endpoint<=0:assert sum(x*y for x,y in zip(w,q))==0
                        tested+=1
    assert tested==2500
    return tested


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--main-sha',required=True)
    args=ap.parse_args();ctx=None
    try:
        ctx=Context('audit',args.binding_sha)
        path=DOC/'meth518_main_result.json';assert ctx.digest(path)==args.main_sha
        raw=json.loads(path.read_bytes());assert raw['binding_sha256']==args.binding_sha and all(raw['gates'].values())
        for v in raw['output_inventory']:assert ctx.digest(v['path'])==v['sha256']
        np=ctx.numpy();tested=tiny_controls()
        m=wire(np,ctx.data('uid'),b'M493U001',180,0,[('m','<u4',(13,)),('hash','u1',(128,))],N)['m']
        occ=wire(np,ctx.data('occurrences'),b'M493O001',68,0,np.dtype(('<u4',(17,))),19962)
        assert np.array_equal(m[:,0],np.arange(N)) and np.all(m[:,2]==11) and np.all(m[:,3]<128)
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1)
        assert np.array_equal(np.bincount(occ[:,1],minlength=N),m[:,7])
        for oi,mi in [(14,1),(11,3),(15,12),(16,10)]:assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        dev,val=(m[:,4]&5)!=0,(m[:,4]&2)!=0;assert np.all(dev^val) and (int(dev.sum()),int(val.sum()))==(11721,5819)
        source=Path(ctx.data('prefix_index.bin')).read_bytes()
        head=struct.unpack_from('<8s6IQ',source);assert head[:7]==(b'M517P001',128,768,3072,32,256,0)
        r=head[-1];assert 0<=r<65536;delta=65536-r;cursor=57384
        prior=np.load(ctx.data('work.npy'),mmap_mode='r',allow_pickle=False)
        assert prior.shape==(N,) and np.all(prior['width']==3072) and np.all(prior['skipped']==0) and np.all(prior['candidates']==0)
        folder=ROOT/'results/native_expert_scaling/meth518_index'
        threshold=(folder/'thresholds.bin').read_bytes()
        assert struct.unpack_from('<8s6IQ',threshold)==(b'M518E001',128,768,3072,32,256,0,r)
        result=np.load(folder/'possible_capacity.npy',mmap_mode='r',allow_pickle=False)
        dtype=np.dtype([('possible','<u2'),('optimistic_width','<u2'),('minimum_addressed_bytes','<u4')])
        assert result.shape==(N,) and result.dtype==dtype
        rebuilt=np.empty(N,dtype=dtype);tcur=40;parents=[];qzero=0;crosses=0
        ctx.r['gates']['independent2500_rank2_necessity_controls_and_original_roles']=True
        for e in range(128):
            cursor+=H*32*4
            pw=np.frombuffer(source,dtype='<u8',count=H,offset=cursor);cursor+=H*8
            rw=np.frombuffer(source,dtype='<u8',count=H,offset=cursor);cursor+=H*8+H*4
            rows=[];originalnorm=[]
            for j in range(H):
                n,p=int(rw[j]),int(pw[j]);W,rem=divmod(n+delta*p,256**4)
                assert rem==0 and W<=768*128**2 and p<=131072*W and n<=W*256**4
                if W==0:assert n==p==0
                rows.append((j,int(W==0),W,n,delta*W));originalnorm.append(W)
            if e:assert np.array(originalnorm,dtype='<u4').tobytes()==np.load(ctx.data(f'row_norm2_e{e:03}.npy'),allow_pickle=False).tobytes()
            def compare(a,b):
                if a[1]!=b[1]:return -1 if a[1] else 1
                if not a[1]:
                    left,right=a[3]*b[4],b[3]*a[4]
                    assert max(left,right)<2**128
                    if left!=right:return -1 if left<right else 1
                return (a[0]>b[0])-(a[0]<b[0])
            ordered=sorted(rows,key=cmp_to_key(compare))
            for row in ordered:
                assert struct.unpack_from('<HHIQQ',threshold,tcur)==row;tcur+=24
            at=np.flatnonzero(m[:,3]==e);maximum=total=0
            for uid in at:
                Q,pq=int(prior['query_norm2'][uid]),int(prior['projected_query_norm2'][uid])
                assert Q<=768*32767**2 and pq<=131072*Q
                if Q==0:possible=H;qzero+=1
                else:
                    lo,hi=0,H
                    while lo<hi:
                        mid=(lo+hi)//2;_,zero,W,n,den=ordered[mid]
                        left,right=pq*den,n*Q;crosses+=1
                        assert max(left,right)<2**128
                        if zero or left>=right:lo=mid+1
                        else:hi=mid
                    possible=lo
                    if possible:
                        _,zero,W,n,den=ordered[possible-1];assert zero or pq*den>=n*Q
                    if possible<H:
                        _,zero,W,n,den=ordered[possible];assert not zero and pq*den<n*Q
                rebuilt[uid]=(possible,3072-possible,506076+772*(3072-possible))
                maximum=max(maximum,possible);total+=possible
            parents.append({'parent':e,'UIDs':len(at),'zero_weight_rows':sum(row[1] for row in rows),
                            'possible_pairs':total,'maximum_possible_rows':maximum})
            ctx.guard()
        assert cursor==len(source)==58253352 and tcur==len(threshold)==9437224
        assert rebuilt.tobytes()==result.tobytes() and parents==raw['parents'] and qzero==raw['query_zero_inputs']
        ctx.r['gates'].update(ALL128_physical_integer_norm_inverse_cross_product_threshold_order=True,
                             ALL17540_independent_search_boundaries_possible_counts_and_lower_cost=True)
        def view(weights):
            ids = np.flatnonzero(weights)
            count = int(weights[ids].sum())
            if count == 0:
                return {'UIDs': 0, 'occurrences': 0, 'mean_minimum_byte_ratio': None, 'p95_minimum_byte_ratio': None}
            weight = weights[ids]
            widths=rebuilt['optimistic_width'][ids].astype('<i8')
            possible=rebuilt['possible'][ids].astype('<i8')
            ratios=rebuilt['minimum_addressed_bytes'][ids].astype('<f8')/2371584
            return {'UIDs':len(ids),'occurrences':count,
                    'mean_minimum_byte_ratio':float(np.sum(weight*ratios)/count),
                    'p95_minimum_byte_ratio':float(np.quantile(np.repeat(ratios,weight),.95)),
                    'mean_optimistic_remaining_rows':float(np.sum(weight*widths)/count),
                    'maximum_possible_rows_per_query':int(possible.max()),
                    'possible_row_occurrences':int(np.sum(weight*possible)),
                    'energy_impossible_row_occurrences':count*3072-int(np.sum(weight*possible)),
                    'zero_possible_occurrences':int(weight[possible==0].sum())}
        views = {'development': view(dev.astype('<i8')), 'consumed': view(val.astype('<i8'))}
        for name, mode in [('natural_consumed', 1), ('teacher_consumed', 0)]:
            weights = np.zeros(N, dtype='<i8')
            for row in occ:
                if row[7] == 1 and row[6] == mode:
                    weights[row[1]] += 1
            views[name] = view(weights)
        books = []
        for book in sorted({int(row[4]) for row in occ if row[7] == 1}):
            weights = np.zeros(N, dtype='<i8')
            for row in occ:
                if row[7] == 1 and row[4] == book:
                    weights[row[1]] += 1
            books.append({'book': book, **view(weights)})
        assert len(books) == 64
        devcount = [int(np.count_nonzero(dev & (m[:, 3] == e))) for e in range(128)]
        rare = {}
        for label, low, high in [('dev_count1_4', 1, 4), ('dev_count5_15', 5, 15)]:
            weights = np.array([int(bool(val[i]) and low <= devcount[int(m[i, 3])] <= high) for i in range(N)], dtype='<i8')
            rare[label] = view(weights)
        eligibility = {'consumed_minimum_mean_byte_ratio_le.75': views['consumed']['mean_minimum_byte_ratio'] <= .75,
                       'natural_minimum_mean_byte_ratio_le.75': views['natural_consumed']['mean_minimum_byte_ratio'] <= .75,
                       'natural_minimum_p95_byte_ratio_le1': views['natural_consumed']['p95_minimum_byte_ratio'] <= 1,
                       'all_consumed_book_minimum_mean_byte_ratio_le1': all(b['mean_minimum_byte_ratio'] <= 1 for b in books),
                       'original_fallback_plus517_index_ratio_le1.20': 5 * (605945856 + len(source)) <= 6 * 605945856}
        assert (views,books,rare,eligibility)==(raw['views'],raw['consumed_books'],raw['rare_consumed'],raw['eligibility'])
        decision='ENERGY_CAPACITY_ADMISSIBLE_ALIGNMENT_STILL_UNRESOLVED' if all(eligibility.values()) else 'FIXED517_RADII_INFORMATION_BLOCKS_SELECTOR_ONLY_REPAIR'
        assert decision==raw['decision'] and raw['integer_Gram_defect']==r
        ctx.r['gates']['ALL_domain_64_book_rare_frozen_optimistic_economic_decisions_independent']=True
        ctx.finish({'main_sha256':args.main_sha,'views':views,'consumed_books':books,'rare_consumed':rare,
                    'eligibility':eligibility,'decision':decision,'physical_parent_thresholds_checked':128*3072,
                    'unique_inputs_audited':N,'independent_U128_search_comparisons':crosses,
                    'tiny_controls_checked':tested,'new_source_projection_solver_model_native_calls':0,
                    'scope':'Energy necessity upper capacity/lower fixed517 byte cost only; not sound activation masks or physical DRAM.'})
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()
