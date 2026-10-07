"""Exact necessary energy thresholds; ideal certificate capacity, not sound skips."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
from bisect import bisect_right
from fractions import Fraction
import json
from math import isqrt
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth518_operations import Context,wire
N,D,H=17540,768,3072
RECORD=struct.Struct('<HHIQQ')


def controls():
    count=0
    for b in [(4,0),(3,1),(-4,1),(1,3)]:
        g=sum(x*x for x in b);r=abs(16-g);delta=16-r
        assert delta>0
        for wx in range(-2,3):
            for wy in range(-2,3):
                for qx in range(-2,3):
                    for qy in range(-2,3):
                        W,Q=wx*wx+wy*wy,qx*qx+qy*qy
                        tw,tq=b[0]*wx+b[1]*wy,b[0]*qx+b[1]*qy
                        pw,pq=tw*tw,tq*tq;RW=W*256-delta*pw;RQ=Q*256-delta*pq
                        assert RW>=0 and RQ>=0
                        cr=lambda n:isqrt(n)+(isqrt(n)**2!=n)
                        actual=tw*tq*16+r*cr(pw*pq)+cr(RW*RQ)
                        possible=W==0 or Q==0 or delta*W*pq>=RW*Q
                        if actual<=0:assert possible and wx*qx+wy*qy<=0
                        if W and Q:
                            assert possible==(Fraction(delta,16)*(Fraction(pw,16*W)+Fraction(pq,16*Q))>=1)
                        count+=1
    assert count==2500 and (768*128**2)*256**4*(768*32767**2)<2**128
    return count


def views(np,m,occ,work,indexbytes):
    dev,val=(m[:,4]&5)!=0,(m[:,4]&2)!=0
    def view(w):
        ids=np.flatnonzero(w);total=int(w[ids].sum())
        if not total:return {'UIDs':0,'occurrences':0,'mean_minimum_byte_ratio':None,'p95_minimum_byte_ratio':None}
        weights=w[ids];ratios=work['minimum_addressed_bytes'][ids].astype('<f8')/2371584
        width=work['optimistic_width'][ids].astype('<i8');possible=work['possible'][ids].astype('<i8')
        return {'UIDs':len(ids),'occurrences':total,
             'mean_minimum_byte_ratio':float(np.sum(weights*ratios)/total),
             'p95_minimum_byte_ratio':float(np.quantile(np.repeat(ratios,weights),.95)),
             'mean_optimistic_remaining_rows':float(np.sum(weights*width)/total),
             'maximum_possible_rows_per_query':int(possible.max()),
             'possible_row_occurrences':int(np.sum(weights*possible)),
             'energy_impossible_row_occurrences':total*3072-int(np.sum(weights*possible)),
             'zero_possible_occurrences':int(weights[possible==0].sum())}
    v={'development':view(dev.astype('<i8')),'consumed':view(val.astype('<i8'))}
    for label,mode in [('natural_consumed',1),('teacher_consumed',0)]:
        w=np.bincount(occ[(occ[:,7]==1)&(occ[:,6]==mode),1],minlength=N);v[label]=view(w)
    books=[]
    for book in np.unique(occ[occ[:,7]==1,4]):
        w=np.bincount(occ[(occ[:,7]==1)&(occ[:,4]==book),1],minlength=N);books.append({'book':int(book),**view(w)})
    assert len(books)==64
    counts=np.bincount(m[dev,3],minlength=128)[m[:,3]]
    rare={label:view((val&flag).astype('<i8')) for label,flag in [
         ('dev_count1_4',(counts>=1)&(counts<=4)),('dev_count5_15',(counts>=5)&(counts<=15))]}
    gates={'consumed_minimum_mean_byte_ratio_le.75':v['consumed']['mean_minimum_byte_ratio']<=.75,
         'natural_minimum_mean_byte_ratio_le.75':v['natural_consumed']['mean_minimum_byte_ratio']<=.75,
         'natural_minimum_p95_byte_ratio_le1':v['natural_consumed']['p95_minimum_byte_ratio']<=1,
         'all_consumed_book_minimum_mean_byte_ratio_le1':all(b['mean_minimum_byte_ratio']<=1 for b in books),
         'original_fallback_plus517_index_ratio_le1.20':5*(605945856+indexbytes)<=6*605945856}
    return v,books,rare,gates


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);args=ap.parse_args();ctx=None
    try:
        ctx=Context('main',args.binding_sha);np=ctx.numpy();tested=controls()
        m=wire(np,ctx.data('uid'),b'M493U001',180,0,[('m','<u4',(13,)),('hash','u1',(128,))],N)['m']
        occ=wire(np,ctx.data('occurrences'),b'M493O001',68,0,np.dtype(('<u4',(17,))),19962)
        assert np.array_equal(m[:,0],np.arange(N)) and np.all(m[:,2]==11) and np.all(m[:,3]<128)
        assert np.array_equal(np.bincount(occ[:,1],minlength=N),m[:,7])
        for oi,mi in [(14,1),(11,3),(15,12),(16,10)]:assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        dev,val=(m[:,4]&5)!=0,(m[:,4]&2)!=0;assert (int(dev.sum()),int(val.sum()))==(11721,5819) and np.all(dev^val)
        prior=np.load(ctx.data('work.npy'),mmap_mode='r',allow_pickle=False)
        assert prior.shape==(N,) and np.all(prior['width']==3072) and np.all(prior['skipped']==0) and np.all(prior['candidates']==0)
        assert np.all(prior['addressed_bytes']==506076+3072*772)
        source=Path(ctx.data('prefix_index.bin')).read_bytes()
        magic,e,d,h,k,s,reserved,r=struct.unpack_from('<8s6IQ',source)
        assert (magic,e,d,h,k,s,reserved)==(b'M517P001',128,768,3072,32,256,0) and 0<=r<65536
        delta=65536-r;cursor=40+768*32*2+32*32*8
        ctx.r['gates']['original_roles_zero517_realized_gates_and2500_necessity_controls']=True
        dtype=[('possible','<u2'),('optimistic_width','<u2'),('minimum_addressed_bytes','<u4')]
        out=np.lib.format.open_memmap(ctx.out/'possible_capacity.npy',mode='w+',dtype=dtype,shape=(N,))
        parents=[];queries_zero=0
        with (ctx.out/'thresholds.bin').open('xb') as f:
            f.write(struct.pack('<8s6IQ',b'M518E001',128,768,3072,32,256,0,r))
            for parent in range(128):
                cursor+=H*32*4
                pw=np.frombuffer(source,dtype='<u8',count=H,offset=cursor);cursor+=H*8
                rw=np.frombuffer(source,dtype='<u8',count=H,offset=cursor);cursor+=H*8+H*4
                assert np.all(pw<=131072*(768*128**2)) and np.all(rw<=(768*128**2)*256**4)
                numerator=rw+delta*pw
                assert np.all(numerator%256**4==0)
                W=numerator//256**4;assert np.all(W<=768*128**2)
                if parent:assert W.astype('<u4').tobytes()==np.load(ctx.data(f'row_norm2_e{parent:03}.npy'),allow_pickle=False).tobytes()
                entries=[]
                for j in range(H):
                    wn,n,den=int(W[j]),int(rw[j]),delta*int(W[j])
                    key=Fraction(n,den) if den else Fraction(-1)
                    assert wn or (n==0 and int(pw[j])==0)
                    entries.append((key,j,wn,n,den))
                entries.sort(key=lambda t:(t[0],t[1]));keys=[t[0] for t in entries]
                for _,j,wn,n,den in entries:f.write(RECORD.pack(j,int(wn==0),wn,n,den))
                at=np.flatnonzero(m[:,3]==parent);maximum=total=0
                for uid in at:
                    Q,pq=int(prior['query_norm2'][uid]),int(prior['projected_query_norm2'][uid])
                    assert Q<=768*32767**2 and pq<=131072*Q
                    if Q==0:possible=H;queries_zero+=1
                    else:possible=bisect_right(keys,Fraction(pq,Q))
                    out[uid]=(possible,H-possible,506076+(H-possible)*772)
                    maximum=max(maximum,possible);total+=possible
                parents.append({'parent':parent,'UIDs':len(at),'zero_weight_rows':int(np.count_nonzero(W==0)),
                                'possible_pairs':total,'maximum_possible_rows':maximum})
                ctx.guard()
                if parent%32==31:print(json.dumps({'main_parent_terminal':parent,'seconds':ctx.resources()['seconds']}),flush=True)
        assert cursor==len(source)==58253352 and (ctx.out/'thresholds.bin').stat().st_size==9437224
        out.flush();v,books,rare,gates=views(np,m,occ,out,len(source))
        ctx.r['gates'].update(ALL128_exact_physical_norm_inverse_Fraction_threshold_order=True,
             ALL17540_ideal_capacity_counts_fixed_radii_lower_cost_bounds=True,
             ALL64_books_rare_domain_and_frozen_lower_bound_decisions=True)
        ctx.finish({'views':v,'consumed_books':books,'rare_consumed':rare,'eligibility':gates,'parents':parents,
             'source517_index_bytes':len(source),'threshold_bytes':9437224,'query_zero_inputs':queries_zero,
             'integer_Gram_defect':r,'tiny_controls_checked':tested,
             'decision':'ENERGY_CAPACITY_ADMISSIBLE_ALIGNMENT_STILL_UNRESOLVED' if all(gates.values()) else 'FIXED517_RADII_INFORMATION_BLOCKS_SELECTOR_ONLY_REPAIR',
             'new_source_projection_solver_model_native_calls':0,'physical_DRAM_verified':False,
             'scope':'Necessary energy capacity upper bound and optimistic fixed517 byte lower bound; possible rows are NOT sound skip decisions.'})
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()
