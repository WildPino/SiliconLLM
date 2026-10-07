"""519 dev-only local matched weight/query geometry and exact ideal capacity."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from math import isqrt
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth519_operations import Context,load_queries,write
N,D,H,E,K,S=17540,768,3072,128,32,256
HEADER=struct.Struct('<8s6I')
PARENT=struct.Struct('<IIQ')
INTRODUCED=506084
CAPACITY=[('possible','<u2'),('optimistic_width','<u2'),('minimum_addressed_bytes','<u4'),('query_norm2','<u8'),('projected_query_norm2','<u8')]

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
         'original_fallback_plus519_index_ratio_le1.20':5*(605945856+indexbytes)<=6*605945856}
    return v,books,rare,gates


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);args=ap.parse_args();ctx=None
    try:
        ctx=Context('main',args.binding_sha);np=ctx.numpy();tested=controls()
        m,occ,inputs,norms=load_queries(np,ctx);q=inputs['core']['q'];dev=(m[:,4]&5)!=0
        ctx.r['gates']['original_QUERY_role_probability_contracts_and_exact_necessity_controls']=True
        vectors=np.lib.format.open_memmap(ctx.out/'selected_solver_vectors.npy',mode='w+',dtype='<f8',shape=(E,D,K))
        spectra=np.lib.format.open_memmap(ctx.out/'solver_eigenvalues.npy',mode='w+',dtype='<f8',shape=(E,D))
        fit=[];indexpath=ctx.out/'local_index.bin'
        weight_moment_MACs=query_moment_MACs=0
        with indexpath.open('xb') as physical:
            physical.write(HEADER.pack(b'M519L001',E,D,H,K,S,0))
            for e in range(E):
                w=ctx.weights(np,e);wf=w.astype('<f8')
                if e:
                    wn=np.load(ctx.data(f'row_norm2_e{e:03}.npy'),allow_pickle=False).astype('<i8')
                else:
                    w32=w.astype('<i4');wn=np.sum(w32*w32,axis=1,dtype='<i8')
                assert np.all(wn>=0) and np.all(wn<=D*128**2)
                validw=wn>0
                matrix=np.zeros((D,D),dtype='<f8')
                if validw.any():
                    unitw=wf[validw]/np.sqrt(wn[validw].astype('<f8'))[:,None]
                    matrix=(unitw.T @ unitw)/int(validw.sum())
                    weight_moment_MACs+=int(validw.sum())*D*D
                train=np.flatnonzero(dev&(m[:,3]==e));validq=train[norms[train]>0]
                if len(validq):
                    unitq=q[validq].astype('<f8')/np.sqrt(norms[validq].astype('<f8'))[:,None]
                    matrix+=(unitq.T @ unitq)/len(validq)
                    query_moment_MACs+=len(validq)*D*D
                expected_trace=int(validw.any())+int(bool(len(validq)))
                assert abs(float(np.trace(matrix))-expected_trace)<=1e-10
                eigen,v=np.linalg.eigh(matrix);order=np.argsort(-eigen,kind='stable');eigen=eigen[order];v=v[:,order[:K]]
                for j in range(K):
                    pivot=int(np.argmax(np.abs(v[:,j])))
                    if v[pivot,j]<0:v[:,j]*=-1
                residual=float(np.linalg.norm(matrix @ v-v*eigen[:K][None,:])/max(float(np.linalg.norm(matrix)),1e-300))
                orthogonality=float(np.max(np.abs(v.T @ v-np.eye(K))))
                assert residual<=1e-10 and orthogonality<=1e-10
                basis=np.rint(v*S).astype('<i2');assert np.abs(basis.astype('<i4')).max()<=S
                b64=basis.astype('<i8');gram=b64.T @ b64
                R=int(np.max(np.sum(np.abs(np.eye(K,dtype='<i8')*S*S-gram),axis=1,dtype='<i8')))
                assert R<S*S
                delta=S*S-R
                projected_float=wf @ basis.astype('<f8')
                assert np.array_equal(projected_float,np.rint(projected_float)) and np.abs(projected_float).max()<=D*128*S<2**31
                projection=projected_float.astype('<i4');p64=projection.astype('<i8')
                pw=np.sum(p64*p64,axis=1,dtype='<i8');assert np.all(pw<=2*S*S*wn)
                rw=wn*S**4-delta*pw;assert np.all(rw>=0) and np.all(rw<=wn*S**4)
                roots=np.fromiter((isqrt(int(x)) for x in rw),dtype='<u4',count=H)
                physical.write(PARENT.pack(e,len(train),R));physical.write(basis.tobytes());physical.write(gram.astype('<i8').tobytes())
                physical.write(projection.tobytes());physical.write(pw.astype('<u8').tobytes());physical.write(rw.astype('<u8').tobytes());physical.write(roots.tobytes())
                vectors[e]=v;spectra[e]=eigen
                fit.append({'parent':e,'development_UIDs':len(train),'nonzero_dev_queries':len(validq),'nonzero_WI_rows':int(validw.sum()),
                    'moment_trace':float(np.trace(matrix)),'moment_sha256':hashlib.sha256(matrix.tobytes()).hexdigest(),
                    'basis_sha256':hashlib.sha256(basis.tobytes()).hexdigest(),'integer_Gram_defect':R,
                    'selected_solver_residual':residual,'selected_solver_orthogonality':orthogonality,
                    'selected_mean_energy_objective':float(eigen[:K].sum()),'no_dev_uses_WI_term_only':not len(validq)})
                ctx.guard()
                if e%8==7:print(json.dumps({'fit_parent_terminal':e,'seconds':ctx.resources()['seconds']}),flush=True)
        vectors.flush();spectra.flush()
        assert indexpath.stat().st_size==32+128*512016==65538080
        write(ctx.out/'basis_freeze.json',{'all128_bases_frozen_before_any_consumed_projection':True,
            'index_sha256':ctx.digest(indexpath),'parents':fit,'rank':K,'scale':S,'WI_query_mixture':[1,1],
            'solver_scope':'stored top32 vectors and complete scalar spectra; no full-spectrum eigenvector optimality proof'})
        ctx.r['gates']['ALL128_development_only_matched_local_bases_Gram_and_integer_dictionary_before_consumed']=True
        capacity=np.lib.format.open_memmap(ctx.out/'ideal_capacity.npy',mode='w+',dtype=CAPACITY,shape=(N,))
        source=indexpath.read_bytes();cursor=32;parents=[];zeroQ=0
        for e in range(E):
            parent,count,R=PARENT.unpack_from(source,cursor);cursor+=16
            assert parent==e and count==fit[e]['development_UIDs'] and R==fit[e]['integer_Gram_defect'];delta=S*S-R
            basis=np.frombuffer(source,dtype='<i2',count=D*K,offset=cursor).reshape(D,K);cursor+=D*K*2+K*K*8+H*K*4
            pw=np.frombuffer(source,dtype='<u8',count=H,offset=cursor);cursor+=H*8
            rw=np.frombuffer(source,dtype='<u8',count=H,offset=cursor);cursor+=H*8+H*4
            recover=rw+delta*pw;assert np.all(recover%S**4==0);wn=recover//S**4
            keys=sorted((Fraction(int(rw[j]),delta*int(wn[j])) if wn[j] else Fraction(-1),j) for j in range(H))
            thresholds=[a for a,_ in keys]
            at=np.flatnonzero(m[:,3]==e);total=maximum=0
            for start in range(0,len(at),128):
                ids=at[start:start+128];projected=q[ids].astype('<f8') @ basis.astype('<f8')
                assert np.array_equal(projected,np.rint(projected))
                qi=projected.astype('<i8');pq=np.sum(qi*qi,axis=1,dtype='<i8')
                assert np.all(pq>=0) and np.all(pq<=2*S*S*norms[ids])
                for local,uid in enumerate(ids):
                    Q=int(norms[uid]);p=int(pq[local])
                    if Q:possible=bisect_right(thresholds,Fraction(p,Q))
                    else:possible=H;zeroQ+=1
                    capacity[uid]=(possible,H-possible,INTRODUCED+(H-possible)*772,Q,p)
                    total+=possible;maximum=max(maximum,possible)
            parents.append({'parent':e,'UIDs':len(at),'possible_pairs':total,'maximum_possible_rows':maximum,'integer_Gram_defect':R})
            ctx.guard()
        capacity.flush();assert cursor==len(source)==65538080
        v,books,rare,gates=views(np,m,occ,capacity,len(source))
        ctx.r['gates'].update(ALL17540_exact_query_projections_necessary_energy_counts_and_lower_cost=True,
            ALL_original_fallback_local_header_storage_ALL64_book_rare_frozen_decisions=True)
        ctx.finish({'views':v,'consumed_books':books,'rare_consumed':rare,'eligibility':gates,'parents':parents,'fit':fit,
            'index_bytes':len(source),'introduced_logical_bytes_per_UID':INTRODUCED,'original_fallback_bank_bytes':605945856,
            'query_zero_inputs':zeroQ,'tiny_controls_checked':tested,'weight_moment_MACs':weight_moment_MACs,
            'development_query_moment_MACs':query_moment_MACs,'new_WI_projection_MACs':E*H*D*K,'new_query_projection_MACs':N*D*K,
            'new_WI_WO_model_native_function_evaluations':0,'physical_DRAM_verified':False,
            'decision':'NEXT_ACTUAL_LOCAL_CERTIFICATE_BYTE_AND_OPERATOR_COST' if all(gates.values()) else 'CLOSE_THIS_LOCAL_MATCHED_RANK32_ENERGY_RECIPE',
            'scope':'Source-derived local dictionaries and exact necessary capacity/optimistic byte floor only; possible rows are not sound masks or whole quality/rate.'})
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()

