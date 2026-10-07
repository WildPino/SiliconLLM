"""Independent weighted moments, I64 physical projections and ideal capacity."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
from fractions import Fraction
from functools import cmp_to_key
import hashlib
import json
from math import isqrt
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth519_operations import Context,ROOT,DOC,load_queries
N,D,H=17540,768,3072


def controls():
    tested=0
    for b in [((4,0),(0,4)),((4,1),(0,3)),((3,-1),(1,3)),((3,1),(1,4))]:
        g=[[sum(b[k][i]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        r=max(sum(abs(16*int(i==j)-g[i][j]) for j in range(2)) for i in range(2));delta=16-r
        assert delta>0
        for wx in range(-2,3):
            for wy in range(-2,3):
                for qx in range(-2,3):
                    for qy in range(-2,3):
                        W,Q=wx*wx+wy*wy,qx*qx+qy*qy
                        u=[wx*b[0][j]+wy*b[1][j] for j in range(2)]
                        v=[qx*b[0][j]+qy*b[1][j] for j in range(2)]
                        pw,pq=sum(x*x for x in u),sum(x*x for x in v)
                        RW,RQ=W*256-delta*pw,Q*256-delta*pq
                        assert RW>=0 and RQ>=0
                        root=lambda n:0 if n==0 else 1+isqrt(n-1)
                        upper=16*sum(x*y for x,y in zip(u,v))+r*root(pw*pq)+root(RW*RQ)
                        if W and Q:
                            possible=Fraction(delta,16)*(Fraction(pw,16*W)+Fraction(pq,16*Q))>=1
                            assert possible==(delta*W*pq>=RW*Q)
                            if upper<=0:assert possible
                        tested+=1
    assert tested==2500
    return tested


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--main-sha',required=True)
    args=ap.parse_args();ctx=None
    try:
        ctx=Context('audit',args.binding_sha);path=DOC/'meth519_main_result.json'
        assert ctx.digest(path)==args.main_sha;raw=json.loads(path.read_bytes())
        assert all(raw['gates'].values()) and raw['binding_sha256']==args.binding_sha
        for v in raw['output_inventory']:assert ctx.digest(v['path'])==v['sha256']
        np=ctx.numpy();tested=controls();m,occ,inputs,normcache=load_queries(np,ctx)
        q=inputs['core']['q'];dev,val=(m[:,4]&5)!=0,(m[:,4]&2)!=0
        folder=ROOT/'results/native_expert_scaling/meth519_index';payload=(folder/'local_index.bin').read_bytes()
        assert struct.unpack_from('<8s6I',payload)==(b'M519L001',128,768,3072,32,256,0)
        vectors=np.load(folder/'selected_solver_vectors.npy',mmap_mode='r',allow_pickle=False)
        eigen=np.load(folder/'solver_eigenvalues.npy',mmap_mode='r',allow_pickle=False)
        frozen=json.loads((folder/'basis_freeze.json').read_bytes())
        assert frozen['all128_bases_frozen_before_any_consumed_projection'] and frozen['index_sha256']==ctx.digest(folder/'local_index.bin')
        assert frozen['parents']==raw['fit'] and frozen['WI_query_mixture']==[1,1]
        assert vectors.shape==(128,768,32) and eigen.shape==(128,768)
        result=np.load(folder/'ideal_capacity.npy',mmap_mode='r',allow_pickle=False)
        dtype=np.dtype([('possible','<u2'),('optimistic_width','<u2'),('minimum_addressed_bytes','<u4'),('query_norm2','<u8'),('projected_query_norm2','<u8')])
        assert result.shape==(N,) and result.dtype==dtype
        rebuilt=np.empty(N,dtype=dtype);cursor=32;parents=[];zeroQ=0
        ctx.r['gates']['original_QUERY_contracts_prior_basis_freeze_and2500_independent_controls']=True
        for e in range(128):
            parent,devcount,R=struct.unpack_from('<IIQ',payload,cursor);cursor+=16
            assert parent==e and R<65536;delta=65536-R
            basis=np.frombuffer(payload,dtype='<i2',count=768*32,offset=cursor).reshape(768,32);cursor+=49152
            gram=np.frombuffer(payload,dtype='<i8',count=32*32,offset=cursor).reshape(32,32);cursor+=8192
            projection=np.frombuffer(payload,dtype='<i4',count=H*32,offset=cursor).reshape(H,32);cursor+=393216
            pw=np.frombuffer(payload,dtype='<u8',count=H,offset=cursor);cursor+=24576
            rw=np.frombuffer(payload,dtype='<u8',count=H,offset=cursor);cursor+=24576
            roots=np.frombuffer(payload,dtype='<u4',count=H,offset=cursor);cursor+=12288
            b64=basis.astype('<i8');assert np.abs(b64).max()<=256
            truegram=np.einsum('ki,kj->ij',b64,b64,dtype='<i8')
            assert truegram.tobytes()==gram.tobytes()
            independentR=max(sum(abs(65536*int(i==j)-int(gram[i,j])) for j in range(32)) for i in range(32))
            assert independentR==R
            w=ctx.weights(np,e).astype('<i8');wn=np.einsum('ij,ij->i',w,w,dtype='<i8')
            if e:assert wn.astype('<u4').tobytes()==np.load(ctx.data(f'row_norm2_e{e:03}.npy'),allow_pickle=False).tobytes()
            train=np.flatnonzero(dev&(m[:,3]==e));validq=train[normcache[train]>0];assert devcount==len(train)
            validw=wn>0;matrix=np.zeros((D,D),dtype='<f8')
            if validw.any():
                x=w[validw].astype('<f8');matrix=x.T @ (x/wn[validw,None])/int(validw.sum())
            if len(validq):
                x=q[validq].astype('<f8');matrix+=x.T @ (x/normcache[validq,None])/len(validq)
            V=vectors[e];values=eigen[e]
            assert np.isfinite(V).all() and np.isfinite(values).all() and np.all(values[:-1]>=values[1:])
            assert np.max(np.abs(V.T @ V-np.eye(32)))<=1e-10
            assert np.linalg.norm(matrix @ V-V*values[:32][None,:])/max(float(np.linalg.norm(matrix)),1e-300)<=1e-10
            fit=raw['fit'][e]
            assert fit['development_UIDs']==len(train) and fit['nonzero_dev_queries']==len(validq) and fit['nonzero_WI_rows']==int(validw.sum())
            assert abs(float(np.trace(matrix))-fit['moment_trace'])<=1e-10
            assert abs(float(values[:32].sum())-fit['selected_mean_energy_objective'])<=1e-10
            for j in range(32):assert V[int(np.argmax(np.abs(V[:,j]))),j]>=0
            assert np.rint(V*256).astype('<i2').tobytes()==basis.tobytes()
            assert hashlib.sha256(basis.tobytes()).hexdigest()==fit['basis_sha256'] and fit['integer_Gram_defect']==R
            projected=w @ b64
            assert np.abs(projected).max()<2**31 and projected.astype('<i4').tobytes()==projection.tobytes()
            pnorm=np.einsum('ij,ij->i',projected,projected,dtype='<i8');residual=wn*256**4-delta*pnorm
            assert np.all(residual>=0) and np.all(pnorm<=131072*wn)
            assert pnorm.astype('<u8').tobytes()==pw.tobytes() and residual.astype('<u8').tobytes()==rw.tobytes()
            assert np.fromiter((isqrt(int(x)) for x in residual),dtype='<u4',count=H).tobytes()==roots.tobytes()
            rows=[(j,int(wn[j]),int(residual[j]),delta*int(wn[j])) for j in range(H)]
            def cmp(a,b):
                if (a[1]==0)!=(b[1]==0):return -1 if a[1]==0 else 1
                left,right=a[2]*b[3],b[2]*a[3];assert max(left,right)<2**128
                if left!=right:return -1 if left<right else 1
                return (a[0]>b[0])-(a[0]<b[0])
            ordered=sorted(rows,key=cmp_to_key(cmp));at=np.flatnonzero(m[:,3]==e);total=maximum=0
            for start in range(0,len(at),128):
                ids=at[start:start+128];x=q[ids].astype('<i8');nq=np.einsum('ij,ij->i',x,x,dtype='<i8')
                assert nq.tobytes()==normcache[ids].tobytes()
                query=x @ b64;pq=np.einsum('ij,ij->i',query,query,dtype='<i8')
                assert np.all(pq>=0) and np.all(pq<=131072*nq)
                for local,uid in enumerate(ids):
                    Q,P=int(nq[local]),int(pq[local])
                    if Q==0:possible=H;zeroQ+=1
                    else:
                        lo,hi=0,H
                        while lo<hi:
                            mid=(lo+hi)//2;_,W,n,den=ordered[mid]
                            left,right=den*P,n*Q;assert max(left,right)<2**128
                            if W==0 or left>=right:lo=mid+1
                            else:hi=mid
                        possible=lo
                        if lo:
                            _,W,n,den=ordered[lo-1];assert W==0 or den*P>=n*Q
                        if lo<H:
                            _,W,n,den=ordered[lo];assert W>0 and den*P<n*Q
                    rebuilt[uid]=(possible,H-possible,506084+772*(H-possible),Q,P)
                    total+=possible;maximum=max(maximum,possible)
            parents.append({'parent':e,'UIDs':len(at),'possible_pairs':total,'maximum_possible_rows':maximum,'integer_Gram_defect':R})
            ctx.guard()
            if e%8==7:print(json.dumps({'audit_parent_terminal':e,'seconds':ctx.resources()['seconds']}),flush=True)
        assert cursor==len(payload)==65538080 and rebuilt.tobytes()==result.tobytes() and parents==raw['parents']
        assert raw['query_zero_inputs']==zeroQ
        ctx.r['gates'].update(ALL128_independent_weighted_moments_stored_top32_solver_and_encoded_Gram=True,
            ALL128_I64_native_projection_full_norm_radius_root_physical_inverse=True,
            ALL17540_I64_query_norm_projection_cross_product_capacity_boundaries=True)
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
                       'original_fallback_plus519_index_ratio_le1.20': 5 * (605945856 + len(payload)) <= 6 * 605945856}
        assert (views,books,rare,eligibility)==(raw['views'],raw['consumed_books'],raw['rare_consumed'],raw['eligibility'])
        decision='NEXT_ACTUAL_LOCAL_CERTIFICATE_BYTE_AND_OPERATOR_COST' if all(eligibility.values()) else 'CLOSE_THIS_LOCAL_MATCHED_RANK32_ENERGY_RECIPE'
        assert raw['decision']==decision and raw['index_bytes']==len(payload) and raw['introduced_logical_bytes_per_UID']==506084
        ctx.r['gates']['ALL64_book_rare_domain_and_frozen_ideal_economic_decisions_independent']=True
        ctx.finish({'main_sha256':args.main_sha,'views':views,'consumed_books':books,'rare_consumed':rare,
            'eligibility':eligibility,'decision':decision,'index_bytes':len(payload),'physical_parents_audited':128,
            'query_capacity_boundaries_audited':N,'original_projected_WI_coefficients_audited':128*3072*32,
            'independent_tiny_controls_checked':tested,'new_WI_WO_model_native_function_evaluations':0,
            'physical_DRAM_verified':False,'scope':'Independent physical local dictionaries/weighted moment residuals/exact necessary capacity; not sound masks or model quality/rate.'})
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()
