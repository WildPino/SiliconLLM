"""517 independent I64 transforms, exact wide predicates and physical inverse."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
import hashlib
import json
from math import isqrt
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth517_operations import Context,ROOT,DOC,load_inputs
N,D,H=17540,768,3072


def upperroot(n):
    if n==0:return 0
    return 1+isqrt(n-1)


def controls():
    count=0
    for b in [((4,0),(0,4)),((4,1),(0,3)),((3,-1),(1,3)),((3,1),(1,4))]:
        g=[[sum(b[k][i]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        defect=max(sum(abs(16*int(i==j)-g[i][j]) for j in range(2)) for i in range(2))
        assert defect<16
        for wx in range(-2,3):
            for wy in range(-2,3):
                for qx in range(-2,3):
                    for qy in range(-2,3):
                        w,q=(wx,wy),(qx,qy)
                        u=[sum(w[k]*b[k][j] for k in range(2)) for j in range(2)]
                        v=[sum(q[k]*b[k][j] for k in range(2)) for j in range(2)]
                        a=sum(x*y for x,y in zip(u,v));pw=sum(x*x for x in u);pq=sum(x*x for x in v)
                        rw=sum(x*x for x in w)*256-(16-defect)*pw
                        rq=sum(x*x for x in q)*256-(16-defect)*pq
                        ew=[16*w[k]-sum(b[k][j]*u[j] for j in range(2)) for k in range(2)]
                        eq=[16*q[k]-sum(b[k][j]*v[j] for j in range(2)) for k in range(2)]
                        dot=sum(x*y for x,y in zip(w,q))
                        error=sum(u[i]*(16*int(i==j)-g[i][j])*v[j] for i in range(2) for j in range(2))
                        assert dot*256==sum(x*y for x,y in zip(ew,eq))+a*16+error
                        assert rw>=sum(x*x for x in ew) and rq>=sum(x*x for x in eq)
                        if a*16+defect*upperroot(pw*pq)+upperroot(rw*rq)<=0:
                            assert dot<=0 and -a*16>=isqrt(rw)*isqrt(rq)
                        count+=1
    for n in range(201):
        r=upperroot(n);assert r*r>=n and (r==0 or (r-1)*(r-1)<n)
    assert count==2500
    assert (768*128**2)*(768*32767**2)*256**8 <2**128
    assert 2*256**2*upperroot((768*128**2)*(768*32767**2)) <2**53
    return count


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--main-sha',required=True)
    args=ap.parse_args();ctx=None
    try:
        ctx=Context('audit',args.binding_sha)
        mainpath=DOC/'meth517_main_result.json';assert ctx.digest(mainpath)==args.main_sha
        raw=json.loads(mainpath.read_bytes());assert all(raw['gates'].values()) and raw['binding_sha256']==args.binding_sha
        for item in raw['output_inventory']:assert ctx.digest(item['path'])==item['sha256']
        np=ctx.numpy();tested=controls()
        m,occ,inputs,normcache,hidden,codes,scales=load_inputs(np,ctx)
        q=inputs['core']['q'];dev,val=(m[:,4]&5)!=0,(m[:,4]&2)!=0
        folder=ROOT/'results/native_expert_scaling/meth517_index'
        payload=(folder/'prefix_index.bin').read_bytes()
        magic,ec,dc,hc,kc,s,reserved,defect=struct.unpack_from('<8s6IQ',payload)
        assert (magic,ec,dc,hc,kc,s,reserved)==(b'M517P001',128,768,3072,32,256,0)
        cursor=40
        basis=np.frombuffer(payload,dtype='<i2',count=768*32,offset=cursor).reshape(768,32);cursor+=768*32*2
        encodedgram=np.frombuffer(payload,dtype='<i8',count=32*32,offset=cursor).reshape(32,32);cursor+=32*32*8
        b64=basis.astype('<i8');gram=np.einsum('ki,kj->ij',b64,b64,dtype='<i8')
        assert gram.tobytes()==encodedgram.tobytes() and np.abs(b64).max()<=256
        independent_defect=max(sum(abs(65536*int(i==j)-int(gram[i,j])) for j in range(32)) for i in range(32))
        assert independent_defect==defect<65536 and raw['integer_Gram_defect']==defect
        stage=json.loads((folder/'basis_stage.json').read_bytes())
        assert stage['basis_sha256']==hashlib.sha256(basis.tobytes()).hexdigest() and stage['basis_frozen_before_any_consumed_projection_or_hidden_check']
        train=q[dev].astype('<i8')
        covariance=train.T @ train
        stored=np.load(folder/'development_second_moment.npy',allow_pickle=False)
        assert stored.dtype==np.dtype('<i8') and stored.tobytes()==covariance.tobytes()
        vectors=np.load(folder/'development_eigenvectors.npy',allow_pickle=False)
        values=np.load(folder/'development_eigenvalues.npy',allow_pickle=False)
        assert vectors.shape==(768,768) and values.shape==(768,) and np.all(values[:-1]>=values[1:])
        assert np.max(np.abs(vectors.T @ vectors-np.eye(768)))<=1e-10
        assert np.linalg.norm(covariance.astype('<f8') @ vectors-vectors*values[None,:])/np.linalg.norm(covariance.astype('<f8'))<=1e-10
        for j in range(768):assert vectors[int(np.argmax(np.abs(vectors[:,j]))),j]>=0
        assert np.rint(vectors[:,:32]*256).astype('<i2').tobytes()==basis.tobytes()
        ctx.r['gates'].update(independent_rank2_exact_algebra_zero_sign_roots_extreme_controls=True,
              ALL_development_only_exact_integer_covariance_stored_solver_and_Gram_defect=True)
        masks=np.load(folder/'omitted_masks.npy',mmap_mode='r',allow_pickle=False)
        work=np.load(folder/'work.npy',mmap_mode='r',allow_pickle=False)
        dtype=np.dtype([('width','<u2'),('skipped','<u2'),('candidates','<u2'),('reserved','<u2'),('addressed_bytes','<u4'),('query_norm2','<u8'),('projected_query_norm2','<u8')])
        assert masks.shape==(N,384) and work.shape==(N,) and work.dtype==dtype
        rebuilt_width=np.empty(N,dtype='<i8');rebuilt_norm=np.empty(N,dtype='<i8')
        rebuilt_projectednorm=np.empty(N,dtype='<i8');rebuilt_candidates=np.empty(N,dtype='<i8')
        parents=[]
        for e in range(128):
            projection=np.frombuffer(payload,dtype='<i4',count=H*32,offset=cursor).reshape(H,32);cursor+=H*32*4
            pw=np.frombuffer(payload,dtype='<u8',count=H,offset=cursor);cursor+=H*8
            rw=np.frombuffer(payload,dtype='<u8',count=H,offset=cursor);cursor+=H*8
            roots=np.frombuffer(payload,dtype='<u4',count=H,offset=cursor);cursor+=H*4
            w=ctx.weights(np,e).astype('<i8');project=w @ b64
            assert project.astype('<i4').tobytes()==projection.tobytes() and np.abs(project).max()<2**31
            wnorm=np.einsum('ij,ij->i',w,w,dtype='<i8');projectnorm=np.einsum('ij,ij->i',project,project,dtype='<i8')
            assert projectnorm.astype('<u8').tobytes()==pw.tobytes()
            residual=wnorm*256**4-(65536-defect)*projectnorm
            assert np.all(residual>=0) and residual.astype('<u8').tobytes()==rw.tobytes()
            assert np.fromiter((isqrt(int(x)) for x in residual),dtype='<u4',count=H).tobytes()==roots.tobytes()
            if e:assert wnorm.astype('<u4').tobytes()==np.load(ctx.data(f'row_norm2_e{e:03}.npy'),allow_pickle=False).tobytes()
            at=np.flatnonzero(m[:,3]==e);candidate_total=certified=0
            for start in range(0,len(at),128):
                ix=at[start:start+128];x=q[ix].astype('<i8')
                nq=np.einsum('ij,ij->i',x,x,dtype='<i8');query=x @ b64
                assert nq.tobytes()==normcache[ix].tobytes()
                pq=np.einsum('ij,ij->i',query,query,dtype='<i8')
                assert np.all(pq>=0) and np.all(pq<=131072*nq)
                central=query @ project.T
                rq=[int(a)*256**4-(65536-defect)*int(b) for a,b in zip(nq,pq)]
                assert min(rq)>=0
                qroots=np.array([isqrt(v) for v in rq],dtype='<u8')
                low=np.multiply(qroots[:,None],roots[None,:],dtype='<u8')
                thresholds=(low>>16)+(np.bitwise_and(low,65535)!=0)
                candidate=(central<=0)&((-central).astype('<u8')>=thresholds)
                reject=np.zeros(candidate.shape,dtype=bool)
                for local in range(len(ix)):
                    for row in np.flatnonzero(candidate[local]):
                        gramguard=defect*upperroot(int(projectnorm[row])*int(pq[local]))
                        residualguard=upperroot(int(residual[row])*rq[local])
                        reject[local,row]=-int(central[local,row])*65536>=gramguard+residualguard
                assert np.packbits(reject,axis=1,bitorder='little').tobytes()==masks[ix].tobytes()
                skips=np.count_nonzero(reject,axis=1);widths=H-skips;counts=np.count_nonzero(candidate,axis=1)
                check=np.zeros(len(ix),dtype=dtype);check['width'],check['skipped'],check['candidates']=widths,skips,counts
                introduced=40+49152+1536+393216+24576+24576+12288+384+28+256+16+8
                check['addressed_bytes']=introduced+772*widths;check['query_norm2'],check['projected_query_norm2']=nq,pq
                assert check.tobytes()==work[ix].tobytes()
                rebuilt_width[ix],rebuilt_norm[ix],rebuilt_projectednorm[ix],rebuilt_candidates[ix]=widths,nq,pq,counts
                h=hidden[ix].copy();h[reject]=np.float32(0)
                assert h.tobytes()==hidden[ix].tobytes() and np.isfinite(h).all() and np.all(h>=0)
                maximum=np.max(np.abs(h),axis=1)
                assert maximum.tobytes()==np.max(np.abs(hidden[ix]),axis=1).tobytes()
                alpha=maximum/np.float32(32767);alpha[maximum==0]=1
                assert alpha.tobytes()==scales[ix].tobytes()
                assert np.rint(h/alpha[:,None]).clip(-32767,32767).astype('<i2').tobytes()==codes[ix].tobytes()
                certified+=int(skips.sum());candidate_total+=int(counts.sum());ctx.guard()
            parents.append({'parent':e,'UIDs':len(at),'candidates':candidate_total,'certified_cells':certified})
            if e%16==15:print(json.dumps({'audit_parent_terminal':e,'seconds':ctx.resources()['seconds']}),flush=True)
        assert cursor==len(payload)==58253352 and parents==raw['parents']
        ctx.r['gates'].update(ALL128_original_I64_projection_physical_inverse_integer_radii_roots=True,
             ALL53882880_I64_centres_exact_wide_predicates_masks_work_independent=True,
             ALL17540_original_hidden_max_scale_codes_probability_WO_BYTE_identity=True)
        introduced,baseline=506076,2371584
        def view(weights):
            ids = np.flatnonzero(weights)
            count = int(weights[ids].sum())
            if count == 0:
                return {'UIDs': 0, 'occurrences': 0, 'mean_byte_ratio': None, 'p95_byte_ratio': None}
            weight = weights[ids]
            widths = rebuilt_width[ids]
            ratios = (introduced + widths * 772).astype('<f8') / baseline
            fraction = np.divide(rebuilt_projectednorm[ids].astype('<f8'), rebuilt_norm[ids] * (256*256),
                                 out=np.zeros(len(ids)), where=rebuilt_norm[ids] != 0)
            return {'UIDs': len(ids), 'occurrences': count,
                    'mean_byte_ratio': float(np.sum(weight * ratios) / count),
                    'p95_byte_ratio': float(np.quantile(np.repeat(ratios, weight), .95)),
                    'mean_remaining_rows': float(np.sum(weight * widths) / count),
                    'maximum_remaining_rows': int(widths.max()),
                    'unchanged_full_width_occurrences': int(weight[widths == H].sum()),
                    'certified_row_occurrences': int(np.sum(weight * (3072 - widths))),
                    'mean_basis_query_energy_ratio': float(np.sum(weight * fraction) / count),
                    'prefix_integer_MACs': count * 3072 * 32,
                    'fallback_integer_MACs': int(np.sum(weight * widths)) * 768,
                    'query_energy_integer_products': count * 800,
                    'row_floor_root_U64_products': count * 3072,
                    'query_transform_integer_MACs': count * 768 * 32, 'bitmap_words': count * 48,
                    'exact_U128_ceil_sqrts': 2 * int(np.sum(weight * rebuilt_candidates[ids])),
                    'introduced_logical_bytes': count * 506076,
                    'fallback_logical_bytes': int(np.sum(weight * widths)) * 772}

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
        eligibility = {'consumed_mean_byte_ratio_le.75': views['consumed']['mean_byte_ratio'] <= .75,
                       'natural_mean_byte_ratio_le.75': views['natural_consumed']['mean_byte_ratio'] <= .75,
                       'natural_p95_byte_ratio_le1': views['natural_consumed']['p95_byte_ratio'] <= 1,
                       'all_consumed_book_mean_byte_ratio_le1': all(b['mean_byte_ratio'] <= 1 for b in books),
                       'complete_fallback_bank_plus_index_ratio_le1.20': 5 * (605945856 + len(payload)) <= 6 * 605945856}
        assert (views,books,rare,eligibility)==(raw['views'],raw['consumed_books'],raw['rare_consumed'],raw['eligibility'])
        decision='NEXT_NATIVE_EXACT_PARITY_AND_COMPLETE_COST' if all(eligibility.values()) else 'CLOSE_THIS_SHARED_RANK32_INTEGER_CERTIFICATE'
        assert raw['decision']==decision and raw['introduced_logical_bytes_per_UID']==introduced
        assert raw['basis_rank']==32 and raw['basis_scale']==256 and raw['rho']==defect/65536
        ctx.r['gates']['ALL64_books_rare_views_and_frozen_economic_decisions_independent']=True
        ctx.finish({'main_sha256':args.main_sha,'views':views,'consumed_books':books,'rare_consumed':rare,
              'eligibility':eligibility,'decision':decision,'index_bytes':len(payload),'unique_inputs_audited':N,
              'partial_cells_audited':N*H,'physical_parents_audited':128,'independent_tiny_controls_checked':tested,
              'integer_Gram_defect':defect,'old_full_WI_WO_model_evaluations':0,'physical_DRAM_verified':False})
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()
