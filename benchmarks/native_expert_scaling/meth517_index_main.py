"""517 one dev-only fixedpoint shared input basis and exact WI row certificates."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
import argparse
import hashlib
import json
from math import isqrt
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth517_operations import Context, load_inputs, write
N,D,H,E,K,S,BATCH = 17540,768,3072,128,32,256,128
HEADER=struct.Struct('<8s6IQ')
RECORD_BYTES=H*K*4+H*8+H*8+H*4
BASELINE=H*(D+4)
WORK=[('width','<u2'),('skipped','<u2'),('candidates','<u2'),('reserved','<u2'),('addressed_bytes','<u4'),('query_norm2','<u8'),('projected_query_norm2','<u8')]
INTRODUCED=40 + D*K*2 + 2*D + RECORD_BYTES + H//8 + 28 + K*8 + 16 + 8

def ceilroot(n):
    root=isqrt(n)
    return root + (root*root != n)

def summaries(np, m, occ, work, norms, index_bytes):
    dev, val = (m[:, 4] & 5) != 0, (m[:, 4] & 2) != 0

    def view(weights):
        at = np.flatnonzero(weights)
        total = int(weights[at].sum())
        if not total:
            return {'UIDs': 0, 'occurrences': 0, 'mean_byte_ratio': None, 'p95_byte_ratio': None}
        w = weights[at]
        ratios = work['addressed_bytes'][at].astype('<f8') / BASELINE
        width = work['width'][at].astype('<i8')
        energy = np.divide(work['projected_query_norm2'][at].astype('<f8'), norms[at] * (S*S),
                           out=np.zeros(len(at)), where=norms[at] != 0)
        return {'UIDs': len(at), 'occurrences': total,
                'mean_byte_ratio': float(np.sum(w * ratios) / total),
                'p95_byte_ratio': float(np.quantile(np.repeat(ratios, w), .95)),
                'mean_remaining_rows': float(np.sum(w * width) / total),
                'maximum_remaining_rows': int(width.max()),
                'unchanged_full_width_occurrences': int(w[width == H].sum()),
                'certified_row_occurrences': int(np.sum(w * (H - width))),
                'mean_basis_query_energy_ratio': float(np.sum(w * energy) / total),
                'prefix_integer_MACs': total * H * K,
                'fallback_integer_MACs': int(np.sum(w * width)) * D,
                'query_energy_integer_products': total * (D + K),
                'row_floor_root_U64_products': total * H,
                'query_transform_integer_MACs': total * D * K, 'bitmap_words': total * (H // 64),
                'exact_U128_ceil_sqrts': 2 * int(np.sum(w * work['candidates'][at].astype('<i8'))),
                'introduced_logical_bytes': total * INTRODUCED,
                'fallback_logical_bytes': int(np.sum(w * width)) * (D + 4)}

    views = {'development': view(dev.astype('<i8')), 'consumed': view(val.astype('<i8'))}
    for label, mode in [('natural_consumed', 1), ('teacher_consumed', 0)]:
        weights = np.bincount(occ[(occ[:, 7] == 1) & (occ[:, 6] == mode), 1], minlength=N)
        views[label] = view(weights)
    books = []
    for book in np.unique(occ[occ[:, 7] == 1, 4]):
        w = np.bincount(occ[(occ[:, 7] == 1) & (occ[:, 4] == book), 1], minlength=N)
        books.append({'book': int(book), **view(w)})
    assert len(books) == 64
    dev_counts = np.bincount(m[dev, 3], minlength=E)[m[:, 3]]
    rare = {label: view((val & flag).astype('<i8')) for label, flag in [
        ('dev_count1_4', (dev_counts >= 1) & (dev_counts <= 4)),
        ('dev_count5_15', (dev_counts >= 5) & (dev_counts <= 15))]}
    eligibility = {'consumed_mean_byte_ratio_le.75': views['consumed']['mean_byte_ratio'] <= .75,
                   'natural_mean_byte_ratio_le.75': views['natural_consumed']['mean_byte_ratio'] <= .75,
                   'natural_p95_byte_ratio_le1': views['natural_consumed']['p95_byte_ratio'] <= 1,
                   'all_consumed_book_mean_byte_ratio_le1': all(b['mean_byte_ratio'] <= 1 for b in books),
                   'complete_fallback_bank_plus_index_ratio_le1.20': (605945856 + index_bytes) * 5 <= 605945856 * 6}
    return views, books, rare, eligibility


def tiny_controls():
    tested = 0
    for b in [(4, 0), (3, 1), (-4, 1), (1, 3)]:
        scale, g = 4, sum(z*z for z in b)
        defect = abs(scale*scale-g)
        assert defect < scale*scale
        for wx in range(-2,3):
            for wy in range(-2,3):
                for qx in range(-2,3):
                    for qy in range(-2,3):
                        w,q=(wx,wy),(qx,qy)
                        tw=sum(x*y for x,y in zip(w,b)); tq=sum(x*y for x,y in zip(q,b))
                        pw,pq=tw*tw,tq*tq
                        rw=sum(x*x for x in w)*scale**4-(scale*scale-defect)*pw
                        rq=sum(x*x for x in q)*scale**4-(scale*scale-defect)*pq
                        assert rw>=0 and rq>=0
                        ew=[scale*scale*x-y*tw for x,y in zip(w,b)]
                        eq=[scale*scale*x-y*tq for x,y in zip(q,b)]
                        actual=sum(x*y for x,y in zip(w,q))
                        assert actual*scale**4 == sum(x*y for x,y in zip(ew,eq))+tw*tq*scale*scale+tw*tq*(scale*scale-g)
                        assert sum(x*x for x in ew)<=rw and sum(x*x for x in eq)<=rq
                        upper=tw*tq*scale*scale+defect*ceilroot(pw*pq)+ceilroot(rw*rq)
                        if upper<=0:
                            assert actual<=0
                            assert -tw*tq*scale*scale>=isqrt(rw)*isqrt(rq)
                        tested+=1
    assert tested==2500
    assert (D*128**2)*(D*32767**2) < 2**64
    assert (D*128**2)*S**4 < 2**63
    assert (D*128**2)*(D*32767**2)*S**8 < 2**128
    assert 2*S*S*ceilroot((D*128**2)*(D*32767**2)) < 2**53
    return tested


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True)
    args=ap.parse_args();ctx=None
    try:
        ctx=Context('main',args.binding_sha);np=ctx.numpy()
        tested=tiny_controls()
        m,occ,inputs,norms,hidden,codes,scales=load_inputs(np,ctx)
        q=inputs['core']['q'];dev=(m[:,4]&5)!=0
        ctx.r['gates']['original_QUERY_roles_parent_probability_and_2500_exact_controls']=True
        train=q[dev].astype('<f8')
        covariance=train.T @ train
        assert np.array_equal(covariance,np.rint(covariance))
        values,vectors=np.linalg.eigh(covariance)
        order=np.argsort(-values,kind='stable');values=values[order];vectors=vectors[:,order]
        for column in range(D):
            pivot=int(np.argmax(np.abs(vectors[:,column])))
            if vectors[pivot,column]<0:vectors[:,column]*=-1
        basis=np.rint(vectors[:,:K]*S).astype('<i2')
        assert np.max(np.abs(basis.astype('<i4')))<=S
        gram=basis.astype('<i8').T @ basis.astype('<i8')
        defect=int(np.max(np.sum(np.abs(np.eye(K,dtype='<i8')*S*S-gram),axis=1,dtype='<i8')))
        assert 0<=defect<S*S
        np.save(ctx.out/'development_second_moment.npy',covariance.astype('<i8'),allow_pickle=False)
        np.save(ctx.out/'development_eigenvalues.npy',values,allow_pickle=False)
        np.save(ctx.out/'development_eigenvectors.npy',vectors,allow_pickle=False)
        write(ctx.out/'basis_stage.json',{'development_UIDs':11721,'rank':K,'scale':S,'integer_Gram_defect':defect,
              'basis_sha256':hashlib.sha256(basis.tobytes()).hexdigest(),'basis_bytes':D*K*2,
              'basis_frozen_before_any_consumed_projection_or_hidden_check':True,
              'selection':'ONE uncentred dev second moment eigh, descending values stable ties, largest-absolute lowest-index coordinate positive; round-to-even times256.'})
        ctx.r['gates']['ONE_development_only_shared_rank32_encoded_Gram_defect_before_consumed']=True
        masks=np.lib.format.open_memmap(ctx.out/'omitted_masks.npy',mode='w+',dtype='u1',shape=(N,H//8))
        work=np.lib.format.open_memmap(ctx.out/'work.npy',mode='w+',dtype=WORK,shape=(N,))
        parents=[];indexpath=ctx.out/'prefix_index.bin'
        with indexpath.open('xb') as physical:
            physical.write(HEADER.pack(b'M517P001',E,D,H,K,S,0,defect))
            physical.write(basis.tobytes());physical.write(gram.astype('<i8').tobytes())
            basisfloat=basis.astype('<f8')
            for e in range(E):
                at=np.flatnonzero(m[:,3]==e)
                w=ctx.weights(np,e)
                projected_float=w.astype('<f8') @ basisfloat
                assert np.array_equal(projected_float,np.rint(projected_float))
                assert np.max(np.abs(projected_float))<=D*128*S <2**31
                projected=projected_float.astype('<i4');p64=projected.astype('<i8')
                pw=np.sum(p64*p64,axis=1,dtype='<i8')
                if e:
                    wn=np.load(ctx.data(f'row_norm2_e{e:03}.npy'),allow_pickle=False).astype('<i8')
                else:
                    w32=w.astype('<i4');wn=np.sum(w32*w32,axis=1,dtype='<i8')
                assert np.all(pw<=2*S*S*wn)
                rw=wn*S**4-(S*S-defect)*pw
                assert np.all(rw>=0) and np.all(rw<=wn*S**4)
                roots=np.fromiter((isqrt(int(v)) for v in rw),dtype='<u4',count=H)
                physical.write(projected.tobytes());physical.write(pw.astype('<u8').tobytes())
                physical.write(rw.astype('<u8').tobytes());physical.write(roots.tobytes())
                candidate_total=certified=0
                for start in range(0,len(at),BATCH):
                    ix=at[start:start+BATCH]
                    transformed_float=q[ix].astype('<f8') @ basisfloat
                    assert np.array_equal(transformed_float,np.rint(transformed_float))
                    tq=transformed_float.astype('<i8')
                    pq=np.sum(tq*tq,axis=1,dtype='<i8')
                    assert np.all(pq>=0) and np.all(pq<=2*S*S*norms[ix])
                    central_float=transformed_float @ projected.astype('<f8').T
                    assert np.array_equal(central_float,np.rint(central_float))
                    assert np.max(np.abs(central_float)) < 2**53
                    central=central_float.astype('<i8')
                    rq=[int(n)*S**4-(S*S-defect)*int(p) for n,p in zip(norms[ix],pq)]
                    assert min(rq)>=0
                    rqroots=np.fromiter((isqrt(v) for v in rq),dtype='<u8',count=len(ix))
                    lower=roots.astype('<u8')[None,:]*rqroots[:,None]
                    threshold=lower//(S*S)+(lower%(S*S)!=0)
                    candidates=(central<=0)&((-central).astype('<u8')>=threshold)
                    reject=np.zeros(candidates.shape,dtype=bool)
                    for local in range(len(ix)):
                        for j in np.flatnonzero(candidates[local]):
                            upper=int(central[local,j])*S*S+defect*ceilroot(int(pw[j])*int(pq[local]))+ceilroot(int(rw[j])*rq[local])
                            reject[local,j]=upper<=0
                    skips=np.count_nonzero(reject,axis=1);width=H-skips
                    counts=np.count_nonzero(candidates,axis=1)
                    masks[ix]=np.packbits(reject,axis=1,bitorder='little')
                    rows=np.zeros(len(ix),dtype=WORK)
                    rows['width'],rows['skipped'],rows['candidates']=width,skips,counts
                    rows['addressed_bytes']=INTRODUCED+width*(D+4)
                    rows['query_norm2'],rows['projected_query_norm2']=norms[ix],pq
                    work[ix]=rows
                    modified=np.where(reject,np.float32(0),hidden[ix])
                    assert modified.tobytes()==hidden[ix].tobytes()
                    maximum=np.max(np.abs(modified),axis=1)
                    assert maximum.tobytes()==np.max(np.abs(hidden[ix]),axis=1).tobytes()
                    alpha=(maximum/np.float32(32767)).astype('<f4');alpha[maximum==0]=1
                    quantized=np.rint(modified/alpha[:,None]).clip(-32767,32767).astype('<i2')
                    assert alpha.tobytes()==scales[ix].tobytes() and quantized.tobytes()==codes[ix].tobytes()
                    certified+=int(skips.sum());candidate_total+=int(counts.sum());ctx.guard()
                parents.append({'parent':e,'UIDs':len(at),'candidates':candidate_total,'certified_cells':certified})
                if e%16==15:print(json.dumps({'main_parent_terminal':e,'seconds':ctx.resources()['seconds']}),flush=True)
        masks.flush();work.flush()
        indexbytes=indexpath.stat().st_size
        assert indexbytes==HEADER.size+D*K*2+K*K*8+E*RECORD_BYTES==58253352
        assert INTRODUCED==506076 and sum(v['UIDs'] for v in parents)==N
        ctx.r['gates'].update(ALL128_original_WI_integer_projection_radius_floor_root_physical=True,
                              ALL17540_exact_wide_integer_certificates_hidden_max_scale_A16_BYTE=True,
                              complete_fallback_ALL_domain_query_transform_candidate_and_byte_cost=True)
        views,books,rare,eligible=summaries(np,m,occ,work,norms,indexbytes)
        ctx.finish({'index_bytes':indexbytes,'original_fallback_bank_bytes':605945856,'integer_Gram_defect':defect,
             'rho':defect/(S*S),'basis_rank':K,'basis_scale':S,'parents':parents,'views':views,
             'consumed_books':books,'rare_consumed':rare,'eligibility':eligible,'tiny_controls_checked':tested,
             'baseline_WI_logical_bytes_per_UID':BASELINE,'introduced_logical_bytes_per_UID':INTRODUCED,
             'new_query_transform_MACs':N*D*K,'new_query_prefix_MACs':N*H*K,
             'original_WI_projection_MACs':E*H*D*K,'development_covariance_MACs':11721*D*D,
             'new_model_native_capture_calls':0,'old_full_WI_WO_model_evaluations':0,'physical_DRAM_verified':False,
             'decision':'NEXT_NATIVE_EXACT_PARITY_AND_COMPLETE_COST' if all(eligible.values()) else 'CLOSE_THIS_SHARED_RANK32_INTEGER_CERTIFICATE'})
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()

