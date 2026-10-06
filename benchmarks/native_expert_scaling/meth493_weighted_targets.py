"""All source bank11 identities; no model, optimization, spectrum or selection."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import traceback
from meth493_operations import Context, ROOT, DOC, write

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--binding-sha',required=True)
    args=ap.parse_args(); ctx=Context(ROOT/'results/native_expert_scaling/meth493_weighted_targets',Path(args.out).resolve(),180,256<<20)
    try:
        binding=ctx.admit(args.binding_sha); ctx.binding=binding
        import numpy as np
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=1); ctx.r['numerical_imports']=True
        assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
        def product(pb,fb,accepted):
            if not accepted: return 0
            x=np.array([pb,fb],dtype='<u4').view('<f4')
            return int(np.multiply(x[0],x[1],dtype=np.float32).view('<u4'))
        fixtures=[(0x3f000000,0x80000000,1,0x80000000),
            (0x3f000000,1,1,0),(0x3f000000,3,1,2),(0x3f000000,0x80000001,1,0x80000000),
            (0x3f000000,0x80000003,1,0x80000002),(0x3f000000,0x01000000,1,0x00800000),
            (0x3f000000,0x00ffffff,1,0x00800000),(0x3f800000,0x7f7fffff,1,0x7f7fffff),
            (0x3f000001,0x3fc00000,1,0x3f400002),(0x3f000003,0x3fc00000,1,0x3f400004),
            (0x3f000000,0x80000003,0,0)]
        controls=[]
        for pb,fb,accepted,expected in fixtures:
            actual=product(pb,fb,accepted); assert actual==expected
            controls.append({'p_bits':pb,'F_bits':fb,'accepted':accepted,'expected_bits':expected,'actual_bits':actual})
        write(ctx.out/'rounding_controls.json',controls)
        ctx.r['gates']['eleven_fixed_RNE_signed_zero_subnormal_boundary_and_rejection_controls']=True
        U,Q,N,T=238872,387036,17540,19962
        ctx.phase='full_original_UID_metadata_ownership_and_occurrence_join'
        def readwire(name,magic,width,reserved,count,dtype):
            stream=Path(binding['data'][name]['path']).open('rb')
            assert stream.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
            for start in range(0,count,1024):
                data=stream.read(min(1024,count-start)*width); assert len(data)==min(1024,count-start)*width
                yield start,np.frombuffer(data,dtype); ctx.guard()
            assert not stream.read(1); stream.close()
        dtype=np.dtype([('meta','<u4',(12,)),('hashes','u1',(64,)),('value','<f8',(3,)),('input','<f4',(768,)),('score','<f4',(128,))])
        meta=np.empty((U,12),'<u4'); pbits=np.empty(U,'<u4')
        for start,part in readwire('unique_inputs.bin',b'M479UNI1',3720,768,U,dtype):
            meta[start:start+len(part)]=part['meta']; p=part['value'][:,1].astype('<f4')
            assert p.astype('<f8').tobytes()==part['value'][:,1].tobytes()
            pbits[start:start+len(part)]=p.view('<u4')
        del part,p
        owner=np.empty((U,8),'<u4'); links=np.empty((Q,13),'<u4')
        for start,part in readwire('ownership.bin',b'M479OWN1',32,0,U,np.dtype(('<u4',(8,)))): owner[start:start+len(part)]=part
        for start,part in readwire('query_links.bin',b'M479LNK1',52,0,Q,np.dtype(('<u4',(13,)))): links[start:start+len(part)]=part
        assert np.array_equal(meta[:,0],np.arange(U)) and np.array_equal(owner[:,0],meta[:,0])
        assert np.array_equal(owner[:,1],meta[:,2]) and np.array_equal(links[:,0],np.arange(Q)) and np.all(links[:,12]<U)
        assert np.array_equal(links[:,6],meta[links[:,12],2]) and np.array_equal(links[:,9],meta[links[:,12],7])
        uid=np.flatnonzero(meta[:,2]==11); selected=np.flatnonzero(links[:,6]==11)
        assert len(uid)==N and len(selected)==T and np.all(meta[uid,7]<128)
        assert np.all((owner[uid,2]&5!=0) ^ (owner[uid,2]&2!=0))
        assert (int(np.sum(owner[uid,2]&5!=0)),int(np.sum(owner[uid,2]&2!=0)))==(11721,5819)
        local=np.full(U,-1,'<i4'); local[uid]=np.arange(N)
        info=np.zeros((N,180),dtype='u1'); seen=np.zeros(N,dtype=bool); counts=np.zeros(N,'<u4')
        rows=info[:,:52].copy().view('<u4').reshape(N,13)
        rows[:,:11]=np.column_stack((np.arange(N),uid,meta[uid,2],meta[uid,7],owner[uid,2:8],pbits[uid]))
        # Fields11/12 are first qid and F32 alpha bits; remaining128bytes are four raw SHA digests.
        targetpath=ctx.out/'weighted_targets.bin'; info_path=ctx.out/'uid_info.bin'; occurrencepath=ctx.out/'occurrences.bin'
        targets=targetpath.open('x+b'); targets.write(struct.pack('<8sIIQ',b'M493Y001',3072,768,N)); targets.truncate(24+N*3072)
        occurrence=occurrencepath.open('xb'); occurrence.write(struct.pack('<8sIIQ',b'M493O001',68,0,T))
        query=Path(binding['weighted_source']['query_inputs.npy']['path']).open('rb')
        ref=Path(binding['weighted_source']['reference_functions.npy']['path']).open('rb')
        qtype=np.dtype([('ledger_record','<u8'),('book','<u2'),('case','u1'),('mode','u1'),('role','u1'),('accepted','u1'),
            ('expert','<u2'),('index','<u4'),('alpha','<f4'),('probability','<f4'),('input','<f4',(768,)),('codes','<i2',(768,)),
            ('input_sha','u1',(32,)),('code_sha','u1',(32,)),('pair_sha','u1',(32,))])
        assert qtype.itemsize==4732
        assert query.read(8)==ref.read(8)==b'\x93NUMPY\x01\x00'
        import ast
        qh=ast.literal_eval(query.read(struct.unpack('<H',query.read(2))[0]).decode('ascii'))
        fh=ast.literal_eval(ref.read(struct.unpack('<H',ref.read(2))[0]).decode('ascii'))
        expected_descr=np.lib.format.dtype_to_descr(qtype)[:12]+[(name,'|S32') for name in ('input_sha','code_sha','pair_sha')]
        assert qh=={'descr':expected_descr,'fortran_order':False,'shape':(T,)}
        assert fh=={'descr':'<f4','fortran_order':False,'shape':(T,768)} and query.tell()==448 and ref.tell()==128
        canonical=Path(binding['data']['unique_inputs.bin']['path']).open('rb')
        source=Path(binding['data']['source_fields.bin']['path']).open('rb')
        ledger=Path(binding['weighted_source']['ledger']['path']).open('rb')
        assert ledger.read(20)==struct.pack('<8sIQ',b'MQ471L01',118,Q)
        lw=struct.Struct('<H6BHIff32s32s32s'); counts_by_view=np.zeros((192,2,2),'<u4')
        ctx.phase='ALL19962_source_BYTE_joins_and_F32_targets'
        for qid in range(T):
            qb=query.read(4732); fb=ref.read(3072); assert len(qb)==4732 and len(fb)==3072
            q=np.frombuffer(qb,qtype)[0]; li=int(q['ledger_record']); link=links[li]
            assert li==int(selected[qid]) and link[6]==11
            assert tuple(map(int,link[[1,2,3,4,5,9,8,10]]))==(128,int(q['book']),int(q['case']),int(q['mode']),int(q['role']),int(q['expert']),int(q['index']),int(q['accepted']))
            assert int(q['role'])==(0 if q['book']<64 else 1 if q['book']<128 else 2)
            source.seek(24+72*li); sb=source.read(72); assert sb[:48]==link[:12].tobytes()
            pb=q['probability'].tobytes(); ab=q['alpha'].tobytes()
            assert struct.pack('<f',struct.unpack_from('<d',sb,56)[0])==pb
            assert 0 < int(q['probability'].view('<u4')) <= 0x3f800000 and np.isfinite(q['alpha']) and q['alpha']>0
            ledger.seek(20+118*li); lb=ledger.read(118)
            n,mode,book,case,bank,accepted,role,expert,index,alpha,p,hi,hc,hp=lw.unpack(lb)
            assert (n,mode,book,case,bank,accepted,role,expert,index)==(128,int(q['mode']),int(q['book']),int(q['case']),11,int(q['accepted']),int(q['role']),int(q['expert']),int(q['index']))
            assert lb[14:18]==ab and lb[18:22]==pb
            ib=q['input'].tobytes(); cb=q['codes'].tobytes()
            assert hi==q['input_sha'].tobytes()==hashlib.sha256(ib).digest()
            assert hc==q['code_sha'].tobytes()==hashlib.sha256(cb).digest()
            assert hp==q['pair_sha'].tobytes()==hashlib.sha256(cb+ab).digest()
            globaluid=int(link[12]); k=int(local[globaluid]); assert k>=0
            canonical.seek(24+3720*globaluid); ub=canonical.read(3720)
            assert ub[48:80]==hi and ub[136:3208]==ib
            assert struct.pack('<f',struct.unpack_from('<d',ub,120)[0])==pb
            assert meta[globaluid,7]==expert and int(pbits[globaluid])==int(q['probability'].view('<u4'))
            f=np.frombuffer(fb,'<f4'); assert np.isfinite(f).all()
            # The actual whole bank11 is accepted. A different control domain must fail, never be silently narrowed.
            assert accepted==1 and owner[globaluid,4]==2
            y=np.multiply(f,q['probability'],dtype=np.float32); assert np.isfinite(y).all(); yb=y.tobytes()
            hashes=hi+hc+hp+hashlib.sha256(fb).digest()
            targets.seek(24+3072*k)
            if seen[k]:
                assert targets.read(3072)==yb, ('duplicate_target_BYTE_contradiction',qid,globaluid)
                assert info[k,52:].tobytes()==hashes and rows[k,12]==q['alpha'].view('<u4'), ('duplicate_source_BYTE_contradiction',qid,globaluid)
            else:
                targets.write(yb); info[k,52:]=np.frombuffer(hashes,'u1'); rows[k,11:]=[qid,q['alpha'].view('<u4')]; seen[k]=True
            occurrence.write(np.asarray([qid,k,*link,int(q['alpha'].view('<u4')),int(q['probability'].view('<u4'))],'<u4').tobytes())
            counts[k]+=1; counts_by_view[book,mode,accepted]+=1
            if qid%256==0: ctx.guard()
        assert not query.read(1) and not ref.read(1)
        for stream in (query,ref,canonical,source,ledger,targets,occurrence): stream.close()
        assert seen.all() and np.array_equal(counts,owner[uid,5])
        info[:,:52]=rows.astype('<u4').view('u1').reshape(N,52)
        with info_path.open('xb') as stream:
            stream.write(struct.pack('<8sIIQ',b'M493U001',180,0,N)); stream.write(info.tobytes())
        ctx.r['gates']['ALL17540_UID_and19962_occurrence_input_p_A16_hash_alpha_role_expert_and_control_BYTE_joins']=True
        ctx.r['gates']['ALL_duplicate_UID_reference_and_target_BYTES_equal_no_averaging']=True
        cells=[]
        for expert in range(128):
            at=uid[meta[uid,7]==expert]
            cells.append({'expert':expert,'unique_count':len(at),'development':int(np.sum(owner[at,2]&5!=0)),
                'consumed_validation':int(np.sum(owner[at,2]&2!=0)),'occurrences':int(owner[at,5].sum())})
        views=[{'book':book,'role':0 if book<64 else 1 if book<128 else 2,'mode':mode,'accepted':accepted,
                'count':int(counts_by_view[book,mode,accepted])} for book in range(192) for mode in range(2) for accepted in range(2)]
        assert sum(v['unique_count'] for v in cells)==N and sum(v['occurrences'] for v in cells)==sum(v['count'] for v in views)==T
        ctx.r['gates']['ALL128_cells_768_book_mode_control_views_and_denominators_preserved']=True
        result=ctx.finish({'unique_inputs':N,'development_inputs':11721,'consumed_validation_inputs':5819,'occurrences':T,
            'target_coordinates':N*768,'occurrence_product_coordinates':T*768,'duplicate_occurrences':T-N,
            'accepted_occurrences':T,'rejected_occurrences':0,'cells':cells,'views':views,
            'native_calls':0,'optimizer_updates':0,'SVD_calls':0,'decision':'FULL_BANK11_WEIGHTED_SOURCE_TARGETS_COMPILED_PENDING_INDEPENDENT_AUDIT',
            'scope':'Source supervision only. No functional student, fresh quality, physical cost or goal completion.'})
        print(json.dumps({'gates':result['gates'],'unique_inputs':N,'occurrences':T,'resource':ctx.terminal_resources['resource']}))
    except BaseException as exc:
        ctx.fail(exc); print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True); raise

if __name__=='__main__': main()
