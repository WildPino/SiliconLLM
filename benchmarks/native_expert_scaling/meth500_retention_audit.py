"""Independent original-source supports, masks, selector and partial integer outputs."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
from fractions import Fraction
import json
from pathlib import Path
import struct
from meth500_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);ap.add_argument('--raw-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth500_retention',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha);rp=DOC/'meth500_overlap_result.json';assert ctx.digest(rp)==a.raw_sha
        raw=json.loads(rp.read_bytes());assert all(raw['gates'].values());folder=ROOT/'results/native_expert_scaling/meth500_overlap'
        inventory=[]
        for v in raw['output_inventory']:ctx.exact(v);inventory.append(v)
        terminal=json.loads((folder/'terminal_resources.json').read_bytes());assert terminal['raw_sha256']==a.raw_sha
        ctx.r['gates']['complete_main_RAW_ALL_outputs_and_terminal_SHA']=True
        import numpy as np
        import threadpoolctl as T
        T.threadpool_limits(1);assert all(v['num_threads']==1 for v in T.threadpool_info())
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        N,D,H,C,B=17540,768,3072,4,1536
        def arr(name):return np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        hidden,codes,alpha=arr('hidden'),arr('hidden_codes'),arr('hidden_scales')
        masks,sm,owner,score,centres,hist=[arr(n) for n in ['masks','static_masks','mandatory_owner','mask_scores','centres','centroid_history']]
        actual,shadow,static,metrics=[arr(n) for n in ['child_actual','child_source_scale','static_actual','metrics']]
        ut=np.dtype([('m','<u4',(13,)),('hash','u1',(128,))]);it=np.dtype([('e','<u4'),('accept','<u4'),('alpha','<f4'),('x','<f4',(D,)),('q','<i2',(D,))])
        def wire(key,magic,width,reserved,dtype,count):
            p=Path(b['data'][key]['path']);assert p.stat().st_size==24+count*dtype.itemsize
            with p.open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
            return np.memmap(p,mode='r',offset=24,dtype=dtype,shape=(count,))
        u=wire('uid',b'M493U001',180,0,ut,N);m=u['m'];ni=wire('inputs',b'M499INP1',4624,D,np.dtype([('core',it),('p','<f4')]),N)
        x=ni['core'];p=ni['p'];ref=wire('targets',b'M499F001',3072,D,np.dtype(('<f4',(D,))),N)
        occ=wire('occurrences',b'M493O001',68,0,np.dtype(('<u4',(17,))),19962)
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(x['e'],m[:,3]) and (int(dev.sum()),int(val.sum()))==(11721,5819) and np.all(dev^val)
        assert np.array_equal(np.bincount(occ[:,1],minlength=N),m[:,7]) and np.all(occ[:,12]==1)
        assert hidden.shape==codes.shape==(N,H) and actual.shape==shadow.shape==(N,C,D) and static.shape==(N,D)
        assert masks.shape==(128,C,H) and np.all((masks==0)|(masks==1))
        payload=np.memmap(b['data']['payload']['path'],mode='r',dtype='u1');exp=json.loads((DOC/'meth380_switch_base128_export_result.json').read_bytes())
        def source(e,kind,shape):
            v=exp['tensors'][f'decoder.block.11.layer.2.mlp.experts.expert_{e}.{kind}.weight']
            assert v['shape']==list(shape)
            return np.frombuffer(payload,'i1',v['elements'],v['offset']).reshape(shape),np.frombuffer(payload,'<f4',shape[0],v['scale_offset'])
        def q16(h):
            maxima=np.amax(np.abs(h),axis=-1);s=np.divide(maxima,np.float32(32767),dtype=np.float32);s[maxima==0]=1
            return np.minimum(32767,np.maximum(-32767,np.rint(h/s[:,None]))).astype('<i2'),s
        def down(q,s,w,so):
            # Independent coding; exact integer products/sums bounded below 2^53.
            d=np.matmul(q.astype('<f8'),w.T.astype('<f8'))
            assert np.array_equal(d,np.trunc(d))
            return np.multiply(np.multiply(d,so.astype('<f8')),s.astype('<f8')[:,None]).astype('<f4')
        def normed(z):
            z=z.astype('<f8');n=np.sqrt(np.sum(z*z,axis=1));return np.divide(z,n[:,None],out=np.zeros_like(z),where=n[:,None]>0)
        # Independent tiny exact dot and complete-cover combinatorial witness.
        exact=sum(Fraction(v)*Fraction(w) for v,w in zip([32767,32767],[-128,127]));assert exact==-32767
        cq,ca=q16(np.array([[0,0],[1,.5]],'<f4'));assert cq.tolist()==[[0,0],[32767,16384]] and ca[0]==1
        fresh_rows=0;complete_cases=0;quantizer_changes=0
        ctx.phase='independent_ALL_source_integer_WI_hidden_quantization_partial_WO_and_selection'
        for e in range(128):
            at=np.flatnonzero(x['e']==e);di=at[dev[at]]
            if not len(at):assert e==0 and not masks[e].any() and not centres[e].any();continue
            wi,si=source(e,'wi',(H,D));wo,so=source(e,'wo',(D,H));mk=masks[e].astype(bool)
            assert np.all(mk.sum(axis=1)==B) and np.all(mk.any(axis=0)) and np.bincount(owner[e],minlength=C).tolist()==[H//C]*C
            assert np.all(mk[owner[e],np.arange(H)])
            z=normed(x['x'][di]);seeds=[0];distance=np.sum((z-z[0])**2,axis=1)
            for _ in range(3):
                avail=np.ones(len(z),bool);avail[seeds]=False
                k=int(np.argmax(np.where(avail,distance,-np.inf))) if avail.any() else 0
                seeds.append(k);distance=np.minimum(distance,np.sum((z-z[k])**2,axis=1))
            pc=z[seeds].copy();assert np.max(np.abs(pc-hist[e,0]))<=1e-12
            for iteration in range(8):
                labels=np.argmax(z@pc.T,axis=1)
                for c in range(C):
                    members=z[labels==c]
                    if len(members):pc[c]=normed(np.mean(members,axis=0)[None,:])[0]
                assert np.max(np.abs(pc-hist[e,iteration+1]))<=1e-12
            assert np.max(np.abs(pc-centres[e]))<=1e-12
            labels=np.argmax(z@centres[e].T,axis=1);hv=codes[di].astype('<f8')*alpha[di].astype('<f8')[:,None]
            norms=np.sum((wo.astype('<f8')*so.astype('<f8')[:,None])**2,axis=0)
            gs=np.mean(hv*hv,axis=0)*norms;ss=np.array([np.mean(hv[labels==c]**2,axis=0)*norms if np.any(labels==c) else gs for c in range(C)])
            assert ss.tobytes()==score[e].tobytes()
            order=sorted(range(C*H),key=lambda k:(-float(ss[k//H,k%H]),k%H,k//H))
            own=np.full(H,-1,'<i4');count=[0]*C;expected=np.zeros((C,H),bool)
            for k in order:
                c,j=divmod(k,H)
                if own[j]<0 and count[c]<H//C:own[j]=c;count[c]+=1;expected[c,j]=True
            assert own.tobytes()==owner[e].tobytes()
            for c in range(C):
                candidates=sorted(np.flatnonzero(~expected[c]).tolist(),key=lambda j:(-float(ss[c,j]),j))
                expected[c,candidates[:B-H//C]]=True
            assert expected.tobytes()==mk.tobytes()
            gj=sorted(range(H),key=lambda j:(-float(gs[j]),j))[:B];st=np.zeros(H,bool);st[gj]=True;assert st.tobytes()==sm[e].astype(bool).tobytes()
            selected=np.argmax(x['x'][at].astype('<f8')@centres[e].T,axis=1);assert np.array_equal(selected,metrics['selected'][at])
            for k in range(0,len(at),64):
                ix=at[k:k+64];up=down(x['q'][ix],x['alpha'][ix],wi,si);h=np.where(up<0,np.float32(0),up).astype('<f4')
                assert h.tobytes()==hidden[ix].tobytes();qh,ah=q16(h)
                assert qh.tobytes()==codes[ix].tobytes() and ah.tobytes()==alpha[ix].tobytes()
                original=down(qh,ah,wo,so);assert original.tobytes()==ref[ix].tobytes();fresh_rows+=len(ix)
                rr=ref[ix].astype('<f8');assert np.array_equal(np.sum(rr*rr,axis=1),metrics['reference_energy'][ix])
                assert np.array_equal(np.count_nonzero(qh,axis=1),metrics['support'][ix])
                for c in range(C):
                    j=np.flatnonzero(mk[c]);qc,ac=q16(h[:,j]);ff=down(qc,ac,wo[:,j],so);fs=down(qh[:,j],ah,wo[:,j],so)
                    assert ff.tobytes()==actual[ix,c].tobytes() and fs.tobytes()==shadow[ix,c].tobytes()
                    assert np.array_equal(np.sum((ff.astype('<f8')-rr)**2,axis=1),metrics['actual_error'][ix,c])
                    assert np.array_equal(np.sum((fs.astype('<f8')-rr)**2,axis=1),metrics['source_scale_error'][ix,c])
                    om=np.count_nonzero(qh[:,~mk[c]],axis=1);assert np.array_equal(om,metrics['omitted'][ix,c])
                    kept=np.amax(h[:,j],axis=1)==np.amax(h,axis=1);assert np.array_equal(kept,metrics['max_kept'][ix,c])
                    complete_cases+=int(np.sum(om==0));assert ff[om==0].tobytes()==ref[ix[om==0]].tobytes()
                    quantizer_changes+=int(np.sum(ac.view('<u4')!=ah.view('<u4')))
                sq,sa=q16(h[:,gj]);sf=down(sq,sa,wo[:,gj],so);assert sf.tobytes()==static[ix].tobytes()
                assert np.array_equal(np.sum((sf.astype('<f8')-rr)**2,axis=1),metrics['static_error'][ix])
                assert np.array_equal(np.argmin(metrics['actual_error'][ix],axis=1),metrics['oracle'][ix]);ctx.guard()
            ctx.log(completed_expert=e,source_rows=fresh_rows)
            if e%16==15:print(json.dumps({'audited_expert':e,'seconds':ctx.resources()['wall_seconds']}),flush=True)
        assert fresh_rows==N
        ctx.r['gates']['ALL17540_new_source_supports_original_F_BYTE_and_quantizers']=True
        ctx.r['gates']['ALL127_input_only_regions_dev_only_masks_union_width_and_selector']=True
        ctx.r['gates']['ALL157860_partial_outputs_and_energies_BYTE_independently_reconstructed']=True
        ctx.r['gates']['ALL_complete_support_identities_and_original_case_denominators']=True
        # Recompute every reported aggregate from independently validated rows.
        ix=np.arange(N);er=metrics['actual_error'];en=metrics['reference_energy'];sel=metrics['selected'];oracle=metrics['oracle']
        for name,view in raw['views'].items():
            base_name=name.removesuffix('_source_p')
            if base_name=='development':take=np.flatnonzero(dev);w=np.ones(N)
            elif base_name=='consumed':take=np.flatnonzero(val);w=np.ones(N)
            else:
                flag=(occ[:,7]==1)&(occ[:,6]==(1 if base_name=='natural_consumed' else 0));w=np.bincount(occ[flag,1],minlength=N).astype('<f8');take=np.flatnonzero(w)
            if name.endswith('_source_p'):w=w*p.astype('<f8')**2
            assert len(take)==view['UIDs'];den=float(np.sum(en[take]*w[take]))
            for label,values in [('routed',er[ix,sel]),('oracle',er[ix,oracle]),('static',metrics['static_error']),('routed_source_scale',metrics['source_scale_error'][ix,sel]),('oracle_source_scale',np.min(metrics['source_scale_error'],axis=1))]:
                num=float(np.sum(values[take]*w[take]));assert num==view[label]['error_energy'] and den==view[label]['reference_energy']
                assert float(np.sqrt(num/den))==view[label]['RMS']
                ratio=np.sqrt(np.divide(values[take],en[take],out=np.zeros(len(take)),where=en[take]!=0))
                assert float(np.quantile(ratio,.95))==view[label]['p95_UID_RMS'] and float(ratio.max())==view[label]['max_UID_RMS']
        v=raw['views'];expected={'oracle_all_consumed_RMS_le_2pct':v['consumed']['oracle']['RMS']<=.02,
            'routed_all_consumed_RMS_le_2pct':v['consumed']['routed']['RMS']<=.02,
            'oracle_natural_consumed_source_p_RMS_le_2pct':v['natural_consumed_source_p']['oracle']['RMS']<=.02,
            'routed_natural_consumed_source_p_RMS_le_2pct':v['natural_consumed_source_p']['routed']['RMS']<=.02,
            'routed_natural_consumed_p95_UID_RMS_le_5pct':v['natural_consumed']['routed']['p95_UID_RMS']<=.05}
        assert expected==raw['eligibility'];ctx.r['gates']['ALL_view_energy_RMS_p95_and_frozen_eligibility_independent']=True
        result=ctx.finish({'main_completed':True,'raw':{'path':str(rp),'sha256':a.raw_sha},'retained_inventory':inventory,
            'unique_inputs_audited':N,'partial_functions_audited':N*9,'new_source_FFN_rows_reconstructed':fresh_rows,
            'complete_support_child_cases':complete_cases,'child_quantizer_scale_changes':quantizer_changes,'eligibility':expected,'decision':raw['decision'],
            'scope':'Independent local apparatus and eligibility audit; no original-parent routing, fresh model quality or physical rate/DRAM qualification.'})
        print(json.dumps({'gates':result['gates'],'decision':result['decision'],'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as e:ctx.done.set();ctx.watchdog.cancel();ctx.fail(e);raise
if __name__=='__main__':main()
