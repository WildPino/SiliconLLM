"""ONE complete original-bank mask/selector screen; no full model or timing claim."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import struct
import traceback
from meth500_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth500_overlap',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha)
        import numpy as np
        import threadpoolctl as T
        import meth500_overlap_math as M
        T.threadpool_limits(1);pools=T.threadpool_info();assert pools and all(v['num_threads']==1 for v in pools)
        for v in pools:assert any(d['path']==str(Path(v['filepath']).resolve()) for d in b['catalog'])
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        ctx.r['runtime']={'pools':pools,'affinity':[0],'module':str(Path(np.__file__).resolve())}
        ctx.phase='NEW_four_axis_union_and_integer_controls';write(ctx.out/'controls.json',M.controls())
        ctx.r['gates']['NEW_fixed_mask_region_A16_integer_controls']=True
        N=17540;D=768;H=3072;C=4;B=1536
        ut=np.dtype([('m','<u4',(13,)),('hash','u1',(128,))])
        it=np.dtype([('e','<u4'),('accept','<u4'),('alpha','<f4'),('x','<f4',(D,)),('q','<i2',(D,))])
        def wire(key,magic,width,reserved,dtype,count):
            p=Path(b['data'][key]['path']);assert p.stat().st_size==24+count*dtype.itemsize
            with p.open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
            return np.memmap(p,mode='r',offset=24,dtype=dtype,shape=(count,))
        uid=wire('uid',b'M493U001',180,0,ut,N);m=uid['m']
        ni=wire('inputs',b'M499INP1',4624,D,np.dtype([('core',it),('p','<f4')]),N);x=ni['core'];p=ni['p']
        ref=wire('targets',b'M499F001',3072,D,np.dtype(('<f4',(D,))),N)
        occ=wire('occurrences',b'M493O001',68,0,np.dtype(('<u4',(17,))),19962)
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert (int(dev.sum()),int(val.sum()))==(11721,5819) and not np.any(dev&val) and np.all(dev|val)
        assert np.array_equal(m[:,0],np.arange(N)) and np.all(m[:,2]==11) and np.all(x['accept']==1)
        assert np.array_equal(x['e'],m[:,3]) and p.tobytes()==m[:,10].tobytes()
        for k in range(0,N,128):
            q,alpha=M.quant(x['x'][k:k+128]);assert q.tobytes()==x['q'][k:k+128].tobytes() and alpha.tobytes()==x['alpha'][k:k+128].tobytes();ctx.guard()
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,1]<N) and np.all(occ[:,8]==11) and np.all(occ[:,12]==1)
        assert np.array_equal(np.bincount(occ[:,1],minlength=N),m[:,7])
        for oi,mi in [(14,1),(11,3),(15,12),(16,10)]:assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        ctx.r['gates']['ALL_original_UID_input_mass_A16_role_occurrence_contracts']=True
        export=json.loads((DOC/'meth380_switch_base128_export_result.json').read_bytes());entries=export['tensors']
        payload=np.memmap(b['data']['payload']['path'],mode='r',dtype='u1')
        def weights(e):
            ans=[]
            for kind,shape in [('wi',(H,D)),('wo',(D,H))]:
                v=entries[f'decoder.block.11.layer.2.mlp.experts.expert_{e}.{kind}.weight']
                assert v['shape']==list(shape) and v['encoding']==1
                w=np.frombuffer(payload,dtype='i1',count=v['elements'],offset=v['offset']).reshape(shape)
                s=np.frombuffer(payload,dtype='<f4',count=shape[0],offset=v['scale_offset'])
                import hashlib
                assert hashlib.sha256(w.tobytes()).hexdigest()==v['sha256'] and hashlib.sha256(s.tobytes()).hexdigest()==v['scale_sha256']
                ans.extend((w,s))
            return ans
        def output(name,shape,dtype):return np.lib.format.open_memmap(ctx.out/name,mode='w+',dtype=dtype,shape=shape)
        hidden=output('hidden.npy',(N,H),'<f4');codes=output('hidden_codes.npy',(N,H),'<i2');scales=output('hidden_scales.npy',(N,),'<f4')
        actual=output('child_actual.npy',(N,C,D),'<f4');shadow=output('child_source_scale.npy',(N,C,D),'<f4');static=output('static_actual.npy',(N,D),'<f4')
        masks=output('masks.npy',(128,C,H),'u1');maskstatic=output('static_masks.npy',(128,H),'u1')
        owners=output('mandatory_owner.npy',(128,H),'<i4');scores=output('mask_scores.npy',(128,C,H),'<f8')
        centres=output('centres.npy',(128,C,D),'<f8');history=output('centroid_history.npy',(128,9,C,D),'<f8')
        mt=np.dtype([('selected','<u4'),('oracle','<u4'),('support','<u4'),('omitted','<u4',(C,)),('max_kept','u1',(C,)),
                     ('source_scale_error','<f8',(C,)),('actual_error','<f8',(C,)),('reference_energy','<f8'),('static_error','<f8')])
        metrics=output('metrics.npy',(N,),mt);rows=[];source_calls=0
        ctx.phase='ALL128_original_supports_and_development_only_region_masks'
        for e in range(128):
            at=np.flatnonzero(x['e']==e);di=at[dev[at]];vi=at[val[at]];wi,si,wo,so=weights(e)
            if not len(at):
                assert e==0 and not len(di);masks[e]=0;maskstatic[e]=0;owners[e]=-1;scores[e]=0;centres[e]=0;history[e]=0
                rows.append({'expert':e,'development':0,'consumed':0,'status':'EMPTY_UNIDENTIFIED_NO_PROMOTION'});continue
            assert len(di)>0
            for k in range(0,len(at),128):
                ix=at[k:k+128];up=M.projection(x['q'][ix],x['alpha'][ix],wi,si)
                h=np.where(up<0,np.float32(0),up).astype('<f4');qh,ah=M.quant(h);f=M.projection(qh,ah,wo,so)
                assert f.tobytes()==ref[ix].tobytes(),('original_source_contract',e,k)
                hidden[ix]=h;codes[ix]=qh;scales[ix]=ah;source_calls+=len(ix);ctx.guard()
            pc,labels,seeds,states=M.regions(x['x'][di],di)
            hdev=codes[di].astype('<f8')*scales[di].astype('<f8')[:,None]
            ss,gs=M.scores(hdev,labels,wo,so);mk,own=M.masks(ss)
            gidx=np.lexsort((np.arange(H),-gs))[:B];sm=np.zeros(H,bool);sm[gidx]=True
            masks[e]=mk;maskstatic[e]=sm;owners[e]=own;scores[e]=ss;centres[e]=pc;history[e]=states
            selected=np.argmax(x['x'][at].astype('<f8')@pc.T,axis=1);metrics['selected'][at]=selected
            for k in range(0,len(at),128):
                ix=at[k:k+128];hc=hidden[ix];qh=codes[ix];ah=scales[ix];rr=ref[ix].astype('<f8')
                metrics['support'][ix]=np.count_nonzero(qh,axis=1);metrics['reference_energy'][ix]=np.sum(rr*rr,axis=1)
                for c in range(C):
                    j=np.flatnonzero(mk[c]);cj,ca=M.quant(hc[:,j])
                    ff=M.projection(cj,ca,wo[:,j],so);sf=M.projection(qh[:,j],ah,wo[:,j],so)
                    actual[ix,c]=ff;shadow[ix,c]=sf
                    metrics['actual_error'][ix,c]=np.sum((ff.astype('<f8')-rr)**2,axis=1)
                    metrics['source_scale_error'][ix,c]=np.sum((sf.astype('<f8')-rr)**2,axis=1)
                    metrics['omitted'][ix,c]=np.count_nonzero(qh[:,~mk[c]],axis=1)
                    metrics['max_kept'][ix,c]=(np.max(hc[:,j],axis=1)==np.max(hc,axis=1))
                    complete=metrics['omitted'][ix,c]==0
                    assert ff[complete].tobytes()==ref[ix[complete]].tobytes(),('complete_source_support_identity',e,c,k)
                sj,sa=M.quant(hc[:,gidx]);sf=M.projection(sj,sa,wo[:,gidx],so);static[ix]=sf
                metrics['static_error'][ix]=np.sum((sf.astype('<f8')-rr)**2,axis=1)
                metrics['oracle'][ix]=np.argmin(metrics['actual_error'][ix],axis=1);ctx.guard()
            rows.append({'expert':e,'development':len(di),'consumed':len(vi),'seed_UIDs':di[seeds].tolist(),
                         'development_cluster_sizes':np.bincount(labels,minlength=C).tolist(),'mask_widths':mk.sum(axis=1).tolist(),
                         'union_width':int(mk.any(axis=0).sum()),'multiplicity_histogram':np.bincount(mk.sum(axis=0),minlength=5).tolist(),
                         'selector_MULs':C*D,'selected_FFN_MACs':2*D*B})
            ctx.log(completed_expert=e,source_FFN_rows=source_calls);print(json.dumps({'expert':e,'rows':len(at),'seconds':ctx.resources()['wall_seconds']}),flush=True)
        assert source_calls==N and len(rows)==128
        ctx.r['gates']['ALL17540_new_hidden_supports_original_saved_F_BYTE']=True
        ctx.r['gates']['ALL127_development_only_fixed_regions_overlapping_masks_and_union3072']=True
        ctx.r['gates']['ALL_complete_retained_nonzero_support_child_functions_BYTE_identity']=True
        for v in [hidden,codes,scales,actual,shadow,static,masks,maskstatic,owners,scores,centres,history,metrics]:v.flush()
        ctx.phase='all_original_views_and_error_decomposition'
        idx=np.arange(N);selected=metrics['selected'];oracle=metrics['oracle'];er=metrics['actual_error'];es=metrics['source_scale_error'];en=metrics['reference_energy']
        uid_weights={name:np.ones(N) for name in ['development','consumed']}
        def stats(take,weight):
            ans={};den=float(np.sum(en[take]*weight[take]));assert den>0
            for name,values in [('routed',er[idx,selected]),('oracle',er[idx,oracle]),('static',metrics['static_error']),('routed_source_scale',es[idx,selected]),('oracle_source_scale',np.min(es,axis=1))]:
                num=float(np.sum(values[take]*weight[take]));ratio=np.sqrt(np.divide(values[take],en[take],out=np.zeros(len(take)),where=en[take]!=0))
                ans[name]={'RMS':float(np.sqrt(num/den)),'error_energy':num,'reference_energy':den,'p95_UID_RMS':float(np.quantile(ratio,.95)),'max_UID_RMS':float(np.max(ratio)),
                           'zero_ref_nonzero_error':int(np.sum((en[take]==0)&(values[take]>0)))}
            ans['UIDs']=len(take);ans['weight_sum']=float(weight[take].sum())
            ans['routed_complete_support_count']=int(np.sum(metrics['omitted'][take,selected[take]]==0))
            ans['any_child_complete_support_count']=int(np.sum(np.any(metrics['omitted'][take]==0,axis=1)))
            ans['source_support_mean']=float(np.mean(metrics['support'][take]));ans['source_support_max']=int(np.max(metrics['support'][take]))
            return ans
        views={}
        for role,take in [('development',np.flatnonzero(dev)),('consumed',np.flatnonzero(val))]:
            views[role]=stats(take,np.ones(N));views[role+'_source_p']=stats(take,p.astype('<f8')**2)
        for name,flag in [('natural_consumed',(occ[:,7]==1)&(occ[:,6]==1)),('teacher_consumed',(occ[:,7]==1)&(occ[:,6]==0))]:
            weight=np.bincount(occ[flag,1],minlength=N).astype('<f8');take=np.flatnonzero(weight)
            views[name]=stats(take,weight);views[name+'_source_p']=stats(take,weight*p.astype('<f8')**2)
        rare={};development_counts=np.bincount(x['e'][dev],minlength=128)[x['e']]
        for name,flag in [('dev_count1_4',(development_counts>0)&(development_counts<=4)),
                          ('dev_count5_15',(development_counts>=5)&(development_counts<=15))]:
            take=np.flatnonzero(flag&val);rare[name]=stats(take,np.ones(N)) if len(take) else {'UIDs':0}
        eligibility={'oracle_all_consumed_RMS_le_2pct':views['consumed']['oracle']['RMS']<=.02,
                     'routed_all_consumed_RMS_le_2pct':views['consumed']['routed']['RMS']<=.02,
                     'oracle_natural_consumed_source_p_RMS_le_2pct':views['natural_consumed_source_p']['oracle']['RMS']<=.02,
                     'routed_natural_consumed_source_p_RMS_le_2pct':views['natural_consumed_source_p']['routed']['RMS']<=.02,
                     'routed_natural_consumed_p95_UID_RMS_le_5pct':views['natural_consumed']['routed']['p95_UID_RMS']<=.05}
        decision='NEXT_NATIVE_AND_FRESH_ELIGIBILITY' if all(eligibility.values()) else ('RECONSIDER_MASK_GEOMETRY' if not eligibility['oracle_all_consumed_RMS_le_2pct'] else 'RECONSIDER_INPUT_SELECTOR')
        write(ctx.out/'summary.json',{'views':views,'rare_consumed':rare,'eligibility':eligibility,'decision':decision})
        result=ctx.finish({'experts':rows,'views':views,'rare_consumed':rare,'eligibility':eligibility,'decision':decision,
            'source_FFN_rows_reconstructed':source_calls,'partial_source_outputs':N*9,'model_native_GPU_calls':0,'gradient_updates':0,
            'stored_FFN_redundancy':2,'active_FFN_fraction':.5,'selector_MACs_per_call':3072,
            'scope':'ONE bank original-domain diagnostic; source parent ID/p oracle fixed. No head/composed/fresh/rate/DRAM/LUT/general-n or whole-donor-quality claim.'})
        print(json.dumps({'gates':result['gates'],'eligibility':eligibility,'decision':decision,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as e:ctx.done.set();ctx.watchdog.cancel();ctx.fail(e);raise
if __name__=='__main__':main()
