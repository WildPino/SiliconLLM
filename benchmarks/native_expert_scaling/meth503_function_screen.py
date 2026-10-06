"""One full-original-domain inquiry: new variable masks, oracle and input selector."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse,hashlib,json,struct
from pathlib import Path
from meth503_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth503_functions',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha)
        import numpy as np
        import threadpoolctl as T
        import meth503_function_math as M
        T.threadpool_limits(1);pools=T.threadpool_info();assert pools and all(v['num_threads']==1 for v in pools)
        for v in pools:assert any(d['path']==str(Path(v['filepath']).resolve()) for d in b['catalog'])
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        ctx.r['runtime']={'pools':pools,'affinity':[0]};write(ctx.out/'controls.json',M.controls())
        ctx.r['gates']['NEW_exact_integer_A16_centroid_unsupported_tie_controls']=True
        N,D,H=17540,768,3072
        def wire(key,magic,width,reserved,dtype,count):
            p=Path(b['data'][key]['path']);dtype=np.dtype(dtype);assert p.stat().st_size==24+count*dtype.itemsize
            with p.open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
            return np.memmap(p,mode='r',offset=24,dtype=dtype,shape=(count,))
        m=wire('uid',b'M493U001',180,0,[('m','<u4',(13,)),('hash','u1',(128,))],N)['m']
        it=np.dtype([('e','<u4'),('accept','<u4'),('alpha','<f4'),('x','<f4',(D,)),('q','<i2',(D,))])
        ni=wire('inputs',b'M499INP1',4624,D,[('core',it),('p','<f4')],N);x=ni['core'];p=ni['p']
        ref=wire('targets',b'M499F001',3072,D,np.dtype(('<f4',(D,))),N)
        occ=wire('occurrences',b'M493O001',68,0,np.dtype(('<u4',(17,))),19962)
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(N)) and np.all(m[:,2]==11) and (int(dev.sum()),int(val.sum()))==(11721,5819) and np.all(dev^val)
        assert np.array_equal(x['e'],m[:,3]) and np.all(x['accept']==1) and p.tobytes()==m[:,10].tobytes()
        assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,12]==1) and np.all(occ[:,8]==11) and np.array_equal(np.bincount(occ[:,1],minlength=N),m[:,7])
        for oi,mi in [(14,1),(11,3),(15,12),(16,10)]:assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        def arr(k):return np.load(b['data'][k]['path'],mmap_mode='r',allow_pickle=False)
        hidden,codes,alpha,assignment,diag=[arr(k) for k in ['hidden','codes','scales','assignment','diagnostics']]
        assert hidden.shape==codes.shape==(N,H) and hidden.dtype==np.dtype('<f4') and codes.dtype==np.dtype('<i2') and np.all(hidden>=0)
        for k in range(0,N,128):
            q,s=M.quant(hidden[k:k+128]);assert q.tobytes()==codes[k:k+128].tobytes() and s.tobytes()==alpha[k:k+128].tobytes();ctx.guard()
        ctx.r['gates']['ALL_original_UID_roles_occurrences_parent_mass_saved_hidden_A16_contracts']=True
        entries=json.loads((DOC/'meth380_switch_base128_export_result.json').read_bytes())['tensors'];payload=np.memmap(b['data']['payload']['path'],mode='r',dtype='u1')
        geometry=json.loads(Path(b['data']['raw502']['path']).read_bytes())
        mt=np.dtype([('selected','<u4'),('oracle','<u4'),('reference_energy','<f8'),('routed_error','<f8'),('oracle_error','<f8'),('routed_omitted','<u4'),('any_covered','u1')])
        metrics=np.lib.format.open_memmap(ctx.out/'metrics.npy',mode='w+',shape=(N,),dtype=mt)
        rows=[];evaluations=complete_cases=unsupported=0;ctx.phase='ALL127_new_actual_functions_and_development_only_centroids'
        for e in range(128):
            at=np.flatnonzero(m[:,3]==e);di=at[dev[at]];C=geometry['experts'][e]['children']
            if not len(at):assert e==0 and C==0;rows.append({'expert':e,'status':'EMPTY_UNCOMPILED_UNPROMOTED'});continue
            mask=np.load(b['masks'][str(e)]['path'],mmap_mode='r');assert mask.shape==(C,H) and np.all(mask.sum(axis=1)<=1536) and np.all(mask.any(axis=0))
            pc,count=M.prototypes(x['x'][di],assignment[di],C);score=x['x'][at].astype('<f8')@pc.T;score[:,count==0]=-np.inf;sel=np.argmax(score,axis=1)
            np.save(ctx.out/f'centres_e{e:03}.npy',pc);assert np.all(count[sel]>0)
            v=entries[f'decoder.block.11.layer.2.mlp.experts.expert_{e}.wo.weight'];assert v['shape']==[D,H] and v['encoding']==1
            wo=np.frombuffer(payload,'i1',v['elements'],v['offset']).reshape(D,H);so=np.frombuffer(payload,'<f4',D,v['scale_offset'])
            assert hashlib.sha256(wo.tobytes()).hexdigest()==v['sha256'] and hashlib.sha256(so.tobytes()).hexdigest()==v['scale_sha256']
            actual=np.lib.format.open_memmap(ctx.out/f'actual_e{e:03}.npy',mode='w+',shape=(len(at),C,D),dtype='<f4')
            errors=np.lib.format.open_memmap(ctx.out/f'errors_e{e:03}.npy',mode='w+',shape=(len(at),C),dtype='<f8')
            omitted=np.column_stack([np.count_nonzero(codes[at][:,mask[c]==0],axis=1) for c in range(C)])
            for c in range(C):
                j=np.flatnonzero(mask[c]);wc=wo[:,j]
                for k in range(0,len(at),64):
                    ix=at[k:k+64];q,s=M.quant(hidden[ix][:,j]);ff=M.project(q,s,wc,so);actual[k:k+len(ix),c]=ff
                    errors[k:k+len(ix),c]=np.sum((ff.astype('<f8')-ref[ix].astype('<f8'))**2,axis=1)
                    complete=omitted[k:k+len(ix),c]==0;assert ff[complete].tobytes()==ref[ix[complete]].tobytes();complete_cases+=int(complete.sum());evaluations+=len(ix);ctx.guard()
            oracle=np.argmin(errors,axis=1);en=np.sum(ref[at].astype('<f8')**2,axis=1)
            metrics['selected'][at]=sel;metrics['oracle'][at]=oracle;metrics['reference_energy'][at]=en
            metrics['routed_error'][at]=errors[np.arange(len(at)),sel];metrics['oracle_error'][at]=errors[np.arange(len(at)),oracle]
            metrics['routed_omitted'][at]=omitted[np.arange(len(at)),sel];metrics['any_covered'][at]=np.any(omitted==0,axis=1)
            assert np.array_equal(metrics['any_covered'][at],diag['any'][at])
            rows.append({'expert':e,'development':len(di),'consumed':int(val[at].sum()),'children':C,'development_group_counts':count.tolist(),
                'unsupported_selector_children':int(np.sum(count==0)),'selector_MULs':D*C,'active_width_ceiling':1536})
            unsupported+=int(np.sum(count==0));actual.flush();errors.flush();del actual,errors
            ctx.log(expert=e,child_functions=evaluations)
            if e%16==15:print(json.dumps({'expert':e,'seconds':ctx.resources()['wall_seconds']}),flush=True)
        metrics.flush();assert evaluations==127570
        ctx.r['gates']['ALL_fixed_masks_development_only_input_prototypes_and_selection_no_consumed_fit']=True
        ctx.r['gates']['ALL127570_actual_child_functions_original_I8_WO_own_A16_and_complete_support_BYTE_identity']=True
        ctx.phase='ALL_original_views_best_child_bound_and_frozen_eligibility'
        def stats(take,w):
            en=metrics['reference_energy'];den=float(np.sum(en[take]*w[take]));assert den>0;result={}
            for label,key in [('routed','routed_error'),('oracle','oracle_error')]:
                er=metrics[key];num=float(np.sum(er[take]*w[take]));zero=int(np.sum((en[take]==0)&(er[take]>0)))
                ratio=np.sqrt(np.divide(er[take],en[take],out=np.zeros(len(take)),where=en[take]!=0))
                result[label]={'RMS':float(np.sqrt(num/den)),'error_energy':num,'reference_energy':den,'p95_UID_RMS':None if zero else float(np.quantile(ratio,.95)),
                    'max_UID_RMS':None if zero else float(ratio.max()),'zero_ref_nonzero_error':zero}
            result.update(UIDs=len(take),weight_sum=float(w[take].sum()),routed_complete_support_count=int(np.sum(metrics['routed_omitted'][take]==0)),any_child_complete_support_count=int(np.sum(metrics['any_covered'][take])))
            return result
        views={}
        for name,flag in [('development',dev),('consumed',val)]:
            take=np.flatnonzero(flag);views[name]=stats(take,np.ones(N));views[name+'_source_p']=stats(take,p.astype('<f8')**2)
        for name,flag in [('natural_consumed',(occ[:,7]==1)&(occ[:,6]==1)),('teacher_consumed',(occ[:,7]==1)&(occ[:,6]==0))]:
            w=np.bincount(occ[flag,1],minlength=N).astype('<f8');take=np.flatnonzero(w);views[name]=stats(take,w);views[name+'_source_p']=stats(take,w*p.astype('<f8')**2)
        counts=np.bincount(m[dev,3],minlength=128)[m[:,3]];rare={}
        for name,flag in [('dev_count1_4',(counts>=1)&(counts<=4)),('dev_count5_15',(counts>=5)&(counts<=15))]:
            take=np.flatnonzero(flag&val);rare[name]=stats(take,np.ones(N)) if len(take) else {'UIDs':0}
        eligibility={}
        for label in ['oracle','routed']:
            eligibility[label+'_all_consumed_RMS_le_2pct']=views['consumed'][label]['RMS']<=.02 and views['consumed'][label]['zero_ref_nonzero_error']==0
            eligibility[label+'_natural_consumed_source_p_RMS_le_2pct']=views['natural_consumed_source_p'][label]['RMS']<=.02 and views['natural_consumed_source_p'][label]['zero_ref_nonzero_error']==0
            v=views['natural_consumed'][label];eligibility[label+'_natural_consumed_p95_UID_RMS_le_5pct']=v['p95_UID_RMS'] is not None and v['p95_UID_RMS']<=.05
        oracle_ok=all(v for k,v in eligibility.items() if k.startswith('oracle'))
        decision='NEXT_PHYSICAL_HEAD_AND_FRESH_ELIGIBILITY' if all(eligibility.values()) else ('RECONSIDER_REPRESENTATION_OR_SUPPORT_DOMAIN' if not oracle_ok else 'RECONSIDER_INPUT_SELECTOR')
        ctx.r['gates']['ALL_views_original_denominators_oracle_lower_bound_tail_and_frozen_decisions']=True
        result=ctx.finish({'experts':rows,'views':views,'rare_consumed':rare,'eligibility':eligibility,'decision':decision,'partial_source_outputs':evaluations,
            'complete_support_child_cases':complete_cases,'unsupported_selector_children':unsupported,'new_source_WI_or_full_F_rows':0,'model_native_GPU_calls':0,'gradient_updates':0,
            'scope':'Fixed parent ID/p, ONE bank and original finite domain. New partial functions only. No fresh/composed/head/physical/rate/LUT/useful-n/family or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'eligibility':eligibility,'decision':decision,'views':views,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as e:ctx.done.set();ctx.watchdog.cancel();ctx.fail(e);raise
if __name__=='__main__':main()
