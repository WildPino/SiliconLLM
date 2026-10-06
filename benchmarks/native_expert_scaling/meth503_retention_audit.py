"""Independent actual child quantization/projection, input selection and all views."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse,hashlib,json,struct
from pathlib import Path
from meth503_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);ap.add_argument('--raw-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth503_retention',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha);rp=DOC/'meth503_function_result.json';assert ctx.digest(rp)==a.raw_sha
        raw=json.loads(rp.read_bytes());assert all(raw['gates'].values())
        for v in raw['output_inventory']:ctx.exact(v)
        folder=ROOT/'results/native_expert_scaling/meth503_functions';assert json.loads((folder/'terminal_resources.json').read_bytes())['raw_sha256']==a.raw_sha
        ctx.r['gates']['sole_main_RAW_ALL_outputs_and_terminal_SHA']=True
        import numpy as np
        import threadpoolctl as T
        T.threadpool_limits(1);assert all(v['num_threads']==1 for v in T.threadpool_info());ctx.r['numerical_imports']=True
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');N,D,H=17540,768,3072
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
        assert np.array_equal(m[:,0],np.arange(N)) and np.array_equal(x['e'],m[:,3]) and np.all(x['accept']==1) and p.tobytes()==m[:,10].tobytes()
        assert (int(dev.sum()),int(val.sum()))==(11721,5819) and np.all(dev^val) and np.all(occ[:,12]==1) and np.array_equal(occ[:,0],np.arange(19962))
        assert np.array_equal(np.bincount(occ[:,1],minlength=N),m[:,7])
        for oi,mi in [(14,1),(11,3),(15,12),(16,10)]:assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
        def arr(k):return np.load(b['data'][k]['path'],mmap_mode='r',allow_pickle=False)
        hidden,codes,alpha,assignment,diag=[arr(k) for k in ['hidden','codes','scales','assignment','diagnostics']]
        metrics=np.load(folder/'metrics.npy',mmap_mode='r');payload=np.memmap(b['data']['payload']['path'],mode='r',dtype='u1')
        entries=json.loads((DOC/'meth380_switch_base128_export_result.json').read_bytes())['tensors']
        def q16(h):
            maxima=np.amax(abs(h),axis=1) if h.shape[1] else np.zeros(len(h),'<f4');s=np.divide(maxima,np.float32(32767),dtype=np.float32);s[maxima==0]=1
            return np.minimum(32767,np.maximum(-32767,np.rint(h/s[:,None]))).astype('<i2'),s
        def down(q,s,w,so):
            d=np.matmul(q.astype('<f8'),w.T.astype('<f8'));assert np.array_equal(d,np.trunc(d))
            return np.multiply(np.multiply(d,so.astype('<f8')),s.astype('<f8')[:,None]).astype('<f4')
        def normalize(z):
            z=z.astype('<f8');norm=np.sqrt(np.sum(z*z,axis=1));return np.divide(z,norm[:,None],out=np.zeros_like(z),where=norm[:,None]>0)
        assert int(np.dot(np.array([32767,32767],'<i8'),np.array([-128,127],'<i8')))==-32767
        cq,ca=q16(np.array([[0,0],[1,.5]],'<f4'));assert cq.tolist()==[[0,0],[32767,16384]] and ca[0]==1
        ctx.r['gates']['independent_integer_and_A16_controls']=True
        for k in range(0,N,256):
            q,s=q16(hidden[k:k+256]);assert q.tobytes()==codes[k:k+256].tobytes() and s.tobytes()==alpha[k:k+256].tobytes();ctx.guard()
        ctx.r['gates']['ALL_original_roles_parent_mass_and_retained_hidden_quantization_independent']=True
        evaluations=complete_cases=unsupported=0;ctx.phase='ALL127_independent_new_child_functions_input_centroids_and_oracle'
        for e,row in enumerate(raw['experts']):
            at=np.flatnonzero(m[:,3]==e);di=at[dev[at]]
            if not len(at):assert e==0 and row['status']=='EMPTY_UNCOMPILED_UNPROMOTED';continue
            masks=np.load(b['masks'][str(e)]['path']);C=len(masks)
            pc=np.load(folder/f'centres_e{e:03}.npy');actual=np.load(folder/f'actual_e{e:03}.npy',mmap_mode='r');errors=np.load(folder/f'errors_e{e:03}.npy',mmap_mode='r')
            assert actual.shape==(len(at),C,D) and actual.dtype==np.dtype('<f4') and errors.shape==(len(at),C)
            z=normalize(x['x'][di]);count=np.bincount(assignment[di],minlength=C);expected=np.zeros((C,D),'<f8')
            for c in range(C):
                if count[c]:expected[c]=normalize(np.mean(z[assignment[di]==c],axis=0)[None,:])[0]
            assert np.max(np.abs(expected-pc))<=1e-12 and row['development_group_counts']==count.tolist() and row['unsupported_selector_children']==int(np.sum(count==0))
            score=x['x'][at].astype('<f8')@expected.T;score[:,count==0]=-np.inf;sel=np.argmax(score,axis=1)
            assert np.array_equal(sel,metrics['selected'][at]) and np.all(count[sel]>0);unsupported+=int(np.sum(count==0))
            v=entries[f'decoder.block.11.layer.2.mlp.experts.expert_{e}.wo.weight'];assert v['shape']==[D,H] and v['encoding']==1
            wo=np.frombuffer(payload,'i1',v['elements'],v['offset']).reshape(D,H);so=np.frombuffer(payload,'<f4',D,v['scale_offset'])
            assert hashlib.sha256(wo.tobytes()).hexdigest()==v['sha256'] and hashlib.sha256(so.tobytes()).hexdigest()==v['scale_sha256']
            omitted=np.column_stack([np.count_nonzero(codes[at][:,masks[c]==0],axis=1) for c in range(C)])
            for c in range(C):
                j=np.flatnonzero(masks[c]);wc=wo[:,j]
                for k in range(0,len(at),128):
                    ix=at[k:k+128];q,s=q16(hidden[ix][:,j]);ff=down(q,s,wc,so)
                    assert ff.tobytes()==actual[k:k+len(ix),c].tobytes()
                    assert np.array_equal(np.sum((ff.astype('<f8')-ref[ix].astype('<f8'))**2,axis=1),errors[k:k+len(ix),c])
                    complete=omitted[k:k+len(ix),c]==0;assert ff[complete].tobytes()==ref[ix[complete]].tobytes();complete_cases+=int(complete.sum());evaluations+=len(ix);ctx.guard()
            oracle=np.argmin(errors,axis=1);en=np.sum(ref[at].astype('<f8')**2,axis=1)
            assert np.array_equal(metrics['oracle'][at],oracle) and np.array_equal(metrics['reference_energy'][at],en)
            assert np.array_equal(metrics['routed_error'][at],errors[np.arange(len(at)),sel]) and np.array_equal(metrics['oracle_error'][at],errors[np.arange(len(at)),oracle])
            assert np.array_equal(metrics['routed_omitted'][at],omitted[np.arange(len(at)),sel]) and np.array_equal(metrics['any_covered'][at],np.any(omitted==0,axis=1)) and np.array_equal(metrics['any_covered'][at],diag['any'][at])
            ctx.log(expert=e,child_functions=evaluations)
            if e%32==31:print(json.dumps({'expert':e,'seconds':ctx.resources()['wall_seconds']}),flush=True)
            del actual,errors
        assert evaluations==raw['partial_source_outputs']==127570 and complete_cases==raw['complete_support_child_cases'] and unsupported==raw['unsupported_selector_children']
        ctx.r['gates']['ALL127570_outputs_and_energies_BYTE_quantizers_exact_support_identities_independent']=True
        ctx.r['gates']['ALL_input_only_dev_prototypes_supported_children_selected_and_oracle_independent']=True
        def verify(view,take,w):
            en=metrics['reference_energy'];den=float(np.sum(en[take]*w[take]));assert view['UIDs']==len(take) and view['weight_sum']==float(w[take].sum())
            for label,key in [('routed','routed_error'),('oracle','oracle_error')]:
                er=metrics[key];num=float(np.sum(er[take]*w[take]));zero=int(np.sum((en[take]==0)&(er[take]>0)));ratio=np.sqrt(np.divide(er[take],en[take],out=np.zeros(len(take)),where=en[take]!=0))
                expected={'RMS':float(np.sqrt(num/den)),'error_energy':num,'reference_energy':den,'p95_UID_RMS':None if zero else float(np.quantile(ratio,.95)),
                    'max_UID_RMS':None if zero else float(ratio.max()),'zero_ref_nonzero_error':zero};assert view[label]==expected
            assert view['routed_complete_support_count']==int(np.sum(metrics['routed_omitted'][take]==0)) and view['any_child_complete_support_count']==int(np.sum(metrics['any_covered'][take]))
        for name,view in raw['views'].items():
            base=name.removesuffix('_source_p')
            if base in ['development','consumed']:take=np.flatnonzero(dev if base=='development' else val);w=np.ones(N)
            else:w=np.bincount(occ[(occ[:,7]==1)&(occ[:,6]==(1 if base=='natural_consumed' else 0)),1],minlength=N).astype('<f8');take=np.flatnonzero(w)
            if name.endswith('_source_p'):w=w*p.astype('<f8')**2
            verify(view,take,w)
        counts=np.bincount(m[dev,3],minlength=128)[m[:,3]]
        for name,flag in [('dev_count1_4',(counts>=1)&(counts<=4)),('dev_count5_15',(counts>=5)&(counts<=15))]:
            take=np.flatnonzero(flag&val)
            if len(take):verify(raw['rare_consumed'][name],take,np.ones(N))
            else:assert raw['rare_consumed'][name]=={'UIDs':0}
        eligibility={};views=raw['views']
        for label in ['oracle','routed']:
            eligibility[label+'_all_consumed_RMS_le_2pct']=views['consumed'][label]['RMS']<=.02 and views['consumed'][label]['zero_ref_nonzero_error']==0
            eligibility[label+'_natural_consumed_source_p_RMS_le_2pct']=views['natural_consumed_source_p'][label]['RMS']<=.02 and views['natural_consumed_source_p'][label]['zero_ref_nonzero_error']==0
            v=views['natural_consumed'][label];eligibility[label+'_natural_consumed_p95_UID_RMS_le_5pct']=v['p95_UID_RMS'] is not None and v['p95_UID_RMS']<=.05
        assert eligibility==raw['eligibility'];oracle_ok=all(v for k,v in eligibility.items() if k.startswith('oracle'))
        decision='NEXT_PHYSICAL_HEAD_AND_FRESH_ELIGIBILITY' if all(eligibility.values()) else ('RECONSIDER_REPRESENTATION_OR_SUPPORT_DOMAIN' if not oracle_ok else 'RECONSIDER_INPUT_SELECTOR');assert decision==raw['decision']
        ctx.r['gates']['ALL_views_rare_denominators_oracle_tail_lower_bound_and_frozen_decisions_independent']=True
        result=ctx.finish({'main_completed':True,'raw':{'path':str(rp),'sha256':a.raw_sha},'unique_inputs_audited':N,'partial_functions_audited':evaluations,
            'views':views,'rare_consumed':raw['rare_consumed'],'eligibility':eligibility,'decision':decision,'new_source_WI_or_full_F_rows':0,
            'scope':'Independent new partial functions and finite original-domain selection, no full source replay or fresh/physical promotion.'})
        print(json.dumps({'gates':result['gates'],'decision':decision,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as e:ctx.done.set();ctx.watchdog.cancel();ctx.fail(e);raise
if __name__=='__main__':main()
