"""Independent Boolean unions: audit ONLY continued tails, no retained-prefix replay."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse,json,struct
from pathlib import Path
from meth502_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);ap.add_argument('--raw-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth502_retention',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha);rp=DOC/'meth502_variable_cover_result.json';assert ctx.digest(rp)==a.raw_sha
        raw=json.loads(rp.read_bytes());assert all(raw['gates'].values())
        for v in raw['output_inventory']:ctx.exact(v)
        folder=ROOT/'results/native_expert_scaling/meth502_variable_cover';assert json.loads((folder/'terminal_resources.json').read_bytes())['raw_sha256']==a.raw_sha
        ctx.r['gates']['sole_main_ALL_output_and_terminal_SHA']=True
        import numpy as np
        ctx.r['numerical_imports']=True;N,H,B=17540,3072,1536
        def arr(k):return np.load(b['data'][k]['path'],mmap_mode='r',allow_pickle=False)
        qh,old,base,previous,assign0=[arr(k) for k in ['codes','metrics','base','final','assignment']]
        m=np.memmap(b['data']['uid']['path'],mode='r',offset=24,shape=(N,),dtype=[('m','<u4',(13,)),('hash','u1',(128,))])['m']
        dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0;assert int(dev.sum())==11721 and int(val.sum())==5819 and np.all(dev^val)
        assignment=np.load(folder/'development_assignment.npy',mmap_mode='r');diag=np.load(folder/'support_diagnostics.npy',mmap_mode='r')
        raw0=json.loads(Path(b['data']['raw501']['path']).read_bytes())
        with Path(b['data']['order']['path']).open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',b'M501ORD1',4,0,11721);order=np.fromfile(f,'<u4')
        trace=[json.loads(v) for v in (folder/'resume_trace.jsonl').read_text().splitlines()];cursor=tc=0
        assert all(json.loads((folder/'controls.json').read_bytes()).values())
        # Independent elementary capacity arithmetic, without importing main math.
        u=np.eye(6,dtype=bool).reshape(3,2,6).any(axis=1);assert u.sum()==6 and np.all(u.any(axis=0)) and np.all(u.sum(axis=1)==2)
        ctx.r['gates']['independent_Boolean_capacity_control']=True
        ctx.phase='ALL128_independent_retained_prefix_resume_fill_and_cost'
        for e,row in enumerate(raw['experts']):
            masks=np.load(folder/f'masks_e{e:03}.npy');at=np.flatnonzero(m[:,3]==e);di=at[dev[at]]
            assert row['expert']==e and masks.dtype==np.dtype('u1') and np.all(masks<=1)
            if not len(di):assert e==0 and not len(at) and masks.shape==(0,H) and row['children']==0;continue
            eo=order[cursor:cursor+len(di)].tolist();cursor+=len(di)
            assert eo==sorted(map(int,di),key=lambda k:(-int(np.count_nonzero(qh[k])),k))
            prefix=[k for k in eo if assign0[k]>=0];remaining=[k for k in eo if assign0[k]<0]
            assert eo==prefix+remaining and len(prefix)==raw0['experts'][e]['assigned_development']
            u=[v.astype(bool).copy() for v in base[e]]
            assert all(np.all((qh[k]==0)|u[int(assign0[k])]) and assignment[k]==assign0[k] for k in prefix)
            if row['status']=='BASELINE501_BYTE_UNCHANGED':
                assert np.array_equal(masks,previous[e]) and not remaining and row['extension_counts'] is None
            else:
                created=0
                for k in remaining:
                    support=qh[k]!=0;assert np.count_nonzero(support)<=B
                    before=[int(v.sum()) for v in u];added=[int(np.sum(support&~v)) for v in u]
                    choices=[c for c in range(len(u)) if before[c]+added[c]<=B];new=not choices
                    if new:c=len(u);u.append(np.zeros(H,bool));before.append(0);added.append(int(support.sum()));created+=1
                    else:c=min(choices,key=lambda c:(added[c],before[c]+added[c],c))
                    assert trace[tc]=={'expert':e,'UID':k,'child':c,'created':new,'before':before[c],'added':added[c]};tc+=1
                    assert assignment[k]==c;u[c]|=support
                S=sum(int(v.sum()) for v in u);U=int(np.any(u,axis=0).sum());missing=np.flatnonzero(~np.any(u,axis=0));fillnew=0
                for j in missing:
                    sizes=[int(v.sum()) for v in u];choices=[c for c,n in enumerate(sizes) if n<B]
                    if not choices:c=len(u);u.append(np.zeros(H,bool));fillnew+=1
                    else:c=min(choices,key=lambda c:(sizes[c],c))
                    u[c][j]=True
                assert np.array_equal(masks,np.array(u,dtype='u1'))
                assert row['extension_counts']=={'S':S,'U':U,'missing':H-U,'copies':H+S-U,'packing_new_children':created,'global_union_new_children':fillnew}
            assert masks.shape==(row['children'],H) and np.all(masks.any(axis=0)) and np.all(masks.sum(axis=1)<=B)
            assert row['widths']==masks.sum(axis=1).tolist() and row['copies']==int(masks.sum()) and row['copy_ratio']==row['copies']/H
            assert row['retained_prefix_UIDs']==len(prefix) and row['resumed_UIDs']==len(remaining) and row['development']==len(di) and row['consumed']==int(val[at].sum())
            assert all(np.all((qh[k]==0)|masks[int(assignment[k])].astype(bool)) for k in di)
            M,C=row['copies'],row['children'];assert row['logical_bytes']=={'I8_coefficients':2*768*M,'WI_F32_scales':4*M,'WO_F32_scales':4*768*C,'u16_indices':2*M}
            support=qh[at]!=0;omit=np.column_stack([np.sum(support&~v.astype(bool),axis=1) for v in masks]);cover=omit==0
            assert np.array_equal(diag['any'][at],cover.any(axis=1)) and np.array_equal(diag['old'][at],cover[np.arange(len(at)),old['selected'][at]])
            assert np.array_equal(diag['min_omitted'][at],omit.min(axis=1)) and np.array_equal(diag['first_cover'][at],np.where(cover.any(axis=1),cover.argmax(axis=1),-1))
            ctx.guard()
        assert cursor==11721 and tc==len(trace) and np.all(assignment[val]==-2)
        rs=raw['experts'][1:];s=raw['summary']
        checks={'baseline_experts':sum(r['status']=='BASELINE501_BYTE_UNCHANGED' for r in rs),'resumed_experts':sum(r['resumed_UIDs']>0 for r in rs),
            'resumed_UIDs':sum(r['resumed_UIDs'] for r in rs),'children':sum(r['children'] for r in rs),'children_min':min(r['children'] for r in rs),'children_max':max(r['children'] for r in rs),
            'copies':sum(r['copies'] for r in rs),'bank_copy_ratio':sum(r['copies'] for r in rs)/(127*H),'development_any_child_complete_support':int(np.sum(dev&(diag['any']!=0))),
            'consumed_any_child_complete_support':int(np.sum(val&(diag['any']!=0))),'consumed_old_selector_complete_support':int(np.sum(val&(diag['old']!=0))),
            'logical_bytes':{k:sum(r['logical_bytes'][k] for r in rs) for k in rs[0]['logical_bytes']}}
        assert all(s[k]==v for k,v in checks.items()) and s['development_UIDs']==11721 and s['consumed_UIDs']==5819 and s['nonempty_experts']==127
        assert raw['eligibility']=={'ALL127_finite_development_exact_support_cover':True}
        ctx.r['gates']['ALL_retained_prefixes_BYTE_baseline_new_tail_fill_and_exact_partition_copy_cost_independent']=True
        ctx.r['gates']['ALL127_development_subsets_global_union_width_and_logical_bytes_independent']=True
        ctx.r['gates']['ALL_consumed_diagnostics_complete_denominators_independent']=True
        result=ctx.finish({'main_completed':True,'raw':{'path':str(rp),'sha256':a.raw_sha},'summary':s,'eligibility':raw['eligibility'],'decision':raw['decision'],
            'unique_inputs_audited':N,'source_model_native_GPU_gradient_calls':0,'scope':'Boolean continuation/copy certificates, no original-prefix pack replay or new functions.'})
        print(json.dumps({'gates':result['gates'],'summary':s,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as e:ctx.done.set();ctx.watchdog.cancel();ctx.fail(e);raise
if __name__=='__main__':main()
