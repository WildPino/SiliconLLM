"""Complete original-domain support-cover witnesses; no source function recomputation."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import struct
from meth501_operations import Context,ROOT,DOC,write
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth501_support_cover',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha)
        import numpy as np
        import meth501_cover_math as M
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True
        write(ctx.out/'controls.json',M.controls());ctx.r['gates']['NEW_constructive_tie_first_failure_duplicate_capacity_controls']=True
        N,H,C,B=17540,3072,4,1536
        uidpath=Path(b['data']['uid']['path']);ut=np.dtype([('m','<u4',(13,)),('hash','u1',(128,))])
        with uidpath.open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',b'M493U001',180,0,N)
        uid=np.memmap(uidpath,dtype=ut,mode='r',offset=24,shape=(N,));meta=uid['m'];dev=(meta[:,4]&5)!=0;val=(meta[:,4]&2)!=0
        qh=np.load(b['data']['codes']['path'],mmap_mode='r',allow_pickle=False);old=np.load(b['data']['metrics']['path'],mmap_mode='r',allow_pickle=False)
        assert qh.shape==(N,H) and qh.dtype==np.dtype('<i2') and np.all(qh>=0)
        assert np.array_equal(meta[:,0],np.arange(N)) and np.all(meta[:,2]==11) and np.all(meta[:,3]<128)
        assert (int(dev.sum()),int(val.sum()))==(11721,5819) and np.all(dev^val) and old.shape==(N,) and np.all(old['selected']<C)
        assert np.array_equal(np.count_nonzero(qh,axis=1),old['support'])
        def out(name,shape,dtype):return np.lib.format.open_memmap(ctx.out/(name+'.npy'),mode='w+',shape=shape,dtype=dtype)
        base=out('base_unions',(128,C,H),'u1');final=out('final_masks',(128,C,H),'u1');regions=out('old_region_unions',(128,C,H),'u1')
        assignment=out('development_assignment',(N,),'<i4');assignment[:]=-2;assignment[dev]=-1
        coverage=out('complete_support_coverage',(N,C),'u1');coverage[:]=0
        orderfile=(ctx.out/'development_order.bin').open('xb');orderfile.write(struct.pack('<8sIIQ',b'M501ORD1',4,0,11721))
        trace=(ctx.out/'packing_trace.jsonl').open('x',encoding='utf8');rows=[];success=[];failures=[]
        ctx.phase='ALL128_fixed_development_support_packing_before_consumed_diagnostics'
        for e in range(128):
            at=np.flatnonzero(meta[:,3]==e);di=at[dev[at]];base[e]=0;final[e]=0;regions[e]=0
            if not len(di):
                assert e==0 and not len(at);rows.append({'expert':e,'development':0,'consumed':0,'status':'EMPTY_NO_EXPOSURE_UNPROMOTED'});continue
            support=qh[di]!=0;order,assigned,u,steps,first=M.pack(support,di,C,B)
            orderfile.write(di[order].astype('<u4').tobytes());assignment[di]=assigned;base[e]=u
            for step in steps:trace.write(json.dumps({'expert':e,'UID':step[0],'child':step[1],'before':step[2:6],'added':step[6:10]})+'\n')
            for c in range(C):
                group=support[old['selected'][di]==c]
                if len(group):regions[e,c]=np.any(group,axis=0)
            counts={'base_slots':int(u.sum()),'base_union':int(u.any(axis=0).sum()),'duplicate_slots':int(u.sum()-u.any(axis=0).sum())}
            row={'expert':e,'development':len(di),'consumed':int(np.sum(val[at])),'source_development_union':int(support.any(axis=0).sum()),
                 'old_selected_region_union_sizes':regions[e].sum(axis=1).tolist(),'base_child_sizes':u.sum(axis=1).tolist(),
                 'assigned_development':int(np.sum(assigned>=0)),'unprocessed_or_unplaced':int(np.sum(assigned<0)),
                 'first_unplaced_UID':first,'packing_counts':counts}
            if first is not None:
                row['status']='GREEDY_FIRST_NO_FEASIBLE_CHILD_INCONCLUSIVE';failures.append(e)
            else:
                f,fillcounts=M.fill(u,B);fillcounts.pop('missing_leaf_placements',None);row['fill_counts']=fillcounts
                if f is None:row['status']='THIS_PARTITION_GLOBAL_UNION_COPY_BUDGET_FAIL_INCONCLUSIVE';failures.append(e)
                else:
                    assert np.all(np.logical_or(~support,f[assigned]));final[e]=f
                    row['status']='EXACT_DEVELOPMENT_SUPPORT_COVER_WITNESS';success.append(e)
            rows.append(row);ctx.guard()
            if e%16==15:ctx.log(completed_expert=e,positive_witnesses=len(success));print(json.dumps({'expert':e,'positive_witnesses':len(success)}),flush=True)
        orderfile.close();trace.close()
        ctx.r['gates']['ALL128_fixed_development_only_pack_first_failure_and_unprocessed_UIDs_retained']=True
        ctx.r['gates']['ALL_positive_development_subset_width_global_union_and_copy_budget_witnesses']=True
        ctx.phase='fixed_masks_ALL_consumed_support_diagnostics_no_selection'
        for e in success:
            at=np.flatnonzero(meta[:,3]==e);support=qh[at]!=0
            for c in range(C):coverage[at,c]=np.all(np.logical_or(~support,final[e,c].astype(bool)),axis=1)
        for v in [base,final,regions,assignment,coverage]:v.flush()
        ex=np.isin(meta[:,3],success);covered=coverage.any(axis=1);selected=coverage[np.arange(N),old['selected']]
        summary={'nonempty_experts':127,'positive_witness_experts':success,'inconclusive_experts':failures,
                 'positive_witness_count':len(success),'inconclusive_count':len(failures),'full_development_UIDs':11721,'full_consumed_UIDs':5819,
                 'positive_development_UIDs':int(np.sum(dev&ex)),'positive_consumed_UIDs':int(np.sum(val&ex)),
                 'positive_consumed_any_child_exact_support_UIDs':int(np.sum(val&ex&covered)),
                 'positive_consumed_old_selector_exact_support_UIDs':int(np.sum(val&ex&selected)),
                 'full_consumed_any_child_exact_support_UIDs':int(np.sum(val&covered)),
                 'old_nonempty_selected_regions_union_above1536':sum(int(v)>B for r in rows if r['development'] for v in r['old_selected_region_union_sizes']),
                 'source_development_union_above1536_experts':sum(r.get('source_development_union',0)>B for r in rows),
                 'scope':'Complete original cohort reported; coverage on positive-mask subset is not complete-cohort eligibility.'}
        eligible=len(success)==127;decision='NEXT_NEW_MASK_FUNCTION_AND_INPUT_SELECTION_ELIGIBILITY' if eligible else 'UNRESOLVED_SUPPORT_PACKING_OR_DOMAIN_GEOMETRY'
        ctx.r['gates']['fixed_positive_masks_consumed_support_diagnostics_original_ALL_denominators']=True
        result=ctx.finish({'experts':rows,'summary':summary,'eligibility':{'ALL127_complete_development_cover':eligible},'decision':decision,
            'source_model_native_GPU_gradient_calls':0,'scope':'Exact finite development support-cover eligibility only. Greedy failure is not nonexistence. No new F/own-state/head/fresh/physical-cost/rate/capacity or goal promotion.'})
        print(json.dumps({'gates':result['gates'],'summary':summary,'decision':decision,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as e:ctx.done.set();ctx.watchdog.cancel();ctx.fail(e);raise
if __name__=='__main__':main()
