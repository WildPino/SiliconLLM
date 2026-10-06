"""Independent integer-bitset support packing/fill/subset certificates, no main/math import."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import struct
from meth501_operations import Context,ROOT,DOC,write
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);ap.add_argument('--raw-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth501_retention',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha);rp=DOC/'meth501_support_cover_result.json';assert ctx.digest(rp)==a.raw_sha
        raw=json.loads(rp.read_bytes());assert all(raw['gates'].values());folder=ROOT/'results/native_expert_scaling/meth501_support_cover'
        for v in raw['output_inventory']:ctx.exact(v)
        assert json.loads((folder/'terminal_resources.json').read_bytes())['raw_sha256']==a.raw_sha
        ctx.r['gates']['sole_complete_main_RAW_ALL_output_and_terminal_SHA']=True
        import numpy as np
        ctx.r['numerical_imports']=True;N,H,C,B=17540,3072,4,1536;FULL=(1<<H)-1
        def arr(name):return np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        base,final,region,assignment,coverage=[arr(n) for n in ['base_unions','final_masks','old_region_unions','development_assignment','complete_support_coverage']]
        qh=np.load(b['data']['codes']['path'],mmap_mode='r',allow_pickle=False);old=np.load(b['data']['metrics']['path'],mmap_mode='r',allow_pickle=False)
        assert qh.shape==(N,H) and qh.dtype==np.dtype('<i2') and np.all(qh>=0)
        up=Path(b['data']['uid']['path']);ut=np.dtype([('m','<u4',(13,)),('hash','u1',(128,))])
        with up.open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',b'M493U001',180,0,N)
        uid=np.memmap(up,mode='r',dtype=ut,offset=24,shape=(N,));m=uid['m'];dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(N)) and np.all(m[:,2]==11) and np.all(dev^val) and (int(dev.sum()),int(val.sum()))==(11721,5819)
        def bits(v):return int.from_bytes(np.packbits(v!=0,bitorder='little').tobytes(),'little')
        supports=[]
        for k in range(0,N,256):
            part=np.packbits(qh[k:k+256]!=0,axis=1,bitorder='little')
            supports.extend(int.from_bytes(v.tobytes(),'little') for v in part);ctx.guard()
        assert all(v.bit_count()==int(old['support'][k]) for k,v in enumerate(supports))
        control=json.loads((folder/'controls.json').read_bytes());assert len(control)==3 and all(control.values())
        # Arithmetic witness independent of main's bool arrays: missing4 slots vs0 free.
        fixture=[15]*4;assert 8-(fixture[0]|fixture[1]|fixture[2]|fixture[3]).bit_count()==4 and 4*4-sum(v.bit_count() for v in fixture)==0
        ctx.r['gates']['independent_integer_bitset_cardinality_and_duplicate_capacity_control']=True
        trace=[json.loads(v) for v in (folder/'packing_trace.jsonl').read_text().splitlines()];cursor=0
        with (folder/'development_order.bin').open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M501ORD1',4,0,11721);order_saved=np.fromfile(f,'<u4');assert len(order_saved)==11721
        order_cursor=0;positive=[];failed=[];dev_positive=cons_positive=any_positive=sel_positive=0;old_above=union_above=0
        expected_coverage=np.zeros((N,C),'u1');ctx.phase='ALL128_independent_integer_bitset_original_support_witnesses'
        for e in range(128):
            row=raw['experts'][e];assert row['expert']==e
            at=[k for k in range(N) if int(m[k,3])==e];di=[k for k in at if dev[k]];vi=[k for k in at if val[k]]
            assert row['development']==len(di) and row['consumed']==len(vi)
            if not di:assert e==0 and not at and not base[e].any() and not final[e].any();continue
            order=sorted(di,key=lambda k:(-supports[k].bit_count(),k));assert order==order_saved[order_cursor:order_cursor+len(di)].tolist();order_cursor+=len(di)
            unions=[0]*4;assigned={};first=None;old_unions=[0]*4;whole=0
            for k in di:whole|=supports[k];old_unions[int(old['selected'][k])]|=supports[k]
            for k in order:
                before=[v.bit_count() for v in unions];added=[(supports[k]&~v).bit_count() for v in unions]
                feasible=[c for c in range(C) if before[c]+added[c]<=B]
                c=min(feasible,key=lambda c:(added[c],before[c]+added[c],c)) if feasible else -1
                assert trace[cursor]=={'expert':e,'UID':k,'child':c,'before':before,'added':added};cursor+=1
                if c<0:first=k;break
                assigned[k]=c;unions[c]|=supports[k]
            assert first==row['first_unplaced_UID'] and len(assigned)==row['assigned_development'] and len(di)-len(assigned)==row['unprocessed_or_unplaced']
            assert all(int(assignment[k])==assigned.get(k,-1) for k in di)
            assert all(bits(base[e,c])==unions[c] and bits(region[e,c])==old_unions[c] for c in range(C))
            assert row['base_child_sizes']==[v.bit_count() for v in unions] and row['old_selected_region_union_sizes']==[v.bit_count() for v in old_unions]
            assert row['source_development_union']==whole.bit_count();old_above+=sum(v.bit_count()>B for v in old_unions);union_above+=whole.bit_count()>B
            u=0
            for v in unions:u|=v
            S=sum(v.bit_count() for v in unions);assert row['packing_counts']=={'base_slots':S,'base_union':u.bit_count(),'duplicate_slots':S-u.bit_count()}
            feasible_fill=H-u.bit_count()<=C*B-S
            if first is not None:assert row['status']=='GREEDY_FIRST_NO_FEASIBLE_CHILD_INCONCLUSIVE';failed.append(e);assert not final[e].any();continue
            if not feasible_fill:assert row['status']=='THIS_PARTITION_GLOBAL_UNION_COPY_BUDGET_FAIL_INCONCLUSIVE';failed.append(e);assert not final[e].any();continue
            f=unions.copy();placements=[]
            for j in range(H):
                if not u&(1<<j):
                    c=min((c for c in range(C) if f[c].bit_count()<B),key=lambda c:(f[c].bit_count(),c));f[c]|=1<<j;placements.append([j,c])
            for c in range(C):
                for j in range(H):
                    if f[c].bit_count()==B:break
                    if not f[c]&(1<<j):f[c]|=1<<j
            assert row['fill_counts']=={'base_slots':S,'base_union':u.bit_count(),'duplicate_slots':S-u.bit_count(),'missing_source_leaves':H-u.bit_count(),'available_slots':C*B-S,'filled':True}
            assert all(bits(final[e,c])==f[c] and f[c].bit_count()==B for c in range(C)) and (f[0]|f[1]|f[2]|f[3])==FULL
            assert all(supports[k]&~f[c]==0 for k,c in assigned.items()) and row['status']=='EXACT_DEVELOPMENT_SUPPORT_COVER_WITNESS'
            positive.append(e);dev_positive+=len(di);cons_positive+=len(vi)
            for k in at:
                covered=[int(supports[k]&~v==0) for v in f];expected_coverage[k]=covered
                if val[k]:any_positive+=int(any(covered));sel_positive+=covered[int(old['selected'][k])]
            ctx.guard()
        assert cursor==len(trace) and order_cursor==11721 and np.array_equal(expected_coverage,coverage) and np.all(assignment[val]==-2)
        summary=raw['summary'];expected={'positive_witness_experts':positive,'inconclusive_experts':failed,'positive_witness_count':len(positive),'inconclusive_count':len(failed),
            'positive_development_UIDs':dev_positive,'positive_consumed_UIDs':cons_positive,'positive_consumed_any_child_exact_support_UIDs':any_positive,
            'positive_consumed_old_selector_exact_support_UIDs':sel_positive,'full_consumed_any_child_exact_support_UIDs':any_positive,
            'old_nonempty_selected_regions_union_above1536':old_above,'source_development_union_above1536_experts':union_above}
        assert all(summary[k]==v for k,v in expected.items()) and len(positive)+len(failed)==127
        assert summary['full_development_UIDs']==11721 and summary['full_consumed_UIDs']==5819 and summary['nonempty_experts']==127
        assert raw['eligibility']=={'ALL127_complete_development_cover':len(positive)==127}
        ctx.r['gates']['ALL_original_UID_order_first_conflict_partial_and_positive_geometry_independent']=True
        ctx.r['gates']['ALL_positive_mask_development_exact_subset_global_union_and_copy_budget_independent']=True
        ctx.r['gates']['ALL_consumed_fixed_mask_subset_diagnostics_original_denominators_and_eligibility']=True
        result=ctx.finish({'main_completed':True,'raw':{'path':str(rp),'sha256':a.raw_sha},'summary':summary,'eligibility':raw['eligibility'],'decision':raw['decision'],
            'unique_inputs_audited':N,'source_model_native_GPU_gradient_calls':0,'scope':'Independent support-cover certificates only. Greedy failure is inconclusive; finite support coverage is not fresh quality, routing or physical speed.'})
        print(json.dumps({'gates':result['gates'],'summary':summary,'decision':result['decision'],'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as e:ctx.done.set();ctx.watchdog.cancel();ctx.fail(e);raise
if __name__=='__main__':main()
