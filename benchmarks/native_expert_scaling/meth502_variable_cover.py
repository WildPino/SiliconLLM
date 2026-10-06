"""ONE extra-storage construction, starting from admitted501 retained prefixes."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse,json,struct
from pathlib import Path
from meth502_operations import Context,ROOT,DOC,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--binding-sha',required=True);a=ap.parse_args()
    ctx=Context(ROOT/'results/native_expert_scaling/meth502_variable_cover',Path(a.out).resolve())
    try:
        b=ctx.admit(a.binding_sha)
        import numpy as np
        import meth502_extension_math as M
        ctx.r['numerical_imports']=True;write(ctx.out/'controls.json',M.controls())
        ctx.r['gates']['NEW_resume_append_fill_tie_variable_width_controls']=True
        N,H,B=17540,3072,1536
        with Path(b['data']['uid']['path']).open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',b'M493U001',180,0,N)
        uid=np.memmap(b['data']['uid']['path'],mode='r',offset=24,shape=(N,),dtype=[('m','<u4',(13,)),('hash','u1',(128,))])
        m=uid['m'];dev=(m[:,4]&5)!=0;val=(m[:,4]&2)!=0
        assert np.array_equal(m[:,0],np.arange(N)) and np.all(m[:,2]==11) and np.all(dev^val) and (int(dev.sum()),int(val.sum()))==(11721,5819)
        def arr(key):return np.load(b['data'][key]['path'],mmap_mode='r',allow_pickle=False)
        qh,old,base,previous,assign0=[arr(k) for k in ['codes','metrics','base','final','assignment']]
        assert qh.shape==(N,H) and qh.dtype==np.dtype('<i2') and np.all(qh>=0)
        raw0=json.loads(Path(b['data']['raw501']['path']).read_bytes())
        with Path(b['data']['order']['path']).open('rb') as f:
            assert f.read(24)==struct.pack('<8sIIQ',b'M501ORD1',4,0,11721);order=np.fromfile(f,'<u4');assert len(order)==11721
        def bits(v):return int.from_bytes(np.packbits(v!=0,bitorder='little').tobytes(),'little')
        support=[]
        for k in range(0,N,256):
            support.extend(int.from_bytes(v.tobytes(),'little') for v in np.packbits(qh[k:k+256]!=0,axis=1,bitorder='little'));ctx.guard()
        assert all(v.bit_count()==int(old['support'][k]) for k,v in enumerate(support))
        assignment=np.lib.format.open_memmap(ctx.out/'development_assignment.npy',mode='w+',shape=(N,),dtype='<i4');assignment[:]=assign0
        trace=(ctx.out/'resume_trace.jsonl').open('x',encoding='utf8');rows=[];cursor=0
        ctx.phase='ALL128_development_only_retained_prefix_extension'
        for e in range(128):
            at=np.flatnonzero(m[:,3]==e);di=at[dev[at]];r0=raw0['experts'][e];assert r0['expert']==e
            if not len(di):
                assert e==0 and not len(at);np.save(ctx.out/f'masks_e{e:03}.npy',np.zeros((0,H),'u1'));rows.append({'expert':e,'children':0,'status':'EMPTY_NO_EXPOSURE_UNPROMOTED'});continue
            eo=order[cursor:cursor+len(di)].tolist();cursor+=len(di)
            assert eo==sorted(map(int,di),key=lambda k:(-support[k].bit_count(),k))
            prefix=[k for k in eo if assignment[k]>=0];remaining=[k for k in eo if assignment[k]<0]
            assert eo==prefix+remaining and len(prefix)==r0['assigned_development']
            unions=[bits(v) for v in base[e]]
            assert all(support[k]&~unions[int(assignment[k])]==0 for k in prefix)
            if r0['status']=='EXACT_DEVELOPMENT_SUPPORT_COVER_WITNESS':
                assert not remaining;f=[bits(v) for v in previous[e]];masks=np.asarray(previous[e]).copy();counts=None;steps=[];status='BASELINE501_BYTE_UNCHANGED'
            else:
                if remaining:assert r0['first_unplaced_UID']==remaining[0]
                else:assert r0['status']=='THIS_PARTITION_GLOBAL_UNION_COPY_BUDGET_FAIL_INCONCLUSIVE'
                f,steps,counts=M.extend(unions,remaining,support,B,H)
                masks=np.array([np.unpackbits(np.frombuffer(v.to_bytes(H//8,'little'),dtype='u1'),bitorder='little') for v in f],dtype='u1')
                for s in steps:assignment[s['UID']]=s['child'];trace.write(json.dumps({'expert':e,**s})+'\n')
                status='EXTENDED_DEVELOPMENT_SUPPORT_COVER'
            assert all(v.bit_count()<=B for v in f) and all(support[k]&~f[int(assignment[k])]==0 for k in di)
            whole=0
            for v in f:whole|=v
            assert whole==(1<<H)-1
            np.save(ctx.out/f'masks_e{e:03}.npy',masks)
            Mtotal=sum(v.bit_count() for v in f);C=len(f)
            rows.append({'expert':e,'status':status,'development':len(di),'consumed':int(val[at].sum()),'retained_prefix_UIDs':len(prefix),'resumed_UIDs':len(remaining),
                'children':C,'widths':[v.bit_count() for v in f],'copies':Mtotal,'copy_ratio':Mtotal/H,'extension_counts':counts,
                'logical_bytes':{'I8_coefficients':2*768*Mtotal,'WI_F32_scales':4*Mtotal,'WO_F32_scales':4*768*C,'u16_indices':2*Mtotal}})
            ctx.guard()
            if e%32==31:ctx.log(expert=e);print(json.dumps({'expert':e}),flush=True)
        trace.close();assignment.flush();assert cursor==11721 and np.all(assignment[dev]>=0) and np.all(assignment[val]==-2)
        ctx.r['gates']['ALL127_complete_development_subset_and_full_source_union_fixed_active_ceiling']=True
        ctx.r['gates']['61_baseline_BYTE_preserved_52_resumed_14_reused_no_prefix_replay_exact_copy_cost']=True
        dt=np.dtype([('any','u1'),('old','u1'),('min_omitted','<u2'),('first_cover','<i4')])
        diag=np.lib.format.open_memmap(ctx.out/'support_diagnostics.npy',mode='w+',shape=(N,),dtype=dt)
        ctx.phase='fixed_masks_ALL_original_consumed_support_diagnostics'
        for r in rows[1:]:
            e=r['expert'];at=np.flatnonzero(m[:,3]==e);f=[bits(v) for v in np.load(ctx.out/f'masks_e{e:03}.npy')]
            for k in at:
                omitted=[(support[k]&~v).bit_count() for v in f];covered=[c for c,v in enumerate(omitted) if v==0]
                diag[k]=(bool(covered),omitted[int(old['selected'][k])]==0,min(omitted),covered[0] if covered else -1)
            ctx.guard()
        diag.flush();rs=rows[1:];totals={k:sum(r['logical_bytes'][k] for r in rs) for k in rs[0]['logical_bytes']}
        summary={'nonempty_experts':127,'development_UIDs':11721,'consumed_UIDs':5819,'baseline_experts':sum(r['status']=='BASELINE501_BYTE_UNCHANGED' for r in rs),
            'resumed_experts':sum(r['resumed_UIDs']>0 for r in rs),'resumed_UIDs':sum(r['resumed_UIDs'] for r in rs),'children':sum(r['children'] for r in rs),
            'children_min':min(r['children'] for r in rs),'children_max':max(r['children'] for r in rs),'copies':sum(r['copies'] for r in rs),
            'bank_copy_ratio':sum(r['copies'] for r in rs)/(127*H),'development_any_child_complete_support':int(np.sum(dev&(diag['any']!=0))),
            'consumed_any_child_complete_support':int(np.sum(val&(diag['any']!=0))),'consumed_old_selector_complete_support':int(np.sum(val&(diag['old']!=0))),
            'logical_bytes':totals,'active_width_ceiling':B,'empty_expert_0':'UNCOMPILED_UNPROMOTED'}
        ctx.r['gates']['ALL_consumed_diagnostics_after_development_masks_no_consumed_selection']=True
        result=ctx.finish({'experts':rows,'summary':summary,'eligibility':{'ALL127_finite_development_exact_support_cover':True},'decision':'NEXT_ACTUAL_FUNCTION_ORACLE_AND_INPUT_SELECTOR',
            'source_model_native_GPU_gradient_calls':0,'scope':'Finite support/copy witness. No optimal-memory, fresh quality, cheap routing, physical layout/DRAM/rate or family promotion.'})
        print(json.dumps({'gates':result['gates'],'summary':summary,'resource':ctx.terminal_resources['resource']}),flush=True)
    except BaseException as e:ctx.done.set();ctx.watchdog.cancel();ctx.fail(e);raise
if __name__=='__main__':main()
