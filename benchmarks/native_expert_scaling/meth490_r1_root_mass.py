"""One development-only 12-root affine sigmoid fit and finite native evaluation."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
from decimal import Decimal,localcontext
import json
from pathlib import Path
import struct
import traceback
from meth490_r1_operations import ROOT,DOC,Context,write

OUT=ROOT/'results/native_expert_scaling/meth490_r1_root_mass'
RAW=DOC/'meth490_r1_root_mass_result.json'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);ap.add_argument('--binding-sha',required=True);a=ap.parse_args()
    assert a.out.resolve()==RAW.resolve()
    ctx=Context(OUT,RAW,600,1<<30)
    try:
        b=ctx.admit(a.binding_sha);ctx.binding=b
        import numpy as np
        import threadpoolctl
        import meth490_r1_root_mass_math as M
        ctx.r['numerical_imports']=True
        threadpoolctl.threadpool_limits(limits=1)
        for pool in threadpoolctl.threadpool_info():assert pool['num_threads']==1
        assert np.__version__==b['runtime']['packages']['numpy']['version']
        tiny=np.array([1],'<u8').view('<f8');assert (tiny+tiny).view('<u8')[0]==2
        assert float(np.float32(1+2**-24))==1
        ctx.r['threadpools']=threadpoolctl.threadpool_info()
        U,Q,D=238872,387036,768
        UNIQUE=np.dtype([('meta','<u4',(12,)),('input_sha','u1',(32,)),('score_sha','u1',(32,)),('value','<f8',(3,)),('input','<f4',(768,)),('score','<f4',(128,))])
        NODE=np.dtype([('maximum','<f4',(2,)),('winner','<u2',(2,)),('mass','<f8',(2,))])
        TARGET=np.dtype([('unique_id','<u4'),('nodes',NODE,(127,))])
        def chunks(key,magic,width,reserved,count,dtype):
            with Path(b['data'][key]['path']).open('rb') as f:
                assert f.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
                for start in range(0,count,2048):
                    n=min(2048,count-start);data=f.read(n*width);assert len(data)==n*width
                    yield start,np.frombuffer(data,dtype=dtype)
                assert not f.read(1)
        def dense(key,magic,width,reserved,count,columns):
            rows=[v.copy().reshape(-1,columns) for _,v in chunks(key,magic,width,reserved,count,'<u4')]
            return np.concatenate(rows)
        ctx.phase='ALL_UID_roles_links_and_root_targets_streaming'
        meta=np.empty((U,12),'<u4')
        for start,rows in chunks('unique_inputs.bin',b'M479UNI1',3720,768,U,UNIQUE):meta[start:start+len(rows)]=rows['meta']
        owner=dense('ownership.bin',b'M479OWN1',32,0,U,8)
        links=dense('query_links.bin',b'M479LNK1',52,0,Q,13)
        uid=np.arange(U,dtype='<u4');assert np.array_equal(meta[:,0],uid) and np.array_equal(owner[:,0],uid)
        assert np.array_equal(meta[:,2],owner[:,1]) and np.array_equal(links[:,6],meta[links[:,12],2]) and np.array_equal(links[:,9],meta[links[:,12],7])
        dev=(owner[:,2]&5)!=0;val=(owner[:,2]&2)!=0
        assert (int(dev.sum()),int(val.sum()))==(159414,79458) and not np.any(dev&val)
        masses=np.empty((U,2),'<f8');maxima=np.empty((U,2),'<f4');winners=np.empty((U,2),'<u2')
        for start,rows in chunks('group_targets.bin',b'M479TGT1',3560,127,U,TARGET):
            assert np.array_equal(rows['unique_id'],uid[start:start+len(rows)])
            masses[start:start+len(rows)]=rows['nodes']['mass'][:,0]
            maxima[start:start+len(rows)]=rows['nodes']['maximum'][:,0]
            winners[start:start+len(rows)]=rows['nodes']['winner'][:,0]
        assert np.isfinite(masses).all() and np.all(masses>=0) and np.all(masses.sum(axis=1)>0)
        qsource=masses/masses.sum(axis=1)[:,None]
        membership=json.loads(Path(b['data']['membership.json']['path']).read_bytes());assert [v['bank'] for v in membership]==list(range(12))
        ctx.r['gates']['ALL_UID_roles_links_and_root_target_schema']=True
        original_out=Path(b['original_out'])
        first=json.loads(Path(b['inherited']['raw']['path']).read_bytes())
        first_ret=json.loads(Path(b['inherited']['retention']['path']).read_bytes())
        assert first['completed_roots']==2 and all(first['gates'].values()) and first_ret['main_completed'] is False and not first_ret['numerical_imports']
        binary=Path(b['inherited_binary']['path'])
        ctx.r['inherited_saved_updates']=64;ctx.r['new_optimizer_updates']=0
        ctx.r['gates']['original_compile_sigmoid_AVX_FPU_negative_controls_retained_pending_numerical_audit']=True
        ctx.phase='ALL12_development_only_fixed32_Adam_mass_fit'
        heads=np.empty((12,769),'<f4');models=[];native_expected=np.empty(U,'<f4');standardized_real=np.empty(U,'<f8')
        for bank in range(12):
            ctx.quiet('fit_bank'+str(bank))
            indices=np.flatnonzero(meta[:,2]==bank);count=len(indices)
            assert count==(22272 if bank<6 else 17540)
            x=np.empty((count,D),'<f4');score=np.empty((count,128),'<f4');values=np.empty((count,3),'<f8');at=0
            for _,rows in chunks('unique_inputs.bin',b'M479UNI1',3720,768,U,UNIQUE):
                selected=rows[rows['meta'][:,2]==bank];n=len(selected)
                if n:
                    assert np.array_equal(selected['meta'][:,0],indices[at:at+n]);x[at:at+n]=selected['input'];score[at:at+n]=selected['score'];values[at:at+n]=selected['value'];at+=n
            assert at==count and np.isfinite(x).all() and np.isfinite(score).all()
            tree=membership[bank]['nodes'];groups=[tree[tree[0][side]]['ids'] for side in ('left','right')]
            assert [len(v) for v in groups]==[64,64] and sorted(groups[0]+groups[1])==list(range(128))
            chosen=np.argmax(score,axis=1);assert np.array_equal(chosen,meta[indices,7])
            terms=np.exp((score-score[np.arange(count),chosen,None]).astype('<f4').astype('<f8')).astype('<f4').astype('<f8')
            root=np.cumsum(terms,axis=1,dtype='<f8')[:,-1]
            assert root.tobytes()==values[:,2].tobytes()
            assert (1/root).astype('<f4').tobytes()==values[:,1].astype('<f4').tobytes()
            for side,ids in enumerate(groups):
                ids=np.array(ids);atmax=np.argmax(score[:,ids],axis=1);childwinner=ids[atmax]
                assert childwinner.astype('<u2').tobytes()==winners[indices,side].tobytes()
                assert score[np.arange(count),childwinner].tobytes()==maxima[indices,side].tobytes()
                assert np.cumsum(terms[:,ids],axis=1,dtype='<f8')[:,-1].tobytes()==masses[indices,side].tobytes()
            local=np.flatnonzero(dev[indices]);train_uid=indices[local]
            omega_global=np.zeros(U,'<f8');books=np.zeros(U,'<u4')
            for book in [*range(64),*range(128,192)]:
                selected=np.unique(links[(links[:,2]==book)&(links[:,6]==bank),12]);assert len(selected) and np.all(dev[selected])
                omega_global[selected]+=1/(128*len(selected));books[selected]+=1
            assert np.array_equal(books,owner[:,6]*(owner[:,1]==bank)) and np.all(omega_global[val]==0)
            omega=omega_global[train_uid];assert len(local)==(14848 if bank<6 else 11721)
            def stepguard():ctx.guard();ctx.quiet('optimizer_step_bank'+str(bank))
            path=(original_out if bank<2 else OUT)/('bank%02d_history.npz'%bank)
            if bank<2:
                with np.load(path,allow_pickle=False) as archive:history={k:archive[k].copy() for k in archive.files}
                assert np.array_equal(history['training_uid'],train_uid) and np.array_equal(history['omega'],omega)
            else:
                history=M.fit(x[local],qsource[train_uid,1],omega,stepguard)
                np.savez(path,**history,training_uid=train_uid.astype('<u4'),omega=omega)
                ctx.r['new_optimizer_updates']+=32
            heads[bank]=history['head'];theta=history['after'][-1]
            for start in range(0,len(x),2048):
                part=x[start:start+2048];partids=indices[start:start+len(part)]
                native_expected[partids]=M.ordered_logits(part,heads[bank])
                full=part.astype('<f8');full-=history['mu'];full/=history['sigma']
                standardized_real[partids]=full@theta[:-1]+theta[-1];del full

            models.append({'bank':bank,'unique_inputs':count,'development_inputs':len(local),'consumed_validation_inputs':int(val[indices].sum()),
                           'history_path':str(path),'steps':32,'training_weight_sum':float(omega.sum()),
                           'initial_loss':float(history['loss'][0]),'final_loss':float(history['final_loss'])})
            ctx.r['completed_roots']=bank+1;ctx.log(bank_complete=bank,development=len(local));ctx.guard()
            del x,score,terms,values,history,omega_global,books,theta
        ctx.r['gates']['ALL12_original_root_targets_and_development_only_384_updates']=True
        with (OUT/'heads.bin').open('xb') as f:f.write(struct.pack('<8sIIQ',b'M490HED1',3076,768,12));f.write(heads.tobytes())
        ctx.phase='ALL238872_native_root_predictions_and_complete_views'
        ctx.run([binary,'predict',OUT/'heads.bin',b['data']['unique_inputs.bin']['path'],OUT/'predictions.bin'],'predict')
        dtype=np.dtype([('uid','<u4'),('logit','<f4'),('pair','<f4',(2,))])
        with (OUT/'predictions.bin').open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',b'M490PRD1',16,0,U);pred=np.frombuffer(f.read(),dtype=dtype)
        assert len(pred)==U and np.array_equal(pred['uid'],uid) and pred['logit'].tobytes()==native_expected.tobytes()
        assert np.isfinite(pred['pair']).all() and np.all((pred['pair']>=0)&(pred['pair']<=1))
        sum_error=np.abs(pred['pair'].astype('<f8').sum(axis=1)-1)
        assert np.all(sum_error<=2**-24)
        right=(maxima[:,1]>maxima[:,0])|((maxima[:,0]==maxima[:,1])&(winners[:,1]<winners[:,0]))
        q=pred['pair'][uid,right.astype(int)].astype('<f8');source=qsource[uid,right.astype(int)]
        assert np.all(source>0)
        log_error=np.full(U,np.inf);positive=q>0;log_error[positive]=np.abs(np.log(q[positive])-np.log(source[positive]))
        relative=np.abs(q-source)/source
        diagnostics=np.dtype([('uid','<u4'),('source_pair','<f8',(2,)),('selected_right','<u4'),('log_error','<f8'),('relative_error','<f8'),('standardized_real_logit','<f8')])
        diag=np.empty(U,diagnostics);diag['uid']=uid;diag['source_pair']=qsource;diag['selected_right']=right;diag['log_error']=log_error;diag['relative_error']=relative;diag['standardized_real_logit']=standardized_real
        with (OUT/'diagnostics.bin').open('xb') as f:f.write(struct.pack('<8sIIQ',b'M490DIA1',diagnostics.itemsize,0,U));f.write(diag.tobytes())
        def summarize(indices):
            n=len(indices)
            if not n:return {'count':0,'over_log_budget':0,'zero_candidate_selected':0,'max_log_error':None,'mean_log_error':None,'max_relative_error':None}
            errors=log_error[indices];finite=np.isfinite(errors)
            return {'count':n,'over_log_budget':int(np.count_nonzero(errors>M.ETA)),'zero_candidate_selected':int(np.count_nonzero(q[indices]==0)),
                    'max_log_error':float(errors.max()) if finite.all() else None,'mean_log_error':float(errors.mean()) if finite.all() else None,
                    'max_relative_error':float(relative[indices].max())}
        reports={k:[] for k in ('unique','occurrence','book','source_ID')}
        for bank in [-1,*range(12)]:
            mask=np.ones(U,bool) if bank<0 else meta[:,2]==bank
            for role,r in (('ALL',np.ones(U,bool)),('development',dev),('consumed_validation',val)):
                reports['unique'].append({'bank':bank,'role':role,**summarize(np.flatnonzero(mask&r))})
        for bank in range(12):
            for role in range(3):
                for mode in range(2):
                    rows=links[(links[:,6]==bank)&(links[:,5]==role)&(links[:,4]==mode)]
                    reports['occurrence'].append({'bank':bank,'role':role,'mode':mode,**summarize(rows[:,12])})
                    for expert in range(128):reports['source_ID'].append({'bank':bank,'role':role,'mode':mode,'source_ID':expert,**summarize(rows[rows[:,9]==expert,12])})
        for book in range(192):
            bookrows=links[links[:,2]==book]
            for bank in range(12):
                for mode in range(2):
                    rows=bookrows[(bookrows[:,6]==bank)&(bookrows[:,4]==mode)]
                    reports['book'].append({'book':book,'bank':bank,'mode':mode,**summarize(rows[:,12])})
        assert [len(reports[k]) for k in ('unique','occurrence','book','source_ID')]==[39,72,4608,9216]
        for k in ('occurrence','book','source_ID'):assert sum(r['count'] for r in reports[k])==Q
        write(OUT/'complete_reports.json',reports)
        ctx.r['gates']['ALL_finite_C_dot_pair_and_complete_occurrence_book_ID_reports']=True
        feasibility={'EVERY_development_selected_root_log_error_within_eta_over7':bool(np.all(log_error[dev]<=M.ETA)),
                     'EVERY_consumed_validation_selected_root_log_error_within_eta_over7':bool(np.all(log_error[val]<=M.ETA))}
        result=ctx.finish({'models':models,'unique_inputs':U,'occurrences':Q,'optimizer_updates':384,'steps_per_root':32,'learning_rate':.03,
                          'log_error_budget':M.ETA,'source_zero_root_child_mass_fields':int(np.count_nonzero(masses==0)),
                          'maximum_pair_sum_error':float(sum_error.max()),'validation_training_examples':0,
                          'feasibility':feasibility,'overall':reports['unique'][:3],
                          'decision':'ROOT_RECIPE_ELIGIBLE_PENDING_ALL_NODES_AND_WINNER' if all(feasibility.values()) else 'FIXED_ROOT_MASS_RECIPE_FAIL_NOT_A_CLASS_IMPOSSIBILITY',
                          'scope':'No winner fitting, new capture, model inference, fresh held-out quality, LUT timing or DRAM claim.'})
        print(json.dumps({'raw':str(RAW),'gates':result['gates'],'feasibility':feasibility,'resource':result['resource']}),flush=True)
    except BaseException as e:
        ctx.fail(e);ctx.done.set();ctx.watchdog.cancel();traceback.print_exc();raise

if __name__=='__main__':main()
