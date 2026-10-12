"""Frozen positive affine rays of actual F32 readout; oracle, not an encoder."""
import argparse, ctypes, json, math, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'; SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def prepare(a):
    import numpy as np,psutil
    report=DOC/'native_head_domain_preflight_result_v2_20261012.json'; audit=DOC/'native_head_domain_preflight_adjudication_v2_20261012.json'
    p=json.loads(report.read_bytes()); ad=json.loads(audit.read_bytes())
    assert ad['report']==extent(report) and ad['decision']=='INDEPENDENT_STORED_PASS'
    prior=json.loads(Path(p['binding']['path']).read_bytes()); native=prior['native_report']
    engine_path=DOC/'native_scalar_readout_preparation_20261012.json'; engine=json.loads(engine_path.read_bytes())
    reference_path=DOC/'gauged_native_reference_preparation_20261012.json'; reference=json.loads(reference_path.read_bytes())
    ids=np.concatenate([np.load(s['file']['path'],allow_pickle=False)['teacher_ids'] for s in p['splits']]); targets=np.unique(ids)
    assert len(ids)==8808 and len(targets)==2361
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',report,audit,Path(p['binding']['path']),engine_path,reference_path,
           DOC/'NATIVE_HEAD_AFFINE_RAYS_PROTOCOL_20261012.md',Path(sys.executable),Path(sys.executable).parent/'python312.dll',
           Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(psutil.__file__)]
    paths += [Path(native[k]['path']) for k in ['head_F32','bias_F32','epsilon_F32']]
    paths += [Path(s['file']['path']) for s in p['splits']]
    paths += [Path(engine[k]['path']) for k in ['source','dll','log','compiler','engine']]
    paths += [Path(reference[k]['path']) for k in ['dll','log','assembly']]
    paths += [Path(x['path']) for x in reference['inputs']]
    paths += list((SITE/'numpy.libs').glob('*.dll'))+list((SITE/'psutil').glob('*.pyd'))
    binding=dict(schema='NATIVE_HEAD_AFFINE_RAYS_BINDING_V1',python=str(Path(sys.executable).resolve()),preflight=p,prior=prior,
                 engine=engine,reference=reference,targets=targets.tolist(),batch=64,threads=6,margins=[0.,1.],
                 scalar_cap=1e6,scalar_growth=1.01,scalar_add=.01,
                 precision=dict(score_absolute=1e-4,norm_relative=2e-5,inverse_relative=2e-5),
                 limits=dict(seconds=900,OS_bytes=4<<30,output_bytes=1<<30),
                 inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
                 scope='All2361 distinct donor winners/all8808 existing-label associations. Head-derived fixed positive '
                       'rays,full affine0/1-margin intervals and actual F32 kernel codes. Oracle target lookup,not causal '
                       'routing/encoder/chatbot;empty interval excludes this ray only. Full real categorical KL with '
                       'audited sufficient moments and native score-range bounds,not exact native teacher KL.')
    write(a.out,binding); print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha; b=json.loads(a.binding.read_bytes())
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert all(os.environ.get(k)==str(b['threads']) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'])
    assert not a.directory.exists() and not a.out.exists()
    proc=psutil.Process(); proc.cpu_affinity(list(range(6))); mapping=None
    receipt=dict(schema='NATIVE_HEAD_AFFINE_RAYS_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard(); assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def save(name,**fields):
        p=a.directory/(name+'.npz'); np.savez(p,**fields); return extent(p)
    def intervals(slopes,rhs,targets):
        rows=[]
        for index,j in enumerate(targets):
            use=np.arange(65537)!=j; aa=slopes[index]; bb=rhs[index]; positive=(aa>0)&use; negative=(aa<0)&use; zero=(aa==0)&use
            lower=0.; upper=math.inf; lower_k=upper_k=-1
            if positive.any():
                kk=np.flatnonzero(positive); values=bb[kk]/aa[kk]; k=int(np.argmax(values))
                if values[k]>0:lower=float(values[k]); lower_k=int(kk[k])
            if negative.any():
                kk=np.flatnonzero(negative); values=bb[kk]/aa[kk]; k=int(np.argmin(values)); upper=float(values[k]); upper_k=int(kk[k])
            bad=np.flatnonzero(zero&(bb>0)); zero_k=int(bad[0]) if len(bad) else -1
            feasible=zero_k<0 and math.isfinite(lower) and lower<=upper
            rows.append([lower,upper,lower_k,upper_k,zero_k,int(feasible)])
        return np.array(rows,'f8')
    try:
        check(); a.directory.mkdir(); native=b['prior']['native_report']; targets=np.array(b['targets'],'i8'); n=len(targets)
        C32=np.fromfile(native['head_F32']['path'],'<f4').reshape(65537,256); beta32=np.fromfile(native['bias_F32']['path'],'<f4')
        eps=float(np.fromfile(native['epsilon_F32']['path'],'<f4')[0]); C=C32.astype('f8'); beta=beta32.astype('f8'); center=C.mean(axis=0)
        directions=C[targets]-center; lengths=np.sqrt(np.sum(directions**2,axis=1)); assert np.all(lengths>0); directions/=lengths[:,None]
        zero_intervals=np.empty((n,6),'f8'); margin_intervals=np.empty_like(zero_intervals); scalars=np.full(n,np.nan,'f8'); code_rows=np.full(n,-1,'i8'); t_list=[]
        for lo in range(0,n,64):
            guard(); hi=min(lo+64,n); jj=targets[lo:hi]; score=directions[lo:hi]@C.T
            slopes=score[np.arange(hi-lo),jj,None]-score; rhs=beta[None,:]-beta[jj,None]
            zero_intervals[lo:hi]=intervals(slopes,rhs,jj); margin_intervals[lo:hi]=intervals(slopes,rhs+1.,jj)
            for row in range(lo,hi):
                lower,upper,_,_,_,feasible=margin_intervals[row]
                if not feasible:continue
                candidate=max(lower*b['scalar_growth'],lower+b['scalar_add'])
                if candidate>upper:candidate=lower+(upper-lower)/2
                if not math.isfinite(candidate) or candidate>b['scalar_cap']:continue
                scalars[row]=candidate; code_rows[row]=len(t_list); t_list.append(candidate*directions[row])
        assert t_list,'No candidate codes: preserve terminal and diagnose,not a general head capacity theorem'
        t64=np.array(t_list,'f8'); t32=np.ascontiguousarray(t64,dtype='<f4'); valid=np.flatnonzero(code_rows>=0); winner_targets=targets[valid]
        radius=np.sqrt(np.mean(t32.astype('f8')**2,axis=1)+eps); v=np.column_stack([t32.astype('f8')/radius[:,None],math.sqrt(256*eps)/radius])
        normalized=np.empty_like(t32); inverse=np.empty(len(t32),'<f4'); ids=np.empty(len(t32),'i8'); real_ids=np.empty_like(ids)
        native_margin=np.empty(len(t32),'f8'); real_margin=np.empty_like(native_margin); raw_margin=np.empty_like(native_margin)
        logz=np.empty_like(native_margin); error_range=np.empty_like(native_margin); absolute_error=np.empty_like(native_margin)
        scorepath=a.directory/'native_scores.f32'; mapping=np.memmap(scorepath,mode='w+',dtype='<f4',shape=(len(t32),65537))
        dll=ctypes.CDLL(b['engine']['dll']['path']); fn=dll.scalar_readout
        fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_float,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]; fn.restype=ctypes.c_int
        batches=[]
        for lo in range(0,len(t32),64):
            guard(); hi=min(lo+64,len(t32)); count=hi-lo; out=np.empty((count,65537),'<f4'); began=time.monotonic()
            assert fn(C32.ctypes.data,beta32.ctypes.data,t32[lo:hi].ctypes.data,count,eps,normalized[lo:hi].ctypes.data,inverse[lo:hi].ctypes.data,out.ctypes.data)==0
            seconds=time.monotonic()-began; mapping[lo:hi]=out
            raw=t32[lo:hi].astype('f8')@C.T+beta; real=raw/radius[lo:hi,None]; delta=out.astype('f8')-real
            logz[lo:hi]=real.max(axis=1)+np.log(np.exp(real-real.max(axis=1,keepdims=True)).sum(axis=1))
            error_range[lo:hi]=delta.max(axis=1)-delta.min(axis=1); absolute_error[lo:hi]=np.max(np.abs(delta),axis=1)
            ids[lo:hi]=np.argmax(out,axis=1); real_ids[lo:hi]=np.argmax(real,axis=1)
            target=winner_targets[lo:hi]; real_target=real[np.arange(count),target]; raw_target=raw[np.arange(count),target]; native_target=out[np.arange(count),target].astype('f8')
            other_real=real.copy(); other_real[np.arange(count),target]=-np.inf
            other_raw=raw.copy(); other_raw[np.arange(count),target]=-np.inf
            other_native=out.astype('f8'); other_native[np.arange(count),target]=-np.inf
            real_margin[lo:hi]=real_target-other_real.max(axis=1); raw_margin[lo:hi]=raw_target-other_raw.max(axis=1); native_margin[lo:hi]=native_target-other_native.max(axis=1)
            normerr=float(np.linalg.norm(normalized[lo:hi].astype('f8')-v[lo:hi,:256])/np.linalg.norm(v[lo:hi,:256])); inverr=float(np.max(np.abs(inverse[lo:hi].astype('f8')*radius[lo:hi]-1)))
            batches.append(dict(lo=lo,hi=hi,seconds=seconds,norm_relative=normerr,inverse_relative=inverr,score_absolute=float(absolute_error[lo:hi].max())))
        mapping.flush(); mapping._mmap.close(); mapping=None
        codes=save('ray_codes',targets=targets,center=center,directions=directions,direction_lengths=lengths,zero_intervals=zero_intervals,
                   margin_intervals=margin_intervals,scalars=scalars,code_rows=code_rows,t=t64,t_f32=t32,v=v,normalized_f32=normalized,inverse_f32=inverse,
                   ids=ids,real_ids=real_ids,raw_margin=raw_margin,real_margin=real_margin,native_margin=native_margin,logz=logz,error_range=error_range,absolute_error=absolute_error)
        scores=extent(scorepath); outputs=[codes,scores]; splits=[]
        for s in b['preflight']['splits']:
            old=load(s['file']); target_indices=np.searchsorted(targets,old['teacher_ids']); rows=code_rows[target_indices]; available=rows>=0
            loss=np.full(len(rows),np.nan,'f8'); kl_lower=loss.copy(); kl_upper=loss.copy(); actual_ids=np.full(len(rows),-1,'i8')
            rr=rows[available]; loss[available]=logz[rr]-np.sum(old['moments'][available]*v[rr],axis=1)-old['entropy'][available]
            kl_lower[available]=np.maximum(0.,loss[available]-error_range[rr]); kl_upper[available]=loss[available]+error_range[rr]; actual_ids[available]=ids[rr]
            successful=available&(actual_ids==old['teacher_ids']); w=old['weights']
            packet=save(s['split']+'_associations',teacher_ids=old['teacher_ids'],target_indices=target_indices,code_rows=rows,available=available,native_ids=actual_ids,
                        real_KL=loss,native_KL_lower=kl_lower,native_KL_upper=kl_upper,weights=w)
            outputs.append(packet); splits.append(dict(split=s['split'],rows=len(rows),file=packet,candidate_labels=int(available.sum()),native_winner_labels=int(successful.sum()),
                                                       case_candidate_fraction=float(w@available),case_native_winner_fraction=float(w@successful),
                                                       available_weighted_real_KL=float(w[available]@loss[available]),
                                                       available_weighted_native_KL_lower=float(w[available]@kl_lower[available]),available_weighted_native_KL_upper=float(w[available]@kl_upper[available])))
        precision={k:all(x[k]<=b['precision'][k] for x in batches) for k in b['precision']}
        check(); guard()
        result=dict(schema='NATIVE_HEAD_AFFINE_RAYS_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,targets=n,zero_interval_nonempty=int(zero_intervals[:,5].sum()),
                    margin1_interval_nonempty=int(margin_intervals[:,5].sum()),candidate_codes=len(t32),native_target_winners=int(np.count_nonzero(ids==winner_targets)),
                    native_to_real_ID_changes=int(np.count_nonzero(ids!=real_ids)),minimum_raw_margin=float(raw_margin.min()),minimum_native_margin=float(native_margin.min()),
                    maximum_score_absolute=float(absolute_error.max()),native_calls=len(batches),batches=batches,precision_flags=precision,outputs=outputs,codes=codes,scores=scores,splits=splits,
                    decision='COMPLETED_RAY_ORACLE_PENDING_INDEPENDENT_AUDIT',before_after_hashes_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
                    scope=b['scope'],general_head_infeasibility_proved=False,causal_encoder_implemented=False,timing_admitted=False)
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(targets=n,candidates=len(t32),native_target_winners=result['native_target_winners'],seconds=result['seconds'])),flush=True)
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        if mapping is not None:mapping._mmap.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset); write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args(); assert a.prepare!=a.run
    prepare(a) if a.prepare else run(a)
