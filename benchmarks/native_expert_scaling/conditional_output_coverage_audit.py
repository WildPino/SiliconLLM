"""Independent full expert/error matrix via norm-minus-projected-energy identity."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    re,te=extent(a.report),extent(a.terminal)
    assert re['sha256']==a.report_sha and te['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes())
    t=json.loads(a.terminal.read_bytes())
    assert t['exit_code']==0 and t['result']==re and extent(r['binding']['path'])==r['binding']
    b=json.loads(Path(r['binding']['path']).read_bytes())
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists()
    proc=psutil.Process()
    proc.cpu_affinity(list(range(6)))
    stage='startup'
    maxima={}
    gates={}
    rows=[]
    count=0
    countIDs=0
    independent_argmin_differences=0
    negative=0
    receipt=dict(schema='CONDITIONAL_OUTPUT_COVERAGE_AUDIT_TERMINAL_V1',command=sys.orig_argv,
                 pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['audit_seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[r['binding'],re,te]+[q['file'] for q in r['records']]:
            guard()
            assert extent(e['path'])==e,e['path']
    def metric(key,value,tol):
        value=float(value)
        assert np.isfinite(value)
        maxima[key]=max(maxima.get(key,0.),value)
        gates[key]=bool(gates.get(key,True) and value<=tol)
    def meta(x,y):
        if isinstance(y,dict):
            assert x.keys()==y.keys()
            for k in y:
                meta(x[k],y[k])
        elif isinstance(y,(bool,int,str)):
            assert x==y,(stage,x,y)
        else:
            metric('metadata_absolute_or_relative',abs(x-y)/max(abs(y),1.),b['numeric']['scalar_absolute_or_relative'])
    try:
        check()
        ad=json.loads(Path(b['parent_adjudication']['path']).read_bytes())
        assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==b['parent_report']
        assert len(r['records'])==144
        for model in b['models']:
            site=model['site']
            banks={}
            for q in model['banks']:
                with np.load(q['file']['path'],allow_pickle=False) as z:
                    banks[q['name']]=(z['U'].copy(),z['mean_Y'].copy())
            for e in [q for q in b['records'] if q['site']==site]:
                stage=f'{site}/{e["id"]}'
                guard()
                with np.load(e['native']['path'],allow_pickle=False) as z:
                    Y=z['reconstructed_denominator__raw_scores'].astype('f8')
                original=next(q for q in r['records'] if q['site']==site and q['id']==e['id'])
                with np.load(original['file']['path'],allow_pickle=False) as z:
                    arrays={k:z[k].copy() for k in z.files}
                routefile=next(q['file'] for q in b['route_packets'] if q['site']==site and q['id']==e['id'])
                assert set(arrays)=={name+'__'+key for name in banks for key in ['all_errors','oracle_IDs','actual_IDs']}
                stats={}
                for name,(U,means) in banks.items():
                    K=len(U)
                    errors=arrays[name+'__all_errors']
                    oracle=arrays[name+'__oracle_IDs']
                    actual=arrays[name+'__actual_IDs']
                    assert errors.dtype==np.dtype('<f8') and errors.shape==(e['history'],K) and (errors>=0).all()
                    assert np.isfinite(errors).all() and oracle.dtype==actual.dtype==np.dtype('<u4')
                    with np.load(routefile['path'],allow_pickle=False) as z:
                        assert np.array_equal(actual,z[name+'__IDs'])
                    assert np.array_equal(oracle,np.argmin(errors,axis=1).astype('u4'))
                    reference=np.empty_like(errors)
                    for j in range(K):
                        guard()
                        u=U[j]
                        # The parent basis orthogonality is qualified; use the full
                        # Gram correction so this identity also includes its roundoff.
                        gram=u.T@u
                        for lo in range(0,len(Y),32):
                            centered=Y[lo:lo+32]-means[j]
                            p=centered@u
                            norm=np.einsum('td,td->t',centered,centered)
                            pnorm=np.einsum('tr,tr->t',p,p)
                            corrected=np.einsum('tr,rs,ts->t',p,gram,p,optimize=True)
                            err=norm-2*pnorm+corrected
                            reference[lo:lo+32,j]=err
                            negative+=int((err<0).sum())
                            scaled=np.abs(errors[lo:lo+32,j]-err)/np.maximum(norm,1e-30)
                            metric('token_error_scaled_absolute',float(np.max(scaled)),b['numeric']['token_error_scaled_absolute'])
                    # Argmin labels may differ on nearly exact/near-tied projections.
                    # Preserve the measured count; all stored labels are exact minima
                    # of the independently numerically verified full error matrix.
                    independent_argmin_differences+=int(np.count_nonzero(np.argmin(reference,axis=1)!=oracle))
                    count+=errors.size
                    countIDs+=2*len(oracle)
                    idx=np.arange(len(Y))
                    best=errors[idx,oracle]
                    routed=errors[idx,actual]
                    assert (best<=routed).all()
                    total=float(np.einsum('td,td->',Y,Y))
                    osse=float(best.sum())
                    rsse=float(routed.sum())
                    pos=e['positions']
                    labelenergy=float(np.sum(Y[pos]**2))
                    stats[name]=dict(oracle_relative_RMS=float(np.sqrt(osse/total)),
                         routed_projection_relative_RMS=float(np.sqrt(rsse/total)),oracle_squared_energy=osse,
                         routed_squared_energy=rsse,target_energy=total,selector_excess_relative_squared=float((rsse-osse)/total),
                         oracle_ID_different_from_actual=int(np.count_nonzero(oracle!=actual)),
                         oracle_label_relative_RMS=float(np.sqrt(best[pos].sum()/labelenergy)),
                         oracle_case_one_percent=float(np.sqrt(osse/total))<=b['criterion_each_case'])
                meta(original['metrics'],stats)
                rows.append(dict(id=e['id'],site=site,split=e['split'],domain=e['domain'],metrics=stats))
                if len(rows)%24==0:
                    print(json.dumps(dict(complete=len(rows),seconds=time.monotonic()-START)),flush=True)
        aa={}
        for site in [0,12,23]:
            aa[str(site)]={}
            for split in ['FIT','DEV']:
                rr=[q for q in rows if q['site']==site and q['split']==split]
                assert len(rr)==24
                aa[str(site)][split]={}
                for name in banks:
                    v=[q['metrics'][name] for q in rr]
                    aa[str(site)][split][name]=dict(oracle_mean=float(np.mean([q['oracle_relative_RMS'] for q in v])),
                         oracle_worst=max(q['oracle_relative_RMS'] for q in v),routed_mean=float(np.mean([q['routed_projection_relative_RMS'] for q in v])),
                         oracle_one_percent_passed=sum(q['oracle_case_one_percent'] for q in v),
                         oracle_different_IDs=sum(q['oracle_ID_different_from_actual'] for q in v),
                         selector_excess_mean_relative_squared=float(np.mean([q['selector_excess_relative_squared'] for q in v])))
        meta(r['aggregates'],aa)
        assert all(r[k]==0 for k in ['source_forwards','native_calls','basis_updates','router_updates'])
        check()
        guard()
        result=dict(schema='CONDITIONAL_OUTPUT_COVERAGE_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,
                    report=re,terminal=te,binding=r['binding'],all_token_expert_errors_verified=count,
                    all_actual_and_oracle_IDs_verified=countIDs,negative_roundoff_errors_in_identity=negative,
                    independent_identity_argmin_differences=independent_argmin_differences,
                    maximum=maxima,verification_gates=gates,aggregates=aa,before_after_hashes_exact=True,
                    decision='INDEPENDENT_STORED_PASS' if all(gates.values()) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',
                    seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
                    scope='Every frozen affine plane/every FIT+DEV token error by independent expanded norm identity, '
                          'full minima/actual route excess and statistics. Target-aware all-expert diagnosis, '
                          'not an admissible router or native/chatbot quality-speed claim.')
        write(a.out,result)
        receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(adjudication=extent(a.out),decision=result['decision'],maximum=maxima,
                              errors=count,independent_argmin_differences=independent_argmin_differences)),flush=True)
    except BaseException as ex:
        import traceback
        receipt.update(exit_code=1,error=repr(ex),stage=stage,traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['report','terminal','out']:
        p.add_argument('--'+k,type=Path,required=True)
    for k in ['report-sha','terminal-sha','freeze']:
        p.add_argument('--'+k,required=True)
    main(p.parse_args())
