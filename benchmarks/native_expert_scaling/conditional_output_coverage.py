"""Exhaustive target-aware fixed-bank coverage diagnosis; never an admissible router."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def prepare(a):
    import numpy as np,psutil
    rp=DOC/'conditional_output_function_result_20261013.json'
    ap=DOC/'conditional_output_function_adjudication_20261013.json'
    r=json.loads(rp.read_bytes())
    ad=json.loads(ap.read_bytes())
    assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp)
    pb=json.loads(Path(r['binding']['path']).read_bytes())
    stp=DOC/'conditional_output_coverage_storage_20261013.json'
    st=json.loads(stp.read_text(encoding='utf-8-sig'))
    assert not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(st['new_output_directory']).resolve()
    paths=[Path(__file__),B/'conditional_output_coverage_audit.py',B/'falcon_auxiliary_s6_transport.py',
           DOC/'CONDITIONAL_OUTPUT_COVERAGE_PROTOCOL_20261013.md',stp,rp,ap,Path(r['binding']['path']),
           Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),
           Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)]
    nr=Path(np.__file__).resolve().parent
    pr=Path(psutil.__file__).resolve().parent
    paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))
    paths+=[Path(q['file']['path']) for m in r['models'] for q in m['banks']]
    paths+=[Path(q['native']['path']) for q in pb['records']]
    paths+=[Path(q['file']['path']) for q in r['records']]
    paths+=[ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
           'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']]
    b=dict(schema='CONDITIONAL_OUTPUT_COVERAGE_BINDING_V1',freeze=a.freeze,python=str(Path(sys.executable).resolve()),
           versions=dict(numpy=np.__version__,psutil=psutil.__version__),parent_report=extent(rp),
           parent_adjudication=extent(ap),parent_binding=r['binding'],records=pb['records'],models=r['models'],
           route_packets=r['records'],inputs=[extent(p) for p in dict.fromkeys(paths)],
           output_directory=str(a.directory.resolve()),
           numeric=dict(token_error_scaled_absolute=1e-10,scalar_absolute_or_relative=1e-10),
           limits=dict(seconds=900,audit_seconds=1200,OS_bytes=6<<30,output_bytes=64<<20),
           criterion_each_case=.01,
           scope='Exhaustive response-aware oracle over frozen affine image spaces; diagnoses bank '
                 'coverage versus router assignment. Requires true target Y and all experts, '
                 'NOT an executable cheap routing candidate/no retraining/source/native/GPU/DEV '
                 'selection/causal/chatbot/quality-speed/n admission.')
    assert time.monotonic()-START<900 and psutil.Process().memory_info().peak_wset<6<<30
    write(a.out,b)
    print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha
    b=json.loads(a.binding.read_bytes())
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil']
    assert not a.directory.exists() and not a.out.exists() and a.directory.resolve()==Path(b['output_directory']).resolve()
    proc=psutil.Process()
    proc.cpu_affinity(list(range(6)))
    rows=[]
    stage='startup'
    receipt=dict(schema='CONDITIONAL_OUTPUT_COVERAGE_TERMINAL_V1',command=sys.orig_argv,
                 pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
        if a.directory.exists():
            assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:
            guard()
            assert extent(e['path'])==e,e['path']
    try:
        check()
        a.directory.mkdir()
        for model in b['models']:
            site=model['site']
            banks={}
            for q in model['banks']:
                with np.load(q['file']['path'],allow_pickle=False) as z:
                    banks[q['name']]=(z['U'].copy(),z['mean_Y'].copy())
            for r in [q for q in b['records'] if q['site']==site]:
                stage=f'{site}/{r["id"]}'
                guard()
                with np.load(r['native']['path'],allow_pickle=False) as z:
                    Y=z['reconstructed_denominator__raw_scores'].astype('f8')
                assert Y.shape==(r['history'],2048) and np.isfinite(Y).all()
                routefile=next(q['file'] for q in b['route_packets'] if q['site']==site and q['id']==r['id'])
                arrays={}
                stats={}
                for name,(U,means) in banks.items():
                    K=len(U)
                    error=np.empty((len(Y),K))
                    for j in range(K):
                        guard()
                        centered=Y-means[j]
                        residual=centered-(centered@U[j])@U[j].T
                        error[:,j]=np.einsum('td,td->t',residual,residual)
                    assert (error>=0).all() and np.isfinite(error).all()
                    oracle=error.argmin(1).astype('u4')
                    with np.load(routefile['path'],allow_pickle=False) as z:
                        actual=z[name+'__IDs'].copy()
                    optimal=error[np.arange(len(Y)),oracle]
                    routed=error[np.arange(len(Y)),actual]
                    total=float(np.sum(Y*Y))
                    osse=float(optimal.sum())
                    rsse=float(routed.sum())
                    pos=r['positions']
                    labelenergy=float(np.sum(Y[pos]**2))
                    stats[name]=dict(oracle_relative_RMS=float(np.sqrt(osse/total)),
                        routed_projection_relative_RMS=float(np.sqrt(rsse/total)),
                        oracle_squared_energy=osse,routed_squared_energy=rsse,target_energy=total,
                        selector_excess_relative_squared=float((rsse-osse)/total),
                        oracle_ID_different_from_actual=int(np.count_nonzero(oracle!=actual)),
                        oracle_label_relative_RMS=float(np.sqrt(optimal[pos].sum()/labelenergy)),
                        oracle_case_one_percent=float(np.sqrt(osse/total))<=b['criterion_each_case'])
                    arrays[name+'__all_errors']=error
                    arrays[name+'__oracle_IDs']=oracle
                    arrays[name+'__actual_IDs']=actual
                p=a.directory/f'{r["id"]}.site{site:02d}.npz'
                np.savez(p,**arrays)
                rows.append(dict(id=r['id'],site=site,split=r['split'],domain=r['domain'],history=r['history'],
                                 file=extent(p),metrics=stats))
                print(json.dumps(dict(stage=stage,complete=len(rows),seconds=time.monotonic()-START)),flush=True)
        check()
        guard()
        aa={}
        for site in [0,12,23]:
            aa[str(site)]={}
            for split in ['FIT','DEV']:
                rr=[q for q in rows if q['site']==site and q['split']==split]
                aa[str(site)][split]={}
                for name in banks:
                    vals=[q['metrics'][name] for q in rr]
                    aa[str(site)][split][name]=dict(
                        oracle_mean=float(np.mean([q['oracle_relative_RMS'] for q in vals])),
                        oracle_worst=max(q['oracle_relative_RMS'] for q in vals),
                        routed_mean=float(np.mean([q['routed_projection_relative_RMS'] for q in vals])),
                        oracle_one_percent_passed=sum(q['oracle_case_one_percent'] for q in vals),
                        oracle_different_IDs=sum(q['oracle_ID_different_from_actual'] for q in vals),
                        selector_excess_mean_relative_squared=float(np.mean([q['selector_excess_relative_squared'] for q in vals])))
        result=dict(schema='CONDITIONAL_OUTPUT_COVERAGE_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,
                    records=rows,aggregates=aa,before_after_hashes_exact=True,source_forwards=0,native_calls=0,
                    basis_updates=0,router_updates=0,decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',
                    output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),seconds=time.monotonic()-START,
                    OS_peak=proc.memory_info().peak_wset,scope=b['scope'])
        write(a.out,result)
        receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(result=extent(a.out),aggregates=aa)),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true')
    m.add_argument('--run',action='store_true')
    for k in ['binding','directory','out']:
        p.add_argument('--'+k,type=Path)
    for k in ['binding-sha','freeze']:
        p.add_argument('--'+k)
    a=p.parse_args()
    if a.run:
        run(a)
    else:
        import psutil
        proc=psutil.Process()
        receipt=dict(schema='CONDITIONAL_OUTPUT_COVERAGE_PREPARATION_TERMINAL_V1',command=sys.orig_argv,
                     pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
        try:
            prepare(a)
            receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as e:
            import traceback
            receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc())
            raise
        finally:
            receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
            write(a.out.with_suffix('.preparation_terminal.json'),receipt)
