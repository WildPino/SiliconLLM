"""CPU-only full probabilities and gradient audit of stored GPUF64 outputs."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'; SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes()); terminal=json.loads(a.terminal.read_bytes()); assert terminal['exit_code']==0 and terminal['result']==extent(a.report)
    assert extent(r['binding']['path'])==r['binding']; b=json.loads(Path(r['binding']['path']).read_bytes())
    proc=psutil.Process(); proc.cpu_affinity(list(range(6))); maxrel=maxscalar=0.
    extra=[extent(__file__),extent(DOC/'NATIVE_HEAD_GPU_PREFLIGHT_AUDIT_PROTOCOL_20261012.md'),extent(a.report),extent(a.terminal),r['binding']]
    receipt=dict(schema='NATIVE_HEAD_GPU_PREFLIGHT_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():assert time.monotonic()-START<=600 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+r['outputs']+extra:guard(); assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def compare(x,y,tolerance=1e-9):
        nonlocal maxrel
        error=float(np.linalg.norm(x-y)/max(1.,float(np.linalg.norm(y)))); maxrel=max(maxrel,error); assert error<=tolerance,error
    def scalar(x,y,tolerance=1e-8):
        nonlocal maxscalar
        error=abs(float(x)-float(y)); maxscalar=max(maxscalar,error); assert error<=tolerance*max(1.,abs(float(y))),(x,y)
    try:
        check(); nr=b['prior']['native_report']; C=np.fromfile(nr['head_F32']['path'],'<f4').reshape(65537,256).astype('f8'); beta=np.fromfile(nr['bias_F32']['path'],'<f4').astype('f8')
        epsilon=float(np.fromfile(nr['epsilon_F32']['path'],'<f4')[0]); augmented=np.column_stack([C,beta/math.sqrt(256*epsilon)])
        rows=0; probability_words=0; splitchecks=[]
        for s in r['splits']:
            old=load(next(x for x in b['preflight']['splits'] if x['split']==s['split'])['file']); stored=load(s['file']); n=len(old['v']); full_batches=0
            assert stored['loss'].dtype==stored['gradient'].dtype==np.dtype('f8') and stored['ids'].dtype==np.dtype('i8')
            assert stored['gradient'].shape==(n,257) and stored['loss'].shape==stored['ids'].shape==(n,)
            for lo in range(0,n,64):
                guard(); hi=min(lo+64,n); v=old['v'][lo:hi]; m=old['moments'][lo:hi]; H=old['entropy'][lo:hi]
                score=np.zeros((hi-lo,65537),'f8')
                for col in range(0,256,32):score+=v[:,col:col+32]@C[:,col:col+32].T
                score+=v[:,-1,None]*(beta/math.sqrt(256*epsilon)); logz=np.logaddexp.reduce(score,axis=1); probability=np.exp(score-logz[:,None])
                compare(probability.sum(axis=1),np.ones(hi-lo),1e-11)
                gradient=-m.copy()
                for vocab in range(0,65537,4096):gradient+=probability[:,vocab:vocab+4096]@augmented[vocab:vocab+4096]
                linear=np.zeros(hi-lo,'f8')
                for col in range(0,257,8):linear+=np.sum(m[:,col:col+8]*v[:,col:col+8],axis=1)
                loss=logz-linear-H
                compare(stored['loss'][lo:hi],loss); compare(stored['gradient'][lo:hi],gradient)
                compare(old['loss'][lo:hi],loss); compare(old['gradient'][lo:hi],gradient)
                assert np.array_equal(stored['ids'][lo:hi],np.argmax(score,axis=1)) and np.array_equal(old['ids'][lo:hi],stored['ids'][lo:hi])
                batch=s['batches'][full_batches]; assert batch['lo']==lo and batch['hi']==hi and batch['FG_seconds']>=0
                full_batches+=1; rows+=hi-lo; probability_words+=(hi-lo)*65537
            calc=dict(loss_absolute=float(np.max(np.abs(stored['loss']-old['loss']))),gradient_absolute=float(np.max(np.abs(stored['gradient']-old['gradient']))),
                      gradient_relative=float(np.linalg.norm(stored['gradient']-old['gradient'])/max(1.,float(np.linalg.norm(old['gradient'])))),argmax_changes=int(np.count_nonzero(stored['ids']!=old['ids'])))
            flags={k:calc[k]<=b['numeric'][k] for k in calc}; assert flags==s['flags']
            for key in calc:scalar(calc[key],s['numeric'][key])
            assert s['rows']==n and len(s['batches'])==full_batches
            splitchecks.append(dict(split=s['split'],rows=n,flags=flags,probability_rows_full_v=True))
            print(json.dumps(dict(audited=s['split'],rows=n,seconds=time.monotonic()-START)),flush=True)
        assert rows==8808 and probability_words==577249896 and r['calls']==148 and r['all148_calls_float64'] and r['repeated_outputs_bit_exact']
        assert r['gpu']==b['gpu'] and r['versions']==b['versions'] and not r['optimization_implemented'] and not r['timing_admitted']
        bench=r['benchmark']; assert len(bench['GPU_seconds'])==7 and bench['batch']==64 and not bench['admission']
        scalar(np.median(bench['GPU_seconds']),bench['median_seconds']); scalar(bench['median_seconds']*139,bench['nominal139_batch_evaluations_seconds'])
        scalar(bench['CPU_prior_median_seconds'],b['preflight']['benchmark']['median_seconds']); scalar(bench['CPU_prior_nominal_seconds'],b['preflight']['benchmark']['nominal139_batch_evaluations_seconds'])
        allpass=all(all(s['flags'].values()) for s in splitchecks); assert r['decision']==('GPU_INITIAL_NUMERIC_PASS_PENDING_INDEPENDENT_AUDIT' if allpass else 'COMPLETED_GPU_INITIAL_NUMERIC_FAILURE')
        assert r['OS_peak']<=b['limits']['OS_bytes'] and r['GPU_allocated_peak']<=b['limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=b['limits']['GPU_reserved_bytes']
        assert r['seconds']<=b['limits']['seconds'] and sum(x['bytes'] for x in r['outputs'])<=b['limits']['output_bytes']
        check(); guard()
        result=dict(schema='NATIVE_HEAD_GPU_PREFLIGHT_AUDIT_V1',source=extent(__file__),freeze=a.freeze,report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],
                    rows=rows,full_probability_words=probability_words,all8808_full_loss_gradients_and_ids_verified=True,split_checks=splitchecks,max_relative=maxrel,max_scalar_delta=maxscalar,
                    before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',GPU_numeric_pass=allpass,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
                    scope='Independent CPU full-vocabulary moment objective/probability/gradient audit of stored GPUF64 output. No GPU replay/optimization/native-code/quality/speed admission.')
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset); write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
