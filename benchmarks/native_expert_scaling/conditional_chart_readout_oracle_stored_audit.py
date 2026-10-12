"""Every fixed-chart decoder/source-head candidate and probability choice."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes())
    assert r['schema']=='CONDITIONAL_CHART_READOUT_ORACLE_RESULT_V1' and t['exit_code']==0 and t['error'] is None and t['result']==extent(a.report)
    b=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding'] and not a.out.exists()
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;maxrel=maxscalar=0.;candidate_count=0
    extra=[extent(a.report),extent(a.terminal),extent(__file__),extent(DOC/'CONDITIONAL_CHART_READOUT_ORACLE_STORED_AUDIT_PROTOCOL_20261012.md')]
    receipt=dict(schema='CONDITIONAL_CHART_READOUT_ORACLE_AUDIT_TERMINAL_V1',command=sys.orig_argv,
        freeze=a.freeze,source=extent(__file__),pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=600 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    def check():
        for item in b['inputs']+r['outputs']+[r['binding']]+extra:guard();assert extent(item['path'])==item,item['path']
    def compare(x,y):
        nonlocal maxrel
        rel=float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)),1e-300));maxrel=max(maxrel,rel);assert rel<=1e-10,rel
    def scalar(x,y):
        nonlocal maxscalar
        d=abs(float(x)-float(y));maxscalar=max(maxscalar,d);assert d<=1e-8*max(1.,abs(float(y))),(x,y)
    def partial(x,A):
        out=np.zeros((*x.shape[:-1],A.shape[0]),'f8')
        for lo in range(0,2048,256):out+=x[...,lo:lo+256]@A[:,lo:lo+256].T
        return out
    try:
        check();get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np)
        cfg=json.loads((Path(b['source'])/'config.json').read_bytes());gamma=get('model.final_layernorm.weight')
        with np.load(b['states']['path'],allow_pickle=False) as pack:H=pack['raw']
        ref=np.fromfile(b['reference']['path'],'<f8').reshape(72,65537);logq=ref-np.logaddexp.reduce(ref,axis=1)[:,None];q=np.exp(logq);true=np.argmax(ref,axis=1)
        for family in r['families']:
            K=family['K'];prior=next(x for x in b['families'] if x['K']==K)
            with np.load(family['file']['path'],allow_pickle=False) as pack:
                assert np.array_equal(pack['raw'],H);decoded=pack['decoded'];radius=pack['true_radius'];own=pack['own_radius'];scores=pack['scores'];stored_losses=pack['KL'];stored_ids=pack['argmax']
            compare(radius,np.sqrt(np.mean(H**2,axis=1)+b['epsilon']))
            for j,chart in enumerate(prior['charts']):
                guard()
                with np.load(chart['file']['path'],allow_pickle=False) as pack:mu=pack['center'];V=pack['V'];W=pack['W']
                z=np.zeros((72,256),'f8')
                for lo in range(0,2048,256):z+=(H[:,lo:lo+256]-mu[lo:lo+256])@W[lo:lo+256]
                compare(decoded[j],mu+z@V.T)
            compare(own,np.sqrt(np.mean(decoded**2,axis=2)+b['epsilon']))
            normalized=np.stack([decoded/radius[None,:,None],decoded/own[:,:,None]])
            for lo in range(0,65537,1024):
                guard();hi=min(65537,lo+1024);A=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
                compare(partial(normalized,A),scores[:,:,:,lo:hi]);compare(partial(H/radius[:,None],A),ref[:,lo:hi])
            ids=np.argmax(scores,axis=3);assert np.array_equal(stored_ids,ids) and np.array_equal(ids[0],ids[1])
            for mi,mode in [(0,'oracle_norm'),(1,'own_norm')]:
                losses=np.empty((K,72),'f8')
                for j in range(K):
                    guard();lp=scores[mi,j]-np.logaddexp.reduce(scores[mi,j],axis=1)[:,None];losses[j]=np.sum(q*(logq-lp),axis=1);candidate_count+=72
                compare(losses,stored_losses[mi]);chosen=np.argmin(losses,axis=0);best=losses[chosen,np.arange(72)];winning=ids[mi,chosen,np.arange(72)];exists=np.any(ids[mi]==true[None,:],axis=0)
                for i,label in enumerate(b['labels']):
                    row=next(x for x in family['rows'] if x['mode']==mode and all(x[k]==v for k,v in label.items()))
                    assert row['best_KL_chart']==int(chosen[i]) and row['best_KL_argmax']==int(winning[i]) and row['source_argmax']==int(true[i]) and row['source_argmax_available']==bool(exists[i])
                    scalar(row['best_KL'],best[i]);compare(np.array(row['all_chart_KL']),losses[:,i]);assert row['all_chart_argmax']==ids[mi,:,i].tolist()
                ss=next(x for x in family['summary'] if x['mode']==mode)
                scalar(ss['mean_KL'],math.fsum(best.tolist())/72);scalar(ss['worst_KL'],best.max());scalar(ss['disagreement_rate'],np.mean(winning!=true))
                assert ss['best_KL_choice_counts']==np.bincount(chosen,minlength=K).tolist()
                assert ss['source_argmax_available_count']==int(exists.sum()) and ss['source_argmax_unavailable_count']==int((~exists).sum())
                assert ss['proxy_gates']=={k:ss[k]<=v for k,v in b['proxy'].items()}
            print(json.dumps(dict(audited_K=K,seconds=time.monotonic()-START)),flush=True)
            del decoded,scores,normalized
        expected='FROZEN_DECODER_ORACLE_PROBABILITY_PASS' if any(all(all(x['proxy_gates'].values()) for x in f['summary']) for f in r['families']) else 'COMPLETED_FROZEN_DECODER_ORACLE_FAILURE'
        assert expected==r['decision'] and candidate_count==2880
        stream.close();stream=None;check();guard()
        result=dict(schema='CONDITIONAL_CHART_READOUT_ORACLE_STORED_AUDIT_V1',source=extent(__file__),freeze=a.freeze,
            report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],
            all20_fixed_decoders_and2880_full_vocab_candidate_probability_rows_checked=True,
            all288_exact_minimum_KL_choices_and_source_argmax_existence_rows_checked=True,
            max_numeric_relative=maxrel,max_scalar_delta=maxscalar,before_after_hashes_exact=True,
            seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,decision='INDEPENDENT_STORED_PASS',scientific_decision=expected,
            scope='Source-width partial encoding/head and independent logaddexp probability/choice audit; '
                'no fit/source model/native/GPU/optimizer/T4 calls. Frozen encoders/decoders only, no capacity bound or causal router proof.')
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
