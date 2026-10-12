"""All fixed charts on DEV72: permissive decoder-choice diagnostics, no fit."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def prepare(a):
    import numpy as np,psutil
    report=DOC/'conditional_raw_residual_charts_result_20261012.json'
    audit=DOC/'conditional_raw_residual_charts_stored_adjudication_20261012.json'
    r=json.loads(report.read_bytes());ad=json.loads(audit.read_bytes())
    assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(report)
    parent=json.loads(Path(r['binding']['path']).read_bytes());families=[x for x in r['families'] if x['boundary']==24]
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',report,audit,Path(r['binding']['path']),
        DOC/'CONDITIONAL_CHART_READOUT_ORACLE_PROTOCOL_20261012.md',Path(parent['source'])/'model.safetensors',
        Path(parent['source'])/'config.json',Path(r['readouts'][0]['file']['path']),Path(r['readouts'][0]['reference']['path']),
        Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(psutil.__file__)]
    paths+=list((ROOT/'results/native_expert_scaling/chatbot_source_runtime/site/numpy.libs').glob('*.dll'))
    paths+=list((ROOT/'results/native_expert_scaling/chatbot_source_runtime/site/psutil').glob('*.pyd'))
    paths += [Path(c['file']['path']) for f in families for c in f['charts']]
    b=dict(schema='CONDITIONAL_CHART_READOUT_ORACLE_BINDING_V1',python=str(Path(sys.executable).resolve()),
        source=parent['source'],families=families,states=r['readouts'][0]['file'],reference=r['readouts'][0]['reference'],
        labels=r['readouts'][0]['labels'],nearest_center_control=r['readouts'],epsilon=parent['epsilon'],
        proxy=parent['readout_proxy'],limits=dict(seconds=600,OS_bytes=4<<30,output_bytes=2<<30),
        inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
        scope='All FIXED fitted final charts, FIXED source encoders and source72 states. Best exact KL/argmax-existence '
            'choices access full teacher probabilities and every chart, hence not a deployable router. No fit or new directions. '
            'Bounds only these frozen decoded predictions, not arbitrary chart coordinates or conditional256 capacity. '
            'No source/shadow/optimizer/native/GPU/T4 calls or DEV calibration.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['schema']=='CONDITIONAL_CHART_READOUT_ORACLE_BINDING_V1' and Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert not a.out.exists() and not a.directory.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None
    receipt=dict(schema='CONDITIONAL_CHART_READOUT_ORACLE_TERMINAL_V1',command=sys.orig_argv,freeze=a.freeze,
        binding=extent(a.binding),pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
    def check():
        for item in b['inputs']:guard();assert extent(item['path'])==item,item['path']
    try:
        check();a.directory.mkdir();get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np)
        cfg=json.loads((Path(b['source'])/'config.json').read_bytes());gamma=get('model.final_layernorm.weight')
        with np.load(b['states']['path'],allow_pickle=False) as pack:H=pack['raw']
        assert H.shape==(72,2048);radius=np.sqrt(np.mean(H*H,axis=1)+b['epsilon'])
        ref=np.fromfile(b['reference']['path'],'<f8').reshape(72,65537);logq=ref-ref.max(axis=1,keepdims=True);logq-=np.log(np.exp(logq).sum(axis=1,keepdims=True));q=np.exp(logq)
        true_id=np.argmax(ref,axis=1);families=[]
        for family in b['families']:
            K=family['K'];decoded=[]
            for chart in family['charts']:
                guard()
                with np.load(chart['file']['path'],allow_pickle=False) as pack:mu=pack['center'];V=pack['V'];W=pack['W']
                decoded.append(mu+((H-mu)@W)@V.T)
            decoded=np.stack(decoded);own=np.sqrt(np.mean(decoded*decoded,axis=2)+b['epsilon']);scores=np.empty((2,K,72,65537),'f8')
            for lo in range(0,65537,1024):
                guard();hi=min(65537,lo+1024);A=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
                scores[0,:,:,lo:hi]=(decoded/radius[None,:,None])@A.T
                scores[1,:,:,lo:hi]=(decoded/own[:,:,None])@A.T
            assert np.isfinite(scores).all();losses=np.empty((2,K,72),'f8');ids=np.argmax(scores,axis=3)
            assert np.array_equal(ids[0],ids[1]);rows=[];summaries=[]
            for mi,mode in [(0,'oracle_norm'),(1,'own_norm')]:
                for j in range(K):
                    guard();lp=scores[mi,j]-scores[mi,j].max(axis=1,keepdims=True);lp-=np.log(np.exp(lp).sum(axis=1,keepdims=True))
                    losses[mi,j]=np.sum(q*(logq-lp),axis=1)
                choices=np.argmin(losses[mi],axis=0);best=losses[mi,choices,np.arange(72)];chosenids=ids[mi,choices,np.arange(72)]
                available=np.any(ids[mi]==true_id[None,:],axis=0)
                summary=dict(mode=mode,mean_KL=math.fsum(best.tolist())/72,worst_KL=float(best.max()),
                    disagreement_rate=float(np.mean(chosenids!=true_id)),best_KL_choice_counts=np.bincount(choices,minlength=K).tolist(),
                    source_argmax_available_count=int(available.sum()),source_argmax_unavailable_count=int((~available).sum()))
                summary['proxy_gates']={k:summary[k]<=v for k,v in b['proxy'].items()};summaries.append(summary)
                for i,label in enumerate(b['labels']):
                    rows.append(dict(**label,mode=mode,best_KL_chart=int(choices[i]),best_KL=float(best[i]),
                        best_KL_argmax=int(chosenids[i]),source_argmax=int(true_id[i]),source_argmax_available=bool(available[i]),
                        all_chart_KL=losses[mi,:,i].tolist(),all_chart_argmax=ids[mi,:,i].tolist()))
            file=a.directory/f'K{K:02d}.all_readouts.npz';np.savez(file,raw=H,decoded=decoded,true_radius=radius,own_radius=own,scores=scores,KL=losses,argmax=ids)
            families.append(dict(K=K,file=extent(file),rows=rows,summary=summaries));del scores,losses,decoded
            print(json.dumps(dict(K=K,summary=summaries,seconds=time.monotonic()-START)),flush=True)
        stream.close();stream=None;check();guard();outputs=[extent(p) for p in sorted(a.directory.iterdir())]
        assert sum(x['bytes'] for x in outputs)<=b['limits']['output_bytes']
        result=dict(schema='CONDITIONAL_CHART_READOUT_ORACLE_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,
            families=families,outputs=outputs,inputs_before_after_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            decision='FROZEN_DECODER_ORACLE_PROBABILITY_PASS' if any(all(all(x['proxy_gates'].values()) for x in f['summary']) for f in families) else 'COMPLETED_FROZEN_DECODER_ORACLE_FAILURE',scope=b['scope'])
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.prepare:prepare(a)
    else:assert all((a.binding,a.binding_sha,a.freeze,a.directory));run(a)
