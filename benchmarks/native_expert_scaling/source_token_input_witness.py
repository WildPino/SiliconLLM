"""Actual layer0 SSM inputs:finite-corpus repeated-token identity witness."""
import argparse,json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site';sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    proc=psutil.Process();proc.cpu_affinity([0]);receipt=dict(schema='SOURCE_TOKEN_INPUT_WITNESS_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    try:
        assert not a.out.exists();cp=DOC/'true_generator_input_capture_result_v3_20261013.json';ap=DOC/'true_generator_input_capture_adjudication_20261013.json';r=json.loads(cp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['decision']=='INDEPENDENT_DATA_AND_SOURCE_CONTINUITY_PASS' and ad['report']==extent(cp);bp=DOC/'true_generator_input_capture_binding_v4_20261013.json';b=json.loads(bp.read_bytes());cfg=Path(b['source'])/'config.json';config=json.loads(cfg.read_bytes());paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',DOC/'SOURCE_TOKEN_INPUT_WITNESS_PROTOCOL_20261013.md',cp,ap,bp,cfg,SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py',Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(psutil.__file__)]+[Path(q['sites'][0]['fields']['H']['path']) for q in r['records']];inputs=[extent(p) for p in dict.fromkeys(paths)];seen={};mismatches=[];positions=0;repeated=0
        def check():
            for e in inputs:assert time.monotonic()-START<=60 and proc.memory_info().peak_wset<=1<<30 and extent(e['path'])==e,e['path']
        check()
        for rec in r['records']:
            e=rec['sites'][0]['fields']['H'];h=np.fromfile(e['path'],'<u2').reshape(e['shape']);assert h.shape==(rec['history'],2048) and e['dtype']=='BF16' and rec['history']==len(rec['input_ids'])
            for ti,token in enumerate(rec['input_ids']):
                assert 0<=token<config['vocab_size'];value=h[ti].tobytes();positions+=1
                if token not in seen:seen[token]=dict(value=value,id=rec['id'],position=ti,count=1)
                else:
                    old=seen[token];old['count']+=1;repeated+=1
                    if old['value']!=value:mismatches.append(dict(token=token,id=rec['id'],position=ti,first_id=old['id'],first_position=old['position'],different_words=int(np.count_nonzero(np.frombuffer(old['value'],'<u2')!=h[ti]))))
        check();assert positions==22547;words={str(k):dict(first_id=v['id'],first_position=v['position'],observations=v['count'],H_BF16_sha256=hashlib.sha256(v['value']).hexdigest()) for k,v in sorted(seen.items())};width=sum(b['fields'][k]['width'] for k in ['gate','x','B','C','delta']);assert width==6704;result=dict(schema='SOURCE_TOKEN_INPUT_WITNESS_V1',freeze=a.freeze,capture=extent(cp),adjudication=extent(ap),inputs=inputs,positions=positions,unique_tokens=len(seen),repeated_observations=repeated,repeated_token_keys=sum(v['count']>1 for v in seen.values()),mismatches=mismatches,word_witnesses=words,decision='EXACT_REPEATED_TOKEN_H0_INVARIANCE' if not mismatches else 'TOKEN_H0_DEPENDENCE_COUNTEREXAMPLE',proposed_generator_table=dict(vocabulary=config['vocab_size'],width=width,BF16_bytes=config['vocab_size']*width*2),before_after_hashes_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope='Finite corpus exact repeated-token H0 identity plus pinned source graph/config. Initial embedding->RMSNorm->SSMin input is structurally token-only in source code. No full-vocab function table/kernel built or speed/ternary/whole-model admission. Conv and state remain history-dependent.');write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(decision=result['decision'],positions=positions,unique_tokens=len(seen),repeated=repeated,mismatches=len(mismatches),table=result['proposed_generator_table'],seconds=result['seconds'])),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
