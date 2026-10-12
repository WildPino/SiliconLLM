"""Independent dense reconstruction and full-V probability probe audit."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes())
    assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.report)
    assert r['schema']=='RESIDUAL_BASIS_READOUT_PROBE_RESULT_V1'
    assert extent(r['binding']['path'])==r['binding'];b=json.loads(Path(r['binding']['path']).read_bytes())
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None
    inputs=[extent(__file__),extent(a.report),extent(a.terminal),r['binding'],*b['inputs'],*r['outputs']]
    def guard():
        assert time.monotonic()-START<=600 and proc.memory_info().peak_wset<=4<<30
        assert not proc.children(recursive=True)
    def check():
        for item in inputs:guard();assert extent(item['path'])==item,item['path']
    check();directory=Path(r['outputs'][0]['path']).parent;H=[];labels=[]
    for rec in b['samples']:
        raw=(np.fromfile(rec['h']['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(rec['history'],2048).astype('f8')
        H.append(raw[rec['positions']]);labels.extend((rec['id'],i,p) for i,p in zip(rec['label_indices'],rec['positions']))
    H=np.concatenate(H);assert np.array_equal(H,np.fromfile(directory/'source_selected_h24.f64','<f8').reshape(72,2048))
    with np.load(b['learned_basis']['path'],allow_pickle=False) as z:R=z['basis'].copy()
    old=np.fromfile(b['old_basis']['path'],'<f4').reshape(2048,256).astype('f8')
    hn=(H@R)@R.T;ho=(H@old)@old.T
    cfg=json.loads((Path(b['source'])/'config.json').read_bytes());eps=cfg['rms_norm_eps'];scale=cfg['lm_head_multiplier']
    norm=np.sqrt(np.sum(H*H,axis=1,keepdims=True)/2048+eps)
    normn=np.sqrt(np.sum(hn*hn,axis=1,keepdims=True)/2048+eps)
    normo=np.sqrt(np.sum(ho*ho,axis=1,keepdims=True)/2048+eps)
    getter,stream=tensor_reader(Path(b['source'])/'model.safetensors',np)
    gamma=getter('model.final_layernorm.weight')
    features=dict(full_source=H/norm*gamma,new_true_source_norm=hn/norm*gamma,
        new_restricted_norm=hn/normn*gamma,old_true_source_norm=ho/norm*gamma,old_restricted_norm=ho/normo*gamma)
    arrays={mode:np.memmap(directory/(mode+'.f64'),'<f8','r',shape=(72,65537)) for mode in b['modes']}
    heads={key:np.memmap(directory/(key+'_head.f64'),'<f8','r',shape=(65537,256)) for key in ('new','old')}
    max_score_relative=0;max_head_relative=0
    try:
        for lo in range(0,65537,2048):
            guard();hi=min(lo+2048,65537);W=getter('lm_head.weight',list(range(lo,hi)))
            for name,basis in [('new',R),('old',old)]:
                expected=W@(gamma[:,None]*basis);actual=heads[name][lo:hi]
                delta=float(np.max(np.abs(expected-actual))/max(float(np.max(np.abs(expected))),1e-300))
                max_head_relative=max(max_head_relative,delta);assert delta<=1e-10
            for mode,value in features.items():
                # Distinct feature-axis partial sums, no producer's projected-head prediction.
                score=np.zeros((72,hi-lo),dtype='f8')
                for k in range(0,2048,256):score+=(value[:,k:k+256]@W[:,k:k+256].T)*scale
                actual=arrays[mode][:,lo:hi];assert np.isfinite(actual).all()
                delta=float(np.max(np.abs(score-actual))/max(float(np.max(np.abs(score))),1e-300))
                max_score_relative=max(max_score_relative,delta);assert delta<=1e-10
        stream.close();stream=None;max_metric=0;summaries=[]
        lookup={(row['mode'],row['id'],row['label_index']):row for row in r['rows']}
        for mode in b['modes'][1:]:
            losses=[];dis=0
            for i,(case,index,position) in enumerate(labels):
                q=np.array(arrays['full_source'][i]);q-=q.max();q-=math.log(math.fsum(map(float,np.exp(q))))
                p=np.array(arrays[mode][i]);p-=p.max();p-=math.log(math.fsum(map(float,np.exp(p))))
                value=math.fsum(map(float,np.exp(q)*(q-p)));wrong=int(np.argmax(q)!=np.argmax(p))
                row=lookup[(mode,case,index)];assert row['position']==position and row['argmax_disagreement']==wrong
                max_metric=max(max_metric,abs(value-row['KL']));losses.append(value);dis+=wrong
            summary=dict(mode=mode,mean_KL=math.fsum(losses)/72,worst_KL=max(losses),disagreements=dis,disagreement_rate=dis/72)
            previous=next(x for x in r['summary'] if x['mode']==mode)
            for key in ('mean_KL','worst_KL'):max_metric=max(max_metric,abs(summary[key]-previous[key]))
            assert dis==previous['disagreements']
            summary['proxy_gates']={key:summary[key]<=bound for key,bound in b['proxy'].items()}
            assert summary['proxy_gates']==previous['proxy_gates'];summaries.append(summary)
        assert len(labels)==72 and len(lookup)==288 and max_metric<=1e-8
        check();guard()
        result=dict(schema='RESIDUAL_BASIS_READOUT_STORED_AUDIT_V1',source=extent(__file__),report=extent(a.report),
            terminal=extent(a.terminal),all72_states5_full_vocab_arrays2_projected_heads_checked=True,
            all288_probability_rows_and_modes_checked=True,max_score_relative=max_score_relative,
            max_projected_head_relative=max_head_relative,max_metric_delta=max_metric,summary=summaries,
            all_inputs_outputs_before_after_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            scope='Independent dense reconstruction/reduction order and fsum categorical audit; direct affine readout recomputed, '
                'no neural/source generation/eigensolver/optimizer/native/GPU replay; no oracle norm or whole pipeline admission.')
        write(a.out,result)
    finally:
        if stream is not None:stream.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--report-sha',required=True)
    p.add_argument('--terminal',type=Path,required=True);p.add_argument('--terminal-sha',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:main(a)
    except BaseException as error:
        import traceback
        write(a.out.with_suffix('.fault.json'),dict(error=repr(error),traceback=traceback.format_exc(),command=sys.orig_argv,
            source=extent(__file__),seconds=time.monotonic()-START));raise
