"""Direct actual-source F64 readout on fixed DEV72 states; no model replay."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def prepare(a):
    import numpy as np,psutil
    report=DOC/'source_residual_coordinate_feasibility_result_20261012.json'
    audit=DOC/'source_residual_coordinate_stored_adjudication_20261012.json'
    r=json.loads(report.read_bytes());ad=json.loads(audit.read_bytes());assert ad['report']==extent(report)
    assert ad['all7_covariances_full_eigensystems_and336_case_metrics_checked']
    b=json.loads(Path(r['binding']['path']).read_bytes());basis=next(x['file'] for x in r['bases'] if x['boundary']==24)
    corpus=ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    records={x['id']:x for x in json.loads(corpus.read_bytes())['records']};samples=[];paths=[]
    for rec in b['records']:
        if rec['split']!='DEV':continue
        cr=records[rec['id']];assert len(cr['positions'])>=3 and len(cr['student_input_ids'])==rec['history']
        indices=[0,len(cr['positions'])//2,len(cr['positions'])-1];assert len(set(indices))==3
        h=next(x['h'] for x in rec['boundaries'] if x['boundary']==24);paths.append(Path(h['path']))
        samples.append(dict(id=rec['id'],history=rec['history'],label_indices=indices,positions=[cr['positions'][i] for i in indices],h=h))
    source=ROOT/'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
    paths += [Path(__file__),B/'falcon_auxiliary_s6_transport.py',report,audit,Path(r['binding']['path']),corpus,
        Path(basis['path']),Path(b['basis']['path']),source/'model.safetensors',source/'config.json',Path(sys.executable),
        DOC/'RESIDUAL_BASIS_READOUT_PROBE_PROTOCOL_20261012.md',Path(np.__file__),Path(np._core._multiarray_umath.__file__),
        Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)]
    paths+=list((ROOT/'results/native_expert_scaling/chatbot_source_runtime/site/numpy.libs').glob('*.dll'))
    assert len(samples)==24 and sum(len(x['positions']) for x in samples)==72
    binding=dict(schema='RESIDUAL_BASIS_READOUT_PROBE_BINDING_V1',python=str(Path(sys.executable).resolve()),source=str(source.resolve()),
        learned_basis=basis,old_basis=b['basis'],samples=samples,
        modes=['full_source','new_true_source_norm','new_restricted_norm','old_true_source_norm','old_restricted_norm'],
        proxy=dict(mean_KL=.01,worst_KL=.1,disagreement_rate=.01),limits=dict(seconds=600,OS_bytes=4<<30,output_bytes=512<<20),
        inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
        scope='Fixed72 first/middle/last DEV labels, actual raw source states and coefficients, F64 readout. '
            'Source norm oracle diagnostic not deployment;no source/shadow/backward/native/GPU/T4 call. '
            'No model-own-history or preserved composition/useful-n/speed/general-capacity claim.')
    write(a.out,binding);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['schema']=='RESIDUAL_BASIS_READOUT_PROBE_BINDING_V1' and not a.out.exists() and not a.directory.exists()
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='before_hash';stream=None
    receipt=dict(schema='RESIDUAL_BASIS_READOUT_PROBE_TERMINAL_V1',binding=extent(a.binding),freeze=a.freeze,command=sys.orig_argv,
        pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
    def check():
        for item in b['inputs']:guard();assert extent(item['path'])==item,item['path']
    try:
        check();a.directory.mkdir()
        with np.load(b['learned_basis']['path'],allow_pickle=False) as z:new=z['basis'].copy()
        old=np.fromfile(b['old_basis']['path'],'<f4').reshape(2048,256).astype('f8')
        H=[];labels=[]
        for item in b['samples']:
            raw=(np.fromfile(item['h']['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(item['history'],2048).astype('f8')
            H.append(raw[item['positions']]);labels.extend(dict(id=item['id'],label_index=i,position=p) for i,p in zip(item['label_indices'],item['positions']))
        H=np.concatenate(H);assert H.shape==(72,2048)
        cfg=json.loads((Path(b['source'])/'config.json').read_bytes());eps=cfg['rms_norm_eps'];scale=cfg['lm_head_multiplier']
        true_norm=np.sqrt(np.mean(H*H,axis=1,keepdims=True)+eps)
        zn=H@new;zo=H@old
        new_norm=np.sqrt(np.sum((zn@(new.T@new))*zn,axis=1,keepdims=True)/2048+eps)
        old_norm=np.sqrt(np.sum((zo@(old.T@old))*zo,axis=1,keepdims=True)/2048+eps)
        getter,stream=tensor_reader(Path(b['source'])/'model.safetensors',np)
        gamma=getter('model.final_layernorm.weight');ref_input=H/true_norm*gamma
        scores={mode:np.empty((72,65537),dtype='f8') for mode in b['modes']}
        new_head=np.empty((65537,256),dtype='f8');old_head=np.empty_like(new_head)
        stage='all_readout_rows'
        for lo in range(0,65537,2048):
            guard();hi=min(lo+2048,65537);W=getter('lm_head.weight',list(range(lo,hi)))
            hn=(W*gamma)@new;ho=(W*gamma)@old;new_head[lo:hi]=hn;old_head[lo:hi]=ho
            scores['full_source'][:,lo:hi]=(ref_input@W.T)*scale
            scores['new_true_source_norm'][:,lo:hi]=((zn/true_norm)@hn.T)*scale
            scores['new_restricted_norm'][:,lo:hi]=((zn/new_norm)@hn.T)*scale
            scores['old_true_source_norm'][:,lo:hi]=((zo/true_norm)@ho.T)*scale
            scores['old_restricted_norm'][:,lo:hi]=((zo/old_norm)@ho.T)*scale
        stream.close();stream=None;guard()
        new_head.astype('<f8').tofile(a.directory/'new_head.f64');old_head.astype('<f8').tofile(a.directory/'old_head.f64')
        for mode,value in scores.items():assert np.isfinite(value).all();value.astype('<f8').tofile(a.directory/(mode+'.f64'))
        H.astype('<f8').tofile(a.directory/'source_selected_h24.f64')
        rows=[];summaries=[]
        for mode in b['modes'][1:]:
            losses=[];dis=0
            for i in range(72):
                q=scores['full_source'][i].copy();q-=q.max();q-=math.log(float(np.exp(q).sum()))
                p=scores[mode][i].copy();p-=p.max();p-=math.log(float(np.exp(p).sum()))
                loss=float(np.sum(np.exp(q)*(q-p)));wrong=int(np.argmax(q)!=np.argmax(p));losses.append(loss);dis+=wrong
                rows.append(dict(**labels[i],mode=mode,KL=loss,argmax_disagreement=wrong))
            summary=dict(mode=mode,mean_KL=math.fsum(losses)/72,worst_KL=max(losses),disagreements=dis,disagreement_rate=dis/72)
            summary['proxy_gates']={key:summary[key]<=value for key,value in b['proxy'].items()};summaries.append(summary)
        stage='after_hash';check();guard();outputs=[extent(p) for p in sorted(a.directory.iterdir())]
        assert sum(item['bytes'] for item in outputs)<=b['limits']['output_bytes']
        r=dict(schema='RESIDUAL_BASIS_READOUT_PROBE_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,
            rows=rows,summary=summaries,outputs=outputs,input_output_shapes=dict(states=[72,2048],scores=[72,65537],heads=[65537,256]),
            inputs_before_after_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope=b['scope'])
        write(a.out,r);receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(summary=summaries,seconds=r['seconds'],OS_peak=r['OS_peak'])),flush=True)
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage);raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.prepare:prepare(a)
    else:assert all((a.binding,a.binding_sha,a.freeze,a.directory));run(a)
