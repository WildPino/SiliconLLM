"""Saved-only F64 metrics and actual whole-recovery model/Adam/RNG audit.

Supports a terminal failed/incomplete attempt without inventing final outputs.
No source/student forward, generation, backward, optimizer update or CUDA use.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))


def bind(a):
    original_binding = DOC / 'chatbot_hybrid_recovery_binding_20261009.json'
    original = json.loads(original_binding.read_bytes())
    receipt = json.loads(a.receipt.read_bytes())
    assert 'exit_code' in receipt and receipt['binding_sha256']==sha(original_binding)
    result_path = DOC / 'chatbot_hybrid_recovery_result_20261009.json'
    full = result_path.exists() and receipt['exit_code']==0
    if full:
        assert receipt['result_sha256']==sha(result_path)
    state = a.attempt / ('learner.pt' if (a.attempt/'learner.pt').exists() else 'recovery.pt')
    original_launcher = DOC / 'chatbot_hybrid_recovery_launcher_frozen_20261009.py.txt'
    files = [Path(__file__),B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
             Path(sys.executable),DOC/'CHATBOT_HYBRID_RECOVERY_AUDIT_PROTOCOL_20261009.md',
             original_binding,a.receipt,Path(original['corpus']),Path(original['checkpoint']),state,original_launcher]
    files += sorted(p for p in a.attempt.iterdir() if p.is_file() and p.suffix=='.json')
    files += sorted(a.attempt.glob('*.logits.f32'))
    if full:
        files.append(result_path)
    corpus = json.loads(Path(original['corpus']).read_bytes())
    files += [Path(r['logits']['path']) for r in corpus['records']]
    files += [SITE/'torch'/v for v in ('serialization.py','optim/adamw.py','optim/optimizer.py')]
    files = list(dict.fromkeys(p.resolve() for p in files))
    # Bind actual original inputs too, even after a failed launcher did not run
    # its success-only postcheck. This consumes no model observations.
    for item in original['inputs']:
        path=Path(item['path'])
        if path.resolve()==(B/'chatbot_falcon_usability_launch.py').resolve():
            path=original_launcher
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
    write(a.out,dict(schema='HYBRID_RECOVERY_AUDIT_BINDING_V1',python=str(Path(sys.executable).resolve()),
         worker_path=str(Path(__file__).resolve()),attempt=str(a.attempt.resolve()),
         original_binding=str(original_binding.resolve()),receipt=str(a.receipt.resolve()),
         original_result=str(result_path.resolve()) if full else None,state=str(state.resolve()),
         full_primary=full,limits=dict(seconds=300,OS_bytes=12<<30,output_bytes=64<<20),
         runtime_binding_scope='Actual terminal partial/full packets and model/Adam/RNG; isolated NumPy/Torch/Python versions and selected file hashes. F64 arithmetic independent of training reductions, not independent libraries or a CUDA replay.',
         inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files),full_primary=full)),flush=True)


def worker(a):
    start=time.monotonic()
    assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes())
    assert b['schema']=='HYBRID_RECOVERY_AUDIT_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve() and sys.version_info[:3]==(3,12,10)
    a.directory.mkdir(exist_ok=False)
    try:
        import numpy as np
        import psutil
        import torch
        assert np.__version__=='2.4.6' and psutil.__version__=='7.2.2' and torch.__version__=='2.6.0+cu124'
        assert not torch.cuda.is_initialized()
        proc=psutil.Process()
        proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        original=json.loads(Path(b['original_binding']).read_bytes())
        corpus=json.loads(Path(original['corpus']).read_bytes())
        attempt=Path(b['attempt'])
        primary=json.loads(Path(b['original_result']).read_bytes()) if b['full_primary'] else None
        def guard():
            assert time.monotonic()-start<=b['limits']['seconds'],'deadline'
            assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'OS cap'
            assert not proc.children(recursive=True),'worker subprocess'
            assert not torch.cuda.is_initialized(),'unexpected CUDA initialization'
        def logp(x):
            maximum=x.max(-1,keepdims=True)
            shifted=x-maximum
            return shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))
        def aggregate(rows):
            def values(chosen):
                n=sum(r['labels'] for r in chosen)
                return dict(cases=len(chosen),labels=n,case_KL=sum(r['KL'] for r in chosen)/len(chosen),
                      label_KL=sum(r['KL']*r['labels'] for r in chosen)/n,
                      disagreements=sum(r['disagreements'] for r in chosen),
                      label_disagreement=sum(r['disagreements'] for r in chosen)/n)
            summary={s:values([r for r in rows if r['split']==s]) for s in ('FIT','DEV') if any(r['split']==s for r in rows)}
            domains=sorted({r['domain'] for r in rows})
            summary['domains']={d:{s:values([r for r in rows if r['split']==s and r['domain']==d]) for s in ('FIT','DEV') if any(r['split']==s and r['domain']==d for r in rows)} for d in domains}
            return summary
        observations={}
        maximum_label_KL_delta=0.0
        for label in ('before','after'):
            rows=[]
            for rec in corpus['records']:
                row_path=attempt/(label+'.'+rec['id']+'.json')
                raw_path=attempt/(label+'.'+rec['id']+'.logits.f32')
                if not row_path.exists():
                    assert not raw_path.exists(),'uncommitted observation packet'
                    continue
                stored=json.loads(row_path.read_bytes())
                assert stored['id']==rec['id'] and stored['domain']==rec['domain'] and stored['split']==rec['split']
                assert stored['labels']==len(rec['output_ids']) and stored['input_tokens']==len(rec['student_input_ids'])
                assert not stored['training_dither']
                packet=stored['logits']
                assert Path(packet['path']).resolve()==raw_path.resolve()
                assert raw_path.stat().st_size==packet['bytes']==len(rec['output_ids'])*65537*4
                assert sha(raw_path)==packet['sha256']
                bits=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4')
                bits<<=16
                source=bits.view('<f4').reshape(rec['logits']['shape']).astype(np.float64)
                target=np.fromfile(raw_path,dtype='<f4').reshape(rec['logits']['shape']).astype(np.float64)
                assert np.isfinite(source).all() and np.isfinite(target).all()
                lp=logp(source)
                lq=logp(target)
                kl=(np.exp(lp)*(lp-lq)).sum(-1)
                assert np.isfinite(kl).all() and kl.min()>=-1e-10
                delta=float(np.max(np.abs(kl-np.array(stored['label_KL']))))
                maximum_label_KL_delta=max(maximum_label_KL_delta,delta)
                assert delta<=1e-4,'F64 versus recorded label KL'
                assert abs(float(kl.mean())-stored['KL'])<=1e-4
                ids=target.argmax(-1).tolist()
                assert ids==stored['predicted_ids']
                disagreement=sum(x!=y for x,y in zip(ids,rec['output_ids'],strict=True))
                assert disagreement==stored['disagreements']
                rows.append(dict(id=rec['id'],domain=rec['domain'],split=rec['split'],labels=len(ids),
                                 KL=float(kl.mean()),disagreements=disagreement))
                guard()
            observations[label]=dict(cases=rows,summary=aggregate(rows) if rows else None)
        updates=[json.loads(p.read_bytes()) for p in sorted(attempt.glob('update_*.json'))]
        flat=[identifier for order in original['orders'] for identifier in order]
        assert len(updates)<=512
        for index,row in enumerate(updates):
            assert row['step']==index+1 and row['epoch']==index//128+1 and row['id']==flat[index]
            assert row['training_dither'] and row['split']=='FIT'
            assert np.isfinite(row['KL']) and np.isfinite(row['gradient_norm_before_clip'])
            for group in row['layer_gradient_norms']:
                assert all(0<group[k]<float('inf') for k in ('core','experts','norms'))
        state=torch.load(b['state'],map_location='cpu',weights_only=True)
        old=torch.load(original['checkpoint'],map_location='cpu',weights_only=True)
        boundary=state['completed_recovery_updates']
        assert boundary==len(updates),'checkpoint/durable update boundary disagreement'
        assert state['target_schema']==old['target_schema']=='COMPACT_FALCON_TERNARY_TARGET_V1'
        assert state['config']==old['config'] and state['model'].keys()==old['model'].keys()
        elements=0
        changed=[]
        for name,value in state['model'].items():
            prior=old['model'][name]
            assert value.device.type=='cpu' and value.dtype==prior.dtype==torch.float32 and value.shape==prior.shape
            array=value.numpy()
            assert np.isfinite(array).all()
            if name.endswith(('_scale',)):
                assert array.min()>=1e-8
            elements+=value.numel()
            if not np.array_equal(array,prior.numpy()):
                changed.append(name)
            guard()
        assert elements==254932736 and len(state['model'])==211
        actual_bytes=0
        prior_states=old['optimizer']['state']
        assert state['optimizer']['state'].keys()==prior_states.keys()
        groups=state['optimizer']['param_groups']
        assert len(groups)==1 and groups[0]['lr']==5e-5 and groups[0]['weight_decay']==0
        assert tuple(groups[0]['betas'])==(.9,.999) and groups[0]['eps']==1e-8 and not groups[0]['foreach']
        for key,slot in state['optimizer']['state'].items():
            assert int(slot['step'].item())==8+boundary
            for name in ('exp_avg','exp_avg_sq'):
                tensor=slot[name]
                assert tensor.device.type=='cpu' and tensor.dtype==torch.float32 and tensor.shape==prior_states[key][name].shape
                array=tensor.numpy()
                assert np.isfinite(array).all()
                if name=='exp_avg_sq':
                    assert array.min()>=0
                actual_bytes+=tensor.numel()*tensor.element_size()
            guard()
        assert actual_bytes==2039461888
        assert state['torch_CPU_rng'].dtype==torch.uint8 and state['torch_CPU_rng'].ndim==1
        assert len(state['torch_CUDA_rng'])==1 and state['torch_CUDA_rng'][0].dtype==torch.uint8
        if boundary:
            assert state['orders']==original['orders'] and state['training_contract']['normalized_AQ_dither']==.025
        recovery_gates=None
        if b['full_primary']:
            assert boundary==512 and len(observations['before']['cases'])==len(observations['after']['cases'])==160
            before=observations['before']['summary']; after=observations['after']['summary']; c=original['criteria']
            recovery_gates=dict(complete_512_updates=True,
               FIT_case_KL_recovery=after['FIT']['case_KL']<=c['FIT_KL_ratio_max']*before['FIT']['case_KL'],
               DEV_case_KL_recovery=after['DEV']['case_KL']<=c['DEV_KL_ratio_max']*before['DEV']['case_KL'],
               DEV_absolute_case_KL=after['DEV']['case_KL']<=c['DEV_case_KL_max'],
               DEV_absolute_label_disagreement=after['DEV']['label_disagreement']<=c['DEV_label_disagreement_max'],
               DEV_every_domain_case_KL=all(v['DEV']['case_KL']<=c['domain_DEV_case_KL_max'] for v in after['domains'].values()),
               DEV_every_domain_label_disagreement=all(v['DEV']['label_disagreement']<=c['domain_DEV_label_disagreement_max'] for v in after['domains'].values()))
            counts=json.loads((attempt/'support.json').read_bytes())
            routing=True
            for label,domains in counts.items():
                for domain,values in domains.items():
                    matrix=np.array(values,dtype=np.int64)
                    assert matrix.shape==(12,72) and matrix.min()>=0
                    selected=[r for r in corpus['records'] if r['domain']==domain and (label!='training' or r['split']=='FIT')]
                    expected=sum(len(r['student_input_ids']) for r in selected)*8*(4 if label=='training' else 1)
                    assert (matrix.sum(-1)==expected).all()
                total=sum(np.array(v,dtype=np.int64) for v in domains.values())
                routing &= all(np.count_nonzero(row)>=8 for row in total)
            recovery_gates['routing_support']=bool(routing)
            assert recovery_gates==primary['gates'],'F64 recovery decision differs'
        guard()
        result=dict(schema='HYBRID_RECOVERY_AUDIT_RESULT_V1',decision='SAVED_RECOVERY_AUDIT_PASS',
             freeze=a.freeze,binding_sha256=a.binding_sha,process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
             primary_result_sha256=sha(b['original_result']) if b['full_primary'] else None,
             checkpoint=dict(path=b['state'],bytes=Path(b['state']).stat().st_size,sha256=sha(b['state'])),
             corpus=dict(path=original['corpus'],sha256=sha(original['corpus'])),
             full_primary=b['full_primary'],checkpoint_boundary=boundary,logged_updates=len(updates),
             observations=observations,maximum_label_KL_F64_difference=maximum_label_KL_delta,
             model_elements=elements,changed_tensors=len(changed),changed_tensor_names=changed,
             optimizer_moment_bytes=actual_bytes,optimizer_step=8+boundary,RNG_storage_present=True,
             reproduced_recovery_gates=recovery_gates,elapsed_seconds=time.monotonic()-start,
             worker_OS_peak_snapshot=proc.memory_info().peak_wset,source_calls=0,student_calls=0,
             CUDA_initialized=False,quality_admission=False,
             scope='Saved F64 labels/IDs/decision and actual finite model/Adam/RNG transport; logged gradients are not independently recomputed. Partial primary stays incomplete.')
        write(a.out,result)
        print(json.dumps(dict(decision=result['decision'],boundary=boundary,full_primary=b['full_primary'],seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(fault=repr(error),elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bind',action='store_true')
    p.add_argument('--attempt',type=Path)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--binding',type=Path)
    p.add_argument('--binding-sha')
    p.add_argument('--freeze')
    p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    bind(a) if a.bind else worker(a)
