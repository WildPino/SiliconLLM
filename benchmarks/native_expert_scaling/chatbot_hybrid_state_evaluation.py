"""Complete inference evaluation of an audited interrupted recovery boundary.

No source execution, optimizer restoration/update, generation or inference dither.
All original-prefix logits are retained once at this new fixed checkpoint.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def bind(a):
    audit_path = DOC/'chatbot_hybrid_recovery_audit_result_20261009.json'
    audit = json.loads(audit_path.read_bytes())
    terminal_path = audit_path.with_suffix('.terminal.json')
    terminal = json.loads(terminal_path.read_bytes())
    assert audit['decision']=='SAVED_RECOVERY_AUDIT_PASS' and not audit['full_primary']
    assert terminal['exit_code']==0 and terminal['resource_gates'] and terminal['result_sha256']==sha(audit_path)
    assert audit['checkpoint_boundary']==286 and audit['optimizer_step']==294
    assert len(audit['observations']['before']['cases'])==160 and not audit['observations']['after']['cases']
    audit_binding_path=DOC/'chatbot_hybrid_recovery_audit_binding_20261009.json'
    assert sha(audit_binding_path)==audit['binding_sha256']==terminal['binding_sha256']
    audit_binding=json.loads(audit_binding_path.read_bytes())
    state, corpus_path = Path(audit['checkpoint']['path']), Path(audit['corpus']['path'])
    assert state.stat().st_size==audit['checkpoint']['bytes'] and sha(state)==audit['checkpoint']['sha256']
    assert sha(corpus_path)==audit['corpus']['sha256']
    original_path = DOC/'chatbot_hybrid_recovery_binding_20261009.json'
    original_identity=next(v for v in audit_binding['inputs'] if Path(v['path']).resolve()==original_path.resolve())
    assert sha(original_path)==original_identity['sha256']
    original = json.loads(original_path.read_bytes())
    target_identity=next(v for v in original['inputs'] if Path(v['path']).resolve()==(B/'chatbot_hybrid_target.py').resolve())
    assert sha(B/'chatbot_hybrid_target.py')==target_identity['sha256']
    corpus = json.loads(corpus_path.read_bytes())
    records = corpus['records']
    assert len(records)==160 and sum(len(r['output_ids']) for r in records)==5746
    assert max(len(r['student_input_ids']) for r in records)<=512
    assert all(r['split'] in ('FIT','DEV') for r in records)
    files = [Path(__file__),B/'chatbot_hybrid_target.py',B/'chatbot_falcon_usability.py',
             B/'chatbot_falcon_usability_launch.py',Path(sys.executable),audit_path,terminal_path,
             original_path,audit_binding_path,state,corpus_path,DOC/'CHATBOT_HYBRID_STATE_EVALUATION_PROTOCOL_20261009.md']
    assert all(Path(r['logits']['path']).stat().st_size==r['logits']['bytes'] and sha(r['logits']['path'])==r['logits']['sha256'] for r in records)
    files += [Path(r['logits']['path']) for r in records]
    files += [SITE/'transformers'/v for v in ('models/falcon_h1/modeling_falcon_h1.py',
              'models/falcon_h1/configuration_falcon_h1.py','cache_utils.py','utils/import_utils.py')]
    files += [SITE/'torch'/v for v in ('serialization.py','utils/checkpoint.py')]
    foreign = {'benchmarks/donor_adaptation/configs/_manifest.json':'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
      'benchmarks/donor_adaptation/density/build_document_holdout.py':'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
      'docs/research/RESEARCH_INDEX.md':'99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}
    assert all(sha(ROOT/k)==v for k,v in foreign.items())
    files += [ROOT/k for k in foreign]
    files = list(dict.fromkeys(p.resolve() for p in files))
    write(a.out,dict(schema='HYBRID_STATE_EVALUATION_BINDING_V1',python=str(Path(sys.executable).resolve()),
         worker_path=str(Path(__file__).resolve()),checkpoint=dict(audit['checkpoint']),
         boundary=286,audit=str(audit_path.resolve()),corpus=str(corpus_path.resolve()),
         criteria=original['criteria'],limits=dict(seconds=900,OS_bytes=10<<30,
                 GPU_allocated_bytes=6<<30,GPU_reserved_bytes=7<<30,output_bytes=2<<30),
         runtime_binding_scope='Audited actual boundary286 model and complete original-prefix corpus; original target/no-grad arithmetic, fixed package versions and selected runtime hashes. No whole DLL-tree or CUDA bit determinism certificate.',
         inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding)==a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema']=='HYBRID_STATE_EVALUATION_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve() and sys.version_info[:3]==(3,12,10)
    assert all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
    a.directory.mkdir(exist_ok=False)
    completed=[]; phase='imports'
    try:
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels','mamba_ssm','causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        import tokenizers
        import chatbot_hybrid_target as native
        assert (torch.__version__,transformers.__version__,np.__version__,tokenizers.__version__,psutil.__version__)==('2.6.0+cu124','5.13.1','2.4.6','0.22.2','7.2.2')
        proc=psutil.Process(); proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6); torch.set_num_interop_threads(1); torch.manual_seed(20261009)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        def guard():
            limit=b['limits']
            assert time.monotonic()-start<=limit['seconds']-60,'worker deadline reserve'
            assert proc.memory_info().peak_wset<=limit['OS_bytes'],'OS cap'
            assert torch.cuda.max_memory_allocated()<=limit['GPU_allocated_bytes'],'GPU allocated cap'
            assert torch.cuda.max_memory_reserved()<=limit['GPU_reserved_bytes'],'GPU reserved cap'
            assert not proc.children(recursive=True),'worker subprocess'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=limit['output_bytes'],'output cap'
        def event(stage,**fields):
            print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**fields)),flush=True)
            guard()
        phase='load'
        state=torch.load(b['checkpoint']['path'],map_location='cpu',weights_only=True)
        assert state['target_schema']=='COMPACT_FALCON_TERNARY_TARGET_V1' and state['completed_recovery_updates']==b['boundary']
        target=native.Target(state['config']).to('cuda')
        target.load_state_dict(state['model'],strict=True)
        assert len(list(target.parameters()))==211 and sum(p.numel() for p in target.parameters())==254932736
        target.requires_grad_(False); target.eval()
        del state; gc.collect()
        corpus=json.loads(Path(b['corpus']).read_bytes()); records=corpus['records']
        support={d:np.zeros((12,72),dtype=np.int64) for d in sorted({r['domain'] for r in records})}
        current_domain=None
        def hook(site):
            def collect(module,args):
                scores=module.router(args[0].reshape(-1,512))
                ids=torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:8]
                support[current_domain][site]+=torch.bincount(ids.flatten(),minlength=72).cpu().numpy()
            return collect
        handles=[layer.banks.register_forward_pre_hook(hook(i)) for i,layer in enumerate(target.layers)]
        def lp64(x):
            shifted=x-x.max(-1,keepdims=True)
            return shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))
        def aggregate(chosen):
            n=sum(r['labels'] for r in chosen); errors=sum(r['disagreements'] for r in chosen)
            return dict(cases=len(chosen),labels=n,case_KL=sum(r['KL_F64'] for r in chosen)/len(chosen),
                 label_KL=sum(sum(r['label_KL_F64']) for r in chosen)/n,
                 disagreements=errors,label_disagreement=errors/n,
                 case_disagreement=sum(r['disagreements']/r['labels'] for r in chosen)/len(chosen))
        rows=[]; maximum_delta=0.0
        event('target_ready',boundary=b['boundary'],parameters=254932736,optimizer_restored=False,AQ_dither=False)
        phase='evaluation'
        with torch.no_grad():
            for rec in records:
                guard(); t0=time.monotonic(); current_domain=rec['domain']
                assert rec['student_input_ids']==rec['input_ids']+rec['output_ids'][:-1]
                assert rec['positions']==list(range(len(rec['input_ids'])-1,len(rec['student_input_ids'])))
                bits=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4'); bits<<=16
                source=bits.view('<f4').reshape(rec['logits']['shape'])
                ids=torch.tensor([rec['student_input_ids']],device='cuda')
                positions=torch.tensor(rec['positions'],device='cuda')
                logits=target(ids,positions).squeeze(0)
                assert list(logits.shape)==rec['logits']['shape'] and torch.isfinite(logits).all().item()
                assert all(p.grad is None for p in target.parameters()) and not target.training
                teacher=torch.from_numpy(source).to('cuda'); logp=torch.log_softmax(teacher,-1)
                f32=(logp.exp()*(logp-torch.log_softmax(logits,-1))).sum(-1).cpu().numpy()
                values=logits.cpu().numpy(); p64=lp64(source.astype(np.float64)); q64=lp64(values.astype(np.float64))
                f64=(np.exp(p64)*(p64-q64)).sum(-1)
                assert np.isfinite(f64).all() and f64.min()>=-1e-10
                delta=float(np.max(np.abs(f64-f32))); maximum_delta=max(maximum_delta,delta)
                assert delta<=1e-4,'F32/F64 KL transport'
                winners=values.argmax(-1).tolist()
                assert winners==logits.argmax(-1).cpu().tolist()
                path=a.directory/('evaluation.'+rec['id']+'.logits.f32')
                with path.open('xb') as f:
                    f.write(values.astype('<f4',copy=False).tobytes()); f.flush(); os.fsync(f.fileno())
                torch.cuda.synchronize()
                row=dict(id=rec['id'],split=rec['split'],domain=rec['domain'],labels=len(rec['output_ids']),
                   input_tokens=len(rec['student_input_ids']),KL_F64=float(f64.mean()),label_KL_F64=f64.tolist(),
                   label_KL_F32=f32.tolist(),predicted_ids=winners,
                   disagreements=sum(x!=y for x,y in zip(winners,rec['output_ids'],strict=True)),
                   logits=dict(path=str(path.resolve()),shape=list(values.shape),bytes=path.stat().st_size,sha256=sha(path)),
                   seconds=time.monotonic()-t0,AQ_dither=False)
                write(a.directory/('evaluation.'+rec['id']+'.json'),row); rows.append(row); completed.append(rec['id'])
                event('case',case=rec['id'],split=rec['split'],KL_F64=row['KL_F64'],disagreements=row['disagreements'],labels=row['labels'])
                del source,teacher,logp,logits,values,p64,q64,ids,positions
        for h in handles: h.remove()
        assert len(rows)==160 and sum(r['labels'] for r in rows)==5746
        summary={s:aggregate([r for r in rows if r['split']==s]) for s in ('FIT','DEV')}
        summary['domains']={d:{s:aggregate([r for r in rows if r['domain']==d and r['split']==s]) for s in ('FIT','DEV')} for d in support}
        audit=json.loads(Path(b['audit']).read_bytes()); before=audit['observations']['before']['summary']; c=b['criteria']
        for d,matrix in support.items():
            expected=sum(len(r['student_input_ids']) for r in records if r['domain']==d)*8
            assert (matrix.sum(-1)==expected).all(),'support count'
        total=sum(support.values())
        gates=dict(complete160_cases=True,
             FIT_case_KL_recovery=summary['FIT']['case_KL']<=c['FIT_KL_ratio_max']*before['FIT']['case_KL'],
             DEV_case_KL_recovery=summary['DEV']['case_KL']<=c['DEV_KL_ratio_max']*before['DEV']['case_KL'],
             DEV_absolute_case_KL=summary['DEV']['case_KL']<=c['DEV_case_KL_max'],
             DEV_absolute_label_disagreement=summary['DEV']['label_disagreement']<=c['DEV_label_disagreement_max'],
             DEV_every_domain_case_KL=all(v['DEV']['case_KL']<=c['domain_DEV_case_KL_max'] for v in summary['domains'].values()),
             DEV_every_domain_label_disagreement=all(v['DEV']['label_disagreement']<=c['domain_DEV_label_disagreement_max'] for v in summary['domains'].values()),
             final_routing_support=all(np.count_nonzero(row)>=8 for row in total))
        write(a.directory/'support.json',{d:v.tolist() for d,v in support.items()})
        phase='result'; guard()
        write(a.out,dict(schema='HYBRID_STATE_EVALUATION_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
             process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),checkpoint=b['checkpoint'],boundary=b['boundary'],
             decision='INTERRUPTED_STATE_PREFIX_RECOVERY_CLOSE' if all(gates.values()) else 'INTERRUPTED_STATE_PREFIX_RECOVERY_FAIL',
             gates=gates,summary=summary,cases=rows,before_F64=before,maximum_label_KL_F32_F64_difference=maximum_delta,
             historical_training_support_available=False,primary512_completion=False,
             GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
             worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
             source_calls=0,student_cases=160,optimizer_updates=0,native_C_calls=0,AQ_dither=False,
             quality_admission=False,native_admission=False,
             scope='New fixed-boundary286 original-teacher-prefix evaluation of all160 calibration cases; same related templates,23 partial source replies. No own-history/fresh task/rate/useful-n/family admission; original incomplete512 run remains incomplete.'))
        event('complete',decision='INTERRUPTED_STATE_PREFIX_RECOVERY_CLOSE' if all(gates.values()) else 'INTERRUPTED_STATE_PREFIX_RECOVERY_FAIL',gates=gates)
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(fault=repr(error),phase=phase,completed_cases=completed,elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--bind',action='store_true')
    p.add_argument('--binding',type=Path); p.add_argument('--binding-sha'); p.add_argument('--freeze')
    p.add_argument('--directory',type=Path); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); bind(a) if a.bind else worker(a)
