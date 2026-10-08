"""Saved-only F64 KL/ID and complete checkpoint audit; no model forward."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))


def bind(a):
    result=json.loads(a.result.read_bytes())
    terminal=a.result.with_suffix('.terminal.json')
    t=json.loads(terminal.read_bytes())
    assert t['exit_code']==0 and t['result_sha256']==sha(a.result)
    files=[a.result,terminal,a.parent_binding,Path(__file__),B/'chatbot_falcon_usability.py',
           B/'chatbot_falcon_usability_launch.py',DOC/'CHATBOT_HYBRID_PILOT_AUDIT_PROTOCOL_20261008.md',Path(sys.executable)]
    files += [Path(v['path']) for v in t['output_files']]
    files += [Path(v['scores']['path']) for v in result['supervision_records']]
    initial=json.loads((a.directory/'initial_manifest.json').read_bytes())
    files += [Path(v['path']) for v in initial['components']]
    files=list(dict.fromkeys(p.resolve() for p in files))
    inputs=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    found={v['path']:v for v in inputs}
    for v in t['output_files']:
        assert found[v['path']]['sha256']==v['sha256'] and found[v['path']]['bytes']==v['bytes']
    assert sha(a.parent_binding)==result['binding_sha256']==t['binding_sha256']
    assert (a.directory/'learner.pt').resolve()==Path(result['checkpoint']['path']).resolve()
    b=dict(schema='HYBRID_PILOT_AUDIT_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),result_path=str(a.result.resolve()),pilot_directory=str(a.directory.resolve()),
        parent_binding_path=str(a.parent_binding.resolve()),inputs=inputs,
        limits=dict(seconds=600,OS_bytes=8<<30,output_bytes=16<<20),
        runtime_binding_scope='Saved pilot extents and audit code bound. CPU isolated Torch/NumPy versions checked; no native DLL-tree certificate.')
    write(a.out,b)
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(inputs))),flush=True)


def audit(a):
    start=time.monotonic()
    assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes())
    assert b['schema']=='HYBRID_PILOT_AUDIT_BINDING_V1'
    a.directory.mkdir(exist_ok=False)
    import psutil
    import numpy as np
    import torch
    assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6'
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    proc=psutil.Process()
    proc.cpu_affinity(list(range(6)))
    result=json.loads(Path(b['result_path']).read_bytes())
    parent=Path(b['pilot_directory'])
    summaries={}
    records=[]
    try:
        def read(item):
            assert sha(item['path'])==item['sha256']
            x=np.fromfile(item['path'],dtype='<f4').reshape(item['shape']).astype(np.float64)
            assert np.isfinite(x).all()
            return x
        def logsoftmax(x):
            m=x.max(-1,keepdims=True)
            return x-m-np.log(np.exp(x-m).sum(-1,keepdims=True))
        for label in ('before','after'):
            rows=[]
            for original in result[label]['cases']:
                source=next(v for v in result['supervision_records'] if v['id']==original['id'])
                s=read(source['scores'])
                q=read(original['logits'])
                lp,lq=logsoftmax(s),logsoftmax(q)
                kl=float((np.exp(lp)*(lp-lq)).sum(-1).mean())
                differences=int(np.count_nonzero(s.argmax(-1)!=q.argmax(-1)))
                assert s.argmax(-1).tolist()==source['output_ids']
                assert abs(kl-original['KL'])<=1e-4, (original['id'],label,kl,original['KL'])
                assert differences==original['disagreements']
                row=dict(id=original['id'],split=original['split'],positions=s.shape[0],KL_F64=kl,
                         disagreements=differences,F32_KL_absolute_error=abs(kl-original['KL']))
                rows.append(row)
            summary={}
            for split in ('FIT','DEV'):
                chosen=[r for r in rows if r['split']==split]
                n=sum(r['positions'] for r in chosen)
                summary[split]=dict(positions=n,mean_KL=sum(r['KL_F64']*r['positions'] for r in chosen)/n,
                    disagreements=sum(r['disagreements'] for r in chosen),
                    disagreement_fraction=sum(r['disagreements'] for r in chosen)/n)
                assert abs(summary[split]['mean_KL']-result[label][split]['mean_KL'])<=1e-4
            summaries[label]=summary
            records += [dict(stage=label,**r) for r in rows]
        checkpoint=torch.load(parent/'learner.pt',map_location='cpu',weights_only=True)
        assert checkpoint['freeze']==result['freeze'] and checkpoint['binding_sha256']==result['binding_sha256']
        state=checkpoint['model']
        elements=sum(v.numel() for v in state.values())
        assert elements==result['trainable_parameters']==254932736
        assert all(v.dtype==torch.float32 and torch.isfinite(v).all().item() for v in state.values())
        assert state['embed.weight'].data_ptr()!=state['head.weight'].data_ptr()
        initial=json.loads((parent/'initial_manifest.json').read_bytes())
        changed=[]
        for component in initial['components']:
            assert sha(component['path'])==component['sha256']
            old=torch.load(component['path'],map_location='cpu',weights_only=True)
            for key,value in old.items():
                full=component['component']+'.'+key
                assert value.shape==state[full].shape and torch.isfinite(value).all().item()
                if not torch.equal(value,state[full]):
                    changed.append(full)
            del old
        assert 'head.weight' in changed and 'embed.weight' in changed
        for i in range(12):
            assert any(v.startswith(f'layers.{i}.core.') for v in changed)
            assert any(v.startswith(f'layers.{i}.banks.') for v in changed)
            for key in ('gate_scale','up_scale','down_scale'):
                assert (state[f'layers.{i}.banks.{key}']>=1e-8).all().item()
        optimizer=checkpoint['optimizer']
        moments=optimizer['state']
        ids=optimizer['param_groups'][0]['params']
        assert len(ids)==len(moments)==len(state)
        moment_bytes=0
        for ident,weight in zip(ids,state.values(),strict=True):
            m=moments[ident]
            assert m['step'].item()==8
            for key in ('exp_avg','exp_avg_sq'):
                assert m[key].shape==weight.shape and m[key].dtype==torch.float32 and torch.isfinite(m[key]).all().item()
                moment_bytes+=m[key].numel()*4
            assert (m['exp_avg_sq']>=0).all().item()
        assert moment_bytes==elements*8
        assert len(checkpoint['updates'])==len(result['updates'])==8
        for i,step in enumerate(result['updates'],1):
            assert step==checkpoint['updates'][i-1]==json.loads((parent/f'update_{i:02d}.json').read_bytes())
            assert step['step']==i and math.isfinite(step['KL']) and step['optimizer_moment_bytes']==moment_bytes
            assert len(step['layer_gradient_norms'])==12
            assert all(math.isfinite(row[k]) and row[k]>0 for row in step['layer_gradient_norms'] for k in ('core','experts','norms'))
        gates=dict(complete_finite_gradient_updates=True,
               FIT_KL_improves=summaries['after']['FIT']['mean_KL']<summaries['before']['FIT']['mean_KL'],
               DEV_KL_no_more_than_2pct_worse=summaries['after']['DEV']['mean_KL']<=1.02*summaries['before']['DEV']['mean_KL'])
        assert gates==result['gates']
        decision='PILOT_RECOVERY_PASS' if all(gates.values()) else 'PILOT_RECOVERY_FAIL'
        assert decision==result['decision']
        assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and time.monotonic()-start<=600
        report=dict(schema='HYBRID_PILOT_AUDIT_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
              process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),decision='SAVED_AUDIT_PASS',
              pilot_decision=decision,gates=gates,summaries=summaries,cases=records,parameters=elements,
              moment_bytes=moment_bytes,changed_parameter_tensors=changed,changed_count=len(changed),
              elapsed_seconds=time.monotonic()-start,worker_OS_peak_snapshot=proc.memory_info().peak_wset,
              model_forwards=0,source_generations=0,training_updates=0,
              limits='Numerical/state/provenance audit only; saved metrics are not independent experimental replication or native parity.')
        write(a.out,report)
        print(json.dumps(dict(decision=report['decision'],pilot=decision,changed=len(changed),seconds=report['elapsed_seconds'])),flush=True)
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(fault=repr(error),elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('bind','audit'),default='audit')
    p.add_argument('--binding',type=Path)
    p.add_argument('--binding-sha')
    p.add_argument('--freeze')
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--result',type=Path)
    p.add_argument('--parent-binding',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    bind(a) if a.mode=='bind' else audit(a)
