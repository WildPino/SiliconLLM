"""Price two new FIT forward/backward cases from audited state; no update."""
import argparse
import importlib.util
import json
import os
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
    adopted=DOC/'chatbot_broad_adoption_result_20261009.json'
    adoption=json.loads(adopted.read_bytes())
    terminal=adopted.with_suffix('.terminal.json');receipt=json.loads(terminal.read_bytes())
    assert adoption['decision']=='SAVED_BROAD_TRANSPORT_PASS' and receipt['exit_code']==0
    assert receipt['result_sha256']==sha(adopted)
    corpus=Path(adoption['corpus']['path']);assert sha(corpus)==adoption['corpus']['sha256']
    records=json.loads(corpus.read_bytes())['records']
    fit=sorted([r for r in records if r['split']=='FIT'],key=lambda r:(len(r['student_input_ids']),r['id']))
    assert len(fit)==24
    selected=[fit[0],fit[-1]];assert selected[0]['id']!=selected[-1]['id']
    audit=DOC/'chatbot_hybrid_recovery_audit_result_20261009.json'
    state=json.loads(audit.read_bytes());assert state['decision']=='SAVED_RECOVERY_AUDIT_PASS' and state['optimizer_step']==294
    checkpoint=Path(state['checkpoint']['path'])
    assert sha(checkpoint)==state['checkpoint']['sha256']=='7b95e699a57c9bc0ed320bef016d95ba78ac1df2e79b9ea98a0c12a1ba78eec5'
    files=[adopted,terminal,corpus,checkpoint,audit,audit.with_suffix('.terminal.json'),Path(sys.executable),
           DOC/'CHATBOT_BROAD_COST_PROTOCOL_20261009.md',Path(__file__),B/'chatbot_hybrid_target.py',
           B/'chatbot_hybrid_recovery.py',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py']
    files += [Path(r['logits']['path']) for r in selected]
    files += [SITE/'torch'/name for name in ('utils/checkpoint.py','optim/adamw.py','optim/optimizer.py')]
    files += [SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py']
    files += [ROOT/name for name in ('benchmarks/donor_adaptation/configs/_manifest.json',
               'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    files=list(dict.fromkeys(p.resolve() for p in files))
    inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    write(a.out,dict(schema='BROAD_CHAT_COST_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),checkpoint=str(checkpoint),corpus=str(corpus),
        selected_ids=[r['id'] for r in selected],selection='Shortest/longest FIT teacher-forcing length;ID tie break;metadata only.',
        limits=dict(seconds=900,reserve_seconds=60,OS_bytes=16<<30,GPU_allocated_bytes=11<<30,
                    GPU_reserved_bytes=12<<30,output_bytes=64<<20),normalized_AQ_dither=.025,
        runtime_binding_scope='Audited model/Adam/RNG/new adopted packets/code/Python/selected runtime/foreign hashes;not full native DLL tree.',
        inputs=inputs))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(inputs))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='BROAD_CHAT_COST_BINDING_V1'
    a.directory.mkdir(exist_ok=False);phase='startup';completed=[];torch=None
    try:
        assert sys.version_info[:3]==(3,12,10) and Path(sys.executable).resolve()==Path(b['python']).resolve()
        assert all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels','mamba_ssm','causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        from torch.utils.checkpoint import checkpoint
        import chatbot_hybrid_target as native
        assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='5.13.1' and np.__version__=='2.4.6'
        torch.set_num_threads(6);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        proc=psutil.Process();proc.cpu_affinity(list(range(6)))
        def guard():
            lim=b['limits'];assert time.monotonic()-start<=lim['seconds']-lim['reserve_seconds']
            assert proc.memory_info().peak_wset<=lim['OS_bytes']
            assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes']
            assert torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
            assert not proc.children(recursive=True)
        def robust_aq(x):
            scale=x.detach().abs().amax(-1,keepdim=True).clamp_min(1e-12)/63
            noise=torch.empty_like(x).uniform_(-b['normalized_AQ_dither'],b['normalized_AQ_dither'])
            return torch.round(x.detach()*(1/scale)+noise).clamp(-63,63),scale
        native.aq63=robust_aq
        class CostTarget(native.Target):
            def forward(self,ids,positions):
                x=self.embed(ids)
                for block in self.layers:
                    x=checkpoint(block,x,use_reentrant=False,preserve_rng_state=True)
                return self.head(self.final_norm(x)[:,positions])
        phase='load_audited_state'
        state=torch.load(b['checkpoint'],map_location='cpu',weights_only=True)
        assert state['completed_recovery_updates']==286 and state['target_schema']=='COMPACT_FALCON_TERNARY_TARGET_V1'
        metadata=json.loads(Path(b['corpus']).read_bytes())['records']
        workspace=[]
        for rec in metadata:
            t=len(rec['student_input_ids']);chunks=(t+15)//16;boundaries=chunks+1
            workspace.append(dict(id=rec['id'],split=rec['split'],teacher_forcing_ids=t,chunk_size=16,
                source_chunks=chunks,interchunk_boundaries=boundaries,
                interchunk_product_F32_bytes=boundaries**2*48*16*256*4,
                intra_chunk_product_F32_bytes=chunks*16*16*48*256*4))
        write(a.directory/'workspace_extents.json',dict(records=workspace,
              scope='Exact tensor extents from bound SSD code and fixed target geometry;not an observed allocation or summed peak. Longest DEV may exceed both FIT profiles.'))
        target=CostTarget(state['config']).to('cuda');target.load_state_dict(state['model'],strict=True);target.train()
        assert (target.layers[0].core.chunk_size,target.layers[0].core.num_heads,
                target.layers[0].core.head_dim,target.layers[0].core.ssm_state_size)==(16,48,16,256)
        named=list(target.named_parameters());assert len(named)==211 and sum(p.numel() for _,p in named)==254932736
        optimizer=torch.optim.AdamW(target.parameters(),lr=5e-5,weight_decay=0,foreach=False)
        optimizer.load_state_dict(state['optimizer'])
        assert len(optimizer.state)==211 and all(int(v['step'].item())==294 for v in optimizer.state.values())
        assert sum(v.numel()*v.element_size() for s in optimizer.state.values() for k,v in s.items()
                   if k in ('exp_avg','exp_avg_sq'))==2039461888
        by_id={r['id']:r for r in json.loads(Path(b['corpus']).read_bytes())['records']};rows=[]
        guard()
        for identifier in b['selected_ids']:
            phase=identifier;record=by_id[identifier];assert record['split']=='FIT'
            torch.set_rng_state(state['torch_CPU_rng']);torch.cuda.set_rng_state_all(state['torch_CUDA_rng'])
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();t0=time.monotonic()
            bits=np.fromfile(record['logits']['path'],dtype='<u2').astype('<u4')
            teacher=torch.from_numpy((bits<<16).view('<f4').reshape(record['logits']['shape'])).to('cuda')
            logp=torch.log_softmax(teacher,-1).detach();prob=logp.exp()
            logits=target(torch.tensor([record['student_input_ids']],device='cuda'),
                          torch.tensor(record['positions'],device='cuda')).squeeze(0)
            loss=(prob*(logp-torch.log_softmax(logits,-1))).sum(-1).mean()
            assert torch.isfinite(loss).item();forward_seconds=time.monotonic()-t0
            loss.backward();torch.cuda.synchronize();backward_seconds=time.monotonic()-t0-forward_seconds
            assert all(p.grad is not None and p.grad.shape==p.shape and torch.isfinite(p.grad).all().item() for _,p in named)
            groups=[]
            for index,layer in enumerate(target.layers):
                values={}
                for label in ('core','banks','norm'):
                    selected=[p.grad for name,p in layer.named_parameters()
                              if (name.startswith('core') if label=='core' else name.startswith('banks') if label=='banks' else name.startswith('norm'))]
                    assert selected
                    squared=sum(float(g.double().square().sum().item()) for g in selected)
                    assert 0<squared<float('inf'),(index,label)
                    values[label+'_gradient_norm']=squared**.5
                groups.append(dict(layer=index,**values))
            row=dict(id=identifier,input_ids=len(record['input_ids']),teacher_forcing_ids=len(record['student_input_ids']),
                     labels=len(record['output_ids']),stop=record['stop'],training_dither=True,KL=loss.detach().item(),
                     forward_seconds=forward_seconds,backward_seconds=backward_seconds,
                     seconds_with_gradient_inspection=time.monotonic()-t0,layer_gradient_groups=groups,
                     GPU_allocated_peak_so_far=torch.cuda.max_memory_allocated(),GPU_reserved_peak_so_far=torch.cuda.max_memory_reserved())
            write(a.directory/(identifier+'.json'),row);rows.append(row);completed.append(identifier)
            del logits,loss,teacher,logp,prob,bits;guard();print(json.dumps(row),flush=True)
        phase='unchanged_state_check'
        for name,p in named:
            assert torch.equal(p.detach().cpu(),state['model'][name]),name
        current=optimizer.state_dict()['state'];prior=state['optimizer']['state'];assert current.keys()==prior.keys()
        for key,slot in current.items():
            for name in ('step','exp_avg','exp_avg_sq'):
                assert torch.equal(slot[name].detach().cpu(),prior[key][name]),(key,name)
            guard()
        report=dict(schema='BROAD_CHAT_COST_RESULT_V1',decision='TWO_FIT_COST_PREFLIGHT_PASS',freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),cases=2,observations=rows,
            model_elements=254932736,parameter_tensors=211,optimizer_step=294,model_and_moments_unchanged=True,
            source_generations=0,source_forwards=0,student_forwards=2,student_backwards=2,optimizer_updates=0,native_runs=0,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
            scope='Two new FIT training forward/backward cost/connectivity observations. Original inference geometry,original source SSD routine,training-only AQ dither. No model update,whole cohort evaluation,quality/native/rate admission.')
        write(a.out,report);guard();print(json.dumps(dict(decision=report['decision'])),flush=True)
    except BaseException as error:
        failure=dict(fault=repr(error),phase=phase,completed=completed,elapsed_seconds=time.monotonic()-start)
        if torch is not None:failure.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_failure.json',failure);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
