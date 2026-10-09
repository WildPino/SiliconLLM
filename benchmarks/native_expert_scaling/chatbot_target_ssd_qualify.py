"""Qualify target SSD storage on saved state and actual long-case adjoints."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'benchmarks/native_expert_scaling'
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def bind(a):
    prior_path = DOC/'chatbot_broad_cost_binding_repair1_20261009.json'
    prior = json.loads(prior_path.read_bytes())
    assert sha(prior_path) == 'd6ceee0f96e2d6d911c8e100fbe9da8c28fc4a296b076deb4fa5dc340f0299e6'
    for item in prior['inputs']:
        path=Path(item['path'])
        if path.resolve() != (B/'chatbot_falcon_usability_launch.py').resolve():
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
    records = json.loads(Path(prior['corpus']).read_bytes())['records']
    files = [Path(v['path']) for v in prior['inputs']]
    files += [prior_path, Path(__file__), B/'chatbot_target_ssd_storage.py', B/'chatbot_falcon_ssd_tiles.py',
              DOC/'CHATBOT_TARGET_SSD_STORAGE_PROTOCOL_20261009.md',
              DOC/'chatbot_broad_cost_result_repair1_20261009.launcher_failure.json',
              ROOT/'results/native_expert_scaling/chatbot_broad_cost_repair1_20261009/first_failure.json']
    baseline = {}
    for identifier in prior['selected_ids']:
        p = ROOT/'results/native_expert_scaling/chatbot_broad_cost_repair1_20261009'/(identifier+'.json')
        baseline[identifier] = str(p.resolve()); files.append(p)
    files = list(dict.fromkeys(p.resolve() for p in files))
    assert len(records) == 48
    write(a.out, dict(schema='TARGET_SSD_STORAGE_BINDING_V1', python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()), checkpoint=prior['checkpoint'], corpus=prior['corpus'],
        selected_ids=prior['selected_ids'], baseline=baseline, normalized_AQ_dither=.025,
        criteria=dict(loss_max_abs=1e-5,group_norm_max_relative=1e-4,relative_RMS=1e-5,max_scaled_abs=1e-4),
        limits=dict(seconds=1800,reserve_seconds=60,OS_bytes=32<<30,GPU_allocated_bytes=11<<30,
                    GPU_reserved_bytes=12<<30,output_bytes=1<<30),
        runtime_binding_scope='Actual audited target286/Adam294/RNG, two adopted FIT packets, prior costs, new storage code and selected runtime/Python/foreign hashes; not a full DLL-tree certificate.',
        inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start = time.monotonic(); assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes()); assert b['schema'] == 'TARGET_SSD_STORAGE_BINDING_V1'
    a.directory.mkdir(exist_ok=False); phase='startup'; completed=[]; torch=None
    try:
        assert Path(sys.executable).resolve()==Path(b['python']).resolve() and sys.version_info[:3]==(3,12,10)
        assert all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels','mamba_ssm','causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        from torch.utils.checkpoint import checkpoint
        from transformers.models.falcon_h1 import modeling_falcon_h1 as code
        import chatbot_hybrid_target as native
        import chatbot_target_ssd_storage as storage
        assert (torch.__version__,transformers.__version__,np.__version__)==('2.6.0+cu124','5.13.1','2.4.6')
        torch.set_num_threads(6);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        proc=psutil.Process();proc.cpu_affinity(list(range(6)))
        def guard():
            lim=b['limits']
            assert time.monotonic()-start<=lim['seconds']-lim['reserve_seconds'],'time reserve'
            assert proc.memory_info().peak_wset<=lim['OS_bytes'],'OS cap'
            assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'],'GPU allocated cap'
            assert torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'GPU reserved cap'
            assert not proc.children(recursive=True)
            assert sum(v.stat().st_size for v in a.directory.iterdir() if v.is_file())<=lim['output_bytes'],'output cap'
        def event(stage,**fields):
            print(json.dumps(dict(stage=stage,elapsed_seconds=time.monotonic()-start,**fields)),flush=True);guard()
        enabled=False; captured={}
        def observe(name,inputs,output):
            if not enabled or name in captured:return
            assert all(v.dtype==torch.float32 for v in inputs) and output.dtype==torch.float32
            saved=dict(inputs=[v.detach().cpu().contiguous() for v in inputs],output=output.detach().cpu().contiguous())
            captured[name]=saved
            def adjoint(g):
                assert 'adjoint' not in saved
                saved['adjoint']=g.detach().cpu().contiguous()
            output.register_hook(adjoint)
        generated=storage.install(code,observe)
        (a.directory/'generated_method.py.txt').write_text(generated,encoding='utf8')
        def robust_aq(x):
            scale=x.detach().abs().amax(-1,keepdim=True).clamp_min(1e-12)/63
            noise=torch.empty_like(x).uniform_(-b['normalized_AQ_dither'],b['normalized_AQ_dither'])
            return torch.round(x.detach()*(1/scale)+noise).clamp(-63,63),scale
        native.aq63=robust_aq
        class TrainingTarget(native.Target):
            def forward(self,ids,positions):
                x=self.embed(ids)
                for block in self.layers:x=checkpoint(block,x,use_reentrant=False,preserve_rng_state=True)
                return self.head(self.final_norm(x)[:,positions])
        phase='state'
        state=torch.load(b['checkpoint'],map_location='cpu',weights_only=True)
        assert state['completed_recovery_updates']==286 and state['target_schema']=='COMPACT_FALCON_TERNARY_TARGET_V1'
        target=TrainingTarget(state['config']).to('cuda');target.load_state_dict(state['model'],strict=True);target.train()
        assert (target.layers[0].core.chunk_size,target.layers[0].core.num_heads,target.layers[0].core.head_dim,target.layers[0].core.ssm_state_size)==(16,48,16,256)
        named=list(target.named_parameters());assert len(named)==211 and sum(p.numel() for _,p in named)==254932736
        optimizer=torch.optim.AdamW(target.parameters(),lr=5e-5,weight_decay=0,foreach=False)
        optimizer.load_state_dict(state['optimizer'])
        assert len(optimizer.state)==211 and all(int(v['step'].item())==294 for v in optimizer.state.values())
        assert sum(v.numel()*v.element_size() for s in optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq'))==2039461888
        by_id={r['id']:r for r in json.loads(Path(b['corpus']).read_bytes())['records']};rows=[]
        event('target_ready')
        for index,identifier in enumerate(b['selected_ids']):
            phase=identifier;rec=by_id[identifier];assert rec['split']=='FIT';enabled=index==1
            torch.set_rng_state(state['torch_CPU_rng']);torch.cuda.set_rng_state_all(state['torch_CUDA_rng'])
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();t0=time.monotonic()
            bits=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4')
            teacher=torch.from_numpy((bits<<16).view('<f4').reshape(rec['logits']['shape'])).to('cuda')
            logp=torch.log_softmax(teacher,-1).detach();prob=logp.exp()
            logits=target(torch.tensor([rec['student_input_ids']],device='cuda'),torch.tensor(rec['positions'],device='cuda')).squeeze(0)
            loss=(prob*(logp-torch.log_softmax(logits,-1))).sum(-1).mean()
            assert torch.isfinite(loss).item();torch.cuda.synchronize();forward=time.monotonic()-t0
            loss.backward();torch.cuda.synchronize();backward=time.monotonic()-t0-forward
            assert all(p.grad is not None and p.grad.shape==p.shape and torch.isfinite(p.grad).all().item() for _,p in named)
            groups=[]
            for site,layer in enumerate(target.layers):
                values={}
                for label,parameters in [
                    ('core',list(layer.core.parameters())),
                    ('banks',list(layer.banks.parameters())),
                    ('norm',list(layer.input_norm.parameters())+list(layer.ff_norm.parameters())),
                ]:
                    squared=sum(float(p.grad.double().square().sum().item()) for p in parameters)
                    assert 0<squared<float('inf');values[label+'_gradient_norm']=squared**.5
                groups.append(dict(layer=site,**values))
            prior=json.loads(Path(b['baseline'][identifier]).read_bytes())
            deltas=[abs(g[k]-old[k])/max(abs(old[k]),1e-30) for g,old in zip(groups,prior['layer_gradient_groups'],strict=True)
                    for k in ('core_gradient_norm','banks_gradient_norm','norm_gradient_norm')]
            row=dict(id=identifier,teacher_forcing_ids=len(rec['student_input_ids']),labels=len(rec['output_ids']),
                KL=loss.item(),loss_absolute_difference=abs(loss.item()-prior['KL']),layer_gradient_groups=groups,
                maximum_group_norm_relative_difference=max(deltas),forward_seconds=forward,backward_seconds=backward,
                seconds_with_capture_and_gradient_inspection=time.monotonic()-t0,
                GPU_allocated_peak_so_far=torch.cuda.max_memory_allocated(),GPU_reserved_peak_so_far=torch.cuda.max_memory_reserved())
            write(a.directory/(identifier+'.json'),row);rows.append(row);completed.append(identifier)
            event('whole_case',**{k:v for k,v in row.items() if k!='layer_gradient_groups'})
            assert row['loss_absolute_difference']<=b['criteria']['loss_max_abs'],'whole loss difference'
            assert max(deltas)<=b['criteria']['group_norm_max_relative'],'gradient group norm difference'
            del logits,loss,teacher,logp,prob,bits
        enabled=False;phase='state_equality'
        for name,p in named:assert torch.equal(p.detach().cpu(),state['model'][name]),name
        current=optimizer.state_dict()['state'];prior=state['optimizer']['state'];assert current.keys()==prior.keys()
        for key,slot in current.items():
            for name in ('step','exp_avg','exp_avg_sq'):assert torch.equal(slot[name].detach().cpu(),prior[key][name]),(key,name)
            guard()
        assert set(captured)==set(storage.OPERATIONS)
        for name,saved in captured.items():
            assert 'adjoint' in saved and torch.isfinite(saved['adjoint']).all().item()
            torch.save(saved,a.directory/(name+'.operands.pt'))
        del named,target,optimizer,state,current,prior,p,slot,layer,parameters;gc.collect();torch.cuda.empty_cache()
        event('whole_equality_pass',captured_contractions=len(captured))
        def compare(reference,candidate):
            assert reference.shape==candidate.shape and reference.dtype==candidate.dtype==torch.float32
            r=reference.detach().reshape(-1);v=candidate.detach().reshape(-1)
            squared=0.;reference_squared=0.;maximum=0.;reference_max=0.
            for offset in range(0,r.numel(),1<<20):
                x=r[offset:offset+(1<<20)].double();y=v[offset:offset+(1<<20)].double()
                assert torch.isfinite(x).all().item() and torch.isfinite(y).all().item()
                delta=x-y;squared+=delta.square().sum().item();reference_squared+=x.square().sum().item()
                maximum=max(maximum,delta.abs().max().item());reference_max=max(reference_max,x.abs().max().item())
            return dict(elements=r.numel(),relative_RMS=(squared/max(reference_squared,1e-60))**.5,
                        maximum_absolute_difference=maximum,max_scaled_absolute=maximum/max(reference_max,1e-30))
        comparisons=[]
        for name in storage.OPERATIONS:
            phase='isolated_CPU_'+name;t0=time.monotonic();saved=captured[name]
            inputs=[v.detach().requires_grad_(True) for v in saved['inputs']]
            reference=storage.contract(name,*inputs,False)
            reference_gradients=torch.autograd.grad(reference,inputs,saved['adjoint'])
            reference=reference.detach()
            candidate=storage.contract(name,*inputs,True)
            candidate_gradients=torch.autograd.grad(candidate,inputs,saved['adjoint'])
            metrics=dict(output=compare(reference,candidate),
                         input_VJP=[compare(r,c) for r,c in zip(reference_gradients,candidate_gradients,strict=True)],
                         CPU_tiled_vs_CUDA_tiled_output=compare(candidate,saved['output']))
            comparison=dict(operation=name,input_shapes=[list(v.shape) for v in inputs],output_shape=list(reference.shape),
                metrics=metrics,seconds=time.monotonic()-t0)
            write(a.directory/(name+'.comparison.json'),comparison);comparisons.append(comparison)
            event('isolated_contraction',**comparison)
            for metric in [metrics['output'],*metrics['input_VJP'],metrics['CPU_tiled_vs_CUDA_tiled_output']]:
                assert metric['relative_RMS']<=b['criteria']['relative_RMS'],'contraction/VJP RMS'
                assert metric['max_scaled_absolute']<=b['criteria']['max_scaled_abs'],'contraction/VJP max'
            del reference,candidate,reference_gradients,candidate_gradients,inputs
        phase='result';guard()
        write(a.out,dict(schema='TARGET_SSD_STORAGE_RESULT_V1',decision='TARGET_F32_SSD_STORAGE_PASS',
            freeze=a.freeze,binding_sha256=a.binding_sha,process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
            observations=rows,contractions=comparisons,cases=2,model_and_moments_unchanged=True,
            finite_parameter_gradients=211,source_calls=0,optimizer_updates=0,native_runs=0,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
            scope='Two target training cases plus four actual longest-FIT block0 F32 CPU formula/VJP comparisons and CPU/CUDA tiled outputs. Whole gradient directions were not saved in the old baseline. No quality/native/rate admission.'))
        event('complete',decision='TARGET_F32_SSD_STORAGE_PASS')
    except BaseException as error:
        failure=dict(fault=repr(error),phase=phase,completed=completed,elapsed_seconds=time.monotonic()-start)
        if torch is not None:failure.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_failure.json',failure);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
