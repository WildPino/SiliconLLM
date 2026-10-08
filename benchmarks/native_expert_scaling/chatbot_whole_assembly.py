"""Actual ALL24 compact/core installation and dense resident Adam allocation.

No full-model/MLP forward, loss, backward, optimizer update or endpoint query.
Byte digests come from the installed tensors, not their input files.
"""
import argparse
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import weakref

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    torch=None;stage='binding'
    r=dict(schema='QWEN_WHOLE_ASSEMBLY_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_student_full_forwards=0,
        new_compact_block_forwards=0,new_optimizer_updates=0,new_endpoint_queries=0,forbidden_forward_attempts=0)
    def resource():
        initialized=torch is not None and torch.cuda.is_initialized()
        return dict(elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_peak_allocated=0 if not initialized else torch.cuda.max_memory_allocated(),
            GPU_peak_reserved=0 if not initialized else torch.cuda.max_memory_reserved(),
            GPU_current_allocated=0 if not initialized else torch.cuda.memory_allocated(),
            GPU_current_reserved=0 if not initialized else torch.cuda.memory_reserved())
    def guard():
        v=resource();assert v['elapsed_seconds']<=300 and v['OS_peak_snapshot']<=16<<30,'assembly time/OS'
        assert v['GPU_peak_allocated']<=8<<30 and v['GPU_peak_reserved']<=9<<30,'GPU limits'
        assert not proc.children(recursive=True),'unexpected descendants'
        assert not args.directory.exists() or sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=8<<20,'output cap'
    def forbidden_forward(module,inputs):
        r['forbidden_forward_attempts']+=1
        raise RuntimeError('Assembly forbids all model/layer/compact forwards')
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='whole_assembly'
        for item in b['inputs']:
            path=Path(item['path']);assert str(path.resolve())==item['resolved_path']
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path'];guard()
        def no_subprocess(event,arguments):
            if event=='subprocess.Popen':raise RuntimeError('Assembly forbids subprocess')
        sys.addaudithook(no_subprocess);stage='imports';print(json.dumps(dict(stage=stage)),flush=True)
        import numpy as np
        import torch as torch_module
        torch=torch_module
        import transformers,tokenizers
        from transformers import AutoModelForCausalLM
        from chatbot_compact_geometry import validate
        from chatbot_whole_checkpoint import hydrate_blocks
        from chatbot_whole_transfer_model import install_compact_model,resident_adam
        assert (torch.__version__,transformers.__version__,tokenizers.__version__,np.__version__)==('2.6.0+cu124','5.13.1','0.22.2','2.4.6')
        assert importlib.util.find_spec('pyarrow') is None and importlib.util.find_spec('datasets') is None
        assert not any(n=='pyarrow' or n.startswith('pyarrow.') for n in sys.modules)
        assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3060'
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(261008);torch.cuda.manual_seed_all(261008)
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        args.directory.mkdir();spec=validate(json.loads(Path(b['spec_path']).read_bytes()))
        stage='load_original_BF16_core_without_forward';print(json.dumps(dict(stage=stage)),flush=True)
        model=AutoModelForCausalLM.from_pretrained(b['source_directory'],local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='eager').to(device).eval()
        assert all(p.dtype==torch.bfloat16 and p.device==device for p in model.parameters())
        for parameter in model.parameters():parameter.requires_grad_(False)
        del parameter
        assert all(layer.mlp.gate_proj.bias is layer.mlp.up_proj.bias is layer.mlp.down_proj.bias is None for layer in model.model.layers)
        core_before={name:value for name,value in model.named_parameters() if '.mlp.' not in name}
        dense_mlp_refs=[weakref.ref(layer.mlp) for layer in model.model.layers]
        dense_parameter_refs=[weakref.ref(p) for n,p in model.named_parameters() if '.mlp.' in n]
        assert len(dense_parameter_refs)==72
        hooks=[module.register_forward_pre_hook(forbidden_forward) for module in (model,model.model,*model.model.layers)]
        r['original_load_resource']=resource();guard()
        stage='hydrate_ALL24_NEW_routing_and_qualified_GUD';print(json.dumps(dict(stage=stage)),flush=True)
        blocks,hydration=hydrate_blocks(b['initializer_path'],b['initializer_audit_path'],b['initializer_audit_binding_path'],spec,device)
        hooks.extend(block.register_forward_pre_hook(forbidden_forward) for block in blocks)
        r['hydration']=hydration;guard()
        stage='install_ALL24_compact_MLPs';r['installation']=install_compact_model(model,blocks,spec)
        model.train();gc.collect();torch.cuda.synchronize();guard()
        assert all(ref() is None for ref in dense_mlp_refs+dense_parameter_refs),'dense source MLPs remain live'
        named=dict(model.named_parameters());core={n:v for n,v in named.items() if '.mlp.' not in n}
        assert core.keys()==core_before.keys() and all(core[n] is core_before[n] for n in core)
        assert model.is_gradient_checkpointing and not model.config.use_cache and model.training
        assert model.lm_head.weight is model.model.embed_tokens.weight
        stage='allocate_ALL_dense_gradients_and_Adam_moments';print(json.dumps(dict(stage=stage,resource=resource())),flush=True)
        optimizer=resident_adam(model);torch.cuda.synchronize();guard()
        params=[(n,v) for n,v in named.items() if v.requires_grad];assert len(params)==1224
        assert len(optimizer.param_groups)==1 and len(optimizer.state)==1224
        group=optimizer.param_groups[0]
        assert (group['lr'],group['betas'],group['eps'],group['weight_decay'],group['foreach'],group['fused'])==(1e-4,(.9,.999),1e-8,0.,False,False)
        assert group['params']==[v for n,v in params]
        def tensor_entry(tensor,digest=False,zero=False):
            assert tensor.is_contiguous() and torch.isfinite(tensor).all()
            entry=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),device=str(tensor.device),
                elements=tensor.numel(),bytes=tensor.numel()*tensor.element_size(),pointer=tensor.data_ptr())
            if digest:
                cpu=tensor.detach().cpu().contiguous();view=memoryview(cpu.reshape(-1).view(torch.uint8).numpy())
                entry['sha256']=hashlib.sha256(view).hexdigest();del view,cpu
            if zero:
                assert int(torch.count_nonzero(tensor))==0
                entry['all_bits_zero']=bool((tensor.detach().reshape(-1).view(torch.uint8)==0).all())
                assert entry['all_bits_zero']
            return entry
        manifest=dict(schema='QWEN_WHOLE_ASSEMBLY_TENSOR_MANIFEST_V1',parameters={},buffers={},optimizer={},
            head_alias='model.embed_tokens.weight',dense_source_MLP_modules_collected=24,dense_source_MLP_parameters_collected=72,
            checkpoint_mode='nonreentrant',cache=False,original_and_student_full_forwards=0)
        ranges=[];cpu_step_ranges=[]
        def account(record,label):
            target=ranges if record['device']=='cuda:0' else cpu_step_ranges
            assert record['bytes']>0;target.append((record['pointer'],record['pointer']+record['bytes'],label))
        stage='installed_bytes_and_optimizer_structure'
        for name,value in named.items():
            entry=tensor_entry(value,digest=True);entry['trainable']=value.requires_grad
            manifest['parameters'][name]=entry;account(entry,name)
            if value.requires_grad:
                state=optimizer.state[value];assert set(state)=={'step','exp_avg','exp_avg_sq'}
                assert value.grad is not None and value.grad.shape==value.shape and value.grad.dtype==torch.float32 and value.grad.device==device
                saved=dict(gradient=tensor_entry(value.grad,zero=True))
                for field in ('exp_avg','exp_avg_sq'):
                    tensor=state[field];assert tensor.shape==value.shape and tensor.dtype==torch.float32 and tensor.device==device
                    saved[field]=tensor_entry(tensor,zero=True)
                step=state['step'];assert step.shape==() and step.dtype==torch.float32 and step.device.type=='cpu'
                saved['step']=tensor_entry(step,zero=True);manifest['optimizer'][name]=saved
                for field,record in saved.items():account(record,name+'/'+field)
            else:assert value.grad is None and name in core
            guard()
        for name,value in model.named_buffers():
            entry=tensor_entry(value,digest=True);assert not value.requires_grad and value.device==device
            manifest['buffers'][name]=entry;account(entry,name);guard()
        for sequence in (ranges,cpu_step_ranges):
            sequence.sort();assert all(left[1]<=right[0] for left,right in zip(sequence,sequence[1:])),'resident storage aliases'
        core_bytes=sum(v['bytes'] for v in manifest['parameters'].values() if not v['trainable'])
        param_bytes=sum(v['bytes'] for v in manifest['parameters'].values() if v['trainable'])
        router_bytes=sum(v['bytes'] for n,v in manifest['buffers'].items() if '.mlp.compact.' in n)
        extra_bytes=sum(v['bytes'] for n,v in manifest['buffers'].items() if '.mlp.compact.' not in n)
        assert (core_bytes,param_bytes,router_bytes)==(360492800,660602880,2853888)
        assert sum(v['step']['bytes'] for v in manifest['optimizer'].values())==4896
        mandatory=core_bytes+4*param_bytes+router_bytes;assert mandatory==3005758208
        manifest['allocation']=dict(core_BF16_bytes=core_bytes,trainable_F32_bytes=param_bytes,
            gradient_F32_bytes=param_bytes,first_moment_F32_bytes=param_bytes,second_moment_F32_bytes=param_bytes,
            router_F32_bytes=router_bytes,additional_HF_buffer_bytes=extra_bytes,CPU_step_F32_bytes=4896,
            mandatory_GPU_array_bytes=mandatory,actual_unique_GPU_array_bytes=sum(e-s for s,e,n in ranges),
            all_GPU_arrays_nonoverlapping=True,all_CPU_steps_nonoverlapping=True,
            optimizer=dict(lr=group['lr'],betas=list(group['betas']),eps=group['eps'],weight_decay=0.,foreach=False,fused=False))
        assert manifest['allocation']['actual_unique_GPU_array_bytes']==mandatory+extra_bytes
        torch.cuda.synchronize();resident=resource();assert resident['GPU_current_allocated']>=mandatory+extra_bytes
        manifest_path=args.directory/'installed_tensor_manifest.json';write_once(manifest_path,manifest)
        assert manifest_path.stat().st_size<=4<<20
        r['installed_manifest']=dict(path=str(manifest_path.resolve()),bytes=manifest_path.stat().st_size,sha256=sha(manifest_path))
        r['allocation']=manifest['allocation'];r['resident_after_allocation_and_byte_checks']=resident
        r['procedure_gates'].update(qualified_ALL24_GUD_and_NEW_buffers_hydrated=True,
            source_core_parameter_objects_and_tied_head_preserved=True,ALL_dense_source_MLPs_collected_without_fallback=True,
            ALL_installed_parameter_and_buffer_byte_digests_retained=True,ALL_dense_gradient_moments_CPU_steps_zero_and_nonaliasing=True,
            checkpoint_mode_cache_and_optimizer_contract_exact=True,no_forward_backward_update_or_endpoint=True,
            safe_runtime_frozen_inputs_and_resource_limits=True)
        assert r['forbidden_forward_attempts']==0
        r.update(decision='WHOLE_ASSEMBLED_AND_ALLOCATED_REQUIRE_FIRST_BYTE_COUNT_AUDIT',
            resource_before_final_serialization=resource(),ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            runtime=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(),TF32=False),
            scope='Actual installation/byte witness/resident optimizer prerequisite; no activation peak, gradient/update feasibility, transfer or native quality')
        guard();write_once(args.out,r);assert args.out.stat().st_size<=1<<20
        print(json.dumps(dict(decision=r['decision'],allocation=r['allocation'],resource=resource())),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,resource_snapshot=resource(),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
