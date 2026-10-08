"""ONE finite complete-output conversion fit: ALL24 student, cached teacher.

Fixed8 epochs/1280 updates and fixed final checkpoint. Initial/final evaluation
is teacher-forced development, never fresh own-history/native admission.
"""
import argparse
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    torch=None;model=None;blocks=None;stage='binding';operation_start=None
    r=dict(schema='QWEN_WHOLE_OUTPUT_FIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_student_full_forwards=0,
        new_backward_calls=0,new_optimizer_updates=0,new_endpoint_queries=0,epochs_completed=0,outputs={},
        fixed_schedule=dict(epochs=8,updates=1280,ordering='manifest FIT order repeated',seed=261008,global_gradient_clip=1.0))
    phase=dict(mode='idle',routes=None)
    def resource():
        initialized=torch is not None and torch.cuda.is_initialized()
        return dict(elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_peak_allocated=0 if not initialized else torch.cuda.max_memory_allocated(),
            GPU_peak_reserved=0 if not initialized else torch.cuda.max_memory_reserved())
    def guard():
        value=resource();assert value['elapsed_seconds']<=7200 and value['OS_peak_snapshot']<=16<<30,'fit time/OS'
        assert value['GPU_peak_allocated']<=8<<30 and value['GPU_peak_reserved']<=9<<30,'GPU limits'
        allowance=90 if stage.startswith('epoch0_case0_') else 30
        assert operation_start is None or time.monotonic()-operation_start<=allowance,'per case/update limit'
        assert not proc.children(recursive=True),'unexpected descendants'
        assert not args.directory.exists() or sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=6<<30,'output cap'
    def tensor_hash(tensor):
        value=tensor.detach().cpu().contiguous();return hashlib.sha256(memoryview(value.reshape(-1).view(torch.uint8).numpy())).hexdigest()
    def receipt(path):
        path=Path(path);guard();return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path))
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='whole_output_fit'
        for item in b['inputs']:
            path=Path(item['path']);assert str(path.resolve())==item['resolved_path']
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path'];guard()
        def no_subprocess(event,arguments):
            if event=='subprocess.Popen':raise RuntimeError('Whole output fit forbids subprocess')
        sys.addaudithook(no_subprocess);stage='imports';print(json.dumps(dict(stage=stage)),flush=True)
        import numpy as np
        import torch as torch_module
        torch=torch_module
        import transformers,tokenizers
        from transformers import AutoModelForCausalLM
        from chatbot_whole_checkpoint import hydrate_blocks
        from chatbot_whole_transfer_model import install_compact_model,resident_adam,zero_resident_gradients,output_objective
        from chatbot_whole_output_data import load_case,logit_metrics,summarize,transfer_gates
        from chatbot_joint_pilot import save_tensors,coefficients
        from chatbot_compact_geometry import validate
        assert (torch.__version__,transformers.__version__,tokenizers.__version__,np.__version__)==('2.6.0+cu124','5.13.1','0.22.2','2.4.6')
        assert importlib.util.find_spec('pyarrow') is None and importlib.util.find_spec('datasets') is None
        assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3060'
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(261008);torch.cuda.manual_seed_all(261008)
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        args.directory.mkdir();spec=validate(json.loads(Path(b['spec_path']).read_bytes()))
        cohort=json.loads(Path(b['cohort_path']).read_bytes());stage='load_qualified_cached_teacher'
        cases=[load_case(summary) for summary in cohort['conversations']]
        fit=[case for case in cases if case['split']=='fit'];assert len(cases)==200 and len(fit)==160
        assert sum(c['label_count'] for c in fit)==2437 and sum(c['label_count'] for c in cases if c['split']=='development')==613
        stage='reuse_qualified_whole_installation';print(json.dumps(dict(stage=stage)),flush=True)
        model=AutoModelForCausalLM.from_pretrained(b['source_directory'],local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='eager').to(device).eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        del parameter
        blocks,hydration=hydrate_blocks(b['initializer_path'],b['initializer_audit_path'],b['initializer_audit_binding_path'],spec,device)
        installation=install_compact_model(model,blocks,spec);r['installation']=installation
        optimizer=resident_adam(model);named=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
        assert len(named)==1224 and sum(v.numel() for n,v in named)==165150720
        params=[v for n,v in named];gc.collect();guard()
        assembled=json.loads(Path(b['installed_manifest_path']).read_bytes())
        core={n:v for n,v in model.named_parameters() if not v.requires_grad}
        assert len(core)==218
        for name,value in core.items():assert tensor_hash(value)==assembled['parameters'][name]['sha256']
        frozen_buffers={n:tensor_hash(v) for n,v in model.named_buffers()}
        assert model.lm_head.weight is model.model.embed_tokens.weight and model.is_gradient_checkpointing and not model.config.use_cache
        for li,block in enumerate(blocks):
            original_route=block.routes
            def tracked_route(x,original_route=original_route,li=li):
                ids,mass=original_route(x)
                if phase['routes'] is not None:
                    assert phase['routes'][li] is None,'one actual forward route per layer'
                    phase['routes'][li]=torch.bincount(ids.detach().reshape(-1),minlength=16).cpu().tolist()
                return ids,mass
            block.routes=tracked_route
        def run(case,retain=False):
            input_ids=torch.tensor(case['input_ids'],dtype=torch.int64,device=device)[None]
            positions=torch.tensor(case['positions'],dtype=torch.int64,device=device)
            teacher=case['teacher_logits'].to(device)
            result=model(input_ids=input_ids,use_cache=False,logits_to_keep=positions)
            r['new_student_full_forwards']+=1;student=result.logits
            assert student.shape==teacher.shape==(1,case['label_count'],151936) and student.dtype==torch.bfloat16
            assert torch.isfinite(student).all() and student.requires_grad==torch.is_grad_enabled()
            if retain:student.retain_grad()
            return student,teacher
        def evaluate(label):
            nonlocal operation_start,stage
            stage='evaluate_'+label;model.eval();records=[]
            wire=args.directory/(label+'.student_logits.bf16.bin');journal=args.directory/(label+'.cases.jsonl')
            with wire.open('xb') as ws,journal.open('xb') as js,torch.no_grad():
                ws.write(struct.pack('<8s4I',b'QWSL0001',151936,2,0,0))
                for case in cases:
                    operation_start=time.monotonic();phase.update(mode=label,routes=[None]*24)
                    student,teacher=run(case);metrics=logit_metrics(student,teacher,case['source_generated_ids'])
                    bits=student[0].contiguous().cpu().view(torch.uint16).numpy();payload=bits.tobytes(order='C')
                    assert len(payload)==case['label_count']*303872
                    assert all(v is not None and len(v)==16 and sum(v)==len(case['input_ids'])*4 for v in phase['routes'])
                    record=dict(id=case['id'],split=case['split'],category=case['category'],**metrics,
                        input_ids=case['input_ids'],decision_positions=case['positions'],source_generated_ids=case['source_generated_ids'],
                        student_logit_offset=ws.tell(),student_logit_bytes=len(payload),student_logit_sha256=hashlib.sha256(payload).hexdigest(),
                        student_route_occurrences_ALL_input_rows=phase['routes'])
                    ws.write(payload);ws.flush();js.write((json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode());js.flush()
                    records.append(record);phase.update(mode='idle',routes=None)
                    del student,teacher,bits,payload;guard();operation_start=None
                for stream in (ws,js):os.fsync(stream.fileno())
            r['outputs'][label+'_logits']=receipt(wire);r['outputs'][label+'_cases']=receipt(journal)
            result=summarize(records)
            print(json.dumps(dict(stage=stage,metrics=result,resource=resource())),flush=True)
            return result
        r['initial_metrics']=evaluate('initial');r['procedure_gates']['initial_ALL200_complete_output_baseline_retained']=True
        stage='fixed8_epoch_training';update_journal=args.directory/'updates.jsonl';sample_indexes=[torch.tensor([0,v.numel()//2,v.numel()-1],device=device,dtype=torch.int64) for n,v in named]
        with update_journal.open('xb') as journal:
            for epoch in range(8):
                model.train();epoch_start=time.monotonic();epoch_losses=[]
                for case_index,case in enumerate(fit):
                    operation_start=time.monotonic();stage=f'epoch{epoch}_case{case_index}_forward';phase.update(mode='training',routes=None)
                    r['pending_update']=dict(epoch=epoch,case_id=case['id'],case_index=case_index,completed_updates=r['new_optimizer_updates'])
                    zero_resident_gradients(optimizer);student,teacher=run(case,retain=(r['new_optimizer_updates']==0))
                    mask=torch.ones(student.shape[:2],dtype=torch.float32,device=device)
                    loss=output_objective(student,teacher,mask);assert torch.isfinite(loss)
                    stage=f'epoch{epoch}_case{case_index}_backward';phase['mode']='backward';loss.backward();r['new_backward_calls']+=1
                    first=r['new_optimizer_updates']==0
                    if first:
                        norms=torch.stack([p.grad.detach().norm() for p in params]);assert torch.isfinite(norms).all()
                        per_layer={str(li):float(torch.linalg.vector_norm(norms[[i for i,(n,p) in enumerate(named) if n.startswith(f'model.layers.{li}.')]])) for li in range(24)}
                        assert all(value>0 for value in per_layer.values()),'zero layer gradient'
                        first_tensors=dict(teacher_logits=teacher.detach().cpu(),student_logits=student.detach().cpu(),
                            student_logit_gradient=student.grad.detach().cpu(),mask=mask.cpu(),gradient_parameter_norms_before_clip=norms.cpu(),
                            before=torch.stack([p.detach().reshape(-1)[idx].cpu() for (n,p),idx in zip(named,sample_indexes)]),
                            gradient_before_clip=torch.stack([p.grad.detach().reshape(-1)[idx].cpu() for (n,p),idx in zip(named,sample_indexes)]))
                        r['first_update']=dict(case_id=case['id'],loss=float(loss.detach()),decision_positions=case['positions'],input_ids=case['input_ids'],
                            parameter_names=[n for n,p in named],parameter_elements=[p.numel() for n,p in named],layer_gradient_norms=per_layer)
                    norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True,foreach=False)
                    assert torch.isfinite(norm) and float(norm)>=0
                    if first:assert float(norm)>0
                    if first:
                        first_tensors['scaled_gradient']=torch.stack([p.grad.detach().reshape(-1)[idx].cpu() for (n,p),idx in zip(named,sample_indexes)])
                        r['first_update'].update(norm_before_clip=float(norm),resource_before_Adam=resource())
                        before_path=args.directory/'first_before_Adam.pt'
                        r['outputs']['first_before_Adam']=save_tensors(before_path,first_tensors)
                        assert r['outputs']['first_before_Adam']['bytes']<=32<<20
                        write_once(args.directory/'first_before_Adam.json',r['first_update'])
                        r['outputs']['first_before_Adam_metadata']=receipt(args.directory/'first_before_Adam.json')
                    stage=f'epoch{epoch}_case{case_index}_Adam_update';optimizer.step();r['new_optimizer_updates']+=1
                    if first:
                        first_tensors['after']=torch.stack([p.detach().reshape(-1)[idx].cpu() for (n,p),idx in zip(named,sample_indexes)])
                        for field in ('exp_avg','exp_avg_sq'):
                            first_tensors[field]=torch.stack([optimizer.state[p][field].detach().reshape(-1)[idx].cpu() for (n,p),idx in zip(named,sample_indexes)])
                        assert any(bool((first_tensors['after'][i]!=first_tensors['before'][i]).any()) for i in range(1224))
                        first_path=args.directory/'first_update.pt';first_receipt=save_tensors(first_path,first_tensors);assert first_receipt['bytes']<=32<<20
                        r['outputs']['first_update']=first_receipt;r['first_update'].update(norm_before_clip=float(norm),resource=resource())
                        write_once(args.directory/'first_update.json',r['first_update']);r['outputs']['first_update_metadata']=receipt(args.directory/'first_update.json')
                        del first_tensors,norms
                    assert all(float(optimizer.state[p]['step'])==r['new_optimizer_updates'] for p in params) if first else True
                    update=dict(epoch=epoch,case_id=case['id'],case_index=case_index,update=r['new_optimizer_updates'],labels=case['label_count'],
                        loss=float(loss.detach()),gradient_norm_before_clip=float(norm),elapsed_seconds=time.monotonic()-operation_start)
                    journal.write((json.dumps(update,separators=(',',':'),allow_nan=False)+'\n').encode());journal.flush();os.fsync(journal.fileno())
                    epoch_losses.append(update['loss']);r.pop('pending_update')
                    del student,teacher,mask,loss,norm;phase['mode']='idle';guard();operation_start=None
                    if r['new_optimizer_updates']%40==0:print(json.dumps(dict(epoch=epoch,updates=r['new_optimizer_updates'],last_loss=update['loss'],resource=resource())),flush=True)
                r['epochs_completed']+=1
                print(json.dumps(dict(epoch_completed=epoch+1,mean_case_KL=sum(epoch_losses)/160,seconds=time.monotonic()-epoch_start)),flush=True)
        r['outputs']['updates']=receipt(update_journal)
        assert r['new_optimizer_updates']==r['new_backward_calls']==1280 and r['epochs_completed']==8
        assert all(float(optimizer.state[p]['step'])==1280 for p in params)
        stage='save_fixed_final_ALL24';finals=[]
        for li,block in enumerate(blocks):
            assert all(torch.isfinite(p).all() for p in block.parameters())
            path=args.directory/f'layer{li:02d}.final.pt';item=save_tensors(path,coefficients(block));assert item['bytes']<=28<<20
            finals.append(dict(layer=li,**item));guard()
        r['final_checkpoints']=finals;r['final_metrics']=evaluate('final')
        assert r['new_student_full_forwards']==1680
        for name,value in core.items():assert value.grad is None and not value.requires_grad and tensor_hash(value)==assembled['parameters'][name]['sha256']
        assert model.lm_head.weight is model.model.embed_tokens.weight and {n:tensor_hash(v) for n,v in model.named_buffers()}==frozen_buffers
        r['transfer_gates']=transfer_gates(r['initial_metrics'],r['final_metrics'])
        r['procedure_gates'].update(qualified_inputs_cache_no_original_or_endpoint_queries=True,
            ALL24_current_student_hidden_states_and_routes_recomputed=True,first_full_loss_backward_ALL24_gradients_Adam_witness_saved=True,
            fixed8_epochs_1280_updates_no_checkpoint_selection=True,final_ALL200_logits_and_fixed24_checkpoints_retained=True,
            frozen_core_router_and_tied_head_unchanged=True,all_finite_and_original_resource_caps=True)
        r.update(decision='WHOLE_OUTPUT_FIT_REQUIRE_FIRST_AUDIT' if all(r['transfer_gates'].values()) else 'CLOSE_FIXED_WHOLE_OUTPUT_FIT_REQUIRE_FIRST_AUDIT',
            resource_before_final_serialization=resource(),ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='Finite teacher-forced complete-output conversion only; BF16 stream autograd surrogate, no fresh own-history, codec/C quality, useful n or rate',
            runtime=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(),TF32=False))
        guard();write_once(args.out,r);assert args.out.stat().st_size<=2<<20;print(json.dumps(dict(decision=r['decision'],transfer_gates=r['transfer_gates'],resource=resource())),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,resource_snapshot=resource(),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
