"""Bind and validate zero-common upcycling of an audited recovery checkpoint."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))


def bind(a):
    audit_path=DOC/'chatbot_hybrid_recovery_audit_result_20261009.json'
    evaluation_path=DOC/'chatbot_hybrid_state_evaluation_result_20261009.json'
    audit=json.loads(audit_path.read_bytes()); evaluation=json.loads(evaluation_path.read_bytes())
    for path in (audit_path,evaluation_path):
        terminal=json.loads(path.with_suffix('.terminal.json').read_bytes())
        assert terminal['exit_code']==0 and terminal['resource_gates'] and terminal['result_sha256']==sha(path)
    assert audit['decision']=='SAVED_RECOVERY_AUDIT_PASS' and audit['checkpoint_boundary']==286
    assert evaluation['boundary']==286 and evaluation['checkpoint']==audit['checkpoint']
    evaluation_binding_path=DOC/'chatbot_hybrid_state_evaluation_binding_20261009.json'
    assert sha(evaluation_binding_path)==evaluation['binding_sha256']
    evaluation_binding=json.loads(evaluation_binding_path.read_bytes())
    base_identity=next(v for v in evaluation_binding['inputs'] if Path(v['path']).resolve()==(B/'chatbot_hybrid_target.py').resolve())
    assert sha(B/'chatbot_hybrid_target.py')==base_identity['sha256']
    state=Path(audit['checkpoint']['path']); corpus_path=Path(audit['corpus']['path'])
    assert sha(state)==audit['checkpoint']['sha256'] and state.stat().st_size==audit['checkpoint']['bytes']
    assert sha(corpus_path)==audit['corpus']['sha256']
    corpus=json.loads(corpus_path.read_bytes()); fit=[r for r in corpus['records'] if r['split']=='FIT']
    shortest=min(fit,key=lambda r:(len(r['student_input_ids']),r['id']))
    longest=min(fit,key=lambda r:(-len(r['student_input_ids']),r['id']))
    assert longest['id']=='fit_reading_08' and len(longest['student_input_ids'])==183
    chosen=[shortest,longest]; rows={r['id']:r for r in evaluation['cases']}
    references={r['id']:rows[r['id']]['logits'] for r in chosen}
    for v in references.values(): assert Path(v['path']).stat().st_size==v['bytes'] and sha(v['path'])==v['sha256']
    assert sha(longest['logits']['path'])==longest['logits']['sha256']
    files=[Path(__file__),B/'chatbot_hybrid_shared_private_target.py',B/'chatbot_hybrid_target.py',
       B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',Path(sys.executable),
       DOC/'CHATBOT_HYBRID_SHARED_PRIVATE_INIT_PROTOCOL_20261009.md',audit_path,audit_path.with_suffix('.terminal.json'),
       evaluation_path,evaluation_path.with_suffix('.terminal.json'),evaluation_binding_path,state,corpus_path,Path(longest['logits']['path'])]
    files += [Path(v['path']) for v in references.values()]
    files += [SITE/'transformers'/v for v in ('models/falcon_h1/modeling_falcon_h1.py','models/falcon_h1/configuration_falcon_h1.py','cache_utils.py')]
    files += [SITE/'torch'/v for v in ('serialization.py','optim/adamw.py','optim/optimizer.py','utils/checkpoint.py')]
    foreign={'benchmarks/donor_adaptation/configs/_manifest.json':'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
      'benchmarks/donor_adaptation/density/build_document_holdout.py':'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
      'docs/research/RESEARCH_INDEX.md':'99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}
    assert all(sha(ROOT/k)==v for k,v in foreign.items()); files += [ROOT/k for k in foreign]
    files=list(dict.fromkeys(p.resolve() for p in files))
    write(a.out,dict(schema='HYBRID_SHARED_PRIVATE_INIT_BINDING_V1',python=str(Path(sys.executable).resolve()),
       worker_path=str(Path(__file__).resolve()),parent_checkpoint=audit['checkpoint'],corpus=str(corpus_path.resolve()),
       audit=str(audit_path.resolve()),evaluation=str(evaluation_path.resolve()),case_ids=[r['id'] for r in chosen],
       longest_case=longest['id'],references=references,common_precision='float32',common_seed_private_ids=[0,36],
       limits=dict(seconds=300,OS_bytes=12<<30,GPU_allocated_bytes=7<<30,GPU_reserved_bytes=9<<30,output_bytes=4<<30),
       runtime_binding_scope='Actual audited parent/model/moments and selected saved references; versioned common/private module,original operators,isolated versions and selected runtime hashes. No CUDA bit determinism or full DLL-tree certificate.',
       inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files),cases=[r['id'] for r in chosen])),flush=True)


def worker(a):
    start=time.monotonic(); assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes()); assert b['schema']=='HYBRID_SHARED_PRIVATE_INIT_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve() and sys.version_info[:3]==(3,12,10)
    assert all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
    a.directory.mkdir(exist_ok=False); phase='imports'; completed=[]
    try:
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels','mamba_ssm','causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        import tokenizers
        import chatbot_hybrid_shared_private_target as variant
        assert (torch.__version__,transformers.__version__,np.__version__,tokenizers.__version__,psutil.__version__)==('2.6.0+cu124','5.13.1','2.4.6','0.22.2','7.2.2')
        proc=psutil.Process();proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(20261009)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def guard():
            v=b['limits']
            assert time.monotonic()-start<=v['seconds']-30,'worker deadline reserve'
            assert proc.memory_info().peak_wset<=v['OS_bytes'],'OS cap'
            assert torch.cuda.max_memory_allocated()<=v['GPU_allocated_bytes'],'GPU allocated cap'
            assert torch.cuda.max_memory_reserved()<=v['GPU_reserved_bytes'],'GPU reserved cap'
            assert not proc.children(recursive=True),'worker subprocess'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=v['output_bytes'],'output cap'
        def event(stage,**fields):
            print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**fields)),flush=True);guard()
        def same_bits(left,right):
            assert left.dtype==right.dtype==torch.float32 and left.shape==right.shape
            return torch.equal(left.detach().cpu().contiguous().view(torch.int32),right.detach().cpu().contiguous().view(torch.int32))
        def save(path,value):
            with path.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
            return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path))
        phase='conversion'
        original=torch.load(b['parent_checkpoint']['path'],map_location='cpu',weights_only=True)
        assert original['target_schema']=='COMPACT_FALCON_TERNARY_TARGET_V1' and original['completed_recovery_updates']==286
        target=variant.Target(original['config'],b['common_precision']).to('cuda')
        mapped=variant.mapped_model(original['model']);target.load_state_dict(mapped,strict=True)
        named=list(target.named_parameters());names=[n for n,_ in named]
        assert len(named)==283 and sum(p.numel() for _,p in named)==variant.PARAMETERS
        for name,p in named:
            assert p.dtype==torch.float32 and torch.isfinite(p).all().item()
            if '.banks.common.' not in name:assert same_bits(p,original['model'][variant.canonical_private_name(name)])
        for i,layer in enumerate(target.layers):
            common=layer.banks.common; old=f'layers.{i}.banks.'
            for label in ('gate','up','gate_scale','up_scale'):
                assert same_bits(getattr(common,label),original['model'][old+label][[0,36]])
            assert torch.count_nonzero(common.down).item()==0 and torch.all(common.down_scale>0).item()
        remapped,mapping=variant.mapped_optimizer(original['model'],original['optimizer'],names)
        assert all(int(v['step'].item())==294 for v in remapped['state'].values())
        packet=dict(model=mapped,optimizer=remapped,optimizer_parameter_names=names,config=original['config'],
           target_schema=variant.SCHEMA,common_precision='float32',common_count=2,common_seed_private_ids=[0,36],
           parent_checkpoint=b['parent_checkpoint'],prior_optimizer_step=294,source_recovery_boundary=286,common_updates=0,
           torch_CPU_rng=original['torch_CPU_rng'],torch_CUDA_rng=original['torch_CUDA_rng'],
           prior_recovery_orders=original['orders'],freeze=a.freeze,binding_sha256=a.binding_sha)
        checkpoint=save(a.directory/'zero_common.pt',packet)
        write(a.directory/'parameter_mapping.json',dict(private=mapping,new_common=[n for n in names if '.banks.common.' in n]))
        event('converted',parameters=variant.PARAMETERS,tensors=283,old_slots=211,new_slots=0,checkpoint=checkpoint)
        records={r['id']:r for r in json.loads(Path(b['corpus']).read_bytes())['records']}
        common_checks=[0]*12;collect=True
        def zero_hook(site):
            def check(module,args,output):
                if collect:
                    assert torch.isfinite(output).all().item() and torch.count_nonzero(output).item()==0,(site,'common zero')
                    common_checks[site]+=1
            return check
        handles=[layer.banks.common.register_forward_hook(zero_hook(i)) for i,layer in enumerate(target.layers)]
        def inputs(rec):
            assert rec['student_input_ids']==rec['input_ids']+rec['output_ids'][:-1]
            return torch.tensor([rec['student_input_ids']],device='cuda'),torch.tensor(rec['positions'],device='cuda')
        def raw(path,values):
            with path.open('xb') as f:f.write(values.astype('<f4',copy=False).tobytes());f.flush();os.fsync(f.fileno())
            return dict(path=str(path.resolve()),shape=list(values.shape),bytes=path.stat().st_size,sha256=sha(path))
        target.requires_grad_(False);target.eval();phase='zero_prefix';prefixes=[]
        with torch.no_grad():
            for identifier in b['case_ids']:
                rec=records[identifier];ids,positions=inputs(rec);logits=target(ids,positions).squeeze(0)
                assert torch.isfinite(logits).all().item();values=logits.cpu().numpy()
                ref=b['references'][identifier];reference=np.fromfile(ref['path'],dtype='<f4').reshape(ref['shape'])
                assert values.shape==reference.shape
                mismatch=int(np.count_nonzero(values.view('<u4')!=reference.view('<u4')))
                row=dict(id=identifier,input_tokens=len(rec['student_input_ids']),labels=len(rec['output_ids']),
                     differing_F32_bits=mismatch,logits=raw(a.directory/(identifier+'.zero.logits.f32'),values))
                write(a.directory/(identifier+'.zero.json'),row);prefixes.append(row);completed.append(identifier)
                event('zero_prefix',case=identifier,labels=row['labels'],differing_F32_bits=mismatch)
        assert common_checks==[2]*12 and all(r['differing_F32_bits']==0 for r in prefixes),'zero-prefix preservation'
        collect=False
        for handle in handles:handle.remove()
        phase='optimizer'
        optimizer=variant.make_optimizer(target,packet)
        by_name=dict(named);old_ids=dict(zip(original['model'],original['optimizer']['param_groups'][0]['params'],strict=True))
        def check_private():
            total=0
            for m in mapping:
                p=by_name[m['new_name']];old=original['optimizer']['state'][m['original_id']];slot=optimizer.state[p]
                assert p.grad is None and same_bits(p,original['model'][m['original_name']])
                assert int(slot['step'].item())==int(old['step'].item())==294
                for label in ('exp_avg','exp_avg_sq'):
                    assert same_bits(slot[label],old[label]);total+=slot[label].numel()*4
                guard()
            assert total==2039461888
            return total
        moments=check_private()
        down=[layer.banks.common.down for layer in target.layers]
        assert all(p not in optimizer.state for name,p in named if '.banks.common.' in name)
        for p in down:p.requires_grad_(True)
        target.train();phase='probe_gradient'
        longest=records[b['longest_case']];ids,positions=inputs(longest)
        bits=np.fromfile(longest['logits']['path'],dtype='<u2').astype('<u4');bits<<=16
        source=bits.view('<f4').reshape(longest['logits']['shape'])
        teacher=torch.from_numpy(source).to('cuda');teacher_lp=torch.log_softmax(teacher,-1).detach()
        logits=target(ids,positions).squeeze(0)
        loss=(teacher_lp.exp()*(teacher_lp-torch.log_softmax(logits,-1))).sum(-1).mean()
        assert torch.isfinite(loss).item();before_train_loss=loss.item();loss.backward()
        gradients=[]
        for i,p in enumerate(down):
            assert p.grad is not None and p.grad.shape==p.shape and torch.isfinite(p.grad).all().item()
            norms=p.grad.double().square().sum((1,2)).sqrt().cpu().tolist()
            assert len(norms)==2 and all(0<v<float('inf') for v in norms)
            gradients.append(dict(layer=i,common_down_branch_norms=norms))
        assert all(p.grad is None for n,p in named if not n.endswith('.common.down'))
        norm=torch.nn.utils.clip_grad_norm_(down,1.0,error_if_nonfinite=True).item();optimizer.step();optimizer.zero_grad(set_to_none=True)
        changed=sum(torch.count_nonzero(p).item()>0 for p in down);assert changed>0
        for p in down:
            slot=optimizer.state[p]
            assert int(slot['step'].item())==1 and torch.isfinite(p).all().item()
            assert torch.isfinite(slot['exp_avg']).all().item() and torch.isfinite(slot['exp_avg_sq']).all().item()
            assert torch.all(slot['exp_avg_sq']>=0).item()
        check_private();target.eval();phase='post_probe'
        with torch.no_grad():
            after=target(ids,positions).squeeze(0);assert torch.isfinite(after).all().item()
            after_values=after.cpu().numpy()
        def logp64(value):
            shifted=value-value.max(-1,keepdims=True);return shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))
        lp=logp64(source.astype(np.float64));lq=logp64(after_values.astype(np.float64))
        after_kl=(np.exp(lp)*(lp-lq)).sum(-1);assert np.isfinite(after_kl).all() and after_kl.min()>=-1e-10
        after_raw=raw(a.directory/'probe.after.logits.f32',after_values)
        small=dict(common={f'layers.{i}.banks.common.{k}':v.detach().cpu() for i,layer in enumerate(target.layers) for k,v in layer.banks.common.state_dict().items()},
          new_down_optimizer={f'layers.{i}.banks.common.down':{k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in optimizer.state[p].items()} for i,p in enumerate(down)},
          zero_checkpoint=checkpoint,parent_checkpoint=b['parent_checkpoint'],common_updates=1,common_precision='float32',
          torch_CPU_rng=torch.get_rng_state(),torch_CUDA_rng=torch.cuda.get_rng_state_all(),
          scope='One fixed-longest-FIT common-down-only connectivity probe,not a quality-selected pilot checkpoint.')
        probe=save(a.directory/'common_probe.pt',small)
        event('probe_complete',common_down_changed=changed,old_slots_unchanged=211,old_step=294,new_down_step=1,
              training_STE_KL_before=before_train_loss,inference_KL_F64_after=float(after_kl.mean()))
        guard();phase='result'
        write(a.out,dict(schema='HYBRID_SHARED_PRIVATE_INIT_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
          process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),decision='SHARED_PRIVATE_INIT_PASS',
          checkpoint=checkpoint,probe_checkpoint=probe,target_schema=variant.SCHEMA,common_precision='float32',
          total_parameters=variant.PARAMETERS,parameter_tensors=283,original_fields_bit_preserved=211,
          old_optimizer_slots_bit_preserved=211,old_optimizer_moment_bytes=moments,old_optimizer_step=294,
          initial_new_optimizer_slots=0,probe_new_down_slots=12,probe_new_down_step=1,
          common_seed_private_ids=[0,36],zero_prefixes=prefixes,common_zero_calls_by_layer=common_checks,
          gradient_probe=dict(case=longest['id'],labels=len(longest['output_ids']),layer_norms=gradients,
             global_norm_before_clip=norm,changed_common_down_tensors=changed,training_STE_KL_before=before_train_loss,
             inference_KL_F64_after=float(after_kl.mean()),after_logits=after_raw),
          GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
          worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
          source_calls=0,old_target_calls=0,new_variant_forward_calls=4,optimizer_updates=1,native_C_calls=0,
          quality_admission=False,native_admission=False,
          scope='Exact state/Adam upcycle,two new long/short zero-common prefixes vs saved286 rows,one common-down-only float32 connectivity update. Original DEV failure remains;common ternary QAT/export/engine.c/fresh quality+50/useful-n/families unqualified.'))
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(fault=repr(error),phase=phase,completed_zero_prefixes=completed,elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bind',action='store_true');p.add_argument('--binding',type=Path)
    p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();bind(a) if a.bind else worker(a)
