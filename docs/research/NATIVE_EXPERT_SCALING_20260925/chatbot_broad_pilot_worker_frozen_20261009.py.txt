"""One fixed broader-data pass from audited Adam294 into the original target."""
import argparse
import gc
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
    qualification_path=DOC/'chatbot_target_ssd_storage_result_20261009.json'
    qualification=json.loads(qualification_path.read_bytes())
    terminal_path=qualification_path.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert qualification['decision']=='TARGET_F32_SSD_STORAGE_PASS' and terminal['exit_code']==0 and terminal['resource_gates']
    assert terminal['result_sha256']==sha(qualification_path)
    qualified_binding_path=DOC/'chatbot_target_ssd_storage_binding_20261009.json'
    qualified=json.loads(qualified_binding_path.read_bytes())
    assert sha(qualified_binding_path)==qualification['binding_sha256']==terminal['binding_sha256']
    for item in qualified['inputs']:
        path=Path(item['path'])
        if path.resolve()!=(B/'chatbot_falcon_usability_launch.py').resolve():
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
    old_evaluation_path=DOC/'chatbot_hybrid_state_evaluation_result_20261009.json'
    old=json.loads(old_evaluation_path.read_bytes())
    old_terminal=old_evaluation_path.with_suffix('.terminal.json');receipt=json.loads(old_terminal.read_bytes())
    assert old['boundary']==286 and old['student_cases']==160 and receipt['result_sha256']==sha(old_evaluation_path) and receipt['exit_code']==0
    old_binding_path=DOC/'chatbot_hybrid_state_evaluation_binding_20261009.json'
    old_binding=json.loads(old_binding_path.read_bytes());assert sha(old_binding_path)==old['binding_sha256']
    old_corpus=Path(old_binding['corpus']);old_records=json.loads(old_corpus.read_bytes())['records']
    retention=[r for r in old_records if r['split']=='DEV'];assert len(retention)==32 and sum(len(r['output_ids']) for r in retention)==1103
    corpus=Path(qualified['corpus']);records=json.loads(corpus.read_bytes())['records']
    fit=[r for r in records if r['split']=='FIT'];domains=sorted({r['domain'] for r in fit})
    buckets={d:sorted([r['id'] for r in fit if r['domain']==d]) for d in domains}
    assert len(domains)==12 and all(len(v)==2 for v in buckets.values())
    order=[buckets[d][i] for i in range(2) for d in domains];assert len(set(order))==24
    files=[Path(v['path']) for v in qualified['inputs']]
    files += [Path(__file__),qualified_binding_path,qualification_path,terminal_path,
              DOC/'CHATBOT_BROAD_PILOT_PROTOCOL_20261009.md',old_evaluation_path,old_terminal,old_binding_path,old_corpus]
    files += [Path(r['logits']['path']) for r in records+retention]
    files=list(dict.fromkeys(p.resolve() for p in files))
    write(a.out,dict(schema='BROAD_CHAT_PILOT_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),checkpoint=qualified['checkpoint'],corpus=str(corpus),
        old_corpus=str(old_corpus),retention_ids=[r['id'] for r in retention],old_DEV_baseline=old['summary']['DEV'],
        order=order,updates=24,initial_optimizer_step=294,final_optimizer_step=318,
        normalized_AQ_dither=.025,learning_rate=5e-5,clip_norm=1,
        criteria=dict(FIT_KL_ratio_max=.5,DEV_KL_ratio_max=.75,DEV_case_KL_max=1.,DEV_label_disagreement_max=.2,
                      domain_DEV_case_KL_max=2.,domain_DEV_label_disagreement_max=.35,retention_ratio_max=1.1),
        limits=dict(seconds=2400,reserve_seconds=90,OS_bytes=16<<30,GPU_allocated_bytes=11<<30,
                    GPU_reserved_bytes=12<<30,output_bytes=12<<30),
        runtime_binding_scope='Audited286/Adam294/RNG, qualified storage helper, adopted new48/old32 packets, old exact DEV baseline, explicit fixed order/code/Python/selected runtime and foreign hashes; not full DLL-tree.',
        inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='BROAD_CHAT_PILOT_BINDING_V1'
    a.directory.mkdir(exist_ok=False);phase='startup';durable=0;updates=[];torch=None
    try:
        assert sys.version_info[:3]==(3,12,10) and Path(sys.executable).resolve()==Path(b['python']).resolve()
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
            lim=b['limits'];assert time.monotonic()-start<=lim['seconds']-lim['reserve_seconds'],'deadline reserve'
            assert proc.memory_info().peak_wset<=lim['OS_bytes'],'OS cap'
            assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'],'GPU allocated cap'
            assert torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'GPU reserved cap'
            assert not proc.children(recursive=True),'unexpected child'
            assert sum(v.stat().st_size for v in a.directory.iterdir() if v.is_file())<=lim['output_bytes'],'output cap'
        def event(stage,**fields):
            print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**fields)),flush=True);guard()
        generated=storage.install(code);(a.directory/'generated_method.py.txt').write_text(generated,encoding='utf8')
        dither=False;original_aq=native.aq63
        def robust_aq(x):
            if not dither or not torch.is_grad_enabled():return original_aq(x)
            scale=x.detach().abs().amax(-1,keepdim=True).clamp_min(1e-12)/63
            noise=torch.empty_like(x).uniform_(-b['normalized_AQ_dither'],b['normalized_AQ_dither'])
            return torch.round(x.detach()*(1/scale)+noise).clamp(-63,63),scale
        native.aq63=robust_aq
        class PilotTarget(native.Target):
            def forward(self,ids,positions):
                x=self.embed(ids)
                for block in self.layers:
                    x=checkpoint(block,x,use_reentrant=False,preserve_rng_state=True) if self.training and torch.is_grad_enabled() else block(x)
                return self.head(self.final_norm(x)[:,positions])
        phase='load'
        old=torch.load(b['checkpoint'],map_location='cpu',weights_only=True)
        assert old['completed_recovery_updates']==286 and old['target_schema']=='COMPACT_FALCON_TERNARY_TARGET_V1'
        config=old['config'];initial_cpu_rng=old['torch_CPU_rng'];initial_cuda_rng=old['torch_CUDA_rng']
        target=PilotTarget(config).to('cuda');target.load_state_dict(old['model'],strict=True)
        named=list(target.named_parameters());parameters=[p for _,p in named]
        assert len(named)==211 and sum(p.numel() for p in parameters)==254932736 and all(p.dtype==torch.float32 for p in parameters)
        optimizer=torch.optim.AdamW(parameters,lr=b['learning_rate'],weight_decay=0,foreach=False)
        optimizer.load_state_dict(old['optimizer'])
        assert len(optimizer.state)==211 and all(int(s['step'].item())==294 for s in optimizer.state.values())
        for group in optimizer.param_groups:
            assert tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and not group['amsgrad'] and not group['maximize']
            group.update(lr=b['learning_rate'],weight_decay=0,foreach=False)
        del old;gc.collect()
        records=json.loads(Path(b['corpus']).read_bytes())['records'];by_id={r['id']:r for r in records}
        old_records=json.loads(Path(b['old_corpus']).read_bytes())['records'];old_by_id={r['id']:r for r in old_records}
        retention=[old_by_id[v] for v in b['retention_ids']]
        assert len(records)==48 and sum(len(r['output_ids']) for r in records)==8808
        stage_records=dict(before=records,training=[by_id[v] for v in b['order']],after=records,retention=retention)
        support={stage:{d:np.zeros((12,72),dtype=np.int64) for d in sorted({r['domain'] for r in rs})}
                 for stage,rs in stage_records.items()};collection=None
        def hook(site):
            def collect(module,args):
                if collection is None:return
                stage,domain=collection
                with torch.no_grad():
                    scores=module.router(args[0].reshape(-1,512))
                    ids=torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:8]
                    support[stage][domain][site]+=torch.bincount(ids.flatten(),minlength=72).cpu().numpy()
            return collect
        handles=[layer.banks.register_forward_pre_hook(hook(i)) for i,layer in enumerate(target.layers)]
        def observe(rec,stage,retain):
            nonlocal collection
            assert rec['student_input_ids']==rec['input_ids']+rec['output_ids'][:-1]
            bits=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4')
            teacher=torch.from_numpy((bits<<16).view('<f4').reshape(rec['logits']['shape'])).to('cuda')
            logp=torch.log_softmax(teacher,-1).detach();prob=logp.exp()
            collection=(stage,rec['domain'])
            try:logits=target(torch.tensor([rec['student_input_ids']],device='cuda'),torch.tensor(rec['positions'],device='cuda')).squeeze(0)
            finally:collection=None
            assert list(logits.shape)==rec['logits']['shape'] and torch.isfinite(logits).all().item()
            losses=(prob*(logp-torch.log_softmax(logits,-1))).sum(-1);loss=losses.mean()
            assert torch.isfinite(loss).item()
            winners=logits.detach().argmax(-1).cpu().tolist()
            row=dict(id=rec['id'],domain=rec['domain'],split=rec['split'],labels=len(rec['output_ids']),stop=rec['stop'],
                     teacher_forcing_ids=len(rec['student_input_ids']),KL=loss.item(),label_KL=losses.detach().cpu().tolist(),
                     predicted_ids=winners,disagreements=sum(x!=y for x,y in zip(winners,rec['output_ids'],strict=True)),training_dither=dither)
            if retain:
                path=a.directory/(stage+'.'+rec['id']+'.logits.f32')
                with path.open('xb') as f:
                    f.write(logits.detach().cpu().numpy().astype('<f4',copy=False).tobytes());f.flush();os.fsync(f.fileno())
                row['logits']=dict(path=str(path.resolve()),shape=list(logits.shape),bytes=path.stat().st_size,sha256=sha(path))
            return loss,row
        def values(rows):
            n=sum(r['labels'] for r in rows);errors=sum(r['disagreements'] for r in rows)
            return dict(cases=len(rows),labels=n,case_KL=sum(r['KL'] for r in rows)/len(rows),
                label_KL=sum(sum(r['label_KL']) for r in rows)/n,disagreements=errors,label_disagreement=errors/n,
                case_disagreement=sum(r['disagreements']/r['labels'] for r in rows)/len(rows))
        def evaluate(stage):
            nonlocal dither,phase
            phase=stage;dither=False;target.eval();rows=[]
            with torch.no_grad():
                for rec in stage_records[stage]:
                    guard();t0=time.monotonic();_,row=observe(rec,stage,True);torch.cuda.synchronize()
                    row['seconds']=time.monotonic()-t0;write(a.directory/(stage+'.'+rec['id']+'.json'),row);rows.append(row)
                    event(stage,id=rec['id'],KL=row['KL'],disagreements=row['disagreements'],labels=row['labels'])
            summary=dict(overall=values(rows),domains={d:values([r for r in rows if r['domain']==d]) for d in support[stage]})
            if stage!='retention':summary.update({s:values([r for r in rows if r['split']==s]) for s in ('FIT','DEV')})
            if stage!='retention':summary['DEV_domains']={d:values([r for r in rows if r['domain']==d and r['split']=='DEV']) for d in support[stage]}
            result=dict(cases=rows,summary=summary);write(a.directory/(stage+'.json'),result)
            event(stage+'_summary',summary=summary);return result
        def support_record():return {stage:{d:v.tolist() for d,v in domains.items()} for stage,domains in support.items()}
        def cpu_copy(value):
            if isinstance(value,torch.Tensor):return value.detach().to('cpu',copy=True)
            if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
            if isinstance(value,list):return [cpu_copy(v) for v in value]
            if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
            return value
        def save_boundary(completed):
            snapshot=dict(model=cpu_copy(target.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),config=config,
                target_schema='COMPACT_FALCON_TERNARY_TARGET_V1',candidate_schema='BROAD_24_TRANSFER_CANDIDATE_V1',
                completed_recovery_updates=286,completed_broad_updates=completed,prior_optimizer_step=294,
                torch_CPU_rng=torch.get_rng_state(),torch_CUDA_rng=torch.cuda.get_rng_state_all(),
                freeze=a.freeze,binding_sha256=a.binding_sha,updates=list(updates),order=b['order'],support=support_record(),
                before=before,checkpoint_selection='Fixed final broad24/Adam318;no DEV selection',
                training_contract=dict(normalized_AQ_dither=.025,learning_rate=5e-5,clip_norm=1,storage_helper_sha256=sha(B/'chatbot_target_ssd_storage.py')))
            assert len(snapshot['model'])==211 and len(snapshot['optimizer']['state'])==211
            for v in list(snapshot['model'].values())+[v for slot in snapshot['optimizer']['state'].values() for k,v in slot.items() if k in ('exp_avg','exp_avg_sq')]:
                assert np.isfinite(v.numpy()).all(),'nonfinite snapshot'
            assert all(int(v['step'].item())==294+completed for v in snapshot['optimizer']['state'].values())
            temporary=a.directory/'candidate.next.pt';path=a.directory/'candidate.pt'
            with temporary.open('xb') as f:torch.save(snapshot,f);f.flush();os.fsync(f.fileno())
            os.replace(temporary,path)
            del snapshot;gc.collect();return path
        torch.set_rng_state(initial_cpu_rng);torch.cuda.set_rng_state_all(initial_cuda_rng)
        event('target_ready',initial_optimizer_step=294)
        before=evaluate('before')
        torch.set_rng_state(initial_cpu_rng);torch.cuda.set_rng_state_all(initial_cuda_rng)
        candidate=None
        for step,identifier in enumerate(b['order'],1):
            phase='training_'+str(step);guard();t0=time.monotonic();target.train();dither=True;optimizer.zero_grad(set_to_none=True)
            loss,row=observe(by_id[identifier],'training',False);loss.backward()
            assert all(p.grad is not None and p.grad.shape==p.shape and torch.isfinite(p.grad).all().item() for p in parameters)
            groups=[]
            for index,layer in enumerate(target.layers):
                norms={}
                for label,pars in [('core',list(layer.core.parameters())),('banks',list(layer.banks.parameters())),
                                   ('norm',list(layer.input_norm.parameters())+list(layer.ff_norm.parameters()))]:
                    squared=sum(p.grad.double().square().sum().item() for p in pars);assert 0<squared<float('inf')
                    norms[label]=squared**.5
                groups.append(dict(layer=index,**norms))
            norm=torch.nn.utils.clip_grad_norm_(parameters,b['clip_norm'],error_if_nonfinite=True).item();optimizer.step()
            with torch.no_grad():
                for layer in target.layers:
                    for scale in (layer.banks.gate_scale,layer.banks.up_scale,layer.banks.down_scale):scale.clamp_(min=1e-8)
            dither=False;optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
            update=dict(**row,step=step,optimizer_step=294+step,gradient_norm_before_clip=norm,layer_gradient_norms=groups,
                        seconds_before_snapshot=time.monotonic()-t0)
            updates.append(update);candidate=save_boundary(step);durable=step
            update['seconds_with_snapshot']=time.monotonic()-t0
            write(a.directory/(f'update_{step:02d}.json'),update);write(a.directory/'support.json',support_record())
            event('update',step=step,id=identifier,KL=row['KL'],seconds_with_snapshot=update['seconds_with_snapshot'],durable=durable)
            del loss
        assert durable==24 and candidate is not None
        after=evaluate('after');old_after=evaluate('retention')
        for stage,domains in support.items():
            for domain,matrix in domains.items():
                expected=sum(len(r['student_input_ids'])*8 for r in stage_records[stage] if r['domain']==domain)
                assert (matrix.sum(-1)==expected).all(),(stage,domain,'support count')
        for h in handles:h.remove()
        c=b['criteria'];pre=before['summary'];post=after['summary'];ret=old_after['summary']['overall'];old=b['old_DEV_baseline']
        gates=dict(FIT_relative=post['FIT']['case_KL']<=c['FIT_KL_ratio_max']*pre['FIT']['case_KL'],
            DEV_relative=post['DEV']['case_KL']<=c['DEV_KL_ratio_max']*pre['DEV']['case_KL'],
            DEV_absolute_KL=post['DEV']['case_KL']<=c['DEV_case_KL_max'],
            DEV_absolute_disagreement=post['DEV']['label_disagreement']<=c['DEV_label_disagreement_max'],
            DEV_every_domain_KL=all(v['case_KL']<=c['domain_DEV_case_KL_max'] for v in post['DEV_domains'].values()),
            DEV_every_domain_disagreement=all(v['label_disagreement']<=c['domain_DEV_label_disagreement_max'] for v in post['DEV_domains'].values()),
            old_DEV_KL_retention=ret['case_KL']<=c['retention_ratio_max']*old['case_KL'],
            old_DEV_disagreement_retention=ret['label_disagreement']<=c['retention_ratio_max']*old['label_disagreement'])
        phase='result';write(a.directory/'support.json',support_record());guard()
        decision='BROAD24_PREFIX_TRANSFER_PASS' if all(gates.values()) else 'BROAD24_PREFIX_TRANSFER_FAIL'
        write(a.out,dict(schema='BROAD_CHAT_PILOT_RESULT_V1',decision=decision,freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),new_updates=24,final_optimizer_step=318,
            checkpoint=dict(path=str(candidate.resolve()),bytes=candidate.stat().st_size,sha256=sha(candidate)),
            before=before,after=after,retention=old_after,old_DEV_baseline=old,gates=gates,updates=updates,
            support_counts_verified=True,parameter_tensors=211,model_parameters=254932736,optimizer_moment_bytes=2039461888,
            source_calls=0,native_runs=0,reserved_queries=0,quality_admission=False,native_admission=False,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
            scope='One fixed new24 FIT update pass;new48 before/after andold32 DEV retention,actual support/finite gradients/state/moments. Teacher-prefix metrics,not own-history/chat/tasks/native/accepted50/useful-n/DRAM.'))
        event('complete',decision=decision,gates=gates)
    except BaseException as error:
        failure=dict(fault=repr(error),phase=phase,durable_broad_boundary=durable,attempted_complete_updates=len(updates),elapsed_seconds=time.monotonic()-start)
        if torch is not None:failure.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_failure.json',failure);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
