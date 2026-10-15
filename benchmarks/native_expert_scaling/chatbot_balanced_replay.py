"""Fixed longer joint distillation with balanced old/new FIT replay."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(B), str(ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent, write
from fixed_depth_quality import load, check_inputs

PROTOCOL = DOC / 'CHATBOT_BALANCED_REPLAY_PROTOCOL_20261015.md'
PRIOR = DOC / 'chatbot_broad_pilot_result_repair1_20261009.json'
PRIOR_BINDING = DOC / 'chatbot_broad_pilot_binding_repair1_20261009.json'
RUNTIME = DOC / 'fixed_depth_quality_io_binding_20261014.json'
BINDING = DOC / 'chatbot_balanced_replay_binding_20261015.json'
RESULT = DOC / 'chatbot_balanced_replay_result_20261015.json'
FAMILY = Path('E:/SiliconLLM/results/native_expert_scaling/chatbot_balanced_replay_20261015')
LIMITS = dict(seconds=14400, GPU_seconds=10800, reserve_seconds=180,
              OS_bytes=16 << 30, GPU_allocated_bytes=6 << 30,
              GPU_reserved_bytes=10 << 30, output_bytes=12 << 30, CPU_threads=6)


def training_order(old_records, new_records, new_ids):
    old_fit=sorted((r for r in old_records if r['split']=='FIT'),key=lambda r:r['id'])
    new_fit=[next(r for r in new_records if r['id']==name) for name in new_ids]
    assert len(old_fit)==128 and len(new_fit)==24 and len({r['id'] for r in new_fit})==24
    assert all(r['split']=='FIT' for r in new_fit)
    order=[]
    for epoch in range(2):
        for i,row in enumerate(old_fit):order.extend([row['id'],new_fit[(epoch*128+i)%24]['id']])
    return old_fit,new_fit,order


def prepare(freeze):
    assert not BINDING.exists() and not FAMILY.exists()
    prior, old, runtime = load(PRIOR), load(PRIOR_BINDING), load(RUNTIME)
    closure_path=DOC/'fixed_depth_evaluation6_recovery_closure_20261015.json'
    closure=load(closure_path)
    assert closure['decision']=='STORED_DIAGNOSTIC_QUALIFIED_WITH_EXPLICIT_ORIGINAL_POSTHASH_FAILURE'
    assert closure['current_inputs_outputs_exact'] and not closure['original_posthash_complete']
    assert extent(closure['audit']['path'])==closure['audit'] and load(closure['audit']['path'])['all_numeric_pass']
    prior_terminal=load(PRIOR.with_suffix('.terminal.json'))
    assert prior_terminal['exit_code']==0 and prior_terminal['resource_gates']
    assert prior_terminal['result_sha256']==extent(PRIOR)['sha256']
    assert prior['decision'] == 'BROAD24_PREFIX_TRANSFER_FAIL' and prior['final_optimizer_step'] == 318
    assert extent(prior['checkpoint']['path']) == prior['checkpoint']
    corpora = [load(old[k])['records'] for k in ('old_corpus', 'corpus')]
    old_fit,new_fit,order=training_order(corpora[0],corpora[1],old['order'])
    records = corpora[0] + corpora[1]
    assert len({r['id'] for r in records}) == len(records) == 208
    assert len(order) == 512 and all(next(r for r in records if r['id'] == name)['split'] == 'FIT' for name in order)
    controls = sorted(new_fit, key=lambda r:(len(r['student_input_ids']), r['id']))
    control_ids = [controls[0]['id'], controls[-1]['id']]
    helpers = [Path(__file__), PROTOCOL, B/'chatbot_hybrid_target.py', B/'chatbot_target_ssd_storage.py',
               B/'chatbot_falcon_ssd_tiles.py', PRIOR, PRIOR_BINDING,
               Path(old['old_corpus']), Path(old['corpus']), Path(prior['checkpoint']['path']),
               PRIOR.with_suffix('.terminal.json'),closure_path,Path(closure['audit']['path'])]
    for path in (Path(__file__), PROTOCOL):
        blob = subprocess.check_output(['git','show',freeze+':'+path.relative_to(ROOT).as_posix()], cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n') == path.read_bytes().replace(b'\r\n',b'\n')
    helpers += [Path(r['logits']['path']) for r in records]
    for name in control_ids:
        row = next(r for r in prior['after']['cases'] if r['id'] == name)
        helpers.append(Path(row['logits']['path']))
    pins = runtime['inputs'] + [extent(RUNTIME)] + [extent(p) for p in dict.fromkeys(helpers)]
    check_inputs(dict(inputs=pins))
    write(BINDING, dict(schema='BALANCED_JOINT_REPLAY_V1', freeze=freeze, inputs=pins, runtime=extent(RUNTIME),
        checkpoint=prior['checkpoint'], prior=extent(PRIOR), corpora=[old['old_corpus'],old['corpus']],
        order=order, control_ids=control_ids, limits=LIMITS, initial_step=318, final_step=830,
        updates=512, checkpoint_interval=16, learning_rate=5e-5, AQ_dither=.025, clip_norm=1.,
        family=str(FAMILY), result=str(RESULT), worker=str(Path(__file__).resolve()),
        scope='Longer balanced known-FIT joint distillation; no fresh heldout, native/rate/DRAM/useful-n admission.'))
    print(json.dumps(dict(binding=extent(BINDING),updates=512,initial_step=318,final_step=830)),flush=True)


def worker(binding_sha):
    import numpy as np
    import psutil
    import torch
    import transformers
    from torch.utils.checkpoint import checkpoint
    from transformers.models.falcon_h1 import modeling_falcon_h1 as source_code
    import chatbot_hybrid_target as native
    import chatbot_target_ssd_storage as storage
    assert extent(BINDING)['sha256'] == binding_sha
    b = load(BINDING)
    assert b['limits'] == LIMITS and b['updates'] == 512 and not FAMILY.exists() and not RESULT.exists()
    assert (np.__version__,torch.__version__,transformers.__version__) == ('2.4.6','2.6.0+cu124','5.13.1')
    proc = psutil.Process();proc.cpu_affinity(list(range(6)))
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();gpu_start=None;gpu_end=None;last_descendant=0.;last_family_scan=0.;durable=0;stage='input_hashes'
    receipt=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    def guard():
        nonlocal last_descendant,last_family_scan
        now=time.monotonic()
        assert now-start <= LIMITS['seconds']-LIMITS['reserve_seconds'],'total deadline'
        assert proc.memory_info().peak_wset <= LIMITS['OS_bytes'],'OS cap'
        if now-last_descendant>=1:
            assert not proc.children(recursive=True),'unexpected child';last_descendant=now
        if gpu_start is not None and gpu_end is None:
            assert now-gpu_start <= LIMITS['GPU_seconds'],'GPU deadline'
            assert torch.cuda.max_memory_allocated()<=LIMITS['GPU_allocated_bytes'],'GPU allocation cap'
            assert torch.cuda.max_memory_reserved()<=LIMITS['GPU_reserved_bytes'],'GPU reservation cap'
        if FAMILY.exists() and now-last_family_scan>=1:
            assert sum(p.stat().st_size for p in FAMILY.iterdir() if p.is_file())<=LIMITS['output_bytes'],'output cap'
            last_family_scan=now
    def event(**fields):
        guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**fields)),flush=True)
    try:
        check_inputs(b,guard)
        assert not torch.cuda.is_initialized()
        gpu=subprocess.check_output(['C:/Windows/System32/nvidia-smi.exe','--query-gpu=uuid,driver_version,memory.free',
                                     '--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
        expected=load(RUNTIME)['GPU_query'].split(',')
        assert gpu[0].strip()==expected[0].strip() and gpu[1].strip()==expected[2].strip() and int(gpu[-1])>=7168
        FAMILY.mkdir();stage='restore';gpu_start=time.monotonic()
        generated=storage.install(source_code)
        (FAMILY/'generated_method.py.txt').write_text(generated,encoding='utf8',newline='\n')
        dither=False;original_aq=native.aq63
        def aq(x):
            if not dither or not torch.is_grad_enabled():return original_aq(x)
            scale=x.detach().abs().amax(-1,keepdim=True).clamp_min(1e-12)/63
            noise=torch.empty_like(x).uniform_(-.025,.025)
            return torch.round(x.detach()*(1/scale)+noise).clamp(-63,63),scale
        native.aq63=aq
        class Target(native.Target):
            def forward(self,ids,positions):
                x=self.embed(ids)
                for block in self.layers:
                    x=checkpoint(block,x,use_reentrant=False,preserve_rng_state=True) if self.training and torch.is_grad_enabled() else block(x)
                return self.head(self.final_norm(x)[:,positions])
        old=torch.load(b['checkpoint']['path'],map_location='cpu',weights_only=True)
        assert old['candidate_schema']=='BROAD_24_TRANSFER_CANDIDATE_V1' and old['completed_broad_updates']==24
        assert old['completed_recovery_updates']==286 and len(old['model'])==211
        assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in old['model'].values())
        assert len(old['optimizer']['state'])==211
        for restored_slot in old['optimizer']['state'].values():
            assert int(restored_slot['step'].item())==318
            assert all(torch.isfinite(restored_slot[k]).all() for k in ('exp_avg','exp_avg_sq'))
        target=Target(old['config']).to('cuda');target.load_state_dict(old['model'],strict=True)
        parameters=list(target.parameters());assert len(parameters)==211 and sum(p.numel() for p in parameters)==254932736
        assert all(p.dtype==torch.float32 for p in parameters)
        optimizer=torch.optim.AdamW(parameters,lr=5e-5,weight_decay=0,foreach=False)
        optimizer.load_state_dict(old['optimizer'])
        assert len(optimizer.state)==211 and all(int(s['step'].item())==318 for s in optimizer.state.values())
        for group in optimizer.param_groups:
            assert tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and not group['amsgrad'] and not group['maximize']
            group.update(lr=5e-5,weight_decay=0,foreach=False)
        config=old['config'];cpu_rng=old['torch_CPU_rng'];gpu_rng=old['torch_CUDA_rng']
        del old;gc.collect()
        records=sum([load(path)['records'] for path in b['corpora']],[]);by_id={r['id']:r for r in records}
        assert len(by_id)==208 and all(by_id[name]['split']=='FIT' for name in b['order'])
        counts=np.zeros((12,72),dtype='i8');collect=False;mass_max=0.
        def route_hook(site):
            def hook(module,args):
                nonlocal mass_max
                if not collect:return
                with torch.no_grad():
                    scores=module.router(args[0].reshape(-1,512));ids=torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:8]
                    mass=torch.softmax(scores.gather(1,ids),-1)
                    assert torch.isfinite(mass).all() and (mass>=0).all()
                    mass_max=max(mass_max,float((mass.sum(-1)-1).abs().max()))
                    assert mass_max<=1e-6
                    counts[site]+=torch.bincount(ids.flatten(),minlength=72).cpu().numpy()
            return hook
        handles=[layer.banks.register_forward_pre_hook(route_hook(i)) for i,layer in enumerate(target.layers)]
        def observe(rec,retain=None):
            nonlocal collect
            assert rec['student_input_ids']==rec['input_ids']+rec['output_ids'][:-1]
            bits=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4')
            teacher=torch.from_numpy((bits<<16).view('<f4').reshape(rec['logits']['shape'])).to('cuda')
            logq=torch.log_softmax(teacher,-1).detach();prob=logq.exp();collect=target.training
            try:
                logits=target(torch.tensor([rec['student_input_ids']],device='cuda'),torch.tensor(rec['positions'],device='cuda')).squeeze(0)
            finally:collect=False
            losses=(prob*(logq-torch.log_softmax(logits,-1))).sum(-1);loss=losses.mean()
            assert torch.isfinite(logits).all() and torch.isfinite(loss)
            ids=logits.detach().argmax(-1).cpu().tolist()
            row=dict(id=rec['id'],domain=rec['domain'],split=rec['split'],labels=len(ids),KL=loss.item(),
                     KL_per_label=losses.detach().cpu().tolist(),predicted_ids=ids,
                     disagreements=sum(x!=y for x,y in zip(ids,rec['output_ids'],strict=True)))
            if retain is not None:
                path=FAMILY/(retain+'.'+rec['id']+'.f32');logits.detach().cpu().numpy().astype('<f4').tofile(path)
                row['scores']=dict(**extent(path),shape=list(logits.shape))
            return loss,row
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
        stage='current_runtime_control';target.eval();prior=load(PRIOR);controls=[]
        with torch.no_grad():
            for name in b['control_ids']:
                _,row=observe(by_id[name],'control');reference=next(r for r in prior['after']['cases'] if r['id']==name)
                shape=reference['logits']['shape']
                actual=np.memmap(row['scores']['path'],'f4','r',shape=tuple(shape))
                expected=np.memmap(reference['logits']['path'],'f4','r',shape=tuple(shape))
                error=float(np.linalg.norm(actual.astype('f8')-expected.astype('f8'))/max(np.linalg.norm(expected.astype('f8')),1e-300))
                assert error<=1e-4 and row['predicted_ids']==reference['predicted_ids'],'current runtime control'
                controls.append(dict(id=name,relative=error,scores=row['scores']))
                event(operation='control',id=name,relative=error)
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
        updates=[];trained_tokens=0
        def cpu_copy(value):
            if isinstance(value,torch.Tensor):return value.detach().to('cpu',copy=True)
            if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
            if isinstance(value,list):return [cpu_copy(v) for v in value]
            if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
            return value
        def save(step):
            snapshot=dict(model=cpu_copy(target.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),config=config,
                target_schema='COMPACT_FALCON_TERNARY_TARGET_V1',candidate_schema='BALANCED_512_TRANSFER_CANDIDATE_V1',
                completed_balanced_updates=step,initial_optimizer_step=318,optimizer_step=318+step,
                torch_CPU_rng=torch.get_rng_state(),torch_CUDA_rng=torch.cuda.get_rng_state_all(),
                binding=extent(BINDING),order=b['order'],route_counts=counts.tolist(),updates=updates)
            for value in list(snapshot['model'].values())+[v for slot in snapshot['optimizer']['state'].values() for k,v in slot.items() if k in ('exp_avg','exp_avg_sq')]:
                assert np.isfinite(value.numpy()).all(),'nonfinite snapshot'
            assert all(int(s['step'].item())==318+step for s in snapshot['optimizer']['state'].values())
            temporary=FAMILY/'candidate.next.pt';assert not temporary.exists()
            with temporary.open('xb') as stream:torch.save(snapshot,stream);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,FAMILY/'candidate.pt');del snapshot;gc.collect()
        stage='training'
        for step,name in enumerate(b['order'],1):
            guard();begin=time.monotonic();target.train();dither=True;optimizer.zero_grad(set_to_none=True)
            loss,row=observe(by_id[name]);loss.backward()
            assert all(p.grad is not None and p.grad.shape==p.shape and torch.isfinite(p.grad).all() for p in parameters)
            groups=[]
            for i,layer in enumerate(target.layers):
                item=dict(site=i)
                for label,ps in [('core',list(layer.core.parameters())),('banks',list(layer.banks.parameters())),
                                  ('norm',list(layer.input_norm.parameters())+list(layer.ff_norm.parameters()))]:
                    squared=sum(p.grad.double().square().sum().item() for p in ps)
                    assert 0<squared<float('inf');item[label]=squared**.5
                groups.append(item)
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True).item();optimizer.step()
            with torch.no_grad():
                for layer in target.layers:
                    for scale in (layer.banks.gate_scale,layer.banks.up_scale,layer.banks.down_scale):scale.clamp_(min=1e-8)
            dither=False;optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
            trained_tokens+=len(by_id[name]['student_input_ids'])
            assert np.all(counts.sum(1)==trained_tokens*8),'route count or checkpoint double collection'
            update=dict(step=step,id=name,optimizer_step=318+step,KL=row['KL'],gradient_norm=norm,groups=groups,
                        compute_seconds=time.monotonic()-begin,trained_tokens=trained_tokens,mass_max=mass_max)
            updates.append(update);write(FAMILY/(f'update_{step:04d}.json'),update)
            if step%16==0:save(step);durable=step
            event(operation='update',step=step,id=name,durable=durable,compute_seconds=update['compute_seconds'],KL=row['KL'])
            del loss
        assert durable==512 and all(int(s['step'].item())==830 for s in optimizer.state.values())
        stage='fixed_final_evaluation';target.eval();evaluation=[]
        # Fixed final model only; DEV never participates in updates or checkpoint choice.
        wanted=[r for r in records if r['split']=='DEV' or (r['split']=='FIT' and r['id'] in set(load(PRIOR_BINDING)['order']))]
        assert len(wanted)==80
        with torch.no_grad():
            for rec in wanted:
                _,row=observe(rec,'final');evaluation.append(row);write(FAMILY/(rec['id']+'.metrics.json'),row)
                event(operation='evaluated',id=rec['id'])
        candidate=extent(FAMILY/'candidate.pt')
        # Complete saved model/Adam/RNG roundtrip, without a second forward or fit.
        saved=torch.load(candidate['path'],map_location='cpu',weights_only=True)
        assert saved['completed_balanced_updates']==512 and saved['optimizer_step']==830
        for name,tensor in target.state_dict().items():assert torch.equal(saved['model'][name],tensor.detach().cpu())
        current=optimizer.state_dict()
        for index,slot in current['state'].items():
            for key,tensor in slot.items():assert torch.equal(saved['optimizer']['state'][index][key],tensor.detach().cpu())
        assert torch.equal(saved['torch_CPU_rng'],torch.get_rng_state())
        assert all(torch.equal(x,y) for x,y in zip(saved['torch_CUDA_rng'],torch.cuda.get_rng_state_all(),strict=True))
        for handle in handles:handle.remove()
        # Loop-local parameter groups also own GPU tensors; release those owners.
        del saved,current,optimizer,target,parameters,group,ps,layer,tensor,slot
        gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        gc.collect();torch.cuda.empty_cache()
        gpu_end=time.monotonic();assert torch.cuda.memory_allocated()<=64<<20,'GPU ownership release'
        stage='posthash';check_inputs(b,guard);assert extent(candidate['path'])==candidate
        rows=[]
        old_ids={r['id'] for r in load(b['corpora'][0])['records']}
        for cohort in ('old_DEV','new_DEV','new_FIT'):
            selected=[r for r in evaluation if (r['split']=='FIT')==(cohort=='new_FIT') and ((r['id'] in old_ids)==(cohort=='old_DEV'))]
            assert len(selected)==(32 if cohort=='old_DEV' else 24)
            domains={}
            for domain in sorted({r['domain'] for r in selected}):
                subset=[r for r in selected if r['domain']==domain]
                domains[domain]=dict(KL=sum(r['KL'] for r in subset)/len(subset),
                    disagreement=sum(r['disagreements'] for r in subset)/sum(r['labels'] for r in subset))
            summary=dict(cohort=cohort,KL=sum(r['KL'] for r in selected)/len(selected),
                disagreement=sum(r['disagreements'] for r in selected)/sum(r['labels'] for r in selected),domains=domains)
            summary['absolute_pass']=summary['KL']<=1 and summary['disagreement']<=.2 and all(v['KL']<=2 and v['disagreement']<=.35 for v in domains.values())
            rows.append(summary)
        outputs=[extent(p) for p in sorted(FAMILY.iterdir()) if p.is_file()]
        assert sum(e['bytes'] for e in outputs)<=LIMITS['output_bytes']
        write(RESULT,dict(schema='BALANCED_JOINT_REPLAY_RESULT_V1',binding=extent(BINDING),checkpoint=candidate,
            updates=512,initial_step=318,final_step=830,update_records=updates,current_runtime_controls=controls,
            summaries=rows,evaluation=evaluation,route_counts=counts.tolist(),mass_max=mass_max,
            saved_state_roundtrip_exact=True,inputs_before_after_exact=True,GPU_released_before_posthash=True,
            GPU_phase_seconds=gpu_end-gpu_start,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            outputs=outputs,RESERVED_queries=0,DEV_training_queries=0,quality_admissible=False,
            decision='KNOWN_PREFIX_GATES_PASS' if all(r['absolute_pass'] for r in rows) else 'KNOWN_PREFIX_GATES_FAIL',
            scope=b['scope']+' Actual native/own-generation/same-artifact rate/functional bank utility and fresh heldout remain required.'))
        receipt.update(exit_code=0,error=None,result=extent(RESULT))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage,durable_updates=durable)
        raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(RESULT.with_suffix('.worker_terminal.json'),receipt)


def launch(binding_sha):
    import psutil
    from original_falcon_whole_recovery import memory_reader
    assert extent(BINDING)['sha256']==binding_sha and not RESULT.exists() and not FAMILY.exists()
    b=load(BINDING);check_inputs(b)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0')
    argv=[sys.executable,'-I','-S','-B','-X','utf8',str(Path(__file__).resolve()),'worker','--binding-sha',binding_sha]
    proc=psutil.Process();proc.cpu_affinity([11]);reader=memory_reader();start=time.monotonic();child=None;peak=0;offset=0
    receipt=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),worker_argv=argv,binding=extent(BINDING))
    try:
        with RESULT.with_suffix('.worker.log').open('xb') as stream:
            child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time())
            write(RESULT.with_suffix('.owned_process.json'),receipt)
            while child.poll() is None:
                peak=max(peak,reader(child))
                assert time.monotonic()-start<=LIMITS['seconds'],'holder deadline'
                assert peak+proc.memory_info().peak_wset<=LIMITS['OS_bytes'],'combined OS cap'
                with RESULT.with_suffix('.worker.log').open('rb') as log:
                    log.seek(offset);data=log.read();end=data.rfind(b'\n')+1
                    if end:print(data[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.5)
        receipt['worker_exit_code']=child.returncode;assert child.returncode==0
        check_inputs(b);receipt.update(exit_code=0,error=None,result=extent(RESULT))
    except BaseException as error:
        if child is not None and child.poll() is None:
            assert abs(psutil.Process(child.pid).create_time()-receipt['worker_creation_time'])<.001
            child.terminate();child.wait(timeout=15)
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),worker_exit_code=None if child is None else child.returncode)
        raise
    finally:
        receipt.update(seconds=time.monotonic()-start,worker_OS_peak=peak,OS_peak=proc.memory_info().peak_wset)
        write(RESULT.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','launch','worker'])
    parser.add_argument('--freeze');parser.add_argument('--binding-sha');args=parser.parse_args()
    if args.mode=='prepare':
        import psutil
        proc=psutil.Process();start=time.monotonic();receipt=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
        try:prepare(args.freeze);receipt.update(exit_code=0,error=None,result=extent(BINDING))
        except BaseException as error:receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
        finally:
            receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
            write(BINDING.with_suffix('.preparation_terminal.json'),receipt)
    elif args.mode=='launch':launch(args.binding_sha)
    else:worker(args.binding_sha)
