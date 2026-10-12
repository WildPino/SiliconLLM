"""Fully trained24-layer source operators in restricted residual coordinates.

Floating shadow diagnostic, not original packed/LUT/selective deployment.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent
from chatbot_falcon_usability import write,normalized
START=time.monotonic()
LIMITS=dict(seconds=1200,reserve_seconds=120,OS_bytes=8<<30,GPU_allocated_bytes=6<<30,
            GPU_reserved_bytes=8<<30,output_bytes=3<<30)


def target_shape(key,shape):
    shape=list(shape)
    if key in ('model.embed_tokens.weight','lm_head.weight'):shape[1]=256
    elif key=='model.final_layernorm.weight' or key.endswith(('input_layernorm.weight','pre_ff_layernorm.weight')):shape=[256]
    elif key.endswith(('mamba.in_proj.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
                       'feed_forward.gate_proj.weight','feed_forward.up_proj.weight')):shape[1]=256
    elif key.endswith(('mamba.out_proj.weight','self_attn.o_proj.weight','feed_forward.down_proj.weight')):shape[0]=256
    return shape


def prepare(a):
    import numpy as np,psutil,torch,transformers
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    from original_engine_chat import ChatTokenizer
    from original_falcon_whole_recovery import memory_reader
    parent=DOC/'falcon_projected_s6_transport_result_20261012.json'
    pr=json.loads(parent.read_bytes());assert pr['decision']=='RESTRICTED_D256_OPERATOR_AND_F32_COEFFICIENTS_QUALIFIED'
    audit=DOC/'falcon_projected_s6_stored_adjudication_20261012.json'
    au=json.loads(audit.read_bytes());assert au['report']==extent(parent) and au['all_exported_parameter_fields_checked']
    native=DOC/'recovered_categorical_native_chat_binding_20261012.json'
    nb=json.loads(native.read_bytes());source=Path(nb['source'])
    baseline=json.loads((DOC/'original_engine_baseline_binding_20261009.json').read_bytes())
    donor=baseline['source_result'];assert extent(donor['path'])==donor
    assert json.loads(Path(donor['path']).read_bytes())['correct']==14
    corpus=ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    dev=[r for r in json.loads(corpus.read_bytes())['records'] if r['split']=='DEV']
    assert len(dev)==24 and sum(len(r['positions']) for r in dev)==4386
    tokenizer=ChatTokenizer(source)
    assert all(tokenizer.encode(c['messages'])==c['input_ids'] for c in nb['cases'])
    with (source/'model.safetensors').open('rb') as f:
        n=int.from_bytes(f.read(8),'little');header=json.loads(f.read(n))
    fields=[dict(name=k,source_shape=v['shape'],target_shape=target_shape(k,v['shape']))
            for k,v in header.items() if k!='__metadata__']
    assert len(fields)==411 and sum(math.prod(x['source_shape']) for x in fields)==1554863488
    assert sum(math.prod(x['target_shape']) for x in fields)==194801792
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',B/'chatbot_falcon_usability.py',
        B/'chatbot_falcon_ssd_tiles.py',B/'original_engine_chat.py',B/'chatbot_hybrid_engine_chat.py',
        parent,audit,native,corpus,Path(pr['basis']['path']),Path(donor['path']),
        DOC/'original_engine_baseline_binding_20261009.json',B/'original_falcon_whole_recovery.py',
        DOC/'FALCON_PRETRAINED_PROJECTED_SHADOW_PROTOCOL_20261012.md',
        DOC/'FALCON_PRETRAINED_PROJECTED_SHADOW_V2_STARTUP_PROTOCOL_20261012.md',
        DOC/'FALCON_PRETRAINED_PROJECTED_SHADOW_V3_RUNTIME_PROTOCOL_20261012.md',
        ROOT/'benchmarks/phase60/engine.c',source/'model.safetensors',source/'config.json']
    paths += [source/n for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json')]
    paths += [Path(r['logits']['path']) for r in dev]
    paths += [Path(item['path']) for item in pr['outputs'] if item['path'].endswith('.frames.npz')]
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if isinstance(p,str) and Path(p).is_file():paths.append(Path(p))
    paths += [Path(sys.executable),Path(sys.executable).parent/'python312.dll']
    paths += sorted((SITE/'torch/lib').glob('*.dll'))+sorted((SITE/'numpy.libs').glob('*.dll'))
    inputs=[extent(path) for path in dict.fromkeys(p.resolve() for p in paths)]
    b=dict(schema='FALCON_PRETRAINED_PROJECTED_SHADOW_BINDING_V3',python=str(Path(sys.executable).resolve()),
        source=str(source.resolve()),basis=pr['basis'],operator_control=extent(parent),
        operator_audit=extent(audit),fields=fields,cases=nb['cases'],followup=nb['followup'],
        special_ids=nb['special_ids'],DEV=dev,eos_ids=[11,228],limits=LIMITS,inputs=inputs,
        gates=dict(local_operator_relative_RMS=1e-4,correct_min=12,every_category_min=2,
                   blank_max=1,DEV_case_KL=1.,DEV_case_disagreement=.20,domain_case_KL=2.,domain_case_disagreement=.35),
        inherited_quality_gates=nb['gates'],donor_behavior=donor,
        scope='Fixed P256/all24 trained source layers; no optimizer/source model/new teacher/native/T4. '
            'Principal loaded runtime+Torch DLL binding, not full dependency tree; host timing uncontrolled.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),inputs=len(inputs),seconds=time.monotonic()-START)),flush=True)


def assemble(b,torch,np,guard,event):
    from transformers import FalconH1Config,FalconH1ForCausalLM
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    from safetensors import safe_open
    from chatbot_falcon_ssd_tiles import install
    cfg=json.loads((Path(b['source'])/'config.json').read_bytes())
    assert cfg['hidden_size']==2048 and cfg['num_hidden_layers']==24 and not cfg['tie_word_embeddings']
    assert not cfg['mamba_proj_bias'] and not cfg['projectors_bias'] and not cfg['attention_bias'] and not cfg['mlp_bias']
    target_cfg={**cfg,'hidden_size':256,'torch_dtype':'float32'}
    config=FalconH1Config(**target_cfg);config._attn_implementation='eager'
    torch.manual_seed(0)
    model=FalconH1ForCausalLM(config).to('cuda').eval().requires_grad_(False)
    assert not code.is_fast_path_available
    p64=np.fromfile(b['basis']['path'],'<f4').reshape(2048,256).astype('f8')
    p=torch.from_numpy(p64.astype('f4')).cuda()
    metric=torch.from_numpy((p64.T@p64).astype('f4')).cuda()
    model.register_buffer('projected_metric',metric,persistent=True)
    class MetricNorm(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(256,device='cuda'),requires_grad=False)
            self.register_buffer('metric',metric,persistent=False);self.variance_epsilon=8*cfg['rms_norm_eps']
        def forward(self,x):
            xf=x.float();den=torch.sqrt(((xf@self.metric)*xf).sum(-1,keepdim=True)/256+self.variance_epsilon)
            return (xf/den*self.weight).to(x.dtype)
    for block in model.model.layers:
        block.input_layernorm=MetricNorm();block.pre_ff_layernorm=MetricNorm()
        block.mamba.norm.variance_epsilon=cfg['rms_norm_eps']
    model.model.final_layernorm=MetricNorm()
    params=dict(model.named_parameters());assert set(params)=={row['name'] for row in b['fields']}
    assert sum(x.numel() for x in params.values())==194801792
    patch=install(code);used=set();ledger=[]
    with safe_open(Path(b['source'])/'model.safetensors',framework='pt',device='cpu') as sf,torch.no_grad():
        def get(key):used.add(key);return sf.get_tensor(key).float().cuda()
        for row in b['fields']:
            guard();key=row['name'];dst=params[key]
            assert list(dst.shape)==row['target_shape'],(key,dst.shape,row['target_shape'])
            if key in ('model.embed_tokens.weight','lm_head.weight'):
                source=sf.get_slice(key);used.add(key)
                gamma=get('model.final_layernorm.weight') if key=='lm_head.weight' else None
                for lo in range(0,65537,4096):
                    guard();value=source[lo:min(lo+4096,65537)].float().cuda()
                    if gamma is not None:value=value*gamma
                    result=value@p
                    if gamma is not None:result=result*math.sqrt(8)
                    dst[lo:lo+len(result)].copy_(result);del value,result
                kind='head_final_gamma_fold' if gamma is not None else 'embedding_projection'
            elif key=='model.final_layernorm.weight' or key.endswith(('input_layernorm.weight','pre_ff_layernorm.weight')):
                used.add(key);dst.fill_(1);kind='gamma_folded_into_downstream_input_maps'
            elif key.endswith(('mamba.in_proj.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
                               'feed_forward.gate_proj.weight','feed_forward.up_proj.weight')):
                prefix=key.split('.mamba.')[0] if '.mamba.' in key else key.split('.self_attn.')[0] if '.self_attn.' in key else key.split('.feed_forward.')[0]
                norm='pre_ff_layernorm' if '.feed_forward.' in key else 'input_layernorm'
                gamma=get(prefix+'.'+norm+'.weight');value=get(key)
                dst.copy_(((value*gamma)@p)*math.sqrt(8));kind='input_gamma_fold'
                del value,gamma
            elif key.endswith(('mamba.out_proj.weight','self_attn.o_proj.weight','feed_forward.down_proj.weight')):
                value=get(key);dst.copy_(p.T@value);kind='output_projection';del value
            else:
                value=get(key);assert dst.shape==value.shape;dst.copy_(value);kind='source_copy';del value
            assert torch.isfinite(dst).all();ledger.append(dict(**row,operation=kind))
            if key.endswith('mamba.out_proj.weight'):event(operation='trained_layer_export',layer=key.split('.')[2])
    assert used=={row['name'] for row in b['fields']}
    return model,ledger,patch


def local_controls(model,b,directory,torch,np,guard):
    r=json.loads(Path(b['operator_control']['path']).read_bytes());rows=[]
    for item in r['outputs']:
        if not item['path'].endswith('.frames.npz'):continue
        guard();site=int(Path(item['path']).name[4:6]);m=model.model.layers[site].mamba;values={}
        with np.load(item['path'],allow_pickle=False) as packet:
            z=torch.from_numpy(packet['z'].astype('f4')).cuda()[None]
            norm=model.model.layers[site].input_layernorm(z)
            def proj(module,args,out):
                v=out*m.mup_vector;g,_,dt=v.split((3072,3584,48),-1)
                values['gate']=g;values['delta']=torch.nn.functional.softplus(dt+m.dt_bias)
            def conv(module,args,out):
                v=torch.nn.functional.silu(out[...,:13]).transpose(1,2)
                values['x'],values['B'],values['C']=v.split((3072,256,256),-1)
            def before_norm(module,args):values['y']=args[0]
            hooks=[m.in_proj.register_forward_hook(proj),m.conv1d.register_forward_hook(conv),
                   m.norm.register_forward_pre_hook(before_norm)]
            try:values['output']=m(norm)
            finally:
                for hook in hooks:hook.remove()
            gated=values['y']*torch.nn.functional.silu(values['gate'])
            values['denominator']=torch.sqrt(gated.square().mean(-1,keepdim=True)+m.norm.variance_epsilon)
            arrays={key:value[0].detach().cpu().numpy().astype('f4') for key,value in values.items()}
            errors={key:float(np.linalg.norm(value.astype('f8')-packet['source_'+key])/
                             max(float(np.linalg.norm(packet['source_'+key])),1e-300)) for key,value in arrays.items()}
            np.savez(directory/f'local.site{site:02d}.npz',**arrays)
            rows.append(dict(site=site,relative_RMS=errors,numeric_gate=all(v<=b['gates']['local_operator_relative_RMS'] for v in errors.values())))
    assert len(rows)==3;return rows


def categorical(rec,values,np):
    v=65537;n=len(rec['positions']);teacher=np.memmap(rec['logits']['path'],'<u2','r',shape=(n,v))
    losses=[];wrong=0
    for i in range(n):
        q=(teacher[i].astype('<u4')<<16).view('<f4').astype('f8');q-=q.max();q-=math.log(float(np.exp(q).sum()))
        p=values[i].astype('f8');p-=p.max();p-=math.log(float(np.exp(p).sum()))
        losses.append(float(np.sum(np.exp(q)*(q-p))));wrong+=int(np.argmax(p)!=np.argmax(q))
    return dict(id=rec['id'],domain=rec['domain'],labels=n,KL=float(np.mean(losses)),KL_per_label=losses,
                disagreement=wrong,disagreement_rate=wrong/n)


def worker(a):
    assert extent(a.binding)['sha256']==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='FALCON_PRETRAINED_PROJECTED_SHADOW_BINDING_V3' and b['limits']==LIMITS
    assert not a.directory.exists() and not a.out.exists() and not a.out.with_suffix('.terminal.json').exists()
    import numpy as np,psutil,torch,transformers
    from original_engine_chat import ChatTokenizer
    assert (torch.__version__,np.__version__,psutil.__version__,transformers.__version__)==('2.6.0+cu124','2.4.6','7.2.2','5.13.1')
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='before_hash';model=None;tasks=[];dev=[]
    receipt=dict(schema='FALCON_PRETRAINED_PROJECTED_SHADOW_TERMINAL_V3',binding=extent(a.binding),
        command=sys.orig_argv,freeze=a.freeze,pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=LIMITS['seconds']-LIMITS['reserve_seconds'],'worker reserve'
        assert proc.memory_info().peak_wset<=LIMITS['OS_bytes'],'OS cap'
        assert torch.cuda.max_memory_allocated()<=LIMITS['GPU_allocated_bytes'],'GPU allocated cap'
        assert torch.cuda.max_memory_reserved()<=LIMITS['GPU_reserved_bytes'],'GPU reserved cap'
        assert not proc.children(recursive=True),'unexpected child'
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=LIMITS['output_bytes']
    def check():
        for item in b['inputs']:guard();assert extent(item['path'])==item,item['path']
    def event(**fields):guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-START,**fields)),flush=True)
    try:
        check();a.directory.mkdir();stage='trained_export'
        model,ledger,patch=assemble(b,torch,np,guard,event)
        assert all(p.dtype==torch.float32 for p in model.parameters())
        identity={name:(id(p),p._version) for name,p in model.named_parameters()}
        write(a.directory/'construction.json',dict(parameters=ledger,source_elements=1554863488,target_elements=194801792,
            config=model.config.to_dict(),input_metric_epsilon=8e-5,postgate_epsilon=1e-5,source_ssd_patch=patch))
        state={key:value.detach().cpu().contiguous() for key,value in model.state_dict().items()}
        torch.save(state,a.directory/'model.f32.pt');del state;guard()
        stage='local_F32_runtime_controls'
        with torch.inference_mode():controls=local_controls(model,b,a.directory,torch,np,guard)
        write(a.directory/'local_controls.json',controls);event(operation='local_controls',all_pass=all(r['numeric_gate'] for r in controls))
        tokenizer=ChatTokenizer(b['source']);followup=None
        def generate(messages,label):
            guard();ids=tokenizer.encode(messages);inputs=torch.tensor([ids],device='cuda',dtype=torch.long)
            with torch.inference_mode():
                out=model.generate(inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=64,do_sample=False,
                    use_cache=True,eos_token_id=b['eos_ids'],pad_token_id=0,return_dict_in_generate=True,output_scores=True)
            new=out.sequences[0,len(ids):].tolist();scores=torch.stack(out.scores).squeeze(1).float().cpu().numpy()
            assert scores.shape==(len(new),65537) and np.isfinite(scores).all() and scores.argmax(-1).tolist()==new
            path=a.directory/(label+'.scores.f32');scores.astype('<f4').tofile(path)
            text=tokenizer.decode(new);row=dict(label=label,messages=messages,input_ids=ids,output_ids=new,output_text=text,
                normalized=normalized(text),scores=extent(path),state_reset=True,
                special_leak=any(i in set(b['special_ids'])-set(b['eos_ids']) for i in new),
                stop_policy=bool(new) and not any(i in b['eos_ids'] for i in new[:-1]) and (new[-1] in b['eos_ids'] or len(new)==64))
            del out,inputs,scores;torch.cuda.empty_cache();guard();return row
        stage='canonical_tasks'
        for case in b['cases']:
            row=generate(case['messages'],case['id']);assert row['input_ids']==case['input_ids']
            row.update(id=case['id'],category=case['category'],expected=case['expected'],correct=row['normalized']==case['expected'])
            tasks.append(row);write(a.directory/(case['id']+'.task.json'),row)
            event(operation='task',id=case['id'],correct=row['correct'],answer=row['output_text'])
            if case['id']==b['followup']['case_id']:
                messages=case['messages']+[dict(role='assistant',content=row['output_text']),dict(role='user',content=b['followup']['prompt'])]
                followup=generate(messages,'own_answer_followup');followup.update(expected=b['followup']['expected'],correct=followup['normalized']==b['followup']['expected'])
                write(a.directory/'own_answer_followup.json',followup)
        stage='all24_DEV'
        for rec in b['DEV']:
            guard();ids=torch.tensor([rec['student_input_ids']],device='cuda',dtype=torch.long)
            positions=torch.tensor(rec['positions'],device='cuda',dtype=torch.long)
            trajectory={}
            def initial(module,args):trajectory[0]=args[0].detach().clone()
            handles=[model.model.layers[0].register_forward_pre_hook(initial)]
            for boundary in (4,8,12,16,20,24):
                def capture(module,args,out,index=boundary):trajectory[index]=out[0].detach().clone()
                handles.append(model.model.layers[boundary-1].register_forward_hook(capture))
            try:
                with torch.inference_mode():
                    h=model.model(ids,use_cache=False).last_hidden_state[:,positions]
                    values=(model.lm_head(h)*model.model.lm_head_multiplier)[0].float().cpu().numpy()
            finally:
                for handle in handles:handle.remove()
            trace=[]
            for boundary in (0,4,8,12,16,20,24):
                array=trajectory[boundary][0].cpu().numpy().astype('<f4')
                assert array.shape==(len(rec['student_input_ids']),256) and np.isfinite(array).all()
                trace_path=a.directory/(rec['id']+f'.h{boundary:02d}.f32');array.tofile(trace_path)
                trace.append(dict(boundary=boundary,shape=list(array.shape),**extent(trace_path)))
            del trajectory,array
            assert values.shape==(len(rec['positions']),65537) and np.isfinite(values).all()
            path=a.directory/(rec['id']+'.labels.f32');values.astype('<f4').tofile(path)
            row=categorical(rec,values,np);row['scores']=extent(path);row['trajectory']=trace;dev.append(row)
            write(a.directory/(rec['id']+'.DEV.json'),row);event(operation='DEV',id=rec['id'],KL=row['KL'])
            del ids,positions,h,values;torch.cuda.empty_cache()
        assert {name:(id(p),p._version) for name,p in model.named_parameters()}==identity
        assert len(tasks)==16 and followup is not None and len(dev)==24
        stage='after_hash';check();guard()
        outputs=[extent(path) for path in sorted(a.directory.iterdir()) if path.is_file()]
        categories={cat:sum(r['correct'] for r in tasks if r['category']==cat) for cat in sorted({r['category'] for r in tasks})}
        quality=dict(correct_min=sum(categories.values())>=12,every_category_min=all(v>=2 for v in categories.values()),
            blank_max=sum(not r['output_text'].strip() for r in tasks)<=1,no_special_leak=not any(r['special_leak'] for r in tasks),
            stop_policy=all(r['stop_policy'] for r in tasks))
        domains={d:dict(case_KL=sum(r['KL'] for r in dev if r['domain']==d)/sum(r['domain']==d for r in dev),
                       case_disagreement=sum(r['disagreement_rate'] for r in dev if r['domain']==d)/sum(r['domain']==d for r in dev))
                 for d in sorted({r['domain'] for r in dev})}
        dev_gates=dict(DEV_case_KL=sum(r['KL'] for r in dev)/24<=1.,
            DEV_case_disagreement=sum(r['disagreement_rate'] for r in dev)/24<=.20,
            all_domain_KL=all(r['case_KL']<=2. for r in domains.values()),
            all_domain_disagreement=all(r['case_disagreement']<=.35 for r in domains.values()))
        result=dict(schema='FALCON_PRETRAINED_PROJECTED_SHADOW_RESULT_V3',binding=extent(a.binding),freeze=a.freeze,
            controls=controls,tasks=tasks,correct=sum(categories.values()),categories=categories,quality_gates=quality,
            own_answer_followup=followup,DEV=dev,DEV_case_KL=sum(r['KL'] for r in dev)/24,
            DEV_case_disagreement=sum(r['disagreement_rate'] for r in dev)/24,
            DEV_domains=domains,DEV_quality_gates=dev_gates,
            target_elements=194801792,source_elements=1554863488,model=extent(a.directory/'model.f32.pt'),
            outputs=outputs,inputs_before_after_exact=True,parameters_before_after_identity_exact=True,
            seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            optimizer_updates=0,new_teacher_generations=0,native_calls=0,engine_admission=False,speed_admission=False,
            scope='Fixed fully pretrained projected floating shadow; sequential residual projection approximate; '
                'full source feature/state/attention/FFN widths and depth provisional; no original LUT/ternary selective deployment. '
                'Fresh reset for followup includes actual own answer; no incremental reuse claim. '
                'DEV source labels inherited BF16 cached source, target full-prefill F32; rounding distinction preserved.')
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage,
                       completed_tasks=len(tasks),completed_DEV=len(dev));raise
    finally:
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.out.with_suffix('.terminal.json'),receipt)


def launch(a):
    import subprocess
    import psutil
    from original_falcon_whole_recovery import memory_reader
    assert extent(a.binding)['sha256']==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['limits']==LIMITS
    argv=[b['python'],'-I','-S','-B','-X','utf8',str(Path(__file__).resolve()),'--worker',
        '--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--freeze',a.freeze,
        '--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
    log=a.out.with_suffix('.worker.log');holder=a.out.with_suffix('.holder.json')
    assert not log.exists() and not holder.exists() and not a.out.exists() and not a.directory.exists()
    proc=psutil.Process();reader=memory_reader();peak=0;child=None;offset=0
    record=dict(schema='FALCON_PROJECTED_SHADOW_HOLDER_V3',binding=extent(a.binding),freeze=a.freeze,
        command=argv,holder_pid=proc.pid,holder_creation_time=proc.create_time(),timing_control_admitted=False)
    try:
        with log.open('xb') as stream:
            child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,creationflags=8)
            record.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time())
            while child.poll() is None:
                peak=max(peak,reader(child))
                assert time.monotonic()-START<=LIMITS['seconds'],'holder deadline'
                assert peak+proc.memory_info().peak_wset<=LIMITS['OS_bytes'],'combined OS cap'
                with log.open('rb') as f:
                    f.seek(offset);data=f.read();end=data.rfind(b'\n')+1
                    if end:print(data[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.2)
        peak=max(peak,reader(child));assert child.returncode==0,log.read_text(encoding='utf8')[-5000:]
        record.update(exit_code=0,error=None,result=extent(a.out),worker_terminal=extent(a.out.with_suffix('.terminal.json')))
    except BaseException as error:
        if child is not None and child.poll() is None:child.kill();child.wait()
        record.update(exit_code=child.returncode if child is not None else None,error=repr(error));raise
    finally:
        record.update(seconds=time.monotonic()-START,worker_OS_peak_through_exit=peak,
            holder_OS_peak=proc.memory_info().peak_wset)
        write(holder,record)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--worker',action='store_true');m.add_argument('--launch',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.prepare:prepare(a)
    else:
        assert all((a.binding,a.binding_sha,a.freeze,a.directory))
        (launch if a.launch else worker)(a)
