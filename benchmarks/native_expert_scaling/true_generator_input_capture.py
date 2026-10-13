"""Actual BF16 all-layer generator inputs and three-site original-source continuity."""
import argparse,hashlib,importlib.util,json,os,struct,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from chatbot_falcon_usability import sha,write
from original_falcon_whole_recovery import extent,check_inputs,memory_reader


def bind(a):
    import numpy as np,psutil,torch,transformers,safetensors
    begin=time.monotonic();assert not a.out.exists() and not a.directory.exists() and not torch.cuda.is_initialized();prior=DOC/'original_falcon_recurrent_capture_binding_20261009.json';pb=json.loads(prior.read_bytes());adopt=DOC/'original_falcon_recurrent_adopted_result_20261009.json';old=json.loads(adopt.read_bytes());corpus=Path(pb['corpus']);records=json.loads(corpus.read_bytes())['records'];assert len(records)==48 and sum(len(r['student_input_ids']) for r in records)==22547;source=Path(pb['source']);weights=source/'model.safetensors';method=Path(old['records'][0]['sites'][0]['fields']['x']['path']).parent/'source_method.json';storage=DOC/'true_generator_input_capture_storage_20261013.json';st=json.loads(storage.read_text(encoding='utf-8-sig'));assert str(a.directory.resolve())==str(Path(st['new_output_directory']).resolve()) and shutil.disk_usage(a.directory.parent).free>8<<30
    tensors={}
    with weights.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size));offset=8+size
        for name,spec in header.items():
            if name=='__metadata__':continue
            assert spec['dtype']=='BF16';lo,hi=spec['data_offsets'];f.seek(offset+lo);data=f.read(hi-lo);assert len(data)==hi-lo;tensors[name]=dict(shape=spec['shape'],bytes=hi-lo,sha256=hashlib.sha256(data).hexdigest())
    assert len(tensors)==411;rb=json.loads((DOC/'gate_input_recovery_binding_20261013.json').read_bytes());assert extent(weights)==next(e for e in rb['inputs'] if Path(e['path']).resolve()==weights.resolve());nv=Path(shutil.which('nvidia-smi'));gpu=subprocess.run([str(nv),'--query-gpu=uuid,name,driver_version,memory.total,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15).stdout.strip();assert len(gpu.splitlines())==1 and 'GPU-1cce7fba-8103-2d4b-ad9b-df8e162f8221' in gpu and 'RTX 3060' in gpu;fields=gpu.split(',');assert int(fields[-2])-int(fields[-1])>=11264
    paths=[Path(__file__),B/'true_generator_input_capture_audit.py',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',B/'original_falcon_whole_recovery.py',B/'chatbot_falcon_ssd_tiles.py',B/'original_falcon_recurrent_capture.py',B/'falcon_auxiliary_s6_transport.py',DOC/'TRUE_GENERATOR_INPUT_CAPTURE_PROTOCOL_20261013.md',DOC/'TRUE_GENERATOR_INPUT_CAPTURE_AUDIT_PROTOCOL_20261013.md',storage,prior,adopt,corpus,method,DOC/'gate_input_recovery_binding_20261013.json',DOC/'recovered_input_temporal_result_20261013.json',DOC/'recovered_input_temporal_adjudication_20261013.json',DOC/'recovered_input_temporal_bf16_witness_v2_20261013.json',DOC/'temporal_error_decomposition_adjudication_v2_20261013.json',Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__),Path(torch.__file__),Path(torch._C.__file__),Path(transformers.__file__),Path(safetensors.__file__),nv]
    paths+=list(source.glob('*'));nr=Path(np.__file__).resolve().parent;tr=Path(torch.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;sr=Path(safetensors.__file__).resolve().parent;paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list((tr/'lib').glob('*.dll'))+list(pr.glob('*.pyd'))+list(sr.glob('*.pyd'))
    paths += [SITE/'transformers'/q for q in ['modeling_utils.py','configuration_utils.py','cache_utils.py','models/falcon_h1/modeling_falcon_h1.py','models/falcon_h1/configuration_falcon_h1.py']];paths += [p for p in [Path('C:/Windows/System32/nvcuda.dll'),Path('C:/Windows/System32/nvml.dll'),Path('C:/Windows/System32/nvapi64.dll')] if p.exists()]
    for rec in old['records']:
        for site in [0,12,23]:paths += [Path(e['path']) for e in rec['sites'][site]['fields'].values()]
    paths += [ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=[extent(p) for p in dict.fromkeys(paths) if p.is_file()];assert time.monotonic()-begin<=900 and psutil.Process().memory_info().peak_wset<=6<<30 and not torch.cuda.is_initialized()
    b=dict(schema='TRUE_GENERATOR_INPUT_CAPTURE_BINDING_V1',python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),freeze=a.freeze,source=str(source.resolve()),source_revision=pb['source_revision'],corpus=str(corpus.resolve()),ordered_ids=pb['ordered_ids'],prior_capture=extent(adopt),prior_source_method=extent(method),checkpoint_tensors=tensors,MuP_bits=rb['semantics']['MuP_bits'],sites=list(range(24)),continuity_sites=[0,12,23],continuity_fields=['x','B','C','gate','delta','y','output'],fields=dict(pb['fields'],H=dict(width=2048,dtype='BF16')),cases=48,history_tokens=22547,expected_payload_bytes=4231620960,criteria=dict(continuity_each_case_field_relative_RMS=1e-4),limits=dict(seconds=1800,reserve_seconds=90,OS_bytes=12<<30,GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,output_bytes=6<<30,audit_seconds=1200,audit_OS_bytes=6<<30),versions=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,psutil=psutil.__version__),gpu_query=gpu,output_directory=str(a.directory.resolve()),inputs=inputs,preparation_seconds=time.monotonic()-begin,scope='Actual all24 BF16 in_proj input bits after ssm_in_multiplier/3-site complete original recurrent continuity,48 full zero-state source base prefills,no LMhead/answers/RESERVED/optimizer/native quality/speed admission. All411 source coefficients verified before/after;primary resolved runtime DLLs/driver pinned.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),gpu=gpu,seconds=b['preparation_seconds'])),flush=True)


def launch(a):
    import psutil
    begin=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and str(a.directory.resolve())==b['output_directory'];assert not a.directory.exists() and not a.out.exists();log=a.out.with_suffix('.worker.log');assert not log.exists();proc=psutil.Process();proc.cpu_affinity([11]);child=None;peak=0;reader=memory_reader();receipt=dict(schema='TRUE_GENERATOR_INPUT_CAPTURE_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def stop_owned():
        if child is not None and child.poll() is None:
            assert abs(psutil.Process(child.pid).create_time()-receipt['worker_creation_time'])<.001;child.terminate();child.wait(timeout=15)
    try:
        check_inputs(b);env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0');argv=[b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker','--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
        with log.open('xb') as stream:
            child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=0x08000000);receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time(),worker_argv=argv);write(a.out.with_suffix('.owned_process.json'),receipt);offset=0;lastsize=begin
            while child.poll() is None:
                peak=max(peak,reader(child));assert time.monotonic()-begin<=b['limits']['seconds'];assert peak+proc.memory_info().peak_wset<=b['limits']['OS_bytes'];assert log.stat().st_size<=4<<20
                try:assert not psutil.Process(child.pid).children(recursive=True)
                except psutil.NoSuchProcess:assert child.poll() is not None
                if time.monotonic()-lastsize>5 and a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())+log.stat().st_size<=b['limits']['output_bytes'];lastsize=time.monotonic()
                with log.open('rb') as f:
                    f.seek(offset);chunk=f.read();cut=chunk.rfind(b'\n')+1
                    if cut:print(chunk[:cut].decode('utf-8',errors='replace'),end='',flush=True);offset+=cut
                time.sleep(.5)
        peak=max(peak,reader(child));receipt.update(worker_exit_code=child.returncode);assert child.returncode==0,log.read_text(errors='replace')[-5000:];check_inputs(b);r=json.loads(a.out.read_bytes());assert r['cases']==r['source_base_forwards']==48 and r['H_fields']==1152 and r['payload_bytes']==b['expected_payload_bytes'] and r['LM_head_calls']==0;assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())+log.stat().st_size<=b['limits']['output_bytes'];receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),elapsed=time.monotonic()-begin)),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc(),termination_requested=child is not None and child.poll() is None);stop_owned();receipt['worker_exit_code']=None if child is None else child.returncode;raise
    finally:
        receipt.update(seconds=time.monotonic()-begin,worker_OS_peak=peak,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


def verify_parameters(model,b,torch,np,guard):
    rows=[];total=0
    assert set(dict(model.named_parameters()))==set(b['checkpoint_tensors'])
    for name,p in model.named_parameters():
        guard()
        spec=b['checkpoint_tensors'][name];assert list(p.shape)==spec['shape'] and p.device.type=='cuda' and not p.requires_grad
        v=p.detach().cpu().contiguous();assert v.dtype in [torch.bfloat16,torch.float32]
        if v.dtype==torch.float32:
            bv=v.to(torch.bfloat16);assert torch.equal(v,bv.float());v=bv
        bits=v.view(torch.uint16).numpy().astype('<u2',copy=False);digest=hashlib.sha256(bits.tobytes()).hexdigest();assert digest==spec['sha256'],name
        rows.append(dict(name=name,shape=list(p.shape),runtime_dtype=str(p.dtype),canonical_BF16_sha256=digest));total+=p.numel()
    assert total==1554863488 and len(rows)==411
    return dict(tensors=rows,canonical_coefficient_words=total,exact_source_BF16_values=True)

def worker(a):
    start = time.monotonic(); assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes()); a.directory.mkdir(exist_ok=False)
    stage = 'startup'; completed = []; forwards = 0; total_bytes = 0
    torch = None; proc = None
    try:
        assert sys.version_info[:3] == (3, 12, 10)
        assert Path(sys.executable).resolve() == Path(b['python']).resolve()
        assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(k) is not None for k in ('kernels', 'mamba_ssm', 'causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        from transformers import FalconH1ForCausalLM
        from transformers.models.falcon_h1 import modeling_falcon_h1 as code
        from chatbot_falcon_ssd_tiles import install
        assert (torch.__version__, transformers.__version__, np.__version__, psutil.__version__) == ('2.6.0+cu124', '5.13.1', '2.4.6', '7.2.2')
        proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6); torch.set_num_interop_threads(1); torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

        def guard():
            lim = b['limits']
            assert time.monotonic()-start <= lim['seconds']-lim['reserve_seconds'], 'worker reserve'
            assert proc.memory_info().peak_wset <= lim['OS_bytes'], 'worker OS cap'
            assert torch.cuda.max_memory_allocated() <= lim['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= lim['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'unexpected child'
            assert total_bytes <= lim['output_bytes'], 'payload cap'

        def event(**values):
            guard(); print(json.dumps(dict(seconds=time.monotonic()-start, **values)), flush=True)

        def save(site, name, tensor):
            nonlocal total_bytes
            spec = b['fields'][name]
            assert tensor.shape == (1, current['history'], spec['width']), (site, name, tensor.shape)
            assert tensor.dtype == (torch.bfloat16 if spec['dtype'] == 'BF16' else torch.float32)
            assert torch.isfinite(tensor).all().item(), (site, name, 'nonfinite')
            value = tensor[0].detach().cpu().contiguous()
            array = value.view(torch.uint16).numpy().astype('<u2', copy=False) if spec['dtype'] == 'BF16' else value.numpy().astype('<f4', copy=False)
            path = a.directory/f"{current['id']}.site{site:02d}.{name}.{'bf16' if spec['dtype'] == 'BF16' else 'f32'}"
            temp = path.with_suffix(path.suffix+'.tmp')
            with temp.open('xb') as f:
                f.write(array.tobytes()); f.flush(); os.fsync(f.fileno())
            temp.replace(path)
            item = dict(**extent(path), shape=list(array.shape), dtype=spec['dtype'])
            current['sites'][site]['fields'][name] = item; total_bytes += item['bytes']
            assert item['bytes'] == current['history']*spec['width']*(2 if spec['dtype'] == 'BF16' else 4)

        method=install(code);assert method==json.loads(Path(b['prior_source_method']['path']).read_bytes());write(a.directory/'source_method.json',method)
        stage = 'source_load'
        model = FalconH1ForCausalLM.from_pretrained(b['source'], local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        identities = {name: (id(p), p._version) for name, p in model.named_parameters()};parameter_before=verify_parameters(model,b,torch,np,guard);write(a.directory/'parameter_before.json',parameter_before)
        assert len(identities) == 411 and sum(p.numel() for p in model.parameters()) == 1554863488
        assert model.config.mamba_chunk_size == 128 and not code.is_fast_path_available
        assert model.config.mamba_d_state == 256 and model.config.mamba_n_groups == 1
        assert model.config.mamba_rms_norm and not model.config.mamba_norm_before_gate
        current = None; staged = {}; head_calls = 0

        def head_hook(module, args):
            nonlocal head_calls
            head_calls += 1
            raise AssertionError('LM head must not be called')

        hooks = [model.lm_head.register_forward_pre_hook(head_hook)]
        def input_observer(site, module, args):
            assert current is not None and len(args)==1 and 'H' not in current['sites'][site]['fields']
            save(site,'H',args[0])


        def in_observer(site, mixer, module, args, output):
            assert current is not None and site not in staged
            projected = output * mixer.mup_vector
            gate, raw_bc, raw_delta = projected.split([3072, 3584, 48], dim=-1)
            delta = torch.nn.functional.softplus(raw_delta+mixer.dt_bias)
            delta = torch.clamp(delta, mixer.time_step_limit[0], mixer.time_step_limit[1])
            staged[site] = dict(gate=gate, delta=delta)

        def conv_observer(site, mixer, module, args, output):
            assert site in staged
            activated = mixer.act(output[..., :current['history']]).transpose(1, 2)
            xx, bb, cc = activated.split([3072, 256, 256], dim=-1)
            staged[site].update(x=xx, B=bb, C=cc)

        def norm_observer(site, module, args):
            nonlocal stage
            stage = f"{current['id']}/site{site:02d}/norm"
            yy, gate = args
            assert torch.equal(gate, staged[site]['gate'])
            for name in ('x', 'B', 'C', 'gate', 'delta'): save(site, name, staged[site][name])
            save(site, 'y', yy)
            del staged[site]
            guard()

        def output_observer(site, module, args, output):
            save(site, 'output', output)

        for site, layer in enumerate(model.model.layers):
            mixer = layer.mamba
            assert mixer.mup_vector.dtype==torch.bfloat16 and mixer.mup_vector.shape==(1,1,6704)
            actual_mu=mixer.mup_vector[0,0].detach().cpu().view(torch.uint16).numpy()
            assert np.array_equal(actual_mu,np.array(b['MuP_bits'],dtype='u2'))
            hooks.append(mixer.in_proj.register_forward_pre_hook(lambda m,args,site=site:input_observer(site,m,args)))
            if site in b['continuity_sites']:
                hooks.append(mixer.in_proj.register_forward_hook(lambda m, args, out, site=site, mixer=mixer: in_observer(site, mixer, m, args, out)))
                hooks.append(mixer.conv1d.register_forward_hook(lambda m, args, out, site=site, mixer=mixer: conv_observer(site, mixer, m, args, out)))
                hooks.append(mixer.norm.register_forward_pre_hook(lambda m, args, site=site: norm_observer(site, m, args)))
                hooks.append(mixer.register_forward_hook(lambda m, args, out, site=site: output_observer(site, m, args, out)))
        records = {r['id']: r for r in json.loads(Path(b['corpus']).read_bytes())['records']}
        event(stage='source_loaded', parameters=len(identities), capture_cases=b['cases'])
        for identifier in b['ordered_ids']:
            rec = records[identifier]; stage = identifier
            current = dict(id=identifier, split=rec['split'], domain=rec['domain'], history=len(rec['student_input_ids']),
                input_ids=rec['student_input_ids'], positions=rec['positions'],
                sites=[dict(site=site, fields={}) for site in b['sites']])
            guard()
            with torch.inference_mode():
                ids = torch.tensor([current['input_ids']], device='cuda', dtype=torch.long)
                result = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                forwards += 1
                assert torch.isfinite(result.last_hidden_state).all().item()
            assert not staged and all(set(v['fields'])==({'H'}|set(b['continuity_fields']) if v['site'] in b['continuity_sites'] else {'H'}) for v in current['sites'])
            assert current['history']*(24*4096+3*29792) == sum(f['bytes'] for v in current['sites'] for f in v['fields'].values())
            write(a.directory/(identifier+'.json'), current); completed.append(current)
            event(stage='captured_case', id=identifier, split=rec['split'], history=current['history'],
                cases=len(completed), payload_bytes=total_bytes, source_base_forwards=forwards)
            del ids, result; current = None
        for hook in hooks: hook.remove()
        assert {name: (id(p), p._version) for name, p in model.named_parameters()} == identities
        parameter_after=verify_parameters(model,b,torch,np,guard);assert parameter_before==parameter_after;write(a.directory/'parameter_after.json',parameter_after);assert head_calls == 0 and total_bytes == b['expected_payload_bytes']
        result = dict(schema='TRUE_GENERATOR_INPUT_CAPTURE_RESULT_V1', freeze=a.freeze,
            binding_sha256=a.binding_sha, decision='COMPLETE_PENDING_INDEPENDENT_CONTINUITY_AUDIT',
            cases=len(completed), sites_per_case=24, history_tokens=b['history_tokens'], records=completed,
            payload_bytes=total_bytes, source_base_forwards=forwards, LM_head_calls=head_calls,
            source_generations=0, optimizer_updates=0, native_calls=0, reserved_queries=0,
            source_parameter_identities_versions_unchanged=True,source_parameter_bytes_exact_before_after=True,parameter_before=extent(a.directory/'parameter_before.json'),parameter_after=extent(a.directory/'parameter_after.json'),source_method=extent(a.directory/'source_method.json'),H_fields=48*24,continuity_sites=b['continuity_sites'],continuity_fields=b['continuity_fields'],MuP_all24_BF16_bits_exact=True,
            full_prefill_scope='One full forced-history base-model call per case, zero initial state; '
                'not a byte-equivalence claim to earlier incremental cached teacher logits.',
            quality_admission=False, speed_admission=False,
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-start)
        write(a.out, result); event(stage='complete', decision=result['decision'], result_sha256=sha(a.out))
    except BaseException as error:
        write(a.directory/'first_fault.json', dict(stage=stage, error=repr(error),
            completed_ids=[r['id'] for r in completed], source_base_forwards=forwards,
            payload_bytes=total_bytes, elapsed_seconds=time.monotonic()-start,
            worker_OS_peak_snapshot=None if proc is None else proc.memory_info().peak_wset,
            GPU_allocated_peak=None if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=None if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_reserved()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--bind',action='store_true');m.add_argument('--launch',action='store_true');m.add_argument('--worker',action='store_true');p.add_argument('--freeze',required=True);p.add_argument('--out',type=Path,required=True)
    for name in ['binding','directory']:p.add_argument('--'+name,type=Path)
    p.add_argument('--binding-sha');a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
