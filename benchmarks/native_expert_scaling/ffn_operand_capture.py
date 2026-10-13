"""Observe actual all24 source FFN nodes on the qualified own-state chatbot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'benchmarks/native_expert_scaling'
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE = ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0] = [str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
from full_width_native_chat import NativeBridge,check_inputs
from true_generator_input_capture_v3 import verify_parameters,descendant_snapshot

FIELDS = dict(x=2048,up=4608,gate_raw=4608,gate_scaled=4608,features=4608,output=2048)
LIMITS = dict(seconds=7200,reserve_seconds=180,OS_bytes=8 << 30,
              GPU_allocated_bytes=6 << 30,GPU_reserved_bytes=10 << 30,
              output_bytes=32 << 30,audit_seconds=7200)


def load(path): return json.loads(Path(path).read_bytes())


class StrictSink:
    """Validate an observer's actual writes without supplying model inputs."""
    def __init__(self,descriptor,np):
        self.np = np; self.descriptor = descriptor; self.offset = 0
        self.values = np.memmap(descriptor['path'],'u2','r',shape=tuple(descriptor['shape'])).reshape(-1)
        self.digest = hashlib.sha256(); self.closed = False
    def write(self,data):
        assert not self.closed and len(data)%2 == 0
        words = self.np.frombuffer(data,dtype='u2'); hi = self.offset+len(words)
        assert hi <= len(self.values)
        assert self.np.array_equal(words,self.values[self.offset:hi]), (
            'parent trace mismatch',self.descriptor['path'],self.offset,hi)
        self.digest.update(data); self.offset = hi
        return len(data)
    def flush(self): pass
    def close(self):
        assert self.offset == len(self.values)
        assert self.digest.hexdigest() == self.descriptor['sha256']
        self.closed = True


class ObservedBridge(NativeBridge):
    def reset_reference(self,reference):
        assert self.streams is None
        self.close_contexts(); self.contexts = [self.create() for _ in range(24)]
        assert all(self.contexts)
        self.steps = 0; self.consumed = []; self.token = None
        self.streams = [StrictSink(s['trace'],self.np) for s in reference['sites']]
    def finish_reference(self,directory,label,reference):
        np,torch = self.np,self.torch
        assert self.steps == reference['seen'] and self.consumed == reference['consumed_ids']
        digests = []
        for sink in self.streams:
            sink.close(); digests.append(dict(bytes=sink.offset*2,sha256=sink.digest.hexdigest()))
        self.streams = None
        snapshots = []
        for site,context in enumerate(self.contexts):
            state = np.empty((3072,96),'f4'); ring = np.empty((3,3584),'u2'); seen = np.zeros(1,'u8')
            assert self.snapshot(context,state.ctypes.data,ring.ctypes.data,seen.ctypes.data) == 0
            last = np.array([self.last(context)],'u4')
            with np.load(reference['sites'][site]['final']['path'],allow_pickle=False) as old:
                assert np.array_equal(old['state'].view('u4'),state.view('u4'))
                assert np.array_equal(old['ring'],ring) and np.array_equal(old['seen'],seen)
                assert np.array_equal(old['last'],last)
            path = directory/f'{label}.site{site:02d}.state.npz'
            np.savez(path,state=state,ring=ring,seen=seen,last=last)
            snapshots.append(extent(path))
        attention = []
        for site,layer in enumerate(self.cache.layers):
            for key in ('keys','values'):
                tensor = getattr(layer,key,None); assert tensor is not None
                bits = tensor.detach().cpu().contiguous().view(torch.uint16).numpy()
                attention.append(dict(site=site,field=key,shape=list(tensor.shape),dtype=str(tensor.dtype),
                    sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
        assert attention == reference['attention']
        self.close_contexts()
        return dict(actual_native_trace_digests=digests,final_snapshots=snapshots,
                    attention=attention,all_parent_words_and_states_exact=True)


def bind(a):
    import numpy as np
    import psutil
    import torch
    import transformers
    import safetensors
    begin = time.monotonic()
    assert not torch.cuda.is_initialized() and not a.directory.exists() and not a.out.exists()
    rp = DOC/'full_width_native_chat_result_20261013.json'
    ap = DOC/'full_width_native_chat_adjudication_20261013.json'
    cp = DOC/'full_width_native_chat_session_closure_20261013.json'
    r,ad,cl = [load(p) for p in (rp,ap,cp)]
    assert ad['decision'] == 'INDEPENDENT_STORED_QUALITY_PASS' and ad['report'] == extent(rp)
    assert cl['adjudication'] == extent(ap) and cl['decision'] == 'KNOWN_CORPUS_FULL_WIDTH_NATIVE_BACKBONE_QUALIFIED'
    pb = load(r['binding']['path']); assert extent(r['binding']['path']) == r['binding']
    original = {x['id']:x for x in pb['records']}
    records = [dict(id=x['id'],split=x['split'],domain=x['domain'],
        input_ids=x['sequence']['consumed_ids'],positions=original[x['id']]['positions'],parent=x)
        for x in r['records']]
    assert len(records) == 48 and sum(len(x['input_ids']) for x in records) == 22547
    assert sum(len(x['positions']) for x in records) == 8808
    config = load(Path(pb['source'])/'config.json'); assert config['mlp_bias'] is False
    alpha,beta = config['mlp_multipliers']
    fields_bytes = 22547*24*sum(FIELDS.values())*2
    snapshots_bytes = 48*24*(3072*96*4+3*3584*2+1024)
    labels_bytes = 8808*65537*4
    assert fields_bytes == 24381063168
    assert fields_bytes+snapshots_bytes+labels_bytes < LIMITS['output_bytes']-(1 << 30)
    assert shutil.disk_usage(a.directory.parent).free > LIMITS['output_bytes']+(2 << 30)
    paths = [Path(__file__),B/'ffn_operand_capture_audit.py',
             DOC/'FFN_OPERAND_CAPTURE_PROTOCOL_20261013.md',rp,ap,cp,Path(r['binding']['path']),ROOT/'AGENTS.md']
    paths += [Path(e['path']) for e in pb['inputs']+r['outputs']]
    paths += [B/n for n in ('composed_operator_mixed.py','full_width_native_chat.py',
        'row_state.py','token_state.py','full_channel_native.py','falcon_auxiliary_s6_transport.py',
        'true_generator_input_capture_v3.py','original_falcon_whole_recovery.py',
        'chatbot_falcon_ssd_tiles.py','chatbot_falcon_usability.py')]
    # Pin full numerical/framework package bodies used by this new collector.
    # These current pins do not repair the parent's historical pin limitations.
    for module in (np,psutil,torch,transformers,safetensors):
        base = Path(module.__file__).resolve().parent
        paths += [p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in ('.py','.pyd','.dll')]
    inputs = [extent(p) for p in dict.fromkeys(paths)]
    parent_pins = {e['path']:e for e in pb['inputs']+r['outputs']}
    for e in inputs:
        if e['path'] in parent_pins: assert e == parent_pins[e['path']],e['path']
    gpu = subprocess.run([shutil.which('nvidia-smi'),
        '--query-gpu=uuid,name,driver_version,memory.total,memory.used',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
    assert 'GPU-1cce7fba-8103-2d4b-ad9b-df8e162f8221' in gpu
    assert int(gpu.split(',')[-2])-int(gpu.split(',')[-1]) >= 10240
    binding = dict(schema='FFN_OPERAND_CAPTURE_BINDING_V1',freeze=a.freeze,python=pb['python'],
        source=pb['source'],source_revision=pb['source_revision'],checkpoint_tensors=pb['checkpoint_tensors'],
        MuP_bits=pb['MuP_bits'],prior_source_method=pb['prior_source_method'],versions=pb['versions'],
        compile=pb['compile'],table=pb['table'],exports=pb['exports'],records=records,
        parent_report=extent(rp),parent_audit=extent(ap),parent_closure=extent(cp),
        fields=FIELDS,gate_multiplier=alpha,down_multiplier=beta,
        fields_payload_bytes=fields_bytes,calculated_payload_bound=fields_bytes+snapshots_bytes+labels_bytes,
        inputs=inputs,limits=LIMITS,GPU_query=gpu,output_directory=str(a.directory.resolve()),
        worker_path=str(Path(__file__).resolve()),preparation_seconds=time.monotonic()-begin,
        criteria=dict(source_matrix_each_case_field_relative_RMS=.01,
                      actual_feature_node_relative_RMS=1e-4,gate_scalar_bits_exact=True,
                      parent_native_traces_states_cache_and_label_bits_exact=True),
        scope='Actual own pre-FFN normalized input/all24 source FFN operands on known48 '
              'histories,qualified native states unchanged. Six BF16 nodes with raw/scaled '
              'gate separation. No expert fit/model substitution/new generation/RESERVED/T4/TPS admission.')
    assert not torch.cuda.is_initialized()
    write(a.out,binding); print(json.dumps(dict(binding=extent(a.out),fields_bytes=fields_bytes,
        calculated_bound=binding['calculated_payload_bound'],input_pins=len(inputs),GPU_query=gpu)),flush=True)


def worker(a):
    begin = time.monotonic(); assert extent(a.binding)['sha256'] == a.binding_sha
    b = load(a.binding); assert b['limits'] == LIMITS and not a.out.exists() and not a.directory.exists()
    import numpy as np
    import psutil
    import torch
    import transformers
    from transformers import FalconH1ForCausalLM
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    from chatbot_falcon_ssd_tiles import install
    assert sys.version_info[:3] == (3,12,10) and Path(sys.executable).resolve() == Path(b['python']).resolve()
    assert (torch.__version__,transformers.__version__,np.__version__,psutil.__version__) == (
        '2.6.0+cu124','5.13.1','2.4.6','7.2.2')
    assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
    proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    torch.set_num_threads(6); torch.set_num_interop_threads(1); torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    stage = 'hashes'; bridge = None; hooks = []; records = []; opened = []; total = 0; current = None
    receipt = dict(schema='FFN_OPERAND_CAPTURE_WORKER_TERMINAL_V1',command=sys.orig_argv,
        pid=proc.pid,creation_time=proc.create_time(),binding=extent(a.binding),freeze=a.freeze)
    def guard():
        assert time.monotonic()-begin <= LIMITS['seconds']-LIMITS['reserve_seconds'],'worker deadline'
        assert proc.memory_info().peak_wset <= LIMITS['OS_bytes'],'OS cap'
        assert torch.cuda.max_memory_allocated() <= LIMITS['GPU_allocated_bytes'],'GPU allocated cap'
        assert torch.cuda.max_memory_reserved() <= LIMITS['GPU_reserved_bytes'],'GPU reserved cap'
        assert not descendant_snapshot(proc)['fresh'],'unexpected descendant'
        assert total <= LIMITS['output_bytes'],'payload cap'
    def event(**values):
        guard(); print(json.dumps(dict(stage=stage,seconds=time.monotonic()-begin,**values)),flush=True)
    def save(site,name,tensor):
        nonlocal total
        assert current is not None and tensor.dtype == torch.bfloat16 and tensor.shape == (1,1,FIELDS[name])
        t = bridge.steps; assert current['counts'][site][name] == t
        bits = tensor[0].detach().cpu().contiguous().view(torch.uint16).numpy()
        assert not np.any((bits&0x7f80) == 0x7f80),'nonfinite FFN node'
        data = bits.tobytes(); current['streams'][site][name].write(data)
        current['counts'][site][name] += 1; total += len(data)
    try:
        check_inputs(b,guard); a.directory.mkdir(); stage = 'source_load'
        method = install(code); assert method == load(b['prior_source_method']['path'])
        write(a.directory/'source_method.json',method)
        model = FalconH1ForCausalLM.from_pretrained(b['source'],local_files_only=True,
            trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        before = verify_parameters(model,b,torch,np,guard); write(a.directory/'parameter_before.json',before)
        identities = {n:(id(p),p._version) for n,p in model.named_parameters()}
        assert not code.is_fast_path_available and model.config.mamba_chunk_size == 128
        assert model.config.hidden_size == 2048 and len(model.model.layers) == 24
        assert model.config.mamba_d_state == 256 and model.config.mamba_n_groups == 1
        assert model.config.mamba_rms_norm and not model.config.mamba_norm_before_gate
        assert all(m.mamba.mup_vector[0,0].detach().cpu().view(torch.uint16).tolist() == b['MuP_bits']
                   for m in model.model.layers)
        bridge = ObservedBridge(model,b,np,torch)
        for site,layer in enumerate(model.model.layers):
            f = layer.feed_forward
            assert f.gate_multiplier == b['gate_multiplier'] and f.down_multiplier == b['down_multiplier']
            assert f.gate_proj.bias is f.up_proj.bias is f.down_proj.bias is None
            def inp(module,args,i=site):
                assert len(args) == 1; save(i,'x',args[0])
            def up(module,args,out,i=site): save(i,'up',out)
            def gate(module,args,out,i=site,alpha=f.gate_multiplier):
                save(i,'gate_raw',out); save(i,'gate_scaled',out*alpha)
            def features(module,args,i=site):
                assert len(args) == 1; save(i,'features',args[0])
            def output(module,args,out,i=site): save(i,'output',out)
            hooks += [f.register_forward_pre_hook(inp),f.up_proj.register_forward_hook(up),
                f.gate_proj.register_forward_hook(gate),f.down_proj.register_forward_pre_hook(features),
                f.register_forward_hook(output)]
        event(operation='source_verified_and_observers_installed')
        stage = 'known48_histories'
        with torch.inference_mode():
            for rec in b['records']:
                guard(); n = len(rec['input_ids']); reference = rec['parent']['sequence']
                bridge.reset_reference(reference)
                paths = [{name:a.directory/f'{rec["id"]}.site{i:02d}.{name}.bf16' for name in FIELDS} for i in range(24)]
                current = dict(streams=[{name:path.open('xb') for name,path in row.items()} for row in paths],
                               counts=[dict.fromkeys(FIELDS,0) for _ in range(24)])
                opened = [stream for row in current['streams'] for stream in row.values()]
                parent_scores = np.memmap(rec['parent']['scores']['path'],'f4','r',shape=(len(rec['positions']),65537))
                positions = set(rec['positions']); scores = []; label = 0
                for t,token in enumerate(rec['input_ids']):
                    if t%64 == 0: event(operation='progress',id=rec['id'],position=t)
                    value = bridge.step(token,t in positions)
                    if value is not None:
                        assert np.array_equal(value.view('u4'),parent_scores[label].view('u4')),'parent label logits mismatch'
                        scores.append(value); label += 1
                assert label == len(rec['positions']) and all(v == n for row in current['counts'] for v in row.values())
                for stream in opened: stream.flush(); os.fsync(stream.fileno()); stream.close()
                opened = []
                field_desc = [{name:dict(**extent(path),shape=[n,FIELDS[name]],dtype='BF16')
                               for name,path in row.items()} for row in paths]
                for row in field_desc:
                    for name,e in row.items(): assert e['bytes'] == n*FIELDS[name]*2
                controls = bridge.finish_reference(a.directory,rec['id'],reference)
                score_path = a.directory/(rec['id']+'.labels.f32'); np.stack(scores).astype('<f4').tofile(score_path)
                record = dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,
                    input_ids=rec['input_ids'],positions=rec['positions'],sites=field_desc,
                    controls=controls,scores=extent(score_path),parent_score_bits_exact=True)
                records.append(record); write(a.directory/(rec['id']+'.receipt.json'),record)
                total = sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())
                current = None; event(operation='history_complete',id=rec['id'],histories=len(records))
        stage = 'after_hash'
        assert {n:(id(p),p._version) for n,p in model.named_parameters()} == identities
        after = verify_parameters(model,b,torch,np,guard); assert after == before
        write(a.directory/'parameter_after.json',after)
        assert all(m.mamba.mup_vector[0,0].detach().cpu().view(torch.uint16).tolist() == b['MuP_bits']
                   for m in model.model.layers)
        assert all(f.feed_forward.gate_multiplier == b['gate_multiplier'] and
                   f.feed_forward.down_multiplier == b['down_multiplier'] for f in model.model.layers)
        check_inputs(b,guard); guard(); assert len(records) == 48
        outputs = [extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        payload = sum(e['bytes'] for rec in records for row in rec['sites'] for e in row.values())
        assert payload == b['fields_payload_bytes']
        result = dict(schema='FFN_OPERAND_CAPTURE_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,
            records=records,outputs=outputs,field_files=48*24*6,FFN_positions=22547*24,
            fields_payload_bytes=payload,output_bytes=sum(e['bytes'] for e in outputs),
            all_parent_native_traces_states_cache_and_label_bits_exact=True,
            all411_source_parameters_exact_before_after=True,before_after_hashes_exact=True,
            new_generations=0,optimizer_updates=0,RESERVED_queries=0,
            source_base_forwards=22547,LM_head_calls=8808,
            decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',scope=b['scope'],
            seconds=time.monotonic()-begin,OS_peak=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        assert result['output_bytes'] <= LIMITS['output_bytes']; write(a.out,result)
        receipt.update(exit_code=0,error=None,result=extent(a.out)); event(operation='completed',result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),stage=stage,traceback=traceback.format_exc(),
                       completed_histories=len(records)); raise
    finally:
        for hook in hooks: hook.remove()
        for stream in opened: stream.close()
        if bridge is not None: bridge.close_contexts()
        receipt.update(seconds=time.monotonic()-begin,OS_peak=proc.memory_info().peak_wset,
                       GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.out.with_suffix('.worker_terminal.json'),receipt)


def launch(a):
    import psutil
    from original_falcon_whole_recovery import memory_reader
    begin = time.monotonic(); assert extent(a.binding)['sha256'] == a.binding_sha
    b = load(a.binding); assert b['limits'] == LIMITS and not a.directory.exists() and not a.out.exists()
    check_inputs(b)
    env = dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0')
    argv = [b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker','--binding',str(a.binding.resolve()),
        '--binding-sha',a.binding_sha,'--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
    log = a.out.with_suffix('.worker.log'); assert not log.exists()
    proc = psutil.Process(); proc.cpu_affinity([11]); reader = memory_reader(); peak = 0; offset = 0; child = None
    receipt = dict(schema='FFN_OPERAND_CAPTURE_HOLDER_V1',binding=extent(a.binding),command=sys.orig_argv,
                   pid=proc.pid,creation_time=proc.create_time(),worker_argv=argv,freeze=a.freeze)
    try:
        with log.open('xb') as stream:
            child = subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time())
            write(a.out.with_suffix('.owned_process.json'),receipt)
            while child.poll() is None:
                peak = max(peak,reader(child))
                assert time.monotonic()-begin <= LIMITS['seconds'],'holder deadline'
                assert peak+proc.memory_info().peak_wset <= LIMITS['OS_bytes'],'combined OS cap'
                with log.open('rb') as f:
                    f.seek(offset); data = f.read(); end = data.rfind(b'\n')+1
                    if end:
                        print(data[:end].decode('utf8',errors='replace'),end='',flush=True); offset += end
                time.sleep(.5)
        peak = max(peak,reader(child)); receipt['worker_exit_code'] = child.returncode
        assert child.returncode == 0,log.read_text(encoding='utf8',errors='replace')[-5000:]
        check_inputs(b); r = load(a.out)
        assert len(r['records']) == 48 and r['field_files'] == 6912 and r['fields_payload_bytes'] == b['fields_payload_bytes']
        assert r['all_parent_native_traces_states_cache_and_label_bits_exact']
        wt = load(a.out.with_suffix('.worker_terminal.json'))
        assert wt['exit_code'] == 0 and wt['result'] == extent(a.out)
        receipt.update(exit_code=0,error=None,result=extent(a.out),worker_terminal=extent(a.out.with_suffix('.worker_terminal.json')))
    except BaseException as error:
        if child is not None and child.poll() is None:
            assert abs(psutil.Process(child.pid).create_time()-receipt['worker_creation_time']) < .001
            child.terminate(); child.wait(timeout=15)
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),
                       worker_exit_code=None if child is None else child.returncode); raise
    finally:
        receipt.update(seconds=time.monotonic()-begin,worker_OS_peak=peak,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); m = p.add_mutually_exclusive_group(required=True)
    for mode in ('bind','worker','launch'): m.add_argument('--'+mode,action='store_true')
    for key in ('binding','directory','out'): p.add_argument('--'+key,type=Path)
    for key in ('binding-sha','freeze'): p.add_argument('--'+key,required=key == 'freeze')
    a = p.parse_args()
    if a.worker: worker(a)
    elif a.launch: launch(a)
    else:
        import psutil
        start = time.monotonic(); proc = psutil.Process()
        receipt = dict(schema='FFN_OPERAND_CAPTURE_BIND_TERMINAL_V1',command=sys.orig_argv,
                       pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze)
        try: bind(a); receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as error:
            receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
        finally:
            receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
            write(a.out.with_suffix('.preparation_terminal.json'),receipt)
