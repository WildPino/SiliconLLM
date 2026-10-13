"""Full-width Falcon quality bridge with 24 independently owned native SSMs.

Source attention/FFN/generators/head still run on CUDA. This is a quality
experiment, not a CPU deployment or a conditional-capacity speed claim.
"""
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
import types

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0] = [str(B), str(SITE)]
from falcon_auxiliary_s6_transport import extent, write
from true_generator_input_capture_v3 import verify_parameters, descendant_snapshot

LIMITS = dict(seconds=7200, reserve_seconds=180, OS_bytes=8 << 30,
              GPU_allocated_bytes=6 << 30, GPU_reserved_bytes=10 << 30,
              output_bytes=24 << 30, audit_seconds=7200)


def load(path):
    return json.loads(Path(path).read_bytes())


def check_inputs(binding, guard=lambda: None):
    for item in binding['inputs']:
        guard()
        assert extent(item['path']) == item, item['path']


def metrics(record, scores, np):
    teacher = np.memmap(record['logits']['path'], '<u2', 'r',
                        shape=(len(record['positions']), 65537))
    loss = []
    wrong = 0
    for i, value in enumerate(scores):
        q = (teacher[i].astype('<u4') << 16).view('<f4').astype('f8')
        p = value.astype('f8')
        q -= q.max(); q -= np.log(np.exp(q).sum())
        p -= p.max(); p -= np.log(np.exp(p).sum())
        loss.append(float(np.sum(np.exp(q) * (q-p))))
        wrong += int(np.argmax(q) != np.argmax(p))
    assert np.isfinite(loss).all()
    return dict(id=record['id'], split=record['split'], domain=record['domain'],
                labels=len(loss), KL=float(np.mean(loss)), KL_per_label=loss,
                disagreement=wrong, disagreement_rate=wrong/len(loss))


def quality(tasks, followup, rows, controls, gates):
    categories = {c: sum(t['correct'] for t in tasks if t['category'] == c)
                  for c in sorted({t['category'] for t in tasks})}
    dev = [r for r in rows if r['split'] == 'DEV']
    domains = {d: dict(KL=sum(r['KL'] for r in dev if r['domain'] == d) /
                         sum(r['domain'] == d for r in dev),
                      disagreement=sum(r['disagreement_rate'] for r in dev if r['domain'] == d) /
                         sum(r['domain'] == d for r in dev))
               for d in sorted({r['domain'] for r in dev})}
    decisions = dict(
        task_min=sum(categories.values()) >= gates['task_correct_min'],
        category_min=all(v >= gates['category_correct_min'] for v in categories.values()),
        blank_max=sum(not t['output_text'].strip() for t in tasks) <= gates['blank_max'],
        no_special_leak=not any(t['special_leak'] for t in tasks+[followup]),
        stop_policy=all(t['stop_policy'] for t in tasks+[followup]),
        own_answer_followup=followup['correct'],
        DEV_case_KL=sum(r['KL'] for r in dev)/len(dev) <= gates['DEV_case_KL'],
        DEV_case_disagreement=sum(r['disagreement_rate'] for r in dev)/len(dev) <= gates['DEV_case_disagreement'],
        domain_KL=all(d['KL'] <= gates['domain_case_KL'] for d in domains.values()),
        domain_disagreement=all(d['disagreement'] <= gates['domain_case_disagreement'] for d in domains.values()),
        prefix_logits_routes_bit_exact=all(c['exact'] for c in controls),
        mass_defect=True)  # Exactly one selected source function, mass 1 at every site/token.
    return dict(categories=categories, correct=sum(categories.values()), domains=domains,
                DEV_case_KL=sum(r['KL'] for r in dev)/len(dev),
                DEV_case_disagreement=sum(r['disagreement_rate'] for r in dev)/len(dev),
                gates=decisions, all_pass=all(decisions.values()))


def bind(a):
    import numpy as np
    import psutil
    import torch
    import transformers
    from original_engine_chat import ChatTokenizer
    begin = time.monotonic()
    assert not torch.cuda.is_initialized() and not a.out.exists() and not a.directory.exists()
    source_binding_path = DOC / 'true_generator_input_capture_binding_v4_20261013.json'
    sb = load(source_binding_path)
    capture_path = DOC/'true_generator_input_capture_result_v3_20261013.json'
    capture_audit_path = DOC/'true_generator_input_capture_adjudication_20261013.json'
    capture, capture_audit = load(capture_path), load(capture_audit_path)
    assert capture['binding_sha256'] == extent(source_binding_path)['sha256']
    assert capture_audit['decision'] == 'INDEPENDENT_DATA_AND_SOURCE_CONTINUITY_PASS'
    assert capture_audit['report'] == extent(capture_path)
    rp = DOC / 'row_state_result_20261013.json'
    ap = DOC / 'row_state_adjudication_20261013.json'
    result, audit = load(rp), load(ap)
    assert audit['decision'] == 'INDEPENDENT_STORED_PASS' and audit['report'] == extent(rp)
    row_binding_path = Path(result['binding']['path'])
    rb = load(row_binding_path)
    native_path = DOC / 'recovered_categorical_native_chat_binding_20261012.json'
    nb = load(native_path)
    corpus_path = Path(sb['corpus'])
    records = load(corpus_path)['records']
    assert len(records) == 48 and sum(len(r['student_input_ids']) for r in records) == 22547
    assert sum(len(r['positions']) for r in records) == 8808
    for r in records:
        assert r['positions'] == sorted(set(r['positions']))
        assert all(0 <= i < len(r['student_input_ids']) for i in r['positions'])
        assert r['logits']['shape'] == [len(r['positions']), 65537]
    tokenizer = ChatTokenizer(sb['source'])
    assert len(nb['cases']) == 16
    assert all(tokenizer.encode(c['messages']) == c['input_ids'] for c in nb['cases'])
    baseline_path = DOC / 'original_engine_baseline_binding_20261009.json'
    baseline = load(baseline_path)['source_result']
    assert extent(baseline['path']) == baseline and load(baseline['path'])['correct'] == 14
    # Reuse resolved runtime/source pins, not the old capture's response archives.
    old_paths = [Path(e['path']) for e in sb['inputs']]
    paths = [p for p in old_paths if p.suffix.lower() not in ('.bf16','.f32','.npz')]
    paths += [Path(__file__), B/'full_width_native_chat_audit.py',
              B/'row_state.py', B/'token_state.py', B/'row_state.c',
              B/'full_channel_native.c', B/'full_channel_native.py', B/'integer_readout.c',
              B/'true_generator_input_capture_v3.py', B/'falcon_auxiliary_s6_transport.py',
              B/'chatbot_falcon_ssd_tiles.py', B/'original_engine_chat.py',
              B/'chatbot_hybrid_engine_chat.py', B/'chatbot_falcon_usability.py',
              B/'original_falcon_whole_recovery.py',
              DOC/'FULL_WIDTH_NATIVE_CHAT_PROTOCOL_20261013.md',
              source_binding_path, capture_path, capture_audit_path, row_binding_path, rp, ap, native_path,
              baseline_path, Path(baseline['path']), corpus_path,
              Path(sb['prior_source_method']['path']), Path(rb['table']['path']),
              Path(rb['compile']['dll']['path'])]
    paths += [Path(e['file']['path']) for e in rb['exports']]
    paths += [Path(r['logits']['path']) for r in records]
    paths += [ROOT/p for p in ['benchmarks/phase60/engine.c',
        'benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py',
        'docs/research/RESEARCH_INDEX.md']]
    inputs = [extent(p) for p in dict.fromkeys(paths)]
    # 23 contextual [H2048,row6704,out2048] + initial out2048, BF16.
    trace_stride = 23*(2048+6704+2048)*2+2048*2
    histories = sum(len(r['student_input_ids']) for r in records)
    prompt_bound = sum(len(c['input_ids']) for c in nb['cases'])
    # Followup can tokenize 64 generated tokens into more tokens after decoding.
    # Explicit hard bound 2048, two prefix-control copies of every known prompt.
    processed_bound = histories + prompt_bound*3 + 17*64 + 2048
    payload_bound = (processed_bound*trace_stride + (48+17+32)*24*(3072*96*4+3*3584*2+64)
                     + (8808+17*64+32)*65537*4)
    assert payload_bound < LIMITS['output_bytes'] - (1 << 30)
    assert shutil.disk_usage(a.directory.parent).free > LIMITS['output_bytes'] + (2 << 30)
    gpu = subprocess.run([shutil.which('nvidia-smi'),
        '--query-gpu=uuid,name,driver_version,memory.total,memory.used',
        '--format=csv,noheader,nounits'], capture_output=True, text=True,
        check=True, timeout=15).stdout.strip()
    assert 'GPU-1cce7fba-8103-2d4b-ad9b-df8e162f8221' in gpu
    assert int(gpu.split(',')[-2])-int(gpu.split(',')[-1]) >= 10240
    binding = dict(schema='FULL_WIDTH_NATIVE_CHAT_BINDING_V1', freeze=a.freeze,
        python=sb['python'], source=sb['source'], source_revision=sb['source_revision'],
        checkpoint_tensors=sb['checkpoint_tensors'], MuP_bits=sb['MuP_bits'],
        prior_source_method=sb['prior_source_method'], versions=sb['versions'],
        source_binding=extent(source_binding_path), row_binding=extent(row_binding_path),
        compile=rb['compile'], table=rb['table'], exports=rb['exports'],
        parent_report=extent(rp), parent_audit=extent(ap), inputs=inputs,
        corpus=extent(corpus_path), records=records, cases=nb['cases'],
        followup=nb['followup'], eos_ids=[11,228], special_ids=nb['special_ids'],
        gates=nb['gates'], numeric=rb['numeric'], contextual_row_relative=0.01,
        baseline=baseline, limits=LIMITS, output_directory=str(a.directory.resolve()),
        worker_path=str(Path(__file__).resolve()), trace_stride_bytes=trace_stride,
        processed_tokens_bound=processed_bound, calculated_payload_bound=payload_bound,
        GPU_query=gpu, preparation_seconds=time.monotonic()-begin,
        scope='Full hidden2048/source FFN/attention/head. Initial source-only F0, later23 '
              'contextual source generators,24 persistent native state96/p6 I16 readouts. '
              'CUDA quality bridge only;no final CPU/conditional useful-n/accepted TPS claim.',
        custody='Known repaired corpus reuses teacher logits from original failed family; '
                'no new donor generation or retroactive exit-status admission.',
        prefix_policy='Batch1/no padding; all source operators evaluated one token at a time. '
                      'Full-prefix wrapper vs separate continuation calls exact in this policy. '
                      'No claim of GPU parallel-prefill bit equivalence.')
    assert not torch.cuda.is_initialized()
    write(a.out, binding)
    print(json.dumps(dict(binding=extent(a.out), payload_bound=payload_bound,
                          processed_bound=processed_bound, GPU_query=gpu)), flush=True)


class NativeBridge:
    def __init__(self, model, binding, np, torch):
        from row_state import interface
        from token_state import arrays, ptrs
        self.np, self.torch, self.model = np, torch, model
        self.dll, self.create, self.destroy, self.tokens, self.rows, self.snapshot = interface(binding)
        self.last = self.dll.ts_last
        import ctypes
        self.last.argtypes = [ctypes.c_void_p]; self.last.restype = ctypes.c_uint32
        self.table = np.memmap(binding['table']['path'], dtype='u2', mode='r', shape=(65537,6704))
        self.params, self.weights, self.conv, self.outputs = [], [], [], []
        self.contexts = []
        self.token = None
        self.steps = 0
        self.cache = None
        self.streams = None
        self.consumed = []
        for e in binding['exports']:
            with np.load(e['file']['path'], allow_pickle=False) as z:
                p = {k: z[k].copy() for k in z.files}
            self.params.append(p)
            self.weights.append(ptrs([p[k] for k in ['Ws','Vs','Alog','D','gamma']]))
            self.conv.append(ptrs([p[k] for k in ['cw','cb','db']]))
            self.outputs.append(arrays(1,np))
        assert len(self.params) == 24
        for i, layer in enumerate(model.model.layers):
            def native(m, hidden_states, cache_params=None, attention_mask=None, site=i):
                return self.mix(site, m, hidden_states, cache_params, attention_mask)
            layer.mamba.forward = types.MethodType(native, layer.mamba)

    def close_contexts(self):
        for context in self.contexts:
            self.destroy(context)
        self.contexts = []
        self.cache = None

    def reset(self, directory, label):
        assert self.streams is None
        self.close_contexts()
        self.contexts = [self.create() for _ in range(24)]
        assert all(self.contexts)
        self.steps = 0; self.consumed = []; self.token = None
        self.trace_paths = [directory/f'{label}.site{i:02d}.trace.bf16' for i in range(24)]
        self.streams = [p.open('xb') for p in self.trace_paths]

    def mix(self, site, mixer, hidden, cache, mask):
        from token_state import arguments
        np, torch = self.np, self.torch
        assert hidden.shape == (1,1,2048) and hidden.dtype == torch.bfloat16
        assert self.streams is not None and self.token is not None and cache is not None
        assert mask is None or bool(torch.all(mask == 1).item())
        # Earlier layers have already appended their attention KV this step.
        assert cache.get_seq_length(site) == self.steps
        tags = np.array([self.token], 'u4')
        p, z = self.params[site], self.outputs[site]
        if site == 0:
            rc = self.tokens(self.contexts[site], 1, tags.ctypes.data,
                self.table.ctypes.data, self.weights[site], self.conv[site],
                p['q'].ctypes.data, p['scale'].ctypes.data, *arguments(z,0,1,np))
        else:
            h = hidden * mixer.ssm_in_multiplier
            row = mixer.in_proj(h) * mixer.mup_vector
            assert row.shape == (1,1,6704) and row.dtype == torch.bfloat16
            hbits = h[0].detach().cpu().contiguous().view(torch.uint16).numpy()
            rowbits = row[0].detach().cpu().contiguous().view(torch.uint16).numpy()
            rc = self.rows(self.contexts[site], 1, tags.ctypes.data,
                rowbits.ctypes.data, self.weights[site], self.conv[site],
                p['q'].ctypes.data, p['scale'].ctypes.data, *arguments(z,0,1,np))
            self.streams[site].write(hbits.tobytes())
            self.streams[site].write(rowbits.tobytes())
        assert rc == 0, (site,self.steps,rc)
        assert self.last(self.contexts[site]) == self.token
        self.streams[site].write(z['output'].tobytes())
        # Copy avoids aliasing reusable native output memory during async GPU work.
        out = torch.from_numpy(z['output'].copy()).view(torch.bfloat16).to(hidden.device)
        return out.reshape(1,1,2048)

    def step(self, token, need_logits=True):
        torch = self.torch
        assert 0 <= token < 65537
        self.token = token
        assert self.cache is None or self.cache.get_seq_length() == self.steps
        ids = torch.tensor([[token]], dtype=torch.long, device='cuda')
        out = self.model.model(input_ids=ids, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.steps += 1; self.consumed.append(token)
        assert self.cache.get_seq_length() == self.steps
        if need_logits:
            value = self.model.lm_head(out.last_hidden_state) * self.model.model.lm_head_multiplier
            assert value.dtype == torch.bfloat16 and value.shape == (1,1,65537)
            score = value[0,0].float().cpu().numpy()
            assert self.np.isfinite(score).all()
            return score
        return None

    def prefill(self, ids):
        value = None
        for i, token in enumerate(ids):
            value = self.step(token, i == len(ids)-1)
        return value

    def finish(self, directory, label):
        np = self.np
        assert self.streams is not None
        for stream in self.streams:
            stream.flush(); os.fsync(stream.fileno()); stream.close()
        self.streams = None
        records = []
        for i, context in enumerate(self.contexts):
            state = np.empty((3072,96),'f4'); ring = np.empty((3,3584),'u2')
            seen = np.zeros(1,'u8')
            assert self.snapshot(context,state.ctypes.data,ring.ctypes.data,seen.ctypes.data) == 0
            assert seen[0] == self.steps and self.last(context) == self.consumed[-1]
            path = directory/f'{label}.site{i:02d}.state.npz'
            np.savez(path, state=state, ring=ring, seen=seen,
                     last=np.array([self.last(context)],'u4'))
            width = 2048 if i == 0 else 10800
            assert self.trace_paths[i].stat().st_size == self.steps*width*2
            records.append(dict(site=i, trace=dict(**extent(self.trace_paths[i]),
                shape=[self.steps,width], dtype='BF16'), final=extent(path)))
        # Attention cache witness, as well as native states, belongs to the prefix control.
        attention = []
        for i, layer in enumerate(self.cache.layers):
            for key in ('keys','values'):
                tensor = getattr(layer,key,None)
                if tensor is not None:
                    bits = tensor.detach().cpu().contiguous().view(self.torch.uint16).numpy()
                    attention.append(dict(site=i,field=key,shape=list(tensor.shape),
                        dtype=str(tensor.dtype),sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
        assert len(attention) == 48
        result = dict(label=label, consumed_ids=self.consumed.copy(), seen=self.steps,
                      sites=records, attention=attention, selected_functions_per_step=24,
                      selection_policy='site index, one function per site, mass exactly 1')
        self.close_contexts()
        return result


def worker(a):
    begin = time.monotonic()
    assert extent(a.binding)['sha256'] == a.binding_sha
    b = load(a.binding)
    assert b['limits'] == LIMITS and not a.directory.exists() and not a.out.exists()
    import numpy as np
    import psutil
    import torch
    import transformers
    from transformers import FalconH1ForCausalLM
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    from chatbot_falcon_ssd_tiles import install
    from original_engine_chat import ChatTokenizer
    from chatbot_falcon_usability import normalized
    assert (torch.__version__, transformers.__version__, np.__version__, psutil.__version__) == (
        '2.6.0+cu124','5.13.1','2.4.6','7.2.2')
    assert sys.version_info[:3] == (3,12,10)
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
    proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    torch.set_num_threads(6); torch.set_num_interop_threads(1); torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    bridge = None; stage = 'before_hash'; tasks = []; rows = []; controls = []; sequences = []
    receipt = dict(schema='FULL_WIDTH_NATIVE_CHAT_WORKER_TERMINAL_V1', command=sys.orig_argv,
        binding=extent(a.binding), freeze=a.freeze, pid=proc.pid, creation_time=proc.create_time(),
        model_admitted=False, timing_admitted=False)
    def guard():
        assert time.monotonic()-begin <= LIMITS['seconds']-LIMITS['reserve_seconds'], 'worker deadline'
        assert proc.memory_info().peak_wset <= LIMITS['OS_bytes'], 'OS cap'
        assert torch.cuda.max_memory_allocated() <= LIMITS['GPU_allocated_bytes'], 'GPU allocation cap'
        assert torch.cuda.max_memory_reserved() <= LIMITS['GPU_reserved_bytes'], 'GPU reserve cap'
        assert not descendant_snapshot(proc)['fresh'], 'unexpected descendant'
        if a.directory.exists():
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= LIMITS['output_bytes']
    def event(**items):
        guard(); print(json.dumps(dict(stage=stage,seconds=time.monotonic()-begin,**items)),flush=True)
    try:
        check_inputs(b,guard); a.directory.mkdir()
        stage = 'source_load'
        method = install(code)
        assert method == load(b['prior_source_method']['path'])
        write(a.directory/'source_method.json', method)
        model = FalconH1ForCausalLM.from_pretrained(b['source'], local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16,
            attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        identities = {name:(id(p),p._version) for name,p in model.named_parameters()}
        before = verify_parameters(model,b,torch,np,guard)
        write(a.directory/'parameter_before.json',before)
        assert not code.is_fast_path_available and model.config.mamba_chunk_size == 128
        assert model.config.hidden_size == 2048 and len(model.model.layers) == 24
        assert model.config.mamba_d_state == 256 and model.config.mamba_n_groups == 1
        assert model.config.mamba_rms_norm and not model.config.mamba_norm_before_gate
        mup = [m.mamba.mup_vector[0,0].detach().cpu().view(torch.uint16).tolist()
               for m in model.model.layers]
        assert all(v == b['MuP_bits'] for v in mup)
        bridge = NativeBridge(model,b,np,torch)
        tokenizer = ChatTokenizer(b['source'])
        event(operation='source_verified_and_native_installed')
        with torch.inference_mode():
            stage = 'prefix_controls'
            for case in b['cases']:
                label = case['id']+'.prefix_wrapper'
                bridge.reset(a.directory,label)
                x = bridge.prefill(case['input_ids'])
                first = bridge.finish(a.directory,label)
                label = case['id']+'.prefix_steps'
                bridge.reset(a.directory,label)
                for i, token in enumerate(case['input_ids']):
                    y = bridge.step(token,i == len(case['input_ids'])-1)
                second = bridge.finish(a.directory,label)
                # NPZ container timestamps are not a semantic equality control.
                native_exact = True
                for s,t in zip(first['sites'],second['sites'],strict=True):
                    with np.load(s['final']['path']) as zs, np.load(t['final']['path']) as zt:
                        native_exact &= all(np.array_equal(zs[k].view('u1'),zt[k].view('u1')) for k in zs.files)
                exact = bool(np.array_equal(x.view('u4'),y.view('u4')) and
                    first['attention'] == second['attention'] and native_exact and
                    all(s['trace']['sha256'] == t['trace']['sha256']
                        for s,t in zip(first['sites'],second['sites'],strict=True)))
                score_path = a.directory/(case['id']+'.prefix_scores.npz')
                np.savez(score_path, wrapper=x, continuation=y)
                controls.append(dict(id=case['id'],exact=exact,scores=extent(score_path),
                                     wrapper=first,continuation=second))
                event(operation='prefix_control',id=case['id'],exact=exact)
            def generate(messages,label):
                ids = tokenizer.encode(messages)
                assert len(ids) <= 2048
                bridge.reset(a.directory,label)
                value = None
                for i,token in enumerate(ids):
                    if i % 32 == 0: guard()
                    value = bridge.step(token,i == len(ids)-1)
                new = []; scores = []
                for i in range(64):
                    guard(); scores.append(value.copy()); token = int(value.argmax()); new.append(token)
                    if token in b['eos_ids'] or i == 63: break
                    value = bridge.step(token)
                path = a.directory/(label+'.scores.f32')
                np.stack(scores).astype('<f4').tofile(path)
                seq = bridge.finish(a.directory,label); sequences.append(seq)
                text = tokenizer.decode(new)
                return dict(label=label,input_ids=ids,messages=messages,output_ids=new,
                    output_text=text,normalized=normalized(text),scores=extent(path),
                    sequence=seq,state_reset=True,
                    special_leak=any(i in set(b['special_ids'])-set(b['eos_ids']) for i in new),
                    stop_policy=bool(new) and not any(i in b['eos_ids'] for i in new[:-1]) and
                                (new[-1] in b['eos_ids'] or len(new) == 64))
            stage = 'canonical_tasks'
            followup = None
            for case in b['cases']:
                row = generate(case['messages'],case['id'])
                assert row['input_ids'] == case['input_ids']
                row.update(id=case['id'],category=case['category'],expected=case['expected'],
                           correct=row['normalized'] == case['expected'])
                tasks.append(row); write(a.directory/(case['id']+'.task.json'),row)
                event(operation='task',id=case['id'],correct=row['correct'],answer=row['output_text'])
                if case['id'] == b['followup']['case_id']:
                    messages = case['messages']+[dict(role='assistant',content=row['output_text']),
                        dict(role='user',content=b['followup']['prompt'])]
                    followup = generate(messages,'own_answer_followup')
                    followup.update(expected=b['followup']['expected'],
                                    correct=followup['normalized'] == b['followup']['expected'])
                    write(a.directory/'own_answer_followup.json',followup)
            stage = 'all48_known_histories'
            for rec in b['records']:
                bridge.reset(a.directory,rec['id']); positions = set(rec['positions']); scores = []
                for t,token in enumerate(rec['student_input_ids']):
                    if t % 64 == 0: event(operation='history_progress',id=rec['id'],position=t)
                    value = bridge.step(token,t in positions)
                    if value is not None: scores.append(value)
                values = np.stack(scores)
                assert values.shape == (len(rec['positions']),65537)
                path = a.directory/(rec['id']+'.labels.f32'); values.astype('<f4').tofile(path)
                seq = bridge.finish(a.directory,rec['id']); sequences.append(seq)
                row = metrics(rec,values,np)
                row.update(scores=extent(path),sequence=seq)
                rows.append(row); write(a.directory/(rec['id']+'.metrics.json'),row)
                event(operation='history_complete',id=rec['id'],KL=row['KL'],disagreement=row['disagreement_rate'])
        stage = 'after_hash'
        assert {name:(id(p),p._version) for name,p in model.named_parameters()} == identities
        after = verify_parameters(model,b,torch,np,guard)
        assert after == before
        write(a.directory/'parameter_after.json',after)
        assert [m.mamba.mup_vector[0,0].detach().cpu().view(torch.uint16).tolist()
                for m in model.model.layers] == mup
        check_inputs(b,guard)
        assert len(tasks) == 16 and followup is not None and len(rows) == 48 and len(controls) == 16
        q = quality(tasks,followup,rows,controls,b['gates'])
        outputs = [extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        result = dict(schema='FULL_WIDTH_NATIVE_CHAT_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,
            tasks=tasks,followup=followup,records=rows,prefix_controls=controls,quality=q,outputs=outputs,
            processed_tokens=sum(s['seen'] for s in sequences)+
                sum(c[k]['seen'] for c in controls for k in ('wrapper','continuation')),
            coefficient_words_verified_before_after=1554863488,
            all411_source_parameters_exact=True, before_after_hashes_exact=True,
            optimizer_updates=0, RESERVED_queries=0,
            decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',scope=b['scope'],custody=b['custody'],
            seconds=time.monotonic()-begin, OS_peak=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            output_bytes=sum(e['bytes'] for e in outputs), timing_admitted=False)
        assert result['processed_tokens'] <= b['processed_tokens_bound']
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
        event(operation='completed',quality=q)
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage,
                       completed_tasks=len(tasks),completed_histories=len(rows)); raise
    finally:
        if bridge is not None:
            if bridge.streams is not None:
                for stream in bridge.streams: stream.close()
            bridge.close_contexts()
        receipt.update(seconds=time.monotonic()-begin,OS_peak=proc.memory_info().peak_wset,
                       GPU_allocated_peak=torch.cuda.max_memory_allocated(),
                       GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.out.with_suffix('.worker_terminal.json'),receipt)


def launch(a):
    import psutil
    from original_falcon_whole_recovery import memory_reader
    begin = time.monotonic()
    assert extent(a.binding)['sha256'] == a.binding_sha
    b = load(a.binding)
    assert b['limits'] == LIMITS and not a.directory.exists() and not a.out.exists()
    check_inputs(b)
    env = dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0')
    argv = [b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker',
        '--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,
        '--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
    log = a.out.with_suffix('.worker.log')
    assert not log.exists()
    proc = psutil.Process(); proc.cpu_affinity([11]); reader = memory_reader()
    receipt = dict(schema='FULL_WIDTH_NATIVE_CHAT_HOLDER_V1',binding=extent(a.binding),
        command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,
        worker_argv=argv,timing_admitted=False)
    child = None; peak = 0; offset = 0
    try:
        with log.open('xb') as stream:
            child = subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time())
            write(a.out.with_suffix('.owned_process.json'),receipt)
            while child.poll() is None:
                peak = max(peak,reader(child))
                assert time.monotonic()-begin <= LIMITS['seconds'], 'holder deadline'
                assert peak+proc.memory_info().peak_wset <= LIMITS['OS_bytes'], 'combined OS cap'
                with log.open('rb') as f:
                    f.seek(offset); data = f.read(); end = data.rfind(b'\n')+1
                    if end:
                        print(data[:end].decode('utf8',errors='replace'),end='',flush=True); offset += end
                time.sleep(.5)
        peak = max(peak,reader(child))
        receipt['worker_exit_code'] = child.returncode
        assert child.returncode == 0, log.read_text(encoding='utf8',errors='replace')[-5000:]
        check_inputs(b)
        r = load(a.out)
        assert len(r['tasks']) == 16 and len(r['records']) == 48 and len(r['prefix_controls']) == 16
        receipt.update(exit_code=0,error=None,result=extent(a.out),
                       worker_terminal=extent(a.out.with_suffix('.worker_terminal.json')))
    except BaseException as error:
        if child is not None and child.poll() is None:
            assert abs(psutil.Process(child.pid).create_time()-receipt['worker_creation_time']) < .001
            child.terminate(); child.wait(timeout=15)
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),
                       worker_exit_code=None if child is None else child.returncode); raise
    finally:
        receipt.update(seconds=time.monotonic()-begin,worker_OS_peak=peak,
                       OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('bind','launch','worker'): modes.add_argument('--'+mode,action='store_true')
    for key in ('binding','directory','out'): parser.add_argument('--'+key,type=Path)
    for key in ('binding-sha','freeze'): parser.add_argument('--'+key,required=key == 'freeze')
    args = parser.parse_args()
    if args.bind:
        import psutil
        start = time.monotonic(); proc = psutil.Process()
        terminal = dict(schema='FULL_WIDTH_NATIVE_CHAT_BIND_TERMINAL_V1',command=sys.orig_argv,
                        pid=proc.pid,creation_time=proc.create_time(),freeze=args.freeze)
        try:
            bind(args); terminal.update(exit_code=0,error=None,result=extent(args.out))
        except BaseException as error:
            terminal.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
        finally:
            terminal.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
            write(args.out.with_suffix('.preparation_terminal.json'),terminal)
    elif args.launch: launch(args)
    else: worker(args)
