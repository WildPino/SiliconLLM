"""Bound metadata-only operator accounting. No tensor values or inference."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
WIDTH = {'BF16': 2, 'F16': 2, 'F32': 4, 'I8': 1, 'U16': 2, 'I32': 4}
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
FOREIGN = {
    'benchmarks/donor_adaptation/configs/_manifest.json': 'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
    'benchmarks/donor_adaptation/density/build_document_holdout.py': 'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
    'docs/research/RESEARCH_INDEX.md': '99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_once(path, obj):
    raw = (json.dumps(obj, indent=2, allow_nan=False) + '\n').encode('utf8')
    assert len(raw) <= 2 << 20, 'output cap'
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


class Job:
    def __init__(self, args):
        self.start = time.monotonic()
        self.proc = psutil.Process()
        self.proc.cpu_affinity([10])
        self.args = args
        self.inputs = []
        self.hashed_bytes = 0
        self.r = {'scope': 'NEW_STATIC_OPERATOR_COUNTS_NOT_INFERENCE_OR_DRAM',
                  'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'process_instance': {'pid': self.proc.pid, 'create_time_unix': self.proc.create_time()},
                  'source_commit': args.source_commit, 'pipeline_complete': False,
                  'new_tensor_value_bytes_read': 0, 'new_model_or_C_calls': 0}
        assert sys.version_info[:3] == (3, 12, 10) and psutil.__version__ == '7.2.2'
        self.r['runtime'] = dict(python=sys.version, executable=sys.executable,
                                 psutil=psutil.__version__, cpu_affinity=self.proc.cpu_affinity(),
                                 tensor_libraries_imported=False, direct_children_created=0)
        assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
        own = {self.proc.pid, *(p.pid for p in self.proc.parents())}
        inventory = []
        for p in psutil.process_iter(['name', 'cmdline']):
            name, argv = (p.info['name'] or '').lower(), p.info['cmdline'] or []
            if p.pid in own:
                continue
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                inventory.append({'pid': p.pid, 'name': name, 'preserved_publisher': True})
                continue
            assert not name.startswith(('python', 'clang', 'meth')), ('overlap', p.pid, name)
        self.r['process_inventory'] = inventory
        self.timer = threading.Timer(120, self.deadline)
        self.timer.daemon = True
        self.timer.start()

    def resource(self):
        return {'seconds': time.monotonic() - self.start,
                'OS_peak_bytes': self.proc.memory_info().peak_wset,
                'hashed_input_extent_bytes': self.hashed_bytes,
                'counter_scope': 'Actual extents read by this job; no subprocesses or tensor values.',
                'limits_seconds_peak_bytes': [120, 256 << 20]}

    def guard(self):
        assert time.monotonic() - self.start < 120, 'deadline'
        assert self.proc.memory_info().peak_wset <= 256 << 20, 'OS peak'

    def deadline(self):
        try:
            self.r.update(fault='hard deadline', resource=self.resource(), inputs=self.inputs)
            write_once(self.args.out.with_suffix('.failure.json'), self.r)
        finally:
            os._exit(124)

    def read(self, entry):
        path = Path(entry['path']).resolve()
        before = path.stat()
        assert before.st_size == entry['file_bytes'], ('size drift', str(path))
        n = entry['bytes']
        assert 0 <= n <= before.st_size and n <= 64 << 20
        with path.open('rb') as f:
            raw = f.read(n)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        assert len(raw) == n and digest(raw) == entry['sha256'], ('extent SHA', str(path))
        self.hashed_bytes += n
        self.inputs.append(dict(path=str(path), offset=0, bytes=n, file_bytes=before.st_size,
                                file_mtime_ns=before.st_mtime_ns, sha256=digest(raw),
                                whole_file_SHA_verified=n == before.st_size))
        self.guard()
        return raw

    def preserve(self):
        for rel, expected in FOREIGN.items():
            path=ROOT / rel
            self.read(dict(path=str(path),bytes=path.stat().st_size,file_bytes=path.stat().st_size,sha256=expected))
        assert not any(p.is_file() for p in (ROOT / 'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))


def headers(job, entries, index=None):
    tensors, metadata = {}, {}
    for entry in entries:
        raw = job.read(entry)
        length = struct.unpack('<Q', raw[:8])[0]
        assert length + 8 == len(raw), 'exact header extent'
        body = json.loads(raw[8:])
        metadata[Path(entry['path']).name] = body.pop('__metadata__', {})
        ranges = []
        for name, field in body.items():
            assert name not in tensors
            shape = field['shape']
            assert isinstance(shape, list) and all(type(v) is int and v > 0 for v in shape)
            begin, end = field['data_offsets']
            assert type(begin) is int and type(end) is int and 0 <= begin < end
            assert end - begin == math.prod(shape) * WIDTH[field['dtype']]
            if index is not None:
                assert index[name] == Path(entry['path']).name, ('index disagreement', name)
            tensors[name] = dict(field, file=Path(entry['path']).name, payload_bytes=end-begin,
                                 absolute_offset=8+length+begin)
            ranges.append((begin, end))
        ranges.sort()
        assert ranges and ranges[0][0] == 0
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
        assert ranges[-1][1] + 8 + length == entry['file_bytes']
    if index is not None:
        assert set(index) == set(tensors), 'complete index coverage'
    return tensors, metadata


class Ledger:
    def __init__(self, tensors):
        self.tensors = tensors
        self.rows = {}
        self.mac = Counter()

    def take(self, name, shape, cat, fraction=Fraction(1), mat=False, dtype='BF16', path='main'):
        assert name not in self.rows, ('duplicate operation binding', name)
        t = self.tensors[name]
        assert t['shape'] == list(shape) and t['dtype'] == dtype, ('operator shape/type', name, t)
        elements = math.prod(shape)
        active = elements * fraction
        b = t['payload_bytes'] * fraction
        assert active.denominator == b.denominator == 1
        self.rows[name] = dict(category=cat, execution_path=path, shape=t['shape'], dtype=dtype,
                               stored_named_elements=elements, stored_payload_bytes=t['payload_bytes'],
                               active_budget_elements=active.numerator,
                               logical_coefficient_budget_bytes=b.numerator,
                               allocation_fraction=[fraction.numerator, fraction.denominator])
        if mat:
            self.mac[cat] += active.numerator

    def mlp(self, prefix, d, h, cat, fraction=Fraction(1), path='main'):
        for projection, shape in [('gate_proj', (h,d)), ('up_proj', (h,d)), ('down_proj', (d,h))]:
            self.take(prefix+projection+'.weight', shape, cat, fraction, True, path=path)

    def close(self):
        assert set(self.rows) == set(self.tensors), ('unbound names', sorted(set(self.tensors)-set(self.rows)))
        sums = defaultdict(Counter)
        for row in self.rows.values():
            sums[row['category']].update({k: row[k] for k in ('stored_named_elements', 'stored_payload_bytes',
                                                              'active_budget_elements', 'logical_coefficient_budget_bytes')})
        # Independent conservation: original header ranges/dimensions, rather than operator formulas.
        original_elements = sum(math.prod(t['shape']) for t in self.tensors.values())
        original_bytes = sum(t['payload_bytes'] for t in self.tensors.values())
        assert sum(v['stored_named_elements'] for v in sums.values()) == original_elements
        assert sum(v['stored_payload_bytes'] for v in sums.values()) == original_bytes
        return {'categories': dict(sums), 'matrix_MACs_by_category_per_decode_token': dict(self.mac),
                'matrix_MACs_total_per_decode_token': sum(self.mac.values()),
                'named_elements': original_elements, 'payload_bytes': original_bytes,
                'logical_coefficient_payload_budget_bytes_per_decode_token': sum(v['logical_coefficient_budget_bytes'] for v in sums.values()),
                'allocation_scope': 'Routed K/E is a bank-total budget allocation, NOT a per-expert access probability or per-name upper bound. Conditional alias repeats and LUT footprint use explicit upper budgets.',
                'coverage_tensor_names': len(self.rows), 'complete_name_shape_dtype_coverage': True,
                'dimension_and_offset_conservation': True, 'descriptors': self.rows}


def source_ledger(tensors, c):
    l = Ledger(tensors)
    d, layers, nh, v = (c[k] for k in ('hidden_size','num_hidden_layers','num_attention_heads','vocab_size'))
    tied = c['tie_word_embeddings']
    l.take('model.embed_tokens.weight', (v,d), 'embedding_tied_head' if tied else 'embedding',
           Fraction(1) if tied else Fraction(1,v), mat=tied)
    if not tied:
        l.take('lm_head.weight', (v,d), 'head', mat=True)
    l.take('model.norm.weight', (d,), 'norm')
    qwen = c['model_type'] == 'qwen2'
    assert c['hidden_act'] == 'silu'
    if qwen:
        assert c.get('use_sliding_window') is False
        hd = c.get('head_dim', d//nh)
        assert nh*hd == d and nh % c['num_key_value_heads'] == 0
        kv = c['num_key_value_heads']*hd
        cache_elements = layers*2*kv
        att_mac_per_context = layers*nh*2*hd
        rotary_pairs = layers*(nh+c['num_key_value_heads'])*hd//2
        depth = layers
    else:
        assert c['model_type'] == 'deepseek_v3' and not c['attention_bias']
        assert c['scoring_func'] == 'sigmoid' and c['topk_method'] == 'noaux_tc'
        assert c['norm_topk_prob'] is True and c['ep_size'] == 1
        assert c['n_routed_experts'] % c['n_group'] == 0
        assert 0 < c['num_experts_per_tok'] <= c['n_routed_experts']
        qk = c['qk_nope_head_dim']+c['qk_rope_head_dim']
        cache_elements = layers*nh*(qk+c['v_head_dim'])
        att_mac_per_context = cache_elements
        rotary_pairs = layers*(nh+1)*c['qk_rope_head_dim']//2
        depth = layers+c['num_nextn_predict_layers']
    layer_rows = []
    for i in range(depth):
        p = f'model.layers.{i}.'
        mtp = i >= layers
        zero = Fraction(0) if mtp else Fraction(1)
        path = 'MTP_NOT_EXECUTED_BY_MAIN_GENERATION' if mtp else 'main'
        prefixcat = 'mtp_' if mtp else ''
        before = Counter(l.mac)
        for n in ['input_layernorm','post_attention_layernorm']:
            l.take(p+n+'.weight', (d,), prefixcat+'norm', zero, path=path)
        if qwen:
            for n, shape in [('q_proj',(d,d)), ('k_proj',(kv,d)), ('v_proj',(kv,d)), ('o_proj',(d,d))]:
                l.take(p+'self_attn.'+n+'.weight', shape, 'attention', mat=True)
                if n != 'o_proj':
                    l.take(p+'self_attn.'+n+'.bias', (shape[0],), 'attention_bias')
        else:
            a = p+'self_attn.'
            if c['q_lora_rank'] is None:
                l.take(a+'q_proj.weight', (nh*qk,d), prefixcat+'attention', zero, True, path=path)
            else:
                r = c['q_lora_rank']
                l.take(a+'q_a_proj.weight', (r,d), prefixcat+'attention', zero, True, path=path)
                l.take(a+'q_a_layernorm.weight', (r,), prefixcat+'norm', zero, path=path)
                l.take(a+'q_b_proj.weight', (nh*qk,r), prefixcat+'attention', zero, True, path=path)
            r = c['kv_lora_rank']
            for n, shape in [('kv_a_proj_with_mqa',(r+c['qk_rope_head_dim'],d)),
                             ('kv_b_proj',(nh*(c['qk_nope_head_dim']+c['v_head_dim']),r)),
                             ('o_proj',(d,nh*c['v_head_dim']))]:
                l.take(a+n+'.weight', shape, prefixcat+'attention', zero, True, path=path)
            l.take(a+'kv_a_layernorm.weight', (r,), prefixcat+'norm', zero, path=path)
        moe = not qwen and i >= c['first_k_dense_replace'] and i % c['moe_layer_freq'] == 0
        if moe:
            e, k, h = c['n_routed_experts'], c['num_experts_per_tok'], c['moe_intermediate_size']
            l.take(p+'mlp.gate.weight', (e,d), prefixcat+'router', zero, True, path=path)
            l.take(p+'mlp.gate.e_score_correction_bias', (e,), prefixcat+'router_choice_buffer', zero, path=path)
            fraction = zero*Fraction(k,e)
            fused = p+'mlp.experts.gate_up_proj' in tensors
            if fused:
                l.take(p+'mlp.experts.gate_up_proj', (e,2*h,d), prefixcat+'routed_ffn', fraction, True, path=path)
                l.take(p+'mlp.experts.down_proj', (e,d,h), prefixcat+'routed_ffn', fraction, True, path=path)
            else:
                for j in range(e):
                    l.mlp(p+f'mlp.experts.{j}.',d,h,prefixcat+'routed_ffn',fraction,path)
            l.mlp(p+'mlp.shared_experts.',d,h*c['n_shared_experts'],prefixcat+'shared_ffn',zero,path)
            active_h = (k+c['n_shared_experts'])*h if not mtp else 0
            routing = dict(source_experts=e, selected_experts=k, groups=c['n_group'],
                           top_groups=c['topk_group'], all_router_dot_MACs=e*d if not mtp else 0,
                           sigmoid_scores=e if not mtp else 0,
                           choice='sigmoid(logit)+correction; group top2 sum; masked topK',
                           normalized_mass='selected ORIGINAL sigmoid scores / (sum selected + 1e-20)',
                           denominator_terms=k if not mtp else 0,
                           full_E_softmax_denominator=False, routed_scaling_factor=c['routed_scaling_factor'],
                           proposed_target_n=None, exact_search_complexity='all E logits/choice entries; topK algorithm backend-dependent')
        else:
            h = c['intermediate_size']
            l.mlp(p+'mlp.',d,h,prefixcat+'dense_ffn',zero,path)
            active_h = h if not mtp else 0
            routing = None
        if mtp:
            for n, shape in [('eh_proj',(d,2*d)),('embed_tokens',(v,d)),('enorm',(d,)),
                             ('hnorm',(d,)),('shared_head.head',(v,d)),('shared_head.norm',(d,))]:
                l.take(p+n+'.weight',shape,'mtp_auxiliary',Fraction(0),path=path)
        layer_rows.append(dict(layer=i, path=path, routing=routing, active_SiLU_channels=active_h,
                               gate_times_up_products=active_h,
                               matrix_MACs=sum(l.mac.values())-sum(before.values())))
    result = l.close()
    if qwen:
        closed_form = v*d+layers*(2*d*d+2*d*kv+3*d*c['intermediate_size'])
    else:
        qa = nh*qk*d if c['q_lora_rank'] is None else c['q_lora_rank']*(d+nh*qk)
        attention = qa+(c['kv_lora_rank']+c['qk_rope_head_dim'])*d+c['kv_lora_rank']*nh*(c['qk_nope_head_dim']+c['v_head_dim'])+d*nh*c['v_head_dim']
        moe_layers=sum(i>=c['first_k_dense_replace'] and i%c['moe_layer_freq']==0 for i in range(layers))
        closed_form=v*d+layers*attention+(layers-moe_layers)*3*d*c['intermediate_size']+moe_layers*(3*d*c['moe_intermediate_size']*(c['num_experts_per_tok']+c['n_shared_experts'])+c['n_routed_experts']*d)
    assert closed_form == result['matrix_MACs_total_per_decode_token'], 'closed-form source MAC witness'
    result['independent_closed_form_matrix_MACs']=closed_form
    result['closed_form_matrix_MAC_conservation']=True
    result['layers'] = layer_rows
    result['attention'] = dict(cache_elements_per_cached_token_all_base_layers=cache_elements,
                              direct_attention_MACs_per_context_token_all_layers=att_mac_per_context,
                              rotary_pairs_per_decode_token=rotary_pairs,
                              layout='GQA K/V before repeat_kv' if qwen else 'EXPANDED MLA K/V passed to Cache.update in bound local Transformers source',
                              cache_dtype_observed_by_inference=False,
                              cache_formula='batch * context * elements_per_cached_token * scalar_bytes',
                              cache_examples_bytes={str(s): {str(t): s*t*cache_elements for t in [128,2048,8192]} for s in [2,4]},
                              dynamic_attention_MAC_formula='batch * context * direct_attention_MACs_per_context_token')
    if not qwen:
        result['attention']['compressed_MLA_alternative'] = dict(
            elements_per_cached_token=layers*(c['kv_lora_rank']+c['qk_rope_head_dim']),
            actual_bound_backend=False, prerequisites='absorbed projections/latent attention, changed arithmetic, cache/source parity and physical allocation qualification')
    result['head_contract'] = dict(tied_by_config=tied, separately_named_head_present='lm_head.weight' in tensors,
                                   storage_alias_values_verified=False, full_head_MACs=v*d,
                                   head_coefficient_bytes=tensors['model.embed_tokens.weight' if tied else 'lm_head.weight']['payload_bytes'],
                                   embedding_row_read_bytes=d*WIDTH[tensors['model.embed_tokens.weight']['dtype']],
                                   head_output_logits=v, head_bias=False)
    result['logical_forward_coefficient_budget_bytes'] = result['logical_coefficient_payload_budget_bytes_per_decode_token'] + (d*2 if tied else 0)
    return result


def archive_ledger(tensors, metadata, catalog):
    d,h,v,depth = 896,4864,151936,24
    # Independent fixed C catalog comparison; never read/hash archive payload again.
    pattern = r'\{"([^"]+)",(\d+),(\d+),\{([^}]+)\},(\d+)ULL,(\d+)ULL\}'
    fields = re.findall(pattern, catalog)
    assert len(fields) == len(tensors) == 725
    kind = {1:'F32',2:'BF16',3:'I8',4:'U16',5:'I32',6:'F16'}
    for name,k,rank,shape,offset,size in fields:
        t = tensors[name]
        dims = [int(x) for x in shape.split(',')][:int(rank)]
        assert (kind[int(k)],dims,int(offset),int(size)) == (t['dtype'],t['shape'],t['absolute_offset'],t['payload_bytes'])
    counts = [int(x) for x in re.search(r'cc_function_counts\[24\]=\{([^}]+)\}',catalog)[1].split(',')]
    m = next(iter(metadata.values()))
    assert m['diagnostic_only'] == 'true' and m['native_promotion_qualified'] == 'false'
    assert json.loads(m['actual_function_counts_json']) == counts
    l = Ledger(tensors)
    l.take('model.embed_tokens.weight.bf16',(v,d),'embedding_tied_head',mat=True)
    l.take('model.embed_tokens.weight.q',(v,d),'unused_head_proposal',Fraction(0),dtype='I8')
    l.take('model.embed_tokens.weight.scale',(v,),'unused_head_proposal',Fraction(0),dtype='F16')
    l.take('model.norm.weight',(d,),'norm',dtype='F32')
    l.take('ffn.silu_table',(513,),'LUT_footprint_upper',dtype='F32')
    for i in range(depth):
        p=f'model.layers.{i}.'
        for n,shape in [('q_proj',(d,d)),('k_proj',(128,d)),('v_proj',(128,d)),('o_proj',(d,d))]:
            l.take(p+'self_attn.'+n+'.weight',shape,'attention',mat=True)
            if n!='o_proj':
                l.take(p+'self_attn.'+n+'.bias',(shape[0],),'attention_bias',dtype='F32')
        for n in ['input_layernorm','post_attention_layernorm']:
            l.take(p+n+'.weight',(d,),'norm',dtype='F32')
        f=f'ffn.{i}.'
        for n,shape in [('gate',(h,d)),('up',(h,d)),('down',(d,h))]:
            l.take(f+n+'.q',shape,'full_dense_quantized_ffn',mat=True,dtype='I8')
            l.take(f+n+'.scale',(shape[0],),'row_scales',dtype='F32')
        l.take(f+'down.ids',(d,32),'escape_control_indices',dtype='U16')
        l.take(f+'down.escape',(d,32),'escape_ffn',mat=True)
        for n in ['bias','residual.bias']:
            l.take(f+n,(d,),'ffn_bias',dtype='F32')
        l.take(f+'residual.right',(32,h),'residual32',mat=True)
        l.take(f+'residual.left',(d,32),'residual32',mat=True)
        l.take(f+'private.ids',(128,),'private_control_indices',dtype='U16')
        for n in ['gate','up']:
            l.take(f+'private.'+n,(128,d),'private_recompute_in_addition',mat=True)
        b=f'bank.{i}.'
        l.take(b+'router',(58880,),'factor_router',mat=True,dtype='F32')
        l.take(b+'child_projection',(32,d),'child_query',mat=True,dtype='F32')
        l.take(b+'child_keys',(128,10,32),'selected_parent_child_keys',Fraction(4,128),True,dtype='F32')
        l.take(b+'a',(128,8,d),'conditional_A',Fraction(4,128),True)
        l.take(b+'b',(counts[i],d,8),'conditional_B',Fraction(4,counts[i]),True)
        l.take(b+'leaf_map',(1280,),'selected_alias_indices',Fraction(4,1280),dtype='I32')
    result=l.close()
    closed_form=v*d+depth*(2*d*d+2*d*128+3*d*h+2*128*d+d*32+32*h+d*32+58880+32*d+4*10*32+2*4*8*d)
    assert closed_form == result['matrix_MACs_total_per_decode_token'], 'closed-form C-loop MAC witness'
    result['independent_closed_form_matrix_MACs']=closed_form
    result['closed_form_matrix_MAC_conservation']=True
    result['logical_forward_coefficient_budget_bytes'] = result['logical_coefficient_payload_budget_bytes_per_decode_token']+d*2
    result.update(catalog_matches_all_shapes_dtypes_offsets=True,
                  archive_payload_freshly_verified=False,
                  retained_full_archive_SHA='4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9',
                  actual_profile='meth285_model_operator.h -> meth284_source_operator.h; used by 295/296',
                  dense_active_features_per_layer=h, private_recomputed_features_per_layer=128,
                  distinct_stored_leaf_slots_from_catalog=sum(counts),
                  value_distinctness_reused_from_276_not_reverified=True,
                  cache=dict(scalar_bytes=4,elements_per_cached_token=depth*2*128,
                             formula='2 * 24 * maxseq * 128 * 4; actual C float arrays',
                             capacity_4096_bytes=depth*2*4096*128*4),
                  router=dict(parent_axes=[8,16],axis_topK=4,candidate_products=16,chosen_parents=4,
                              child_keys_per_selected_parent=10,chosen_children_per_parent=1,
                              mass='exp over four selected product scores only, then BF16 rounding',
                              source_router_equivalent=False,full128_parent_softmax=False,
                              full1280_child_mass_normalized=False,proposed_larger_n_qualified=False),
                  notes=['Full gate/up integer dots computed even for private rows; 128 BF16 rows overwrite afterwards.',
                         'Down I8 loop includes zero escape holes; 32 BF16 escapes are additional work.',
                         'B consulted upper counts four executions even if resolved aliases repeat.',
                         'Unused Q8/F16 head remains loaded in archive; actual full head reads tied BF16 embedding.',
                         'LUT bytes are full footprint upper, not exact per-token index trace.'])
    return result


def run(job):
    args=job.args
    bp=args.binding.resolve()
    raw=bp.read_bytes()
    assert digest(raw)==args.expected_binding_sha
    job.hashed_bytes += len(raw)
    job.inputs.append(dict(path=str(bp),offset=0,bytes=len(raw),file_bytes=len(raw),sha256=digest(raw),whole_file_SHA_verified=True))
    binding=json.loads(raw)
    assert binding['schema']=='CHATBOT_OPERATOR_BINDING_V1'
    for e in binding['common']:
        job.read(e)
    donor=binding['donors'][args.donor]
    preflight=json.loads(job.read(donor['preflight']))
    config=json.loads(job.read(donor['config']))
    assert config==preflight['config']
    weight_map=json.loads(job.read(donor['index']))['weight_map'] if donor.get('index') else None
    tensors,_=headers(job,donor['headers'],weight_map)
    result=source_ledger(tensors,config)
    old=preflight['tensor_inventory']
    assert (result['named_elements'], result['payload_bytes'],len(tensors)) == (
        old['named_parameter_elements'],sum(old['value_bytes_by_dtype'].values()),old['tensor_names'])
    mtp=sum(math.prod(t['shape']) for n,t in tensors.items() if re.match(r'model\.layers\.(\d+)\.',n) and int(n.split('.')[2])>=config['num_hidden_layers'])
    assert mtp==old['layers_at_or_beyond_base_depth_elements']
    result['main_generation_named_elements']=result['named_elements']-mtp
    result['MTP_named_elements']=mtp
    job.r.update(model_id=preflight['model_id'],declared_revision=preflight['declared_revision'],
                 revision_remote_verified=False,binding_SHA=args.expected_binding_sha,source=result)
    if args.donor=='qwen':
        a=binding['qwen_archive']
        tensors,meta=headers(job,[a['header']])
        catalog=job.read(a['catalog']).decode('utf8')
        job.r['existing_engine_artifact']=archive_ledger(tensors,meta,catalog)
    job.preserve()
    job.guard()
    job.r.update(decision='COMPLETE_STATIC_ACCOUNTING_PIPELINE_NOT_QUALIFIED',
                 inputs=job.inputs,resource=job.resource(),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 limitations=['Coefficient bytes are logical payload/address upper counts, not measured DRAM transactions.',
                              'Named elements include buffers/indices/copies; numeric uniqueness and intelligence are not inferred.',
                              'Matrix MACs exclude elementwise nonlinear/reduction, softmax/topK, quantization, packing and control cycles.',
                              'Dynamic attention adds the reported context term; prefill and accepted decoding clocks are unmeasured.',
                              'Source values unchanged assertion is limited to bound headers/file sizes; no fresh whole source SHA.',
                              'No compact conversion, quality, accepted50 or large-n admission follows from static counts.'])
    write_once(args.out,job.r)
    job.timer.cancel()
    print(json.dumps({'output':str(args.out),'decision':job.r['decision'],'resource':job.r['resource']}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--expected-binding-sha',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--donor',choices=['qwen','gigachat'],required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    job=None
    try:
        job=Job(args)
        run(job)
    except BaseException as error:
        if job is not None:
            job.timer.cancel()
            job.r.update(fault=repr(error),inputs=job.inputs,resource=job.resource())
            write_once(args.out.with_suffix('.failure.json'),job.r)
        raise
