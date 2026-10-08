"""Offline target-aware dimension accounting; no model, weight values or rate claims."""
import argparse
import ctypes
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(entry):
    raw = Path(entry['path']).read_bytes()
    if len(raw) != entry['bytes'] or sha(raw) != entry['sha256']:
        raise ValueError('bound input changed: ' + entry['path'])
    return raw


def integer(c, key):
    v = c[key]
    if type(v) is not int or v <= 0:
        raise ValueError('positive integer required: ' + key)
    return v


def attention(c, layers):
    d = integer(c, 'hidden_size')
    nh, nk = integer(c, 'num_attention_heads'), integer(c, 'num_key_value_heads')
    hd = c.get('head_dim') or d//nh
    if type(hd) is not int or nh*hd != d or nh % nk:
        raise ValueError('unsupported attention dimensions')
    kv = nk*hd
    return layers*(2*d*d+2*d*kv), layers*2*kv, layers*2*d


def ssm2(c, layers):
    d = integer(c, 'hidden_size')
    m = c.get('mamba_d_ssm') or int(c['mamba_expand']*d)
    n, groups, heads = (integer(c, k) for k in ('mamba_d_state', 'mamba_n_groups', 'mamba_n_heads'))
    if m != heads*integer(c, 'mamba_d_head'):
        raise ValueError('SSM2 head conservation failed')
    conv_dim = m+2*groups*n
    # in_proj=[m+conv_dim+heads,D], out_proj=[D,m]; conv is scalar work.
    products = layers*d*(m+conv_dim+heads+m)
    return products, layers*m*n, layers*conv_dim*integer(c, 'mamba_d_conv')


def source(entry):
    c = json.loads(read(entry['config']))
    d, l, v = (integer(c,k) for k in ('hidden_size','num_hidden_layers','vocab_size'))
    kind = c['model_type']
    result = {'model': entry['model'], 'revision': entry['revision'], 'kind': kind,
              'D':d, 'L':l, 'V':v, 'activation':c['hidden_act'],
              'source_weight_values_observed':False, 'precision':c.get('dtype',c.get('torch_dtype')),
              'operator_compatibility_with_original':'adaptation required; no adapter/inference parity admission'}
    if entry['retained_census']:
        old = json.loads(read(entry['retained_census']))['source']
        cats = old['matrix_MACs_by_category_per_decode_token']
        cats = {k:x for k,x in cats.items() if x}
        result.update(products=cats, total_products=sum(cats.values()),
                      source_count_scope='reused qualified census bytes; not a source observation replay',
                      original_named_elements=old['named_elements'], original_payload_bytes=old['payload_bytes'],
                      main_named_elements=old['main_generation_named_elements'],
                      f32_dense_adam_main_lower_bound_bytes=16*old['main_generation_named_elements'])
        result['source_head_products'] = old['head_contract']['full_head_MACs']
        result['missing_transformations'] = ['compact recurrent core', 'ternary small experts', 'structured router', 'packed-only engine loader']
        return result
    products = {'head':v*d}
    att_layers, recurrent_layers = l, 0
    if kind in ('falcon_h1','bitnet'):
        products['ffn'] = 3*l*d*integer(c,'intermediate_size')
        if kind == 'falcon_h1':
            # Bound local decoder code creates and executes both branches in every layer.
            recurrent_layers = l
            products['ssm_projection'], state, conv = ssm2(c,l)
        else:
            state = conv = 0
    elif kind == 'granitemoehybrid':
        types = c['layer_types']
        if len(types)!=l or set(types)-{'mamba','attention'}:
            raise ValueError('unsupported Granite layer type')
        att_layers, recurrent_layers = types.count('attention'), types.count('mamba')
        n, k, h = (integer(c,key) for key in ('num_local_experts','num_experts_per_tok','intermediate_size'))
        if not 0 < k <= n:
            raise ValueError('invalid MoE selection')
        products['routed_ffn'] = 3*l*d*h*k
        products['shared_ffn'] = 3*l*d*integer(c,'shared_intermediate_size')
        products['router'] = l*n*d
        products['ssm_projection'], state, conv = ssm2(c,recurrent_layers)
        result['expert_count_per_layer'], result['selected_per_layer'] = n,k
    else:
        raise ValueError('unsupported metadata family: '+kind)
    products['attention'], cache, per_context = attention(c,att_layers)
    result.update(products=products, total_products=sum(products.values()), source_head_products=v*d,
                  attention_layers=att_layers,recurrent_layers=recurrent_layers,
                  cached_kv_elements_per_context_token=cache,
                  attention_extra_products_per_context_token=per_context,
                  recurrent_state_elements=state, conv_state_elements=conv,
                  source_count_scope='config/operator-derived matrix products, NOT tensor-header completeness',
                  original_named_elements=None, original_payload_bytes=None,
                  matrix_only_f32_dense_adam_lower_bound_bytes=16*sum(products.values()),
                  missing_transformations=['core active-cost reduction','ternary conditional experts','structured router','original engine operator/interaction mapping'])
    if kind == 'bitnet':
        result['format_affinity']='Producer reports native ternary; master BF16 values/cardinality and packed operator semantics not observed here'
        result['missing_transformations']=['compact recurrent core','conditional FFN partition','squared-ReLU/SubLN versus gated-dReLU mapping','structured router','original engine operator/interaction mapping']
    return result


def tree(n, query, beam, fanout):
    # Complete padded routing rows for a bottom-up fanout tree; no expert values.
    nodes, level_nodes, depth = 0,n,0
    while level_nodes > 1:
        level_nodes = (level_nodes+fanout-1)//fanout
        nodes += level_nodes
        depth += 1
    return {'internal_nodes':nodes, 'depth':depth, 'parameters':nodes*fanout*(query+1),
            'score_products_upper':query*fanout*(1+max(0,depth-1)*beam),
            'scope':'proposed trained beam tree, NOT exact search of arbitrary flat router or an implemented LUT router'}


def target(v, dimensions, n):
    d,l,nswa,k,h = (dimensions[x] for x in ('D','L','swa','k','h'))
    dn, state, rank, window, conv = 2*d,96,d//16,128,4
    ssm_l = l-nswa
    core_mat = ssm_l*(3*dn*d+dn*(rank+2*state)+dn*rank)+nswa*4*d*d
    core_scalars = ssm_l*dn*(conv+3+state)+(2*l+1)*d
    head, embedding = v*d,v*d  # Untied native baseline arrays; no free storage alias.
    expert, selected = 3*l*d*h*n,3*l*d*h*k
    scales, active_scales = l*n*(2*h+d),l*k*(2*h+d)
    flat_router = l*n*(d+1)
    tr = tree(n,32,k,32)
    tree_params = l*(d*32+tr['parameters'])
    tree_products = l*(d*32+tr['score_products_upper'])
    fixed = core_mat+core_scalars+head+embedding
    encoded = []
    for num,den,label in [(1,2,'byte_pair'),(1,4,'nibble_pair')]:
        codes, active_codes = expert*num//den,selected*num//den
        if expert*num % den or selected*num % den:
            raise ValueError('unsupported packed alignment')
        encoded.append({'layout':label,'packed_expert_code_bytes':codes,
                        'packed_only_model_coefficients_bytes_flat_router':4*(fixed+scales+flat_router)+codes,
                        'packed_only_model_coefficients_bytes_tree_proposal':4*(fixed+scales+tree_params)+codes,
                        'legacy_loader_model_coefficient_copies_bytes':4*(fixed+scales+flat_router)+5*expert+codes,
                        'selected_expert_code_and_scale_bytes':active_codes+4*active_scales,
                        'logical_decode_coefficients_bytes_tree_proposal':4*(core_mat+core_scalars+head+d+active_scales+tree_products)+active_codes})
    return {**dimensions,'V':v,'n_per_layer':n,'total_expert_functions':l*n,
            'expert_coefficients':expert,'active_expert_coefficients':selected,
            'core_matrix_products':core_mat,'full_head_products':head,
            'flat_router_products':l*n*d,'tree_router_products_upper_proposal':tree_products,
            'tree':tr,'per_decode_matrix_products_upper_tree_proposal':core_mat+head+selected+tree_products,
            'fixed_recurrent_and_swa_cache_payload_bytes':4*(ssm_l*dn*(state+conv)+nswa*2*window*d),
            'f32_dense_adam_all_target_coefficients_bytes':16*(fixed+expert+scales+tree_params),
            'layouts':encoded,'physical_dram_and_speed':'UNMEASURED',
            'scope':'prospective original-Mamba1/SWA shape; all transfer/precision/behavior gates open'}


def peak_bytes():
    if os.name != 'nt':
        return None
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    c=Counters(); c.cb=ctypes.sizeof(c)
    process=ctypes.windll.kernel32.GetCurrentProcess
    process.restype=ctypes.c_void_p
    fn=ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    if not fn(process(),ctypes.byref(c),c.cb):
        raise ctypes.WinError()
    return c.PeakWorkingSetSize


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inputs',required=True,type=Path)
    ap.add_argument('--expected-inputs-sha',required=True)
    ap.add_argument('--source-commit',required=True)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    start=time.monotonic()
    raw=args.inputs.read_bytes()
    if sha(raw)!=args.expected_inputs_sha: raise ValueError('manifest SHA')
    manifest=json.loads(raw)
    if manifest['schema']!='engine-target-triage-inputs-v1': raise ValueError('input schema')
    for e in manifest['support']: read(e)
    read(manifest['acquirer'])
    rows=[source(e) for e in manifest['sources']]
    profiles=[]
    for row in rows:
        for dims in [{'D':256,'L':6,'swa':1,'k':8,'h':128}, {'D':512,'L':12,'swa':2,'k':8,'h':128}]:
            per_expert=3*dims['L']*dims['D']*dims['h']
            for n in (32,128,math.ceil(10_000_000_000/per_expert),math.ceil(100_000_000_000/per_expert)):
                profiles.append({'donor':row['model'],'target':target(row['V'],dims,n)})
    elapsed=time.monotonic()-start
    peak=peak_bytes()
    if elapsed>60 or (peak is not None and peak>256<<20): raise RuntimeError('resource cap')
    result={'schema':'engine-target-triage-result-v1','started_commit':args.source_commit,
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'inputs_sha256':sha(raw),'sources':rows,'prospective_targets':profiles,
            'resource':{'seconds_before_serialization':elapsed,'OS_peak_bytes_before_serialization':peak},
            'scope':'new static target-aware accounting, not model/quality/native/rate or complete peak qualification',
            'limitations':['matrix products omit elementwise/selection/normalization/nonlinear/dispatch cost',
                           'logical coefficients are not actual DRAM traffic',
                           'fixed cache payload is not full runtime workspace',
                           'Adam figures exclude activation/teacher/copy/workspace and are explicit chosen precision models',
                           'large-n target pools contain no trained weights; stored parameters are not useful capacity',
                           'config-derived sources do not establish complete header parameter count or weight values',
                           'tree bounds describe a proposed learned router, not a measured or exact arbitrary flat-router algorithm'],
            'new_weight_value_bytes':0,'model_native_calls':0,'goal_complete':False}
    output=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
    if len(output)>5<<20: raise RuntimeError('output cap')
    with args.out.open('xb') as f: f.write(output)
    print(json.dumps({'out':str(args.out),'sha256':sha(output),'bytes':len(output),'seconds':time.monotonic()-start,
                      'OS_peak_bytes':peak_bytes(),'sources':len(rows),'targets':len(profiles)}))


if __name__=='__main__': main()
