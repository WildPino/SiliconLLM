"""Pinned official Switch metadata and inferred decode ledger; no weights."""
import time
TOTAL_START=time.monotonic()
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import time
import requests
import psutil
import torch
import transformers
from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersTop1Router
import meth303_compact_i8_preflight as M
import meth299_gigachat_block_selection as P
MODELS=(('google/switch-base-128','86c815ec05361a33a8b49fc717277da9c0a4e711','pytorch_model.bin.index.json'),
        ('google/switch-base-256','cdac1724c078ea4974b4087c59634799561e7835','pytorch_model.bin.index.json'),
        ('google/switch-large-128','f5573ca2d15edb5ebc609438693db5bbf2e24f58','model.safetensors.index.json'))
DOC=M.DOC;PROTOCOL=DOC/'METH_321_SWITCH_METADATA_COMPATIBILITY_PROTOCOL_20261003.md'
OUT=M.OUT/'meth320_switch_metadata'
FAILURE=DOC/'meth320_switch_metadata_screen_result.failure.json'
FAILURE_SHA='f5cfa26e823e1369765795259cb0c34c300237344976a7b01a51508a43a580fd'
ASSET_PINS={
 'google_switch-base-128_config.json':'b932aa6e954dfcfe647c60ef457eb14f878aca681dc6987f44616f00368d28b7',
 'google_switch-base-128_pytorch_model.bin.index.json':'8bfab7baaab848f12572bd8199a6e289b3dcbf86e9cb63aa84b8a6e35c3b8c5e',
 'google_switch-base-256_config.json':'7e9e2a8bb723b5755e8ab94dfa6178b62907a2345d93a2f080f9238e6fde2778',
 'google_switch-base-256_pytorch_model.bin.index.json':'346ce23272114432696d1624955469a345032c942fe060c765c14e9b4752d51f',
 'google_switch-large-128_config.json':'8ecf96926b7083259e9691115c79c2284acef0e8baa7a344e283bf6428a883cf',
 'google_switch-large-128_model.safetensors.index.json':'4464fa31c4f68c27f9c9aa76e2d3d7a2dda753624f58a03cbb57e011bb623e9a'}

def fetch(model,revision,name,start):
    filename=model.replace('/','_')+'_'+name;path=OUT/filename;raw=path.read_bytes()
    assert len(raw)<=4*(1<<20) and M.digest(path)==ASSET_PINS[filename] and time.monotonic()-start<=300
    return json.loads(raw),{'url':f'https://huggingface.co/{model}/resolve/{revision}/{name}',
        'path':str(path),'bytes':len(raw),'sha256':M.digest(path)}

def active_cost(rows,precision):
    weight=0;scales=0;products=0
    for row in rows:
        shape=row['shape'];count=row['elements'];fixed=row['fixed_f32']
        if fixed or precision=='FP32':weight+=count*4
        elif precision=='BF16':weight+=count*2
        else:weight+=count;scales+=shape[0]*4
        if not fixed:products+=count
    return {'weight_bytes':weight,'row_scale_bytes':scales,'addressed_weight_bytes':weight+scales,
            'large_matrix_coefficients':products,'under560MB_descriptor':weight+scales<=560000000,
            'precision_unverified':precision!='FP32'}

def analyze(model,revision,config,index,bindings):
    assert config['model_type']=='switch_transformers' and config['is_encoder_decoder']
    cfg=SwitchTransformersConfig(**config)
    assert config.get('num_selected_experts',1)==1 and cfg.dense_act_fn=='relu' and not cfg.is_gated_act
    assert cfg.num_experts in (128,256) and cfg.router_dtype=='float32'
    with torch.device('meta'):network=SwitchTransformersForConditionalGeneration(cfg)
    routers_built=[v for v in network.modules() if type(v) is SwitchTransformersTop1Router]
    assert len(routers_built)==cfg.num_sparse_encoder_layers+cfg.num_sparse_decoder_layers and len(routers_built)>0
    assert all(v.num_experts==cfg.num_experts for v in routers_built)
    state=network.state_dict();names=set(index['weight_map']);expected=set(state)
    missing=sorted(expected-names);extra=sorted(names-expected)
    shapes={name:tuple(value.shape) for name,value in state.items()}
    assert not missing and not extra,('official_index_current_builtin_namespace_mismatch',missing[:20],extra[:20])
    assert all(value.device.type=='meta' for value in state.values())
    unique=sum(value.numel() for value in network.parameters())
    indexed=sum(state[name].numel() for name in names)
    declared=index['metadata']['total_size'];possible=[unique*2,unique*4,indexed*2,indexed*4]
    # Serialized aliases/dtypes are not inferred beyond this consistency check.
    assert declared in possible,('index_total_size_not_explained_by_F32_BF16_alias_accounting',declared,possible)
    active=[];prefill_only=[];source_sparse=set();norms=0
    for name in sorted(names):
        shape=shapes[name];count=state[name].numel()
        if '.experts.expert_' in name:
            source_sparse.add(name.split('.experts.')[0])
            if '.experts.expert_0.' not in name:continue
        if name.startswith('encoder.') or name.startswith('shared.') or name=='lm_head.weight':continue
        if not name.startswith('decoder.'):raise AssertionError(('unclassified_source_tensor',name))
        if name=='decoder.embed_tokens.weight':continue
        fixed=len(shape)<2 or '.router.' in name or 'relative_attention_bias' in name
        row={'name':name,'shape':list(shape),'elements':count,'fixed_f32':fixed}
        if '.EncDecAttention.k.' in name or '.EncDecAttention.v.' in name:prefill_only.append(row)
        else:active.append(row)
    assert 'shared.weight' in state and cfg.tie_word_embeddings
    head_shape=shapes['shared.weight'];assert head_shape==(cfg.vocab_size,cfg.d_model)
    active.append({'name':'shared.weight-as-tied-full-logit-head','shape':list(head_shape),
                   'elements':state['shared.weight'].numel(),'fixed_f32':False})
    ledgers={precision:active_cost(active,precision) for precision in ('FP32','BF16','W8')}
    # Embedding row is additionally consulted each generated token, not a new head.
    for precision,binding in ledgers.items():
        binding['embedding_row_bytes']=cfg.d_model*({'FP32':4,'BF16':2,'W8':1}[precision])+(4 if precision=='W8' else 0)
        binding['addressed_weight_bytes']+=binding['embedding_row_bytes']
        binding['under560MB_descriptor']=binding['addressed_weight_bytes']<=560000000
    routers=[v for v in active if '.router.' in v['name']]
    stored=[]
    for precision in ('FP32','BF16','W8'):
        byte_count=0;scales=0
        for name,value in network.named_parameters():
            shape=tuple(value.shape);n=value.numel();fixed=len(shape)<2 or '.router.' in name or 'relative_attention_bias' in name
            byte_count+=n*(4 if fixed or precision=='FP32' else 2 if precision=='BF16' else 1)
            if precision=='W8' and not fixed:scales+=shape[0]*4
        stored.append({'precision':precision,'unique_weight_bytes':byte_count,'row_scale_bytes':scales,'total_bytes':byte_count+scales})
    result={'model':model,'revision':revision,'bindings':bindings,'official_config':config,
        'unique_parameter_count_inferred':unique,'index_entry_parameter_count_inferred':indexed,
        'index_total_size':declared,'actual_weights_or_function_uniqueness_verified':False,
        'top1_proof':{'explicit_config_selected_experts':config.get('num_selected_experts'),'exact_builtin_router_class':'SwitchTransformersTop1Router','router_modules':len(routers_built)},
        'index_namespace_matches_current_builtin':True,'tensors':{name:list(shape) for name,shape in sorted(shapes.items())},
        'decode_active_rows':active,'decoder_cross_KV_prefill_only_rows':prefill_only,
        'num_source_sparse_layers':len(source_sparse),'decode_router_coefficients':sum(v['elements'] for v in routers),
        'decode_ledgers':ledgers,'unique_storage_scenarios':stored,
        'FP32_cross_attention_cached_KV_bytes_at_source128':2*128*cfg.d_model*cfg.num_decoder_layers*4,
        'FP32_self_attention_cached_KV_bytes_at_decode128':2*128*cfg.d_model*cfg.num_decoder_layers*4,
        'decision':'eligible_for_separate_actual_source_binding_and_reference_semantics' if ledgers['BF16']['under560MB_descriptor'] else 'active_cost_screen_closed_for_unchanged_BF16_geometry',
        'limits':'Shape/alias counts inferred from official index names and installed built-in meta graph, not tensor-header or weight verification. Full encoder and cross-KV prefill, cache reads/attention/context growth, source router capacity and weighted top1 semantics still require implementation/quality/cost. FP32/BF16/W8 descriptor is not native speed; non-FP32 quality unverified.'}
    del state,network;gc.collect();return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start=TOTAL_START;stage='bindings';result={'experiment':'METH-321-Switch-explicit-top1-meta-proof','models':[]}
    try:
        for path in (Path(__file__),PROTOCOL,FAILURE):P.committed(path)
        assert M.digest(FAILURE)==FAILURE_SHA
        previous=json.loads(FAILURE.read_text())
        result['preserved320_failure_sha256']=FAILURE_SHA
        assert M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='f1eab2d3e02e70d93d20d6fb07dd3acd20519fe6f88a57d7a1b214bc5b305ecd'
        assert M.digest(inspect.getfile(SwitchTransformersConfig))=='4a8446dcc0e5b95388f51532be67d012895b16c77fc425335ea4f0205c80744f'
        result.update({'controller_sha256':M.digest(Path(__file__)),'protocol_sha256':M.digest(PROTOCOL),
            'libraries':{'torch':torch.__version__,'transformers':transformers.__version__},
            'builtin_model_source_sha256':M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration)),
            'builtin_config_source_sha256':M.digest(inspect.getfile(SwitchTransformersConfig)),
            'source_weight_payload_downloaded_bytes':0})
        OUT.mkdir(exist_ok=True)
        for model,revision,index_name in MODELS:
            stage=model+'_metadata';config,cb=fetch(model,revision,'config.json',start);index,ib=fetch(model,revision,index_name,start)
            stage=model+'_meta_graph_and_ledger';entry=analyze(model,revision,config,index,[cb,ib]);result['models'].append(entry)
            old=next((v for v in previous['models'] if v['model']==model),None)
            if old:
                for key in old:assert entry[key]==old[key],('completed320_metadata_must_match',model,key)
            entry['exact_completed320_case_match']=old is not None
            rss=psutil.Process().memory_info().rss;assert rss<=2*(1<<30) and time.monotonic()-start<=300
            print(json.dumps({'model':model,'unique_parameters_inferred':entry['unique_parameter_count_inferred'],
                'decode_ledgers':entry['decode_ledgers'],'decision':entry['decision']}),flush=True)
        result['scope']='Official immutable metadata/namespace and inferred full decoder-active descriptor only. No source weight acquisition, measured native cost, quality, useful expert uniqueness or completed pretrained transfer. Encoder-decoder infilling use does not replace final goal.'
        result['resource']={'seconds':time.monotonic()-start,'end_rss_bytes':psutil.Process().memory_info().rss}
        M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'seconds':time.monotonic()-start})
        M.write(args.out.with_suffix('.failure.json'),result);raise

if __name__=='__main__':main()
