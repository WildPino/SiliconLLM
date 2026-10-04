"""Bounded original Ling-mini immutable headers and source-specific active ledger."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time

import psutil
import requests
import meth324_switch_reference as M

MODELS = (
 ('ibm-granite/granite-3.1-1b-a400m-base','408b6e90baab8cf24f4aa9f8e19703ffa0a53b29',(1024,512,32,8,24,16,8,49152)),
 ('ibm-granite/granite-3.1-3b-a800m-base','d4dd87aa3a6c201bc374851d7d7ff4cf39a0b82a',(1536,512,40,8,32,24,8,49152)))
PROTOCOL = M.DOC / 'METH_411_GRANITE_MOE_SOURCE_HEADERS_PROTOCOL_20261004.md'
OUT = M.ROOT / 'results/native_expert_scaling/meth411_granite_moe_source_headers'
PREVIOUS = M.DOC/'meth410_ling_static_pair_lut_result.json'
PREVIOUS_SHA = 'bd345a1b3239c5f319432541059006fb94cb6a002593e52d80890a0058707f87'
REFERENCE = M.ROOT/'results/native_expert_scaling/meth324_switch_reference/venv/Lib/site-packages/transformers/models/granitemoe'
REFERENCE_HASHES = {'modeling_granitemoe.py':'6bd8829f07902a061168d0f600774b95fca9382aa9e102a6c1def37f95b614ed',
                    'configuration_granitemoe.py':'6d930c9775bc203cfd55f6cdf7c637f4ad762bb0b6326d03d7ee35be7832c5f5'}
DTYPE_BYTES = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'I64': 8, 'I32': 4, 'I16': 2, 'U8': 1, 'I8': 1, 'BOOL': 1}
def analyze(config,tensors,geometry):
    d,h,n,k,layers,heads,kv,vocab=geometry
    assert tuple(config[x] for x in ('hidden_size','intermediate_size','num_local_experts','num_experts_per_tok','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size'))==geometry
    assert config['architectures']==['GraniteMoeForCausalLM'] and config['model_type']=='granitemoe'
    assert config['hidden_act']=='silu' and not config['attention_bias'] and config['tie_word_embeddings']
    assert config['attention_multiplier']==.015625 and config['embedding_multiplier']==12 and config['residual_multiplier']==.22 and config['logits_scaling']==6
    assert config['rms_norm_eps']==1e-6 and config['rope_scaling'] is None and config['max_position_embeddings']==131072
    assert config['bos_token_id']==config['eos_token_id']==config['pad_token_id']==0
    expected={'model.embed_tokens.weight':[vocab,d],'model.norm.weight':[d]}
    banks=[];qkvd=kv*(d//heads);core=router=norms=0;core_rows=0
    for l in range(layers):
        p=f'model.layers.{l}.'
        for name,shape in [('q_proj',[d,d]),('k_proj',[qkvd,d]),('v_proj',[qkvd,d]),('o_proj',[d,d])]:
            expected[p+'self_attn.'+name+'.weight']=shape;core+=math.prod(shape);core_rows+=shape[0]
        for name in ('input_layernorm','post_attention_layernorm'):expected[p+name+'.weight']=[d];norms+=d
        expected[p+'block_sparse_moe.router.layer.weight']=[n,d];router+=n*d
        expected[p+'block_sparse_moe.input_linear.weight']=[n,2*h,d]
        expected[p+'block_sparse_moe.output_linear.weight']=[n,d,h]
        banks.append({'layer':l,'source_slots':n,'packed_input_shape':[n,2*h,d],'packed_output_shape':[n,d,h],
                      'stored_routed_coefficients':3*n*d*h,'selected_routed_coefficients':3*k*d*h})
    norms+=d
    # Missing lm_head is the standard serialized tied alias; if stored, equality
    # cannot be verified from metadata, and must be checked on actual values.
    alias='lm_head.weight' in tensors
    if alias:expected['lm_head.weight']=[vocab,d]
    assert set(tensors)==set(expected),('unexpected_missing_tensor_names',sorted(set(tensors)-set(expected)),sorted(set(expected)-set(tensors)))
    for name,shape in expected.items():assert tensors[name]['shape']==shape and tensors[name]['dtype']=='BF16',(name,tensors[name])
    routed_stored=sum(x['stored_routed_coefficients'] for x in banks);routed_active=sum(x['selected_routed_coefficients'] for x in banks)
    head=vocab*d;named=sum(x['elements'] for x in tensors.values())
    architectural_unique=core+routed_stored+router+norms+head
    assert named==architectural_unique+(head if alias else 0)
    active_coded=core+routed_active;active_rows=core_rows+layers*k*(2*h+d)
    # ONE proposed all-row I8/F32-scale variant, BF16 lookup/F32 controls.
    active={'coded_core_and_selected_expert_I8_bytes':active_coded,'coded_row_F32_scale_bytes':active_rows*4,
            'full_head_I8_bytes':head,'head_row_F32_scale_bytes':vocab*4,
            'all_router_F32_bytes':router*4,'norm_F32_bytes':norms*4,'one_BF16_lookup_row_bytes':d*2}
    active['descriptor_bytes']=sum(active.values());active['under560MB']=active['descriptor_bytes']<=560000000
    stored={'coded_core_and_all_expert_I8_bytes':core+routed_stored,'coded_row_F32_scale_bytes':(core_rows+layers*n*(2*h+d))*4,
            'head_I8_bytes':head,'head_row_F32_scale_bytes':vocab*4,'full_lookup_BF16_bytes':head*2,'router_and_norm_F32_bytes':(router+norms)*4}
    stored['total_bytes']=sum(stored.values())
    return {'config_geometry':geometry,'all_named_parameters':named,'architecture_unique_parameters_conditional_on_declared_tie':architectural_unique,
            'stored_lm_head_alias':alias,'tied_value_equality_verified':False,
            'categories':{'attention':core,'routed_stored':routed_stored,'router':router,'norms':norms,'shared_original_embedding_and_head':head},
            'banks':banks,'active_parameters_with_head_and_one_lookup_row':active_coded+head+router+norms+d,
            'hypothetical_row_I8_active':active,'hypothetical_row_I8_storage':stored,
            'original_slot_function_distinctness_and_usefulness_verified':False}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; stage = 'bindings'
    result = {'experiment': 'METH-411-Granite-MoE-two-source-immutable-headers-and-I8-cost-eligibility', 'requests': [], 'models': []}
    bodies = 0

    def guard():
        nonlocal peak
        peak = max(peak, psutil.Process().memory_info().rss)
        assert peak <= 1 << 30 and time.monotonic()-start <= 600 and bodies <= 64 << 20, 'metadata_10min_1GiB_64MiB'

    def get(url, maximum, byte_range=None, total=None):
        nonlocal bodies
        guard(); assert len(result['requests']) < 250
        headers = {'Accept-Encoding': 'identity'}
        if byte_range: headers['Range'] = f'bytes={byte_range[0]}-{byte_range[1]}'
        entry = {'url': url, 'range': byte_range}; result['requests'].append(entry)
        with requests.get(url, headers=headers, stream=True, timeout=(15,30)) as response:
            entry['status'] = response.status_code; response.raise_for_status()
            if byte_range:
                assert response.status_code == 206, ('range_ignored', response.status_code)
                assert response.headers.get('Content-Range') == f'bytes {byte_range[0]}-{byte_range[1]}/{total}', ('range_mismatch', response.headers.get('Content-Range'))
            data = bytearray()
            for block in response.iter_content(1 << 16):
                bodies += len(block); data.extend(block); guard(); assert len(data) <= maximum, 'response_body_limit'
            raw = bytes(data)
            if byte_range: assert len(raw) == byte_range[1]-byte_range[0]+1
            entry.update({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
            return raw

    try:
        for p in (Path(__file__),PROTOCOL,PREVIOUS,Path(M.__file__)):M.committed(p)
        assert M.digest(PREVIOUS)==PREVIOUS_SHA and all(json.loads(PREVIOUS.read_text(encoding='utf-8'))['gates'].values())
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),
                       'previous410_sha256':PREVIOUS_SHA,'source_weight_value_payload_downloaded_bytes':0})
        result['reference_modules']=[]
        OUT.mkdir(parents=True)
        for name,expected_hash in REFERENCE_HASHES.items():
            source=REFERENCE/name;raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected_hash
            (OUT/name).write_bytes(raw);result['reference_modules'].append({'name':name,'sha256':expected_hash,'path':str(source),'stored_copy':str(OUT/name),'executed':False,'version':'Transformers4.57.6'})
        # Native library presence is not original-runtime or generation qualification.
        for number,(model,revision,geometry) in enumerate(MODELS):
            where=OUT/f'source{number}';where.mkdir();entry={'model':model,'revision':revision,'shards':[],'assets':[]};result['models'].append(entry)
            stage=f'{model}:api_config_index'
            api_raw=get(f'https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true',4<<20)
            (where/'api.json').write_bytes(api_raw);api=json.loads(api_raw);assert api['sha']==revision
            siblings={x['rfilename']:x for x in api['siblings']};root=f'https://huggingface.co/{model}/resolve/{revision}/'
            for name,maximum in (('config.json',1<<20),('model.safetensors.index.json',16<<20),
                                 ('generation_config.json',1<<20),('tokenizer.json',4<<20),('tokenizer_config.json',1<<20),('special_tokens_map.json',1<<20)):
                raw=get(root+name,maximum);assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==siblings[name]['blobId']
                (where/name).write_bytes(raw);entry['assets'].append({'name':name,'path':str(where/name),'sha256':hashlib.sha256(raw).hexdigest(),'git_blob_id':siblings[name]['blobId'],'bytes':len(raw)})
            config=json.loads((where/'config.json').read_text(encoding='utf-8'))
            index=json.loads((where/'model.safetensors.index.json').read_text(encoding='utf-8'))
            names=sorted(set(index['weight_map'].values()));assert len(names)==2
            assert set(names)=={x for x in siblings if x.endswith('.safetensors')}
            tensors={};dtype_parameters=Counter();archives=0
            for name in names:
                stage=f'{model}:{name}';source=siblings[name];lfs=source['lfs'];size=lfs['size'];archives+=size
                file_sha=lfs.get('sha256') or lfs.get('oid');assert re.fullmatch('[0-9a-f]{64}',file_sha)
                prefix=get(root+name+'?download=true&metadata_range=0-7',8,(0,7),size)
                length=struct.unpack('<Q',prefix)[0];assert 2<=length<=4<<20 and 8+length<size
                raw=get(root+name+f'?download=true&metadata_range=8-{7+length}',length,(8,7+length),size)
                saved=where/(name+'.header.json');saved.write_bytes(raw);header=json.loads(raw)
                items={k:v for k,v in header.items() if k!='__metadata__'}
                assert set(items)=={k for k,v in index['weight_map'].items() if v==name}
                intervals=[]
                for tensor,item in items.items():
                    assert tensor not in tensors and item['dtype'] in DTYPE_BYTES
                    shape=item['shape'];assert all(isinstance(v,int) and v>=0 for v in shape)
                    elements=math.prod(shape);a,b=item['data_offsets'];assert 0<=a<=b<=size-8-length
                    assert b-a==elements*DTYPE_BYTES[item['dtype']]
                    intervals.append((a,b));dtype_parameters[item['dtype']]+=elements
                    tensors[tensor]={'shape':shape,'dtype':item['dtype'],'elements':elements,'shard':name,'data_offsets':[a,b]}
                position=0
                for a,b in sorted(intervals):assert a==position;position=b
                assert position+8+length==size
                entry['shards'].append({'name':name,'declared_LFS_size':size,'declared_LFS_sha256':file_sha,
                    'header_bytes':length,'header_sha256':hashlib.sha256(raw).hexdigest(),'header_path':str(saved),'tensor_names':len(items),'all_offsets_exact':True})
                guard()
            assert set(tensors)==set(index['weight_map']);count_bytes=sum(v*DTYPE_BYTES[k] for k,v in dtype_parameters.items())
            assert count_bytes==index['metadata']['total_size'];stage=f'{model}:actual_operator_and_active_analysis'
            analysis=analyze(config,tensors,geometry);assert sum(dtype_parameters.values())==analysis['all_named_parameters']
            entry.update({'tensors':tensors,'source_parameters_by_dtype':dict(dtype_parameters),'source_tensor_value_bytes':count_bytes,
                          'source_shard_bytes':archives,'source_config':config,'analysis':analysis})
            entry['decision']='eligible_only_for_full_I8_cpu_cost_and_original_operator_contract' if analysis['hypothetical_row_I8_active']['under560MB'] else 'unchanged_full_row_I8_active_geometry_closed_before_values'
            print(json.dumps({'model':model,'actual_tensor_names':len(tensors),'named_parameters':analysis['all_named_parameters'],
                              'I8_descriptor':analysis['hypothetical_row_I8_active']['descriptor_bytes'],'decision':entry['decision']}),flush=True)
        token_names=('tokenizer.json','tokenizer_config.json','special_tokens_map.json')
        result['shared_tokenizer_bytes_exact']={name:(OUT/'source0'/name).read_bytes()==(OUT/'source1'/name).read_bytes() for name in token_names}
        assert all(result['shared_tokenizer_bytes_exact'].values())
        result['hardware']={'RAM_total_bytes':psutil.virtual_memory().total,'RAM_available_bytes':psutil.virtual_memory().available}
        result['gates']={'both_configs_indexes_and_tokenizer_assets_git_blob_bound':True,'ALL_four_header_names_offsets_dtypes_extents_exact':True,
                         'ALL24x32_and32x40_packed_expert_shapes_exact':True,'ALL_source_parameter_and_I8_ledgers_reconciled':True,
                         'both_tokenizer_asset_bytes_exact':True,'qualified_environment_Granite_modules_SHA_stored_not_executed':True}
        result['decision']='source0_I8_native_cost_candidate_source1_requires_separate_cost_representation'
        result['resource']={'seconds':time.monotonic()-start,'maximum_checked_rss_bytes':peak,'response_body_bytes':bodies,'http_requests':len(result['requests'])}
        result['scope']='Immutable actual Granite3.1-MoE two-source API/config/index/tokenizer/ALL safetensors headers only. No actual weights/finite value check/tied value equality/ordered expert distinctness/usefulness/original inference/loader/backend qualification/native conversion/quality/rate/physical DRAM. Local Transformers4.57.6 modules stored/hash-bound NOT imported/executed, presence is not runtime contract. I8/A16 with F32 scales/control and BF16 lookup ledgers hypothetical; tied original tensor produces separate head/lookup precision copies. Native costs unmeasured. This is another-family operator/cost applicability, not useful>256/10x or replacement of real~10B/~100B final scope.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'decision':result['decision'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage, 'error':repr(error), 'seconds':time.monotonic()-start, 'response_body_bytes':bodies})
        M.write(args.out.with_suffix('.failure.json'),result); raise


if __name__ == '__main__': main()
