"""Real pretrained cross-source bank-union identity/applicability; no mutation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct
import time
import psutil
import meth324_switch_reference as M
import meth368_switch_bank_manifest as B

PROTOCOL=M.DOC/'METH_401_PRETRAINED_BANK_UNION_APPLICABILITY_PROTOCOL_20261004.md'
EXPORT={256:('meth338_switch_tensor_recovery_result.json','19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'),128:('meth380_switch_base128_export_result.json','6be77281f7ecca73c47eab7f318fed2cb27699f18071b3f3781ab9bce54881ee')}
SHAPE_FIELDS=('d_model','d_ff','num_heads','d_kv','num_layers','num_decoder_layers','expert_capacity','vocab_size','relative_attention_num_buckets','relative_attention_max_distance','encoder_sparse_step','decoder_sparse_step','dense_act_fn','layer_norm_epsilon','router_bias','router_dtype','num_selected_experts')

def representation(row):
    return (tuple(row['shape']),row['encoding'],row['sha256'],row.get('scale_sha256'))

def duplicate_pairs(rows):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row['signature'])].append((row['source_n'],row['source_expert']))
    return [v for v in groups.values() if len(v)>1]

def compatible(a,b):return all(a[k]==b[k] for k in SHAPE_FIELDS)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start=time.monotonic();peak=0;read_bytes=0;stage='bindings'
    result={'experiment':'METH-401-real-pretrained128-plus256-bank-union-applicability','sources':[],'banks':[]}
    def guard():
        nonlocal peak
        peak=max(peak,psutil.Process().memory_info().rss)
        assert peak<=2<<30 and time.monotonic()-start<=600,'union_10min_2GiB'
    def sha(path):
        nonlocal read_bytes
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while block:=f.read(4<<20):read_bytes+=len(block);h.update(block);guard()
        return h.hexdigest()
    try:
        for p in (Path(__file__),PROTOCOL,Path(M.__file__),Path(B.__file__)):M.committed(p)
        result.update({'controller_sha256':sha(__file__),'protocol_sha256':sha(PROTOCOL),'manifest_parser_sha256':sha(B.__file__)})
        exports={};artifacts={};initial={};union=defaultdict(list);original=defaultdict(list)
        for n,(name,expected) in EXPORT.items():
            stage=f'source{n}';p=M.DOC/name;M.committed(p);assert sha(p)==expected
            export=json.loads(p.read_text(encoding='utf-8'));assert all(export['gates'].values());exports[n]=export
            a=export['artifact'];artifacts[n]=a;payload=Path(a['payload']);initial[n]=(payload.stat().st_size,payload.stat().st_mtime_ns)
            assert initial[n][0]==a['bytes'] and sha(payload)==a['sha256']
            assert sha(a['manifest'])==a['manifest_sha256'];B.read_manifest(a['manifest'],export['original_config'],export['tensors'],payload)
            assert export['original_config']['num_experts']==n
            bank_entries=defaultdict(dict)
            for name,row in export['tensors'].items():
                match=B.PATTERN.fullmatch(name)
                if not match:continue
                prefix,e,_,kind=match.groups();e=int(e);assert 0<=e<n and row['encoding']==1
                assert row['shape']==([3072,768] if kind=='wi' else [768,3072])
                assert row['elements']==2359296 and row['bytes']==2359296 and row['scale_bytes']==row['shape'][0]*4
                assert row['offset']+row['bytes']<=a['bytes'] and row['scale_offset']+row['scale_bytes']<=a['bytes']
                bank_entries[prefix][(e,kind)]=(name,row)
            assert len(bank_entries)==12
            source={'n':n,'model':export['model'],'revision':export['revision'],'export_sha256':expected,'artifact':a,'whole_payload_fresh_hash_exact':True,'manifest_all_name_offset_shape_exact':True,'bank_prefixes':sorted(bank_entries),'fresh_each_segment_rehash':False}
            result['sources'].append(source)
            for prefix,entries in sorted(bank_entries.items()):
                assert len(entries)==2*n
                for e in range(n):
                    wn,w=entries[(e,'wi')];on,o=entries[(e,'wo')]
                    sig=[w['sha256'],w['scale_sha256'],o['sha256'],o['scale_sha256']]
                    row={'source_n':n,'source_expert':e,'source_wi_name':wn,'source_wo_name':on,'signature':sig,
                         'source_original_F32_pair_sha256':[w['source_sha256'],o['source_sha256']]}
                    union[prefix].append(row);original[prefix].append(row)
        assert compatible(exports[256]['original_config'],exports[128]['original_config'])
        bad=dict(exports[128]['original_config']);bad['d_ff']+=8;assert not compatible(exports[256]['original_config'],bad)
        all_distinct=True
        for prefix,rows in sorted(union.items()):
            assert len(rows)==384 and {v['source_n'] for v in rows}=={128,256}
            duplicate=duplicate_pairs(rows);all_distinct&=not duplicate
            injected=rows+[dict(rows[0])];assert duplicate_pairs(injected)
            original_signatures=[tuple(v['source_original_F32_pair_sha256']) for v in rows]
            result['banks'].append({'prefix':prefix,'candidate_slots':384,'distinct_I8_scale_ordered_pairs':384-sum(len(v)-1 for v in duplicate),'duplicate_ordered_pairs':duplicate,'distinct_original_ordered_F32_pairs_by_bound_export_hash':len(set(original_signatures)),'rows':rows})
        # Compare shared pretrained core/control bytes, not only compatible dimensions.
        left=exports[256]['tensors'];right=exports[128]['tensors'];common=sorted(set(left)&set(right));matched=[];different=[];router=[]
        for name in common:
            if B.PATTERN.fullmatch(name):continue
            if '.router.classifier.weight' in name:router.append(name);continue
            assert left[name]['shape']==right[name]['shape'] and left[name]['encoding']==right[name]['encoding']
            (matched if representation(left[name])==representation(right[name]) else different).append(name)
        assert len(router)==12
        result['shared_core_comparison']={'same_shape_nonexpert_nonrouter_names':len(matched)+len(different),'byte_identical_names':matched,'different_names':different,'router_names_different_geometry':router,
                                          'shared_tokenizer_or_hidden_semantic_alignment_verified':False}
        candidate_extra=[v for name,v in right.items() if B.PATTERN.fullmatch(name)]
        additional=sum(v['bytes']+v['scale_bytes'] for v in candidate_extra)
        routed_coeff=sum(v['elements'] for v in candidate_extra);assert routed_coeff==7247757312
        extra_router=sum(right[v]['bytes'] for v in router)
        ram=psutil.virtual_memory()
        result['candidate_ledger']={'base256_payload_bytes':artifacts[256]['bytes'],'additional128_routed_I8_and_scales_bytes':additional,
            'additional128_F32_router_bytes':extra_router,'candidate_stored_bytes_with_both_router_sets':artifacts[256]['bytes']+additional+extra_router,
            'candidate_architecture_parameter_positions_base_plus_extra_routed_plus_router':artifacts[256]['source_unique_parameters']+routed_coeff+extra_router//4,
            'routed_candidate_slots_per_bank':384,'total_routed_slots_12_banks':4608,'new_source_I8_pair_slots_byte_distinct_ALL_banks':all_distinct,
            'selected_one_pair_coefficients_12_banks_unchanged':12*2*768*3072,'selected_one_pair_I8_and_scales_12_banks_bytes':12*(2*768*3072+4*(3072+768)),
            'flat_384_F32_router_addressed_bytes_12_banks':12*384*768*4,'flat_256_F32_router_addressed_bytes_12_banks':12*256*768*4,
            'RAM_total_bytes':ram.total,'RAM_available_bytes':ram.available,'candidate_fits_current_available_RAM':artifacts[256]['bytes']+additional+extra_router<ram.available,
            'candidate_exported_or_executed':False,'effective_384_choice_selector_learned':False,'function_equivalence_canonicalization_or_individual_usefulness_verified':False}
        for n,a in artifacts.items():assert initial[n]==(Path(a['payload']).stat().st_size,Path(a['payload']).stat().st_mtime_ns)
        result['gates']={'both_complete_qualified_payload_hashes_fresh_exact':True,'both_manifest_all_records_exact':True,'ALL12_banks384_byte_distinct_ordered_pairs':all_distinct,'core_operator_shapes_compatible':True,'duplicate_and_geometry_negative_controls_detected':True,'candidate_fits_available_RAM':result['candidate_ledger']['candidate_fits_current_available_RAM']}
        result['decision']='eligible_only_for_cross_source_function_alignment_and_new_selector_transfer_screen' if all(result['gates'].values()) else 'stop_bank_union_before_alignment_or_model_work'
        result['resource']={'seconds':time.monotonic()-start,'maximum_checked_rss_bytes':peak,'actual_file_bytes_read_for_hashes':read_bytes};guard()
        result['scope']='Read-only source applicability. Fresh full target payload hashes match complete verified exports; segment hashes inherited through exact whole binding, not individually freshly rehashed. Ordered WI/WO byte tuples distinct is not canonical function equivalence, hidden-space alignment, useful capacity or quality. Candidate ledger keeps base256 core/head plus128 experts/router; no cloned bytes, export/model/selector/quality/rate/physical DRAM. Same-family new384 candidates are1.5x256, not10x or another-family proof.'
        M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'decision':result['decision'],'gates':result['gates'],'ledger':result['candidate_ledger'],'shared_core_names':[len(matched),len(different)],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'bytes_hashed':read_bytes});M.write(args.out.with_suffix('.failure.json'),result);raise
if __name__=='__main__':main()
