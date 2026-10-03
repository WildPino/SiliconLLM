"""Full-width additive-weight descriptor; larger banks are analytical only."""
import argparse
import math
from pathlib import Path
import time
import psutil
import meth300_gigachat_vector_lut_cost as H
import meth299_gigachat_block_selection as P
C=P.C
PRIOR=H.DOC/'meth300_gigachat_vector_lut_cost_result.json'
PROTOCOL=H.DOC/'METH_317_GIGACHAT_ADDITIVE_DESCRIPTOR_PROTOCOL_20261003.md'
ENCODED={'mla','dense0','routed','shared'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert not args.out.exists();start=time.monotonic()
    for path in (Path(__file__),PROTOCOL,PRIOR,Path(H.__file__)):P.committed(path)
    assert H.sha(PRIOR)=='9458aec3f6baafbe00eaa266a3ac7da84c43040153cbfee3fc24ea9fd64dcd75'
    assert H.sha(Path(H.__file__))=='986abfee67cc7137bed3aecaf6a32c469d38c61d481b16859275476ce6e92ea4'
    prior=C.read(PRIOR);bindings={};tensors={}
    for kind in ('bf16','q4'):
        binding=prior['source_bindings'][kind]
        _,tensors[kind],bindings[kind]=H.header(Path(binding['path']),binding['file_bytes'])
        assert bindings[kind]['header_sha256']==binding['header_sha256']
    rows=[]
    for original in prior['tensor_rows']:
        name=original['name'];source=tensors['bf16'][name];q4=tensors['q4'][name]
        assert list(source['shape'])==original['gguf_shape'] and list(q4['shape'])==original['gguf_shape']
        assert source['type']==original['source_type'] and q4['type']==original['q4_type']
        row={'name':name,'organ':original['organ'],'shape':original['gguf_shape']}
        if original['organ'] in ENCODED:
            d=original['input_width'];o=original['output_width'];active=original['active_banks'];stored=original['stored_banks']
            assert d%8==0 and original['active_elements']==d*o*active
            row.update({'active_coefficients':d*o*active,'active_banks':active,'stored_banks':stored,
                'active_code_bytes':d*o*active//4,'active_scale_bytes':o*active*4,'active_palette_bytes':active*4096,
                'stored_code_bytes':d*o*stored//4,'stored_scale_bytes':o*stored*4,'stored_palette_bytes':stored*4096,
                'logical_palette_vector_loads':d*o*active//4,'integer_coefficient_products':d*o*active})
        else:
            row.update({'unchanged_active_bytes':original['source_active_bytes'] if original['organ']=='embedding' else original['q4_active_bytes'],
                        'unchanged_stored_bytes':source['bytes'] if original['organ']=='embedding' else q4['bytes']})
        rows.append(row)
    # Independent finite-format and geometry controls, before aggregate reporting.
    assert 8*2//8==2 and 2*256*8==4096
    assert 2*254*63<=32767 and 8960*126*63<2**31
    assert sum(v.get('active_coefficients',0) for v in rows if v['organ']=='routed')==589824000
    assert sum(v.get('active_coefficients',0) for v in rows if v['organ']=='mla')==650051584
    assert sum(v.get('active_code_bytes',0) for v in rows if v['organ']=='routed')==147456000
    scenarios=[]
    for n in (64,640,6400):
        multiplier=n//64
        active=sum(sum(v.get(k,0) for k in ('active_code_bytes','active_scale_bytes','active_palette_bytes','unchanged_active_bytes')) for v in rows)
        stored=sum(sum(v.get(k,0) for k in ('stored_code_bytes','stored_scale_bytes','stored_palette_bytes','unchanged_stored_bytes'))*(multiplier if v['organ'] in ('routed','router') else 1) for v in rows)
        router=sum(v.get('unchanged_active_bytes',0) for v in rows if v['organ']=='router')
        flat=active+router*(multiplier-1)
        # Hypothetical coarse64 then scan n/64 local keys for each four selected parents.
        local_reads=25*4*multiplier*16*2 if n>64 else 0
        local_stored=25*n*16*2 if n>64 else 0
        scenarios.append({'experts_per_layer':n,'actually_sourced_experts_per_layer':64,
            'analytical_only_larger_capacity':n!=64,'selected_experts_per_layer':4,
            'full_flat_router_active_weight_bytes':flat,'full_flat_router_stored_weight_bytes':stored,
            'flat_router_coefficients':25*1536*n,
            'hypothetical_hierarchical_active_weight_bytes':active+local_reads,
            'hypothetical_hierarchical_stored_weight_bytes':stored-router*(multiplier-1)+local_stored,
            'hypothetical_local_key_operand_bytes':local_reads,
            'flat_descriptor_under560MB':flat<=560000000,
            'hierarchical_descriptor_under560MB':active+local_reads<=560000000})
    rss=psutil.Process().memory_info().rss;seconds=time.monotonic()-start
    assert seconds<=120 and rss<=1<<30
    result={'experiment':'METH-317-full-width-two-additive-I8-books-per-eight-weights',
        'controller_sha256':H.sha(Path(__file__)),'protocol_sha256':H.sha(PROTOCOL),'source_headers':bindings,
        'source_payload_read':False,'prior_sha256':H.sha(PRIOR),'tensor_rows':rows,'scenarios':scenarios,
        'format':{'group_width':8,'books':2,'entries_per_book':256,'book_dtype':'I8','book_range':[-63,63],
                  'code_bits_per_coefficient':2,'scale':'one positive F32 per output row','activation':'dynamic S7,[-63,63]',
                  'head':'original Q6_K','router_norm':'original F32','embedding':'original BF16'},
        'decision':'eligible_for_separate_full_geometry_native_decoder_preflight' if scenarios[0]['flat_descriptor_under560MB'] else 'stop_before_native_or_training',
        'resource':{'seconds':seconds,'end_rss_bytes':rss},
        'scope':'Calculated descriptor and integer range only. No learned books/weights, physical DRAM, cache, native latency, quality, new useful experts or accepted rate. Larger banks and hierarchy are unsupported analytical hypotheses. This does not implement AQLM or reuse its quality/rate.'}
    H.write_new(args.out,result)
    print(__import__('json').dumps({'decision':result['decision'],'scenarios':scenarios,'resource':result['resource'],'sha256':H.sha(args.out)}))

if __name__=='__main__':main()
