"""Fixed bijective real expert WI/WO identity controls; no model/runtime mutation."""
import re
import struct
from pathlib import Path

CONFIG_KEYS=('d_model','d_ff','num_heads','d_kv','num_layers','num_decoder_layers','num_experts','expert_capacity','vocab_size','relative_attention_num_buckets','relative_attention_max_distance','encoder_sparse_step','decoder_sparse_step')
PATTERN=re.compile(r'^(.*\.mlp\.experts\.expert_)([0-9]+)(\.(wi|wo)\.weight)$')


def remap(entries,n,offset):
    assert offset in (0,1,127) and n>1
    shifted=dict(entries);mapping={};banks={}
    for name,entry in entries.items():
        match=PATTERN.fullmatch(name)
        if not match:continue
        prefix,expert,suffix,kind=match.groups();expert=int(expert);assert 0<=expert<n
        actual=(expert+offset)%n;source=prefix+str(actual)+suffix
        target=entries[source];assert entry['shape']==target['shape'] and entry['encoding']==target['encoding']==1
        shifted[name]=target;mapping[name]=source;banks.setdefault(prefix,{})[(expert,kind)]=actual
    assert mapping and all(len(values)==2*n for values in banks.values())
    for values in banks.values():
        assert all(values[(e,'wi')]==values[(e,'wo')] for e in range(n))
        assert sorted(values[(e,'wi')] for e in range(n))==list(range(n))
        if offset%n:assert all(values[(e,'wi')]!=e for e in range(n))
    assert all(shifted[name]==entry for name,entry in entries.items() if name not in mapping)
    return shifted,mapping


def read_manifest(path,config,entries,payload):
    # Independent binary parser verifies every actual lookup offset/shape and
    # exact bounded EOF, separately from existing335 writer.
    data=Path(path).read_bytes();assert data[:8]==b'SWI8A001';offset=8
    values=struct.unpack_from('<13IfII',data,offset);offset+=64
    assert list(values[:13])==[config[k] for k in CONFIG_KEYS]
    assert struct.pack('<f',values[13])==struct.pack('<f',config['layer_norm_epsilon'])
    assert values[14:]==(1,len(entries))
    def string():
        nonlocal offset
        size=struct.unpack_from('<I',data,offset)[0];offset+=4
        assert 0<size<=1024 and offset+size<=len(data)
        value=data[offset:offset+size].decode('utf-8');offset+=size;return value
    assert string()==str(Path(payload).resolve());names=[]
    for name,entry in sorted(entries.items()):
        actual=string();assert actual==name;names.append(actual)
        record=struct.unpack_from('<5I3Q',data,offset);offset+=44;shape=entry['shape']
        assert record==(0,len(shape),shape[0],shape[1] if len(shape)==2 else 1,entry['encoding'],entry['offset'],entry['scale_offset'],entry['elements'])
    assert names==sorted(set(entries)) and offset==len(data)
    return {'all_names_offsets_scales_shapes_exact':len(names),'bounded_EOF_exact':True}


def consultations(config,source_tokens,decoder_positions,routes,offset):
    records=[];expected=[]
    for stack,layers,step in (('encoder',config['num_layers'],config['encoder_sparse_step']),('decoder',config['num_decoder_layers'],config['decoder_sparse_step'])):
        banks=[layer for layer in range(layers) if step>0 and (layer%step==1 or step==1)]
        if stack=='encoder':expected.extend((stack,layer,t) for layer in banks for t in range(source_tokens))
        else:expected.extend((stack,layer,t) for t in range(decoder_positions) for layer in banks)
    assert len(expected)==len(routes)
    for (stack,layer,token),route in zip(expected,routes):
        chosen,accepted,probability=route;chosen=int(chosen);accepted=int(accepted)
        assert 0<=chosen<config['num_experts'] and accepted in (0,1)
        actual=(chosen+offset)%config['num_experts'] if accepted else None
        prefix=f'{stack}.block.{layer}.layer.{2 if stack=="decoder" else 1}.mlp.experts.expert_'
        records.append({'stack':stack,'layer':layer,'token':token,'router_slot':chosen,'accepted':accepted,'probability':float(probability),
                        'source_expert':actual,'consulted_wi':prefix+str(actual)+'.wi.weight' if accepted else None,'consulted_wo':prefix+str(actual)+'.wo.weight' if accepted else None})
    return records
