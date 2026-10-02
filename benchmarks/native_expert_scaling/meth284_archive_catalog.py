#!/usr/bin/env python3
"""Generate direct C offsets for the fixed complete archive; no tensor conversion."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'results/native_expert_scaling/meth276_qwen05b_i16_private128_diagnostic_repair1.safetensors'
SHA='4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9'
DTYPES={'F32':(1,4,'torch.float32'),'BF16':(2,2,'torch.bfloat16'),
        'I8':(3,1,'torch.int8'),'U16':(4,2,'torch.uint16'),
        'I32':(5,4,'torch.int32'),'F16':(6,2,'torch.float16')}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def build(path):
    assert not path.exists()
    assert digest(ARCHIVE)==SHA
    with ARCHIVE.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
    meta=header.pop('__metadata__')
    assert meta['format']=='M276_DIAGNOSTIC_I16_INPUT_PRIVATE128_UNIQUE_BF16_V1'
    inventory=json.loads(meta['tensor_inventory_json'])
    config=json.loads(meta['config_json'])
    assert len(header)==len(inventory)==725 and set(header)==set(inventory)
    assert (config['hidden_size'],config['num_hidden_layers'],config['intermediate_size'],config['vocab_size'])==(896,24,4864,151936)
    counts=json.loads(meta['actual_function_counts_json'])
    assert len(counts)==24 and sum(counts)==30556
    rows=[];ranges=[]
    for name,item in sorted(header.items()):
        kind,width,torch_type=DTYPES[item['dtype']];shape=item['shape'];begin,end=item['data_offsets']
        assert 1<=len(shape)<=4 and all(type(x) is int and x>0 for x in shape)
        elements=1
        for dim in shape:elements*=dim
        record=inventory[name]
        assert record['shape']==shape and record['dtype']==torch_type and record['bytes']==end-begin==elements*width
        assert begin>=0 and end+size+8<=ARCHIVE.stat().st_size
        ranges.append((begin,end))
        rows.append('    {'+json.dumps(name)+f',{kind},{len(shape)},'+
                    '{'+','.join(map(str,shape+[0]*(4-len(shape))))+'},'+
                    f'{begin+size+8}ULL,{end-begin}ULL'+ '},')
    ordered=sorted(ranges);assert ordered[0][0]==0
    assert all(a[1]==b[0] for a,b in zip(ordered,ordered[1:]))
    assert ordered[-1][1]+size+8==ARCHIVE.stat().st_size
    text=('/* Generated fixed-artifact catalog; offsets refer to the original safetensors file. */\n'
          '#define CC_FIELD_COUNT 725\n'
          f'#define CC_ARCHIVE_BYTES {ARCHIVE.stat().st_size}ULL\n'
          f'#define CC_ARCHIVE_SHA "{SHA}"\n'
          'typedef struct {const char *name; int dtype,rank; uint64_t shape[4],offset,bytes;} CcField;\n'
          'static const CcField cc_fields[CC_FIELD_COUNT]={\n'+'\n'.join(rows)+'\n};\n'
          'static const int cc_function_counts[24]={'+','.join(map(str,counts))+'};\n')
    path.write_text(text,encoding='utf-8',newline='\n')
    return {'archive_sha256':SHA,'fields':725,'payload_bytes':ordered[-1][1],
            'catalog_sha256':digest(path),'generator_sha256':digest(Path(__file__))}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    print(json.dumps(build(ap.parse_args().out)))
