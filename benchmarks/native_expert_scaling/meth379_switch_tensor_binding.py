"""Whole acquired source binding: tensor bytes, tied copies, real expert tuples."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import time
import psutil
import torch
import numpy as np
import meth324_switch_reference as M

PROTOCOL=M.DOC/'METH_379_SWITCH_TENSOR_BINDING_PROTOCOL_20261004.md'
HEADERS=M.DOC/'meth325_switch_source_headers_result.json'
HEADERS_SHA='f49bd77012045ce2ba2d1ad20d29f86833d229674a644ee0d250e622c39d71e8'
ACQUISITION=M.DOC/'meth378_switch_base128_acquisition_result.json'
ACQUISITION_SHA='89add8d24541423dfa61827f785bf125c2062cf1aa6976214f03da94b7c582b7'
SOURCE=M.ROOT/'results/native_expert_scaling/meth378_switch_base128_source'
EXPERT=re.compile(r'^(encoder|decoder)\.block\.(\d+)\.layer\.(\d+)\.mlp\.experts\.expert_(\d+)\.(wi|wo)\.weight$')


def file_digest(path,start):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(4<<20):
            h.update(block)
            assert time.monotonic()-start<=1200
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start=time.monotonic();stage='bindings';maximum_rss=0
    result={'experiment':'METH-379-full-real-Switch128-tensor-binding','shards':[],'tensors':{}}
    try:
        for path in (Path(__file__),PROTOCOL,HEADERS,ACQUISITION,Path(M.__file__)):M.committed(path)
        assert M.digest(HEADERS)==HEADERS_SHA and M.digest(ACQUISITION)==ACQUISITION_SHA
        acquisition=json.loads(ACQUISITION.read_text(encoding='utf-8'))
        assert acquisition['passed'] and acquisition['manifest_sha256']==HEADERS_SHA
        assert acquisition['model']=='google/switch-base-128' and acquisition['revision']=='86c815ec05361a33a8b49fc717277da9c0a4e711'
        header=next(v for v in json.loads(HEADERS.read_text(encoding='utf-8'))['models'] if v['model']==acquisition['model'])
        assert header['passed']
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),
            'headers_sha256':HEADERS_SHA,'acquisition_sha256':M.digest(ACQUISITION),
            'libraries':{'torch':torch.__version__,'numpy':np.__version__},
            'model':acquisition['model'],'revision':acquisition['revision']})
        assert torch.__version__=='2.6.0+cu124'
        torch.set_num_threads(1)
        for shard in header['shards']:
            stage=shard['name'];path=SOURCE/stage
            acquired=next(v for v in acquisition['files'] if v['name']==stage)
            assert acquired['complete'] and acquired['bytes']==shard['bytes'] and acquired['sha256']==shard['lfs_sha256']
            assert path.stat().st_size==shard['bytes']
            sha=file_digest(path,start);assert sha==shard['lfs_sha256']
            state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            expected={name for name,v in header['actual_tensor_headers'].items() if v['shard']==stage}
            assert set(state)==expected
            elements=0
            for index,(name,tensor) in enumerate(state.items()):
                original=header['actual_tensor_headers'][name]
                assert type(tensor) is torch.Tensor and tensor.device.type=='cpu' and tensor.dtype==torch.float32
                assert list(tensor.shape)==original['shape'] and list(tensor.stride())==original['stride']
                assert tensor.is_contiguous()
                array=tensor.numpy();view=memoryview(array).cast('B');h=hashlib.sha256()
                finite=True;minimum=None;maximum=None
                for offset in range(0,len(view),4<<20):
                    block=view[offset:offset+(4<<20)];h.update(block)
                    values=np.frombuffer(block,dtype='<f4')
                    finite=finite and bool(np.isfinite(values).all())
                    low=float(values.min());high=float(values.max())
                    minimum=low if minimum is None else min(minimum,low)
                    maximum=high if maximum is None else max(maximum,high)
                assert finite,('nonfinite_source',name)
                elements+=tensor.numel()
                result['tensors'][name]={'shape':list(tensor.shape),'dtype':'F32','elements':tensor.numel(),
                    'bytes':len(view),'sha256':h.hexdigest(),'minimum':minimum,'maximum':maximum,'shard':stage}
                del array,view,block,values
                if index%64==0:
                    rss=psutil.Process().memory_info().rss;maximum_rss=max(maximum_rss,rss)
                    assert rss<=16<<30 and time.monotonic()-start<=1200
            del tensor,state
            gc.collect()
            result['shards'].append({'name':stage,'bytes':path.stat().st_size,'sha256':sha,'actual_tensor_count':len(expected),
                                      'serialized_elements':elements})
            print(json.dumps({'completed':stage,'tensors':len(expected),'seconds':time.monotonic()-start,'maximum_checked_rss_bytes':maximum_rss}),flush=True)
        assert set(result['tensors'])==set(header['actual_tensor_headers']) and len(result['tensors'])==3320
        alias_names=('shared.weight','encoder.embed_tokens.weight','decoder.embed_tokens.weight','lm_head.weight')
        aliases=[result['tensors'][name] for name in alias_names]
        assert all(v['shape']==aliases[0]['shape'] and v['sha256']==aliases[0]['sha256'] for v in aliases)
        total_elements=sum(v['elements'] for v in result['tensors'].values())
        unique_architecture_elements=total_elements-3*aliases[0]['elements']
        assert total_elements*4==header['declared_index_bytes'] and unique_architecture_elements==7415217408
        banks={}
        for name,tensor in result['tensors'].items():
            match=EXPERT.fullmatch(name)
            if match:
                stack,block,layer,expert,organ=match.groups()
                key=f'{stack}.block.{block}.layer.{layer}.mlp'
                banks.setdefault(key,{}).setdefault(int(expert),{})[organ]=tensor
        assert len(banks)==12
        result['expert_banks']=[]
        for name,bank in sorted(banks.items()):
            assert set(bank)==set(range(128))
            entries=[]
            for expert,organs in sorted(bank.items()):
                assert set(organs)=={'wi','wo'}
                assert organs['wi']['shape']==[3072,768] and organs['wo']['shape']==[768,3072]
                canonical={'operation':'ReLU(WI*x), then WO','dtype':'F32','wi_shape':organs['wi']['shape'],
                           'wo_shape':organs['wo']['shape'],'wi_sha256':organs['wi']['sha256'],'wo_sha256':organs['wo']['sha256']}
                parameter_tuple_sha=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                entries.append({'expert':expert,'parameter_tuple_sha256':parameter_tuple_sha,
                    'wi_sha256':organs['wi']['sha256'],'wo_sha256':organs['wo']['sha256']})
            result['expert_banks'].append({'name':name,'source_labels':128,
                'distinct_parameter_tuples':len({v['parameter_tuple_sha256'] for v in entries}),'experts':entries,
                'effective_function_or_useful_capacity_verified':False})
        result.update({'tied_aliases_exact':True,'serialized_parameters':total_elements,
            'unique_architecture_parameters_after_confirmed_ties':unique_architecture_elements,
            'original_expert_pair_parameters':sum(v['elements'] for name,v in result['tensors'].items() if EXPERT.fullmatch(name)),
            'same_family_scale_transfer_candidate_only':True,
            'resource':{'seconds':time.monotonic()-start,'maximum_checked_rss_bytes':maximum_rss,
                        'end_rss_bytes':psutil.Process().memory_info().rss},
            'passed':True,'decision':'real_source_bound_for_separate_full_reference_and_native_conversion',
            'scope':'Every complete source shard rehashed against LFS; actual all F32 tensor bytes/shapes/finite values hashed, four tied copies equal. Expert hashes distinguish canonical parameter tuples, not mathematically distinct functions (e.g. hidden permutations/scalings) or useful knowledge. Source operator/whole-model controls, tokenizer, independent quality, native precision/routing/LUT/DRAM and same-artifact accepted rate remain.'})
        assert time.monotonic()-start<=1200 and maximum_rss<=16<<30
        M.write(args.out,result)
        print(json.dumps({'sha256':M.digest(args.out),'resource':result['resource'],'banks':[(v['name'],v['distinct_parameter_tuples']) for v in result['expert_banks']],'passed':True}),flush=True)
    except BaseException as error:
        result.update({'error':repr(error),'stage':stage,'seconds':time.monotonic()-start,'maximum_checked_rss_bytes':maximum_rss})
        M.write(args.out.with_suffix('.failure.json'),result)
        raise


if __name__=='__main__':main()
