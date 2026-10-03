"""Single source-tensor pass, immutable complete compact payload held in bounded RAM."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psutil
import torch
import meth324_switch_reference as M
import meth327_switch_tensor_binding as T
import meth335_switch_w8a8_export as E
import meth337_switch_export_recovery as R

PROTOCOL=M.DOC/'METH_338_SWITCH_TENSOR_RECOVERY_PROTOCOL_20261003.md'
FAILURE=M.DOC/'meth337_switch_export_recovery_result.failure.json'
FAILURE_SHA='f50d585ddc73f6e24dd4ff4c894d056908b4b6b7473931626a4405f82ada93a6'
OUT=M.ROOT/'results/native_expert_scaling/meth338_switch_tensor_recovery'


def layout(bound,header):
    entries={};aliases={};cursor=0
    for shard in header['shards']:
        names=sorted(name for name,record in header['actual_tensor_headers'].items() if record['shard']==shard['name'])
        for name in names:
            shape=bound['tensors'][name]['shape'];elements=int(np.prod(shape));encoding=int(E.compressed(name,shape))
            entry={'shape':shape,'elements':elements,'encoding':encoding,'source_sha256':bound['tensors'][name]['sha256']}
            if name in E.EMBEDDINGS and aliases:
                alias=next(iter(aliases));other=entries[alias]
                assert shape==other['shape'] and entry['source_sha256']==other['source_sha256']
                entry.update({k:other[k] for k in ('offset','bytes','scale_offset','scale_bytes')});entry['physical_alias']=alias
            else:
                cursor=(cursor+63)//64*64;entry['offset']=cursor;entry['bytes']=elements*(1 if encoding else 4);cursor+=entry['bytes']
                if encoding:
                    cursor=(cursor+63)//64*64;entry['scale_offset']=cursor;entry['scale_bytes']=shape[0]*4;cursor+=entry['scale_bytes']
                else:entry.update({'scale_offset':0,'scale_bytes':0})
            if name in E.EMBEDDINGS:aliases[name]=True
            entries[name]=entry
    assert len(entries)==6392 and set(entries)==set(bound['tensors']) and set(aliases)==E.EMBEDDINGS
    return entries,cursor


def verify_tensor(name,tensor,entry,target):
    assert tensor.dtype==torch.float32 and tensor.is_contiguous() and tensor.device.type=='cpu'
    array=tensor.detach().numpy();assert list(array.shape)==entry['shape'] and np.isfinite(array).all()
    assert E.array_digest(array)==entry['source_sha256'],'actual_source_tensor_identity'
    if entry['encoding']:
        codes,scales=R.quantize(array)
        expected_arrays=[('offset','bytes','sha256',codes),('scale_offset','scale_bytes','scale_sha256',scales)]
    else:
        entry['scale_sha256']=None;expected_arrays=[('offset','bytes','sha256',array)]
    for offset_key,size_key,sha_key,expected in expected_arrays:
        offset=entry[offset_key];size=entry[size_key];assert expected.nbytes==size and offset+size<=target.size
        actual=target[offset:offset+size];sha=E.array_digest(expected)
        assert E.array_digest(actual)==sha,'actual335_target_segment_identity'
        entry[sha_key]=sha
    return entry


def bank_tuples(entries):
    banks={};result={}
    for name,entry in entries.items():
        match=T.EXPERT.fullmatch(name)
        if not match:continue
        stack,layer,slot,label,organ=match.groups();key=f'{stack}.block.{layer}.layer.{slot}'
        banks.setdefault(key,{}).setdefault(int(label),{})[organ]=entry
    assert len(banks)==12
    for key,experts in banks.items():
        assert set(experts)==set(range(256));hashes=[]
        for label,organs in sorted(experts.items()):
            assert set(organs)=={'wi','wo'} and all(v['encoding']==1 for v in organs.values())
            description=[[organ,organs[organ]['shape'],organs[organ]['sha256'],organs[organ]['scale_sha256']] for organ in ('wi','wo')]
            hashes.append(hashlib.sha256(json.dumps(description,separators=(',',':')).encode()).hexdigest())
        assert len(set(hashes))==256,'target_code_scale_tuple_collapse'
        result[key]={'labels':256,'distinct_code_and_scale_tuples':256,'tuple_sha256_by_label':hashes}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum_rss=0
    result={'experiment':'METH-338-single-pass-source-tensor-complete-target-identity','shards':[],'verified_source_target_tensors_so_far':0}
    def guard():
        nonlocal maximum_rss
        rss=psutil.Process().memory_info().rss;maximum_rss=max(maximum_rss,rss)
        assert rss<=32<<30 and time.monotonic()-start<=1200,'single_pass_resource_guard'
    try:
        for path in (Path(__file__),PROTOCOL,FAILURE,R.FAILED,E.BOUND,E.QUALIFIED,T.HEADERS,T.ACQUISITION,
                     Path(M.__file__),Path(T.__file__),Path(E.__file__),Path(R.__file__)):
            M.committed(path)
        assert M.digest(FAILURE)==FAILURE_SHA and M.digest(R.FAILED)==R.FAILED_SHA
        assert M.digest(E.BOUND)==E.BOUND_SHA and M.digest(E.QUALIFIED)==E.QUALIFIED_SHA and M.digest(T.HEADERS)==T.HEADERS_SHA
        bound=json.loads(E.BOUND.read_text());qualified=json.loads(E.QUALIFIED.read_text());acquisition=json.loads(T.ACQUISITION.read_text())
        assert bound['passed'] and all(qualified['gates'].values()) and acquisition['passed']
        assert M.digest(T.ACQUISITION)==bound['acquisition_sha256'] and torch.__version__=='2.6.0+cu124'
        failed335=json.loads(R.FAILED.read_text());failed337=json.loads(FAILURE.read_text())
        assert failed335['shards'][-1]['source_tensors_so_far']==6392 and R.PAYLOAD.stat().st_size==14818015744
        before=R.PAYLOAD.stat();assert psutil.virtual_memory().available>=35<<30
        torch.set_num_threads(1);equivalence=R.equivalence_controls();primitive=E.primitive_controls();assert equivalence['passed'] and primitive['passed']
        R.SAFE_SUBNORMAL_VALUES=0
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),'source_binding_sha256':E.BOUND_SHA,
                       'qualified334_sha256':E.QUALIFIED_SHA,'preserved335_failure_sha256':R.FAILED_SHA,'preserved337_failure_sha256':FAILURE_SHA,
                       'exact_quantizer_equivalence':equivalence,'primitive_controls':primitive,'model':bound['model'],'revision':bound['revision'],
                       'source_identity_policy':'FRESH ALL6392 source tensor byte hashes327 and small config/tokenizer files; old326/335 full archive provenance; no new ZIP envelope checksum'})
        stage='source_sizes_and_small_files'
        for item in acquisition['files']:
            path=Path(item['path']);assert item['complete'] and path.stat().st_size==item['bytes']
            if path.suffix!='.bin':assert T.file_digest(path,start)==item['sha256']
        header=next(v for v in json.loads(T.HEADERS.read_text())['models'] if v['model']==bound['model'])
        config=json.loads((T.SOURCE/'config.json').read_text());entries,expected_size=layout(bound,header)
        assert expected_size==before.st_size and expected_size<=24<<30
        OUT.mkdir(parents=True);stage='immutable_target_RAM_load';target=np.fromfile(R.PAYLOAD,dtype=np.uint8)
        target.setflags(write=False);assert target.size==expected_size;guard()
        payload_sha=E.array_digest(target);result['immutable_target_RAM_sha256']=payload_sha
        stage='physical_source_order_all_tensor_byte_verification';verified=set()
        for shard in header['shards']:
            shard_start=time.monotonic();state=torch.load(T.SOURCE/shard['name'],weights_only=True,mmap=True,map_location='cpu')
            expected_names={name for name,record in header['actual_tensor_headers'].items() if record['shard']==shard['name']}
            assert set(state)==expected_names and not verified.intersection(state)
            physical=sorted(state.items(),key=lambda item:(item[1].data_ptr(),item[0]))
            for name,tensor in physical:
                entries[name]=verify_tensor(name,tensor,entries[name],target);verified.add(name)
                result['verified_source_target_tensors_so_far']=len(verified);guard()
            del tensor,physical,state;gc.collect();guard()
            item={'name':shard['name'],'source_tensors_so_far':len(verified),'seconds':time.monotonic()-shard_start}
            result['shards'].append(item);print(json.dumps(item),flush=True)
        assert verified==set(entries) and len(verified)==6392
        # All padding is part of canonical335 layout and must remain zero too.
        segments=sorted({(v['offset'],v['bytes']) for v in entries.values()}|{(v['scale_offset'],v['scale_bytes']) for v in entries.values() if v['encoding']})
        cursor=0
        for offset,size in segments:
            assert offset>=cursor and not np.any(target[cursor:offset]);cursor=offset+size
        assert cursor==expected_size
        result['target_banks']=bank_tuples(entries);spec=OUT/'manifest.bin';E.manifest(config,entries,R.PAYLOAD,spec)
        assert R.PAYLOAD.stat().st_mtime_ns==before.st_mtime_ns and R.PAYLOAD.stat().st_size==before.st_size
        result['tensors']=entries;result['original_config']=config
        result['artifact']={'payload':str(R.PAYLOAD),'bytes':expected_size,'sha256':payload_sha,'manifest':str(spec),'manifest_sha256':M.digest(spec),
                           'source_unique_parameters':14664154368,'original_experts_per_bank':256,'f32_embedding_alias_names':sorted(E.EMBEDDINGS),
                           'f32_embedding_physical_copies':1,'separate_I8_head_view_from_same_tied_source':True,'new_payload_write_bytes':0,
                           'row_I8_tensor_names':sum(v['encoding']==1 for v in entries.values()),'f32_tensor_names':sum(v['encoding']==0 for v in entries.values())}
        result['safe_subnormal_source_values_bypassed_with_exact_zero_code_proof']=R.SAFE_SUBNORMAL_VALUES
        result['gates']={'exact_quantizer_equivalence':True,'all6392_actual_source_tensor_identities':True,
                         'all_actual_target_codes_f32_scales_and_padding_exact':True,'all12_banks256_distinct_code_scale_tuples':True,
                         'original_embedding_controls_exact':True}
        result['prior_interrupted_main_seconds']=[failed335['main_seconds_excluding_imports'],failed337['main_seconds_excluding_imports']]
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_rss_bytes':maximum_rss,'end_rss_bytes':psutil.Process().memory_info().rss}
        result['scope']='Function parameter identity and actual complete compact representation only. ZIP envelope not freshly rehashed; every source coefficient and small config/tokenizer byte identity checked. Quantization approximate; no native/LUT/useful n/quality/accepted-rate proof. All raw failures retained.'
        result['decision']='eligible_for_prepared336_native_integer_and_full_CPU_cost_qualification'
        guard();M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'artifact':result['artifact'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_rss_bytes':maximum_rss})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
