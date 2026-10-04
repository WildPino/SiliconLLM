"""Actual nested learned64/128 expert payloads, exact338 codes/F32 row subsets."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time
import numpy as np
import psutil
import meth324_switch_reference as M
import meth368_switch_bank_manifest as I

OUT=M.ROOT/'results/native_expert_scaling/meth370_switch_nested_bank_export'
PROTOCOL=M.DOC/'METH_370_SWITCH_NESTED_BANK_EXPORT_PROTOCOL_20261004.md'
RECOVERED=M.DOC/'meth338_switch_tensor_recovery_result.json'
RECOVERED_SHA='19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'
USEFUL=M.DOC/'meth369_switch_bank_usefulness_result.json'
USEFUL_SHA='305e2e8997fc7f0a6fc58626900f44bb5fd70f7890bdbd03c9d089baa184e244'
OLD_EXPORT=M.ROOT/'benchmarks/native_expert_scaling/meth335_switch_w8a8_export.py'
TIED=('shared.weight','encoder.embed_tokens.weight','decoder.embed_tokens.weight','lm_head.weight')


def manifest(config, entries, payload, path):
    values = [config[k] for k in ('d_model', 'd_ff', 'num_heads', 'd_kv', 'num_layers',
              'num_decoder_layers', 'num_experts', 'expert_capacity', 'vocab_size',
              'relative_attention_num_buckets', 'relative_attention_max_distance',
              'encoder_sparse_step', 'decoder_sparse_step')]
    with path.open('xb') as stream:
        stream.write(b'SWI8A001')
        stream.write(struct.pack('<13IfII', *values, config['layer_norm_epsilon'], 1, len(entries)))
        name = str(payload.resolve()).encode('utf-8')
        stream.write(struct.pack('<I', len(name))); stream.write(name)
        for name, entry in sorted(entries.items()):
            text = name.encode('utf-8'); stream.write(struct.pack('<I', len(text))); stream.write(text)
            shape = entry['shape']
            stream.write(struct.pack('<5I3Q', 0, len(shape), shape[0], shape[1] if len(shape) == 2 else 1,
                         entry['encoding'], entry['offset'], entry['scale_offset'], entry['elements']))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum=0;written=0
    result={'experiment':'METH-370-real-nested64-128-bank-payload-export','targets':[]}
    def guard():
        nonlocal maximum
        maximum=max(maximum,psutil.Process().memory_info().rss)
        assert maximum<=16<<30 and time.monotonic()-start<=1200 and written<=16<<30,'subset_export_20min_16GiB_RAM_disk'
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    try:
        for path in (Path(__file__),PROTOCOL,RECOVERED,USEFUL,OLD_EXPORT,Path(M.__file__),Path(I.__file__)):M.committed(path)
        assert digest(RECOVERED)==RECOVERED_SHA and digest(USEFUL)==USEFUL_SHA
        recovered=json.loads(RECOVERED.read_text(encoding='utf-8'));useful=json.loads(USEFUL.read_text(encoding='utf-8'))
        assert all(recovered['gates'].values()) and len(useful['controls'])==2 and all(len(v['cases'])==96 for v in useful['controls'])
        # Exact old writer AST is reused, independently parsed by368 helper.
        old=next(v for v in ast.parse(OLD_EXPORT.read_text(encoding='utf-8')).body if isinstance(v,ast.FunctionDef) and v.name=='manifest')
        new=next(v for v in ast.parse(Path(__file__).read_text(encoding='utf-8')).body if isinstance(v,ast.FunctionDef) and v.name=='manifest')
        assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
        artifact=recovered['artifact'];source=Path(artifact['payload']);before=(source.stat().st_size,source.stat().st_mtime_ns)
        assert before[0]==artifact['bytes'] and digest(source)==artifact['sha256'] and digest(artifact['manifest'])==artifact['manifest_sha256']
        I.read_manifest(artifact['manifest'],recovered['original_config'],recovered['tensors'],source)
        assert psutil.virtual_memory().available>=16<<30 and shutil.disk_usage(M.ROOT).free>=20<<30
        result.update({'controller_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),'recovered338_sha256':RECOVERED_SHA,'usefulness369_sha256':USEFUL_SHA,
                       'source_artifact':artifact,'fixed_expert_subsets':{'64':[0,64],'128':[0,128],'256':[0,256]},'writer335_AST_exact':True,
                       'RAM_bytes':psutil.virtual_memory().total,'available_RAM_before_bytes':psutil.virtual_memory().available,'disk_free_before_bytes':shutil.disk_usage(M.ROOT).free})
        original=recovered['tensors'];assert len(original)==6392 and sum(v['elements'] for v in original.values())-3*32128*768==14664154368
        assert len({original[k]['source_sha256'] for k in TIED})==1
        OUT.mkdir(parents=True)
        for n in (64,128):
            stage=f'export{n}';folder=OUT/f'n{n}';folder.mkdir();target=folder/'weights.bin';spec=folder/'manifest.bin';entries={};cache={};copies=[];padding=[]
            config=copy.deepcopy(recovered['original_config']);config['num_experts']=n
            with source.open('rb') as inp,target.open('xb') as out:
                def emit(offset,size,expected,label):
                    nonlocal written
                    key=(offset,size)
                    if key in cache:
                        placed,sha=cache[key];assert expected is None or sha==expected;return placed,sha
                    pad=(-out.tell())%64
                    if pad:padding.append({'offset':out.tell(),'bytes':pad});out.write(b'\0'*pad);written+=pad
                    placed=out.tell();inp.seek(offset);h=hashlib.sha256();remaining=size
                    while remaining:
                        block=inp.read(min(4<<20,remaining));assert block;assert out.write(block)==len(block)
                        remaining-=len(block);written+=len(block);h.update(block);guard()
                    sha=h.hexdigest();assert expected is None or sha==expected,(label,sha,expected)
                    cache[key]=(placed,sha);copies.append({'label':label,'source_offset':offset,'target_offset':placed,'bytes':size,'sha256':sha})
                    return placed,sha
                for name,oldentry in sorted(original.items()):
                    match=I.PATTERN.fullmatch(name)
                    if match and int(match.group(2))>=n:continue
                    entry=copy.deepcopy(oldentry);entry['source_target_offset']=oldentry['offset'];entry['source_target_scale_offset']=oldentry['scale_offset']
                    router='.mlp.router.classifier.weight' in name
                    if router:
                        assert entry['shape']==[256,768] and entry['encoding']==0 and entry['sha256']==entry['source_sha256']
                        entry['shape']=[n,768];entry['elements']=n*768;entry['bytes']=n*768*4
                        entry['source_parent_sha256']=oldentry['source_sha256'];entry['source_rows']=[0,n]
                        entry['offset'],entry['sha256']=emit(oldentry['offset'],entry['bytes'],None,name)
                        entry['source_sha256']=entry['sha256']
                        inp.seek(oldentry['offset']);assert np.isfinite(np.frombuffer(inp.read(entry['bytes']),dtype='<f4')).all()
                    else:entry['offset'],entry['sha256']=emit(oldentry['offset'],entry['bytes'],oldentry['sha256'],name)
                    if entry['encoding']:
                        entry['scale_offset'],entry['scale_sha256']=emit(oldentry['scale_offset'],entry['scale_bytes'],oldentry['scale_sha256'],name+'.scales')
                    else:entry['scale_offset']=0
                    entries[name]=entry
                assert out.tell()==sum(v['bytes'] for v in copies)+sum(v['bytes'] for v in padding)
            assert len(entries)==24*n+248
            retained=sum(v['elements'] for v in entries.values())-3*32128*768
            assert retained=={64:3790748928,128:7415217408}[n]
            assert len({entries[k]['source_sha256'] for k in TIED})==1
            manifest(config,entries,target,spec);checks=I.read_manifest(spec,config,entries,target)
            stage=f'readback{n}'
            with source.open('rb') as inp,target.open('rb') as actual:
                for region in copies:
                    inp.seek(region['source_offset']);actual.seek(region['target_offset']);remaining=region['bytes'];h=hashlib.sha256()
                    while remaining:
                        size=min(4<<20,remaining);a=inp.read(size);b=actual.read(size);assert len(a)==len(b)==size and a==b
                        h.update(b);remaining-=size;guard()
                    assert h.hexdigest()==region['sha256']
                for gap in padding:
                    actual.seek(gap['offset']);assert actual.read(gap['bytes'])==b'\0'*gap['bytes']
                actual.seek(0,2);assert actual.tell()==target.stat().st_size
            banks={}
            for name,entry in entries.items():
                match=I.PATTERN.fullmatch(name)
                if match:
                    prefix,e,suffix,kind=match.groups();banks.setdefault(prefix,{})[(int(e),kind)]=(entry['sha256'],entry['scale_sha256'])
            assert len(banks)==12 and all(len(v)==2*n for v in banks.values())
            distinct={prefix:len({(values[(e,'wi')],values[(e,'wo')]) for e in range(n)}) for prefix,values in banks.items()};assert all(v==n for v in distinct.values())
            metadata=folder/'tensors.json';M.write(metadata,{'original_config':config,'tensors':entries,'copies':copies,'padding':padding,'retained_original_unique_coefficients':retained})
            record={'n':n,'namespace':len(entries),'original_config':config,'retained_original_unique_coefficients':retained,'original_donor_unique_coefficients':14664154368,
                    'payload':str(target),'bytes':target.stat().st_size,'sha256':digest(target),'manifest':str(spec),'manifest_sha256':digest(spec),
                    'metadata':str(metadata),'metadata_sha256':digest(metadata),'distinct_real_WI_WO_code_scale_pairs_per_bank':distinct,'manifest_checks':checks,
                    'all_retained_ranges_exact338_readback':True,'all_padding_and_EOF_exact':True,'router_prefix_F32_selected_rows':[0,n],
                    'logical_decoder_matrix_bytes_per_position':123764736+534016+18432*n,'physical_RAM_fit':target.stat().st_size<psutil.virtual_memory().total,'passed':True}
            assert record['physical_RAM_fit'];result['targets'].append(record);print(json.dumps({k:record[k] for k in ('n','namespace','retained_original_unique_coefficients','bytes','sha256')}),flush=True)
        assert before==(source.stat().st_size,source.stat().st_mtime_ns)
        result['full256_unchanged_reference']=artifact
        result['gates']={'fresh_source338_complete_identity':True,'both_NEW_real64_128_artifacts_exported':len(result['targets'])==2,
                         'all_retained_codes_scales_F32_exact_and_router_rows_exact':True,'all_namespace_alias_unique_counts_and_pair_distinctness_exact':True,
                         'all_physical_padding_EOF_manifests_complete_hashes_exact':True,'source_payload_unchanged':True,'actual_smaller_payload_RAM_fit':True}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_rss_bytes':maximum,'new_payload_and_padding_bytes_written':written,'available_RAM_end_bytes':psutil.virtual_memory().available,'disk_free_end_bytes':shutil.disk_usage(M.ROOT).free}
        result['decision']='actual_real_nested_bank_artifacts_eligible_for_371_full_native_independent_dynamic_n_contract'
        result['scope']='Actual64/128 learnedexpert subsets from SAME14.664B base256, physicallysmaller payloads, unchangedcore/head/precision/capacity and retainedexpert coefficients/F32routerprefix. Not Google independentlytrainedbase128 or small-n retraining. Export fidelity only; no originalquality/usefuladditionaln/acceptedrate/physicalDRAM proof.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':digest(args.out),'gates':result['gates'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_rss_bytes':maximum,'written_payload_bytes':written});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
