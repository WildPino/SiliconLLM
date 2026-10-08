"""Independent saved-byte adoption of one native compact export.

Uses integer RTNE on ALL finite trained F32 coefficients; never imports the
exporter, evaluates a model, reruns a codec producer or compiles native code.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def header(path):
    path=Path(path)
    with path.open('rb') as stream:
        first=stream.read(8);assert len(first)==8;size=struct.unpack('<Q',first)[0]
        assert 0<size<=1<<20;data=stream.read(size);assert len(data)==size
    fields=json.loads(data);metadata=fields.pop('__metadata__',{});base=size+8;ranges=[]
    for name,value in fields.items():
        assert value['dtype'] in ('BF16','F32') and 1<=len(value['shape'])<=3
        count=1
        for dimension in value['shape']:assert type(dimension) is int and dimension>0;count*=dimension
        begin,end=value['data_offsets'];width=2 if value['dtype']=='BF16' else 4
        assert type(begin) is int and type(end) is int and 0<=begin<end and end-begin==count*width
        assert (base+begin)%width==0;ranges.append((begin,end))
    ranges.sort();assert ranges and ranges[0][0]==0 and all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
    assert base+ranges[-1][1]==path.stat().st_size
    return fields,metadata,base


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)));torch=None
    r=dict(schema='QWEN_COMPACT_EXPORT_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),procedure_gates={},
        new_original_BF16_full_forwards=0,new_student_full_forwards=0,new_backward_calls=0,new_optimizer_updates=0,
        new_endpoint_queries=0,native_builds_or_forwards=0)
    def guard():
        assert time.monotonic()-start<=180 and proc.memory_info().peak_wset<=4<<30,'codec audit time/OS'
        assert torch is None or not torch.cuda.is_initialized(),'CPU-only codec audit'
        assert not proc.children(recursive=True),'unexpected codec audit descendants'
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        binding=json.loads(args.binding.read_bytes());assert binding['job']['name']=='compact_export_audit'
        for item in binding['inputs']:
            path=Path(item['path']);assert str(path.resolve())==item['resolved_path']
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path'];guard()
        def no_subprocess(event,arguments):
            if event=='subprocess.Popen':raise RuntimeError('Codec audit forbids subprocess')
        sys.addaudithook(no_subprocess)
        import numpy as np
        import torch as torch_module
        torch=torch_module;assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6'
        torch.set_num_threads(6);torch.set_num_interop_threads(1)
        assert not args.directory.exists();args.directory.mkdir()
        export=json.loads(Path(binding['export_result_path']).read_bytes());manifest=export['export']
        archive=Path(manifest['archive']['path']);fields,metadata,base=header(archive)
        source=Path(binding['source_weights']);original,_,source_base=header(source)
        assert len(fields)==435 and archive.stat().st_size-base==693597568
        assert metadata==manifest['metadata'] and metadata['format']=='QWEN_CHAT_COMPACT_SHARED_SELECTED_BF16_GUD_F32_ROUTER_V1'
        actual=np.memmap(archive,mode='r',dtype='u1');donor=np.memmap(source,mode='r',dtype='u1');checked=set()
        def region(name):
            field=fields[name];begin,end=field['data_offsets'];return actual[base+begin:base+end]
        def digest(value):return hashlib.sha256(memoryview(value)).hexdigest()
        core_elements=0
        for name,field in original.items():
            if '.mlp.' in name:continue
            assert fields[name]==field or (fields[name]['dtype']==field['dtype'] and fields[name]['shape']==field['shape'])
            begin,end=field['data_offsets'];assert np.array_equal(region(name),donor[source_base+begin:source_base+end])
            assert fields[name]['dtype']=='BF16';core_elements+=(end-begin)//2;checked.add(name);guard()
            r['adoption_progress']=dict(core_fields=len(checked),core_elements=core_elements,trained_elements=0,router_bytes=0)
        assert len(checked)==218 and core_elements==180246400
        fit=json.loads(Path(binding['fit_result_path']).read_bytes());encoded_elements=0;router_bytes=0
        for item in fit['final_checkpoints']:
            values=torch.load(item['path'],map_location='cpu',weights_only=True);li=item['layer']
            prefix=f'model.layers.{li}.mlp.compact.'
            for name in ('projection','parent_centers','parent_norms'):
                value=values[name];assert value.dtype==torch.float32 and value.is_contiguous()
                array=value.numpy();assert fields[prefix+name]['shape']==list(array.shape) and fields[prefix+name]['dtype']=='F32'
                assert np.array_equal(region(prefix+name),array.view('u1').reshape(-1));router_bytes+=array.nbytes;checked.add(prefix+name)
            assert torch.equal(values['child_centers'][:,0],values['parent_centers'])
            assert torch.equal(values['child_norms'][:,0],values['parent_norms'])
            for name in ('shared_g','shared_u','shared_b','leaf_g','leaf_u','leaf_b'):
                value=values[name];assert value.dtype==torch.float32 and value.is_contiguous()
                array=value.numpy();bits=array.view('<u4');assert np.isfinite(array).all()
                # Exact round-to-nearest, ties-to-even for finite IEEE F32.
                # Integer arithmetic is independent of Torch BF16 conversion.
                rounding=np.uint32(0x7fff)+((bits>>np.uint32(16))&np.uint32(1))
                expected=((bits+rounding)>>np.uint32(16)).astype('<u2')
                assert not ((expected&np.uint16(0x7fff))>=np.uint16(0x7f80)).any()
                assert fields[prefix+name]['shape']==list(array.shape) and fields[prefix+name]['dtype']=='BF16'
                assert np.array_equal(region(prefix+name),expected.view('u1').reshape(-1))
                encoded_elements+=array.size;checked.add(prefix+name);guard()
                r['adoption_progress']=dict(fields=len(checked),core_elements=core_elements,trained_elements=encoded_elements,router_bytes=router_bytes)
            del values
        assert encoded_elements==165150720 and router_bytes==2803200
        assembled=json.loads(Path(binding['installed_manifest_path']).read_bytes())
        assert fields['native.rope.inv_freq']['dtype']=='F32' and fields['native.rope.inv_freq']['shape']==[32]
        assert digest(region('native.rope.inv_freq'))==assembled['buffers']['model.rotary_emb.inv_freq']['sha256']
        checked.add('native.rope.inv_freq');assert checked==set(fields)
        # Parse every generated catalog entry independently of catalog_text.
        text=Path(manifest['catalog']['path']).read_text(encoding='utf8')
        expression=r'^\s*\{"([^"]+)",(1|2),(1|2|3), \{([0-9,]+)\}, ([0-9]+)ULL, ([0-9]+)ULL\},$'
        entries=re.findall(expression,text,re.MULTILINE);assert len(entries)==435
        seen=set()
        for name,kind,rank,shape,offset,length in entries:
            assert name not in seen;seen.add(name);field=fields[name];begin,end=field['data_offsets']
            assert int(kind)==(1 if field['dtype']=='F32' else 2) and int(rank)==len(field['shape'])
            assert list(map(int,shape.split(',')))==field['shape']+[0]*(3-len(field['shape']))
            assert int(offset)==base+begin and int(length)==end-begin
        assert seen==set(fields) and f'#define CB_ARCHIVE_SHA "{sha(archive)}"' in text
        assert f'#define CB_ARCHIVE_BYTES {archive.stat().st_size}ULL' in text
        for name,count in [('D',896),('L',24),('V',151936),('P',16),('Q',32),('HS',512),('HF',128),('NH',14),('NKV',2),('HD',64),('FIELD_COUNT',435)]:
            assert re.search(rf'^#define CB_{name} {count}$',text,re.MULTILINE)
        source_dir=Path(binding['source_directory']);assert len(export['interaction_files'])==6
        names={'source_config.json':'config.json','generation_config.json':'generation_config.json',
               'tokenizer_config.json':'tokenizer_config.json','tokenizer.json':'tokenizer.json','vocab.json':'vocab.json','merges.txt':'merges.txt'}
        for item in export['interaction_files']:
            path=Path(item['path']);assert path.name in names and sha(path)==sha(source_dir/names[path.name])
        r.update(core_BF16_elements_checked=core_elements,RTNE_BF16_GUD_elements_checked=encoded_elements,
            exact_F32_router_bytes_checked=router_bytes,fields_and_catalog_entries_checked=435,
            decision='COMPACT_EXPORT_BYTES_INDEPENDENTLY_VERIFIED',
            scope='ALL coefficient/core/router/catalog/source-interaction byte adoption; codec diagnostic energies, decoded/native quality and rate unqualified')
        r['procedure_gates'].update(all_core_source_bytes=True,all_GUD_independent_integer_RTNE=True,
            all_NEW_F32_router_bytes=True,installed_RoPE_bytes=True,complete_header_and_catalog=True,
            complete_source_interaction_files=True,no_model_optimizer_compiler_endpoint_or_exporter_execution=True)
        guard();r.update(compute_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat());write_once(args.out,r)
        print(json.dumps(dict(decision=r['decision'],encoded_elements=encoded_elements,seconds=r['compute_seconds'])),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat());write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
