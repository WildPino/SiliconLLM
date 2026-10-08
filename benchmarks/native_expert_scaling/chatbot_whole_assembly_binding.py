"""Bind actual whole assembly and its FIRST CPU installed-byte/count audit."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry,receipt
from chatbot_null_prior_kernel_binding import checked


def main(args):
    assert not args.out.exists();doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    source_binding_path=doc/'chatbot_source_batch4_binding_20261007.json'
    assert sha(source_binding_path)=='b595f2945dc80db21c4e399eadff33b1a83f41cf7074b49064c72c2e1048571e'
    source_binding=json.loads(source_binding_path.read_bytes())
    old_path=doc/'chatbot_joint_baseline_binding_20261007.json'
    assert sha(old_path)=='a3265dbeceaf7de82a903bf7735beba88628d7814be34fb1a46d459424b00b5b'
    old=json.loads(old_path.read_bytes());source_dir=Path(source_binding['source_directory'])
    source=source_dir/'model.safetensors';assert sha(source)=='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
    source_inputs=[v for v in source_binding['inputs'] if Path(v['path']).parent==source_dir];assert len(source_inputs)==7
    for value in source_inputs:assert entry(value['path'])==value
    init_path=doc/'chatbot_balanced_initializer_20261008.json'
    initial,init_term,it=checked(init_path,'61a0ee0b668302f825f90edbd596b411c6d673ff0bdccd65e48836a953e4f452')
    audit_path=doc/'chatbot_balanced_initializer_audit_20261008.json'
    audit,audit_term,at=checked(audit_path,'1943947142b11943a356a4a7742d4231bba2abf2a59fca9e0631626ceeba5d12')
    assert initial['all24_source_initialized_and_NEW_supported'] and audit['all24_source_initialized_and_NEW_supported']
    assert audit['decision']=='BALANCED_ALL24_INITIALIZER_INDEPENDENTLY_VERIFIED'
    assert audit['total_NEW_source_copy_elements_checked']==151388160 and audit['independently_verified_balanced_construction_steps']==480
    init_binding_path=doc/'chatbot_balanced_initializer_binding_20261008.json'
    audit_binding_path=doc/'chatbot_balanced_initializer_audit_binding_20261008.json'
    assert sha(init_binding_path)=='044aff4df33a5fcb9e22e2a5fc732ad539f8aa9fc003812ec17694678f0af77d'
    assert sha(audit_binding_path)=='40dbfda06e6d7aadc689f5992b24568c1ecd8d612a4dde5f30ef3bb1369cd7c9'
    assert Path(at['command'][at['command'].index('--binding')+1]).resolve()==audit_binding_path.resolve()
    audit_binding=json.loads(audit_binding_path.read_bytes())
    assert next(v for v in audit_binding['inputs'] if Path(v['path'])==init_path)['sha256']==sha(init_path)
    closure=doc/'chatbot_balanced_initializer_terminal_20261008.json'
    assert sha(closure)=='5a5ad15f4338639752a818dc12b9837c59322816d1c0bb09d4e16a58f3287e7e'
    spec=code/'chatbot_compact_spec.json';assert sha(spec)=='a5850df0172d9f9152e8391a3588ed5cfe9fb184d049955a161e28b4786cd9d0'
    paths=[source_binding_path,old_path,old['python'],Path(old['python']).with_name('python312.dll'),
        *(v['path'] for v in source_inputs),spec,init_path,init_term,audit_path,audit_term,init_binding_path,audit_binding_path,closure,
        Path(__file__).resolve(),code/'chatbot_whole_assembly.py',code/'chatbot_whole_assembly_audit.py',code/'chatbot_whole_assembly_terminal.ps1',
        code/'chatbot_whole_checkpoint.py',code/'chatbot_whole_transfer_model.py',code/'chatbot_compact_geometry.py',
        code/'chatbot_interaction_launch.py',code/'chatbot_directional_binding.py',code/'chatbot_directional_launch.py',code/'chatbot_null_prior_kernel_binding.py',
        doc/'CHATBOT_BALANCED_INITIALIZER_RESULT_20261008.md',doc/'CHATBOT_WHOLE_ASSEMBLY_NEXT_20261008.md',doc/'CHATBOT_WHOLE_ASSEMBLY_PROTOCOL_20261008.md']
    for layer in initial['layers']:
        for field in ('witness','initial_checkpoint'):
            item=layer[field];assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes'];paths.append(item['path'])
    runtime=old['capture_runtime_roots'];view=old['interaction_view']
    b=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=old['python'],source_directory=str(source_dir),source_weights=str(source),
        spec_path=str(spec),initializer_path=str(init_path),initializer_audit_path=str(audit_path),initializer_audit_binding_path=str(audit_binding_path))
    if args.stage=='assemble':
        worker=code/'chatbot_whole_assembly.py';name='whole_assembly';schema='QWEN_WHOLE_ASSEMBLY_RESULT_V1'
        decisions=['WHOLE_ASSEMBLED_AND_ALLOCATED_REQUIRE_FIRST_BYTE_COUNT_AUDIT']
        limits=dict(worker_seconds=300,family_seconds=600,OS_bytes=16<<30,GPU_allocated_bytes=8<<30,GPU_reserved_bytes=9<<30,output_bytes=8<<20,log_bytes=2<<20)
    else:
        assert args.result is not None and args.result_sha
        raw,term,terminal=checked(args.result,args.result_sha)
        assert raw['decision']=='WHOLE_ASSEMBLED_AND_ALLOCATED_REQUIRE_FIRST_BYTE_COUNT_AUDIT'
        assert raw['new_original_BF16_full_forwards']==raw['new_student_full_forwards']==raw['new_optimizer_updates']==0
        original=Path(terminal['command'][terminal['command'].index('--binding')+1]);paths.extend([args.result,term,original])
        for value in terminal['output_manifest']:paths.append(receipt(value['path'],terminal))
        b.update(assembly_result_path=str(args.result.resolve()),assembly_terminal_path=str(term.resolve()))
        prefixes=('numpy','psutil','torch','functorch','sympy','mpmath','networkx','typing_extensions','packaging','filelock','jinja2','markupsafe','fsspec','safetensors')
        runtime=[v for v in runtime if v['view_name'].startswith(prefixes)];view=[v for v in view if v['view_name'].startswith(prefixes)]
        worker=code/'chatbot_whole_assembly_audit.py';name='whole_assembly_audit';schema='QWEN_WHOLE_ASSEMBLY_AUDIT_RESULT_V1'
        decisions=['WHOLE_ASSEMBLY_BYTE_COUNT_INDEPENDENTLY_VERIFIED'];limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=2<<20)
    b.update(inputs=[entry(p) for p in dict.fromkeys(map(str,paths))],capture_runtime_roots=runtime,interaction_view=view,
        job=dict(name=name,worker_path=str(worker),result_schema=schema,accepted_decisions=decisions,limits=limits))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']),runtime_roots=len(runtime))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('assemble','audit'),required=True)
    p.add_argument('--result',type=Path);p.add_argument('--result-sha');p.add_argument('--out',type=Path,required=True);main(p.parse_args())
