"""Administrative source/runtime content sealing; no decoded tensor/model calls."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,tree,write_once


def main(out):
    assert not out.exists()
    old_path=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_interaction_binding_repair3_20261007.json'
    assert sha(old_path)=='1b57abd45f37670b7f2e9293fda5d3ed299930dbb406003b59132a1f90b5fb97'
    old=json.loads(old_path.read_bytes())
    source=Path(old['donor_source'])
    paths=[Path(old['python']),Path(old['python']).with_name('python312.dll'),old_path]
    paths += [Path(i['path']) for i in old['inputs'] if Path(i['path']).parent==source]
    paths += [source/'model.safetensors']
    paths += [ROOT/p for p in (
        'benchmarks/native_expert_scaling/chatbot_source_capture.py',
        'benchmarks/native_expert_scaling/chatbot_capture_launch.py',
        'benchmarks/native_expert_scaling/chatbot_capture_binding.py',
        'benchmarks/native_expert_scaling/chatbot_capture_cases.py',
        'benchmarks/native_expert_scaling/chatbot_interaction.py',
        'benchmarks/native_expert_scaling/chatbot_interaction_launch.py',
        'docs/research/NATIVE_EXPERT_SCALING_20260925/CHATBOT_SOURCE_CAPTURE_PROTOCOL_20261007.md')]
    inputs=[dict(path=str(p.absolute()),resolved_path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
    weight=next(i for i in inputs if Path(i['path']).name=='model.safetensors')
    assert weight['sha256']=='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe','Original source differs from retained44/57 binding'
    site=ROOT/'.venv/Lib/site-packages'
    print('Sealing installed runtime code/DLL trees, no imports/model calls',flush=True)
    digest,count,size=tree(site)
    b=dict(schema='QWEN_ORIGINAL_CAPTURE_BINDING_V1',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        python=old['python'],source_directory=str(source),inputs=inputs,
        runtime_tree=dict(path=str(site),tree_sha256=digest,files=count,bytes=size),
        interaction_view=[dict(path=i['path'],view_name=i['view_name']) for i in old['runtime_roots']],
        tree_scope='Sorted UTF8 relative path NUL size NUL SHA LF; py,pyd,dll,json,pem,METADATA,WHEEL,RECORD; no pyc',
        version_contract=dict(python='3.12.10',torch='2.6.0+cu124',transformers='5.13.1',tokenizers='0.22.2',numpy='2.4.6',psutil='7.2.2'),
        decoded_tensor_values=0,model_calls=0)
    write_once(out,b)
    print(json.dumps(dict(binding=str(out),sha256=sha(out),runtime_files=count,runtime_bytes=size)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
