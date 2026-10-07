"""Seal restricted capture runtime after retained Arrow faults; no model calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,tree,write_once


def entry(path):
    return dict(path=str(path.absolute()),resolved_path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path))


def main(args):
    assert not args.out.exists() and sha(args.prior)==args.prior_sha
    b=json.loads(args.prior.read_bytes());changes=[]
    allowed={str(ROOT/p) for p in ('benchmarks/native_expert_scaling/chatbot_source_capture.py','benchmarks/native_expert_scaling/chatbot_capture_launch.py')}
    for i,item in enumerate(b['inputs']):
        if item['path'] in allowed:
            current=entry(Path(item['path']));changes.append(dict(path=item['path'],old_sha=item['sha256'],new_sha=current['sha256']))
            b['inputs'][i]=current
    assert len(changes)==2
    receipt=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_source_runtime_view_20261007.json'
    maps=json.loads(receipt.read_bytes())['mappings'];roots=[]
    for item in maps:
        print('Sealing '+item['name'],flush=True)
        h,count,size=tree(item['path']);roots.append(dict(path=item['path'],view_name=item['name'],tree_sha256=h,files=count,bytes=size))
    b['capture_runtime_roots']=roots
    for path in (args.prior,receipt,Path(__file__).resolve(),*(ROOT/p for p in (
        'benchmarks/native_expert_scaling/chatbot_capture_runtime.ps1',
        'benchmarks/native_expert_scaling/chatbot_source_capture_faults.ps1',
        'docs/research/NATIVE_EXPERT_SCALING_20260925/CHATBOT_SOURCE_CAPTURE_REPAIR3_20261007.md',
        'docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_original_capture_repair2_20261007.launcher_failure.json',
        'docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_original_capture_repair2_20261007.worker.log',
        'docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_source_capture_windows_faults_20261007.json'))):
        b['inputs'].append(entry(path))
    b['restricted_runtime_repair']=dict(changes=changes,source_values_and_cases_unchanged=True,
        source_forwards_before_repair=0,removed_import_visibility=['pyarrow','datasets','unrelated optional packages'],
        original_full_runtime_tree='Retained administrative provenance, not enabled/required capture dependencies')
    write_once(args.out,b);print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out),roots=len(roots))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--prior-sha',required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
