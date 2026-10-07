"""Numbered metadata-only pre-observation input update; no runtime/model replay."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def main(args):
    assert sha(args.prior)==args.prior_sha and not args.out.exists()
    b=json.loads(args.prior.read_bytes());changes=[]
    allowed={str(ROOT/p) for p in ('benchmarks/native_expert_scaling/chatbot_source_capture.py',
        'benchmarks/native_expert_scaling/chatbot_capture_launch.py')}
    for item in b['inputs']:
        if item['path'] in allowed:
            path=Path(item['path']);current=sha(path)
            changes.append(dict(path=str(path),old_sha256=item['sha256'],new_sha256=current))
            item.update(bytes=path.stat().st_size,sha256=current,resolved_path=str(path.resolve()))
    assert len(changes)==2
    for path in (args.prior,Path(__file__).resolve(),*(ROOT/p for p in (
        'docs/research/NATIVE_EXPERT_SCALING_20260925/CHATBOT_SOURCE_CAPTURE_REPAIR2_20261007.md',
        'docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_original_capture_20261007.launcher_failure.json',
        'docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_original_capture_20261007.worker.log'))):
        b['inputs'].append(dict(path=str(path.absolute()),resolved_path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path)))
    b['startup_repair']=dict(changes=changes,criteria_unchanged=True,completed_source_forwards_before_repair=0,
        first_decoded_weights='UNKNOWN before forced kill; empty capture namespace proves no entered forward, not no decoding',
        purpose='Identify/prevent subprocess before spawn; retain actual startup stage/unknown native descendants')
    write_once(args.out,b);print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--prior-sha',required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
