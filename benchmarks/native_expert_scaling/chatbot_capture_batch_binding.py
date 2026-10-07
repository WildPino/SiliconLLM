"""Bind a new previously-unentered-case batch using qualified retained operands."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def entry(path):
    return dict(path=str(path.absolute()),resolved_path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path))


def main(args):
    assert not args.out.exists() and sha(args.prior)==args.prior_sha and sha(args.adoption)==args.adoption_sha
    b=json.loads(args.prior.read_bytes());adopt=json.loads(args.adoption.read_bytes())
    assert adopt['schema']=='QWEN_ORIGINAL_FORWARD_PREFIX_ADOPTION_V1' and adopt['all_saved_frame_bytes_and_alignment_qualified']
    assert adopt['missing_case_ids'] and adopt['new_source_forwards']==adopt['new_source_responses']==0
    assert sha(adopt['source_report'])==adopt['source_report_SHA256']
    allowed={str(ROOT/p) for p in ('benchmarks/native_expert_scaling/chatbot_source_capture.py','benchmarks/native_expert_scaling/chatbot_capture_launch.py')}
    changes=[]
    for i,item in enumerate(b['inputs']):
        if item['path'] in allowed:
            new=entry(Path(item['path']));changes.append(dict(path=item['path'],old_sha=item['sha256'],new_sha=new['sha256']));b['inputs'][i]=new
    assert len(changes)==2
    paths=[args.prior,args.adoption,Path(adopt['source_report']),Path(__file__).resolve(),
        ROOT/'benchmarks/native_expert_scaling/chatbot_capture_adopt.py',
        ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/CHATBOT_SOURCE_CAPTURE_BATCH_PROTOCOL_20261007.md']
    for case in adopt['cases']:
        assert sha(case['binary_path'])==case['binary_SHA256'] and sha(case['journal_path'])==case['journal_SHA256']
        paths += [Path(case['binary_path']),Path(case['journal_path'])]
    known={i['path'] for i in b['inputs']}
    for p in paths:
        if str(p.absolute()) not in known:b['inputs'].append(entry(p));known.add(str(p.absolute()))
    b.update(adoption_path=str(args.adoption.resolve()),adoption_SHA256=args.adoption_sha,
        batch_acquisition=dict(maximum_previously_unentered_cases=8,worker_seconds=300,original_ALL48_300s_capture_qualified=False,
            retained_case_coverage=adopt['calibration_case_coverage'],changed_sources=changes,
            I_O='append/flush per forward; fsync each16 and at completed conversation',source_response_replay=0))
    write_once(args.out,b);print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out),next_cases=adopt['missing_case_ids'][:8])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--prior-sha',required=True)
    p.add_argument('--adoption',type=Path,required=True);p.add_argument('--adoption-sha',required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
