"""Freeze only used pilot inputs and the already-qualified restricted runtime."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def entry(path):
    path=Path(path).absolute()
    return dict(path=str(path),resolved_path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path))


def main(args):
    assert not args.out.exists() and sha(args.capture_binding)==args.capture_sha and sha(args.adoption)==args.adoption_sha
    prior=json.loads(args.capture_binding.read_bytes());adopt=json.loads(args.adoption.read_bytes())
    assert adopt['all_saved_frame_bytes_and_alignment_qualified'] and not adopt['missing_case_ids']
    assert (adopt['calibration_case_coverage'],adopt['fit_cases'],adopt['development_cases'],adopt['complete_requested_generations'])==(48,32,16,47)
    source=Path(prior['source_directory'])/'model.safetensors'
    old=next(i for i in prior['inputs'] if Path(i['path'])==source)
    assert sha(source)==old['sha256'] and source.stat().st_size==old['bytes']
    paths=[Path(prior['python']),Path(prior['python']).parent/'python312.dll',source,
        args.capture_binding,args.adoption,Path(adopt['source_report']),Path(__file__).resolve()]
    paths += [ROOT/p for p in (
        'benchmarks/native_expert_scaling/chatbot_joint_pilot.py',
        'benchmarks/native_expert_scaling/chatbot_joint_launch.py',
        'benchmarks/native_expert_scaling/chatbot_compact_geometry.py',
        'benchmarks/native_expert_scaling/chatbot_compact_spec.json',
        'benchmarks/native_expert_scaling/chatbot_interaction_launch.py',
        'docs/research/NATIVE_EXPERT_SCALING_20260925/CHATBOT_JOINT_PILOT_PROTOCOL_20261007.md',
        'docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_source_batch4_20261007.terminal.json',
        'docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_source_capture_terminal_20261007.json')]
    for case in adopt['cases']:
        assert sha(case['binary_path'])==case['binary_SHA256'] and sha(case['journal_path'])==case['journal_SHA256']
        paths += [Path(case['binary_path']),Path(case['journal_path'])]
    result=dict(schema='QWEN_JOINT_LAYER12_PILOT_BINDING_V1',python=prior['python'],
        source_weights=str(source),adoption_path=str(args.adoption.resolve()),
        spec_path=str(ROOT/'benchmarks/native_expert_scaling/chatbot_compact_spec.json'),
        inputs=[entry(p) for p in dict.fromkeys(paths)],capture_runtime_roots=prior['capture_runtime_roots'],
        interaction_view=prior['interaction_view'],
        limits=dict(worker_seconds=900,arm_seconds=240,family_seconds=1200,family_OS_bytes=12<<30,
            GPU_allocated=9<<30,GPU_reserved=10<<30,output_bytes=1<<30),
        completed_original_response_replays=0,whole_chatbot_quality_or_rate_qualified=False)
    write_once(args.out,result);print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out),files=len(result['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture-binding',type=Path,required=True);p.add_argument('--capture-sha',required=True)
    p.add_argument('--adoption',type=Path,required=True);p.add_argument('--adoption-sha',required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
