"""Bind E16 first-fit question; preserve the closed original E160 trial."""
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
    assert not args.out.exists() and sha(args.prior)==args.prior_sha
    binding=json.loads(args.prior.read_bytes())
    assert binding['schema']=='QWEN_JOINT_LAYER12_PILOT_BINDING_V1'
    doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
    result_path=doc/'chatbot_joint_pilot_20261007.json';terminal_path=result_path.with_suffix('.terminal.json')
    result=json.loads(result_path.read_bytes());terminal=json.loads(terminal_path.read_bytes())
    assert terminal['actual_worker_exit_code']==0 and sha(result_path)==terminal['result_sha256']
    assert result['decision']=='CLOSE_FINITE_PILOT_ON_EXPOSURE_NO_FIT' and result['updates']==0
    directory=Path(result['routing_coefficients_F32']['path']).parent
    saved=[directory/n for n in ('routing_F32.pt','E16.routes_F32.pt','E16.exposure.json')]
    for path in saved:
        receipt=next(i for i in terminal['output_manifest'] if Path(i['path'])==path)
        assert path.stat().st_size==receipt['bytes'] and sha(path)==receipt['sha256']
    assert json.loads(saved[2].read_bytes())['eligible']
    paths=[args.prior,result_path,terminal_path,*saved,Path(__file__).resolve(),
        ROOT/'benchmarks/native_expert_scaling/chatbot_joint_baseline.py',
        ROOT/'benchmarks/native_expert_scaling/chatbot_joint_baseline_launch.py',
        doc/'CHATBOT_JOINT_BASELINE_PROTOCOL_20261007.md']
    binding['inputs'] += [entry(p) for p in paths]
    binding.update(schema='QWEN_JOINT_LAYER12_BASELINE_BINDING_V1',original_pilot_result=str(result_path),
        saved_geometry=str(saved[0]),saved_E16_routes=str(saved[1]),saved_E16_exposure=str(saved[2]),
        question='FIRST_E16_JOINT_FIT_NOT_REPAIR_OF_CLOSED_E160_EXPOSURE',
        unchanged_geometry_and_parent_control_replay=0,E160_count_utility_qualified=False)
    write_once(args.out,binding);print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out),files=len(binding['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--prior-sha',required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
