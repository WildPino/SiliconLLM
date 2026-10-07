"""Seal retained audit inputs; no source weights or scientific runtime required."""
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
    prior=json.loads(args.prior.read_bytes());doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
    baseline=doc/'chatbot_joint_baseline_20261007.json';terminal=baseline.with_suffix('.terminal.json')
    raw=json.loads(baseline.read_bytes());end=json.loads(terminal.read_bytes())
    assert end['actual_worker_exit_code']==0 and end['result_sha256']==sha(baseline)
    assert raw['decision']=='CLOSE_FINITE_E16_FIT_NO_E160_OR_WHOLE_PROMOTION'
    adoption=Path(prior['adoption_path']);data=json.loads(adoption.read_bytes())
    exposure_directory=Path(prior['saved_geometry']).parent
    paths=[Path(prior['python']),Path(prior['python']).parent/'python312.dll',args.prior,baseline,terminal,adoption,
        doc/'chatbot_joint_pilot_20261007.json',doc/'chatbot_joint_pilot_20261007.terminal.json',Path(__file__).resolve(),
        ROOT/'benchmarks/native_expert_scaling/chatbot_joint_retained_audit.py',ROOT/'benchmarks/native_expert_scaling/chatbot_joint_retained_launch.py',
        ROOT/'benchmarks/native_expert_scaling/chatbot_interaction_launch.py',doc/'CHATBOT_JOINT_RETAINED_AUDIT_PROTOCOL_20261007.md']
    for e in (16,160):paths += [exposure_directory/f'E{e}.routes_F32.pt',exposure_directory/f'E{e}.exposure.json']
    for case in data['cases']:
        assert sha(case['binary_path'])==case['binary_SHA256'];paths.append(Path(case['binary_path']))
    for encoding in raw['arms'][0]['encodings'].values():
        for receipt in encoding['prediction_files'].values():
            assert sha(receipt['path'])==receipt['sha256'];paths.append(Path(receipt['path']))
    result=dict(schema='QWEN_JOINT_RETAINED_AUDIT_BINDING_V1',python=prior['python'],adoption_path=str(adoption),
        exposure_directory=str(exposure_directory),baseline_result=str(baseline),inputs=[entry(p) for p in dict.fromkeys(paths)],
        capture_runtime_roots=[i for i in prior['capture_runtime_roots'] if i['view_name'].startswith('psutil')],
        interaction_view=[i for i in prior['interaction_view'] if i['view_name'].startswith('psutil')],
        limits=dict(worker_seconds=120,family_seconds=180,family_OS_bytes=512<<20,GPU=0,output_bytes=1<<20),
        source_or_fitted_function_replay=0)
    assert len(result['capture_runtime_roots'])==len(result['interaction_view'])==2
    write_once(args.out,result);print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out),inputs=len(result['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--prior-sha',required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
