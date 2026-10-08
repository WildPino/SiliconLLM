"""Bind only used input bytes; plan first, then new derivatives of retained weights."""
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


def receipt(path,terminal):
    path=Path(path);item=next(v for v in terminal['output_manifest'] if Path(v['path'])==path)
    assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];return path


def main(args):
    assert not args.out.exists() and sha(args.prior)==args.prior_sha
    prior=json.loads(args.prior.read_bytes());assert prior['schema']=='QWEN_JOINT_LAYER12_BASELINE_BINDING_V1'
    doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    adoption=json.loads(Path(prior['adoption_path']).read_bytes())
    pilot_path=doc/'chatbot_joint_pilot_20261007.json';pilot_terminal=pilot_path.with_suffix('.terminal.json')
    terminal=json.loads(pilot_terminal.read_bytes());assert terminal['actual_worker_exit_code']==0 and sha(pilot_path)==terminal['result_sha256']
    saved=[receipt(prior[field],terminal) for field in ('saved_geometry','saved_E16_routes')]
    base_paths=[args.prior,prior['python'],Path(prior['python']).with_name('python312.dll'),
        prior['adoption_path'],*(v['binary_path'] for v in adoption['cases']),pilot_path,pilot_terminal,*saved,
        code/'chatbot_interaction_launch.py',code/'chatbot_joint_retained_audit.py',Path(__file__).resolve(),
        code/'chatbot_directional_launch.py',doc/'CHATBOT_DIRECTIONAL_PROTOCOL_20261008.md']
    binding=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=prior['python'],adoption_path=prior['adoption_path'],
        saved_geometry=prior['saved_geometry'],saved_E16_routes=prior['saved_E16_routes'],
        new_original_BF16_full_forwards=0,whole_chatbot_quality_or_rate_qualified=False)
    if args.job=='plan':
        worker=code/'chatbot_directional_plan.py';base_paths.append(worker)
        runtime=[v for v in prior['capture_runtime_roots'] if v['view_name'].startswith('psutil')]
        view=[v for v in prior['interaction_view'] if v['view_name'].startswith('psutil')]
        job=dict(name='plan',result_schema='QWEN_DIRECTIONAL_PLAN_RESULT_V1',accepted_decisions=['ANCHORS_FROZEN_FOR_FIRST_DIRECTIONAL_VALUES'],
            limits=dict(worker_seconds=45,family_seconds=90,OS_bytes=512<<20,output_bytes=1<<20,log_bytes=2<<20))
    else:
        assert args.plan is not None and args.plan_sha and sha(args.plan)==args.plan_sha
        plan=json.loads(args.plan.read_bytes());assert plan['decision']=='ANCHORS_FROZEN_FOR_FIRST_DIRECTIONAL_VALUES' and all(plan['procedure_gates'].values())
        plan_terminal=args.plan.with_suffix('.terminal.json');plan_receipt=json.loads(plan_terminal.read_bytes())
        assert plan_receipt['actual_worker_exit_code']==0 and plan_receipt['result_sha256']==args.plan_sha
        baseline_path=doc/'chatbot_joint_baseline_20261007.json';baseline_terminal=baseline_path.with_suffix('.terminal.json')
        result=json.loads(baseline_path.read_bytes());baseline_receipt=json.loads(baseline_terminal.read_bytes())
        assert baseline_receipt['actual_worker_exit_code']==0 and sha(baseline_path)==baseline_receipt['result_sha256']
        student=receipt(ROOT/'results/native_expert_scaling/chatbot_joint_baseline_20261007/E16.final_F32.pt',baseline_receipt)
        source=next(v for v in prior['inputs'] if Path(v['path'])==Path(prior['source_weights']))
        assert sha(prior['source_weights'])==source['sha256'] and Path(prior['source_weights']).stat().st_size==source['bytes']
        worker=code/'chatbot_directional_jacobian.py'
        base_paths.extend([args.plan,plan_terminal,code/'chatbot_directional_plan.py',baseline_path,baseline_terminal,student,prior['source_weights'],worker])
        runtime=prior['capture_runtime_roots'];view=prior['interaction_view']
        binding.update(plan_path=str(args.plan.resolve()),plan_sha256=args.plan_sha,source_weights=prior['source_weights'],student_checkpoint=str(student),
            arithmetic='Smooth real-coefficient formula evaluated in F64; NOT rounded-inference-program derivative')
        job=dict(name='jacobian',result_schema='QWEN_DIRECTIONAL_JACOBIAN_RESULT_V1',
            accepted_decisions=['DIRECTIONAL_DEFICIT_SUPPORTED_NEW_STRUCTURAL_FIT_JUSTIFIED','DIRECTIONAL_DEFICIT_UNSUPPORTED_CHANGE_REPRESENTATION'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,GPU_allocated_bytes=4<<30,GPU_reserved_bytes=6<<30,output_bytes=512<<20,log_bytes=2<<20))
    binding.update(job=dict(job,worker_path=str(worker.resolve())),inputs=[entry(p) for p in dict.fromkeys(base_paths)],
        capture_runtime_roots=runtime,interaction_view=view)
    write_once(args.out,binding);print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out),files=len(binding['inputs']),runtime_roots=len(runtime))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--prior-sha',required=True)
    p.add_argument('--job',choices=['plan','jacobian'],required=True);p.add_argument('--plan',type=Path);p.add_argument('--plan-sha')
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
