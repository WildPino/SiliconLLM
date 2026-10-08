"""Freeze a retained-only audit with NumPy/psutil, no source model/runtime Torch."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry


def main(args):
    assert not args.out.exists() and sha(args.jacobian)==args.jacobian_sha
    doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    old_path=doc/'chatbot_directional_jacobian_binding_20261008.json';old=json.loads(old_path.read_bytes())
    terminal_path=args.jacobian.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==args.jacobian_sha and all(terminal['gates'].values())
    files=[]
    for item in terminal['output_manifest']:
        path=Path(item['path']);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];files.append(path)
    plan_path=Path(old['plan_path']);assert sha(plan_path)==old['plan_sha256'];plan=json.loads(plan_path.read_bytes())
    paths=[old_path,args.jacobian,terminal_path,plan_path,old['saved_geometry'],old['python'],Path(old['python']).with_name('python312.dll'),
        *(v['binary_path'] for v in plan['anchors']),*files,
        code/'chatbot_directional_audit.py',Path(__file__).resolve(),code/'chatbot_directional_binding.py',code/'chatbot_directional_launch.py',
        code/'chatbot_interaction_launch.py',code/'chatbot_directional_plan.py',code/'chatbot_joint_retained_audit.py',
        doc/'CHATBOT_DIRECTIONAL_AUDIT_PROTOCOL_20261008.md']
    runtime=[v for v in old['capture_runtime_roots'] if v['view_name'].startswith(('numpy','psutil'))]
    view=[v for v in old['interaction_view'] if v['view_name'].startswith(('numpy','psutil'))]
    assert len(runtime)==5
    b=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=old['python'],saved_geometry=old['saved_geometry'],plan_path=str(plan_path),
        jacobian_path=str(args.jacobian.resolve()),jacobian_directory=str(files[0].parent),inputs=[entry(p) for p in dict.fromkeys(paths)],
        capture_runtime_roots=runtime,interaction_view=view,new_original_BF16_full_forwards=0,whole_chatbot_quality_or_rate_qualified=False,
        job=dict(name='audit',worker_path=str((code/'chatbot_directional_audit.py').resolve()),result_schema='QWEN_DIRECTIONAL_AUDIT_RESULT_V1',
            accepted_decisions=['SAVED_DIRECTIONAL_DIAGNOSTIC_INDEPENDENTLY_VERIFIED'],
            limits=dict(worker_seconds=90,family_seconds=180,OS_bytes=1<<30,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out),files=len(b['inputs']),runtime_roots=len(runtime))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jacobian',type=Path,required=True);p.add_argument('--jacobian-sha',required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
