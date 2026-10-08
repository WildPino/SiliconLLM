"""Pin new retained outputs and reused exact source proof; no Torch runtime audit."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry


def main(args):
    assert not args.out.exists() and sha(args.fit)==args.fit_sha
    code=ROOT/'benchmarks/native_expert_scaling';doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
    prior_path=doc/'chatbot_structural_binding_20261008.json';prior=json.loads(prior_path.read_bytes())
    terminal_path=args.fit.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==args.fit_sha and all(terminal['gates'].values())
    files=[]
    for item in terminal['output_manifest']:
        path=Path(item['path']);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];files.append(path)
    original_audit=doc/'chatbot_joint_retained_audit_20261007.json'
    assert sha(original_audit)=='8eca7c810284a8dd1effa3d64d23a1f4d7010028df4748aeb54a6e0c843c23c8'
    paths=[args.fit,terminal_path,prior_path,prior['python'],Path(prior['python']).with_name('python312.dll'),*files,
        *(v['path'] for v in prior['inputs'] if Path(v['path']).name.endswith(('.bf16.bin','.frames.jsonl'))),
        *(prior[n] for n in ('plan_path','adoption_path','baseline_result','prior_jacobian_result','source_J_path','Q_path','directions_path','warm_checkpoint')),
        original_audit,code/'chatbot_structural_audit.py',Path(__file__).resolve(),code/'chatbot_joint_retained_audit.py',code/'chatbot_interaction_launch.py',
        code/'chatbot_directional_launch.py',code/'chatbot_directional_binding.py',doc/'CHATBOT_STRUCTURAL_AUDIT_PROTOCOL_20261008.md']
    b=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=prior['python'],fit_result=str(args.fit.resolve()),fit_directory=str(files[0].parent),
        **{n:prior[n] for n in ('plan_path','adoption_path','baseline_result','prior_jacobian_result','source_J_path','Q_path','directions_path','warm_checkpoint')},
        original_response_audit=str(original_audit),inputs=[entry(p) for p in dict.fromkeys(paths)],
        capture_runtime_roots=[v for v in prior['capture_runtime_roots'] if v['view_name'].startswith(('numpy','psutil'))],
        interaction_view=[v for v in prior['interaction_view'] if v['view_name'].startswith(('numpy','psutil'))],
        job=dict(name='structural_audit',worker_path=str((code/'chatbot_structural_audit.py').resolve()),result_schema='QWEN_STRUCTURAL_AUDIT_RESULT_V1',
            accepted_decisions=['SAVED_FIRST_STRUCTURAL_FIT_INDEPENDENTLY_VERIFIED'],
            limits=dict(worker_seconds=90,family_seconds=180,OS_bytes=1<<30,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--fit-sha',required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
