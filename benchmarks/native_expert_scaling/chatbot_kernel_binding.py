"""Bind FIT-only kernel inputs or new immutable kernel outputs for first audit."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry,receipt


def main(args):
    assert not args.out.exists();doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    prior_path=doc/'chatbot_coupled_compile_binding_20261008.json'
    assert sha(prior_path)=='7b6f172f27a379b01e0fc7c503be463f909fade30fa80d6544994783dc4130d6'
    prior=json.loads(prior_path.read_bytes());adoption_path=Path(prior['adoption_path']);adoption=json.loads(adoption_path.read_bytes())
    cost=doc/'chatbot_coupled_compile_20261008.json';assert sha(cost)=='9569e690c5975d01e9b9f9e5bc454be64563e93a2fc6c9efa93648055ab93f66'
    audit=doc/'chatbot_coupled_compile_audit_20261008.json';assert sha(audit)=='8ae10f75d05a0744033a5f147fad8266ec68453632badb98af07c7c09dd48551'
    pilot_terminal=doc/'chatbot_joint_pilot_20261007.terminal.json'
    saved=receipt(prior['saved_E16_routes'],json.loads(pilot_terminal.read_bytes()))
    paths=[prior_path,prior['python'],Path(prior['python']).with_name('python312.dll'),adoption_path,cost,audit,
        pilot_terminal,saved,*(v['binary_path'] for v in adoption['cases'] if v['split']=='fit'),
        code/'chatbot_kernel.py',Path(__file__).resolve(),code/'chatbot_joint_retained_audit.py',code/'chatbot_interaction_launch.py',
        code/'chatbot_directional_binding.py',code/'chatbot_directional_launch.py',
        doc/'CHATBOT_FULL_FIT_KERNEL_NEXT_20261008.md',doc/'CHATBOT_KERNEL_PROTOCOL_20261008.md']
    b=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=prior['python'],adoption_path=str(adoption_path),
        saved_E16_routes=str(saved),cost_path=str(cost),capture_runtime_roots=prior['capture_runtime_roots'],interaction_view=prior['interaction_view'])
    if args.job=='kernel':
        worker=code/'chatbot_kernel.py'
        job=dict(name='kernel',result_schema='QWEN_FULL_FIT_KERNEL_RESULT_V1',
            accepted_decisions=['ELIGIBLE_FOR_ONE_FULL_FIT_CONVEX_COMPILER','CLOSE_FULL_FIT_KERNEL_PREREQUISITE'])
    else:
        kernel=doc/'chatbot_kernel_20261008.json';terminal_path=kernel.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
        assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(kernel) and all(terminal['gates'].values())
        for item in terminal['output_manifest']:
            p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];paths.append(p)
        worker=code/'chatbot_kernel_audit.py';paths.extend([kernel,terminal_path,doc/'chatbot_kernel_binding_20261008.json',worker,doc/'CHATBOT_KERNEL_AUDIT_PROTOCOL_20261008.md'])
        b['kernel_path']=str(kernel)
        job=dict(name='kernel_audit',result_schema='QWEN_FULL_FIT_KERNEL_AUDIT_RESULT_V1',accepted_decisions=['SAVED_FULL_FIT_KERNEL_INDEPENDENTLY_VERIFIED'])
    b.update(inputs=[entry(p) for p in dict.fromkeys(paths)],job=dict(job,worker_path=str(worker.resolve()),
        limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=512<<20 if args.job=='kernel' else 1<<20,log_bytes=2<<20)))
    assert len(b['capture_runtime_roots'])==5
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',choices=['kernel','audit'],required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
