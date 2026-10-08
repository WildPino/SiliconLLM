"""Pin all new compiler outputs for their first independent numerical audit."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry


def main(args):
    assert not args.out.exists();doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    prior_path=doc/'chatbot_coupled_compile_binding_20261008.json';prior=json.loads(prior_path.read_bytes())
    assert sha(prior_path)=='7b6f172f27a379b01e0fc7c503be463f909fade30fa80d6544994783dc4130d6'
    compiled=doc/'chatbot_coupled_compile_20261008.json';terminal_path=compiled.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(compiled) and all(terminal['gates'].values())
    paths=[*(v['path'] for v in prior['inputs']),prior_path,compiled,terminal_path]
    for item in terminal['output_manifest']:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];paths.append(p)
    worker=code/'chatbot_coupled_compile_audit.py'
    paths.extend([worker,Path(__file__).resolve(),doc/'CHATBOT_COUPLED_COMPILE_AUDIT_PROTOCOL_20261008.md'])
    b=dict(schema=prior['schema'],python=prior['python'],compile_path=str(compiled),
        **{n:prior[n] for n in ('design_path','Q_path','source_J_path','plan_path','adoption_path','original_response_audit','core_checkpoint','saved_E16_routes')},
        inputs=[entry(p) for p in dict.fromkeys(paths)],capture_runtime_roots=prior['capture_runtime_roots'],interaction_view=prior['interaction_view'],
        job=dict(name='coupled_compile_audit',worker_path=str(worker.resolve()),result_schema='QWEN_COUPLED_COMPILE_AUDIT_RESULT_V1',
            accepted_decisions=['SAVED_COUPLED_COMPILER_FAILURE_INDEPENDENTLY_VERIFIED'],
            limits=dict(worker_seconds=90,family_seconds=180,OS_bytes=2<<30,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
