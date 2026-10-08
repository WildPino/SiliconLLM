"""Pin new convex bank and outputs for its first independent saved-only audit."""
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
    prior_path=doc/'chatbot_kernel_fit_binding_20261008.json';assert sha(prior_path)=='072ef6ec52538814c3a37efdba373a37b89e198d0a82820fe5f7d14105c48db7'
    prior=json.loads(prior_path.read_bytes());fit=doc/'chatbot_kernel_fit_20261008.json';terminal_path=fit.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(fit) and all(terminal['gates'].values())
    paths=[*(v['path'] for v in prior['inputs']),prior_path,fit,terminal_path]
    for item in terminal['output_manifest']:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];paths.append(p)
    worker=code/'chatbot_kernel_fit_audit.py';paths.extend([worker,Path(__file__).resolve(),doc/'CHATBOT_KERNEL_FIT_AUDIT_PROTOCOL_20261008.md'])
    b=dict(schema=prior['schema'],python=prior['python'],fit_path=str(fit),
        **{n:prior[n] for n in ('kernel_path','adoption_path','old_FIT_core_path','plan_path','prefix_journal_path','original_response_audit')},
        inputs=[entry(p) for p in dict.fromkeys(paths)],capture_runtime_roots=prior['capture_runtime_roots'],interaction_view=prior['interaction_view'],
        job=dict(name='kernel_fit_audit',worker_path=str(worker.resolve()),result_schema='QWEN_FULL_FIT_CONVEX_AUDIT_RESULT_V1',
            accepted_decisions=['SAVED_FULL_FIT_CONVEX_FAILURE_INDEPENDENTLY_VERIFIED'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
