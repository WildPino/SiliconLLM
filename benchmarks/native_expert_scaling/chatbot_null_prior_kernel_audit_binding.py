"""Bind first new covariance audit to immutable used inputs and saved outputs."""
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
    prior_path=doc/'chatbot_null_prior_kernel_binding_20261008.json';assert sha(prior_path)=='2b5b63bb4b801066e74362e471c92f0ab203d407194fce7a3c473a9e3b042ee0'
    prior=json.loads(prior_path.read_bytes())
    for item in prior['inputs']:assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
    kernel_path=doc/'chatbot_null_prior_kernel_20261008.json';raw=json.loads(kernel_path.read_bytes())
    assert raw['decision']=='ELIGIBLE_FOR_ONE_NULL_PRIOR_CONVEX_COMPILER' and all(raw['eligibility_gates'].values())
    terminal_path=kernel_path.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(kernel_path) and all(terminal['gates'].values())
    worker=code/'chatbot_null_prior_kernel_audit.py'
    paths=[prior_path,*(v['path'] for v in prior['inputs']),kernel_path,terminal_path,
        *(receipt(v['path'],terminal) for v in terminal['output_manifest']),worker,Path(__file__).resolve(),doc/'CHATBOT_NULL_PRIOR_KERNEL_AUDIT_PROTOCOL_20261008.md']
    b=dict(prior,kernel_path=str(kernel_path),inputs=[entry(p) for p in dict.fromkeys(paths)],
        job=dict(name='null_prior_kernel_audit',worker_path=str(worker.resolve()),result_schema='QWEN_NULL_PRIOR_COVARIANCE_AUDIT_RESULT_V1',
            accepted_decisions=['SAVED_NULL_PRIOR_COVARIANCE_INDEPENDENTLY_VERIFIED'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args())
