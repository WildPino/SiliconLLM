"""Bind FIRST changed-kernel audit to original inputs and sealed NEW outputs."""
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
    prior_path=doc/'chatbot_quadratic_kernel_binding_20261008.json';prior=json.loads(prior_path.read_bytes())
    for item in prior['inputs']:assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
    kernel_path=doc/'chatbot_quadratic_kernel_20261008.json';raw=json.loads(kernel_path.read_bytes());terminal_path=kernel_path.with_suffix('.terminal.json');term=json.loads(terminal_path.read_bytes())
    assert term['actual_worker_exit_code']==0 and term['result_sha256']==sha(kernel_path) and all(term['gates'].values())
    assert raw['decision']=='ELIGIBLE_FOR_ONE_QUADRATIC_CONVEX_COMPILER' and all(raw['eligibility_gates'].values())
    worker=code/'chatbot_quadratic_kernel_audit.py'
    paths=[prior_path,*(i['path'] for i in prior['inputs']),kernel_path,terminal_path,
        *(receipt(v['path'],term) for v in raw['outputs'].values()),worker,Path(__file__).resolve()]
    b=dict(prior,kernel_path=str(kernel_path),inputs=[entry(p) for p in dict.fromkeys(paths)],
        job=dict(name='quadratic_kernel_audit',worker_path=str(worker),result_schema='QWEN_QUADRATIC_KERNEL_AUDIT_RESULT_V1',
            accepted_decisions=['SAVED_QUADRATIC_KERNEL_INDEPENDENTLY_VERIFIED'],limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
