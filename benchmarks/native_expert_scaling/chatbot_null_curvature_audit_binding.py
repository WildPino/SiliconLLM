"""Bind FIRST curvature audit after the original acquisition has actually exited."""
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
    prior_path=doc/'chatbot_null_curvature_binding_20261008.json';prior=json.loads(prior_path.read_bytes())
    for item in prior['inputs']:assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
    result_path=args.curvature.resolve();assert result_path.parent==doc.resolve()
    raw=json.loads(result_path.read_bytes())
    terminal_path=result_path.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(result_path) and all(terminal['gates'].values()) and all(raw['procedure_gates'].values())
    worker=code/'chatbot_null_curvature_audit.py'
    paths=[prior_path,*(item['path'] for item in prior['inputs']),result_path,terminal_path,
        *(receipt(v['path'],terminal) for v in terminal['output_manifest']),worker,Path(__file__).resolve(),doc/'CHATBOT_NULL_CURVATURE_AUDIT_PROTOCOL_20261008.md']
    b=dict(prior,curvature_path=str(result_path),inputs=[entry(p) for p in dict.fromkeys(paths)],
        job=dict(name='null_curvature_audit',worker_path=str(worker),result_schema='QWEN_NULL_CURVATURE_AUDIT_RESULT_V1',accepted_decisions=['SAVED_NULL_CURVATURE_INDEPENDENTLY_VERIFIED'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--curvature',type=Path,required=True);main(p.parse_args())
