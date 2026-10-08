"""Bind first saved convex prior bank audit; no completed scientific rerun."""
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
    prior_path=doc/'chatbot_null_prior_fit_binding_20261008.json';assert sha(prior_path)=='a231235dd59b7e2574ba149046e43e9b29240bf650736b2a6bf0ed5407bd73b7'
    prior=json.loads(prior_path.read_bytes())
    for item in prior['inputs']:assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
    fit_path=doc/'chatbot_null_prior_fit_20261008.json';raw=json.loads(fit_path.read_bytes());term_path=fit_path.with_suffix('.terminal.json');term=json.loads(term_path.read_bytes())
    assert term['actual_worker_exit_code']==0 and term['result_sha256']==sha(fit_path) and all(term['gates'].values())
    assert raw['numerical_gate'] and raw['decision']=='CLOSE_NULL_PRIOR_CONVEX_NOVEL_FIDELITY'
    worker=code/'chatbot_null_prior_fit_audit.py'
    paths=[prior_path,*(v['path'] for v in prior['inputs']),fit_path,term_path,
        *(receipt(v['path'],term) for v in term['output_manifest']),worker,Path(__file__).resolve(),doc/'CHATBOT_NULL_PRIOR_FIT_AUDIT_PROTOCOL_20261008.md']
    b=dict(prior,fit_path=str(fit_path),inputs=[entry(p) for p in dict.fromkeys(paths)],
        job=dict(name='null_prior_fit_audit',worker_path=str(worker.resolve()),result_schema='QWEN_NULL_PRIOR_CONVEX_AUDIT_RESULT_V1',
            accepted_decisions=['SAVED_NULL_PRIOR_CONVEX_FAILURE_INDEPENDENTLY_VERIFIED'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args())
