"""Seal first saved-only audit for the NEW augmented response compiler."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry,receipt
from chatbot_null_prior_kernel_binding import checked


def main(args):
    assert not args.out.exists();doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    priorpath=doc/'chatbot_quadratic_fit_binding_20261008.json';prior=json.loads(priorpath.read_bytes())
    for item in prior['inputs']:assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
    fit=args.fit.resolve();assert fit.parent==doc.resolve();raw,tpath,t=checked(fit)
    assert raw['numerical_gate'] and raw['decision'] in ('CLOSE_QUADRATIC_CONVEX_NOVEL_FIDELITY','ELIGIBLE_FOR_SEPARATE_FULL_NOVEL_SCREEN')
    paths=[priorpath,*(v['path'] for v in prior['inputs']),fit,tpath,*(receipt(v['path'],t) for v in raw['outputs'].values()),Path(__file__).resolve()]
    worker=code/'chatbot_quadratic_fit_audit.py'
    b=dict(prior,fit_path=str(fit),inputs=[entry(p) for p in dict.fromkeys(paths)],
        job=dict(name='quadratic_fit_audit',worker_path=str(worker),result_schema='QWEN_QUADRATIC_CONVEX_AUDIT_RESULT_V1',
            accepted_decisions=['SAVED_QUADRATIC_CONVEX_RESULT_INDEPENDENTLY_VERIFIED'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
