"""Bind only FIRST factor audit inputs, including complete sealed witnesses."""
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
    prior_path=doc/'chatbot_source_quadratic_binding_20261008.json';prior=json.loads(prior_path.read_bytes())
    for item in prior['inputs']:assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
    result_path=doc/'chatbot_source_quadratic_20261008.json';raw=json.loads(result_path.read_bytes())
    terminal_path=result_path.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(result_path) and all(terminal['gates'].values()) and all(raw['qualification_gates'].values())
    names=('kappa','dictionary_norms','scores','residuals','selected','selected_normalized_atoms','coefficients','singular_values','R_BF16','T_BF16')
    worker=code/'chatbot_source_quadratic_audit.py'
    paths=[prior_path,*(i['path'] for i in prior['inputs']),result_path,terminal_path,
        *(receipt(raw['outputs'][n]['path'],terminal) for n in names),worker,Path(__file__).resolve()]
    b=dict(prior,factor_path=str(result_path),inputs=[entry(p) for p in dict.fromkeys(paths)],
        job=dict(name='source_quadratic_audit',worker_path=str(worker),result_schema='QWEN_SOURCE_QUADRATIC_AUDIT_RESULT_V1',accepted_decisions=['SOURCE_QUADRATIC_FACTORS_INDEPENDENTLY_VERIFIED'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
