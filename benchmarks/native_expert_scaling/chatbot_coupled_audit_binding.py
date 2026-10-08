"""New design output binding for its first independent audit."""
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
    prior_path=doc/'chatbot_coupled_design_binding_20261008.json'
    assert sha(prior_path)=='3761ec4ca71e7ce4806c6bfec3a78c367ad9b5a881330bd36f4652f3dfd3f683'
    prior=json.loads(prior_path.read_bytes());design=doc/'chatbot_coupled_design_20261008.json'
    terminal_path=design.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert sha(design)=='6ca8ef5b70de9c2e4a39eb971a063b7fb946aa1a51a283a478045bda6512ff53'
    assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(design) and all(terminal['gates'].values())
    outputs=[]
    for item in terminal['output_manifest']:
        path=Path(item['path']);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];outputs.append(path)
    worker=code/'chatbot_coupled_audit.py'
    paths=[*(item['path'] for item in prior['inputs']),prior_path,design,terminal_path,*outputs,
        worker,Path(__file__).resolve(),doc/'CHATBOT_COUPLED_AUDIT_PROTOCOL_20261008.md']
    b=dict(schema=prior['schema'],python=prior['python'],design_path=str(design),
        **{n:prior[n] for n in ('plan_path','Q_path','saved_geometry')},inputs=[entry(p) for p in dict.fromkeys(paths)],
        capture_runtime_roots=prior['capture_runtime_roots'],interaction_view=prior['interaction_view'],
        job=dict(name='coupled_audit',worker_path=str(worker.resolve()),result_schema='QWEN_COUPLED_AUDIT_RESULT_V1',
            accepted_decisions=['SAVED_COUPLED_DESIGN_INDEPENDENTLY_VERIFIED'],
            limits=dict(worker_seconds=60,family_seconds=120,OS_bytes=512<<20,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
