"""Bind only reused geometry/scales for the source-null covariance prerequisite."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry,receipt


def checked(path,want=None):
    path=Path(path)
    if want:assert sha(path)==want
    raw=json.loads(path.read_bytes());terminal_path=path.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(path) and all(terminal['gates'].values())
    return raw,terminal_path,terminal


def main(args):
    assert not args.out.exists();doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    prior_path=doc/'chatbot_kernel_binding_20261008.json';assert sha(prior_path)=='28c06df3c792251bc2072b2fd13e30e980f4677aaffbdbf76b1cc07243af8810'
    prior=json.loads(prior_path.read_bytes())
    old_path=doc/'chatbot_kernel_20261008.json';old,oldterm,ot=checked(old_path,'d63a24ec90d5c382f9c00655afcecc9f70ca1d43a1e1d876ac744afbd002aa9c')
    assert old['decision']=='ELIGIBLE_FOR_ONE_FULL_FIT_CONVEX_COMPILER'
    audit_path=doc/'chatbot_kernel_audit_20261008.json';audit,aterm,_=checked(audit_path,'ef4bee138ac55c1ffc3f8f6ad0cbd9fbeed22226e10c3fab9bfa76c32c0ced73')
    assert audit['decision']=='SAVED_FULL_FIT_KERNEL_INDEPENDENTLY_VERIFIED'
    used=[receipt(old['outputs'][n]['path'],ot) for n in ('FIT_U','FIT_W','FIT_weights')]
    plan_binding_path=doc/'chatbot_directional_plan_binding_20261008.json';plan_binding=json.loads(plan_binding_path.read_bytes())
    geometry=Path(plan_binding['saved_geometry']);pilotterm=doc/'chatbot_joint_pilot_20261007.terminal.json'
    receipt(geometry,json.loads(pilotterm.read_bytes()))
    original_binding_item=next(v for v in plan_binding['inputs'] if Path(v['path'])==geometry)
    assert sha(geometry)==original_binding_item['sha256']
    plan_path=doc/'chatbot_directional_plan_20261008.json';plan=json.loads(plan_path.read_bytes())
    assert sha(plan_path)=='bcbb9a5921ece474870375c438c42a474b0f115b85f9e3a35366cf1916b165dc'
    assert plan['projection_rank_proof']['exact_real_row_rank']==32
    correction_path=doc/'chatbot_directional_plan_provenance_20261008.json';correction=json.loads(correction_path.read_bytes())
    assert correction['result_sha256']==sha(plan_path) and correction['actual_source_freeze']=='ec197f9efdf1b38dfdc6f8a958e2203a354ee6c8'
    assert correction['terminal_sha256']==sha(plan_path.with_suffix('.terminal.json'))
    design_path=doc/'chatbot_coupled_design_20261008.json';design,designterm,dt=checked(design_path,'6ca8ef5b70de9c2e4a39eb971a063b7fb946aa1a51a283a478045bda6512ff53')
    anchor_mass=receipt(ROOT/'results/native_expert_scaling/chatbot_coupled_design_20261008/W.npy',dt)
    original=doc/'chatbot_joint_retained_audit_20261007.json';assert sha(original)=='8eca7c810284a8dd1effa3d64d23a1f4d7010028df4748aeb54a6e0c843c23c8'
    directional=doc/'chatbot_directional_jacobian_20261008.json';_,directionterm,_=checked(directional,'576f0c5a8495456fe6d4dac9b0055e8a32d964546029d00ddc4f89c337356b22')
    directional_audit=doc/'chatbot_directional_audit_20261008.json';_,dat,_=checked(directional_audit,'3885e228dc7022d7f962edf07cb74d445870320626192c8e4c06f107e61564d2')
    worker=code/'chatbot_null_prior_kernel.py'
    paths=[prior_path,prior['python'],Path(prior['python']).with_name('python312.dll'),old_path,oldterm,audit_path,aterm,*used,
        plan_binding_path,geometry,pilotterm,plan_path,plan_path.with_suffix('.terminal.json'),correction_path,design_path,designterm,anchor_mass,
        original,directional,directionterm,directional_audit,dat,worker,Path(__file__).resolve(),
        code/'chatbot_joint_retained_audit.py',code/'chatbot_interaction_launch.py',code/'chatbot_directional_binding.py',code/'chatbot_directional_launch.py',
        doc/'CHATBOT_NULL_JET_PRIOR_NEXT_20261008.md',doc/'CHATBOT_NULL_PRIOR_KERNEL_PROTOCOL_20261008.md']
    b=dict(schema=prior['schema'],python=prior['python'],old_kernel_path=str(old_path),saved_geometry=str(geometry),anchor_mass_path=str(anchor_mass),
        original_response_audit=str(original),original_directional_report=str(directional),inputs=[entry(p) for p in dict.fromkeys(paths)],
        capture_runtime_roots=prior['capture_runtime_roots'],interaction_view=prior['interaction_view'],
        job=dict(name='null_prior_kernel',worker_path=str(worker.resolve()),result_schema='QWEN_NULL_PRIOR_COVARIANCE_RESULT_V1',
            accepted_decisions=['ELIGIBLE_FOR_ONE_NULL_PRIOR_CONVEX_COMPILER','CLOSE_NULL_PRIOR_COVARIANCE_PREREQUISITE'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=512<<20,log_bytes=2<<20)))
    assert len(b['capture_runtime_roots'])==5
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']),runtime_roots=5)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args())
