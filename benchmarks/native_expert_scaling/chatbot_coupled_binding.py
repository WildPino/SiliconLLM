"""Bind only inputs used by the new whole-budget/FIT-only design screen."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry,receipt


def main(args):
    assert not args.out.exists()
    doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';code=ROOT/'benchmarks/native_expert_scaling'
    prior_path=doc/'chatbot_structural_audit_binding_20261008.json'
    assert sha(prior_path)=='46a4c5ab4fd44bb8e0bf6a4630e2805575c5feff7dd685eb69be9beefe3bd5b6'
    prior=json.loads(prior_path.read_bytes())
    plan_path=Path(prior['plan_path']);plan=json.loads(plan_path.read_bytes())
    assert sha(plan_path)=='bcbb9a5921ece474870375c438c42a474b0f115b85f9e3a35366cf1916b165dc'
    correction_path=doc/'chatbot_directional_plan_provenance_20261008.json'
    correction=json.loads(correction_path.read_bytes())
    assert correction['result_sha256']==sha(plan_path) and correction['actual_source_freeze']=='ec197f9efdf1b38dfdc6f8a958e2203a354ee6c8'
    assert correction['terminal_sha256']==sha(plan_path.with_suffix('.terminal.json'))
    jac_terminal_path=doc/'chatbot_directional_jacobian_20261008.terminal.json'
    jac_terminal=json.loads(jac_terminal_path.read_bytes());q_path=receipt(prior['Q_path'],jac_terminal)
    audit_path=doc/'chatbot_directional_audit_20261008.json'
    assert sha(audit_path)=='3885e228dc7022d7f962edf07cb74d445870320626192c8e4c06f107e61564d2'
    pilot_binding=json.loads((doc/'chatbot_directional_plan_binding_20261008.json').read_bytes())
    # The plan binding's field names are retained; do not acquire another router.
    geometry_path=Path(pilot_binding['saved_geometry'])
    pilot_terminal_path=doc/'chatbot_joint_pilot_20261007.terminal.json'
    receipt(geometry_path,json.loads(pilot_terminal_path.read_bytes()))
    adoption_path=Path(prior['adoption_path']);adoption=json.loads(adoption_path.read_bytes())
    anchors=[a for a in plan['anchors'] if a['split']=='fit'];assert len(anchors)==16
    binaries=[]
    for a in anchors:
        case=next(c for c in adoption['cases'] if c['id']==a['case_id'])
        assert case['split']=='fit' and Path(case['binary_path'])==Path(a['binary_path'])
        assert sha(case['binary_path'])==case['binary_SHA256'];binaries.append(Path(case['binary_path']))
    spec=code/'chatbot_compact_spec.json';ledger=doc/'chatbot_compact_geometry_budget_20261007.json'
    assert sha(spec)=='a5850df0172d9f9152e8391a3588ed5cfe9fb184d049955a161e28b4786cd9d0'
    assert sha(ledger)=='4aad875680fc2fb0ea35e275d395c71a8542ced41ffca6e70fc613bb6dc5cd37'
    worker=code/'chatbot_coupled_design.py'
    paths=[prior_path,prior['python'],Path(prior['python']).with_name('python312.dll'),plan_path,
        plan_path.with_suffix('.terminal.json'),correction_path,jac_terminal_path,audit_path,q_path,
        doc/'chatbot_directional_plan_binding_20261008.json',pilot_terminal_path,geometry_path,adoption_path,*binaries,spec,ledger,
        worker,Path(__file__).resolve(),code/'chatbot_directional_plan.py',code/'chatbot_joint_retained_audit.py',
        code/'chatbot_interaction_launch.py',code/'chatbot_directional_binding.py',code/'chatbot_directional_launch.py',
        doc/'CHATBOT_COUPLED_DESIGN_PROTOCOL_20261008.md',doc/'CHATBOT_COUPLED_JET_NEXT_20261008.md']
    binding=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=prior['python'],plan_path=str(plan_path),
        Q_path=str(q_path),saved_geometry=str(geometry_path),spec_path=str(spec),ledger_path=str(ledger),
        inputs=[entry(p) for p in dict.fromkeys(paths)],capture_runtime_roots=prior['capture_runtime_roots'],interaction_view=prior['interaction_view'],
        job=dict(name='coupled_design',worker_path=str(worker.resolve()),result_schema='QWEN_COUPLED_DESIGN_RESULT_V1',
            accepted_decisions=['CLOSE_COUPLED_AFFINE_COMPLETE_COST','CLOSE_COUPLED_JET_DESIGN','ELIGIBLE_FOR_ONE_SEPARATELY_FROZEN_COUPLED_COMPILER'],
            limits=dict(worker_seconds=60,family_seconds=120,OS_bytes=512<<20,output_bytes=8<<20,log_bytes=2<<20)))
    assert len(binding['capture_runtime_roots'])==5 and all(v['view_name'].startswith(('numpy','psutil')) for v in binding['capture_runtime_roots'])
    write_once(args.out,binding);print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out),files=len(binding['inputs']),runtime_roots=5)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
