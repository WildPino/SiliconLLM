"""Freeze the admitted design, immutable FIT source data and one core checkpoint."""
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
    old_path=doc/'chatbot_structural_binding_20261008.json';old=json.loads(old_path.read_bytes())
    design_path=doc/'chatbot_coupled_design_20261008.json';design=json.loads(design_path.read_bytes())
    assert sha(design_path)=='6ca8ef5b70de9c2e4a39eb971a063b7fb946aa1a51a283a478045bda6512ff53' and all(design['eligibility_gates'].values())
    audit=doc/'chatbot_coupled_audit_20261008.json';assert sha(audit)=='c700b96a722c0fe8f3c5499438e541341b2e0bef13a92d4e17d94fb489e5f237'
    audit_terminal=audit.with_suffix('.terminal.json');terminal=json.loads(audit_terminal.read_bytes())
    assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(audit) and all(terminal['gates'].values())
    fit_terminal_path=doc/'chatbot_structural_fit_20261008.terminal.json'
    core=receipt(ROOT/'results/native_expert_scaling/chatbot_structural_fit_20261008/E16.final_F32.pt',json.loads(fit_terminal_path.read_bytes()))
    assert sha(core)=='8e04c2d59c61b5d822051ad04de107a808b0e2e5d3984b7599d485de5a5c3e32'
    jac=Path(old['prior_jacobian_result']);jac_terminal=jac.with_suffix('.terminal.json')
    source=receipt(old['source_J_path'],json.loads(jac_terminal.read_bytes()));q=receipt(old['Q_path'],json.loads(jac_terminal.read_bytes()))
    plan=Path(old['plan_path']);adoption=Path(old['adoption_path']);adopt=json.loads(adoption.read_bytes())
    pilot_terminal_path=doc/'chatbot_joint_pilot_20261007.terminal.json'
    saved=receipt(old['saved_E16_routes'],json.loads(pilot_terminal_path.read_bytes()))
    original=doc/'chatbot_joint_retained_audit_20261007.json';assert sha(original)=='8eca7c810284a8dd1effa3d64d23a1f4d7010028df4748aeb54a6e0c843c23c8'
    selected=[Path(design['output_arrays'][n]['path']) for n in ('W','K','mu','scale','z','selected_gamma','selected_gradient_x')]
    worker=code/'chatbot_coupled_compile.py'
    paths=[old_path,old['python'],Path(old['python']).with_name('python312.dll'),design_path,
        design_path.with_suffix('.terminal.json'),audit,audit_terminal,fit_terminal_path,core,
        jac,jac_terminal,source,q,plan,adoption,pilot_terminal_path,saved,original,*selected,
        *(v['binary_path'] for v in adopt['cases']),worker,Path(__file__).resolve(),code/'chatbot_joint_retained_audit.py',
        code/'chatbot_directional_plan.py',code/'chatbot_interaction_launch.py',code/'chatbot_directional_binding.py',code/'chatbot_directional_launch.py',
        doc/'CHATBOT_COUPLED_COMPILE_PROTOCOL_20261008.md']
    b=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=old['python'],design_path=str(design_path),
        Q_path=str(q),source_J_path=str(source),jacobian_result=str(jac),core_checkpoint=str(core),
        plan_path=str(plan),adoption_path=str(adoption),saved_E16_routes=str(saved),original_response_audit=str(original),
        inputs=[entry(p) for p in dict.fromkeys(paths)],
        capture_runtime_roots=[v for v in old['capture_runtime_roots'] if v['view_name'].startswith(('numpy','psutil'))],
        interaction_view=[v for v in old['interaction_view'] if v['view_name'].startswith(('numpy','psutil'))],
        job=dict(name='coupled_compile',worker_path=str(worker.resolve()),result_schema='QWEN_COUPLED_COMPILE_RESULT_V1',
            accepted_decisions=['CLOSE_COUPLED_COMPILER_CONSTRAINTS','CLOSE_COUPLED_COMPILER_ENCODING_OR_NOVEL','ELIGIBLE_FOR_SEPARATE_NATIVE_COMPOSITION'],
            limits=dict(worker_seconds=180,family_seconds=300,OS_bytes=4<<30,output_bytes=512<<20,log_bytes=2<<20)))
    assert len(b['capture_runtime_roots'])==5
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
