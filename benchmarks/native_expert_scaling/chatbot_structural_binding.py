"""Bind first changed full-J learner, using saved source information only."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once
from chatbot_directional_binding import entry,receipt


def qualified(path):
    path=Path(path);terminal_path=path.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert terminal['actual_worker_exit_code']==0 and sha(path)==terminal['result_sha256'] and all(terminal['gates'].values())
    return json.loads(path.read_bytes()),terminal,terminal_path


def main(args):
    assert not args.out.exists() and sha(args.prior)==args.prior_sha
    old=json.loads(args.prior.read_bytes());assert old['schema']=='QWEN_DIRECTIONAL_BINDING_V1' and old['job']['name']=='jacobian'
    code=ROOT/'benchmarks/native_expert_scaling';doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
    jpath=doc/'chatbot_directional_jacobian_20261008.json';jac,jterm,jterminal=qualified(jpath)
    apath=doc/'chatbot_directional_audit_20261008.json';audit,aterm,aterminal=qualified(apath)
    assert audit['scientific_decision']==jac['decision']=='DIRECTIONAL_DEFICIT_SUPPORTED_NEW_STRUCTURAL_FIT_JUSTIFIED'
    basepath=doc/'chatbot_joint_baseline_20261007.json';base,bterm,bterminal=qualified(basepath)
    init=receipt(ROOT/'results/native_expert_scaling/chatbot_joint_baseline_20261007/E16.initial.pt',bterm)
    warm=receipt(old['student_checkpoint'],bterm)
    jdir=ROOT/'results/native_expert_scaling/chatbot_directional_jacobian_20261008'
    sources={n:receipt(jdir/file,jterm) for n,file in dict(source_J_path='source_J_F64.npy',Q_path='selector_rowspace_Q_F64.npy',directions_path='directions_F64.npy').items()}
    original_path=doc/'chatbot_joint_baseline_binding_20261007.json';original=json.loads(original_path.read_bytes())
    assert original['schema']=='QWEN_JOINT_LAYER12_BASELINE_BINDING_V1'
    adoption=json.loads(Path(old['adoption_path']).read_bytes())
    paths=[args.prior,old['python'],Path(old['python']).with_name('python312.dll'),jpath,jterminal,apath,aterminal,
        basepath,bterminal,init,warm,old['plan_path'],doc/'chatbot_directional_plan_provenance_20261008.json',
        original_path,old['adoption_path'],old['saved_E16_routes'],original['spec_path'],*sources.values(),
        *(v[field] for v in adoption['cases'] for field in ('binary_path','journal_path')),
        code/'chatbot_structural_fit.py',Path(__file__).resolve(),code/'chatbot_directional_launch.py',code/'chatbot_directional_binding.py',
        code/'chatbot_interaction_launch.py',code/'chatbot_joint_pilot.py',code/'chatbot_compact_geometry.py',
        doc/'CHATBOT_STRUCTURAL_FIT_PROTOCOL_20261008.md']
    binding=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=old['python'],plan_path=old['plan_path'],adoption_path=old['adoption_path'],
        prior_jacobian_result=str(jpath),baseline_result=str(basepath),warm_checkpoint=str(warm),reference_checkpoint=str(init),
        saved_E16_routes=old['saved_E16_routes'],spec_path=original['spec_path'],**{n:str(v) for n,v in sources.items()},
        inputs=[entry(p) for p in dict.fromkeys(paths)],capture_runtime_roots=old['capture_runtime_roots'],interaction_view=old['interaction_view'],
        new_original_BF16_full_forwards=0,new_source_jacobians=0,whole_chatbot_quality_or_rate_qualified=False,
        job=dict(name='structural_fit',worker_path=str((code/'chatbot_structural_fit.py').resolve()),result_schema='QWEN_STRUCTURAL_FIT_RESULT_V1',
            accepted_decisions=['LOCAL_F32_STRUCTURAL_FIDELITY_AVAILABLE_ENCODING_NOT_ADMITTED','CLOSE_ONE_STRUCTURAL_FIT_NO_WHOLE_PROMOTION'],
            limits=dict(worker_seconds=360,family_seconds=540,OS_bytes=4<<30,GPU_allocated_bytes=4<<30,GPU_reserved_bytes=6<<30,output_bytes=512<<20,log_bytes=2<<20)))
    write_once(args.out,binding);print(json.dumps(dict(binding=str(args.out),sha256=sha(args.out),files=len(binding['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--prior-sha',required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
