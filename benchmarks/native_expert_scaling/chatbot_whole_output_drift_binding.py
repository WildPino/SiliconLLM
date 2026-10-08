"""Bind the NEW saved-only final/online trajectory diagnostic."""
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
    assert not args.out.exists();code=ROOT/'benchmarks/native_expert_scaling';doc=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
    fit_path=doc/'chatbot_whole_output_fit_20261008.json';fit,fit_term,ft=checked(fit_path,'34b8a7b317c6435b1ead9298ae75f32116ffd4f39606c402992e15bcd0c5387e')
    audit_path=doc/'chatbot_whole_output_audit_20261008.json';audit,audit_term,at=checked(audit_path,'7609daee3d927d46bdff69b815b35f314f70894cea80e07227c7574e622c1e72')
    assert fit['decision']=='CLOSE_FIXED_WHOLE_OUTPUT_FIT_REQUIRE_FIRST_AUDIT' and not audit['candidate_transfer_eligible']
    assert audit['decision']=='WHOLE_OUTPUT_FIT_INDEPENDENTLY_VERIFIED' and all(audit['procedure_gates'].values())
    old_path=doc/'chatbot_whole_output_audit_binding_20261008.json'
    assert sha(old_path)=='2f863b86e464e3b9db663d5eeb5dfa927275dce3bba44bb6da7438897494506d'
    old=json.loads(old_path.read_bytes());paths=[fit_path,fit_term,audit_path,audit_term,old_path,
        old['python'],Path(old['python']).with_name('python312.dll'),
        doc/'chatbot_whole_output_terminal_20261008.json',code/'chatbot_whole_output_drift.py',Path(__file__).resolve(),
        code/'chatbot_interaction_launch.py',code/'chatbot_directional_binding.py',code/'chatbot_null_prior_kernel_binding.py',
        code/'chatbot_directional_launch.py',doc/'CHATBOT_WHOLE_OUTPUT_DRIFT_PROTOCOL_20261008.md']
    assert sha(doc/'chatbot_whole_output_terminal_20261008.json')=='1f72bea1b75191a107127d72de53d86796c12817a4d7b4f4d290641d3c1fa777'
    for field in ('initial_cases','final_cases','updates'):paths.append(receipt(fit['outputs'][field]['path'],ft))
    binding=dict(schema=old['schema'],python=old['python'],fit_result_path=str(fit_path),fit_audit_result_path=str(audit_path),
        inputs=[entry(p) for p in dict.fromkeys(map(str,paths))],
        capture_runtime_roots=[v for v in old['capture_runtime_roots'] if v['view_name'].startswith('psutil')],
        interaction_view=[v for v in old['interaction_view'] if v['view_name'].startswith('psutil')],
        job=dict(name='whole_output_drift',worker_path=str(code/'chatbot_whole_output_drift.py'),result_schema='QWEN_WHOLE_OUTPUT_DRIFT_RESULT_V1',
            accepted_decisions=['SAVED_WHOLE_OUTPUT_DRIFT_DIAGNOSED_NOT_CAUSAL'],
            limits=dict(worker_seconds=30,family_seconds=90,OS_bytes=256<<20,output_bytes=256<<10,log_bytes=2<<20)))
    write_once(args.out,binding);print(json.dumps(dict(sha256=sha(args.out),files=len(binding['inputs']),runtime_roots=len(binding['capture_runtime_roots']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
