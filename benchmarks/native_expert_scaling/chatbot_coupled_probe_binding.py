"""Pin the first unrounded prefix observable or its first independent proof."""
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
    prior_path=doc/'chatbot_coupled_compile_binding_20261008.json';prior=json.loads(prior_path.read_bytes())
    compiled=doc/'chatbot_coupled_compile_20261008.json';raw=json.loads(compiled.read_bytes())
    audit=doc/'chatbot_coupled_compile_audit_20261008.json';audit_terminal=audit.with_suffix('.terminal.json')
    terminal=json.loads(audit_terminal.read_bytes());assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(audit) and all(terminal['gates'].values())
    journal_path=Path(raw['output_arrays']['novel_prefix64_F32']['path']).with_name('novel_prefix64_journal.json')
    journal=json.loads(journal_path.read_bytes());worker=code/'chatbot_coupled_probe.py'
    paths=[prior_path,prior['python'],Path(prior['python']).with_name('python312.dll'),compiled,compiled.with_suffix('.terminal.json'),
        audit,audit_terminal,journal_path,prior['original_response_audit'],*(item['binary_path'] for item in journal),
        worker,Path(__file__).resolve(),code/'chatbot_joint_retained_audit.py',code/'chatbot_interaction_launch.py',code/'chatbot_directional_binding.py',code/'chatbot_directional_launch.py',
        doc/'CHATBOT_COUPLED_PROBE_PROTOCOL_20261008.md']
    b=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=prior['python'],compile_path=str(compiled),journal_path=str(journal_path),original_response_audit=prior['original_response_audit'])
    if args.job=='probe':
        paths.extend(raw['output_arrays'][n]['path'] for n in ('affine_A_F64','affine_bias_F64','shared_g_BF16','shared_u_BF16','shared_b_BF16','novel_prefix64_F32'))
        job=dict(name='coupled_probe',result_schema='QWEN_COUPLED_PROBE_RESULT_V1',accepted_decisions=[
            'UNROUNDED_FIELD_EXTENSION_FAILURE_DOMINATES_CODEC_ON_PREFIX','UNROUNDED_FIELD_ALSO_FAILS_CODEC_CONTRIBUTION_UNRESOLVED','UNROUNDED_PREFIX_BOUND_INCONCLUSIVE_CODEC_REMAINS_OPEN'])
    else:
        probe=doc/'chatbot_coupled_probe_20261008.json';probe_terminal=probe.with_suffix('.terminal.json');pt=json.loads(probe_terminal.read_bytes())
        assert pt['actual_worker_exit_code']==0 and pt['result_sha256']==sha(probe) and all(pt['gates'].values())
        for item in pt['output_manifest']:
            assert sha(item['path'])==item['sha256'];paths.append(item['path'])
        paths.extend([probe,probe_terminal]);b['probe_path']=str(probe)
        job=dict(name='coupled_probe_audit',result_schema='QWEN_COUPLED_PROBE_AUDIT_RESULT_V1',accepted_decisions=['SAVED_UNROUNDED_PREFIX_EXACT_BOUND_INDEPENDENTLY_VERIFIED'])
    b.update(inputs=[entry(p) for p in dict.fromkeys(paths)],
        capture_runtime_roots=prior['capture_runtime_roots'],interaction_view=prior['interaction_view'],
        job=dict(job,worker_path=str(worker.resolve()),limits=dict(worker_seconds=45,family_seconds=90,OS_bytes=512<<20,output_bytes=2<<20,log_bytes=2<<20)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',choices=['probe','audit'],required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
