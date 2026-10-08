"""Bind finite ALL24 complete-output fit and FIRST saved-output auditor."""
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
    oldpath=doc/'chatbot_whole_assembly_binding_20261008.json'
    assert sha(oldpath)=='3dd279387dcec5bfad25f70fefb5cd682e540cc0c1ffea575433bcee0698e077';old=json.loads(oldpath.read_bytes())
    assemblypath=doc/'chatbot_whole_assembly_20261008.json'
    assembly,assemblyterm,ast=checked(assemblypath,'18d7905032f4a7dc5c48821f157249ddfc5ea6398fc8c81ed12a5ee086468f0b')
    auditpath=doc/'chatbot_whole_assembly_audit_20261008.json'
    audit,auditterm,at=checked(auditpath,'83c5247b0b0387f6be0af919eb303600f8f3c494a39abd5e4921829ad2241885')
    assert audit['decision']=='WHOLE_ASSEMBLY_BYTE_COUNT_INDEPENDENTLY_VERIFIED'
    manifest=receipt(assembly['installed_manifest']['path'],ast)
    assert sha(manifest)=='90cad20a81d98c917ca4bb81c37bcc655689ed8f0f4b27e2ea70afec9a6a0961'
    closure=doc/'chatbot_whole_assembly_terminal_20261008.json'
    assert sha(closure)=='7ada2ebb9a0088eecced80a55eeb1681c36bc722fa7efd839a51eec954cd9840'
    cohortpath=doc/'chatbot_global_cohort_20261008.json'
    cohort,cohortterm,ct=checked(cohortpath,'e3898c2026e65f1edd995b96622c04cc9614f158f2ff6d7ffdcad718dfded4af')
    cauditpath=doc/'chatbot_global_cohort_audit_repair1_20261008.json'
    caudit,cauditterm,cat=checked(cauditpath,'7c3577129a90b9c4b8172ed16cb250729edb17d9345b73cc847b93d02d97280f')
    assert caudit['decision']=='NEW_WHOLE_TRANSFER_COHORT_BYTE_ID_QUALIFIED' and caudit['complete_200_case_cohort']
    fresh=code/'chatbot_whole_fresh_dialogs_v1.json';freshdata=json.loads(fresh.read_bytes())
    endpoint=code/'chatbot_global_endpoint_reserved_v1.json';endpointdata=json.loads(endpoint.read_bytes())
    transfer=code/'chatbot_global_transfer_cases_v1.json';transferdata=json.loads(transfer.read_bytes())
    assert len(freshdata['cases'])==16 and all(len(v['turns'])==2 for v in freshdata['cases'])
    assert (freshdata['max_context_tokens'],freshdata['max_new_tokens_per_turn'],freshdata['EOS'])==(512,64,[151645,151643])
    from chatbot_capture_cases import manifest as old_manifest
    oldcases=old_manifest()['cases']
    signatures=lambda values:{json.dumps(v['messages'],ensure_ascii=False,separators=(',',':')) for v in values}
    other=signatures(transferdata['cases'])|signatures(endpointdata['cases'])|signatures(oldcases)
    firstprompts=[dict(messages=[dict(role='user',content=c['turns'][0]['user'])]) for c in freshdata['cases']]
    assert len(signatures(firstprompts))==16 and not signatures(firstprompts)&other
    paths=[oldpath,assemblypath,assemblyterm,auditpath,auditterm,closure,manifest,
        cohortpath,cohortterm,cauditpath,cauditterm,doc/'chatbot_global_cohort_repair1_terminal_20261008.json',
        *(v['path'] for v in old['inputs']),code/'chatbot_joint_pilot.py',code/'chatbot_capture_cases.py',
        code/'chatbot_whole_output_data.py',code/'chatbot_whole_output_fit.py',code/'chatbot_whole_output_audit.py',
        Path(__file__).resolve(),code/'chatbot_whole_output_terminal.ps1',fresh,endpoint,transfer,
        doc/'CHATBOT_FRESH_BEHAVIOR_CONTRACT_20261008.md',doc/'CHATBOT_WHOLE_OUTPUT_LEARNER_NEXT_20261008.md',
        doc/'CHATBOT_WHOLE_OUTPUT_PROTOCOL_20261008.md',doc/'CHATBOT_WHOLE_ASSEMBLY_RESULT_20261008.md']
    for case in cohort['conversations']:
        for field in ('metadata','logits'):
            path=receipt(case[field+'_path'],ct);assert sha(path)==case[field+'_sha256'];paths.append(path)
    b={key:old[key] for key in ('schema','python','source_directory','source_weights','spec_path','initializer_path','initializer_audit_path','initializer_audit_binding_path')}
    b.update(installed_manifest_path=str(manifest),cohort_path=str(cohortpath),
        excluded_fresh_dialog_path=str(fresh),excluded_endpoint_path=str(endpoint),
        exclusion_scope='No fresh endpoint tokenization/model/answer;16 initial message signatures disjoint from old48/NEW200/endpoint64;semantic domain overlap remains')
    runtime=old['capture_runtime_roots'];view=old['interaction_view']
    if args.stage=='fit':
        worker=code/'chatbot_whole_output_fit.py';name='whole_output_fit';schema='QWEN_WHOLE_OUTPUT_FIT_RESULT_V1'
        decisions=['WHOLE_OUTPUT_FIT_REQUIRE_FIRST_AUDIT','CLOSE_FIXED_WHOLE_OUTPUT_FIT_REQUIRE_FIRST_AUDIT']
        limits=dict(worker_seconds=7200,family_seconds=7800,OS_bytes=16<<30,GPU_allocated_bytes=8<<30,GPU_reserved_bytes=9<<30,output_bytes=6<<30,log_bytes=2<<20)
    else:
        assert args.result is not None and args.result_sha
        raw,term,terminal=checked(args.result,args.result_sha)
        assert raw['new_optimizer_updates']==1280 and raw['new_student_full_forwards']==1680
        original=Path(terminal['command'][terminal['command'].index('--binding')+1]);paths.extend([args.result,term,original])
        for value in terminal['output_manifest']:paths.append(receipt(value['path'],terminal))
        b['fit_result_path']=str(args.result.resolve())
        prefixes=('numpy','psutil','torch','functorch','sympy','mpmath','networkx','typing_extensions','packaging','filelock','jinja2','markupsafe','fsspec','safetensors')
        runtime=[v for v in runtime if v['view_name'].startswith(prefixes)];view=[v for v in view if v['view_name'].startswith(prefixes)]
        worker=code/'chatbot_whole_output_audit.py';name='whole_output_audit';schema='QWEN_WHOLE_OUTPUT_AUDIT_RESULT_V1'
        decisions=['WHOLE_OUTPUT_FIT_INDEPENDENTLY_VERIFIED'];limits=dict(worker_seconds=600,family_seconds=900,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=2<<20)
    b.update(inputs=[entry(p) for p in dict.fromkeys(map(str,paths))],capture_runtime_roots=runtime,interaction_view=view,
        job=dict(name=name,worker_path=str(worker),result_schema=schema,accepted_decisions=decisions,limits=limits))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']),runtime_roots=len(runtime))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('fit','audit'),required=True)
    p.add_argument('--result',type=Path);p.add_argument('--result-sha');p.add_argument('--out',type=Path,required=True);main(p.parse_args())
