"""Bind adopted NEW operands, warm BYTE inputs and first initializer/auditor."""
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
    oldpath=doc/'chatbot_joint_baseline_binding_20261007.json'
    assert sha(oldpath)=='a3265dbeceaf7de82a903bf7735beba88628d7814be34fb1a46d459424b00b5b';old=json.loads(oldpath.read_bytes())
    source=Path(old['source_weights']);source_entry=next(v for v in old['inputs'] if Path(v['path'])==source)
    assert sha(source)==source_entry['sha256'] and source.stat().st_size==source_entry['bytes']
    cohortpath=doc/'chatbot_global_cohort_20261008.json'
    cohort,cohortterm,ct=checked(cohortpath,'e3898c2026e65f1edd995b96622c04cc9614f158f2ff6d7ffdcad718dfded4af')
    auditpath=doc/'chatbot_global_cohort_audit_repair1_20261008.json'
    audit,auditterm,at=checked(auditpath,'7c3577129a90b9c4b8172ed16cb250729edb17d9345b73cc847b93d02d97280f')
    assert audit['decision']=='NEW_WHOLE_TRANSFER_COHORT_BYTE_ID_QUALIFIED' and audit['complete_200_case_cohort']
    assert audit['adopted_x_rows_PER_LAYER']==12833 and len(cohort['conversations'])==200
    previouspath=doc/'chatbot_whole_initializer_20261008.json'
    previous,previousterm,pt=checked(previouspath,'6e5ab7c12042fd8778786fc393d51b3d93a445e936c89daa9fecb37aab712fd2')
    priorauditpath=doc/'chatbot_whole_initializer_audit_20261008.json'
    prioraudit,priorauditterm,pat=checked(priorauditpath,'264add971be9a2727422856d5b23e21e21c087fd2edc8479f6079752ce53786e')
    assert prioraudit['decision']=='PARTIAL_INITIALIZER_SUPPORT_STOP_INDEPENDENTLY_VERIFIED'
    initial0=receipt(previous['layers'][0]['initial_checkpoint']['path'],pt)
    geometry1=receipt(previous['layers'][1]['witness']['path'],pt)
    assert sha(initial0)=='b20a795df0d0f5cf5e73a06dc4d1f95fe080d375c4d9aac709ff7a176d115291'
    basepath=doc/'chatbot_joint_baseline_20261007.json';baseterm=basepath.with_suffix('.terminal.json')
    bt=json.loads(baseterm.read_bytes());assert bt['actual_worker_exit_code']==0 and sha(basepath)==bt['result_sha256'] and all(bt['gates'].values())
    initial12=receipt(ROOT/'results/native_expert_scaling/chatbot_joint_baseline_20261007/E16.initial.pt',bt)
    indices12=receipt(initial12.with_name('E16.initializer.json'),bt)
    assert sha(initial12)=='d85c4b292fcc54d7ad8bd8777e5158354b6d043da62627ed4c04dc3de856822d'
    assert sha(indices12)=='8e1c0409a85c7cc3840b7dcb972a22caa7620dfd6466890fe2cb098306bf1bd0'
    spec=code/'chatbot_compact_spec.json';assert sha(spec)=='a5850df0172d9f9152e8391a3588ed5cfe9fb184d049955a161e28b4786cd9d0'
    paths=[oldpath,old['python'],Path(old['python']).with_name('python312.dll'),source,spec,
        cohortpath,cohortterm,auditpath,auditterm,doc/'chatbot_global_cohort_repair1_terminal_20261008.json',
        previouspath,previousterm,priorauditpath,priorauditterm,basepath,baseterm,initial0,geometry1,initial12,indices12,
        Path(__file__).resolve(),code/'chatbot_interaction_launch.py',code/'chatbot_directional_binding.py',code/'chatbot_directional_launch.py',
        code/'chatbot_null_prior_kernel_binding.py',code/'chatbot_compact_geometry.py',code/'chatbot_joint_pilot.py',code/'chatbot_whole_initializer.py',
        code/'chatbot_new_cohort_initializer.py',code/'chatbot_new_cohort_initializer_audit.py',code/'chatbot_new_cohort_initializer_terminal.ps1',
        doc/'CHATBOT_NEW_COHORT_INITIALIZER_NEXT_20261008.md',doc/'CHATBOT_NEW_COHORT_INITIALIZER_PROTOCOL_20261008.md']
    for case in cohort['conversations']:
        path=receipt(case['x_path'],ct);assert sha(path)==case['x_sha256'];paths.append(path)
    prefixes=('numpy','psutil','torch','functorch','sympy','mpmath','networkx','typing_extensions','packaging','filelock','jinja2','markupsafe','fsspec','safetensors')
    runtime=[v for v in old['capture_runtime_roots'] if v['view_name'].startswith(prefixes)]
    view=[v for v in old['interaction_view'] if v['view_name'].startswith(prefixes)]
    b=dict(schema='QWEN_DIRECTIONAL_BINDING_V1',python=old['python'],source_weights=str(source),
        cohort_result_path=str(cohortpath),cohort_audit_path=str(auditpath),spec_path=str(spec),
        warm_geometry_inputs={str(i):entry(p) for i,p in ((0,initial0),(1,geometry1),(12,initial12))},
        warm_initial_inputs={str(i):entry(p) for i,p in ((0,initial0),(12,initial12))},
        capture_runtime_roots=runtime,interaction_view=view)
    if args.stage=='initialize':
        worker=code/'chatbot_new_cohort_initializer.py';name='new_cohort_initializer';schema='QWEN_NEW_COHORT_INITIALIZER_RESULT_V1'
        decisions=['NEW_COHORT_ALL24_INITIALIZED_REQUIRE_FIRST_AUDIT','CLOSE_NEW_COHORT_ALL24_ON_SUPPORT']
        limits=dict(worker_seconds=900,family_seconds=1500,OS_bytes=16<<30,GPU_allocated_bytes=8<<30,GPU_reserved_bytes=9<<30,output_bytes=1<<30,log_bytes=2<<20)
    else:
        assert args.result is not None and args.result_sha
        raw,term,terminal=checked(args.result,args.result_sha)
        assert raw['schema']=='QWEN_NEW_COHORT_INITIALIZER_RESULT_V1' and all(raw['procedure_gates'].values())
        assert raw['decision'] in ('NEW_COHORT_ALL24_INITIALIZED_REQUIRE_FIRST_AUDIT','CLOSE_NEW_COHORT_ALL24_ON_SUPPORT')
        original=Path(terminal['command'][terminal['command'].index('--binding')+1])
        b['initializer_result_path']=str(args.result.resolve());paths.extend([args.result,term,original])
        for item in terminal['output_manifest']:paths.append(receipt(item['path'],terminal))
        worker=code/'chatbot_new_cohort_initializer_audit.py';name='new_cohort_initializer_audit';schema='QWEN_NEW_COHORT_INITIALIZER_AUDIT_RESULT_V1'
        decisions=['NEW_COHORT_ALL24_INITIALIZER_INDEPENDENTLY_VERIFIED','NEW_COHORT_PARTIAL_SUPPORT_STOP_INDEPENDENTLY_VERIFIED']
        limits=dict(worker_seconds=300,family_seconds=600,OS_bytes=8<<30,output_bytes=1<<20,log_bytes=2<<20)
    paths.append(worker);b['inputs']=[entry(p) for p in dict.fromkeys(paths)]
    b['job']=dict(name=name,worker_path=str(worker),result_schema=schema,accepted_decisions=decisions,limits=limits)
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']),runtime_roots=len(runtime))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('initialize','audit'),required=True)
    p.add_argument('--result',type=Path);p.add_argument('--result-sha');p.add_argument('--out',type=Path,required=True);main(p.parse_args())
