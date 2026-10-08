"""Freeze NEW global cohort producer or FIRST saved data adoption audit."""
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
    oldsource=doc/'chatbot_source_batch4_binding_20261007.json';oldbase=doc/'chatbot_joint_baseline_binding_20261007.json'
    assert sha(oldsource)=='b595f2945dc80db21c4e399eadff33b1a83f41cf7074b49064c72c2e1048571e'
    assert sha(oldbase)=='a3265dbeceaf7de82a903bf7735beba88628d7814be34fb1a46d459424b00b5b'
    source=json.loads(oldsource.read_bytes());base=json.loads(oldbase.read_bytes())
    manifest=code/'chatbot_global_transfer_cases_v1.json';endpoint=code/'chatbot_global_endpoint_reserved_v1.json'
    m=json.loads(manifest.read_bytes());e=json.loads(endpoint.read_bytes())
    assert m['schema']==e['schema']=='QWEN_GLOBAL_COHORT_CASES_V1' and len(m['cases'])==200 and len(e['cases'])==64
    assert sum(c['split']=='fit' for c in m['cases'])==160 and sum(c['split']=='development' for c in m['cases'])==40
    oldadopt=doc/'chatbot_source_adoption_final_20261007.json';assert sha(oldadopt)=='1b54b8fd3f82f93caa35c1aa05b26abfbce555f05165b1428dc4a14536cad7d9'
    a=json.loads(oldadopt.read_bytes())
    def key(c):return json.dumps([c['mode'],c['messages']],ensure_ascii=False,sort_keys=True,separators=(',',':'))
    sets=[{key(c) for c in v['cases']} for v in (m,e,a)]
    assert len(sets[0])==200 and len(sets[1])==64 and not any(x&y for i,x in enumerate(sets) for y in sets[i+1:])
    sourcefiles=[v for v in source['inputs'] if Path(v['path']).parent==Path(source['source_directory'])]
    assert {Path(v['path']).name for v in sourcefiles}=={'model.safetensors','config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt'}
    for item in sourcefiles:assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
    paths=[oldsource,oldbase,source['python'],Path(source['python']).with_name('python312.dll'),manifest,endpoint,oldadopt,
        *(v['path'] for v in sourcefiles),Path(__file__).resolve(),code/'chatbot_global_cohort_cases.py',code/'chatbot_capture_cases.py',
        code/'chatbot_interaction.py',code/'chatbot_interaction_launch.py',code/'chatbot_directional_binding.py',code/'chatbot_global_cohort_launch.py',
        doc/'CHATBOT_GLOBAL_COHORT_NEXT_20261008.md',doc/'CHATBOT_GLOBAL_COHORT_PROTOCOL_20261008.md']
    b=dict(schema='QWEN_GLOBAL_COHORT_BINDING_V1',python=source['python'],source_directory=source['source_directory'],
        manifest_path=str(manifest),endpoint_path=str(endpoint),old_adoption_path=str(oldadopt),
        tokenizer_json_path=str(Path(source['source_directory'])/'tokenizer.json'))
    if args.stage=='collect':
        worker=code/'chatbot_global_cohort_collect.py';runtime=base['capture_runtime_roots'];view=base['interaction_view']
        job=dict(name='global_cohort_collect',result_schema='QWEN_GLOBAL_COHORT_COLLECT_RESULT_V1',allow_NEW_original_full_forwards=True,
            accepted_decisions=['NEW_WHOLE_TRANSFER_COHORT_REQUIRE_FIRST_BYTE_ID_AUDIT'],
            limits=dict(worker_seconds=900,family_seconds=1500,OS_bytes=16<<30,GPU_allocated_bytes=8<<30,GPU_reserved_bytes=9<<30,output_bytes=3<<30,log_bytes=4<<20))
    else:
        assert args.result is not None and args.result_sha and sha(args.result)==args.result_sha
        raw=json.loads(args.result.read_bytes());assert raw['schema']=='QWEN_GLOBAL_COHORT_COLLECT_RESULT_V1' and raw['conversations']
        if 'fault' in raw:
            stem=Path(str(args.result).removesuffix('.failure.json')+'.json');term=stem.with_suffix('.launcher_failure.json')
        else:term=args.result.with_suffix('.terminal.json')
        terminal=json.loads(term.read_bytes());assert terminal['worker_instance']['pid']==raw['process_instance']['pid']
        if 'fault' not in raw:assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==args.result_sha and all(terminal['gates'].values())
        priorbinding=Path(terminal['command'][terminal['command'].index('--binding')+1])
        paths.extend([args.result,term,priorbinding,Path(raw['tokenized_inputs']['path'])])
        assert sha(raw['tokenized_inputs']['path'])==raw['tokenized_inputs']['sha256']
        for summary in raw['conversations']:
            for name in ('metadata','x','logits','journal'):
                path=Path(summary[name+'_path']);assert sha(path)==summary[name+'_sha256'];paths.append(path)
        b.update(collector_result_path=str(args.result.resolve()),collector_actual_worker_exit_code=terminal['actual_worker_exit_code'])
        prefixes=('numpy','psutil','regex');runtime=[v for v in base['capture_runtime_roots'] if v['view_name'].startswith(prefixes)]
        view=[v for v in base['interaction_view'] if v['view_name'].startswith(prefixes)]
        worker=code/'chatbot_global_cohort_audit.py'
        job=dict(name='global_cohort_audit',result_schema='QWEN_GLOBAL_COHORT_AUDIT_RESULT_V1',allow_NEW_original_full_forwards=False,
            accepted_decisions=['NEW_WHOLE_TRANSFER_COHORT_BYTE_ID_QUALIFIED','NEW_COHORT_COMPLETE_CASE_PREFIX_ADOPTED_FULL_COHORT_INCOMPLETE'],
            limits=dict(worker_seconds=300,family_seconds=600,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=4<<20))
    paths.append(worker);b.update(inputs=[entry(p) for p in dict.fromkeys(paths)],capture_runtime_roots=runtime,interaction_view=view,
        job=dict(job,worker_path=str(worker)))
    write_once(args.out,b);print(json.dumps(dict(sha256=sha(args.out),files=len(b['inputs']),runtime_roots=len(runtime))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('collect','audit'),required=True)
    p.add_argument('--result',type=Path);p.add_argument('--result-sha');p.add_argument('--out',type=Path,required=True);main(p.parse_args())
