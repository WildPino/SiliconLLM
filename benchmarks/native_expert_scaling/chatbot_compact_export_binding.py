"""Bind eligible fixed fit, its actual FIRST auditor and one native export."""
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
    fit,fit_term,ft=checked(args.fit,args.fit_sha);audit,audit_term,at=checked(args.audit,args.audit_sha)
    assert fit['decision']=='WHOLE_OUTPUT_FIT_REQUIRE_FIRST_AUDIT' and all(fit['transfer_gates'].values())
    assert audit['decision']=='WHOLE_OUTPUT_FIT_INDEPENDENTLY_VERIFIED' and audit['candidate_transfer_eligible']
    assert fit['new_optimizer_updates']==1280 and audit['final_trainable_elements_checked']==165150720
    old_path=Path(ft['command'][ft['command'].index('--binding')+1]);old=json.loads(old_path.read_bytes())
    assert sha(old_path)==ft['command'][ft['command'].index('--binding-sha')+1]
    audit_binding_path=Path(at['command'][at['command'].index('--binding')+1]);ab=json.loads(audit_binding_path.read_bytes())
    assert sha(audit_binding_path)==at['command'][at['command'].index('--binding-sha')+1]
    assert Path(ab['fit_result_path']).resolve()==args.fit.resolve()
    adopted=next(v for v in ab['inputs'] if Path(v['path']).resolve()==args.fit.resolve())
    assert adopted['sha256']==args.fit_sha and adopted['bytes']==args.fit.stat().st_size
    assert all(fit['procedure_gates'].values()) and all(audit['procedure_gates'].values())
    paths=[args.fit,fit_term,args.audit,audit_term,old_path,audit_binding_path,old['python'],
        Path(old['python']).with_name('python312.dll'),old['source_weights'],old['installed_manifest_path'],
        code/'chatbot_compact_export.py',code/'chatbot_compact_export_worker.py',Path(__file__).resolve(),
        code/'chatbot_interaction_launch.py',code/'chatbot_directional_binding.py',
        code/'chatbot_null_prior_kernel_binding.py',code/'chatbot_directional_launch.py',
        doc/'CHATBOT_COMPACT_NATIVE_NEXT_20261008.md',doc/'CHATBOT_COMPACT_EXPORT_PROTOCOL_20261008.md']
    source=Path(old['source_directory'])
    paths.extend(source/name for name in ('config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt'))
    assert [v['layer'] for v in fit['final_checkpoints']]==list(range(24))
    for value in fit['final_checkpoints']:
        path=receipt(value['path'],ft);assert sha(path)==value['sha256'] and path.stat().st_size==value['bytes'];paths.append(path)
    # Source bytes and installed-manifest receipt are adopted from the original
    # fit binding; no replacement source checkpoint or assembly is allowed.
    for path in [Path(old['source_weights']),Path(old['installed_manifest_path']),*(source/name for name in
                 ('config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt'))]:
        previous=next(v for v in old['inputs'] if Path(v['path'])==path)
        assert sha(path)==previous['sha256'] and path.stat().st_size==previous['bytes']
    prefixes=('numpy','psutil','torch','functorch','sympy','mpmath','networkx','typing_extensions','packaging','filelock','jinja2','markupsafe','fsspec','safetensors')
    runtime=[v for v in old['capture_runtime_roots'] if v['view_name'].startswith(prefixes)]
    view=[v for v in old['interaction_view'] if v['view_name'].startswith(prefixes)]
    binding=dict(schema=old['schema'],python=old['python'],source_weights=old['source_weights'],source_directory=old['source_directory'],
        installed_manifest_path=old['installed_manifest_path'],fit_result_path=str(args.fit.resolve()),fit_audit_result_path=str(args.audit.resolve()),
        inputs=[entry(p) for p in dict.fromkeys(map(str,paths))],capture_runtime_roots=runtime,interaction_view=view,
        job=dict(name='compact_export',worker_path=str(code/'chatbot_compact_export_worker.py'),result_schema='QWEN_COMPACT_EXPORT_RESULT_V1',
            accepted_decisions=['COMPACT_NATIVE_EXPORT_REQUIRE_FIRST_CODEC_AUDIT'],
            limits=dict(worker_seconds=180,family_seconds=360,OS_bytes=4<<30,output_bytes=768<<20,log_bytes=2<<20)))
    if args.stage=='audit':
        assert args.export is not None and args.export_sha
        exported,export_term,et=checked(args.export,args.export_sha)
        assert exported['decision']=='COMPACT_NATIVE_EXPORT_REQUIRE_FIRST_CODEC_AUDIT' and all(exported['procedure_gates'].values())
        export_binding_path=Path(et['command'][et['command'].index('--binding')+1]);eb=json.loads(export_binding_path.read_bytes())
        assert sha(export_binding_path)==et['command'][et['command'].index('--binding-sha')+1]
        assert Path(eb['fit_result_path']).resolve()==args.fit.resolve() and Path(eb['fit_audit_result_path']).resolve()==args.audit.resolve()
        for path,want in ((args.fit,args.fit_sha),(args.audit,args.audit_sha)):
            adopted=next(v for v in eb['inputs'] if Path(v['path']).resolve()==path.resolve())
            assert adopted['sha256']==want and adopted['bytes']==path.stat().st_size
        paths.extend([args.export,export_term,export_binding_path,code/'chatbot_compact_export_audit.py'])
        for item in et['output_manifest']:paths.append(receipt(item['path'],et))
        binding.update(export_result_path=str(args.export.resolve()),inputs=[entry(p) for p in dict.fromkeys(map(str,paths))],
            job=dict(name='compact_export_audit',worker_path=str(code/'chatbot_compact_export_audit.py'),result_schema='QWEN_COMPACT_EXPORT_AUDIT_RESULT_V1',
                accepted_decisions=['COMPACT_EXPORT_BYTES_INDEPENDENTLY_VERIFIED'],
                limits=dict(worker_seconds=180,family_seconds=360,OS_bytes=4<<30,output_bytes=1<<20,log_bytes=2<<20)))
    write_once(args.out,binding);print(json.dumps(dict(sha256=sha(args.out),inputs=len(binding['inputs']),runtime_roots=len(runtime))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('export','audit'),default='export')
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--fit-sha',required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--audit-sha',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--export',type=Path);p.add_argument('--export-sha')
    main(p.parse_args())
