"""Bound one-shot native export; no forward/backward, optimizer or compiler."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)));torch=None
    r=dict(schema='QWEN_COMPACT_EXPORT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),procedure_gates={},
        new_original_BF16_full_forwards=0,new_student_full_forwards=0,new_backward_calls=0,new_optimizer_updates=0,
        new_endpoint_queries=0,native_builds_or_forwards=0)
    def guard():
        assert time.monotonic()-start<=180 and proc.memory_info().peak_wset<=4<<30,'export time/OS'
        assert torch is None or not torch.cuda.is_initialized(),'CPU-only export'
        assert not proc.children(recursive=True),'unexpected export descendants'
        if args.directory.exists():
            assert sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=768<<20,'export output cap'
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        binding=json.loads(args.binding.read_bytes());assert binding['job']['name']=='compact_export'
        for item in binding['inputs']:
            path=Path(item['path']);assert str(path.resolve())==item['resolved_path']
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path'];guard()
        def no_subprocess(event,arguments):
            if event=='subprocess.Popen':raise RuntimeError('Export forbids subprocess')
        sys.addaudithook(no_subprocess)
        import torch as torch_module
        torch=torch_module;assert torch.__version__=='2.6.0+cu124';torch.set_num_threads(6);torch.set_num_interop_threads(1)
        from chatbot_compact_export import export_compact
        report=export_compact(binding['source_weights'],binding['fit_result_path'],binding['fit_audit_result_path'],
            args.directory,binding['installed_manifest_path'],guard)
        # Complete reproducible interaction bundle alongside the native archive.
        # The dense source config is named source_config.json explicitly; it is
        # provenance, not a claim that HF can load this custom compact geometry.
        files=[];source=Path(binding['source_directory'])
        for name,target in [('config.json','source_config.json'),('generation_config.json','generation_config.json'),
                            ('tokenizer_config.json','tokenizer_config.json'),('tokenizer.json','tokenizer.json'),
                            ('vocab.json','vocab.json'),('merges.txt','merges.txt')]:
            destination=args.directory/target;assert not destination.exists();shutil.copyfile(source/name,destination)
            assert sha(destination)==sha(source/name)
            files.append(dict(path=str(destination.resolve()),bytes=destination.stat().st_size,sha256=sha(destination)))
            guard()
        r.update(export=report,interaction_files=files,decision='COMPACT_NATIVE_EXPORT_REQUIRE_FIRST_CODEC_AUDIT',
            scope='Exported bytes only; encoded/native quality, physical DRAM and throughput pending')
        r['procedure_gates'].update(eligible_fixed_fit_and_independent_audit=True,
            exact_source_core_and_NEW_router=True,finite_BF16_GUD_codec=True,actual_catalog_and_blob_bound=True,
            source_interaction_bundle_exact=True,no_model_optimizer_compiler_or_endpoint_execution=True)
        guard();r.update(compute_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        assert len(json.dumps(r,separators=(',',':')).encode('utf8'))<=1<<20
        write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],archive=report['archive'],resources=r['compute_seconds'])),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
