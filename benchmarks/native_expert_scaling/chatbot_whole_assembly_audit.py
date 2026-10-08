"""FIRST CPU-only installed-tensor digest/count/allocation receipt verification.

Does not execute source/student/HF/geometry/optimizer or GPU kernels. Actual
device allocation/zeros/liveness are measured producer assertions, not CPU
re-observations of the exited CUDA process.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)));stage='binding'
    r=dict(schema='QWEN_WHOLE_ASSEMBLY_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_student_full_forwards=0,new_optimizer_updates=0,new_endpoint_queries=0)
    def guard():
        assert time.monotonic()-start<=180 and proc.memory_info().peak_wset<=4<<30
        assert not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='whole_assembly_audit'
        for entry in b['inputs']:
            path=Path(entry['path']);assert str(path.resolve())==entry['resolved_path']
            assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'];guard()
        def no_subprocess(event,arguments):
            if event=='subprocess.Popen':raise RuntimeError('Assembly audit forbids subprocess')
        sys.addaudithook(no_subprocess)
        import torch
        assert torch.__version__=='2.6.0+cu124' and not torch.cuda.is_initialized()
        args.directory.mkdir();raw=json.loads(Path(b['assembly_result_path']).read_bytes())
        terminal=json.loads(Path(b['assembly_terminal_path']).read_bytes())
        assert terminal['result_sha256']==sha(b['assembly_result_path']) and terminal['actual_worker_exit_code']==0
        assert all(terminal['gates'].values()) and all(raw['procedure_gates'].values())
        assert raw['decision']=='WHOLE_ASSEMBLED_AND_ALLOCATED_REQUIRE_FIRST_BYTE_COUNT_AUDIT'
        item=raw['installed_manifest'];assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
        manifest=json.loads(Path(item['path']).read_bytes());assert manifest['schema']=='QWEN_WHOLE_ASSEMBLY_TENSOR_MANIFEST_V1'
        assert manifest['head_alias']=='model.embed_tokens.weight' and manifest['checkpoint_mode']=='nonreentrant' and manifest['cache'] is False
        assert (manifest['dense_source_MLP_modules_collected'],manifest['dense_source_MLP_parameters_collected'])==(24,72)
        params=manifest['parameters'];buffers=manifest['buffers'];states=manifest['optimizer']
        source_path=Path(b['source_weights'])
        with source_path.open('rb') as stream:
            header=json.loads(stream.read(struct.unpack('<Q',stream.read(8))[0]));base=stream.tell()
            core_names={name for name in header if name!='__metadata__' and '.mlp.' not in name}
            assert len(core_names)==218
            assert {n for n,v in params.items() if not v['trainable']}==core_names
            for name in sorted(core_names):
                entry=params[name];src=header[name];assert src['dtype']=='BF16' and entry['dtype']=='torch.bfloat16' and entry['shape']==src['shape']
                begin,end=src['data_offsets'];assert end-begin==entry['bytes'];stream.seek(base+begin)
                h=hashlib.sha256();remaining=end-begin
                while remaining:
                    block=stream.read(min(1<<20,remaining));assert block;h.update(block);remaining-=len(block)
                assert h.hexdigest()==entry['sha256'],name;guard()
        def tensor_sha(tensor):
            assert tensor.device.type=='cpu' and tensor.dtype==torch.float32 and tensor.is_contiguous()
            return hashlib.sha256(memoryview(tensor.reshape(-1).view(torch.uint8).numpy())).hexdigest()
        initializer=json.loads(Path(b['initializer_path']).read_bytes());checked_elements=0;expected_params=set();expected_buffers=set()
        for layer in initializer['layers']:
            li=layer['layer'];prefix=f'model.layers.{li}.mlp.compact.'
            routing=torch.load(layer['witness']['path'],map_location='cpu',weights_only=True)
            initial=torch.load(layer['initial_checkpoint']['path'],map_location='cpu',weights_only=True)
            for field in ('projection','parent_centers','child_centers','parent_norms','child_norms'):
                name=prefix+field;entry=buffers[name];value=routing[field]
                assert entry['dtype']=='torch.float32' and entry['shape']==list(value.shape) and entry['sha256']==tensor_sha(value)
                expected_buffers.add(name)
            for field in ('shared_g','shared_u','shared_b','leaf_g','leaf_u','leaf_b'):
                indexes=range(16) if field.startswith('leaf_') else (None,)
                for leaf in indexes:
                    value=initial[field] if leaf is None else initial[field][leaf].contiguous()
                    name=prefix+field+('' if leaf is None else '.'+str(leaf));entry=params[name]
                    assert entry['trainable'] and entry['dtype']=='torch.float32' and entry['shape']==list(value.shape)
                    assert entry['sha256']==tensor_sha(value),name
                    expected_params.add(name);checked_elements+=value.numel()
            del routing,initial;guard()
        assert len(expected_params)==1224 and checked_elements==165150720
        assert set(params)==core_names|expected_params and set(states)==expected_params
        extra=set(buffers)-expected_buffers
        assert extra=={'model.rotary_emb.inv_freq','model.rotary_emb.original_inv_freq'}
        for name in extra:
            assert buffers[name]['shape']==[32] and buffers[name]['dtype'] in ('torch.float32','torch.bfloat16')
            assert len(buffers[name]['sha256'])==64
        assert buffers['model.rotary_emb.inv_freq']['sha256']==buffers['model.rotary_emb.original_inv_freq']['sha256']
        gpu_ranges=[];cpu_ranges=[]
        def record_check(entry,target):
            size={'torch.float32':4,'torch.bfloat16':2}[entry['dtype']]
            assert math.prod(entry['shape'])==entry['elements'] and entry['bytes']==entry['elements']*size and entry['bytes']>0
            assert entry['device']==('cuda:0' if target is gpu_ranges else 'cpu') and entry['pointer']>0
            target.append((entry['pointer'],entry['pointer']+entry['bytes']))
        for entry in list(params.values())+list(buffers.values()):record_check(entry,gpu_ranges)
        for name,state in states.items():
            assert set(state)=={'gradient','exp_avg','exp_avg_sq','step'}
            for field,entry in state.items():
                assert entry['all_bits_zero'] is True and entry['dtype']=='torch.float32'
                assert entry['shape']==([] if field=='step' else params[name]['shape'])
                record_check(entry,cpu_ranges if field=='step' else gpu_ranges)
        for sequence in (gpu_ranges,cpu_ranges):
            sequence.sort();assert all(left[1]<=right[0] for left,right in zip(sequence,sequence[1:]))
        allocation=manifest['allocation'];assert raw['allocation']==allocation
        expected=dict(core_BF16_bytes=sum(params[n]['bytes'] for n in core_names),
            trainable_F32_bytes=sum(params[n]['bytes'] for n in expected_params),
            router_F32_bytes=sum(buffers[n]['bytes'] for n in expected_buffers),
            additional_HF_buffer_bytes=sum(buffers[n]['bytes'] for n in extra),CPU_step_F32_bytes=sum(e-s for s,e in cpu_ranges))
        assert (expected['core_BF16_bytes'],expected['trainable_F32_bytes'],expected['router_F32_bytes'],expected['CPU_step_F32_bytes'])==(360492800,660602880,2853888,4896)
        for key,value in expected.items():assert allocation[key]==value
        for key in ('gradient_F32_bytes','first_moment_F32_bytes','second_moment_F32_bytes'):assert allocation[key]==660602880
        assert allocation['mandatory_GPU_array_bytes']==3005758208
        assert allocation['actual_unique_GPU_array_bytes']==sum(e-s for s,e in gpu_ranges)==3005758208+expected['additional_HF_buffer_bytes']
        assert allocation['optimizer']==dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.,foreach=False,fused=False)
        assert raw['resident_after_allocation_and_byte_checks']['GPU_current_allocated']>=allocation['actual_unique_GPU_array_bytes']
        assert all(raw[key]==0 for key in ('new_original_BF16_full_forwards','new_student_full_forwards','new_compact_block_forwards','new_optimizer_updates','new_endpoint_queries','forbidden_forward_attempts'))
        assert not torch.cuda.is_initialized()
        r['procedure_gates'].update(ALL218_installed_core_digests_match_source_payload=True,
            ALL165150720_installed_GUD_elements_match_qualified_priors=True,ALL120_NEW_routing_buffer_digests_match_witnesses=True,
            exact_parameter_set_optimizer_shapes_counts_and_recorded_nonaliasing=True,all_recorded_zero_allocations_and_byte_ledger_consistent=True,
            paired_original_exit_and_no_forward_update_endpoint=True,CPU_only_no_HF_geometry_optimizer_reexecution=True)
        r.update(decision='WHOLE_ASSEMBLY_BYTE_COUNT_INDEPENDENTLY_VERIFIED',checked_core_tensors=218,checked_GUD_elements=checked_elements,
            checked_routing_buffers=120,checked_optimizer_parameter_states=1224,allocation=allocation,
            elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='Installed source/qualified GUD/NEW routing digests and count ledger independently checked; device zeros, pointers, module liveness and allocator peaks remain actual producer observations; RoPE derived values not independently recalculated')
        guard();write_once(args.out,r);assert args.out.stat().st_size<=1<<20;print(json.dumps(dict(decision=r['decision'],allocation=allocation)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat());write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
