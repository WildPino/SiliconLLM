"""NEW whole-conversion parameter/optimizer/logit-liveness ledger; no forwards."""
import argparse
import datetime as dt
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def inventory(spec,census,old):
    d=spec['hidden_size'];l=spec['layers'];p=spec['parents'];q=spec['query_width'];h0=spec['shared_width'];h1=spec['function_width']
    assert (d,l,p,q,h0,h1)==(896,24,16,32,512,128)
    src=census['source'];assert src['named_elements']==494032768 and src['coverage_tensor_names']==290
    assert src['complete_name_shape_dtype_coverage'] and src['dimension_and_offset_conservation'] and src['payload_bytes']==988065536
    descriptors=src['descriptors'];core={name:v for name,v in descriptors.items() if v['category']!='dense_ffn'}
    assert len(core)==218 and sum(v['stored_named_elements'] for v in core.values())==180246400
    assert old['source_payload_bytes']==src['payload_bytes'] and old['source_matrix_MAC']==src['matrix_MACs_total_per_decode_token']
    rows=[]
    for children in (1,10):
        prior=next(v for v in old['arms'] if v['children']==children)
        shared=l*3*d*h0;private=l*p*children*3*d*h1;trainable=shared+private
        # Existing executable constructor registers child centers/norms even
        # for C=1. Include their live conversion bytes; native C=1 omits them.
        router_module=l*(d*q+p*q+p*children*q+p+p*children)
        deployed_router_matrices=l*(d*q+p*q+(p*children*q if children>1 else 0))
        deployed_router_norms=l*(p+(p*children if children>1 else 0))
        source_bytes=src['payload_bytes'];frozen_bytes=sum(v['stored_payload_bytes'] for v in core.values())
        weight_bytes=trainable*4;grad_bytes=trainable*4;adam_moments=trainable*8
        # Full resident optimizer lower bound after all parameters were touched,
        # independent teacher and student core. No CPU offload assumption.
        persistent=source_bytes+frozen_bytes+weight_bytes+grad_bytes+adam_moments+router_module*4
        logical=prior['logical_coefficient_bytes_per_token']+2*prior['router_matrix_MAC']
        payload=prior['proposed_mixed_payload_bytes']+2*deployed_router_matrices
        logits=128*spec['vocabulary_size']*4
        recurrent=128*d*4*(l+1)
        rows.append(dict(children=children,trainable_elements=trainable,shared_elements=shared,private_elements=private,
            frozen_student_core_elements=sum(v['stored_named_elements'] for v in core.values()),frozen_student_BF16_bytes=frozen_bytes,
            teacher_BF16_bytes=source_bytes,student_F32_weight_bytes=weight_bytes,max_resident_gradient_bytes=grad_bytes,
            fully_touched_Adam_F32_moment_bytes=adam_moments,actual_constructor_router_buffer_F32_bytes=router_module*4,
            fully_touched_resident_training_persistent_lower_bound=persistent,
            GPU_lower_bound_exceeds_12GiB=persistent>12<<30,
            one_full_F32_logit_tensor_at_128_tokens_bytes=logits,
            minimum_named_teacher_and_student_logits_bytes=2*logits,
            checkpoint_input_F32_shape_account_at_128_tokens_bytes=recurrent,
            liveness_scope='Persistent plus named outputs only; log_softmax/exp/backward/attention/SiLU/allocator/workspaces/temp copies not included. Not a peak bound.',
            deployed_router_F32_elements=deployed_router_matrices+deployed_router_norms,
            proposed_native_payload_bytes=payload,complete_matrix_MAC_per_token=prior['matrix_MAC_per_token'],
            proposed_native_logical_coefficient_bytes_per_token=logical,
            complete_matrix_ratio_exact=str(Fraction(prior['matrix_MAC_per_token'],old['source_matrix_MAC'])),
            complete_logical_ratio_exact=str(Fraction(logical,old['source_logical_coefficient_bytes'])),
            native_dimension_gate=5*prior['matrix_MAC_per_token']<=3*old['source_matrix_MAC'] and 5*logical<=3*old['source_logical_coefficient_bytes'],
            resident_Adam_decision='REJECT_FULLY_RESIDENT_12GiB_ADAM' if persistent>12<<30 else 'NOT_EXCLUDED_BY_PERSISTENT_LOWER_BOUND_PEAK_UNMEASURED',
            conversion_exposure='C=10 original local exposure recipe remains CLOSED; this ledger does not reopen it' if children==10 else 'ALL24 initialization/exposure not qualified'))
    return dict(core_source_parameter_names=list(core),core_tensors=218,core_elements=180246400,arms=rows,
        selected_initial_whole_arm_children=1,source_inference_contract='Qwen2.5-0.5B-Instruct, GQA/BF16 tied head, exact original tokenizer/chat contract',
        global_interface='ALL24 nonlinear MLP installation and masked output KL code implemented; actual complete installation/initialization/fit not executed',
        native_integration='New compact format/catalog/operator/canonical adapter missing; old fixed m284 archive uses source-full FFN',
        cache_and_context='Original native F32 KV capacity4096=100663296B; context attention43008*context MAC, reused old accounting',
        measured_physical_DRAM_or_rate=False,qualified_whole_transfer=False)


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_WHOLE_TRANSFER_INVENTORY_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_source_values_features_J_H_or_fit_calls=0)
    def guard():assert time.monotonic()-start<=45 and proc.memory_info().peak_wset<=512<<20 and not proc.children(recursive=True)
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='whole_inventory'
        for v in b['inputs']:
            p=Path(v['path']);assert str(p.resolve())==v['resolved_path'] and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'];guard()
        args.directory.mkdir();spec=json.loads(Path(b['spec_path']).read_bytes());census=json.loads(Path(b['census_path']).read_bytes());old=json.loads(Path(b['old_budget_path']).read_bytes())
        r['inventory']=inventory(spec,census,old)
        assert all(v['native_dimension_gate'] for v in r['inventory']['arms'])
        assert not r['inventory']['arms'][0]['GPU_lower_bound_exceeds_12GiB'] and r['inventory']['arms'][1]['GPU_lower_bound_exceeds_12GiB']
        r['procedure_gates'].update(bound_existing_metadata_and_actual_new_code=True,no_old_geometry_census_or_source_numeric_replay=True,
            actual_constructor_unused_child_buffers_charged=True,F32_router_native_delta_explicit=True,
            resident_optimizer_lower_bound_not_peak=True,teacher_head_attention_cache_and_output_loss_liveness_explicit=True)
        r.update(decision='E16_COMPLETE_IMPLEMENTATION_PREREQUISITE_E160_RESIDENT_ADAM_REJECTED',
            elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope='New conversion-memory/implementation ledger, not original geometry rerun or training/native admission')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],arms=r['inventory']['arms'])),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
