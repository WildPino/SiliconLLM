"""FIRST independent NEW text/ID/wire/argmax adoption, no source rerun.

Reads complete saved conversations, optionally a retained complete-case prefix
of a failed collector. Pending incomplete conversation is explicitly excluded.
"""
import argparse
import datetime as dt
import hashlib
import json
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
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    stage='binding'
    r=dict(schema='QWEN_GLOBAL_COHORT_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},conversations=[],new_original_BF16_full_forwards=0,new_source_or_student_or_optimizer_or_endpoint_queries=0)
    def guard():assert time.monotonic()-start<=300 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='global_cohort_audit'
        for item in b['inputs']:
            path=Path(item['path']);assert str(path.resolve())==item['resolved_path'] and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];guard()
        import numpy as np
        from chatbot_interaction import IndependentBPE
        assert np.__version__=='2.4.6' and 'torch' not in sys.modules and 'transformers' not in sys.modules
        args.directory.mkdir();raw=json.loads(Path(b['collector_result_path']).read_bytes())
        spec=json.loads(Path(b['manifest_path']).read_bytes());endpoint=json.loads(Path(b['endpoint_path']).read_bytes())
        original=json.loads(Path(b['old_adoption_path']).read_bytes());encoded=json.loads(Path(raw['tokenized_inputs']['path']).read_bytes())
        def key(c):return json.dumps([c['mode'],c['messages']],sort_keys=True,ensure_ascii=False,separators=(',',':'))
        tkeys={key(c) for c in spec['cases']};ekeys={key(c) for c in endpoint['cases']};okeys={key(c) for c in original['cases']}
        assert len(tkeys)==200 and len(ekeys)==64 and not (tkeys&ekeys or tkeys&okeys or ekeys&okeys)
        assert len(encoded['cases'])==200 and [c['id'] for c in encoded['cases']]==[c['id'] for c in spec['cases']]
        bpe=IndependentBPE(json.loads(Path(b['tokenizer_json_path']).read_bytes()))
        for case,tokens in zip(spec['cases'],encoded['cases']):
            stage='NEW_independent_render_and_IDs'
            # Source canonical plain-text grammar, previously qualified adapter;
            # independently reconstruct each NEW rendering without HF/Jinja.
            messages=case['messages'];parts=[]
            if messages[0]['role']!='system':parts.append('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n')
            for i,message in enumerate(messages):
                ending='' if case['mode']=='continue' and i==len(messages)-1 else '<|im_end|>\n'
                parts.append('<|im_start|>'+message['role']+'\n'+message['content']+ending)
            if case['mode']=='generate':parts.append('<|im_start|>assistant\n')
            rendered=''.join(parts);assert rendered==tokens['rendered_text'],case['id']
            assert hashlib.sha256(rendered.encode('utf8')).hexdigest()==tokens['rendered_UTF8_SHA256']
            ids=bpe.encode(rendered);assert ids==tokens['prompt_ids'] and 0<len(ids)<=128
            assert hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()==tokens['prompt_U32LE_SHA256'];guard()
        # Independent source tokenizer grammar and BPE are used only on NEW
        # transfer cases, never on the reserved endpoint or old fixture inputs.
        assert len(raw['conversations'])>0
        assert [c['id'] for c in raw['conversations']]==[c['id'] for c in spec['cases'][:len(raw['conversations'])]]
        complete=len(raw['conversations'])==200 and 'fault' not in raw
        if complete:assert raw['decision']=='NEW_WHOLE_TRANSFER_COHORT_REQUIRE_FIRST_BYTE_ID_AUDIT' and all(raw['procedure_gates'].values())
        full_x_rows=0;full_label_rows=0
        for summary,case,tokens in zip(raw['conversations'],spec['cases'],encoded['cases']):
            stage='complete_saved_NEW_case_wires';current=json.loads(Path(summary['metadata_path']).read_bytes())
            assert current['id']==case['id'] and current['messages']==case['messages'] and current['mode']==case['mode']
            assert current['split']==case['split']==summary['split'] and current['category']==case['category']==summary['category']
            assert current['prompt_ids']==tokens['prompt_ids'] and current['rendered_text']==tokens['rendered_text']
            for name in ('x','logits','journal'):
                assert current[name+'_path']==summary[name+'_path'] and current[name+'_sha256']==summary[name+'_sha256']
                assert sha(current[name+'_path'])==current[name+'_sha256']
            with Path(current['x_path']).open('rb') as stream:assert struct.unpack('<8s4I',stream.read(24))==(b'QWGX0001',24,896,2,0)
            with Path(current['logits_path']).open('rb') as stream:assert struct.unpack('<8s4I',stream.read(24))==(b'QWGL0001',151936,2,0,0)
            frames=[json.loads(line) for line in Path(current['journal_path']).read_bytes().splitlines()]
            assert frames==current['frames'] and len(frames)==len(current['generated_ids'])==summary['source_forwards']
            assert 0<len(frames)<=16
            xo=lo=24;next_ids=[];xrows=0
            with Path(current['x_path']).open('rb') as xs,Path(current['logits_path']).open('rb') as ls:
                xs.seek(24);ls.seek(24)
                for step,frame in enumerate(frames):
                    expected_input=tokens['prompt_ids'] if step==0 else [next_ids[-1]]
                    assert frame['step']==step and frame['input_ids']==expected_input and frame['decision_history_position']==len(tokens['prompt_ids'])+step-1
                    assert frame['x_offset']==xo and frame['logits_offset']==lo
                    assert frame['x_rows']==len(expected_input) and frame['x_bytes']==len(expected_input)*43008 and frame['logits_bytes']==303872
                    xb=xs.read(frame['x_bytes']);lb=ls.read(303872)
                    assert hashlib.sha256(xb).hexdigest()==frame['x_sha256'] and hashlib.sha256(lb).hexdigest()==frame['logits_sha256']
                    xx=np.frombuffer(xb,dtype='<u2');ll=np.frombuffer(lb,dtype='<u2')
                    assert len(xx)==len(expected_input)*24*896 and len(ll)==151936
                    assert not ((xx&0x7fff)>=0x7f80).any() and not ((ll&0x7fff)>=0x7f80).any()
                    values=(ll.astype('<u4')<<16).view('<f4');winner=int(np.argmax(values))
                    assert winner==frame['next_id']==current['generated_ids'][step]
                    assert all(i not in (151645,151643) for i in next_ids), 'Input after generated EOS'
                    next_ids.append(winner);xo+=frame['x_bytes'];lo+=303872;xrows+=len(expected_input)
                assert xs.read(1)==ls.read(1)==b''
            assert xo==current['x_bytes']==summary['x_bytes']==Path(current['x_path']).stat().st_size
            assert lo==current['logits_bytes']==summary['logits_bytes']==Path(current['logits_path']).stat().st_size
            assert xrows==current['x_rows']==summary['x_rows']==len(tokens['prompt_ids'])+len(next_ids)-1
            eos=next_ids[-1] in (151645,151643);expected_stop='eos' if eos else 'length'
            assert eos or len(next_ids)==16
            assert current['termination']==summary['termination']==expected_stop and current['accepted_generated_ids']==next_ids
            full_x_rows+=xrows;full_label_rows+=len(next_ids)
            r['conversations'].append(dict(id=case['id'],split=case['split'],x_rows=xrows,teacher_logit_vectors=len(next_ids),
                all_payload_bytes_finite_and_journal_exact=True,complete_vocabulary_argmax_ALL_exact=True,
                complete_own_history_and_generated_only_BOTH_EOS=True))
            if len(r['conversations'])%16==0:print(json.dumps(dict(audited_complete_cases=len(r['conversations']))),flush=True)
            guard()
        assert full_label_rows<=raw['new_original_BF16_full_forwards']<=3200
        payload=full_x_rows*43008+full_label_rows*303872
        if complete:assert full_label_rows==raw['new_original_BF16_full_forwards'] and payload==raw['tensor_payload_bytes']
        r['procedure_gates'].update(NEW_transfer_old_and_endpoint_exact_disjointness=True,
            ALL200_NEW_renderings_independent_source_grammar_and_BPE_IDs=True,
            complete_case_prefix_incremental_wires_and_journals_exact=True,ALL24_x_full_vocabulary_teacher_labels_finite=True,
            ALL_saved_source_winners_independently_exact=True,own_history_masks_positions_BOTH_EOS_exact=True,
            zero_model_old_fixture_or_endpoint_queries=True)
        r.update(decision='NEW_WHOLE_TRANSFER_COHORT_BYTE_ID_QUALIFIED' if complete else 'NEW_COHORT_COMPLETE_CASE_PREFIX_ADOPTED_FULL_COHORT_INCOMPLETE',
            complete_200_case_cohort=complete,complete_case_coverage=len(r['conversations']),adopted_x_rows_PER_LAYER=full_x_rows,
            adopted_teacher_logit_vectors=full_label_rows,adopted_tensor_payload_bytes=payload,
            pending_incomplete_conversation_excluded='pending_conversation' in raw,
            producer_full_resource_admission=complete and b['collector_actual_worker_exit_code']==0,
            elapsed_before_final_serialization=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='FIRST retained NEW wire/ID/argmax adoption only; no independent source reexecution or compact transfer/endpoint quality/rate claim')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],cases=len(r['conversations']),x_rows=full_x_rows,label_rows=full_label_rows)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,elapsed=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
