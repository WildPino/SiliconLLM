"""FIRST saved whole-output/loss/gradient/sample-Adam/count audit, CPU only.

No source/student/geometry/optimizer numerical reexecution. ALL saved logits
are independently scored in F64; full network differentiation is not certified.
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
    r=dict(schema='QWEN_WHOLE_OUTPUT_AUDIT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},new_original_BF16_full_forwards=0,new_student_full_forwards=0,new_optimizer_updates=0,new_endpoint_queries=0)
    def guard():
        assert time.monotonic()-start<=600 and proc.memory_info().peak_wset<=4<<30
        assert not proc.children(recursive=True)
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='whole_output_audit'
        for entry in b['inputs']:
            path=Path(entry['path']);assert str(path.resolve())==entry['resolved_path']
            assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'];guard()
        def no_subprocess(event,arguments):
            if event=='subprocess.Popen':raise RuntimeError('Whole output audit forbids subprocess')
        sys.addaudithook(no_subprocess)
        import numpy as np
        import torch
        assert np.__version__=='2.4.6' and torch.__version__=='2.6.0+cu124' and not torch.cuda.is_initialized()
        args.directory.mkdir();raw=json.loads(Path(b['fit_result_path']).read_bytes())
        cohort=json.loads(Path(b['cohort_path']).read_bytes());by_id={c['id']:c for c in cohort['conversations']}
        def bf16(bits):
            return (np.ascontiguousarray(bits,dtype='<u2').astype('<u4')<<16).view('<f4').astype('<f8')
        def logprob(values):
            shifted=values-values.max(axis=-1,keepdims=True)
            return shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))
        def kl_values(student,teacher):
            q=logprob(student);p=logprob(teacher)
            return (np.exp(p)*(p-q)).sum(axis=-1),q,p
        summaries={};maximum_KL_gap=0.;maximum_ID_logprob_gap=0.
        for label in ('initial','final'):
            stage=label+'_ALL_saved_logits';records=[json.loads(line) for line in Path(raw['outputs'][label+'_cases']['path']).read_bytes().splitlines()]
            assert [v['id'] for v in records]==[c['id'] for c in cohort['conversations']]
            independent=[];cursor=24
            with Path(raw['outputs'][label+'_logits']['path']).open('rb') as stream:
                assert struct.unpack('<8s4I',stream.read(24))==(b'QWSL0001',151936,2,0,0)
                for record in records:
                    source=by_id[record['id']];meta=json.loads(Path(source['metadata_path']).read_bytes())
                    assert record['category']==source['category'] and record['split']==source['split']
                    prompt=meta['prompt_ids'];generated=meta['generated_ids'];k=len(generated)
                    assert record['input_ids']==prompt+generated[:-1] and record['decision_positions']==list(range(len(prompt)-1,len(prompt)+k-1))
                    assert record['source_generated_ids']==generated and record['labels']==k
                    assert record['student_logit_offset']==cursor==stream.tell() and record['student_logit_bytes']==k*303872
                    payload=stream.read(k*303872);assert hashlib.sha256(payload).hexdigest()==record['student_logit_sha256']
                    saved=np.frombuffer(payload,dtype='<u2').reshape(k,151936);assert not ((saved&0x7fff)>=0x7f80).any()
                    teacher_payload=Path(source['logits_path']).read_bytes();assert struct.unpack('<8s4I',teacher_payload[:24])==(b'QWGL0001',151936,2,0,0)
                    teacher_bits=np.frombuffer(teacher_payload,dtype='<u2',offset=24).reshape(k,151936)
                    student=bf16(saved);teacher=bf16(teacher_bits);KL,q,p=kl_values(student,teacher)
                    student_ids=np.argmax(student,axis=1).tolist();assert student_ids==record['student_ids']
                    errors=sum(a!=z for a,z in zip(student_ids,generated));assert errors==record['disagreements']
                    gap=float(np.max(np.abs(KL-np.array(record['KL_F32_per_label']))));maximum_KL_gap=max(maximum_KL_gap,gap)
                    assert np.all(np.abs(KL-np.array(record['KL_F32_per_label']))<=2e-5*np.maximum(1.,np.abs(KL)))
                    lprob=q[np.arange(k),generated];lp_gap=float(np.max(np.abs(lprob-record['source_ID_log_probability'])))
                    maximum_ID_logprob_gap=max(maximum_ID_logprob_gap,lp_gap);assert lp_gap<=2e-5*max(1.,float(np.max(np.abs(lprob))))
                    assert abs(record['mean_KL']-float(KL.mean()))<=2e-5*max(1.,abs(float(KL.mean())))
                    counts=record['student_route_occurrences_ALL_input_rows'];assert len(counts)==24
                    assert all(len(v)==16 and all(type(i) is int and i>=0 for i in v) and sum(v)==len(record['input_ids'])*4 for v in counts)
                    independent.append(dict(id=record['id'],split=record['split'],category=record['category'],labels=k,KL=KL.tolist(),errors=errors))
                    cursor+=len(payload);del student,teacher,q,p,teacher_payload,teacher_bits,payload,saved;guard()
                assert not stream.read(1) and cursor==Path(raw['outputs'][label+'_logits']['path']).stat().st_size
            summary={}
            for split in ('fit','development'):
                cases=[c for c in independent if c['split']==split];n=sum(c['labels'] for c in cases)
                assert (len(cases),n)==((160,2437) if split=='fit' else (40,613))
                categories={}
                for category in sorted({c['category'] for c in cases}):
                    group=[c for c in cases if c['category']==category];m=sum(c['labels'] for c in group)
                    categories[category]=dict(labels=m,mean_KL=math.fsum(x for c in group for x in c['KL'])/m,disagreement_fraction=sum(c['errors'] for c in group)/m)
                summary[split]=dict(cases=len(cases),labels=n,mean_KL=math.fsum(x for c in cases for x in c['KL'])/n,disagreement_fraction=sum(c['errors'] for c in cases)/n,categories=categories)
                measured=raw[label+'_metrics'][split]
                assert measured['cases']==len(cases) and measured['labels']==n and measured['disagreement_fraction']==summary[split]['disagreement_fraction']
                assert abs(measured['mean_KL']-summary[split]['mean_KL'])<=2e-5*max(1.,abs(summary[split]['mean_KL']))
            summaries[label]=summary
        stage='FIRST_loss_logit_gradient_and_sample_Adam'
        first_meta=json.loads(Path(raw['outputs']['first_update_metadata']['path']).read_bytes())
        first=torch.load(raw['outputs']['first_update']['path'],map_location='cpu',weights_only=True)
        first_source=by_id[first_meta['case_id']];first_source_meta=json.loads(Path(first_source['metadata_path']).read_bytes())
        assert first_meta['input_ids']==first_source_meta['prompt_ids']+first_source_meta['generated_ids'][:-1]
        assert first_meta['decision_positions']==list(range(len(first_source_meta['prompt_ids'])-1,len(first_meta['input_ids'])))
        first_teacher_bytes=memoryview(first['teacher_logits'].reshape(-1).view(torch.uint8).numpy())
        assert hashlib.sha256(first_teacher_bytes).hexdigest()==hashlib.sha256(Path(first_source['logits_path']).read_bytes()[24:]).hexdigest()
        teacher=bf16(first['teacher_logits'].view(torch.uint16).numpy())[0];student=bf16(first['student_logits'].view(torch.uint16).numpy())[0]
        KL,q,p=kl_values(student,teacher);k=len(KL);assert np.array_equal(first['mask'].numpy(),np.ones((1,k)))
        first_loss=float(KL.mean());assert abs(first_meta['loss']-first_loss)<=2e-5*max(1.,abs(first_loss))
        gradient=(np.exp(q)-np.exp(p))/k;observed=bf16(first['student_logit_gradient'].view(torch.uint16).numpy())[0]
        gap=np.abs(gradient-observed);envelope=2e-6/k+(2.**-8)*np.abs(gradient)
        assert np.all(gap<=envelope),'first BF16 logit-gradient envelope'
        norms=first['gradient_parameter_norms_before_clip'].numpy().astype('float64');assert norms.shape==(1224,) and np.isfinite(norms).all()
        norm=math.sqrt(math.fsum(float(v)*float(v) for v in norms));assert abs(first_meta['norm_before_clip']-norm)<=1e-5*max(1.,norm)
        assert len(first_meta['parameter_names'])==len(set(first_meta['parameter_names']))==1224
        assert sum(first_meta['parameter_elements'])==165150720
        for li in range(24):
            n=math.sqrt(math.fsum(float(norms[i])**2 for i,name in enumerate(first_meta['parameter_names']) if name.startswith(f'model.layers.{li}.')))
            assert n>0 and abs(n-first_meta['layer_gradient_norms'][str(li)])<=1e-5*max(1.,n)
        values={key:first[key].numpy().astype('float64') for key in ('before','gradient_before_clip','scaled_gradient','after','exp_avg','exp_avg_sq')}
        assert all(v.shape==(1224,3) and np.isfinite(v).all() for v in values.values())
        tiny=2.**-126;epsilon=2.**-23
        def near(actual,expected,factor):
            assert np.all(np.abs(actual-expected)<=factor*epsilon*np.abs(expected)+tiny)
        scale=min(1.,1./(first_meta['norm_before_clip']+1e-6));g=values['scaled_gradient']
        near(g,values['gradient_before_clip']*scale,32)
        near(values['exp_avg'],.1*g,32);near(values['exp_avg_sq'],.001*g*g,64)
        delta=1e-4*(values['exp_avg']/.1)/(np.sqrt(values['exp_avg_sq']/.001)+1e-8)
        expected=values['before']-delta
        ulp=np.abs(np.spacing(values['before'].astype('<f4')).astype('<f8'))
        assert np.all(np.abs(values['after']-expected)<=4*ulp+64*epsilon*np.abs(delta)+tiny)
        stage='fixed_update_counts_final_checkpoints_and_decision'
        updates=[json.loads(line) for line in Path(raw['outputs']['updates']['path']).read_bytes().splitlines()]
        fit=[c for c in cohort['conversations'] if c['split']=='fit'];assert len(updates)==1280
        for index,item in enumerate(updates):
            assert (item['epoch'],item['case_index'],item['case_id'],item['update'])==(index//160,index%160,fit[index%160]['id'],index+1)
            assert item['labels']==fit[index%160]['source_forwards'] and math.isfinite(item['loss']) and math.isfinite(item['gradient_norm_before_clip'])
        assert raw['new_student_full_forwards']==1680 and raw['new_backward_calls']==raw['new_optimizer_updates']==1280 and raw['epochs_completed']==8
        initializer=json.loads(Path(b['initializer_path']).read_bytes());final_elements=0
        assert [v['layer'] for v in raw['final_checkpoints']]==list(range(24))
        for final,initial in zip(raw['final_checkpoints'],initializer['layers']):
            value=torch.load(final['path'],map_location='cpu',weights_only=True);routing=torch.load(initial['witness']['path'],map_location='cpu',weights_only=True)
            prior=torch.load(initial['initial_checkpoint']['path'],map_location='cpu',weights_only=True)
            li=initial['layer'];prefix=f'model.layers.{li}.mlp.compact.'
            for index,name in enumerate(first_meta['parameter_names']):
                if not name.startswith(prefix):continue
                suffix=name[len(prefix):];parts=suffix.split('.')
                tensor=prior[parts[0]] if len(parts)==1 else prior[parts[0]][int(parts[1])].contiguous()
                assert tensor.numel()==first_meta['parameter_elements'][index]
                indexes=[0,tensor.numel()//2,tensor.numel()-1]
                assert np.array_equal(values['before'][index],tensor.reshape(-1)[indexes].numpy().astype('float64'))
            assert set(value)==set(prior)
            for name,tensor in value.items():
                assert tensor.device.type=='cpu' and tensor.dtype==torch.float32 and tensor.is_contiguous() and torch.isfinite(tensor).all()
                assert tensor.shape==prior[name].shape
                if name in ('projection','parent_centers','child_centers','parent_norms','child_norms'):
                    assert torch.equal(tensor.view(torch.uint8),routing[name].view(torch.uint8))
                else:final_elements+=tensor.numel()
            del value,routing,prior;guard()
        assert final_elements==165150720
        decisions={}
        for split in ('fit','development'):
            current=summaries['final'][split];baseline=summaries['initial'][split]
            decisions[split+'_absolute_KL_le_0p10']=current['mean_KL']<=.1
            decisions[split+'_disagreement_le_0p15']=current['disagreement_fraction']<=.15
            decisions[split+'_relative_KL_le_0p20']=current['mean_KL']<=.2*baseline['mean_KL']
        decisions['DEV_each_category_KL_le_0p20']=all(v['mean_KL']<=.2 for v in summaries['final']['development']['categories'].values())
        decisions['DEV_each_category_disagreement_le_0p25']=all(v['disagreement_fraction']<=.25 for v in summaries['final']['development']['categories'].values())
        assert decisions==raw['transfer_gates'],'gate disagreement: retain audit fault, no promotion'
        assert not torch.cuda.is_initialized()
        r['procedure_gates'].update(ALL400_saved_whole_output_cases_masks_offsets_and_F64_metrics=True,
            FIRST_output_loss_BF16_logit_gradient_envelope_and_ALL24_recorded_gradient_norms=True,
            FIRST_3672_coordinate_Adam_samples_and_clip_envelopes=True,fixed1280_update_order_and_1680_forward_counts=True,
            ALL24_final_shapes_finiteness_and_NEW_routing_buffers_BYTE_unchanged=True,
            frozen_candidate_decision_matches_independent_F64_metrics=True,CPU_only_no_source_student_geometry_optimizer_reexecution=True)
        r.update(decision='WHOLE_OUTPUT_FIT_INDEPENDENTLY_VERIFIED',candidate_transfer_eligible=all(decisions.values()),transfer_gates=decisions,
            independent_metrics=summaries,maximum_KL_gap=maximum_KL_gap,maximum_ID_logprob_gap=maximum_ID_logprob_gap,
            FIRST_loss_F64=first_loss,FIRST_maximum_logit_gradient_gap=float(gap.max()),FIRST_gradient_norm_F64=norm,
            final_trainable_elements_checked=final_elements,elapsed_before_final_serialization=time.monotonic()-start,
            OS_peak_snapshot=proc.memory_info().peak_wset,ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='Saved complete outputs/selected gradient and optimizer coordinates/byte and count audit; full network differentiation, all-coordinate Adam, actual student routing arithmetic and fresh behavior NOT independently reexecuted')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],candidate_transfer_eligible=r['candidate_transfer_eligible'],metrics=summaries)),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat());write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
