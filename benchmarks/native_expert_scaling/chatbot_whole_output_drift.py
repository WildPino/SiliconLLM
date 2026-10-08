"""New saved-only diagnostic: final state versus online training losses/routes.

No original/student forward, derivative, optimizer or old audit is repeated.
Online losses use different evolving checkpoints; their differences are not
causal estimates of forgetting or a capacity impossibility certificate.
"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([10])
    r=dict(schema='QWEN_WHOLE_OUTPUT_DRIFT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),procedure_gates={},
        new_original_BF16_full_forwards=0,new_student_full_forwards=0,new_backward_calls=0,new_optimizer_updates=0,new_endpoint_queries=0)
    def guard():
        assert time.monotonic()-start<=30 and proc.memory_info().peak_wset<=256<<20
        assert not proc.children(recursive=True) and not any(v in sys.modules for v in ('torch','numpy','transformers'))
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='whole_output_drift'
        for item in b['inputs']:
            path=Path(item['path']);assert str(path.resolve())==item['resolved_path']
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];guard()
        assert not args.directory.exists();args.directory.mkdir()
        raw=json.loads(Path(b['fit_result_path']).read_bytes());audit=json.loads(Path(b['fit_audit_result_path']).read_bytes())
        assert audit['decision']=='WHOLE_OUTPUT_FIT_INDEPENDENTLY_VERIFIED' and not audit['candidate_transfer_eligible']
        assert raw['decision']=='CLOSE_FIXED_WHOLE_OUTPUT_FIT_REQUIRE_FIRST_AUDIT'
        read_lines=lambda field:[json.loads(line) for line in Path(raw['outputs'][field]['path']).read_bytes().splitlines()]
        initial=read_lines('initial_cases');final=read_lines('final_cases');updates=read_lines('updates')
        assert len(initial)==len(final)==200 and len(updates)==1280
        assert [v['id'] for v in initial]==[v['id'] for v in final]
        fit=[v for v in final if v['split']=='fit'];dev=[v for v in final if v['split']=='development'];assert (len(fit),len(dev))==(160,40)
        last=updates[1120:];assert [v['case_id'] for v in last]==[v['id'] for v in fit]
        categories=sorted({v['category'] for v in final});records=[]
        for category in categories:
            indexes=[i for i,v in enumerate(fit) if v['category']==category];subset=[fit[i] for i in indexes];online=[last[i] for i in indexes]
            assert len(subset)==20
            delta=[v['mean_KL']-u['loss'] for v,u in zip(subset,online)]
            records.append(dict(category=category,manifest_FIT_indexes=indexes,
                last_epoch_online_case_mean_KL=math.fsum(v['loss'] for v in online)/20,
                fixed_final_FIT_case_mean_KL=math.fsum(v['mean_KL'] for v in subset)/20,
                mean_final_minus_online_case_KL=math.fsum(delta)/20,
                positive_final_minus_online_cases=sum(v>0 for v in delta),negative_final_minus_online_cases=sum(v<0 for v in delta),
                minimum_final_minus_online_case_KL=min(delta),maximum_final_minus_online_case_KL=max(delta)))
            guard()
        positions={}
        for split,subset in (('fit',fit),('development',dev)):
            bins=[]
            for step in range(16):
                selected=[v for v in subset if step<v['labels']]
                if selected:
                    bins.append(dict(generated_step=step,labels=len(selected),
                        mean_KL=math.fsum(v['KL_F32_per_label'][step] for v in selected)/len(selected),
                        disagreements=sum(v['student_ids'][step]!=v['source_generated_ids'][step] for v in selected)))
            assert sum(v['labels'] for v in bins)==sum(v['labels'] for v in subset);positions[split]=bins
        routes=[]
        for li in range(24):
            values={}
            for stage,source in (('initial',initial),('final',final)):
                for split in ('fit','development'):
                    subset=[v for v in source if v['split']==split]
                    counts=[sum(v['student_route_occurrences_ALL_input_rows'][li][p] for v in subset) for p in range(16)]
                    assert sum(counts)==sum(len(v['input_ids'])*4 for v in subset)
                    values[stage+'_'+split]=counts
            entry=dict(layer=li,counts=values,total_variation_of_occurrence_distribution={})
            for split in ('fit','development'):
                before=values['initial_'+split];after=values['final_'+split];assert sum(before)==sum(after)>0
                entry['total_variation_of_occurrence_distribution'][split]=sum(abs(a-c) for a,c in zip(before,after))/(2*sum(before))
            routes.append(entry);guard()
        r.update(category_final_vs_online=records,position_diagnostics=positions,route_occurrence_diagnostics=routes,
            last_epoch_online_case_mean_KL=math.fsum(v['loss'] for v in last)/160,
            fixed_final_FIT_case_mean_KL=math.fsum(v['mean_KL'] for v in fit)/160,
            fixed_final_FIT_label_mean_KL=math.fsum(math.fsum(v['KL_F32_per_label']) for v in fit)/2437,
            decision='SAVED_WHOLE_OUTPUT_DRIFT_DIAGNOSED_NOT_CAUSAL',
            limitations=['Online losses use different pre-update parameter states; no fixed-checkpoint learning curve or counterfactual order effect',
                'Category and manifest position are confounded; signed drift does not isolate causation',
                'Occurrence counts include repeated input rows; not distinct states, same-row agreement, selected mass or expert utility',
                'Teacher-forced prefix bins are not own-history generation or fresh quality',
                'Journal F32 metrics were independently envelope-verified previously; original F64 scoring is not repeated'])
        r['procedure_gates'].update(qualified_fit_audit_and_saved_journals_only=True,complete160_last_epoch_cases_and200_final_cases=True,
            complete16_position_bins_and24_route_count_conservation=True,no_new_source_student_optimizer_endpoint_or_old_audit=True)
        guard();r.update(compute_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat());write_once(args.out,r)
        print(json.dumps(dict(decision=r['decision'],category_drift=records,online=r['last_epoch_online_case_mean_KL'],final=r['fixed_final_FIT_case_mean_KL'])),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat());write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
