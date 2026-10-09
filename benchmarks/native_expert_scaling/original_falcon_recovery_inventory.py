"""Exact stored metadata/byte inventory; timing extrapolations are not measurements."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import sha,write


def main(a):
    start=time.monotonic()
    result=DOC/'original_falcon_transfer_result_20261009.json'
    terminal=result.with_suffix('.terminal.json');t=json.loads(terminal.read_bytes())
    assert t['exit_code']==0 and sha(result)==t['result_sha256']
    r=json.loads(result.read_bytes());assert r['durable_updates']==r['optimizer_updates']==1
    directory=ROOT/'results/native_expert_scaling/original_falcon_transfer_20261009'
    corpus=ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    records=json.loads(corpus.read_bytes())['records'];assert len(records)==48
    native=json.loads((directory/'native.json').read_bytes())
    outputs={Path(i['path']).name:i for i in t['output_files']}
    paths=[Path(__file__),result,terminal,corpus,directory/'native.json',directory/'update1.json']
    paths += [Path(outputs[name]['path']) for name in ('candidate.pt','candidate.packed','native.f32')]
    paths += [Path(rec['logits']['path']) for rec in records]
    receipts=[]
    for p in paths:
        i=dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p));receipts.append(i)
        if p.name in ('candidate.pt','candidate.packed','native.f32'):assert i==outputs[p.name]
    for rec in records:assert sha(rec['logits']['path'])==rec['logits']['sha256']
    splits={}
    for split in ('FIT','DEV'):
        rows=sorted([rec for rec in records if rec['split']==split],key=lambda rec:(rec['domain'],rec['id']))
        assert len(rows)==24 and len({rec['domain'] for rec in rows})==12
        histories=sum(len(rec['student_input_ids']) for rec in rows);labels=sum(len(rec['positions']) for rec in rows)
        splits[split]=dict(cases=24,histories=histories,labels=labels,
            max_history=max(len(rec['student_input_ids']) for rec in rows),teacher_bytes=labels*65537*2,
            full_native_f32_bytes=histories*65537*4,route_bytes=histories*6*64,
            ordered_ids=[rec['id'] for rec in rows])
    token_cost=(r['gradients']['forward_seconds']+r['gradients']['backward_seconds'])/1507
    histories=sum(v['histories'] for v in splits.values())
    adopted=1507
    before_full=(histories-adopted)*65537*4;after_full=histories*65537*4
    before_routes=(histories-adopted)*6*64;after_routes=histories*6*64
    checkpoints=2*outputs['candidate.pt']['bytes'];packed=outputs['candidate.packed']['bytes']
    declared=before_full+after_full+before_routes+after_routes+checkpoints+packed
    value=dict(schema='ORIGINAL_FALCON_RECOVERY_METADATA_INVENTORY_V1',inputs=receipts,splits=splits,
        adopted_before_id='broad_fit_smol_magpie_ultra_022',adopted_before_history=adopted,
        proposed_new_before_native_cases=47,proposed_new_after_native_cases=48,
        proposed_new_updates=24,actual_start_counter=1,proposed_durable_counters=[13,25],
        projected_bytes=dict(before_full=before_full,after_full=after_full,before_routes=before_routes,
            after_routes=after_routes,two_actual_size_checkpoint_estimate=checkpoints,final_packed=packed,
            declared_subtotal=declared,other_queries_witnesses_logs_metrics_not_in_subtotal=True),
        observed_reference=dict(history=1507,forward=r['gradients']['forward_seconds'],
            backward=r['gradients']['backward_seconds'],native_load=native['load_seconds'],native_elapsed=native['elapsed_seconds']),
        extrapolations_not_measured=dict(FIT_forward_backward_linear_seconds=splits['FIT']['histories']*token_cost,
            native_95_history_only_linear_seconds=(2*histories-adopted)/1507*(native['elapsed_seconds']-native['load_seconds']),
            optimizer_inspection_checkpoint_hash_export_launch_costs_excluded=True,
            unions_all_other_cases_unknown=True,union_cardinality_need_not_scale_linearly_with_history=True),
        proposed_caps=dict(seconds=3600,reserve_seconds=300,OS_bytes=32<<30,GPU_allocated_bytes=10<<30,
            GPU_reserved_bytes=11<<30,output_bytes=40<<30),disk_free_at_observation=shutil.disk_usage(ROOT).free,
        new_model_forwards=0,new_teacher_queries=0,new_optimizer_updates=0,GPU_calls=0,
        elapsed_seconds=time.monotonic()-start,implementation_status='Recovery runner/protocol/binding not yet implemented/frozen;inventory is actual stored evidence only.')
    for item in receipts:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    write(a.out,value)
    print(json.dumps(dict(result=str(a.out),sha256=sha(a.out),seconds=value['elapsed_seconds'],
        byte_subtotal=declared,splits=splits,extrapolation=value['extrapolations_not_measured'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
