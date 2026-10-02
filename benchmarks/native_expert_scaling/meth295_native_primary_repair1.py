#!/usr/bin/env python3
"""Process-isolated native continuation of exact saved M295 GPU controls."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import meth295_native_primary_prediction as M

PARTIAL=M.DOC/'meth295_native_primary_prediction_result.partial.json'
PARTIAL_SHA='9e6de46738323b62c5bf60adfddbba047ccdbc55e73f4f44596aa9bf9cdb05fc'
FAILURE=M.DOC/'meth295_native_primary_prediction_result.failure.json'
FAILURE_SHA='958b9ba6e874981428eda527f0771088ab6112de98920b7c845e4832732aeba1'
ORIGINAL_SCRIPT_SHA='23a0b88aa2074c6802a704774ffb5a15fece374de06b0fd1e0a7be1af14a1ec8'
BUNDLE_SHA='9e6b7806a51f1fd763619adbce30ebd184c80142aa72ffceaf9db435202b5f83'

def controls(items):
    for p,s in ((PARTIAL,PARTIAL_SHA),(FAILURE,FAILURE_SHA),(Path(M.__file__),ORIGINAL_SCRIPT_SHA)):
        assert M.P.digest(p)==s,str(p)
    failed=json.loads(FAILURE.read_text(encoding='utf-8'));assert failed['cpu_runtime'] is None
    assert failed['stage']==M.GPU_STORED and failed['error']=='AssertionError: '
    saved=json.loads(PARTIAL.read_text(encoding='utf-8'));assert saved['stage']==M.GPU_STORED
    arms=saved['arms'];assert set(arms)=={*M.ARMS[:2],M.GPU_STORED}
    donor_top={row['source_id']:np.asarray(row['token_top1']) for row in arms[M.ARMS[0]]['prompt_rows']}
    for arm,cell in arms.items():
        assert len(cell['document_rows'])==len(cell['prompt_rows'])==24
        for item,doc,prompt in zip(items,cell['document_rows'],cell['prompt_rows']):
            assert item['source_id']==doc['source_id']==prompt['source_id']
            assert item['category']==doc['category']==prompt['category'] and item['bytes']==doc['bytes']
            n=len(item['document_ids']);assert len(doc['token_nll'])==len(doc['token_top1'])==n
            assert doc['canonical_m17_nats_exact'] is True
            assert np.isfinite(doc['token_nll']).all() and min(doc['token_nll'])>=0 and np.isfinite(doc['nats'])
            assert len(prompt['token_top1'])==prompt['positions']==len(item['prompt_ids'])
            assert prompt['matching']==int((np.asarray(prompt['token_top1'])==donor_top[item['source_id']]).sum())
            for row in (doc,prompt):assert all(0<=t<M.V for t in row['token_top1'])
    return arms,donor_top,saved

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    bundle=M.ART/'meth295_native_primary_repair1_ids.bin';native=M.ART/'meth295_native_primary_repair1.bin'
    log=M.ART/'meth295_native_primary_repair1_stdout.log';err=M.ART/'meth295_native_primary_repair1_stderr.log'
    exe=M.ART/'meth295_native_primary_repair1_cpu.exe'
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.partial.json'),args.out.with_suffix('.failure.json'),bundle,native,log,err))
    start=time.monotonic();stage='bindings';cpu=None
    try:
        items,parent=M.bind();arms,donor_top,saved=controls(items)
        cases=M.cases_for(items);M.write_bundle(bundle,cases)
        assert M.P.digest(M.ART/'meth295_native_primary_ids.bin')==BUNDLE_SHA==M.P.digest(bundle)
        # Importing Torch does not create CUDA tensors; never initialize a CUDA model/context here.
        assert M.torch.cuda.is_initialized() is False
        command=['clang','-O3','-mavx2','-mssse3','-mfma','-fopenmp','-std=c11','-DSILICON_COMPLETE_I16_NATIVE_PRIMARY',
            str(M.P.ROOT/'benchmarks/phase60/engine.c'),'-o',str(exe),'-lm','-lpsapi','-lbcrypt']
        subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        stage='native_CPU_full_head'
        M.dump(args.out.with_suffix('.partial.json'),{'stage':stage,'GPU_partial_sha256':PARTIAL_SHA,
            'cases':len(cases),'bundle_sha256':BUNDLE_SHA,'seconds':time.monotonic()-start})
        print(json.dumps({'stage':stage,'cases':len(cases),'scored_rows':sum(len(c['targets']) for c in cases),
            'prefill_rows':sum(c['first'] for c in cases),'log':str(log),'hard_seconds':M.CPU_SECONDS,
            'CUDA_initialized':False,'reused_complete_GPU_controls_sha256':PARTIAL_SHA}),flush=True)
        cpu=M.run_cpu(exe,bundle,native,log,err)
        assert len(cpu['reports'])==len(cases)+1
        assert [r['case_complete'] for r in cpu['reports'][:-1]]==list(range(1,len(cases)+1))
        report=cpu['reports'][-1]
        assert report['arms']==1 and report['cases']==len(cases) and report['timing_qualification'] is False
        assert report['scored_rows']==sum(len(c['targets']) for c in cases) and report['prefill_rows']==sum(c['first'] for c in cases)
        stage='parse_native';arms[M.ARMS[2]],oracles=M.parse_native(native,cases,items,donor_top)
        M.F.ARMS=M.ARMS;summary=M.F.summarize({k:arms[k] for k in M.ARMS})
        M.F.ARMS=(*M.ARMS[:2],M.GPU_STORED);descriptive=M.F.summarize({k:arms[k] for k in M.F.ARMS});M.F.ARMS=M.ARMS
        gates={'fixed_manifest_source_annotations_and_original_artifact':True,'same_native_operator_body_and_all725_no_fallback':True,
            'all_windows_absolute_positions_targets_and_scored_tokens_verified':True,'all_case_first_row_full_head_score_oracles':True,
            'all_GPU_document_scalars_exact_independent_canonical_M17':True,'CUDA_model_process_terminated_before_native_CPU':True,
            'recreated_input_bundle_byte_identical_original':True,
            'pooled_bpb_vs_both':all(summary['pooled'][k]<=.01 for k in ('candidate_minus_donor_bpb','candidate_minus_bf16_e1280_bpb')),
            'category_bpb_vs_both':all(summary[c][k]<=.02 for c in M.P.CATEGORIES for k in ('candidate_minus_donor_bpb','candidate_minus_bf16_e1280_bpb')),
            'pooled_top1':summary['pooled']['candidate_minus_bf16_e1280_top1']>=-.01,
            'category_top1':all(summary[c]['candidate_minus_bf16_e1280_top1']>=-.02 for c in M.P.CATEGORIES)}
        paths=[Path(__file__),Path(M.__file__),M.P.ROOT/'benchmarks/native_expert_scaling/meth295_native_primary_cpu.c',
            M.P.ROOT/'benchmarks/phase60/engine.c',M.P.ROOT/'benchmarks/native_expert_scaling/meth285_model_operator.h',
            M.P.ROOT/'benchmarks/native_expert_scaling/meth284_source_operator.h',M.P.ROOT/'benchmarks/native_expert_scaling/meth284_archive_catalog.h',
            M.P.ROOT/'benchmarks/native_expert_scaling/meth294_window_forward.h',exe,bundle,native,log,err]
        result={'experiment':'METH-295-actual-native-primary-new-source-prediction-repair1',
            'artifact_sha256':M.CORE_SHA,'manifest_sha256':M.MANIFEST_SHA,'answerability_sha256':M.ANSWER_SHA,
            'annotations_sha256':M.ANNOTATIONS_SHA,'policy_sha256':M.POLICY_SHA,'bridge_sha256':M.BRIDGE_SHA,
            'preserved_failure_sha256':FAILURE_SHA,'GPU_controls_partial_sha256':PARTIAL_SHA,
            'source_sha256':M.P.M57.MODEL_SHA,'parent_checkpoint_sha256':M.P.M57.CHECKPOINT_SHA,
            'child_checkpoint_sha256':M.P.M122.SPECIALIZED_SHA,
            'arms':arms,'summary':summary,'GPU_stored_descriptive_summary':descriptive,'bootstrap':M.bootstrap(arms),
            'gates':gates,'full_head_oracles':oracles,'cpu_runtime':cpu,
            'GPU_reused_evidence':{'partial_seconds':saved['seconds'],'last_observed_stage_seconds':173.76599999999598,
                'end_rss_bytes':3307409408,'peak_allocated_bytes':4697787904,'load_record_not_retained_by_original_partial':True},
            'cases':[{'case':i,'kind':c['kind'],'source_id':items[c['item']]['source_id'],'n':len(c['ids']),
                'offset':c['offset'],'first':c['first'],'document_first':c['document_first']} for i,c in enumerate(cases)],
            'compile_command':command,'sha256':{str(p.relative_to(M.P.ROOT)):M.P.digest(p) for p in paths},
            'seconds':time.monotonic()-start,'CUDA_initialized':M.torch.cuda.is_initialized(),
            'diagnostic_only':True,'native_promotion_qualified':False,
            'decision':'native_primary_prediction_pass_freeze_generation' if all(gates.values()) else 'native_primary_prediction_fail_close_fixed_native_profile',
            'scope':'Exact saved fresh GPU controls plus ALL24-source actual original native profile. Only process isolation/reuse fixes apparatus. No precision/operator/data/threshold change. Numerical5% failures retained; generation/semantics/PIQA/CPU K64/accepted rate/useful n/RAM/DRAM/family unqualified.'}
        assert result['CUDA_initialized'] is False
        M.dump(args.out,result);print(json.dumps({k:result[k] for k in ('decision','summary','gates','seconds')}),flush=True)
    except BaseException as error:
        M.dump(args.out.with_suffix('.failure.json'),{'stage':stage,'error':type(error).__name__+': '+str(error),
            'seconds':time.monotonic()-start,'cpu_runtime':cpu});raise

if __name__=='__main__':main()
