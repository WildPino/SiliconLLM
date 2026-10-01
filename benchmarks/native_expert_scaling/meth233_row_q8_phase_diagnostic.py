#!/usr/bin/env python3
"""Diagnose phase cost without reopening the stopped METH-232 gate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
PRIOR=DOC/'meth232_full_source_row_q8_native_result.json'
PRIOR_SHA='5dad8dcb67d2b7eeae0c747d63b5cada6e2ce6bd6faed94bac32dd6052c94368'
WEIGHTS=ROOT/'results/native_expert_scaling/meth232_full_source_row_q8_fixture.bin'
VECTORS=ROOT/'benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth125_e1280_vectors.bin'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    for key in ('exe','check','out'): ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args(); assert not args.check.exists() and not args.out.exists()
    start,stage=time.monotonic(),'bindings'
    try:
        assert digest(PRIOR)==PRIOR_SHA
        prior=json.loads(PRIOR.read_text())
        assert prior['gates']['native_component_cost'] is False
        assert digest(WEIGHTS)==prior['binary']['sha256']
        assert digest(VECTORS)==prior['vectors_sha256']
        original=args.exe.parent/'meth232_full_source_row_q8_cpu.c'
        assert digest(original)==prior['source_code_sha256']
        stage='one_instrumented_execution'
        run=subprocess.run([str(args.exe),str(WEIGHTS),str(VECTORS),str(args.check)],
            capture_output=True,text=True,check=True,timeout=600)
        timing=json.loads(run.stdout); phases=json.loads(run.stderr)
        stage='bitwise_arithmetic_identity'
        assert digest(args.check)==prior['output_sha256']
        assert timing['threads']==6 and timing['tokens']==256 and timing['layers']==24
        assert timing['peak_working_set_bytes']<=2*1024**3
        assert timing['weight_bytes']==prior['binary']['bytes']
        assert phases['phase_order']==['input_decode_gate_up','silu_product','down_bias_finite']
        assert len(phases['pass_phase_ms_per_token'])==3
        rows=phases['pass_phase_ms_per_token']
        assert all(len(row)==3 and all(v>0 for v in row) for row in rows)
        medians=[statistics.median(row[i] for row in rows) for i in range(3)]
        matrix_share=(medians[0]+medians[2])/sum(medians)
        decision=('matrix_phases_dominate_choose_changed_matrix_operator' if matrix_share>=.9
            else 'activation_or_instrumentation_share_requires_further_diagnosis')
        result={'experiment':'METH-233-row-Q8-phase-cost-diagnostic','prior_result_sha256':PRIOR_SHA,
            'weight_sha256':prior['binary']['sha256'],'vectors_sha256':prior['vectors_sha256'],
            'original_source_sha256':digest(original),'diagnostic_source_sha256':digest(args.exe.with_suffix('.c')),
            'included_meth182_source_sha256':digest(args.exe.parent/'meth182_group64_ffn_cpu.c'),
            'script_sha256':digest(Path(__file__)),'executable_sha256':digest(args.exe),
            'check_sha256':digest(args.check),'timing':timing,'phases':phases,
            'summary':{'median_phase_ms_per_token':dict(zip(phases['phase_order'],medians)),
                'matrix_phase_share_of_sum_of_medians':matrix_share},
            'gates':{'all_bindings_exact':True,'check_bitwise_equals_meth232':True,'peak_rss_within_2GiB':True},
            'decision':decision,'runtime_seconds':time.monotonic()-start,
            'scope':'Instrumented cost decomposition only; includes timer overhead. METH-232 remains stopped. No cost-gate retry/promotion, fitting, new model quality, expert-count or route/LUT claims.'}
        assert result['runtime_seconds']<=10*60
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'decision':decision,'summary':result['summary'],'gates':result['gates'],
            'runtime_seconds':result['runtime_seconds']}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':repr(error),
            'runtime_seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
        raise


if __name__=='__main__': main()
