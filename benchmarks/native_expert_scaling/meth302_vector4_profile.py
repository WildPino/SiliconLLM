#!/usr/bin/env python3
"""Attribute unchanged301 fixture costs; cannot regrade or promote301."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import meth301_vector4_lut_preflight as M

PRIOR=M.DOC/'meth301_vector4_lut_preflight_result.json'
PRIOR_SHA='dc71c85a1d42058fa3eeac5cf83947c27a17e78f5f498b55caec069751aeef2f'
M301_CPU_SHA='d07eedc895a86264ad610790df3123b214c93c8ceabe8c34c0aedfd15fd9c021'
M301_SCRIPT_SHA='9fc3017ffff34674d3e5903b536445d4b43f3bfb45c8694fb69af62cf7d06f04'
CPU=M.ROOT/'benchmarks/native_expert_scaling/meth302_vector4_profile_cpu.c'
EXE=M.OUT/'meth302_vector4_profile.exe'
ORGANS=('mla','routed','shared','dense0','head','router')

def run(path):
    started=time.monotonic();stage='bindings';result={'experiment':'METH-302-unchanged-vector4-native-cost-attribution'}
    try:
        assert M.digest(PRIOR)==PRIOR_SHA and M.digest(M.CPU)==M301_CPU_SHA
        assert M.digest(Path(M.__file__))==M301_SCRIPT_SHA
        p=json.loads(PRIOR.read_text(encoding='utf-8'))
        assert p['decision']=='reject_unchanged_vector4_native_layout_before_palette_training'
        assert M.digest(M.SPEC)==p['spec']['sha256']
        assert not EXE.exists()
        result.update({'prior_sha256':PRIOR_SHA,'immutable301_source_sha256':M301_CPU_SHA,
                       'immutable301_controller_sha256':M301_SCRIPT_SHA,'catalogue':p['spec'],
                       'script_sha256':M.digest(Path(__file__)),'profile_source_sha256':M.digest(CPU),
                       'engine_sha256':M.digest(M.ENGINE),'source_bindings':p['source_bindings']})
        stage='compile'
        command=[str(M.COMPILER),'-O3','-mavx2','-mssse3','-mfma','-fopenmp','-ffp-contract=off','-std=c11',
                 '-DSILICON_VECTOR4_LUT_PROFILE',str(M.ENGINE),'-o',str(EXE),'-lm','-lpsapi']
        c=subprocess.run(command,cwd=M.ROOT,capture_output=True,text=True,timeout=120)
        result['compile']={'command':command,'exit_code':c.returncode,'stdout':c.stdout,'stderr':c.stderr,
                            'compiler_sha256':M.digest(M.COMPILER)}
        assert c.returncode==0,c.stderr
        result['executable_sha256']=M.digest(EXE)
        stage='profile'
        M.OUT=M.OUT/'meth302_profile';M.OUT.mkdir(exist_ok=True);M.EXE=EXE
        run=M.native(64,1);result['native_run']=run
        assert run['ready']['allocated_fixture_bytes']==p['runs'][0]['ready']['allocated_fixture_bytes']
        assert [v['output_hash'] for v in run['tokens']]==[v['output_hash'] for v in p['runs'][0]['tokens']]
        measured=[v for v in run['tokens'] if not v['warmup']]
        def sums(v):return sum(v['table_seconds_by_organ'])+sum(v['matvec_seconds_by_organ'])
        gaps=[v['seconds']-sums(v) for v in run['tokens']]
        assert all(-1e-8<=v<=.001 for v in gaps),'instrumentation closure'
        costs={}
        for i,k in enumerate(ORGANS):
            b=[v['table_seconds_by_organ'][i] for v in measured]
            m=[v['matvec_seconds_by_organ'][i] for v in measured]
            costs[k]={'table_median_seconds':statistics.median(b),'matvec_median_seconds':statistics.median(m),
                      'combined_median_seconds':statistics.median(x+y for x,y in zip(b,m))}
        table=[sum(v['table_seconds_by_organ']) for v in measured]
        router=[v['matvec_seconds_by_organ'][5] for v in measured]
        matvec=[sum(v['matvec_seconds_by_organ'][:5]) for v in measured]
        shares=[m/(m+t+r) for m,t,r in zip(matvec,table,router)]
        residual=[m/(.014-t-r) for m,t,r in zip(matvec,table,router) if t+r<.014]
        result['organ_costs']=costs
        result['aggregate_costs']={'table_median_seconds':statistics.median(table),'router_median_seconds':statistics.median(router),
                                  'matvec_median_seconds':statistics.median(matvec),'matvec_share_median':statistics.median(shares),
                                  'whole_median_seconds':statistics.median(v['seconds'] for v in measured),
                                  'zero_table_same_other_cost_median_seconds':statistics.median(m+r for m,r in zip(matvec,router)),
                                  'required_matvec_speedup_at_unchanged_table_router_median':statistics.median(residual) if residual else None,
                                  'table_router_at_least14ms_states':sum(t+r>=.014 for t,r in zip(table,router)),
                                  'max_unattributed_seconds':max(gaps)}
        result['gates']={'all30_full_output_route_hashes_exact301':True,'numeric_controls':True,
                         'timer_closure_1ms':True,'repeatability_1_10':run['max_over_min_rep_median']<=1.10}
        if not result['gates']['repeatability_1_10']:
            result['decision']='attribution_inconclusive_variation'
        elif result['aggregate_costs']['matvec_share_median']>=.70:
            result['decision']='prioritize_reducing_active_matvec_or_lookup_complexity_not_table_only_tuning'
        else:result['decision']='inspect_construction_router_and_matrix_balance_before_next_mechanism'
        result['scope']='Diagnostic timing of unchanged synthetic source-shaped operators; cannot regrade301, establish donor quality/useful extra n/physical DRAM or accepted end-to-end rate. Conditional timing deductions hold measured other costs fixed.'
        result['total_seconds']=time.monotonic()-started
        M.write(path,result)
        print(json.dumps({k:result[k] for k in ('decision','organ_costs','aggregate_costs','gates','total_seconds')}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'total_seconds':time.monotonic()-started})
        M.write(path.with_suffix('.failure.json'),result);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    run(args.out)
