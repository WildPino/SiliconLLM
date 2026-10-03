#!/usr/bin/env python3
"""Attribute immutable303 math; no regrading or training promotion."""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import meth303_compact_i8_preflight as M

PRIOR=M.DOC/'meth303_compact_i8_preflight_result.json'
PRIOR_SHA='396edaee9c201324ffcb35eb57cadfb915a4ea6899694ca9e1055d8c29dd968b'
SOURCE_SHA='eee66947fec515b1f7805a20bd7de2be7b92e665c8a0c4602d1a7a80e2fcdcf5'
DRIVER_SHA='3b0300ada9b645fbf4576dff301cfa93bc0635ab8254e66d080de83570ef4765'
CPU=M.ROOT/'benchmarks/native_expert_scaling/meth304_compact_i8_profile_cpu.c'
ORGANS=('mla','routed','shared','dense0','macro_router','child_query_search','ffn_input','mixture','head')

def exact_math_tokens():
    old=M.CPU.read_text(encoding='utf-8');new=CPU.read_text(encoding='utf-8')
    original=old[old.index('static void execute(void){'):old.index('static Matrix *matrix(')]
    profile=new[new.index('static void profile_execute(void){'):new.index('int main(int argc')]
    stripped=re.sub(r'(?m)^\s*/\*P\*/[^\n]*\n','',profile).replace('static void profile_execute(void){','static void execute(void){')
    stripped=re.sub(r'P_APPLY\(([^,]+),([^,]+),([^)]+)\)',r'checked_apply(\1,\2)',stripped)
    assert re.sub(r'\s+','',stripped)==re.sub(r'\s+','',original),'changed math token stream'
    return True

def run(path):
    start=time.monotonic();stage='bindings';result={'experiment':'METH-304-unchanged-compact-I8-cost-attribution'}
    try:
        assert M.digest(PRIOR)==PRIOR_SHA and M.digest(M.CPU)==SOURCE_SHA and M.digest(Path(M.__file__))==DRIVER_SHA
        for rel,sha in M.PINS.items():assert M.digest(M.ROOT/rel)==sha,rel
        p=json.loads(PRIOR.read_text(encoding='utf-8'))
        assert p['decision']=='reject_unchanged_compact_I8_cost_before_teacher_collection_or_training'
        result.update({'prior_sha256':PRIOR_SHA,'immutable303_source_sha256':SOURCE_SHA,'immutable303_controller_sha256':DRIVER_SHA,
                       'controller_sha256':M.digest(Path(__file__)),'profile_source_sha256':M.digest(CPU),'engine_sha256':M.digest(M.ENGINE),
                       'math_tokens_exact_after_timer_reversal':exact_math_tokens(),'ledger':M.catalogue(),'gates':{}})
        M.OUT=M.OUT/'meth304_profile';M.OUT.mkdir(exist_ok=True)
        M.EXE=M.OUT/'meth304_compact_i8_profile.exe';M.SPEC=M.OUT/'meth304_spec.bin'
        stage='source_and_compile';result['spec'],result['source_binding']=M.make_spec()
        assert result['spec']['sha256']==p['spec']['sha256'] and result['source_binding']==p['source_binding']
        assert not M.EXE.exists()
        command=[str(M.COMPILER),'-O3','-mavx2','-mssse3','-mfma','-fopenmp','-ffp-contract=off','-std=c11',
                 '-DSILICON_COMPACT_I8_PROFILE',str(M.ENGINE),'-o',str(M.EXE),'-lm','-lpsapi','-lbcrypt']
        c=subprocess.run(command,cwd=M.ROOT,capture_output=True,text=True,timeout=120)
        result['compile']={'command':command,'exit_code':c.returncode,'stdout':c.stdout,'stderr':c.stderr,
                           'compiler_sha256':M.digest(M.COMPILER),'libomp_sha256':M.digest(M.TOOLCHAIN/'bin/libomp.dll')}
        assert c.returncode==0,c.stderr
        result['executable_sha256']=M.digest(M.EXE);stage='native';M.native(result,result['ledger'])
        native=result['native'];obs=native['operators']
        assert [x['output_route_hash'] for x in obs]==[x['output_route_hash'] for x in p['native']['operators']]
        assert native['selftest']==p['native']['selftest']
        for k in ('allocated_bytes','active_i8_coefficients','active_row_scale_bytes','stored_i8_coefficients','functions_per_layer'):
            assert native['ready'][k]==p['native']['ready'][k]
        gaps=[]
        for x in obs:
            a=x['organ_seconds'];b=x['projection_seconds'];assert len(a)==len(b)==len(ORGANS)
            assert all(0<=v<600 for v in a+b) and all(0<=y<=z+1e-8 for y,z in zip(b,a))
            gaps.append(x['seconds']-sum(a))
        assert all(-1e-8<=v<=.001 for v in gaps),'timer closure'
        measured=[x for x in obs if not x['warmup']]
        costs={}
        for i,k in enumerate(ORGANS):
            costs[k]={'total_median_seconds':statistics.median(x['organ_seconds'][i] for x in measured),
                      'projection_median_seconds':statistics.median(x['projection_seconds'][i] for x in measured),
                      'other_median_seconds':statistics.median(x['organ_seconds'][i]-x['projection_seconds'][i] for x in measured),
                      'zero_organ_same_other_cost_median_seconds':statistics.median(sum(x['organ_seconds'])-x['organ_seconds'][i] for x in measured)}
        projections=[sum(x['projection_seconds']) for x in measured]
        other=[sum(x['organ_seconds'])-sum(x['projection_seconds']) for x in measured]
        result['organ_costs']=costs
        result['aggregate_costs']={'whole_median_seconds':statistics.median(x['seconds'] for x in measured),
                                  'projection_median_seconds':statistics.median(projections),'other_median_seconds':statistics.median(other),
                                  'projection_share_median':statistics.median(a/(a+b) for a,b in zip(projections,other)),
                                  'max_unattributed_seconds':max(gaps),'min_unattributed_seconds':min(gaps)}
        result['gates'].update({'all30_output_route_hashes_exact303':True,'selftest_exact303':True,'math_tokens_exact303':True,'timer_closure_1ms':True})
        # 303's cost gate is retained as an observation, NEVER regraded by the profile.
        result['original303_cost_failure_remains']=True
        if not result['gates']['repeatability_1_10']:result['decision']='attribution_inconclusive_variation'
        else:
            result['dominant_organ']=max(costs,key=lambda k:costs[k]['total_median_seconds'])
            result['decision']='inspect_dominant_organ_and_projection_schedule_before_new_cost_candidate'
        result['scope']='Unchanged303 synthetic input-ready math/source head/router. Cost attribution only; no capture/training/quality/useful n/physical DRAM/accepted-token promotion.'
        result['total_seconds']=time.monotonic()-start;M.write(path,result)
        print(json.dumps({k:result.get(k) for k in ('decision','dominant_organ','organ_costs','aggregate_costs','gates','total_seconds')}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'total_seconds':time.monotonic()-start})
        M.write(path.with_suffix('.failure.json'),result);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists();run(args.out)
