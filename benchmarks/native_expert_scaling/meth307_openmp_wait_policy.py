#!/usr/bin/env python3
"""Frozen PASSIVE/ACTIVE/PASSIVE comparison on the unchanged306 executable."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import meth306_packed_i4_preflight as B

M=B.M
PRIOR=M.DOC/'meth306_packed_i4_preflight_result.json'
PRIOR_SHA='bb723d4cdb5027869e26668bf652fdd122ffe6a66385ca2608208fcd24328182'
DRIVER_SHA='21ed4f3563b30dbd9021a3f4f01d9da1f3d9eb14ba784d241ff81f54a9fa1f26'
CPU_SHA='c3a1610dfeef1b98edd9d5d8e41509493084841c8cb3f70259ecafcb4f2e729c'
EXE_SHA='f7a22f25a7531bce4d9f8d67a05839039e764f9889907af69bc2727ccfa16de1'

def effective(stderr,name):
    text=stderr.split('Effective settings',1)[-1]
    values=re.findall(r'^\s*'+re.escape(name)+r'\s*=\s*(.*?)\s*$',text,re.MULTILINE)
    assert len(values)==1,(name,values)
    return values[0].strip("'\"").upper()

def run(path):
    start=time.monotonic();stage='bindings';result={'experiment':'METH-307-exact306-worker-wait-ABA','arms':[]}
    try:
        assert M.digest(PRIOR)==PRIOR_SHA and M.digest(Path(B.__file__))==DRIVER_SHA and M.digest(B.CPU)==CPU_SHA
        assert M.digest(Path(M.__file__))=='3b0300ada9b645fbf4576dff301cfa93bc0635ab8254e66d080de83570ef4765'
        for rel,sha in M.PINS.items():assert M.digest(M.ROOT/rel)==sha,rel
        p=json.loads(PRIOR.read_text(encoding='utf-8'));exe=Path(p['compile']['command'][p['compile']['command'].index('-o')+1])
        assert M.digest(exe)==EXE_SHA and p['decision']=='reject_unchanged_packed_I4_kernel_before_teacher_collection_or_training'
        result.update({'prior_sha256':PRIOR_SHA,'immutable306_controller_sha256':DRIVER_SHA,'immutable306_cpu_sha256':CPU_SHA,
                       'executable_sha256':EXE_SHA,'controller_sha256':M.digest(Path(__file__)),
                       'inherited_omp_kmp_environment':{k:v for k,v in os.environ.items() if k.startswith(('OMP_','KMP_'))}})
        root=M.OUT/'meth307_wait_policy';root.mkdir(exist_ok=True);M.SPEC=root/'meth307_spec.bin'
        stage='fresh_source_binding';result['spec'],result['source_binding']=M.make_spec()
        assert result['spec']['sha256']==p['spec']['sha256'] and result['source_binding']==p['source_binding']
        M.EXE=exe;original_popen=subprocess.Popen
        for label,policy in [('A','PASSIVE'),('B','ACTIVE'),('C','PASSIVE')]:
            stage='native_'+label;M.OUT=root/label;M.OUT.mkdir(exist_ok=True)
            def policy_popen(*args,**kwargs):
                env=kwargs['env'].copy();env.pop('KMP_LIBRARY',None);env.pop('KMP_BLOCKTIME',None)
                env['OMP_WAIT_POLICY']=policy;env['KMP_SETTINGS']='TRUE'
                if policy=='ACTIVE':env['KMP_BLOCKTIME']='200'
                kwargs['env']=env;return original_popen(*args,**kwargs)
            arm={'label':label,'policy':policy,'gates':{}}
            try:
                subprocess.Popen=policy_popen;B.native(arm,B.ledger())
            finally:subprocess.Popen=original_popen
            result['arms'].append(arm);n=arm['native'];n['environment'].update({'OMP_WAIT_POLICY':policy,'KMP_SETTINGS':'TRUE'})
            n['environment']['KMP_BLOCKTIME']='200' if policy=='ACTIVE' else 'removed; runtime PASSIVE default'
            n['environment']['KMP_LIBRARY']='removed for both policies'
            assert n['selftest']==p['native']['selftest']
            for key in ('parent_count','children_per_parent','functions_per_layer','threads','code_bits','row_tile','serial_if_coefficients_less_than','allocated_bytes','active_coefficients','active_row_scale_bytes','stored_coefficients'):
                assert n['ready'][key]==p['native']['ready'][key],key
            assert [v['output_route_hash'] for v in n['operators']]==[v['output_route_hash'] for v in p['native']['operators']]
            stderr=Path(n['stderr_path']).read_text(encoding='utf-8')
            runtime={'OMP_WAIT_POLICY':effective(stderr,'OMP_WAIT_POLICY'),'KMP_BLOCKTIME':effective(stderr,'KMP_BLOCKTIME')}
            assert runtime['OMP_WAIT_POLICY']==policy
            assert re.fullmatch('200(?:MS)?' if policy=='ACTIVE' else '0(?:MS)?',runtime['KMP_BLOCKTIME']),runtime
            arm['effective_runtime_wait_settings']=runtime
        a,b,c=(v['native'] for v in result['arms'])
        medians=[statistics.median(v['rep_medians_seconds']) for v in (a,b,c)]
        drift=max(medians[0],medians[2])/min(medians[0],medians[2])
        result['aggregate']={'arm_medians_seconds':medians,'passive_A_C_ratio':drift,
                             'active_over_passive_A':medians[1]/medians[0],'active_over_passive_C':medians[1]/medians[2]}
        result['gates']={'all90_hashes_and_selftests_exact306':True,'effective_wait_settings_verified':True,
                         'each_arm_repeatability_1_10':all(v['gates']['repeatability_1_10'] for v in result['arms']),
                         'passive_A_C_drift_1_10':drift<=1.10,'active_each_median_14ms':max(b['rep_medians_seconds'])<=.014}
        if not result['gates']['each_arm_repeatability_1_10'] or not result['gates']['passive_A_C_drift_1_10']:
            result['decision']='wait_policy_attribution_inconclusive_variation_no_training'
        elif not result['gates']['active_each_median_14ms']:
            result['decision']='active_wait_profile_cost_stop_before_teacher_collection_or_training'
        else:result['decision']='eligible_only_for_separately_frozen_real_teacher_function_fit_and_I4_quality'
        result['scope']='Exact306 binary/spec/packed fixtures, new declared worker-wait profile. Original PASSIVE failures retained. No trained donor quality/useful n/physical DRAM/causal model or accepted decode rate.'
        result['total_seconds']=time.monotonic()-start;M.write(path,result)
        print(json.dumps({'decision':result['decision'],'gates':result['gates'],'aggregate':result['aggregate'],
                          'rep_medians_ms':[[x*1000 for x in v['native']['rep_medians_seconds']] for v in result['arms']],
                          'result_sha256':M.digest(path),'total_seconds':result['total_seconds']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'total_seconds':time.monotonic()-start})
        M.write(path.with_suffix('.failure.json'),result);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists();run(args.out)
