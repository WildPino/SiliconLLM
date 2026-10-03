#!/usr/bin/env python3
"""Three fresh exact306 processes on one logical CPU per physical core."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
import psutil

import meth307_openmp_wait_policy as W
B=W.B
M=W.M

def topology():
    class U(C.Union):_fields_=[('reserved',C.c_uint64*2)]
    class Info(C.Structure):_fields_=[('mask',C.c_uint64),('relationship',C.c_uint32),('u',U)]
    assert C.sizeof(Info)==32 and C.sizeof(C.c_void_p)==8
    api=C.WinDLL('kernel32',use_last_error=True).GetLogicalProcessorInformation
    api.argtypes=[C.c_void_p,C.POINTER(C.c_uint32)];api.restype=C.c_int
    size=C.c_uint32();assert not api(None,C.byref(size)) and C.get_last_error()==122
    raw=C.create_string_buffer(size.value);assert api(raw,C.byref(size)) and size.value%32==0
    cores=[]
    for off in range(0,size.value,32):
        item=Info.from_buffer_copy(raw.raw[off:off+32])
        if item.relationship==0:cores.append([i for i in range(64) if item.mask&(1<<i)])
    assert len(cores)==6 and sorted(sum(cores,[]))==list(range(12))
    allowed=psutil.Process().cpu_affinity();selected=[min(i for i in core if i in allowed) for core in cores]
    assert len(set(selected))==6
    return {'physical_core_logical_ids':cores,'process_allowed_logical_ids':allowed,'selected_logical_ids':selected,
            'windows_api_bytes':size.value,'api':'GetLogicalProcessorInformation/RelationProcessorCore'}

def bindings(stderr,selected):
    observed={};lines=[]
    for line in stderr.splitlines():
        match=re.search(r'KMP_AFFINITY:.*\bthread\s+(\d+)\s+bound to OS proc set\s+([0-9{},\s-]+)$',line)
        if not match:continue
        thread=int(match[1]);mask=match[2].strip()
        if mask.startswith('{'):assert mask.endswith('}');mask=mask[1:-1].strip()
        assert re.fullmatch(r'\d+',mask),'non-singleton binding'
        cpu=int(mask);assert 0<=thread<6 and cpu==selected[thread],(thread,cpu,selected)
        if thread in observed:assert observed[thread]==cpu
        observed[thread]=cpu;lines.append(line)
    assert sorted(observed)==list(range(6)) and sorted(observed.values())==sorted(selected),'missing actual thread bindings'
    return {'thread_to_logical_cpu':observed,'actual_runtime_binding_lines':lines}

def run(path):
    start=time.monotonic();stage='bindings';result={'experiment':'METH-308-exact306-physical-core-active-profile','runs':[]}
    try:
        assert M.digest(W.PRIOR)==W.PRIOR_SHA and M.digest(Path(B.__file__))==W.DRIVER_SHA and M.digest(B.CPU)==W.CPU_SHA
        assert M.digest(Path(W.__file__))=='3c08734e08a890028d6bc16923358ebde9ac727a32a1271237c0b9464d276352'
        assert M.digest(Path(M.__file__))=='3b0300ada9b645fbf4576dff301cfa93bc0635ab8254e66d080de83570ef4765'
        for rel,sha in M.PINS.items():assert M.digest(M.ROOT/rel)==sha,rel
        prior=json.loads(W.PRIOR.read_text(encoding='utf-8'))
        exe=Path(prior['compile']['command'][prior['compile']['command'].index('-o')+1]);assert M.digest(exe)==W.EXE_SHA
        topo=topology();result['topology']=topo
        affinity='verbose,granularity=thread,proclist=['+','.join(str(i) for i in topo['selected_logical_ids'])+'],explicit'
        result.update({'prior_sha256':W.PRIOR_SHA,'immutable306_controller_sha256':W.DRIVER_SHA,'immutable306_cpu_sha256':W.CPU_SHA,
                       'executable_sha256':W.EXE_SHA,'controller_sha256':M.digest(Path(__file__)),
                       'inherited_omp_kmp_environment':{k:v for k,v in os.environ.items() if k.startswith(('OMP_','KMP_'))}})
        root=M.OUT/'meth308_physical_core';root.mkdir(exist_ok=True);M.SPEC=root/'meth308_spec.bin'
        stage='fresh_source_binding';result['spec'],result['source_binding']=M.make_spec()
        assert result['spec']['sha256']==prior['spec']['sha256'] and result['source_binding']==prior['source_binding']
        M.EXE=exe;original_popen=subprocess.Popen
        for number in range(1,4):
            stage='native_'+str(number);M.OUT=root/str(number);M.OUT.mkdir(exist_ok=True)
            config={'OMP_WAIT_POLICY':'ACTIVE','KMP_BLOCKTIME':'200','KMP_SETTINGS':'TRUE','KMP_AFFINITY':affinity}
            def bound_popen(*args,**kwargs):
                env=kwargs['env'].copy()
                for key in ('KMP_LIBRARY','OMP_PLACES','OMP_PROC_BIND'):env.pop(key,None)
                env.update(config);kwargs['env']=env;return original_popen(*args,**kwargs)
            item={'number':number,'gates':{}};result['runs'].append(item)
            try:
                subprocess.Popen=bound_popen;B.native(item,B.ledger())
            finally:subprocess.Popen=original_popen
            native=item['native'];native['environment'].update(config)
            native['removed_competing_environment_keys']=['KMP_LIBRARY','OMP_PLACES','OMP_PROC_BIND']
            assert native['selftest']==prior['native']['selftest']
            for key in ('parent_count','children_per_parent','functions_per_layer','threads','code_bits','row_tile','serial_if_coefficients_less_than','allocated_bytes','active_coefficients','active_row_scale_bytes','stored_coefficients'):
                assert native['ready'][key]==prior['native']['ready'][key],key
            assert [x['output_route_hash'] for x in native['operators']]==[x['output_route_hash'] for x in prior['native']['operators']]
            stderr=Path(native['stderr_path']).read_text(encoding='utf-8')
            assert W.effective(stderr,'OMP_WAIT_POLICY')=='ACTIVE' and re.fullmatch(r'200(?:MS)?',W.effective(stderr,'KMP_BLOCKTIME'))
            item['actual_bindings']=bindings(stderr,topo['selected_logical_ids'])
        process_medians=[statistics.median(x['native']['rep_medians_seconds']) for x in result['runs']]
        drift=max(process_medians)/min(process_medians)
        result['process_medians_seconds']=process_medians;result['max_over_min_process_median']=drift
        result['gates']={'all90_hashes_selftests_counts_exact306':True,'actual_six_distinct_physical_core_bindings':True,
                         'effective_active200ms':True,'each_process_repeatability_1_10':all(x['gates']['repeatability_1_10'] for x in result['runs']),
                         'cross_process_repeatability_1_10':drift<=1.10,
                         'all_nine_medians_14ms':all(x['gates']['all_operator_medians_14ms'] for x in result['runs'])}
        if not result['gates']['each_process_repeatability_1_10'] or not result['gates']['cross_process_repeatability_1_10']:
            result['decision']='physical_core_profile_inconclusive_variation_no_training'
        elif not result['gates']['all_nine_medians_14ms']:
            result['decision']='physical_core_profile_cost_stop_before_teacher_collection_or_training'
        else:result['decision']='eligible_only_for_separately_frozen_real_teacher_function_fit_and_I4_quality'
        result['scope']='Exact306 binary/packed640-function fixtures/source head/router/norms. Verified physical-core execution profile only; no trained donor quality/useful n/causal model/physical DRAM or accepted rate.'
        result['total_seconds']=time.monotonic()-start;M.write(path,result)
        print(json.dumps({'decision':result['decision'],'gates':result['gates'],'rep_medians_ms':[[v*1000 for v in x['native']['rep_medians_seconds']] for x in result['runs']],
                          'cross_process_ratio':drift,'result_sha256':M.digest(path),'total_seconds':result['total_seconds']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'total_seconds':time.monotonic()-start})
        M.write(path.with_suffix('.failure.json'),result);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists();run(args.out)
