"""Model-free consumed358 phase variability and actual Windows physical topology."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import time
import numpy as np
import psutil
import meth324_switch_reference as M

BASELINE=M.DOC/'meth358_switch_all_a16_cost_resume_result.json'
BASELINE_SHA='1387f8b461d15f6ba12ffbec67a115f893dc4cff9f65e39518b6fd5000ea9836'
PROTOCOL=M.DOC/'METH_359_SWITCH_COST_TOPOLOGY_PROTOCOL_20261004.md'


def physical_topology():
    class Relationship(ctypes.Union):
        _fields_=[('reserved',ctypes.c_uint64*2),('flags',ctypes.c_byte)]
    class Info(ctypes.Structure):
        _fields_=[('mask',ctypes.c_size_t),('relationship',ctypes.c_int),('detail',Relationship)]
    assert ctypes.sizeof(ctypes.c_void_p)==8 and ctypes.sizeof(Info)==32
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);fn=kernel.GetLogicalProcessorInformation
    fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(wintypes.DWORD)];fn.restype=wintypes.BOOL
    length=wintypes.DWORD(0);assert not fn(None,ctypes.byref(length)) and ctypes.get_last_error()==122
    buffer=ctypes.create_string_buffer(length.value);assert fn(buffer,ctypes.byref(length)) and length.value%ctypes.sizeof(Info)==0
    infos=(Info*(length.value//ctypes.sizeof(Info))).from_buffer(buffer)
    cores=[];union=set()
    for entry in infos:
        if entry.relationship==0:
            logical=[i for i in range(64) if entry.mask&(1<<i)];assert logical and not union.intersection(logical);union.update(logical)
            cores.append({'processor_mask':int(entry.mask),'SMT_flag':int(entry.detail.flags),'logical_processors':logical})
    cores.sort(key=lambda c:c['logical_processors'][0]);allowed=psutil.Process().cpu_affinity()
    selected=[min(set(c['logical_processors'])&set(allowed)) for c in cores]
    assert len(cores)==psutil.cpu_count(logical=False)==6 and len(union)==psutil.cpu_count()==12
    assert len(selected)==len(set(selected))==6 and set(selected)<=set(allowed)
    return {'api':'GetLogicalProcessorInformation/RelationProcessorCore','group_scope':'single group/local<=64 logical processors',
            'physical_cores':cores,'process_allowed_logical_processors':allowed,'selected_one_logical_per_physical_core':selected,
            'selected_processor_mask':sum(1<<i for i in selected),'physical_cpu_count':len(cores),'logical_cpu_count':len(union)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start=time.monotonic();stage='bindings';result={'experiment':'METH-359-consumed-cost-and-physical-topology','measurements':[]}
    try:
        for path in (Path(__file__),PROTOCOL,BASELINE,Path(M.__file__)):M.committed(path)
        assert M.digest(BASELINE)==BASELINE_SHA
        raw=json.loads(BASELINE.read_text(encoding='utf-8'));assert not raw['gates']['all_threads6_repeat_ratio_le1p10']
        stage='actual_windows_topology';result['topology']=physical_topology()
        stage='consumed_phase_step_summary'
        for record in raw['measurements']:
            rows=[r for r in record['rows'] if r['repetition']>=0];summaries=[]
            for row in rows:
                steps=np.asarray(row['decode_step_seconds']);assert len(steps)==32 and np.isfinite(steps).all() and np.all(steps>0)
                median=float(np.median(steps));maximum=float(steps.max())
                summaries.append({'repetition':row['repetition'],'encoder_seconds':row['encode_seconds'],'cross_kv_seconds':row['cross_kv_seconds'],
                                  'decode_seconds':row['decode_seconds'],'full_seconds':row['full_seconds'],'median_step_seconds':median,
                                  'maximum_step_seconds':maximum,'maximum_step_position':int(steps.argmax()),'maximum_to_median_step_ratio':maximum/median,
                                  'excess_sum_over_median_seconds':float(np.maximum(steps-median,0).sum()),'maximum_excess_over_median_seconds':maximum-median})
            phases={key:{'max_min_ratio':max(r[key] for r in summaries)/min(r[key] for r in summaries),'range_seconds':max(r[key] for r in summaries)-min(r[key] for r in summaries)}
                    for key in ('encoder_seconds','cross_kv_seconds','decode_seconds','full_seconds')}
            result['measurements'].append({'source_length':record['source_length'],'threads':record['threads'],'rows':summaries,'phases':phases})
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),'baseline358_sha256':BASELINE_SHA,
                       'gates':{'consumed358_exact':True,'single_group_six_physical_twelve_logical_cores_closed':True,'one_allowed_logical_per_physical_core_identified':True},
                       'resource':{'main_seconds_excluding_imports':time.monotonic()-start,'end_rss_bytes':psutil.Process().memory_info().rss},
                       'decision':'test_new_process_affinity_profile_with_same356_binary_fixed_gates_no_unchanged_rerun',
                       'scope':'No inference/native timings/new data, no causal attribution of variance to SMT/migration. Actual topology only motivates a separately frozen profile.358 cost failure and351 quality failures remain; no physical DRAM/useful n proof.'})
        assert time.monotonic()-start<=60 and psutil.Process().memory_info().rss<=256<<20
        M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'topology':result['topology'],'measurements':result['measurements'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
