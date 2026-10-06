"""Bounded terminal metadata admission and derived whole-cost report, no replay."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_484_20261006.json'
REPORT=DOC/'METH_484_SHARED_INTEGER_ELIGIBILITY_RESULT_20261006.md'
START=time.monotonic()
assert not DEST.exists() and not REPORT.exists()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        while b:=f.read(1<<20):
            h.update(b);assert time.monotonic()-START<60
    return h.hexdigest()


def read(name,expected=None):
    p=DOC/name;digest=sha(p)
    if expected:assert digest==expected
    return json.loads(p.read_bytes()),{'path':str(p),'bytes':p.stat().st_size,'sha256':digest}


raw,rawfile=read('meth484_shared_integer_eligibility_result.json','389deaba1ccd0315778388032c1f8e824fe6439d0a7ed3b8947e1eb0f3d001b4')
ret,retfile=read('RETENTION_484_20261006.json','27d3e0c9f2ec9b7160c4bf7a63685f5aef7e1a8b5e60e8226605877055eaf6dc')
binding,bindingfile=read('meth484_source_binding.json','9cbbececd51bc1cf748502db585f826ae1a21f693943b97673121e120d824e1c')
assert ret['raw']==rawfile and raw['binding_sha256']==bindingfile['sha256'] and all(raw['gates'].values())and all(ret['gates'].values())
instances=[]
for record,registration,window in((raw,'meth484_sole_first_registration.json','meth484_windows_terminal.json'),
                                 (ret,'meth484_audit_sole_first_registration.json','meth484_audit_windows_terminal.json')):
    reg,regfile=read(registration);p=ROOT/'results/native_expert_scaling'/window;w=json.loads(p.read_bytes())
    assert reg['actual_exit_code']==0 and reg['no_remaining_live_handle'] and reg['session_id']is None
    assert reg['terminal_seconds']<=60 and reg['terminal_peak_working_set_bytes']<=256<<20
    assert w['query_available'] and w['query_error']is None and not w['matching_scientific_events'] and len(w['instances'])==1
    i=w['instances'][0];assert i['pid']==record['process_instance']['pid'] and i['create_time_unix']==record['process_instance']['create_time_unix']
    for key in('start_utc','end_utc'):
        assert datetime.datetime.fromisoformat(i[key])==datetime.datetime.fromisoformat(record[key])
    assert datetime.datetime.fromisoformat(record['start_utc']).timestamp()-record['process_instance']['create_time_unix']<2
    instances.append({'record':rawfile if record is raw else retfile,'registration':regfile,'sole_exec_chunk':reg['sole_exec_chunk'],
                      'actual_exit_code':0,'terminal_seconds':reg['terminal_seconds'],'terminal_peak_working_set_bytes':reg['terminal_peak_working_set_bytes'],
                      'windows':{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'available':True,'matching_faults':0,'actual_instance':i}})
assert instances[0]['windows']['sha256']==ret['main_windows']['sha256']
for name in('meth484_shared_integer_eligibility_result.failure.json','RETENTION_484_20261006.failure.json'):
    assert not(DOC/name).exists()
for p in binding['helpers']:
    assert sha(p['path'])==p['sha256']
assert not subprocess.check_output(['git','diff','HEAD','--',str(Path(__file__).relative_to(ROOT)),str((DOC/'meth484_sole_first_registration.json').relative_to(ROOT)),str((DOC/'meth484_audit_sole_first_registration.json').relative_to(ROOT))],cwd=ROOT)
sources={}
for n,s in raw['sources'].items():
    assert s['byte_budget']==ret['sources'][n]['budget']
    b=s['byte_budget'];pool=b['groups']['CPU_expert'];router=b['groups']['CPU_F32_router']
    per_n=(pool['value_bytes']+pool['row_scale_bytes']+router['value_bytes'])//int(n)
    fixed=b['payload_bytes']-int(n)*per_n
    constraint=next(v for v in s['whole_cost']['constraints']if v['metric']=='prose'and v['target_accepted_IDs_per_second']==50)
    full=s['whole_cost']['inherited_profile1_aggregate_sum_of_96_case_means_seconds'];f=s['whole_cost']['eligible_measured_fraction']
    calls=s['aggregate_logical_transfer_and_invocations']['proposed_integer_GEMM_invocations']
    limits=[]
    for r in(0.,.5,constraint['maximum_component_multiplier_at_h0']):
        h=constraint['required_whole_cost_ratio']-(1-f)-f*r
        limits.append({'component_multiplier_r':r,'maximum_h':h,'maximum_additional_seconds_for_ALL96':h*full,
                       'maximum_mean_additional_seconds_per_proposed_invocation':h*full/calls})
    sources[n]={'byte_budget':b,'logical_transfer':s['aggregate_logical_transfer_and_invocations'],
                'prose50_whole_constraint':constraint,'illustrative_overhead_limits':limits,
                'fixed_geometry_model_file_law':{'fixed_bytes':fixed,'bytes_per_added_expert_ID_across_ALL_banks':per_n,
                    'formula':'model_file_bytes(n)=fixed_bytes+n*bytes_per_added_expert_ID_across_ALL_banks',
                    '80GiB_model_only_upper_n_excluding_OS_runtime_state_workspaces':((80<<30)-fixed)//per_n,
                    'not_trained_capacity_or_actual_residency_measurement':True}}
assert sources['128']['fixed_geometry_model_file_law']==sources['256']['fixed_geometry_model_file_law']
admission={'experiment':'METH484 BOTH sole invocations and actual Windows metadata final admission',
           'head_at_finalization':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           'finalizer_sha256':sha(__file__),'binding':bindingfile,'instances':instances,'sources':sources,
           'gates':{'main_ALL7_and_independent_ALL6_gates':True,'BOTH_actual_exit0_within60seconds256MiB':True,
                    'BOTH_Windows_available_zero_fault_matching_PID_create_ISOinstants':True,
                    'complete_manifest_alias_and_source_runtime_audit_admitted':True,
                    'derived_whole_overhead_and_fixed_format_RAM_law_exact':True,'no_GPU_model_native_payload_solver_or_completed_inquiry_replay':True},
           'decision':'COMPLETE_METADATA_ELIGIBILITY_ADMITTED_ONE_COMPLETE_BACKEND_CANDIDATE_PERMITTED_NO_SPEED_QUALITY_OR_GENERALITY_PROMOTION',
           'goal_status':'ACTIVE_INCOMPLETE','seconds':time.monotonic()-START}
with DEST.open('xb')as f:f.write((json.dumps(admission,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
y=sources['256'];budget=y['byte_budget'];limits=y['illustrative_overhead_limits'];law=y['fixed_geometry_model_file_law']
text=f'''# METH484: shared integer metadata eligibility independently admitted

6 October2026. COMPLETE model-free eligibility; full goal ACTIVE/INCOMPLETE.
[Prospective protocol](METH_484_SHARED_INTEGER_ELIGIBILITY_PROTOCOL_20261006.md).
[Complete final admission](ADMISSION_484_20261006.json).

## What is established

ALL3320/6392 descriptor names of original128/256 artifacts independently checked:
shapes/codecs/config/offsets/scales/aliases/EOF,169shared matrices per artifact,
all original expert IDs/12F32routers/64F32controls/three tied embedding names.
Every physical component deduplicated; exact file-byte sum, no partial overlap,
no padding bytes. No weights mapped/read or payload digest refreshed. Original
458 payload SHA inherited with unchanged path/size/mtime; this is metadata
eligibility, not fresh payload/driver/numeric qualification. Exact374 and388
kernels match458;389 is contract record, not kernel filename. Inherited compiler
contracts retain no-fast-math/no-FP-contraction. Frozen675tracked source name
catalog includes only existing CUBLAS_WORKSPACE_CONFIG settings and new inquiry
names, no implemented backend identified in that narrow search. No semantic
global absence/novelty claim; untracked/other branches/external libs excluded.

| Geometry budget |128experts|256experts|
|---|---:|---:|
|Sharedattention I8 codes|84,934,656B|84,934,656B|
|Shared denseFFN I8 codes|56,623,104B|56,623,104B|
|Sharedhead I8 codes|24,674,304B|24,674,304B|
|Eligible GPUresident codes|166,232,064B (158.53125MiB)|166,232,064B (158.53125MiB)|
|Maximum GPUoperator scratch|3,932,160B (3.75MiB)|3,932,160B|
|Maximum CPUoperator arrays|7,472,128B (7.12598MiB)|7,472,128B|
|Shared CPUrow-scales|755,200B|755,200B|
|Actual model file|7,541,946,880B|14,818,015,744B|

GPU storage independent of n ONLY for this fixed geometry. Original models have
different weight values/workers; this is not a causal usefulness comparison.
Expert pool/router stayCPU/RAM. CPUarray figure includes existing F32input/output/
A16codes, not an incremental whole-RAM requirement. Driver/library/allocator/
runtime/whole-state workspace/OS overhead unqualified and separately charged.

Derived SAME-FORMAT geometry law from admitted components:

```
model_file_bytes(n)={law['fixed_bytes']}+n*{law['bytes_per_added_expert_ID_across_ALL_banks']}
80GiB model-file-only bound: n<={law['80GiB_model_only_upper_n_excluding_OS_runtime_state_workspaces']}
```

This is a representation byte law, excluding OS/runtime/state/workspaces; no
learned additional functions, no100B payload/quality and no measuredDRAM residency
or bandwidth. n counts IDs PER sparse bank, not the sum over twelve banks.

## Algebra and whole cost

Independent new scalar control checks ALL65536 signed16 values:
q=q0+128q1+16384q2, I8digits ranges[0,127]/[0,127]/[-2,1]. Four columns/query,
fourthZERO; I32partial/I64reconstruct/original AVXlane bounds safe. Source CPU
absmax/RNE/A16codes and F64 row-scale then query-scale then F32 remain required.
No GPU arithmetic operation or new compiled parity inferred from this identity.

458 profile1 aggregate is SUM of96case means. Dense+head fraction for256
f=.45986422578853203; prose43.44347306841726accepted IDs/s. Conditional50 requires
(1-f)+f*r+h<=.8688694613683452; at h=0, r<=.7148494462538275. This is a necessary
whole-budget target (~28.5%stage reduction), not measured backend benefit. All
dense/control shapes are eligible, but their aggregate does not reveal separate
attention/FFN/matrix times. Router SCORE2.28% omits mass/control in residual.

ALL96cases including rejected are logically charged once; proposed256 layout:

```
324381 eligible projection queries; 163101 integer GEMM invocations
H2D digits=1209332736B; D2H four I32partials=5371705344B
GPU weights uploaded once per initialization=166232064B (untimed here)
```

Three digits/encoderpositions belong to one request, not batched user requests.
Counts are not physical reads, measured bandwidth or latency. For the fixed
invocation layout, additional overhead h at ideal r=0 permits at most
{limits[0]['maximum_additional_seconds_for_ALL96']:.9f}s across96cases, or
{limits[0]['maximum_mean_additional_seconds_per_proposed_invocation']*1e6:.6f}us/invocation on average.
At illustrative r=.5, at most{limits[1]['maximum_additional_seconds_for_ALL96']:.9f}s,
{limits[1]['maximum_mean_additional_seconds_per_proposed_invocation']*1e6:.6f}us/invocation.
These conditional ceilings include extra pack/copy/launch/sync/post cost wherever
it lies outside r; no double counting or omitted original CPUquantizer time.
They expose a material latency risk despite the small memory footprint.100prose
cannot be reached by dense+head alone at any nonnegative r,h with these source
fractions; residual+experts already exceed that required ratio. Source profile
numbers are inherited, no old timing/main/model/capture/audit/control replay.

## Complete independent retention and resources

Freeze409390d (complete six files), binding9abdd98; firstmain registration9141a70,
soleexec chunkf88625 actualexit0/2.734s/terminalpeak79,757,312B. Mainrecord's peak
61,304,832B is before output serialization; terminal higher peak explicitly kept.
Audit frozen in409390d, firstregistration5095845, solechunk0f17ba actualexit0/
2.625s/terminalpeak61,898,752B. Main7gates/audit6gates ALLPASS. BOTH Application1000
queries available/zero matching controller faults, exactPID/create/awareISOtimes.
ALL1536inherited rows with3072dense/1536headcounter entries/ALL192case counts,
ALLdescriptors/aliases/budgets/name-catalog and conditional arithmetic admitted.
Source/runtime binding1865files, metadata85MB/89MBhashed per main/audit; no model
or CUDA library import. No firstfault or repair, no remaining live science handle.

RAW SHA{rawfile['sha256']}

RET SHA{retfile['sha256']}

## Decision and remaining goal

ONE complete integer backend candidate is metadata-justified; no speed/quality/
driver promotion. First freeze its complete implementation, inputs, controls,
runtime qualification, C integration, actual whole quality/rate/initialization
cost gates and stop budget before any GPUcontrol. Choose one layout, no sweep.
If runtime/arithmetic/whole-budget gate fails, retain it and stop that recipe.
Compact core acceleration is an execution enabler. Function-compatible transfer,
causal useful distinctn, CPU LUTwinnerANDmass scaling, physicalDRAM, fresh whole
donor-relative quality ANDSAME>=50batch1 IDs/s and actual otherfamilies/~100B remain
open. [Whole reassessment and exact next](METH_484_WHOLE_ALGEBRA_AND_NEXT_20261006.md).
'''
with REPORT.open('xb')as f:f.write(text.replace('\n','\r\n').encode())
print(json.dumps({'admission_sha256':sha(DEST),'result_sha256':sha(REPORT),'gates':admission['gates'],'derived':sources['256']['illustrative_overhead_limits'],'law':law}))
