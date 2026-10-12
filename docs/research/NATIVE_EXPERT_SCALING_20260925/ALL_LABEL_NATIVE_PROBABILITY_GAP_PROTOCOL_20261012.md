# Frozen all-label native probability-image diagnostic

12 October2026. All4422 FIT/all4386 DEV existing source labels,full65537
vocabulary,current actual F32 C/beta/epsilon,independent per-label oracle codes.
Source/head/moments/domain differ from historical paired255 study. No fitting of
head/model,source forward/T4/new/RESERVED/longalign data. Head quality diagnostic,
not held-out learned-model validation or a preserved autonomous chatbot.

## Qualified inputs and convex search

Require independent GPUF64 initial objective qualification PASS. Initial v,m,H,
loss/gradient from full source/operator preflight;original tF32 from audited native
capture. Exact original case/domain weights and quality budgets retained.
GPU0 UUID/driver/runtime/library bytes bound. Deterministic Torch2.6F64,TF32 off,
workspace4096:8,CPU6 threads/affinity0..5,batch64.

A=[C,beta/sqrt(256epsilon)];L=logsumexp(A v)-m^T v-H,g=A^T p-m.
K=half-ball radius16,last>=0. Projection zeroes negative last then radial clip.
128 maximum round-robin outer updates;30 maximum per-batch backtracks. Initial
Lipschitz1;start each outer batch at max(previous/2,1e-12),double unaccepted rows.
Majorant slack1e-12;all rows must accept and loss increase<=1e-9. Accepted rows
remain frozen while other rows backtrack. Per-label update counts retained.
Budget1200s may stop at a batch boundary,including partial last outer;final point
retained with explicit completed/min/max update counts. No convergence inferred.

Lower=loss-g^T v-16sqrt(||g_first256||²+min(g_last,0)²);retain best raw bound with
point/loss/gradient. Numeric conservative display max(0,raw-1e-8). This is a
real-arithmetic support bound evaluated numerically,not an interval certificate
or a global F32-machine floor. Independent point/gradient/probability checks are
mandatory before a numerical reader-domain obstruction is reported.

## Separately executable uppers

Lift relaxed first256 u onto sphere:clamp its norm to sqrt(256-1e-8),last=
sqrt(max(256-||u||²,1e-8)),t=sqrt(256epsilon)*u/last. Cast tF32;recompute real
positive-sphere v from that actual tF32 and re-evaluate full loss. Retain original
warm tF32;replace best code only if real loss improves>1e-12. No relaxed loss
substituted for executable loss,no top1-only selection,no nonfinite candidate.

Final actual original scalar-norm/head C executes every best code. All577249896
native F32 scores plus norm/inverse words retained. Source q recomputed directly
from stored raw H and actual BF16 source head/gamma/multiplier/sourceepsilon1e-5;
native full KL/ID/entropy/case/domain metrics,original budgets. Full real code loss
must agree with GPU selected loss<=1e-8. Native-vs-real sourceKL delta must fit
full logit span(max(delta)-min(delta))+1e-9. Previous1e-5 false flags stay unchanged.
Precision original score1e-4,norm/inverse2e-5 retained as separate flags;quality/
precision failures do not short-circuit full capture/audit.

## Stored points and audit scope

Initial,every16 completed outer,and final packets contain full current v/loss/g,
steps/update counts,best lower points/loss/g/raw/display bound,best tF32/real loss.
Trace records weighted quantities/calls/time/completed iteration/min/max counts.
Complete independent full functions are required for EVERY stored checkpoint,
every final source probability/native bit/metric. Exact CPU replay of unstored
intermediate GPU calls is not claimed;independent final bounds/executable uppers
are the scientific evidence. A concrete mechanical fault warrants separate
trajectory replay;absence of such replay never proves solver convergence.

Decision fields:actual native oracle quality gates;numeric real-domain lower
violations (case mean>.01,worst label>.1,every-case>.05 as bound original quality).
Keep them separate:real-domain floor is not a proved F32-machine floor,and per-
label oracle success is not causal code acquisition. Native result scientific
admission only after complete independent scalar-FMA bit/source/operator audit.

## Frozen resource and operational contract

Capture1800s,solver1200s,OS **6GiB**,GPU allocated/reserved2GiB,output4GiB.
OS raised from prospective4GiB BEFORE execution to cover simultaneously loaded
Torch runtime,1.074GB actual source F64 head and1.159GB split native mapping plus
checkpoint/data buffers. This is a final implementation choice,not observed-gate
relaxation. CPU audit1200s/OS6GiB. Reserve600s for native/source/hash closure.
Monotonic for caps;perf_counter for total/native durations. Output-only ordered D
namespace,fresh result/terminal PID/creation on failure. Complete consumed extents
before/after,external publishers untouched,timing admission=false. No subprocess.
No hours-long job selected;bounded acquisition30min/audit20min. If a later required
run is hours long,announce/launch bound then user-requested PAUSE;no Luna/delegation.
After this bounded image diagnosis,return to compact causal SSM/SWA/ternary code
transfer/useful n/CPU IDs+mass/DRAM/multiple families/same useful artifact>=50.
