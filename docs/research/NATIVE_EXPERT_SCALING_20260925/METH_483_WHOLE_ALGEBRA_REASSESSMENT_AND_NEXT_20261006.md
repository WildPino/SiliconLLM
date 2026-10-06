# After483: whole-artifact decision from algebra, geometry and information

6 October2026. Derived source analysis and next-method proposal; no new kernel,
GPU initialization, model run, fit, timing, download or484admission. Full goal
ACTIVE/INCOMPLETE. [483 independently admitted result](METH_483_CONSTRAINT_ROOT_RESULT_20261006.md).

## What the latest inquiry actually resolves

The active-row apparatus now exposes/retains solver vectors and independently
certifies them against original inputs.93root LPs plus5controls, every consumed
query/role/tie/source-ID/report admitted. Original57LPs were reused. No longer
equivalent LP inquiry. All twelve saved heads miss the fixed sufficient physical
sign requirement; all twelve certified brackets still contain both0 and positive
margins above E. The affine class remains UNRESOLVED, not disproved.

For an IDEAL real-valued affine-score router, a grouped root label is a
difference of support functions:

```
winner belongs to R iff max_{e in R} w_e*x >= max_{e in L} w_e*x
```

with a specified ideal tie convention. In those ideal score coordinates each
winner cell is an intersection of halfspaces; a union of winner cells need not
be convex or affine-separable. Native scores also include the input-dependent
A16 quantizer, row scales and F32 rounding/ties. Therefore the ideal cell
statement is NOT a global native-affinity theorem in original x.483 used the
fully preserved NATIVE labels; its finite-data bounds do not assume affinity
of the native teacher.
The balanced original ID groups need not align with function geometry. A single
head per root is therefore a substantial representation hypothesis. Optimizing
its coefficients longer does not resolve whether it is the useful target.
Finite-data brackets also do not establish generalization or global geometry.

## Three coupled problems, with separate mathematical obligations

| Problem | Required object | Current evidence / missing step |
|---|---|---|
| Capacity | Distinct useful functions compatible with one compact core | Small Qwen learned hierarchy positive; real Switch bank utility mixed. Large-donor transfer and causal useful-n remain open. |
| Retrieval | Query-dependent function choice, probability mass and acceptance state at low charged cost | Fixed support/moment/tree recipes closed;483 affine inquiry unresolved. CPU LUT scaling needs actual quality and physical accesses. |
| Execution | Whole time<=20ms per accepted batch1 ID, under declared context/hardware | Original matched cost decomposed; transformed quality ANDsame rate absent. Core/head cost cannot be ignored. |

Memory feasibility is

```
B_RAM(n)=B_core+n*B_function+B_route(n)+B_state+B_workspace+B_runtime.
```

n is approximately proportional to RAM only when those formats/widths/overheads
are controlled.100B/10B gives approximately10x n only for comparable architecture,
not parameter count alone. O(log n) path work is conditional on useful learned
geometry; O(n) stored heads and their physical cache/DRAM accesses still count.
Duplicated IDs are not extra retained donor information.

For source Switch's accepted top1 function, let c be context/capacity state:

```
g(x,c)=a(x,c)*p_e(x)*f_e(x),   a in {0,1}.
||g_hat-g|| <= |a_hat*p_hat-a*p|*||f_e||
               +a_hat*p_hat*||f_hat-f_e||,  p_hat>=0.
```

This derived bound includes acceptance, not only winner identity. Changing IDs
can change capacity counts even when two functions are similar. Final quality
requires the whole state transition. Root-ID accuracy is only a diagnostic.
Propagation through later layers and predictive readout must be assessed on
fresh ownstate contexts; teacher-state local RMS does not bound that loss.

For mass, d=logZ-max>=0 and p=exp(-d). An actual error bound |d_hat-d|<=epsilon
implies |p_hat/p-1|<=exp(epsilon)-1. A path's log-probability errors add with its
depth. Tail mass must be represented:393 and481 rule out the specified top-k/
second-moment shortcuts. Cheap nonlinear conditional mass is still a proposal,
not something supplied by483's affine labels. No normalization work is free.

## Whole time: why the next inquiry changes component

458's matched256 consumed source profile has fractions
f_dense_control=.400130186198696, f_head=.059734039589836044,
f_expert=.2709438759770766, f_router_score=.02278303693856363,
f_residual=.24640886129582773. The routing SCORE fraction does not include all
normalization/control in residual. Dense/control includes attention AND dense
FFN. These are particular original artifacts/contexts/workers, not an n-only
causal comparison or physicalDRAM measurement.
[458 complete evidence](METH_458_SWITCH_MATCHED_WHOLE_COST_RESULT_20261005.md).

For an eligible component fraction f, a backend cost multiplier r, and additional
charged transfer/padding/control fraction h, conditional algebra is

```
T_new/T_old=(1-f)+f*r+h.
```

The matched256 prose rate43.44347306841726 would require ratio<=.8688694613683452
for50accepted IDs/s. If ALL dense/control+head time were eligible, f=.459864225788532
would require r<=approximately.7148494463 with h=0 (about28.5%stage reduction).
That is a CONDITIONAL target, not a measured speedup or eligibility result.
Exact shared-query quantization460/461's small contribution does not meet this
whole gap. The following proposal changes the arithmetic backend of the eligible
matrix operation; it does not repeat that quantization-sharing recipe.

## Exact integer identity: preserve source arithmetic while moving shared work

Static source374 exposes I8 row weights, original CPU RNE A16 activation codes,
integer dot accumulation, then

```
float_y=(float)(((double)integer_sum*(double)row_scale)*(double)query_scale).
```

mv/head_mv/mv_batch use that order; float fallback and all shapes/encodings must
be separately inventoried. Source A16 clips to[-32767,32767], width<=4096. The
proposal below is valid more generally for signed I16, but an existing kernel's
overflow/codec assumptions must be checked before claiming source bit parity.

For unsigned16 representation u of signed16 q and s=1 if u>=32768 else0:

```
q0=u & 127
q1=(u >> 7) & 127
q2=(u >> 14)-4*s
q=q0+128*q1+16384*q2
q0,q1 in [0,127], q2 in {-2,-1,0,1}.
```

All three digits fit signed I8. Unsigned shifts avoid implementation-dependent
negative shifts. For I8 W, P_j=W*q_j, hence exactly

```
W*q=P0+128*P1+16384*P2.
|P0|,|P1|<=16256*D; |P2|<=256*D.
```

Require16256*D<=INT32_MAX before an I32 dot backend. At sourceD<=4096 this bound
is66,584,576. Reconstruct in I64, with conservative intermediate bound
D*(16256*129+256*16384)=6,291,328*D. The final source integer magnitude is at most
128*32767*4096=17,179,344,896<2^34; conversion to F64 is exact. Original CPU
scaling/cast order then preserves output only after source arithmetic/codec/
compiler contracts have been qualified. Algebra alone is not an executed test.

One matrix multiply can use the three digit vectors as THREE COLUMNS, giving
all three partial outputs at once. If implementation/alignment requires padding,
append declared zero columns and charge them. These are digits of ONE query,
not independent user requests; batch1 acceptance denominators remain unchanged.
Original encoder multiple positions are likewise the same original request.

Proposed implementation on existing RTX3060: keep ONLY eligible compact shared
core/head matrices resident, CPU original quantizer packs digits, integer GEMM
returns I32 partials, CPU I64 reconstructs and performs the ORIGINAL scale/cast.
Experts and routing/LUT remain CPU/RAM. Charge packing, launches, transfers,
padding, synchronization, CPU postprocessing, initialization and storage.
Three digit columns do not imply one physical matrix read or any speed benefit;
measure actual execution. Existing cublas64_12/cudart DLL files were located by
read-only inventory; no usable driver/version/backend qualification follows.

NVIDIA documents GemmEx I8 inputs/I32 outputs with integer compute, and alignment
requirements. This motivates the prospective operator, not a guarantee for our
installed runtime or batch1 shapes.
[Official cuBLAS GemmEx documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex).

The shared GPU footprint would be independent of expert n only for a genuinely
fixed compact core. GPU-offloading a large dense donor with its full active cost
does not conclude pretrained-to-conditional transfer. Wider families and dense
Giga require separate core/function decomposition; no universality claim.

## Exact next action and bounded decision

Propose ONE model-free484 eligibility/budget admission, not a GPU experiment:

1. Freeze complete parser/protocol/runtime/source bindings BEFORE its first
   numerical or GPU control. No old main/audit/capture/timing replay.
2. Read bound original128/256 manifests and every tensor descriptor; classify
   core/control/head, expert pool, router and F32-only organs. Check every
   encoding, dimension, offset, alias, integer-width bound and physical byte
   budget. Distinguish metadata eligibility from payload/driver qualification.
3. Bind374/389 kernels, manifests,458+retention and relevant original arithmetic
   contracts. Check whether another exact backend already exists. Same original
   activation/scaling/state ordering; no precision substitution.
4. Compute real eligible shared bytes, host/GPU scratch/transfer/padding charges
   and a whole-cost requirement. Do NOT attribute measured subcomponent time to
   individual matrices without evidence. Keep the unexplained share explicit.
5. Cost<=60s/256MiB/smallmetadata, ZERO model/solver/GPU calls, no downloads or
   new resources. If eligibility/cost cannot justify a whole candidate, stop.

Only a successful eligibility inquiry permits ONE prospectively frozen complete
integer primitive+C integration: runtime qualification, meaningful extreme/
shape arithmetic controls, whole native parity, and fresh ownstate quality AND
SAME accepted rate with all overhead. No rank/tile/shape sweep or fragment-time
promotion. No kernel experiment has been prepared or executed in this record.

Conditional-capacity track remains coupled: select partitions/features based
on useful conditional FUNCTIONS compatible with the SAME core, retaining all
original cases/IDs/acceptance/mass. Function-weighted or query-dependent loss
is a different proposed objective from the closed global rank/RMS recipes, but
needs its own complete costed protocol and causal capacity evidence. Growing
useful n, exact core operator and efficient normalized retrieval must ultimately
compose on one artifact; none of them alone completes the goal.
