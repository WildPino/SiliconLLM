# Complete compact recurrent/ternary learner: actual pilot result

8 October 2026. Goal ACTIVE/INCOMPLETE. **PILOT_RECOVERY_PASS**, independently
saved-only verified. This admits a numerical/memory prerequisite, not chatbot
quality, native C parity, useful large n or accepted50.

## New procedure made real

[Target](../../../benchmarks/native_expert_scaling/chatbot_hybrid_target.py) and
[pilot](../../../benchmarks/native_expert_scaling/chatbot_hybrid_pilot.py) implement
ONE complete source-informed target: D512/L12,10 SSM768/48heads*16/N256/conv4/
gate-before-RMS, two4head*128 SWA/window128, untied V65537 head/embedding,
72 source-derived expert groups per block/top8/h128, signed ternary SwiGLU.
Master weights/scales/router/core/norms/head all trainable. Actual AQ63 and
ternary integer-dot forward, STE gradients, learned row scales and stable
flat-router IDs/selected normalized mass. No unquantized expert inference path.

Source Falcon-H1-1.5B80ebc50d7799a440b96c93bb6686a3924a09b0cb reused; no download,
16-case selection, Tiny packet/C or Qwen fit repeated. NEW6 calibration cases,
32 donor-generated positions (FIT20/DEV12), full65,537-vocabulary scores. Source
original template/IDs and generation contract retained. Only four cases optimized,
fixed order twice/eight AdamW updates/lr5e-5/global clip1; two DEV cases unoptimized.

Same source FFN feature-slot count is NOT conserved capacity. Shared2048->512
embedding/head basis retains .4719909727573395 of the balanced normalized Gram
trace; max P^T P-I error1.0132789611816406e-6. This is a weight-energy statistic,
not information/quality retention. Width/depth/parallel branches/projection,
approximate norm scaling, ternary coefficients and active selection all need
learned recovery. The target is currently family/shape-specific.

## Full-output result, independently recalculated from saved logits

| Scope | Positions | Initial KL F64 | Final KL F64 | Initial/final greedy disagreements |
|---|---:|---:|---:|---:|
| FIT, four cases |20|27.80247800804052|5.94090893670987|20/20 ->10/20|
| DEV, two cases |12|27.614440992379073|26.51796746991217|12/12 ->12/12|

All fixed pilot gates pass: finite eight full updates/positive core, expert and
norm gradients in ALL12 layers; actual readout change; FIT KL falls; DEV KL is
within1.02*initial. ALL211 parameter tensors actually changed, including embedding,
head and every block core/banks. All254,932,736 parameters finite F32; dense Adam
has2,039,461,888 actual moment bytes, correct shapes/finite/nonnegative second
moments and ALL step8. Learned row scales remain>=1e-8.

DEV rewrite worsens26.672386->26.905113; supplied-history case improves
29.498562->25.743687. Two unoptimized cases/12 positions do not establish broad
generalization, student own-history or retained chatbot usefulness. No absolute
quality admission: DEV greedy disagreement remains100%. This closes the bounded
eight-update pilot; it does not authorize an unbudgeted extension on these cases.

Independent [saved-only audit](chatbot_hybrid_pilot_audit_result_20261008.json)
recalculates ALL64 initial/final complete vocabulary rows with NumPy F64,
max discrepancy from Torch F32 KL6.687997050391914e-6<fixed1e-4, ALL source greedy
IDs/disagreement counts and fixed decisions agree. Initial/final checkpoint,
all eight immutable update metrics and optimizer state checks pass.0 new model
forwards/source calls/updates. This verifies saved numerical/state consistency,
not independent training replication or native arithmetic.

## Actual cost, storage and architectural accounting

- Repair1 worker190.890s/family200.125s; GPU allocated peak4,449,674,752B,
  reserved4,724,883,456B. Worker Windows peak through actual exit8,242,982,912B,
  launcher snapshot27,156,480B. All predeclared resource caps pass.
- Eight update seconds8.688..11.375 INCLUDE forward/backward/full gradient and
  parameter checks/optimizer; EXCLUDE subsequent immutable record write and
  completed CPU recovery snapshot copy (event spacing includes them/guards). These
  are this short-prefix F32/local3060 recipe, not T4/long-context throughput.
- Actual254,932,736 trainable parameters; F32/gradient/Adam array floor
  4,078,923,776B. Complete final learner+Adam3,059,440,986B, SHA
  `75b2317efe99fe66fc16f2b0e6df1f5001ef8b9c243c150b87b24e1f433793d9`.
  Initial component checkpoints/basis/output logit vectors retained; repair1
  output files4,100,236,214B, source score bytes remain in original namespace.
- Source matrix products1,420,036,096/token; target69,632,512/token INCLUDING
  flat72-score router, versus prior unimplemented tree proposal69,435,904.
  Scalars/SSM state/context cost excluded; packed model425,188,352B is still a budget,
  not an actual native export or physical DRAM measurement. No measured token/s.
- Audit11.359s worker/22.609s family, actual exit0, worker OS through-exit
  3,866,517,504B. Scope CPU saved-only, no GPU/kernel performance claim.

## Fault preservation, freezes, bindings and commands

Original main freeze b7a47e15e7979f6a2a06421da60d202fb3bce99f, binding
f484251254db3c0f31eb9157f6a62c1d414971a6eb1b1cfac9c8147f30c9c280:
ALL6 donor packets saved ONCE, then target config tried assigning read-only
layer_types/layers_block_type, before basis/target forward. [Repair1](CHATBOT_HYBRID_PILOT_REPAIR1_20261008.md)
changes access only, same scientific recipe. Original first_failure/launcher
failure/log retained, family117.063s/worker OS through-exit3,740,237,824B/exit1.
No target observation or update was repeated. Repair1 materializes donor weights
for initialization and reuses original ALL6 score packets,0 new source generations.

Repair1 freeze44154bcd0165da8daf5a485e71c12f74e60cd819,
[binding](chatbot_hybrid_pilot_binding_repair1_20261008.json) SHA
7fc91cc634496d215f6fbc9eeaf1bc0f3ddd8d6d383c42c16e75148d0e8e2318 (48 inputs),
[actual result](chatbot_hybrid_pilot_result_repair1_20261008.json),
[terminal/complete output extents](chatbot_hybrid_pilot_result_repair1_20261008.terminal.json).
Audit freeze0c2d4cbc81e51b581b0611cba9a2a52d6763f3d8,
[binding](chatbot_hybrid_pilot_audit_binding_20261008.json) SHA
a8d3f078c94338e176c0f0d08be2f8a1d8b96cf3843bdc6332de4e9c8949b177 (54 inputs),
[terminal](chatbot_hybrid_pilot_audit_result_20261008.terminal.json).
Launcher verifies input bytes before/after and holds Windows worker handle
through exit. Its later audit schema addition is a new frozen revision after
pilot closure; original executable Python sources remain reconstructible in Git.
Runtime bindings are selected files/version/path checks, NOT full DLL-tree hashes.

From repo root, isolated Python312, all required source/runtime/retained extents
available. Commands are provenance, **do not replay completed observations**:

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/chatbot_falcon_usability_launch.py --binding docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_hybrid_pilot_binding_repair1_20261008.json --binding-sha 7fc91cc634496d215f6fbc9eeaf1bc0f3ddd8d6d383c42c16e75148d0e8e2318 --freeze 44154bcd0165da8daf5a485e71c12f74e60cd819 --directory results/native_expert_scaling/chatbot_hybrid_pilot_repair1_20261008 --out docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_hybrid_pilot_result_repair1_20261008.json
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/chatbot_falcon_usability_launch.py --binding docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_hybrid_pilot_audit_binding_20261008.json --binding-sha a8d3f078c94338e176c0f0d08be2f8a1d8b96cf3843bdc6332de4e9c8949b177 --freeze 0c2d4cbc81e51b581b0611cba9a2a52d6763f3d8 --directory results/native_expert_scaling/chatbot_hybrid_pilot_audit_20261008 --out docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_hybrid_pilot_audit_result_20261008.json
```

## Decision and exact continuation

The complete whole target learner is AVAILABLE, actual finite learning/memory
and short-case FIT recovery verified. Initial loading/partitioning alone fails
to retain behavior; broader transfer remains research. Keep original compact
SSM/SWA/ternary/LUT deployment direction. **Next** [packed-only/native continuation](CHATBOT_HYBRID_NATIVE_NEXT_20261008.md)
using this actual checkpoint and final saved logits, before a long adaptation.
Broader balanced calibration/long contexts/own-prefix supervision and separately
excluded fresh dialogue/task criteria then govern sustained recovery. No repeated
pilot selection data, generic donor-runtime detour or isolated1% prerequisite.

All current workers terminal. Three foreign tracked SHA verified unchanged.
No T4/long fit started. C projection/conv/gated RMS/per-head scan/SWA/ternary
SwiGLU LUT/32-bit IDs, packed-only loader, structured CPU LUT winner/mass,
useful-n/RAM, physical DRAM, fresh whole quality+accepted50 SAME artifact and
actual other families/~10B/~100B remain required. Goal not complete.
