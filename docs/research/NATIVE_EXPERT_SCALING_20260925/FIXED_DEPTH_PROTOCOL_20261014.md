# Structural active-depth intervention

14 October 2026. PREPARED / QUALITY NOT MEASURED. This is the next substantive
variable after the [qualified generator cost](CONTEXTUAL_GENERATOR_COST_RESULT_20261014.md).
Actual23 BF generators cost52.1574/50.42935ms, alone far above the3.18275/
2.97615ms recurrent room. Further minor scalar/layout changes cannot close
this observed baseline. This does not bound every alternate representation.

Change ONE structural variable: number of executed source decoder blocks.
Preserve residual2048, norms/FFN/attention/scalars of each kept source block,
source embedding/head and actual native state96/I16 readout. Use the qualified
activation tables. Omitted whole blocks are identity, without generator, norm,
FFN, attention KV or native state updates. No gain is credited to source
operators still executed on GPU. Kept blocks use NEW own-history hidden,
never stored teacher hidden/preconv/state. The first and last sites are kept.

Frozen evenly spaced schedules, integer half-up rounding of23*i/(count-1):

| Active blocks | Original source sites |
| --- | --- |
| 3 | 0,12,23 |
| 6 | 0,5,9,14,18,23 |
| 12 | 0,2,4,6,8,10,13,15,17,19,21,23 |
| 24 control | All original sites |

Source functions at omitted sites remain available in the checkpoint but are
not causally used. Thus fixed pruning is a structural diagnostic, not useful
expert-n evidence, token-varying routing or complete capacity transfer. No
reactivation of a dormant stateful block is permitted. A future selective bank
requires its own causal state/scheduling contract and utility evidence.

Implementation: benchmarks/native_expert_scaling/fixed_depth_bridge.py,
FixedDepthBridge. It uses qualified C parameters/table owners for kept sites,
fresh contexts per sequence, original attention site indexes and actual KV
ownership. Every step invokes24 wrapper slots; only scheduled blocks execute
their operators. Snapshot/log ledger records active/omitted IDs, masses1/0,
seen/last/state/ring, retained parameter bytes and active attention digests.
restore() restores both source block and Mamba methods before another arm.
Meta-device/source-class mechanical probes are not quality observations.

## Frozen quality decision, required next worker

Before any quality observation, freeze a complete bounded worker/auditor and
new actual GPU/runtime/source binding. Old NVIDIA pins do not prove current
GPU continuity. Announce GPU allocation, elapsed-time cap and stop conditions
before use. Run one owned bounded process at a time; do not collide with CPU
timing or user applications. No GPU resources are allocated by this protocol.

1. Mechanical controls: actual source classes/cache interfaces, omitted block
   identity bit-exact and no submodule invocation; retained C seen/ring/state;
   method restoration;24-arm controls against existing qualified bridge on
   identical arithmetic/device. Cached/prefill/step prefixes must match per arm.
2. Source-only parameter/table export provenance and all411 donor parameters
   values/identities/versions before/after; retained operators and multipliers
   unchanged. Read-only native parameters own their lifetime. Fresh input
   captures, no precomputed source h feeding kept downstream blocks.
3. Score all24 FIT known histories for EACH3/6/12/24 arm, using the original
   label positions/teacher distributions. No FIT record or domain exclusion.
   Mean case KL<=1, disagreement<=.20, each domain KL<=2/disagreement<=.35.
   Choose the lowest active count among3/6/12 passing ALL FIT criteria;
   24 is only the control. No selection from DEV, task answers or timings.
4. Run all24 known DEV histories and16 canonical own-generation tasks, plus
   own-answer followup/prefix controls, on the FIT-selected arm. If none
   passes FIT, run6 as an explicitly NONSELECTED defect diagnostic, not an
   admitted replacement. This fallback is frozen now and never promoted
   from a DEV-only pass. Keep failures and complete full numeric audit.
5. Whole quality gates unchanged:>=12/16,each category>=2,blank<=1,valid stop,
   no special leaks, own followup correct; DEV meanKL<=1/disagreement<=.20,
   each domain<=2/.35,all16 own-prefix controls exact,mass defect<=1e-6,
   counters and coverage exact. Local generator1% is a numeric control for
   unchanged kept generators, not a threshold on identity-pruned block output.
6. Independent full replay of every kept native trajectory/state/intermediate
   and integer/MuP controls; omitted traces/KV/state absent, original source
   IDs retained. All stored logits/quality and FIT choice recomputed, source
   params/inputs/output hashes/receipts/worker identities sealed. Source GPU
   FFN/attention arithmetic independence limits remain explicit.

RESERVED104+64 and longalign stay unopened. This known-corpus intervention
cannot prove held-out knowledge conservation. A pass permits a physical CPU
active-depth artifact and recovery/conditional-bank study; a fail motivates
joint function-preserving merge/adaptation rather than blind arbitrary pruning.
No claim that omitting18 blocks preserves the original donor's unused capacity.

Planning cap per separately bound phase:45min,OS8GiB,GPU own allocated6GiB/
reserved10GiB,E32GiB, CPU6 threads. The numerical binding must compute exact
sequence/output bounds and free-space/free-GPU margins before allocation.
If a chosen owned run will last hours, follow the user's announce/launch/pause
instruction. This document neither launches a run nor fabricates a result.

Final same-artifact CPU chatbot quality/useful>=50TPS, useful RAM-driven n,
physical DRAM, multiple donor families/scales and conversion training costs
remain open. Depth pruning alone cannot complete those requirements.
