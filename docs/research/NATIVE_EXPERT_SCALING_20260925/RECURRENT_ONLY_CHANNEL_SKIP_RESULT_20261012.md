# Recurrent-only384 channel projection with full source skip

12 October2026. **COMPLETE / INDEPENDENT_STORED_PASS / LOCAL QUALITY FAIL.** Goal ACTIVE/INCOMPLETE.

## Question and controlled variable

Source ungated SSM output is `y=y_rec+D*x`. Previously the frozen384-channel
scans reconstructed BOTH terms through each head's `V W^T`. Keeping full skip
requires no additional SSD recurrent-state coordinates:

```
y_old = V ym
y_new = V(ym-D W^T x)+D*x = y_old+D*(x-V W^T x)
```

Within each source head, common scalar decay permits fixed channel projection
to commute with recurrence. Reuse both frozen fixed8/adaptive384 maps and actual
latents;all48 source histories, sites0/12/23. No new fit/scan/model forward.
Full source nonlinear generators and gates are retained. `x` is post-conv4,
so stateless skip GIVEN x does not remove the causal convolution buffers or
source feature-generation cost. Channel-state memory alone is 393,216B versus
source3,145,728B per site; this is not total model memory or native cost.

Both comparison arms use controlled F64 arithmetic with BF16 normalized-input
and output rounding. The denominator is recomputed jointly from the COMBINED
gated output;source-denominator mode is a separate oracle. Original F32 output
metrics are repeated as controls. No precision equivalence or native export
claim from the controlled comparison.

## Observed local DEV results

Equal-case mean output relative RMS, with reconstructed denominator:

| Site | fixed8: projected skip | fixed8: full skip | adaptive384: projected skip | adaptive384: full skip |
|---|---:|---:|---:|---:|
| 0 | .571221 | .290536 | .568896 | .286465 |
| 12 | .841039 | .799232 | .832677 | .791990 |
| 23 | .557974 | .304151 | .557106 | .303801 |

All24 DEV cases improve in every site/layout/norm-mode combination. Both layouts
and both modes still fail original local budgets:mean<=.1,worstcase<=.2,worst
domainmean<=.2,centeredmean<=.1. No donor-relative chatbot-quality admission.
Full FIT/DEV/domain/label metrics and gates are in the result JSON.

Error decomposition BEFORE gating/normalization:
`e_old=e_rec+e_skip`; `||e_old||²=||e_rec||²+||e_skip||²+2<e_rec,e_skip>`.
Every case's raw energies/cross term are retained, not inferred from RMS ratios.
DEV recurrent/skip inner product is negative on23/24 fixed8 and24/24 adaptive
histories at12, and24/24 both layouts at23. At12 adaptive ungated RMS barely
changes .577109->.576486 despite substantial skip error (.363601 relative to
source y);mean error-vector cosine is -.275125. Cancellation matters.
This identifies a remaining recurrent representation problem under THESE bases;
it is not a384-channel/nonlinear/conditional capacity lower bound.

## Precision failure and continuation

Original V1 was never run:prelaunch dtype inspection selected explicitF32
coefficient casts for the old precision control in V2. V2 closed1 at firstcase:
old control scalar delta0, F64/BF16-v-F32/BF16 output RMS .00198256 exceeds .001.
No V2 arrays saved;first fault retained. V3 keeps original threshold and records
all flags instead of terminating before complete diagnostic evidence.252/576
precision bridge flags fail;max .00336916. Every old F32 metric reproduces the
prior report exactly (maxdelta0). No precision gate is retrospectively passed.

V3 F64 comparison isolates skip treatment within one precision regime, while
old-F32 equivalence remains failed. Numerical audit checks all new stored outputs
independently;GPU precision bridge magnitudes remain frozen-code observations,
not exact CUDA replay. See [preflight](RECURRENT_ONLY_CHANNEL_SKIP_PREFLIGHT_20261012.md)
and [failure diagnosis](RECURRENT_ONLY_CHANNEL_SKIP_PRECISION_DIAGNOSIS_20261012.md).

## Audit, cost and custody

144 case/site records,67,641 history/site positions,1,108,230,144 BF16 output words,
eight arrays per record;total new payload2,216,841,312B including NPZ metadata.
Independent CPU NumPy audit uses additive skip correction rather than producer
subtract-then-decode, stable SiLU, integer BF16 RNE and blocked full source out
projection. Full arrays/energies/cross terms/scalars/label metrics/aggregates/
gates and all before/after hashes are checked, not sampled.

Independent audit PASS:all1,108,230,144 output words checked, relative output
difference0;max metric delta1.42109e-14,energy relative4.42150e-15. All input/output
hashes before/after exact. This does not qualify the failed old-F32 bridge.
Audit292.91928s,OSpeak720,818,176B;all owned sessions closed, no owned run live.

V3 terminal184.58247s,OSpeak1,348,706,304B,CUDA allocated647,827,968B/reserved
757,071,872B. V2 terminal18.87566s,OSpeak1,252,741,120B,CUDA allocated343,034,880B/
reserved429,916,160B. Timing admission false (desktop sharing/Windows clock);
no native speed measurement. Preparation V1/V2/V3:73.556/6.923/6.596s.
Original producer freeze0d72234;V2d950b7d;V30cb4d90;V3binding SHA256
`6808d0b47090a78d2e2e0dbca9527b44814da426b0ca0fa74a7501af5bc01029`.
Final independent audit source freezec1f1a04,launch6cc1242.

- [Producer V3](../../../benchmarks/native_expert_scaling/recurrent_only_channel_skip_v3.py), [protocol](RECURRENT_ONLY_CHANNEL_SKIP_PROTOCOL_20261012.md), [binding](recurrent_only_channel_skip_v3_binding_20261012.json), [launch](recurrent_only_channel_skip_v3_launch_20261012.json)
- [Result](recurrent_only_channel_skip_v3_result_20261012.json), [terminal](recurrent_only_channel_skip_v3_result_20261012.terminal.json)
- [Audit](../../../benchmarks/native_expert_scaling/recurrent_only_channel_skip_audit.py), [audit protocol](RECURRENT_ONLY_CHANNEL_SKIP_AUDIT_PROTOCOL_20261012.md), [audit launch](recurrent_only_channel_skip_audit_launch_20261012.json)
- [Adjudication](recurrent_only_channel_skip_adjudication_20261012.json), [audit terminal](recurrent_only_channel_skip_adjudication_20261012.terminal.json), [closure](recurrent_only_channel_skip_session_closure_20261012.json)

## Decision

Full skip materially helps0/23, but leaves severe local failure at12. Next fit
channel bases on recurrent-only `y-D*x`, using FIT only, fullskip retained:
[selected next](RECURRENT_OUTPUT_BASES_NEXT_20261012.md). Old bases were fitted on
the entire y;re-evaluating them alone does not test recurrent-only basis capacity.
Exact source generator lifts/state96 tools/moving decoder moments remain useful,
but require explicit composition/native/own-history qualification.
No new chatbot, conditional bank, n/DRAM/routing or same useful artifact>=50
admission. Goal remains active and incomplete.
