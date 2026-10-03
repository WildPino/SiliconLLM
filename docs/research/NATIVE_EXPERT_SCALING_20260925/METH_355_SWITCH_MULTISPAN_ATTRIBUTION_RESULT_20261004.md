# METH-355: consumed-state attribution PASS, upstream precision motivated

Freeze d1631b1; authoritative exec4703 exit0. Raw SHA
`639308005e6f7740e96b747d74a6b29df401e9b401fbef7212d5362d76e58d10`.
MAIN163.812s/max checked RSS1,894,907,904B. ALL96 teacher and natural-common-
prefix input hashes exact351; ALL192 original F32 head oracles close<=1e-6
with exact choices; ALL192 independent native A16 head oracles BYTE EXACT.
Original shared F32/head I8 code/scale identities freshly checked against
327/338. Whole338 qualified identity reused with size/mtime fixed, no fresh
full-payload hash. Complete counterfactual logits retained with raw hashes.

## Fixed-state observations, not changed whole-model quality

| Head condition | All teacher changes/1344 | Masked teacher changes/768 | Common generated-prefix changes/904 |
| --- | --- | --- | --- |
| Actual native upstream A8/HEAD A16 | 51 | 40 | 32 |
| Original final state/SAME I8 HEAD A16 | 12 | 8 | 13 |
| Native final state/original F32 head | 49 | 40 | 36 |
| Native final state/I8 weight/unquantized activation | 50 | 40 | 32 |

Original final state/SAME target head recovers37 of40 current masked changes
but introduces5, leaving8. Native original F32 head recovers6 but introduces6,
leaving40; unquantized head activation leaves40. At natural first divergence,
32cases, original final state/SAME target head matches original in23; native
state/original F32 head matches4; native unquantized activation matches0.
Generated comparison ends INCLUDING first changed output choice, where input
histories are still the same; NEVER compares later different histories.
Encoder state and earlier choices can already differ despite common token IDs.
Decoder route choices differ579/8064 all teacher,243/4608 masked positions,
345/5424 common generated-prefix routes. These are descriptive changes,
not proof that every different route is harmful or that routing alone causes
all output differences. Array counterfactuals are conditional interventions
at the head, not equivalent to propagating changed activations through layers.

## Decision

A further head-only precision change is not supported by this diagnostic.
Upstream state error has a much larger conditional contribution than the
remaining HEAD A16 quantization. Next single execution variable: all I8
matrix inputs use A16 activation codes (core and experts as well as head),
SAME338 I8 weights/scales/F32 controls/router/lookup, SAME cache/attention/
capacity/greedy semantics. Reuse qualified exact I8xI16/I64 primitive, extend
independent full numerical reference to every quantized projection, then
actual CPU cost and NEW original-primary prediction/generation/known-task
criteria. This tests activation error; it does not establish that upstream
weight quantization can be ignored. If it fails, preserve before next variable.
351 masked-only/prose fidelity failures remain,349 scoped prediction PASS
unchanged.354 original rate draft ineligible/unexecuted. No accepted-rate,
useful large-n/LUT/DRAM/family/~100B result. Default engine body exact.
Reproduction: [355 protocol](METH_355_SWITCH_MULTISPAN_ATTRIBUTION_PROTOCOL_20261004.md).
