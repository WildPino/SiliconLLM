# METH-79: removing balance improves retention but fails the E1280 gate

**Decision.** The precommitted E1280 joint gate fails. Stopping the
product-key axis-balance term after update 64 improves fresh-prompt
donor top-1 versus the matched METH-75 balanced continuation, but
the improvement is below the required 0.5 point and the E1280 arm
still declines more than one point from its own update-64 baseline.
Broad expert selection remains; the failure is not explained by a
simple collapse in measured route coverage. Do not promote this
checkpoint to a new external, semantic or native gate.

The [protocol](METH_79_BALANCE_CESSATION_PROTOCOL_20260927.md),
[manifest](meth79_balance_cessation_dev_manifest.json) and
[runner](../../../benchmarks/donor_adaptation/s1/meth79_balance_cessation_continuation.py)
were committed at `f0e708b` before either arm. The manifest SHA-256 is
`0bbbedcbd0c15f9d2a8def3c2bad4a9e03113b874afcc3717bffc69551dacf70`.
Its 24 rows exclude all earlier development sets through METH-75 and
all raw training draws through update 256. The METH-71 update-64
checkpoint, optimizer state and RNG stream were restored per arm.
The only objective change against METH-75 is balance weight 0 rather
than 0.02 at updates 65–256. The runner checks each actual training
draw against the frozen stream. METH-75 update-256 comparators were
scored on the same new prompts before this continuation trained.

| Frozen new-prompt readout | E128 | E1280 |
|---|---:|---:|
| Update-64 parent top-1 | 3,719/3,812 = 97.560% | 3,760/3,812 = 98.636% |
| METH-75 balanced update-256 top-1 | 3,695/3,812 = 96.931% | 3,687/3,812 = 96.721% |
| METH-79 no-balance update-256 top-1 | 3,712/3,812 = 97.377% | 3,701/3,812 = 97.088% |
| No-balance versus balanced | +17 positions = +0.446 point | +14 positions = +0.367 point; **fails ≥0.5** |
| No-balance decline from update 64 | −7 positions = −0.184 point; pass | −59 positions = −1.548 points; **fails ≤1.0** |
| Minimum selected experts/layer | 128/128 | 1,088/1,280 |
| Worst maximum/mean route load | 3.84× | 19.31× |
| Minimum changed B slots/layer | 128/128 | 1,277/1,280 |
| Raw held-out ΔBPB versus donor | −0.009931 | −0.009384 |

Both cessation arms clear the absolute 95% top-1, raw BPB, changed-slot
and route-health screens. E1280 is 0.289 point below the matched E128
cessation arm, within the precommitted 0.5-point cross-E allowance.
The E128 arm passes its individual development gate. These observations
show a modest retention benefit from removing the balance term under
this exact paired run; they do not establish that balance was the only
source of donor drift. CE/KL optimization continues to move factors
and routes. No new pair-permutation utility or semantic result was
measured, and the earlier METH-72/74 failures are unchanged.

The [E128 raw result](meth79_e128_result.json) was committed at
`dcdc652` before E1280 ran; the [E1280 raw result](meth79_e1280_result.json)
at `a6a3a2b`. They include all 192 updates' microbatch losses,
gradients, draw checks, per-prompt counts and per-layer load.
The E128/E1280 terminal checkpoint SHA-256 values are
`3b7cb161e306236c0d8ce54015bba51b2befe6415a57887f25bcbf89301a267c`
and
`ed55ea5cebc18233607fea8bf37c00bda5e7a97f2603be08e126266aaf7e5dc4`.
Runtime was 383.2/502.1 seconds, with GPU peak 2.581/10.231 GB and
RSS 2.875/4.683 GB. Both remained within the local budgets; no T4
was used.

The next training rule needs an explicit donor-retention constraint
that remains effective as factor utility grows. A new quality study
must use fresh development prompts and a separately frozen external
audit; it cannot tune against the viewed METH-79 prompts or infer
large-E success from route coverage alone. The CPU component results
in METH-76/77 remain valid for their measured banks, but the
pretrained-to-native full-model throughput target is still open.
