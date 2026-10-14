# Fixed depth six: complete stored numerical audit, quality rejected

14 October 2026. Goal INCOMPLETE. Producer worker9672 and holder24048,
CPU auditor27444 and closer31144 all exited0. Original V2 producer posthash
completed; no recovery or successful numerical replay was needed.

All24 FIT histories and16 prefix pairs were audited:56 sequences/336 kept-site
sequences,72,972 native positions,149,446,656 BF output words and the same
number of integer readout words. Every stored native output, state, ring and
integer result is exact. Independent F64 controls pass: maximum contextual
generator relative error3.80980e-5, B/C1.34341e-7/1.41453e-7,
scan6.25484e-7, state1.16705e-6; fields and normalization exact.
Auditor480.375s/OS1,146,118,144B/no GPU; closer25.063s/OS37,101,568B.
Full before/after input and output seals and source411 parameter checks pass.
Actual reduced-prefix KV digest values are retained in the producer journals.

Audited FIT KL14.682199397323998 and disagreement98.94407711216799% fail
all four frozen pooled/domain gates. This rejects untrained identity omission
with kept sites0/5/9/14/18/23. It does not reject every adapted depth reduction.
This phase uses original BF contextual generators; the separately screened
I8/I4 packs were not composed into this candidate. Source GPU attention,
FFN and head are witnesses, not independently replayed GPU arithmetic.
No DEV, RESERVED, changed threshold or accepted end-to-end rate admission.

Producer4322.468s/OS4,034,850,816B, own GPU peak3,159,225,344B allocated/
3,225,419,776B reserved. GPU phase713.453s, ownership released to32MiB
before posthash. All12,162 processed tokens and2,936,687,869B output retained.
The longer total includes CPU preparation, custody and I/O; GPU phase is
measured separately. No extrapolated speed claim follows from these durations.

Authoritative artifacts:

- [Producer](fixed_depth_fit6_result_20261014.json), SHA256
  `492a06ab34b716005696c1684bc9ced5e65acff3771db9a0d6d20c283b0bd74f`.
- [Full audit](fixed_depth_fit6_audit_20261014.json), SHA256
  `09d3064de8d70853b8cb919af436e0a10f7177b6dbd152395272fb905634542d`.
- [Closure](fixed_depth_fit6_session_closure_20261014.json), SHA256
  `8c07a88f7a4f612ed43b9771b08d3ba9b985feafeee5ab9f1e6ee4ca6d0affb0`.
- [Audit exit0 receipt](fixed_depth_fit6_audit_20261014.terminal.json) and
  [closure exit0 receipt](fixed_depth_fit6_session_closure_20261014.terminal.json).

Next run only the unexecuted frozen FIT12 arm, then its full stored audit even
if quality fails. FIT-only selection requires the complete3/6/12 family.
No promotion or evaluation may be inferred from this per-phase closure.
