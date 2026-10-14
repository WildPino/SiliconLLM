# Recovered waiting comparison: complete ledger, partial budget fails

14 October2026. Recovery22492/session23997 exit0/51.062s/OS42,532,864B.
Original parent21820 remains exit1 at the0ms parser; P116264 exit0 was reused,
only A33932/P24492 executed after the repair, both exit0/18.297s/16.984s,
OS peaks1,598,046,208B/1,598,115,840B. No GPU, source forward or fitting.

All1152 measured groups and their orders/counts/times/settings/identities
pass the full stored ledger. All warm/measured outputs are checked by frozen
workers against prior archives:566,193,024 generator cells and180,486,144
integer/F64/BF readout cells across P/A/P. Actual runtime blocktime0/200/0ms
and three physical-worker masks verified. Current recovery input/output seal
exact; this does not restore missing original successful final posthash.

Median milliseconds for23 generators plus24 readouts in layer order:

| Encoding | ACTIVE resident | ACTIVE evicted | P1/P2 resident drift | P1/P2 evicted drift |
|---|---:|---:|---:|---:|
|BF16|33.08590|30.50780|1.212797 FAIL|1.063859 PASS|
|I8|29.38405|27.06070|1.208305 FAIL|1.078495 PASS|
|I4-g128|24.12520|22.44160|1.209423 FAIL|1.103444 FAIL|

Every active median exceeds20ms before convolution/state/norm/input
quantization/FFN/attention/head/router/frontend. Four of six passive drift
checks exceed the unchanged1.10 threshold. The P1-to-A interruption is
1211.725s from recorded worker creation/lifetime metadata; original failure
and gap are explicit. No uninterrupted runtime recipe or isolated causal
speedup is admitted. Keep all tails/first processes, no faster rerun selected.

This closes the scheduling/weight-codec-only full-depth route under the tested
profiles. Smaller coefficient storage and active waiting have not supplied a
credible complete50-token artifact. Do not spend another unchanged worker/
wait-policy/group-size grid. The next substantive uncertainty is composed
quality versus active geometry: run only the previously frozen unexecuted
FIT6/12 arms with source width/operators unchanged, complete full numerical
audits, and preserve failures. These are geometry diagnostics, not useful-n
or transfer-through-pruning claims. A failed reduced model requires a changed
source-derived/adapted conditional mechanism before deployment, not relaxed gates.

[Recovered report](encoded_organs_wait_recovered_20261014_result.json),
[recovery closure](encoded_organs_wait_recovered_20261014_closure.json),
[pipeline](encoded_organs_wait_recovered_20261014_pipeline.json),
[exit0 receipt](encoded_organs_wait_recovered_20261014_pipeline.terminal.json),
[original failed parent](encoded_organs_wait_20261014_pipeline.terminal.json).
Full own-model quality, CPU useful50, conditional capacity/IDs+masses/DRAM,
multiple families/10B/100B and sustainable conversion remain incomplete.
