# METH-210: Q8 embedding/head attribution and priced exact-head screen

METH-194 Q8+centered-E1280 nearly preserves BPB but misses pooled/code
donor-top-1 retention by 0.113/0.229 points. METH-188's Q6 attribution
cannot automatically locate this Q8 interaction. METH-128/129 already
provide an approximate-head shortlist and exact selected-row primitive.
Test whether exact embedding/head restore the Q8 development gates,
and whether a proposal with FP16 row scales preserves K=64 inclusion
and exact BF16 row choice on the resulting states.

Bind the exact stored METH-193 core, METH-194 result, donor, centered
E1280 checkpoints and viewed METH-121 manifest. Reconcile BF16+E1280
and full tied-R8 Q8+E1280 per-source document NLL and prompt match
counts exactly. Freeze five arms: BF16+E1280; Q8 FFNs with tied R8;
Q8 FFNs/exact embedding/R8 head; Q8 FFNs/R8 embedding/exact head;
Q8 FFNs/exact tied embedding/head. Changing embedding and head
independently requires explicit untied parameter objects; verify
their pointers and restore tying in the selected exact-tied arm.
Attention, controls and E1280 factors remain unchanged.

The selected candidate is fixed to exact tied embedding/head, not
chosen among the diagnostic arms. Retain all METH-194 quality gates:
pooled/category BPB increase <=.01/.02 and donor prompt top-1 loss
<=1/2 percentage points versus BF16+E1280. Then on every candidate
prompt hidden state use the original stored R8 codes and their scales
rounded once to FP16 as proposal. K=64 must omit zero full-BF16 top-1
tokens and exact BF16 row recomputation with lowest-ID tie breaking
must have zero mismatches. Reuse the METH-128 shortlist evaluator;
report row-score reduction differences, not just gathered-score parity.

Proposed E1280 greedy active ledger: original 559,981,568 bytes minus
303,872 FP32-to-FP16 proposal-scale bytes, plus 114,688 selected BF16
head-row bytes and conservatively 1,792 exact embedding lookup bytes
=559,794,176 bytes/token. Require <=560 MB. Full BF16 head storage
adds 272,269,312 bytes; duplicate RAM capacity is explicit. This is
an arithmetic proposal, not a new artifact or measured DRAM traffic.
It does not include a future third-level E12800 router by assumption.

Every likelihood is scored with a full head for that diagnostic arm;
the K=64 choice experiment is a separate finite-state greedy screen.
Passing licenses a stored exporter and native composition experiment,
not fresh quality, probability parity for a mixed-logit implementation,
universal shortlist correctness or >=50 accepted tok/s. No new model
quality sources are consumed in this attribution experiment.

Local RTX 3060, six host threads, <=15 minutes, <=20 GiB RSS,
<=10.5 GiB allocated GPU, <1 GB result. Stop on bindings, baseline
reconciliation, nonfinite states or budgets; preserve failure record.
No T4 or training. Runner:
`benchmarks/native_expert_scaling/meth210_q8_exact_head_diagnostic.py`.
