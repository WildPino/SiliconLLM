# Balanced joint replay: execution and stored audit pass, generalization fails

15 October2026. Goal INCOMPLETE. Worker26392/holder24160 exit0 after all512
updates and the fixed-final80-case evaluation. Actual final candidate is
Adam830,3,059,861,270B SHA256
3dc2880deac5fed70dc2b2a5d3c7feea4fe6e9ad52a656449309d4e9785eb826.

| Cohort | Producer case KL | Label disagreement | Frozen absolute gates |
| --- | ---: | ---: | --- |
| Old DEV32 | 2.56407253 | 35.99275% | FAIL |
| New DEV24 | 7.11347743 | 81.48655% | FAIL |
| New FIT24 | 0.70077109 | 16.41791% | PASS |

Previous broad24 had new FIT KL4.47428/new DEV6.99079 and old DEV KL3.14427/
disagreement50.408%. Longer balanced replay learns the repeated FIT and
improves old retention,but the new DEV does not improve sufficiently;its KL
is slightly worse while argmax disagreement improves. These observables can
move differently. This is strong finite evidence of a coverage/generalization
gap,not a proof of a representation ceiling or a universal training diagnosis.
No more unchanged fitting on these152 records is the next action.

CPU auditor8872 exited0/207.546s/OS3,668,103,168B/no GPU. All80 cases/9,911
label positions/649,537,207 candidate logit words audited against cached BF
teacher. F64 per-label KL maximum difference1.101036e-5,below frozen1e-4;
every argmax/disagreement integer exact,and all domain/cohort decisions match.
All211 actual final model tensors/254,932,736 F32 words and422 Adam moments/
509,865,472 words finite and shape-correct,all steps830,row scales valid,
CPU/CUDA RNG schema,512 exclusive update files/checkpoint ledger/order/exposure
and resource/current complete before/after seals pass. The two initial runtime
controls match original saved F32 outputs word-for-word.

Route counts match all recorded training prefix tokens and top8 per site.
Sites0..10 consult all72 banks at least once;site11 consults65/72. Counts and
distinct stored tensors do not establish causal expert usefulness. Producer
in-memory roundtrip and GPU training/forward arithmetic are witnesses;the
audit does not independently replay gradients,updates or final model forwards.
No native,own-generation,fresh-heldout,DRAM,useful-n or accepted50 admission.

Worker7328.828s (~2h02m),GPU phase7257.610s,OS5,221,330,944B,GPU allocated
4,730,072,064B/reserved5,291,114,496B;holder7365.719s. All within frozen caps;
GPU released before posthash. Recorded per-update wall intervals before
snapshots total4475.615s. Remaining2853.119s includes initialization,controls,
checkpoints,evaluation,checks and hashes;it is not separately measured I/O.
No snapshot-only timings or hardware write counters were recorded. The32
scheduled durable saves imply at least97,894,170,624B raw master/moment tensor
payload serialized by the executed schedule,a deduction rather than physical
write traffic. Dense optimizer cost remains proportional to total parameters.

Authoritative artifacts:

- [Producer](chatbot_balanced_replay_result_20261015.json),SHA256
  bc67bd00da6b78f2ea7a48f99c8d13872d2fc73c6d7fa9266db6b934e67d8016.
- [Full stored audit](chatbot_balanced_replay_audit_20261015.json),SHA256
  7ed8ffa983ad95866f18f4fdc1f3e034e7a40c58ca6148099e6d61af792e7771.
- [Closure](chatbot_balanced_replay_closure_20261015.json),SHA256
  f9d723dfc3d425fd818be9855a7d007e226b58598f31beab84ff6b140d2d2a73.

The already adopted public corpus contains360 additional eligible FIT and72
unused DEV cases at the original split/group boundaries. [Source coverage
protocol](CHATBOT_SOURCE_COVERAGE_PROTOCOL_20261015.md) fixes ALL432 without
score-based selection;40 LongAlign cases stay deferred,RESERVED untouched.
Capture and separately audit those donor packets before freezing another
learner recipe. A larger teacher cohort alone is not a transfer success.
