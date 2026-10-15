# Complete mixed recipient: numerical rejection before cost

15 October 2026. Attempt frozen at `4b78ac7` under the
[prospective protocol](RECIPIENT_MIXED_BASELINE_PROTOCOL_20261015.md).
**Source-to-packet checks PASS; decoded numerical parity FAIL; cost NOT RUN.**
This is an implementation/reference qualification failure, not evidence of
chatbot quality, donor transfer or an impossibility of the conditional program.

## Retained execution

The bounded controller exited 1 after audit rejection. Export took 20.937 s,
compilation 1.625 s, native probe 1.359 s and audit child 29.375 s. No generation
case, fitting, source download or T4 job ran. First-fault outputs remain in
`results/native_expert_scaling/recipient_mixed_baseline_20261015`.
The [closure receipt](recipient_mixed_failed_closure_20261015.json),
[full audit diagnostics](recipient_mixed_failed_audit_20261015.json) and
[controller failure](recipient_mixed_failed_controller_20261015.json) are
retained verbatim for review; all four owned child identities were absent
at closure. Closure SHA256 is
`7a55024441c183a1889edf7007871c866a7ef34d4edd089fa129c534aeca8b2a`.

The packet contains all 291 roles, including explicit W8 head and retained
BF16 embedding: 630039552 bytes, SHA256
`8bb77b499d76031a128cf8044889a7a772953e96dbe4c65b670e518ae95d00cb`.
All source bytes/scales/codes/padding passed independent verification against
the original 290 BF16 tensors. This does not verify dynamic execution.
The probe is SHA256
`9dfdc44494987db9116608d70261af9e6790f8b5856a0214260077dbf8090d12`.
Its binding is SHA256
`1c555b313197b7bd06c1876c06c5df776a444327da5041b5430f62e7ede420b6`.

| Position | Final norm relative L2 | Whole-head relative L2 | Native / reference top1 |
| ---: | ---: | ---: | ---: |
| 0 | 3.79293e-8 | 1.13570e-9 | 16964 /16964 |
| 7 | 0.0825984 | 0.0921821 | 11 /271 |
| 15 | 0.0241060 | 0.0665110 | 279 /279 |
| 31 | 0.0289408 | 0.0761825 | 16148 /16148 |

The frozen gate requires all four top1 values and each norm/head relative
L2<=1e-3. It fails three numerical rows and one top1. The reference consumed
all 32 prefix tokens and sampled outputs at the four declared positions.
Its internal work took 25.125 s, peak sampled RSS 4052836352 bytes.

## Decision

Do not report a decode rate or launch long recovery on this unqualified mixed
runtime. Preserve the failed attempt; diagnose the first operator/code
disagreement using the [instrumented replay protocol](RECIPIENT_MIXED_DIAGNOSTIC_PROTOCOL_20261015.md).
Floating-point differences amplified by dynamic quantization are a hypothesis,
not yet a demonstrated cause. Neither packed-byte equivalence nor a matching
first token is sufficient numerical qualification.

Months of T4 usage remain accepted for a qualified conditional transfer path.
The overall goal is active/incomplete. This fixed numerical failure does not
change the large-donor requirement or promote a smaller standalone model.
