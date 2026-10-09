# Balanced whole Falcon recovery: bounded interrupted result

9 October 2026. Original attempt TERMINAL/INCOMPLETE, not eligible recovery.
Goal ACTIVE/INCOMPLETE. No T4 allocation.

## Identity and question

Implementation/protocol freeze1f2a30a20418d749c716b81a7610aadf612540d4;
[original binding](chatbot_hybrid_recovery_binding_20261009.json)
SHAed9299ffd4eceb846f0b15518436df8c49af565a0755c49c5ee183cac00e9065,
180 original inputs. Exact launch command and retained instance:
[process record](CHATBOT_HYBRID_RECOVERY_LIVE_20261009.md).
Original launcher bytes are archived under SHA65f8bf570191515ddea770e4ec5e153cd71949729dfa57d9865527a98fa93672.

Question:does a source-informed compact whole SSM/SWA/ternary target recover
balanced source-prefix outputs with finite complete connected learning and
the registered resource/quality gates? Fixed512 updates, final512 selection,
original inference AQ63, training-only dither.025. No metric-driven selection.

## Actual observations and retained state

| Quantity | Actual result |
|---|---:|
| Complete initial cases/labels | 160 / 5746 |
| Initial FIT/DEV equal-case KL | 24.0622717738 / 24.7118660808 |
| Initial FIT/DEV label disagreement | 4634/4643 / 1101/1103 |
| Durable recovery updates | 286 of512 |
| Complete epoch1/2 online mean KL | 6.5398319168 / 2.3660820425 |
| Complete epoch1/2 training seconds | 1336.703 / 1328.663 |
| Final after cases | 0 |
| Retained checkpoint | recovery.pt,3,059,446,998B |
| Original output files/bytes | 611 / 4,566,931,215 |
| Family wall time | 3588.125s |
| Worker OS peak through exit | 8,334,934,016B |
| Worker/launcher exit | 1 / 1 |

First fault is `AssertionError('worker deadline reserve')`,training phase,
3540.563s,after complete update286. The worker's60s reserve inside the3600s
family cap preserved the latest complete CPU model/Adam/RNG snapshot before
exit. See [launcher failure receipt](chatbot_hybrid_recovery_result_20261009.launcher_failure.json)
and the retained attempt'sfirst_failure.json. Original PIDs20000/33244 are gone;
session76414 is closed. This is the original bound attempt, not a repair rerun.

First and first-longest(183-input,update29) backward groups are all positive
finite across12 core/bank/norm sites. Every completed update logs global finite
gradient norm and CPU finite model/moment checks. Independent state verification
passed below. Two online means use different models at each case; their
improvement is not a final held-out quality result. Full512/all160 after gates
cannot pass. Historical support counts were not in recovery snapshots and are
missing. GPU through-exit peak is unavailable in the failed worker result.

Actual family time is below3600s and OS peak below16GiB; total attempt output
is below12GiB. This does not promote the incomplete recovery or supply missing
GPU/final quality evidence. No old source/capture/whole-native run was replayed.

## Saved-only adoption and decision

[Auditor protocol](CHATBOT_HYBRID_RECOVERY_AUDIT_PROTOCOL_20261009.md) and code
syntax checked after original exit. Launcher schema extension frozen
85aaaa828c1a2bd8b889d4953b6612140a3fbb7a. New784-input
[audit binding](chatbot_hybrid_recovery_audit_binding_20261009.json)
SHA9c97ad570e06aa4f23bf3cdbd47f58edd5928c96f8329269c4b2cf17eb0e60b7.
CPU-only audit TERMINAL/PASS,session2480 exit0. [Result](chatbot_hybrid_recovery_audit_result_20261009.json),
[terminal](chatbot_hybrid_recovery_audit_result_20261009.terminal.json):
63.094s family/6,795,464,704B worker OS peak through exit;all fixed resource
and input gates PASS. No CUDA initialization or modelcalls. Original identities
match with archived launcher. ALL160 before/5746 rows independently reduced in
F64;max per-label reduction discrepancy2.115653027345843e-5,all IDs exact.
ALL211 model tensors changed/finite,254,932,736 elements;actual finite Adam
moments2,039,461,888B,optimizerstep294=8+286;CPU/CUDA RNG storage present.
Checkpoint SHA7b95e699a57c9bc0ed320bef016d95ba78ac1df2e79b9ea98a0c12a1ba78eec5.
Durable update count exactly matches checkpoint boundary286. After cases0,
full_primaryfalse,reproduced recovery gatesnull,quality_admissionfalse.
SAVED_RECOVERY_AUDIT_PASS certifies transport/state, not absent final quality.

Follow [engine pipeline priority](ENGINE_PIPELINE_PRIORITY_20261009.md):inspect
the actual audit, then freeze a separate final-state evaluation/continuation
decision. Do not repeat any286 completed updates or initial outputs. Prepared
complete-cohort C tools accept only full eligible recovery and therefore cannot
promote this interrupted attempt. Original native19/32 numerical failure stays.
Useful fresh own-history/chat quality plus>=50 accepted IDs/s on the same
original-LUT/ternary/SSM artifact,large useful n/routing mass/DRAM/families remain.
