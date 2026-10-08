# Balanced whole recovery: verified live instance

9 October2026. LIVE, no final recovery/native/chatbot admission.
Implementation and [protocol](CHATBOT_HYBRID_RECOVERY_PROTOCOL_20261009.md)
frozen at1f2a30a20418d749c716b81a7610aadf612540d4.
[Binding](chatbot_hybrid_recovery_binding_20261009.json) SHA
ed9299ffd4eceb846f0b15518436df8c49af565a0755c49c5ee183cac00e9065,
180 checkpoint/corpus/packet/code/runtime/foreign inputs.

## Authoritative process and available observation

Unified exec session76414; launcher PID20000, worker PID33244/parent20000,
both confirmed through Win32_Process with exact bound commands.
The launcher holds the actual Windows worker handle through exit.
Worker event at12.766s: complete254,932,736-parameter target loaded,
2,039,461,888B Adam moments restored,all initial optimizer steps8.
Longest FIT input selected before values:fit_reading_08.
Original naive eager SSD fallback is expected; optional kernels are absent.
No donor loaded, no T4 or concurrent model job.

ALL160 initial observations COMPLETE by445.5s;5746 F32 full-vocabulary rows
retained. FIT caseKL24.0622717738/labelKL25.7844728281,4634/4643 differing IDs
(99.8062%). DEV caseKL24.7118660808/labelKL26.0431082653,1101/1103
(99.8187%). These are NEW baseline values,not quality or training recovery.
At a later actual snapshot30 complete update records were durable; first update
fit_explanation_15 has all12 positive finite core/bank/norm gradient groups.
First four full updates12.359/10.265/11.531/11.875s including CPU model/moment
snapshot. At this observed rate512 within3600s is unlikely,but final time is
not yet known and the cap remains fixed. First-longest update29,fit_reading_08,
183 input tokens/3 labels,also has ALL12 positive finite core/bank/norm groups;
8.953s before snapshot/9.828s including snapshot. Actual update30 completed
at776.656s. These are changing-model training losses,not final evaluation.
Worker OS peak snapshot8,331,014,144B,
not through-exit measurement. Both exact PIDs still live at this observation.

Prepared [saved-only auditor](CHATBOT_HYBRID_RECOVERY_AUDIT_PROTOCOL_20261009.md)
and code are UNEXECUTED. First compile/review and extend launcher schema AFTER
original exit; never edit original bound launcher during this run. Its exact
bytes are retained in chatbot_hybrid_recovery_launcher_frozen_20261009.py.txt,
SHA65f8bf570191515ddea770e4ec5e153cd71949729dfa57d9865527a98fa93672.
The audit can adopt retained partial state without inventing final outputs.
Original launcher and all3 foreign tracked hashes rechecked intact at the
latest process observation. Current documentation/code commits do not mutate
the frozen worker/target/launcher inputs. No own tracked implementation diff.

Observe the SAME session/
log/result/terminal and current process before deciding status; never relaunch
because observation timed out. Completed initial rows are saved individually.
Full initial160 -> fixed512 balanced updates -> full final160 and actual
checkpoint. Caps3600s family/16GiB OS/11GiB allocated GPU/12GiB reserved/12GiB
output. Worker guard reserve60s for recovery.

## Exact command and outputs

Isolated Python3.12.10 `-I -S -B -X utf8`; shared launcher loaded via
`runpy.run_path` with safe runtime site first. Bound arguments:

```
--binding docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_hybrid_recovery_binding_20261009.json
--binding-sha ed9299ffd4eceb846f0b15518436df8c49af565a0755c49c5ee183cac00e9065
--freeze 1f2a30a20418d749c716b81a7610aadf612540d4
--directory results/native_expert_scaling/chatbot_hybrid_recovery_20261009
--out docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_hybrid_recovery_result_20261009.json
```

`.worker.log` is the live event stream; `.terminal.json` or
`.launcher_failure.json` records actual exit/through-exit cost. Directory
contains immutable completed observations/updates and final or recovery state.
Only a terminal receipt plus output inspection can establish closure. All old
native19/32FAIL,poor quality,useful-n/structured LUT mass/DRAM/family gates remain.
First action on resumption:poll76414 or authoritative processes,then adopt
actual output; no new training namespace or replay.
