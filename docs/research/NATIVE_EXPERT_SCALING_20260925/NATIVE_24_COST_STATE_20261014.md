# Actual 24-layer recurrent cost screen: execution state

14 October 2026. Goal ACTIVE / INCOMPLETE. **COMPLETE / STORED AUDIT PASS /
RECURRENT_MEDIAN_EXHAUSTS_20MS.** All numerical workers are closed.

Freeze ba53952; binding V2 SHA256
f2ac53bb82f1adf760316d7a56b2681c1e3cff4fb5868ddce57efa97a331c73e.
Preparation23940/create1791966127.4061067 exit0; producer13964/create
1791966199.7384613/tool16978 exit0/227.344s; auditor23840/create
1791966451.0765328/tool54592 exit0/183.375s; closure29452/create
1791966681.851034 exit0/3.609s. All declared pins/output hashes and successful
PID+creation identities sealed. E194 files/463,653,248 bytes.
Resident median26.97095ms/parameter-evicted26.14040ms; all observed>24ms.
Full192-site/3072-call continuation/state/readout audit PASS.
[Final result](NATIVE_24_COST_RESULT_20261014.md). Next exact accumulation
candidate, not another FFN grid or rerun of these outputs.

Commands executed from repository root with Python312 -I -S -B -X utf8:

```text
benchmarks/native_expert_scaling/native_24_cost.py prepare --freeze ba53952b7af28de188bf2ff2ff8d4e5bfe6b76df --out docs/research/NATIVE_EXPERT_SCALING_20260925/native_24_cost_binding_v2_20261014.json
benchmarks/native_expert_scaling/native_24_cost.py run --binding docs/research/NATIVE_EXPERT_SCALING_20260925/native_24_cost_binding_v2_20261014.json --binding-sha f2ac53bb82f1adf760316d7a56b2681c1e3cff4fb5868ddce57efa97a331c73e --out docs/research/NATIVE_EXPERT_SCALING_20260925/native_24_cost_result_20261014.json
benchmarks/native_expert_scaling/native_24_cost_audit.py --report docs/research/NATIVE_EXPERT_SCALING_20260925/native_24_cost_result_20261014.json --out docs/research/NATIVE_EXPERT_SCALING_20260925/native_24_cost_adjudication_20261014.json
benchmarks/native_expert_scaling/native_24_cost.py close --report docs/research/NATIVE_EXPERT_SCALING_20260925/native_24_cost_result_20261014.json --audit docs/research/NATIVE_EXPERT_SCALING_20260925/native_24_cost_adjudication_20261014.json --out docs/research/NATIVE_EXPERT_SCALING_20260925/native_24_cost_session_closure_20261014.json
```

Historical commands: exclusive outputs already exist. Do not relaunch in
place. Preserve original failure history below.

## Historical preparation and repair

Original freeze 0adac14. Preparation PID 9576/create 1791965883.8138323,
exit 1 / 150.703 s / OS 51,703,808 bytes. It hashed preceding inputs, then
failed on a nonexistent root engine.c file. The authoritative current
engine path is benchmarks/phase60/engine.c. No support compilation, E family
creation, binding or numerical/timing observations occurred. The original
[failed receipt](native_24_cost_binding_20261014.preparation_terminal.json)
is retained and bound by the corrected preparation.

Repair: remove only the nonexistent additional engine pin; retain the
canonical engine and all other protected files. Preflight existence of
all pin paths before lengthy hashing. New source freeze and binding V2;
same protocol, four histories, last 16 positions each, all 24 source packets,
resident/parameter-evicted modes, thresholds, numeric auditor and resource
limits. Use native_24_cost_binding_v2_20261014.json; never overwrite the
original receipt. E family native_24_cost_20261014 is still unused.

Prepare, then inspect result/receipt before launching. Run result, independent
stored audit and closure each have exclusive files and worker receipts.
Do not run timing concurrently with other numerical research workers.
The source-active experiment is fully audited/closed and scientifically
failed as recorded in FFN_ACTIVE_ROUTER_RESULT_20261014.md.
