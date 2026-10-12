# Shadow v2 runtime guard fault and v3 correction

12 October2026. V2 sourceb9fe670, binding/invocation6272184,session6746 CLOSED
exit1. Worker12060/create1791791944.4272127,28.968s,OS500,912,128B,GPU0.
Full before-input hashes completed; empty new output directory created. No model,
controls/tasks/DEV/result. Preserve all failed holder/terminal/log and source.

The assertion queried `code.is_fast_path_available` BEFORE any mixer constructor.
Actual pinned HF source defines that global inside FalconH1Mixer.__init__,
lines443-467. It is absent at module import. Independent actual META-device
constructor probe verifies411 parameter tensors/194,801,792 elements,head_dim128,
SSM3072,and fast-path FALSE after construction. Probe JSON's `attention_heads`
label refers to the measured head_dim128, not the number of heads (source8).
No weights/data/model inference or GPU allocation in that probe.

Separate v3 moves the SAME fast-path assertion immediately AFTER the constructor.
No skipped fast-path validation or changed numerical/quality/resource criteria;
all original source transforms and v2 trajectory observation unchanged. Fresh
source/binding/output namespace and freeze before run.1200s/8GiBOS/GPU6/8GiB/
output3GiB preserved,DETACHED_PROCESS8 and strict no-worker-child guard retained.
This is an adapter ordering repair, not a source capacity result. No old result
or fault rewritten, no original categorical producer restart. Full behavioral
completion and independent parameter/operand/score/metric/custody audit remain.
