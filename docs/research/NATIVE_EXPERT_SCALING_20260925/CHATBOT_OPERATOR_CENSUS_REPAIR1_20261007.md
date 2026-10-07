# Giga report representation repair1, before execution

Original source/protocol/binding frozena23514a. Qwen completed once, actual
executor0b8d35/exit0; its retained JSON will not be replayed. First Giga metadata
job70948d/exit1 reached report serialization, whose extended JSON exceeded the
fixed2MiB cap. The same serializer failed in the exception handler, losing its
in-memory report and actor/OS-resource measurements. Retained first
[fault record](chatbot_gigachat_operator_census_20261007.failure.json) distinguishes
the actual executor span from unknown Python peak/process instance. No stage
admission. No new tensor values/source responses/native/model calls occurred.

Repair ONLY serialization: use the same frozen operator accounting via import,
compact JSON without changing keys, counts, shapes, arithmetic or2MiB limit.
Since the first process did not retain its computed counts, the FAILED Giga
metadata computation is repeated; explicitly disclose this repetition, do not
claim no census replay. Completed Qwen/operator/native science is not repeated.
All original correctness, timing120s/peak256MiB/CPU10 and output2MiB limits stay.
The repaired failure handler also retains a bounded actor/resource summary if
even compact descriptor serialization exceeds the original cap.

Create a NEW binding referencing the original sealed inputs, plus the repair
source/protocol and original binding/fault; freeze it before the sole repair
execution. No fresh tensor headers need to be opened to extend the binding.
Archive/Qwen inspection remains unexecuted in this Giga-only repair. The original
protocol/source/binding/fault bytes remain unchanged. Actual repair exit/Windows
closure are recorded separately. Pipeline stays NOT QUALIFIED.
