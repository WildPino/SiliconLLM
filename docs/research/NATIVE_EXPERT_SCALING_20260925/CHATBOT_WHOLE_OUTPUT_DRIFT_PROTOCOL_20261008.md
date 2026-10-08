# Whole output: new saved-only drift and coverage diagnosis

8 October2026. Freeze before new diagnostic values. Completed fixed fit/FIRST
audit reject all6 absolute/category gates, pass two relative KL gates. The final
FIT label meanKL0.4316736074 and last epoch online case mean0.2127655486 have
different weights/checkpoints. Their difference motivates diagnosis, not a
causal conclusion or changing the original transfer criteria.

## Question and bounded decision

How does fixed-final error differ from each case's online loss in the last epoch,
and where do residual errors/actual student route occurrences concentrate?
Reuse only independently adopted current fit's initial/final JSONL and1280-step
journal. No model/source/student/gradient/optimizer/operator/old audit replay.
This finite analysis can motivate a separately frozen order/optimization control
or a representation change. It cannot establish either cause by itself.

Prewritten axes: ALL8 categories with manifest case indexes/last epoch online
versus fixed-final case-weighted KL/signed per-case differences; ALL16 generated
position bins by FIT/DEV; ALL24x16 initial/final actual student route occurrence
counts and normalized aggregate total variation separately FIT/DEV. Validate
count conservation. Labels and summaries use existing F32 journal values already
independently checked in F64 envelopes; do not recompute saved logits.

Category and update position can be confounded; online states evolve. Route
occurrences are repeated rows, not distinct states/same-row agreement or selected
mass/utility. Teacher-forced bins are not own-history behavior. No pass/fail
threshold on diagnostic numbers, no checkpoint/epoch/precision ladder and no
promotion of the closed fixed fit or prepared native artifact.

## Execution and retention

`chatbot_whole_output_drift_binding.py` adopts completed fit/FIRST receipts and
the actual four-instance closure, binds only the three used journals plus code/
protocol/Python/DLL/psutil roots. Commit actual binding before executing
`chatbot_whole_output_drift.py` via the existing held-handle launcher in a fresh
namespace. Worker30s/family90s/OS256MiB/output256KiB/log2MiB, CPU10/launcher11,
no Torch/NumPy/Transformers imported, no CUDA/subprocess/endpoints. Original
foreign/publisher protections/through-exit resources apply. First fault/prefix
retained; never replay completed model learning to repair administration.

Verdict SAVED_WHOLE_OUTPUT_DRIFT_DIAGNOSED_NOT_CAUSAL; exact numerical values
and actual hashes/resources/exit will be written only after execution.
