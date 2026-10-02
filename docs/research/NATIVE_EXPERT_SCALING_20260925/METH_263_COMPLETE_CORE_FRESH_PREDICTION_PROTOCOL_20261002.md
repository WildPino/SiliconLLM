# METH-263: frozen independent full-model prediction on the same saved core

Bind261 manifest SHA
`5af457561ad24fb837770735d0dbd0f23baa2f04a0525f38656285c0db7ec050`,
262 answerability SHA
`8adc68cd2b281740d2fcfbb7d6bf9c29936e61dfe1da79e01e88b4378b05e387`,
require all24 source-only gates/anchors/order. Bind259 export/actual
artifact SHA
`3c0949fb2b7f99dc887c6aec1041aa162b35801939349966d18e52b8807a6f71`,
all original helpers/parent/child/source weights, fixed text/token hashes.
No model score was read on this new cohort before this code/protocol freeze.
Do not select new sources or alter the saved candidate after observations.

Run the same three complete arms as260: original BF16 donor with bank
disabled; original BF16 centered E1280; actual saved259 source-core/
unique-bank candidate.24 new sources,8/code/prose/technical, full exact
BF16 tied head for document probabilities and donor-top1 agreement.
Original source controls are measured anew on these held-out sources;
old-cohort score replay is not a claim about this cohort. Candidate loader
uses the complete archive, not donor/checkpoint fallback. No refit or new n.

Require candidate pooled BPB minus both controls<=.01 and per-category<=.02.
Donor-top1 agreement versus BF16 E1280 may lose<=1 percentage point pooled,
<=2 in each category. Preserve all gate thresholds from214/260. Report
source-paired10,000-bootstrap BPB-difference P05/P95 against both controls,
seed263263; descriptive, no post-score threshold selection.

Separate finite prompt check: archived int8/FP16 head proposal, fixedK64,
no omitted full-head top1/no exact-row reranking mismatch on candidate
hidden states. This is not a universal unseen-generation certificate and
does not substitute for full-head probability/ranking comparisons.

All gates pass licenses separately frozen generation/tasks/blind
assessment on these same sources/artifact, then native complete engine/
alias lookup/CPU LUT/real-DRAM/routing and same-artifact>=50 accepted batch1
tok/s. Any failed prediction gate closes this fixed candidate before those
promotions; no artifact tuning, threshold relaxation or source replacement
on the observed cohort. Keep donor/bank/source-core contribution distinct:
this independent prediction screen does not itself establish arbitrarily
large useful n, RAM scaling,100B-capacity transfer or a second donor family.

Local RTX3060/six threads, deterministic/highest/TF32 off,20min after
imports,20GiB RSS/10.5GiB GPU. Cached source only, no T4/acquisition. Preserve
completed-arm partial/failure stages, concise exception logging. Freeze
before one execution:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth263_complete_core_fresh_prediction.py --answerability-sha 8adc68cd2b281740d2fcfbb7d6bf9c29936e61dfe1da79e01e88b4378b05e387 --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth263_complete_core_fresh_prediction_result.json
```
