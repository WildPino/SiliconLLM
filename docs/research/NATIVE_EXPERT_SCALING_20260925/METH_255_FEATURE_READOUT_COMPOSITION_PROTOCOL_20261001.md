# METH-255: frozen fit-only private feature/readout composition diagnosis

METH-253 source/private composition reduces source-prefix function error
81.03% at9.318ms. METH-254 fixed private32 selection under the learned
readout stops at6 positive units on first parent, no bank/validation.
Determine whether the original-source versus adapted readout composition
explains this incompatibility before another function-bank attempt.

Bind254 raw
`453f03f31183e4da4d29fdf6b3aafefce9eef9ab7f44b18704348576ca1e8c35`,
require its stop/no checkpoint/no validation. Bind253 native all-pass
`9262f98cd87021d32697bb79badc1f07d7480a558def3cce6c9e48fb7466611a`,
252 raw/source fixture/all layer12 field hashes, original247 result/
snapshot,238 shared fields, source capture/router. Read actual source
layer12 table/base/rank32 residual/private field bytes from252 fixture;
require table/gate/up shared bytes equal original238/current247 controls.
The private rows/IDs are its fixed source-only choice, no new selection.

Only original512 fit windows, same parent groups. For each parent compute
shared FP32 features and fixed32 source-private FP32 features through
unchanged LUT. Four executed functions:

1. learned247 actual readout with shared features (original control);
2. same learned readout with fixed32 private features;
3. actual original-source245 mixed+BF16rank32 readout with shared features;
4. that same source readout with fixed32 private features.

Use original FP32 subtraction/FP64 SSE. All finite. Require16 unchanged
learned-parent decoded coefficient hashes and all16 original fit SSEs exact,
with same pooled `sum()`/whole target energy. No weights/strength/keys/rank
refit or Jacobian/center adjustment. No validation inputs/targets, private
selection, new C timing or checkpoint. Native qualification is inherited
only for that actual fixed source function; not the learned composition.

Prospective source coherence gate: source-private fit SSE/energy<=.01,
source-private<=.9 source-shared SSE and source-private<=learned-shared SSE.
All identity controls mandatory. All pass licenses a separately frozen
coherent-source-readout matched private-unit pair, not another retry under
the same learned readout. Otherwise change coupled transfer before bank
selection. Record learned-private change without redefining either gate.
No four-function fit score is independent quality or proof of count.

Local RTX3060/six threads,10min after imports,20GiB RSS/10.5GiB GPU,
JSON only, no T4/download/new dataset. Partial/failure stage retained. No
new fit or source acquisition. Freeze before running:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth255_feature_readout_composition.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth255_feature_readout_composition_result.json
```
