# METH-257: frozen complete source-operator core and effective E1280 bank

Move from local layer12 dictionary refinements to a complete physical model.
METH-256 is closed; none of its new amplitudes or learned readouts enter this
candidate. Use the fixed source-only FFN qualified by METH-252/253 in all24
layers and the already learned centered E1280 bank. This is a changed core
representation, not a recovery/retest of METH-214's rejected group-Q8 FFN.
This phase exports and checks apparatus without quality targets.

## Exact provenance and format

Pin METH-252 raw SHA
`7da69e9d44df64719fddbfb62a452a2b4afb3be7044511f1749d16305b6910ce`,
METH-253 raw SHA
`9262f98cd87021d32697bb79badc1f07d7480a558def3cce6c9e48fb7466611a`,
require253 all gates and same fixture SHA
`b90193f090812b9fd991616c0150f28f16a6553f1e2af2c1af8d8f5a489ba395`.
Pin METH-211 core/export, parent training/checkpoint, centered child
checkpoint and original METH-125 native state fixture through existing
fixed hashes. No original dense BF16 donor weights downloaded or loaded.
Config is locally cached original pinned Qwen config, stored in metadata.

The single safetensors format is
`M253_ROWQ8_LUT_MIXED32_RES32_PRIVATE32_EFFECTIVE_E1280_V1`:

- All220 non-FFN/core/head-proposal fields from METH-211 unchanged. This
  supplies exact BF16 tied embedding/head, BF16 attention/control organs
  and existing R8-FP16-scale greedy head proposal. No group-Q8 FFN copied.
- All361 actual METH-252 binary segments byte/hash unchanged: shared513
  FP32 SiLU table and15 fields in each of24 layers, including rowQ8 gate/
  up, indexed32 BF16 down outliers, separateBF16 rank32 residual and
  fixed32 original BF16 private input rows. No new selection or fitting.
- Five effective conditional fields per layer: parent A[128,8,896] BF16,
  centered child B[1280,896,8] BF16 and unchanged FP32 product-key router,
  child projection[32,896] and parent-local keys[128,10,32]. Existing
  METH-128 GPU centering arithmetic computes the original effective B.
  Require all10 sibling A matrices exactly equal before deduplicating A;
  require saved A/B exactly equal to original actual BF16 execution casts,
  and all128 sibling groups have10 distinct effective BF16 B matrices.

Thus701 tensors, all finite/readback exact, payload and file each<1.6GB.
Record every source segment hash and effective bank hashes/counts/bytes.
Saving effective A/B is execution-preserving for this BF16 Python model;
it is not preservation of unrounded FP32 latent training coefficients or
already established F32 native conservation. A is shared within siblings;
B contains genuinely learned values, not copies manufactured to inflate n.
Distinct B is an apparatus property, not new semantic/count quality proof.

## Complete loader and component conservation

Construct the model from the archived config with seed257257, BF16 and
SDPA. Delete all72 random constructor FFN parameters before execution.
Copy all218 remaining named parameters from this archive with exact
effective-dtype equality; verify tied head/embedding pointers. Install
stored FFN/conditional modules and consume all701 tensors, including
proposal fields; no source-weight or training-checkpoint load by loader.

FFN receives BF16 inputs, converts to FP32, executes exactly the qualified
source-feature lookup/readout arithmetic, then casts dense output BF16.
The conditional arm keeps old product-key top4, softmax, child-key argmax
and BF16 exact-SiLU A/B operations. Index shared A by selected child's
parent, B by child; sum the unchanged gated corrections in the old order.
No new LUT is substituted for the conditional SiLU, no route refit.

Compare all256 original stored input states in each of24 layers,6144
vectors total. Require original-source segment versus loaded FFN FP32
outputs bitwise equal, BF16 outputs bitwise equal, old/new parent IDs/
scores/child IDs bitwise equal and full conditional BF16 outputs bitwise
equal. For this conditional control, both old and new bank arms use the
same qualified saved FFN base: this isolates effective-bank conservation
and does not compare the changed core to the full donor. No model logits,
targets, document prediction, generation or task scores in this phase.

## Budget, decision and subsequent required gates

Local RTX3060, six host threads, deterministic/TF32 off;20min after imports,
20GiB RSS/10.5GiB GPU,>=4GiB free disk. Save artifact before loader checks
and preserve failure stage/rows on stop. No T4 or new model/data download.
Freeze code/protocol before executing once. All apparatus gates pass only
licenses a separately frozen complete-model consumed-development screen.
Then independently new source prediction/generation/tasks, actual native
whole-model parity and same-artifact accepted>=50 batch1tok/s remain
required. METH-214's consumed cohort cannot be called fresh for this core.

E1280 is a real existing bank, not proof n can scale arbitrarily with RAM.
Report retained pretrained core, unique source rows, copied source storage,
independently learned B, resident/selected bank bytes and measured route/
LUT/DRAM separately. Do not add9.318ms to unrelated component rates to
claim50tok/s. Large-n/RAM cost and quality, sparse GigaChat reuse and real
10B/100B/multi-family transfer remain open.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth257_full_source_core_export.py --artifact results/native_expert_scaling/meth257_qwen05b_source_operator_e1280.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth257_full_source_core_export_result.json
```
