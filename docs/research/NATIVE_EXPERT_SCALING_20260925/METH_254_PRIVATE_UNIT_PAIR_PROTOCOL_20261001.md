# METH-254: frozen source-bound private32 nonlinear E16/E160 selection

METH-253 qualifies identical private32 BF16 nonlinear features at9.318ms
and retains81.03% local source-error recovery. Test useful conditional
source-unit choice in actual learned parent functions. Both arms use32
original BF16 gate/up rows and identical shared features/LUT/readout.
Only choices differ, with same fixed16/160 Euclidean keys/cells.

## Bindings and fixed fit selector

Bind253 raw
`9262f98cd87021d32697bb79badc1f07d7480a558def3cce6c9e48fb7466611a`,
require all gates. Bind actual METH-247 result/snapshot, METH-240 result/
snapshot, original captured BF16 x/y and router, METH-238 shared fixture,
pinned local BF16 donor file and layer12 gate/up/down tensor hashes.
Retain all247 fields except unused old E160 factors. Source down is
provenance only; the unchanged learned parent readout is executed.

Use original512 fit windows x128 states. Replay labels/counts, all16
old parent coefficient hashes and every original parent FP32 fit SSE
exactly, including pooled Python `sum()` ratio. No readout/routing refit,
rank/strength/mean correction or source Jacobian reset. Private rows are
exact original BF16 source weights. Unit-choice learning is the adaptation.

Compute shared FP32 phi and source BF16-as-FP32 gate/up features through
the same513-point LUT. Their FP64 difference delta gives a source-derived
nonlinear response per unit. Parent coefficient W is decoded mixed plus
decoded BF16 factors in FP64; reference predictions are actual FP32.
For each parent fit support and each of its10 child supports, independently
choose32 units by a fixed **positive-gain greedy** target-error objective:

- r=reference_F32-target_F32 converted FP64.
- a_j=sum_i delta_ij*(r W)_ij, b_j=sum_i delta_ij^2*||W[:,j]||^2.
- Choose largest gain `-(2*a_j+b_j)`, lowest source ID on ties; no repeats.
- Update r by delta[:,j]*W[:,j]; update every a through exact cross-unit
  covariance and W-column inner products. Keep all cross-unit terms.
- Stop if best gain is nonfinite/nonpositive. **Any cell stopping before32
  fails this fixed pair before validation**, no padding/forced IDs.
- Require finite r and final SSE=initial SSE-sum(chosen gains), relative
  closure<=1e-8. This greedy subset is not the globally optimal32-set.

Sort selected IDs ascending for storage. Execute the actual FP32 private
function using selected32 BF16 rows, overwrite shared features, and run
unchanged separated parent readout. Oracle/actual prediction relative L2
<=1e-5; actual FP32 SSE decides fitting/validation, not oracle SSE. All
selected gate/up pairs must differ from shared decoded rows. No slope
estimation is needed for constant-input cells: their source nonlinear
weights exist independently of labels; do not call them fitted slopes.

## Real function, artifact and fit prerequisites

Hash actual full decoded composite gate/up matrices plus unchanged decoded
parent W, not ID metadata alone; require176 distinct parameter signatures.
On the same first64 fit states of each parent, evaluate its private16
function and all10 private160 functions. Every pair among11 must have
relative output Frobenius distance>1e-7 against original reference norm.
This is fit-probe distinctness, not independent quality/useful count.

Store source BF16 gate/up[16 or160,32,896] and uint16 IDs[16 or160,32], plus
all retained actual parent fields. Save after all176 complete choices,
even if later distinctness/fit gates fail. Snapshot<140,000,000bytes,
read every tensor back exactly and verify retained parent fields unchanged.
Record copied source row overlap/choices; do not equate bank storage with
unique donor parameters or extra semantic capacity.

Actual fit E160 SSE/energy<=.01; both arms no worse than original reference,
E160 no worse than private16 (1e-8 relative FP32 rounding allowance).
All identity/oracle/closure/artifact/function prerequisites before targets.
An incomplete selector writes a valid stop record with no completed bank
or validation, not an apparatus exception or padded32-function claim.

## Consumed validation and decision

Only after all fit gates, read original128 consumed validation targets once.
Same source/keys: original reference, private16, private160 and within-parent
rotated private160; original source parent/child priors as controls. All
original per-window energies/reference/prior FP32 scores exact. Score all
actual functions with original FP32 subtraction then FP64 SSE accumulation.
E160 SSE/energy<=.01 and<=.9 each private16, rotated E160, original reference,
source parent prior and source child prior. Paired10,000-window-bootstrap
gain P05>0, seed254255. Failure closes this fixed positive-greedy private32
pair. Pass licenses actual learned-bank C/export next, not full promotion.

Local RTX3060/six threads,20min after imports,20GiB RSS/10.5GiB GPU,
>=2GiB free disk. No T4/download/fresh data. Partial completed cells/failure
stages retained. Bank may reuse donor rows; report resident/unique/function
quantities separately. No native learned-bank dynamic routing/large-RAM
DRAM/full independent quality/generation/task/accepted rate or family/scale
claim. Freeze code/protocol before running:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth254_private_unit_pair.py --checkpoint results/native_expert_scaling/meth254_layer12_private_source_unit_functions.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth254_private_unit_pair_result.json
```
