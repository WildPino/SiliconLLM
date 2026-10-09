# Mixed rank8 history within48 heads: frozen before observation

9 October2026. Goal ACTIVE/INCOMPLETE. Previous fixed384-channel selection and
18-readout repair failed the declared local budget. New variable:linear mixtures
of UNGATED recurrent channels. Original engine/Adam25 unchanged;no source-model,
optimizer,native,RESERVED,T4 or new reply/label call. No previous state96 replay.

ALL48 retained forced histories/sites0/12/23,every history position;24FIT/24DEV/
12domains. One rank8 basis per48 heads/site from FIT only:144 algebraic head fits.
Use equal-case separately trace-normalized F64 y_h^T y_h Grams (64x64),top8 eigens;
centered per-case Grams retained separately. Orthogonality<=1e-10. No DEV fitting,
rank grid or output readout refit. Constant rank384 across all48 source timescales.

Source64-channel head scalar decay A_h/delta_h,shared B/C and same Dskip commute
with fixed linear R_h(64x8) in real arithmetic. Directly generate x'=x_h R_h and
run source chunk128/F32 SSD with8 channels/head/full state256 from zero state.
Use actual stored x/B/C/delta and pinned A_log/D. Bound temporary chunk axis,
same inner source reductions. No model load;only source code reshape/segment/pad
helpers and direct coefficient tensors. R stored F64;actual projected scan R/F32.

Check every case:generated y' versus captured y_h R_h relative RMS<=1e-4;
recurrent-only comparison removes identical projected Dskip*x,also<=1e-4.
Independent sequential F64 scalar recurrence on entire1507-ID FIT history for
eight fixed heads[0,7,13,19,25,31,37,47],coordinates[0..7],eachsite<=1e-4.
Scalar witness uses actual F32 mixed x promoted toF64,not a differentR projection.
Stop on failed numerical qualification before interpreting downstream error.

Reconstruct y_hat_h=y'_h R_h^T using F32. Apply ACTUAL FULL3072 source gate and
BF16 source norm/out_proj under two denominator diagnostics:(1)actual original
source denominator,(2)denominator recomputed from projected gated y_hat.
ALL144 original source gate/norm/output reconstructions<=1e-4 required.
Report full/centered/label-position relative RMS,equal-case/domain means/worst,
ungated y reconstruction error,and FIT perhead centered/uncentered retention.
Center each history/output channel separately. Double metrics/denominator clamp1e-12.
Neither diagnostic is a deployable original-core decoder;both use full source gate.

Predeclared encoder-study budget:EVERYsite DEV mean<=10%,case worst<=20%,domain
mean worst<=20%,centered mean<=10%,same as precedingassay. If any diagnostic mode
passes,this encoder merits a conditional-generator/decoder study only;choose
smaller maxsite mean thenlexicalmode. No useful-chatbot/native/quality/speed admission.
If neither passes,record MIXED_HISTORY_LINEAR_RECONSTRUCTION_BUDGET_FAIL;
do not infer all output-aware/conditional bases impossible or change thresholds.
R^T SiLU(conv(...)) and gate multiplication do not fold exactly through R;
real compact generators and an input-dependent decoder remain required.

Save all actual projected scan latents F32(T,48,8),103,896,576B logical payload
across144 records,with exact shapes/extents/hashes;basis R/Grams/eigens in NPZ,
metrics and held receipts. This allows later decoder fitting without repeating
the projected recurrence. Output cap128MiB (pre-observation increase from the
NEXT planning96MiB to hold actual F32 latents plus~10MB basis statistics).
No BF16 latent-storage approximation introduced. No original file overwrite.

One held family1800s/reserve90s,OSworker+launcher6GiB,GPU4allocated/5reservedGiB,
output128MiB/log4MiB. WorkerCPU0..5/Torch6/inter-op1/NumPy1,launcherCPU11;
deterministic algorithms/TF32off/CUBLAS :4096:8. Principal runtime binaries bound,
not complete DLL tree. Byte checks before/after,held PID/create_time/exit/OS through
exit/GPU/time/firstfault/completedbasis+metrics+latents. No owned overlappingjobs;
exact publisher exception. Observation timeout is not terminal;no unrecorded retry.
Adopt durable bases/latents/metrics in a separately bound missing-only completion
if necessary. Initial capture parent runtime gaps stay unknown.

```text
original_falcon_mixed_history.py --bind --out <new binding>
original_falcon_mixed_history.py --launch --binding <binding> --binding-sha <SHA> --freeze <commit> --directory <new namespace> --out <new result>
```

Python312 -I -S -B -X utf8,repo root. Reuse held channel launcher/schema with
experiment=MIXED_HISTORY_RANK8_V1 and new worker/namespace;freeze before run.
