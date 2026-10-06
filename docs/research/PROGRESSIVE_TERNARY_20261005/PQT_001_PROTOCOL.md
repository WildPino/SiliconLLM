# PQT-001: fixed-budget progressive ternary projection screen

Prospective protocol, 5 October 2026. No donor numerical observation yet.
Source commit and exact files will be recorded in the immutable dispatch bundle
before model/data admission. Subsequent repairs receive new IDs and preserve
all first failures. This is a diagnostic pilot, not complete-model promotion.

## Question and fixed hypotheses

For one full-width pretrained FFN projection, can progressive compensation or
discrete output-aware refinement improve on direct rounding at identical final
code/scale storage? Four arms: direct (D), direct plus refinement (DR),
progressive compensation (P), progressive plus refinement (PR). No method or
hyperparameter is selected after observing evaluation. Report all arms.
This first comparison isolates conversion method, not ternary versus every
available compression format; a favorable result still needs an int4 baseline.

## Immutable source inputs and split

Original donor `Qwen/Qwen2.5-0.5B`, revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, model.safetensors 988,097,824 bytes,
SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
Admit configuration/tokenizer files from that same revision, hash all used files.
No remote model code. FP32 evaluation; SDPA, eval mode, no dropout/autograd/TF32.
Only `model.layers.23.mlp.down_proj.weight`, shape [896,4864], is replaced.
Everything else stays at the original source values, including the readout.

`Salesforce/wikitext`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`,
configuration `wikitext-2-raw-v1`. Train parquet SHA-256
`e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`, 6,357,543 bytes;
test parquet SHA-256
`5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`, 732,610 bytes.
Join all text rows with newline, tokenize without special tokens in original row
order. Train: first 16 nonoverlapping windows of 129 tokens; fit on first 128
positions per window (2048 activations). Test: first 8 such windows, evaluate
first 128 positions (1024 positions) and their next-token labels. Save token
arrays and hashes. No validation split or project-consumed captures are used.
Public WikiText remains potentially project/pretraining-exposed diagnostic data,
even though test activations are excluded from fitting. No broad generalization claim.

## Representation and procedure

Symmetric codes {-1,0,+1}, group size 64 along input columns, one FP32 positive
or zero scale per output-row/group. Choose each scale once from original weights
by the exact L2 support-count optimum; keep these scales fixed in all four arms.
Zero wins ties at +/-scale/2. No bias, residual matrix, float correction or
rank expansion in the deployed converted projection. Pack four codes per byte
using 00=0, 01=+1, 10=-1 (least-significant pair first); 11 is invalid.
Expected payload: 1,089,536 code bytes + 272,384 scale bytes = 1,361,920 bytes,
15.625% of this projection's FP16 weight bytes. Account for file headers and
whole-model mixed precision separately; this is not 6.4x whole-model compression.

Progressive arm: original column order, block size 128, Hessian X^T X/N,
diagonal damping 0.01 times mean diagonal, upper Cholesky factor of its inverse.
Round each column to the fixed grid; compensate uncommitted columns with the
second-order error update. Preserve zero-activation columns; damping regularizes
them. Record columns committed and reject any nonfinite or failed factorization.
Refinement: two fixed column-coordinate passes over the ternary codes, minimize
calibration projection SSE by testing the nearest fixed-grid level to the exact
unconstrained coordinate optimum. Reconstruct residual every 64 columns and
each pass. Calibration loss may not increase above 1e-5 relative + 1e-12 absolute
tolerance. Count applied passes/columns. No optimizer or gradient fine-tuning.

## Metrics and decision fixed before execution

Retain per-window and per-position records, no favorable-window selection.
Layer metric: total held-out SSE divided by source projection squared energy;
also report unnormalized MSE. Full-vocabulary prediction: source-to-candidate
KL, source argmax agreement, and candidate minus source next-token NLL (nats).
All 1024 evaluation positions count. No clamping of nonfinite metrics.
Negative roundoff KL within 1e-7 is recorded, not silently clipped.

Pilot advance gate: at least one behavior-aware arm has >=10% lower held-out
layer normalized SSE than D, mean KL <= D + 1e-6, argmax agreement >= D minus
1 percentage point, and NLL <= D + .01 nats/token. Final payload <=35% of
projection FP16 bytes; conversion/fitting phase <=45 minutes. Report the gate
for every arm, not only a winner. A pass justifies further nonlinear/expert and
generation tests; it does not close the goal. A failure closes this exact recipe
on this projection/split, not all progressive ternarization.

## Apparatus admission and independent audit

Before donor access: tiny double-precision controls with exhaustive ternary
enumeration for optimal scales and coordinate choices; independent scalar
progressive recurrence; packed roundtrip including invalid-code rejection;
nonfinite and zero-curvature rejection. No donor evidence in these controls.
Save original source weights, calibration/evaluation inputs and outputs,
evaluation labels, final-normalized source/candidate states, scales and packed
codes. Those arrays allow reconstruction without importing fitting code.

Independent audit runs in a separate process on cuda:1 and imports no fitting
module. Decode packed bytes with independent implementation; check every code,
scale, reconstructed layer output, storage and aggregate layer metric in FP64.
Reload original readout independently from the admitted safetensors file.
Recompute full-vocabulary prediction for the first 4 positions of each
evaluation window (32 positions) in FP64. This fixed subset bounds the slow
FP64 readout on T4; the main report still uses all 1024 positions.
Record precision disagreements and
source/candidate margins. Main versus audit KL/NLL per audited position must
agree within 2e-5 absolute; source/candidate argmax disagreement is allowed only
if FP64 top-two logit gap <=1e-4. Source input/output numerical consistency
must have relative RMS <=1e-5. A failed audit prohibits scientific adjudication.
This subset audits numerical apparatus, not a substitute for all-position metrics.

## Resources, environment and retention

acct1, private `wildpino/pqt-001-projection-20261005-001`. PQT-ENV-002 confirmed
two Tesla T4 devices (15,636,037,632 bytes each), Python 3.13.15 and torch
2.11.0+cu128. Pin GPU image
`gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`.
Before donor access, install pinned transformers 4.57.6, huggingface-hub 0.36.0,
tokenizers 0.22.2 without dependencies, and assert torch plus these packages,
NumPy 1.26.4, pyarrow 23.0.1, safetensors 0.8.0 and Python. Save the actual
runtime; any mismatch stops the run before numerical observation. These pins
use published wheels compatible with Python 3.13; compatibility is still
qualified by the apparatus and model admission, not inferred from metadata.
Seed 20261005 for Python,
NumPy and PyTorch. One sequential fitting/evaluation process on cuda:0 and
independent audit on cuda:1; memory is separate. Require two actual Tesla T4s.
No local GPU/native timing, no main environment mutation. Server timeout 5400s;
installation/acquisition budget 900s, main process budget 3300s (including fit),
audit budget 900s. Hard subprocess timeouts preserve logs and first failures.
Measure fitting and elapsed times, peak GPU/RSS, actual quota before/after.

All hashes, environment, apparatus checks, stdout/stderr, partial checkpoints,
raw per-position metrics and audit go to the immutable run directory. Large
arrays stay outside Git with a committed hash/size manifest and retrieval path;
small raw JSON/log evidence is committed. Raw remote bytes use -text attributes.
Freeze bundle/source manifest and reproduction command before dispatch.
