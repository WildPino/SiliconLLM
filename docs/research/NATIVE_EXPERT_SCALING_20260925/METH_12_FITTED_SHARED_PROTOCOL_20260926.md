# METH-12: fit a compact shared residual before full-layer routing

**Status at registration:** CPU-only prospective test; no quality number from
this configuration has been observed. No T4/GPU run is planned.

## Distinct question and decision

METH-11 copied 64 donor FFN groups into an always-active shared path and
lost +0.706052 BPB at 31.25% active FFN width. E68's rank-64 linear
shared residual reduced local error with an oracle top-3 mass selector,
but failed complementarity across five layers. STRAT-03's particular
nonlinear shared path and learned top-3 router failed its local gates.
Here the changed mechanism is a **calibration-fitted rank-256 linear
shared residual** with the existing E37 input-only router selecting
top-16 of all 256 donor FFN groups. It is installed on all 28 layers
and judged by paired end-to-end BPB, not local reconstruction alone.
This decides whether a cheap learned core merits joint training/export
design; it does not decide final donor retention.

## Bound source, data and procedure

- Donor: Qwen2.5-1.5B revision
  `8faed761d45a263340a0528343f099c05c9a4323`, CPU fp32/eager.
  E256 coactivation labels have 35 distinct neurons/group, SHA-256
  `c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c`.
  E37 router SHA-256 is
  `42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a`.
- Calibration: existing disjoint `calib` corpus half, 16×512, seed
  424242, IDs SHA-256
  `3075c14d95a05dd69387e3df040370b053c16108bac409fa582db038f89be46e`.
  Use the first 12 windows (6,144 positions) to fit, last four (2,048)
  only to report local reconstruction. No heldout token enters a fit.
- For each intact-donor layer, capture MLP input `x`, hidden SwiGLU `z`
  and output `y`. Score 256 router rows on `x`; stable descending top-16
  with original lower group ID breaking ties. Compute the selected
  `y_sparse` from the original gate/up/down weights. Fit
  `S(x) = (x A_256) V_256^T` to `y-y_sparse` by the existing E68
  float64 reduced-rank ridge routine: lambda = 0.001 times
  `trace(X^T X)/1536`, Cholesky and SVD; cast factors to fp32 for scoring.
  No bias or rescaling is introduced. Donor attention, embedding/head,
  norms and all source FFN weights remain unchanged.
- Target FFN is `S(x) + y_sparse`. The 16 selected groups read 560 donor
  neurons/layer; the shared factors add 2×1536×256 weights/layer.
  Arithmetic active FFN count is 94,273,536 weights/token across 28
  layers, versus 1,156,055,040 dense donor FFN weights/token. The
  E256 router adds 11,010,048 score weights/token if scanned fully.
  These counts exclude the tied fp32 head and do not establish C rate.

## Fixed controls and stop

The existing H1 heldout 24×512 IDs must match SHA-256
`a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`.
Score only its first two windows, exactly as METH-11, 4,396 bytes. This
is a reused **internal pilot**, not a fresh final generalization set.
Before the transformed score, an all-groups-active, zero-shared wrapper
must match donor logits on the first 16 positions within 1e-4 absolute.
Validate finite factors and per-layer 256×35 partition; report local
calibration-validation SSE by layer and control the sparse output against
direct masked-down multiplication. Abort on any identity/data/finite
failure.

The prior METH-11 step-zero screen boundary is retained: if paired
pilot BPB exceeds donor by **more than +0.20**, stop this configuration
before joint training or export. If at/below +0.20, it earns a **new
training/export design only**, not a quality pass. A fresh untouched
heldout, generation/tasks, low-bit native export and ≥50 accepted tok/s
on the same artifact remain mandatory later.

Expected local cost: approximately 10–20 minutes of six-thread CPU fit
and score, no GPU; stop after 30 minutes or if process RAM exceeds
40 GiB. Save source/data hashes, factors, fit diagnostics and exact
command. Do not use heldout output to retune rank, k or ridge lambda.
