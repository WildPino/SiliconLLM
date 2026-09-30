# METH-208: frozen-router raw max-load sampling diagnostic

METH-207 passes fit and reserved chat, but reserved raw has six
max-load failures, worst 1.301374 vs <=1.25. Diagnose whether this
is outside the empirical variation of a 256-sequence sample from
the fit inputs. This is conditional on a router fitted to those
inputs; it does not prove no drift or pass the original screen.

Bind the exact METH-207 result/router hashes. Replay the existing
BF16 teacher/control eight-prompt parity, stored projection and
METH-175 table. Capture only raw fit draws 0:1024 and consumed raw
reserved draws 1024:1280, with unchanged width 512. Require causal
text mode on every sequence. Reconcile every saved raw-cell metric
and gate to 1e-6. Retain per-sequence, per-layer, parent/child route
counts as uint16; verify counts sum to content selections and
reconstruct the exact saved max-load ratio. No key/bias/B fitting.

Seed 208208 generates 2,048 bootstrap replicates, each drawing 256
sequence indices with replacement from the 1,024 fit draws. Preserve
within-sequence route correlation and use identical draw weights
across all 24 layers. Recompute each replicate's ratio as nine
times its largest child count divided by its largest content-parent
count, allowing both maximizing IDs to change. Report per-layer
and worst-layer 5/50/95/99 percentiles, layer-failure-count percentiles,
all-layer pass fraction and marginal/joint empirical tails relative
to the observed reserved maximum and six failures. These are
descriptive empirical tails, not independent significance tests.

GPU count multiplication uses FP32 with TF32 disabled. Integer
count sums are below 2^24; check the first three replicates of each
layer against direct CPU int64 accumulation. Save counts, weights
and all ratios in an exact-readback local artifact so the sampling
diagnostic can be repeated without re-running the model.

Decision frozen before execution: if either observed worst ratio
or failure count exceeds the corresponding 99th fit-bootstrap
percentile, flag changed-calibration investigation. If both are
within their 95th percentiles, classify sampling compatibility and
require a separately frozen larger route-adjudication cohort.
Otherwise classify inconclusive. No result relaxes METH-207's gate,
licenses native/B-training promotion, or uses unopened source cells.

Local RTX 3060, six host threads, <=20 minutes, <=20 GiB RSS,
<=10.5 GiB allocated GPU, <1 GB combined result/count artifact.
Stop on bindings, parity, metric/count reconciliation, exact readback,
integer accumulation, finite values or budget failure; preserve
failures. No T4, fresh quality text, chat replay or new training.

Runner: `benchmarks/native_expert_scaling/meth208_load_sampling.py`.
