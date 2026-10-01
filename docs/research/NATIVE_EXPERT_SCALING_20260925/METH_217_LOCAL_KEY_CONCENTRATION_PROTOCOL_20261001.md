# METH-217: diagnose the fixed local-key raw concentration

Freeze before execution. Read-only METH-216 router/result, SHA256
`c92d603885513ec30d2eaee36975760df243dffb30521830524e32512ec3397e` /
`b2669d8ad250cc6e980717c94ec95641f8469e04437c1937ec9f643351000f0b`.
Reuse pinned BF16 E1280, exact factor bank, child coordinates,
structural table and identical 3072 raw fit sequences. No key/bias/B
updates and no reserved/source inference. Previous turn made progress:
stored-core fresh ranking rejection and local-key fit narrowed the
route failure to hot-parent share in zero-based layers 12/16/17/20.

Require eight-prompt teacher/control exact parity, frozen projection
equality, and original fit totals plus every metric/gate reconciliation
within 1e-6 on those four layers. Capture normalized FP32 q, int16
selected parent IDs and sequence boundaries for content selections.
Persist those four full layer captures under <1 GB so future diagnosis
does not require repeating model inference. Bind their bytes by SHA256.

At every hot parent (>=250 selections), compute exact q-bit multiplicity,
hard counts and soft counts at the saved final temperature 0.0004.
For every hard-share failure (>25%) and two highest-share passing
parents per layer, record all nine counts, unbiased-score dispersion,
biased top-two margins (min/p05/median/p95/max), exact ties, margins
<=1e-6, identical key rows and key cosine similarities. Do not substitute
METH-205's different projection/1024-draw diagnostic.

Fixed interpretation for each failing parent:
- exact-q group >25%: this deterministic feature-only route cannot
  meet share; change features/causal state before recalibration;
- otherwise soft share >25%: final soft calibration itself is unbalanced;
- otherwise: soft-to-hard deployment gap, requiring a hard partition
  calibration or changed decision rule rather than more soft updates.
Report mixed mechanisms if different parents differ. Low support,
ties and margins are descriptive evidence, not grounds to relax gates.
No classification proves held-out route health or useful specialist B.

Local RTX 3060, six host threads, <=20 minutes, 12 GiB RSS,
10.5 GiB allocated GPU and <1 GB new disk total. Preserve partial
metrics/failure stage. No T4, training, native rate or fresh quality.
Runner: `benchmarks/native_expert_scaling/meth217_local_key_concentration.py`.
