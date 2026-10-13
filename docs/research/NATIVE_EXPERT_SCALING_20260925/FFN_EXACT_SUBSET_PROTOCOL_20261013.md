# Frozen exact combined remainder coverage assay

Before observations. Actual FFN operands/all48 histories/sites0,12,23.
Existing64 FIT centers,full2048 outputs. No new centers/source/GPU/RESERVED.
Budgets32/128/512/2048 plus4608 all-neuron identity control. Source BF
coefficients and effective F32 alpha/beta,smooth proxy compared against
actual captured BF output per-case1% numerical gate.

Combined exact feature remainder from FFN_EXACT_REMAINDER_SUBSET_NEXT_20261013.md.
Per-cell token-weighted FIT mean remainder squared times squared norm of
beta-scaled source down column. Stable descending score,ties source neuron
index. Empty FIT cell uses global token-weighted FIT score. Score ignores
cross covariance:it is a defined heuristic,not optimal subset selection.
Save all64x4608 scores/ranks/counts/column norms;no DEV selection/reranking.

Every candidate uses implicit full-output affine plus selected exact remainder.
All4608 reconstructs smooth proxy within1e-10 and is CONTROL ONLY,excluded
from reduced-cost selection. Unchanged FIT mean<=.10/worst<=.20 selects
smallest passing noncontrol budget;DEV same mean/worst/domain<=.20. An
eligible local approximation still requires actual own-history model quality.

720 complete F32 predictions,one uint32 cell ID and F64 mass exactly1 per
token/variant,and3 model NPZ. Exact array bytes2,785,362,780;ZIP headers/keys
add overhead within3GiB E cap. Unique ordered family ffn_exact_subset_20261013.
Producer CPU6/1800s/OS8GiB;full audit CPU6/1800s/OS8GiB. Hash all source/runtime/
parent/input/output pins before/after. Full audits even when quality fails.
Real terminal receipts and PID/create-time absence checked at closure.

Auditor separately computes source center values,direct Euclidean routes,
direct input-delta chain-rule action,exact feature-minus-affine remainder,
grouped FIT squared scores and stable ranks,then dense masked full-output
predictions. FIT scores relative1e-10,column energies1e-12,F32 predictions
1e-6,all case/selection metadata absolute1e-9. Every ranking/ID/mass exact,
every identity control1e-10. Shared tensor reader/BF decoder/NumPy BLAS scope
explicit. Freeze all three scripts before binding. No physical bank,ternary
quantization,scalable CPU router,memory traffic,whole model or speed admission.
