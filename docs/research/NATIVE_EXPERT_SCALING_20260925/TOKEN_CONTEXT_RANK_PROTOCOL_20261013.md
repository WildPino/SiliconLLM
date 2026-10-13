# Token prototype plus contextual generator residual

13 October2026. New variable after full-width chatbot known-corpus qualification:
remove reusable token-dependent information BEFORE applying a small residual
factor. Reuses ACTUAL candidate own-H BF16 traces,no source model/GPU rerun.
Global low-rank generator and fixed output-bank failures remain valid;this
changes the residual being represented,not their old routing thresholds.

For each site1/12/23,full FIT24/11148 rows and DEV24/11399 rows:
H=mu(token)+R. Exact real arithmetic G H=G mu(token)+G R. Lookup stores first
term;second may have lower required rank. Two controls:BF16 FIT global mean,
BF16 per-token FIT mean with same global fallback for unknown tokens. ALL DEV
positions retained;unknown/covered counts reported,never excluded from gates.
These FIT prototypes are not source-only all-vocabulary tables or new knowledge.

FIT residual C=R^T R/N,pseudoinverse cutoff1e-10 lambda_max. Group weights
normalize each of5 source preconv field energy means on FIT. MuP values actual
BF16. M=diag(weights*MuP) G C^(1/2). Descending right singular/eigen vectors V.
Q=C^(-1/2)V,L=G C^(1/2)V. Candidate raw=G mu+R Q_k L_k^T,then source F32/BF16
round BEFORE effective BF16 MuP multiply/BF16. Own causal conv4/bias BF16,
SiLU BF16,delta BF bias/softplus BF16 are checked as additional field gates.
Original source matrix/full G H is the numeric control,1% per-field inherited
source-generator accuracy policy. No source decoder/cache/state is substituted.

Frozen ranks0/128/256/512/1024. Select smallest passing rank using ALL FIT
cases and ALL5 preconv AND causal field relative RMS<=1%. Then apply fixed
selection to ALL DEV with same1% gate. Rank<=256 required for a compact
candidate;larger ranks are descriptive. A field pass is not whole-model quality.
No success declared from averages,known-token-only subsets or oracle DEV tuning.

Separate audit reconstructs every FIT prototype,moment,root,weighted eigen and
source factor identity,and all6x48x5 candidate curves with separate causal-field
equations/manual BF rounding. FIT covariance1e-10,root1e-8,basis/source-factor/
eigen identities1e-6;metric metadata absolute1e-10. Shared NumPy BLAS primitive
is an explicit limit. Complete audit even if all compact candidates fail.

CPU6 cores,producer/audit each1800s cap,OS8GiB,new matrices<=1GiB in unique
E token_context_rank_20261013 family,no GPU/T4/SSM/optimizer/RESERVED. Expected
minutes,not an hours-long model run. Source/parameter/trace/runtime pins before/
after,terminal custody and no restart solely on observation timeout.
If this passes,next must compile/evaluate causal own-model substitution and
actual CPU lookup/residual cost. Final FFN/attention/head active cost,useful-n,
DRAM and same artifact>=50 useful acceptedTPS remain open.
