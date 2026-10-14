# Frozen source remainder-sensitive balanced input routing assay

Before new observations. Same source/actual48 own histories/sites0,12,23,
existing64 FIT cells,active counts512/2048/32 and rank grid64/128/256/512.
Full2048 output/exact selected source remainder. No DEV fitting/new centers/
source model/GPU/RESERVED/optimizer or whole-model substitution.

Metric M and balanced encoder/decoder equations are defined in
FFN_SOURCE_ACTIVE_METRIC_NEXT_20261014.md. FIT token-weighted exact source
remainder gradients,weighted by squared norm of beta-scaled down columns.
This is a surrogate for importance selection,not a score-gradient,optimal
ranking or cross-neuron output-loss metric. It changes input feature objective.

Accumulate all FIT coefficients E[a^2],E[b^2],E[ab],input uncentered moment,
mean/covariance and direct derivative-action Gram on four RNG0 unit vectors.
Materialize M in grouped source products and require relative action1e-10,
symmetry/PSD relative1e-10. Sigma cutoff is EXACTLY1e-12*largest eigenvalue;
save whole spectrum/basis,dropped trace,supported rank/conditioning. Supported
rank must be>=512. Form square root/pseudoinverse root on that support and
T=root Sigma * M * root Sigma,symmetric. Save T/eigenvalues/basis and dual
encoder/decoder512. Duality relative1e-10. No unexplained orthogonality claim.

First FIT history,16 evenly spaced rows,RNG0 unit directions after the four
action directions. Fixed center,central difference step1e-4;check exact
source per-neuron remainder derivative. Weighted discrepancy normalized by
the full SOURCE FEATURE derivative norm (not possibly-zero remainder norm)
<=1e-5. Save indices,directions,analytic/numeric actions/normalizers/errors.
This handles the known zero-remainder anchors without arbitrary score noise.

ID function receives ONLY delta,cell IDs,center constants,encoder and projected
U/G using decoder,down-column energies,rank/count. It never receives original
U/G or reference activations. All rank choices fixed BEFORE full original
reference activations. Evaluate true chosen source remainder afterwards;
implicit full affine/source reference still used for diagnostic. No physical
kernel/latency claim. Source proxy versus actual BF<=1% per case unchanged.

FIT mean<=.10/worst<=.20 selects smallest rank. Same choice tested on DEV,
mean<=.10/worst<=.20/domain<=.20. No promotion through DEV or altered gates.
Save3 complete geometry/source-projection models and576 full F32 predictions,
cell uint32/F64 mass1/uint16 selected neurons. Exact model payload computed
from six2048-square matrices,four2048 spectra/mean,two2048x512 factors,
two4608x512 projected operators,four4608 vectors,action and FD arrays.
All byte arithmetic is bound before run;ordered E family ffn_active_router_20261014,
output4GiB/producer1800s/auditor1800s/OS8GiB/CPU6 each. Full audits on failure.

New binding updates only four known NVIDIA components to current exact
hashes,retaining old/new descriptors and historical CPU environment closure.
All other inherited pins exact. Those GPU components are not executed by
CPU algorithms;no claim of replaying historical GPU capture under new driver.
All actual bound current bytes/source/model/output hashes before/after.

Auditor independently reconstructs concatenated FIT moment/covariance/mean,
source derivatives/averaged coefficients/direct full FIT derivative Gram,
four-term metric,PSD/eigen/root/balanced/duality/source-projection witnesses.
All numeric algebra relative1e-10;source finite difference1e-5. Literal
projected-only IDs replay exactly,then independent direct distances/input
deltas/feature-minus-affine and256-column output sums. All576 F32 predictions
relative1e-6/case/domain/selection metadata absolute1e-9/controls1e-10 replay.
Shared source reader/BF decoder/NumPy BLAS/literal ID rule explicit. Terminal
receipts/PID-create absence/complete579-file extent/protected hashes closed.

This is an incomplete transfer procedure. Any local success still requires
compiled CPU ID/mass/precision/real loaded bytes/DRAM/cost screen before
all24 or large banks,then own-model quality/donor-relative held-out and same
useful50 on the same artifact. Middle width2048 remains costly. Multiple
families/scales/useful RAM-driven n and conversion costs remain open.
