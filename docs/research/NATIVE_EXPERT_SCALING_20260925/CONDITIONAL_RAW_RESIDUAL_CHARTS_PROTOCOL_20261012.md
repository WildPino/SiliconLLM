# Conditional raw residual charts: frozen experiment

12 October2026. Oracle geometric feasibility, not affordable source routing or
causal compact state acquisition. New variable:4/16 affine charts with<=256
active continuous coordinates, on raw source boundaries0/12/24. Reuse audited
source raw states, source reading metrics and global256 controls. No model fit.

## FIT construction fixed before observation

All FIT24 cases/11,148 history states, deterministic case/token order. Base
case/token weight1/(24*T); chart-fit weight w=base/(mean(h^2)+1e-5). M is the
unchanged existing balanced normalized input reading metric at0/12 and the
existing mixture-Fisher actual normalized source head metric at24.

First center is the FIT h maximizing w*(h-weighted_raw_mean)^T M
(h-weighted_raw_mean). Each subsequent center maximizes w*nearest_existing_center
distance. Lowest index breaks ties, previously selected indices excluded;
no random seed or DEV selection. Then exactly32 Lloyd rounds, assign minimum
raw source-space M distance, update nonempty center to w-weighted raw mean.
Distance/r(h)^2 has the same per-state argmin. Empty slot retains previous center.
Final centers/assignments,33 center snapshots/33 assignment and objective rows
and initialization indices are stored. Objective must be nonincreasing within1e-8.
The final assignment may differ from that used for the last center update;
centers are FROZEN at that point, not silently converged or recentered.

Within each final assigned chart C=sum w(h-mu)(h-mu)^T/sum w about that frozen
center. It is a second moment, not assumed exactly centered after32 rounds.
Full C eigensystem; support eigenvalues>1e-12 lambda_max. For Q/lambda on support,
S=Q sqrt(lambda), J=Q/sqrt(lambda), K=S^T M S. Full support K eigensystem U.
V=S U_256,W=J U_256. Actual active=min(support rank,256); unused columns exactly0.
Rank0/empty chart is an explicit frozen-center constant decoder, not a duplicate
or ridge. Empty covariance0 and no fitted chart directions. No change to K after
DEV. Support/occupancy/active dimensions/masses all reported per slot.

For z=W^T(h-mu), hhat=mu+Vz. Numeric W^T V equals diagonal1 on actual active
dimensions and0 on inactive dimensions within1e-8. C/K eigen residual<=1e-10,
negative eigenvalue tolerance1e-10 of max(lambda_max,1). Empirical weighted FIT
quadratic reading loss equals mass*sum discarded K eigenvalues within1e-8 relative
to total chart reading energy, including negligible covariance support loss.
This optimizes the fixed-center linear reading surrogate, not categorical KL.

## Exact norm and offset algebra

Store G=V^T V,c=V^T mu,a=mu^T mu. Own denominator is
sqrt((a+2c^T z+z^T G z)/2048+1e-5); no normalized-state center used as raw center.
Input gamma belongs in each actual source reader A. Affine reading is
(A mu + A V z)/own_radius. It needs a chart-dependent bias even if source has none.
Skips for all chart pairs at representative0->12 and12->24 store
W_new^T V_old and W_new^T(mu_old-mu_new). These are coarse representation maps,
not validated actual source dynamics between those boundaries.

## Held-out measurements and frozen decisions

All48 cases/22,547 full history states at each boundary and each K: source-space
nearest-center ORACLE ID, encoded z from full source h, true/own radii, percase
occupancy and full M-relative reading errors. Both denominators measured.
Same prior proxy: every tested boundary DEV case mean<=.10 and worst<=.20.
All288 case packets persisted; no deployment claim from oracle index/encoding.

First/middle/last HISTORY positions for each48 case give144 inputs at0/12 for
each K. Store full actual source eight projection observables and both decoded
versions, normalized input packet and labels; pooled split/group errors reported.
Projection functions use source gamma/MuP/key multipliers exactly, before
convolution/SiLU/RoPE/attention/recurrence. No nonlinear model preservation claim.

At24 use SAME DEV72 first/middle/last LABEL states and source F64 full-V reference
as the global256 experiment. Full source head reading on reconstructed raw states,
oracle/own denominators; full scores/decoded states saved. Each mode meanKL<=.01,
worstKL<=.1,disagreement<=.01. Actual finite losses, not Fisher approximation.
K qualifies locally only if all three own-norm reading proxies and both final
probability modes pass. Independent full audit regardless of scientific failure.

## Cost, custody and limits

CPU6/affinity0..5;1800s/OS8GiB/output12GiB. Full relevant input SHA before/after,
exact source/protocol freeze,binding/invocation/PID/creation/exit/terminal retained.
Initial source imports principal NumPy/psutil binaries bound, no neural loader.
No source/shadow/optimizer/backward/native/GPU/T4/RESERVED/longalign calls.
External publishers and foreign files retained; no clean timing admission.
No hours-long run selected; if bounds need hours, follow explicit user pause rule.
First mechanical fault retained; repair needs separate source/namespace.

Store full FIT input/weights, centers/assignments/history; complete per-chart C
and support kernel spectra/eigenvectors, V/W/norm/cross/constants; all case
encodings; actual source projections/head scores; all pairwise affine skips.
Account for active coordinates, distinct maps, centers and biases, selected
reader/head shape traffic, K-scaled oracle search and K^2 skip storage. Dense
operation/storage counts do not constitute engine LUT/physical DRAM benchmarks.

This is the representational ceiling before causal compact routing. If it fails,
identify remaining finite readout or internal reading defects and new correction.
If it passes, actual causal IDs/coordinates/composed own-history and ternary bank
conversion are still required, with useful expert scaling and same-artifact>=50.
