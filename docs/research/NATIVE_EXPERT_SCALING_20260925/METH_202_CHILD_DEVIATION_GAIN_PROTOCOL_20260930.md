# METH-202: gain of the trained E12800 child-specific component

METH-201 finds only 1.583% gate-weighted root mean square sibling
output spread around the content-parent mean on real viewed states;
METH-179's route rotations show no robust exact-route advantage.
Before another long specialist training run, isolate whether the
small learned child-specific direction helps loss at all under its
**exact trained route**. This is a viewed-source mechanism diagnostic,
not a candidate model or quality promotion.

Bind the exact METH-175 trained E12800 BF16 bank, METH-173 24-document
manifest, METH-179 nine-route result, source E1280 parent/child
factors and route table by SHA-256. Repeat METH-179's eight-prompt
initial BF16 parity check. For each E1280 parent and each layer,
define `mean` as the arithmetic mean of its nine trained content
child B matrices. Keep the structural child (local 0), mean,
shared A, source parent, top-4 gates, trained exact child route,
attention and core fixed. Score three amplitudes for the nine
content children only:

- λ=0: each content B is `mean`;
- λ=1: the exact stored METH-175 B;
- λ=4: each content B is `mean + 4*(B-mean)`.

Round λ=0 and λ=4 transformed matrices to BF16 before CPU-master
inference, so their effective precision matches the stored bank.
Run λ=1 first and require exact per-document NLL equality with
METH-179 shift 0. Use its same 24×1,024 token IDs, causal windows,
absolute positions and per-document byte counts. Do not consume new
sources or inspect targets to choose λ. Report per-document and
pooled BPB for all arms. Bootstrap the paired per-document λ=0
minus λ=4 BPB on 10,000 resamples, seed 202202.

The **directional child-signal gate** passes only if λ=4 improves
pooled BPB by >=0.0002 versus λ=0 and the paired bootstrap fifth
percentile is positive. If it fails, do not amplify or continue
training the same fixed-route child residuals merely to increase
their norm; change the route/training coupling. If it passes,
develop a bounded training-only mechanism and still require fresh
source-disjoint quality. These are diagnostic thresholds, not
evidence that λ=4 is deployable or that n×10 is useful.

Cap at 20 minutes, 10.5 GiB GPU allocation, 35 GiB RSS and 1 GB
result disk on the local RTX 3060. Stop on bank/readback, baseline
NLL parity, shape, nonfinite or resource failure. No T4 is used.
