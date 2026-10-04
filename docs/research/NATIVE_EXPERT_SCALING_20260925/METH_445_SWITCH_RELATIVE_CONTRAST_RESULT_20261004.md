# M445: weighting alone fails; local and shared output dimension differ

Frozen c7fe39c before first import/eigensolve; terminal exit0.
[Protocol](METH_445_SWITCH_RELATIVE_CONTRAST_PROTOCOL_20261004.md),
[raw](meth445_switch_relative_contrast_result.json) SHA256
ddd1cd0d8c49e5e6a6b5305eca54e46e12f528ebe8e508dcb175661020b8685f.
ALL7 apparatus gates PASS, old444 means/contrasts/energy/covariance byte-exact,
all4 old energy curves reproduced<=1e-12; source provenance is retained444,
not a new native replay. Zero model forwards and optimizer updates.

## Equal-relative-anchor weighting does not repair the fixed global basis

Fit C_rel=mean_i(Z_i^T Z_i/||Z_i||²) on18development anchors only. Each anchor
has equal relative weight. Validation remains6 fixed anchors; no native heads.

| Fixed rank | Macro development private energy | Macro validation private energy | Worst validation anchor |
| --- | ---: | ---: | ---: |
|32|72.5695%|62.7272%|56.7205%|
|64|77.2018%|66.8408%|60.7061%|
|128|83.1418%|71.5742%|65.7276%|
|256|90.4980%|78.5202%|73.7924%|

ALL three prospective relative-energy gates fail for every fixed rank. Equal
weighting removes the88.4354%norm domination by two initial-position inputs,
but does not supply a small portable global basis on these responses. Close
this specific reweighting correction. These macro fractions are a different
estimand from444's absolute-energy-weighted fractions; compare within their
definitions, not as interchangeable quality scores.

## A separate oracle basis at each input is smaller, but unavailable at runtime

128 centered response vectors have rank<=127 exactly by algebra. Their Gram
spectra qualify that bound and full127-energy recovery. Minimum oracle rank
for95%energy is2 for development position0 anchors,57..87 for the other16
development anchors,75..87 for the6validation anchors. Local rank32 retains
only76.78..82.74%of validation energy, rank64 89.52..92.89%, rank96 96.86..98.25%.

This is a spectrum of all128 functions collectively at one already observed
input, not the latent dimension required by EACH expert or a transferable basis.
For example, many distinct rank1 private matrices can collectively span many
output directions. Neither global nor local spectra exclude small PER-EXPERT
matrix deltas with different bases. They also do not bound nonlinear or KL-aware
compression. Validation functions were used to fit their OWN local oracle Gram
spectrum, so this is explicitly not generalization.

The gap between local oracle spectra and development-fit global projection
supports investigating input coverage and bases that depend on expert/input.
It does not provide a cheap predictor of such bases, isolate one failure cause
or demonstrate source-native posterior quality. Energy is not KL/task accuracy.

Next: matrix-level single-donor base/delta feasibility with private ReLU kept
inside each expert, admissible coupled neuron permutations/scales, per-expert
delta bases and complete active-cost accounting BEFORE training. A full shared
base plus private factors saves possible storage; for Switch top1 it adds
arithmetic unless the base itself becomes cheaper. Top-k can reuse the common
down operation on the weighted sum of private activations. No rate assumption
from coefficient counts. Router normalization and physicalDRAM remain separate.

2.078s total/admission1.203s/algebra.875s,553,852,928B peak,
650,842,663B hashed, one15,769,662B NPZ. CPU0/BLAS1; Torch import only, no source
mapping/new head/gradient fit/GPU/T4/network/C edit/rate/DRAM. All first failures
and444 outcomes retained. Goal ACTIVE/INCOMPLETE; original artifacts unchanged.
