# METH-222: conditional actual FFN function instead of full dense compute

Freeze before capture/fitting. Q8 and the one planar-Q6 float-input
kernel miss native cost. The next variable reduces active operations:
learn a small common function plus conditional donor-output functions.
This differs from failed post-hoc neuron carving (METH-11/12/14), global
weight-SVD corrections (METH-192), and weak children added to an accurate
full dense donor (METH-175). No old route is promoted or bias refitted.

## Bounded local transfer

One predetermined layer **12** of pinned Qwen2.5-0.5B-Instruct donor;
BF16 SDPA, no residual experts. Hash-bind actual pretrained model and
existing raw training corpus. Exclude exactly the same 352 legacy
chat/audit rows as METH-136. Seed 222222 chooses 640 different remaining
raw rows without replacement and one 128-token offset each. First 512
rows fit (65,536 token states), last 128 validate (16,384 states).
Sets are raw-row disjoint, not guaranteed source-document disjoint;
this is training-corpus validation, never fresh held-out LLM quality.

Capture actual pre-MLP x and BF16 FFN output y at layer12, no head
logits/generation. Save every BF16 bit, input ID, row/offset and hash,
read back exactly. No new corpus, calibration or source replacement.

Fit a common affine map x896 -> y896 by centered FP64 ridge regression,
lambda=0.001 times mean input covariance diagonal. Round weight BF16;
bias FP32 recomputed from fit means and rounded weight. Learn shared
64-dimensional projection from top fit input-covariance eigenvectors,
whiten by positive eigenvalues, round BF16, retain FP32 centering bias.
This is activation/function regression, not diagonal weight-energy SVD.

Fit 16 Euclidean parent centroids with 20 Lloyd steps, seed222222.
For each parent fit 10 local centroids with 20 steps, seed222223+parent.
Initialization uniformly samples k distinct actual fit states; empty
Lloyd cells retain their prior centroid, no adaptive restart. All means,
assignments and functions depend only on fit inputs/outputs. Lowest
cell ID breaks equal distance. Route always selects **one** parent then
one leaf: E16 versus E160 have equal active function width and shared
common/projection, tenfold stored function count. It is not the prior
four-route E1280 geometry or proof of arbitrary-n CPU scaling.

Residual target is actual y minus BF16-effective common prediction.
Per cell fit a 64-feature affine residual map to y896 with centered
FP64 ridge, lambda=0.001 mean local covariance diagonal (floor1e-12).
Round all 65 coefficient columns, including intercept, to BF16.
Every stored tensor must reload exactly before validation. Fit occupancy
gate: all160 cells occupied, >=90% have >=65 training states. Counts are
not independent samples/rank guarantees. Stop validation scoring if fit fails.

## Frozen local decision

Score only reloaded parameters on all 128 validation rows: common only,
E16, E160, and E160 with each chosen leaf rotated +1 modulo10 while
holding its selected parent and feature fixed. Report each sequence's
SSE and actual donor-output energy. Candidate E160 must satisfy:

- pooled SSE / target squared energy <=0.01;
- SSE <=90% of E16 SSE (useful increased learned choice);
- SSE <=90% of rotated-leaf SSE (function/route alignment);
- 10,000 paired-sequence bootstrap normalized SSE gain over E16 has
  strictly positive 5th percentile, seed222223;
- fit occupancy and every stored parameter readback pass.

No limit or hyperparameter retry after seeing results. Passing licenses
native cost plus multi-layer causal-composition development; it does
not license final quality or claim semantic transfer. Failing closes
this fixed affine-common/PCA64 hierarchical-affine specialist geometry.
It does not refute jointly trained nonlinear common/specialist maps.

## Cost hypothesis, not measurement

At width896/24 layers, BF16 common matrix costs38,535,168 bytes/token,
shared feature projection2,752,512, one896x65 cell2,795,520. FP32 parent
and selected10-leaf keys159,744; common/projection biases92,160. Total
FFN+route hypothetical selected weight bytes **44,335,104**. It replaces
both the old dense Q8 FFNs323,592,192 and old E1280 selected/router
ledger11,280,384. Keeping METH-211 head/attention/control would give
559,794,176 - 323,592,192 - 11,280,384 + 44,335,104 = **269,256,704**.
No whole-model artifact, DRAM trace or native kernel exists for this
combination. BF16 weighted FP32 native arithmetic still needs parity.
Conditional B storage alone is44.728 MB at E16,447.283 MB at E160,
4.473 GB at E1600,44.728 GB at E16000 across24 layers. Real n functions
must be learned and useful; arithmetic count is not conserved capacity.

Budget: local RTX3060, six threads, <=25 minutes after imports,
20 GiB RSS,10.5 GiB GPU, <400 MB additional output. No T4 or new external
quality/source/task. Other layers, full quality, CPU LUT/DRAM at large n,
multi-family/approximately-10B transfer and >=50 accepted tok/s stay open.

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth222_conditional_function_pilot.py --capture results/native_expert_scaling/meth222_layer12_training_states.npz --checkpoint results/native_expert_scaling/meth222_layer12_function_cells.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth222_conditional_function_pilot_result.json
```
