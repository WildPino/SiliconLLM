# METH-201: trained E12800 child function spread on real states

METH-175's tenfold E1280→E12800 bank passes training/load gates but
METH-176 finds no robust held-out quality gain. METH-178 shows that
only 2.318% of route-weighted candidate-minus-control B weight
difference lies in within-parent child variation, and METH-179 finds
no advantage for its exact content-child route over rotations.
Weight-space differences do not measure functional differences on
actual hidden states. This diagnostic separates two next actions:
train more distinct child corrections if functional spread is tiny;
study route learning if distinct functions already exist.

Bind the exact METH-175 candidate BF16 bank, METH-173 already-viewed
manifest, source donor, E1280 parent/child factors and low-recurrence
route table by existing SHA-256 checks. Reuse METH-176's initial
teacher/student parity check. Select manifest document indices
`0,3,6,9,12,15,18,21`, taking the first 128 causal input tokens of
each. No new text or target labels are read. Run the exact trained
E12800 model with its source parents, gates and selected child IDs.
At every one of the 24 MLP layers, capture the pre-MLP hidden states,
top-4 E1280 parent IDs, gates, selected content/structural IDs and
the layer MLP output.

For each top-4 parent at every captured token, apply its exact
shared A and the **nine trained content B children** to the same
hidden state, in BF16 as the model does. Keep the current parent and
gate fixed; do not change downstream model states. Restrict summary
energies to route selections whose current local slot is content
(1–9). Compute the mean output across nine siblings and the mean
squared sibling deviation around it, weighting both by the frozen
parent gate. Report pooled and per-layer root-mean-square sibling
spread divided by (a) root-mean-square content-parent mean residual
and (b) root-mean-square captured layer MLP output. Verify exact
selected-child output parity for every content selection and record
token/selection counts and the range across layers.

The **mechanism decision** is router-first only if at least 18/24
layers have both sibling/mean ratio >=0.10 and sibling/MLP ratio
>=0.01. Otherwise prioritize a different specialist-training signal
before further route optimization. These thresholds are diagnostic,
not quality gates: even a large spread cannot prove that child
selection reduces loss. This sample is already viewed and cannot
promote E12800 quality or native speed.

Cap at 20 minutes, 10.5 GiB allocated GPU memory, 20 GiB RSS and
1 GB output on the local RTX 3060. Stop on hash, parity, selected
output, shape or resource failure. Do not use T4.
