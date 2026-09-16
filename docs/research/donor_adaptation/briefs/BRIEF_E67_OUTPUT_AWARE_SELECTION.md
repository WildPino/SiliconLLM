# E67 — does E38's mass ranking hide output-preserving blocks?

**Status: preregistered design, 2026-09-16; no runner or donor result yet.**
This is a CPU-only diagnostic on existing Qwen2.5-1.5B weights. It authorizes no
T4, engine change, donor promotion, or 10B/50-tok/s claim. Freeze this brief in
git before writing or running an instrument; report every deviation.

**Addendum A (2026-09-16, before any donor run):** E41 §6 was read after the
initial brief freeze. It found two exact float32 group-mass ties among 344,064
rows; the verified fix for the *next* oracle was float64 accumulation. E67 must
therefore compute `sum(z_i²)` in float64 and break any remaining tie by lower
group label, recording tie counts. This preserves E38's **mass criterion**, not
bitwise replication of its float32 `topk` IDs. Recompute the E67 mass baseline
within E67; do not compare its local errors to E38's BPB or require exact
float32 selector IDs. This amendment changes no donor result or E38 verdict.

## Why this is a genuinely different cell

E38's `oracle` sorts each group by the intermediate SwiGLU energy
`m_g = sum_{i in g} z_i²`, with `z = SiLU(W_gate x) * W_up x`. Its score is
`4.131817` BPB at `k=3/256`, above chance. That result closes **mass-based
selection on its frozen partition**, but does not bound every selector: the
output is `y = W_down z = sum_g v_g`, where `v_g = W_down[:,g] z_g`.
The local squared error of selecting groups `A` is
`||y - sum_{g in A} v_g||²`. Because output-column norms differ and group
contributions interact, sorting `m_g` need not minimize this error, much less
end-to-end BPB. E38's registered `G-E38C` is an observed control comparison,
not a mathematical theorem about BPB. Its own brief §8 acknowledges that
gradient- or output-aligned criteria were untested.

The simplest counterexample is `z=(10,1)`, `W_down=(0.001,1)`, `k=1`:
mass chooses component 1 and incurs error `1²`; choosing component 2 incurs
error `0.01²`. This does not predict the donor; it proves the inference from
"top mass" to "best output" invalid. The new test must include this planted
case and a cancellation case as code-level self-tests.

## Frozen object and input

- Pinned donor `Qwen/Qwen2.5-1.5B`, revision
  `8faed761d45a263340a0528343f099c05c9a4323`, fp32/eager, unchanged.
- D0c's existing `E=256` labels from
  `benchmarks/donor_adaptation/density/results/d0c_labels/labels_E256.npz`;
  same 35 neurons/group, no repartition or refit.
- The existing `common.get_slice(tok, "heldout", 24, 512, 1234)` with full
  `ids_sha256=a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`.
  For the local diagnostic, use sequences 0–3 and positions 384–511 of each
  512-token sequence (512 scored positions total). Record the SHA-256 of the
  selected IDs as well. No calibration or fitting uses this held-out slice.
- Layers `[1, 7, 14, 21, 27]`, the D0 deep-layer set. Capture `z` from the
  intact model; no mask is applied during capture. Process contributions in
  token chunks so the 256×hidden contribution tensor is not stored for all
  512 tokens at once.
- Set intra-op threads to six. No quiescence gate: this measures numerical
  reconstruction, not wall-clock rate. Report CPU time and peak RSS only as
  operational observations, never as tok/s.

## Equal-activation arms at `k=3`

1. **Mass:** E38's `sum(z_i²)` criterion, accumulated in float64 per addendum
   A, top three; recompute, do not quote a stored selector output. This is
   the matched criterion baseline, not a bit-identical E38 replay.
2. **Output-norm:** top three by `||v_g||²`, isolating the effect of the
   `W_down` column norms without accounting for cancellation.
3. **Output-greedy:** start residual `r=y`; at each of three steps select an
   unselected group maximizing `2 r·v_g - ||v_g||²`, then set `r -= v_g`.
   This is greedy minimization of the **actual FFN-output squared error**,
   not an exact global `k`-subset optimum and not a BPB oracle.
4. **Random-label null:** use D0c's stored random equal partition with the
   same output-greedy selector and `k=3`. Its purpose is to test whether the
   purported gain requires the learned/coactivation partition; it is not a
   speed candidate.
5. **Dense identity:** all 256 groups, required to reconstruct `W_down z`.

All sparse arms read the same number of selected FFN neurons. Their selectors
are **unattainable at that byte budget** because computing every `v_g` first
requires the dense FFN; they are capacity diagnostics, not executable router
proposals. Report that charge explicitly. Do not price them at E63's mixed
byte rate. The output-greedy arm may lose to mass on some tokens; it is not
declared an upper bound, and no code assertion may force the ordering.

## Checks that must fire before interpreting a donor result

1. Planted unequal-column-norm case above: mass and output-aware choose
   different groups with the stated errors. A separate cancellation case
   verifies the greedy score is computed from the *current residual*, not a
   one-shot output norm.
2. For every sampled token/layer, `sum_g v_g` reconstructs direct `W_down z`
   to relative RMS error `<1e-5`, and the dense identity arm has the same
   result. Fail closed on a dimension, label, or SHA mismatch.
3. Every sparse arm selects exactly three **distinct** labels, whose groups
   contain exactly 35 neurons; identical logical FFN-weight count for each
   arm. The actual row-layout cost remains unmeasured.
4. Recomputed mass scores must match the E38 criterion. Tie handling is
   deterministic (lower label wins); count/report ties rather than hiding
   them. No training, checkpoint search or changing the slice after results.

## Outputs, bands and stopping rule

For each layer and selector report `sum||y-yhat||² / sum||y||²` across all
sampled positions, medians of per-token relative error, and per-sequence
values. Report the mass→output-greedy change both for each of five layers and
as the ratio of summed squared errors over all five; include output-norm and
random-label null, counts of per-token wins/losses, control details, source
hash, donor revision, input/label hashes, and elapsed time in one JSON.

- `STRONG_LOCAL_SIGNAL`: summed squared error falls by **at least 50%** vs
  mass, in **at least four of five** layers individually, with no remaining
  layer worse by more than 20%. This licenses a **new, separately frozen**
  end-to-end BPB/rank protocol, not promotion or GPU training.
- `PARTIAL_LOCAL_SIGNAL`: summed error falls by at least 20% but the strong
  band fails. Report the pattern; no automatic end-to-end run.
- `NO_LOCAL_SIGNAL`: less than 20% summed-error reduction. This rejects this
  *greedy output-aware selector on this partition/sample*, not all
  output-aware criteria and not shared-low-rank-plus-residual operators.

Even `STRONG_LOCAL_SIGNAL` says nothing about downstream loss, free-running
rank, trained one-byte format, engine parity, target-scale transfer or rate.
Conversely, failure cannot close the shared-path hypothesis: a cheap learned
component `S(x)` changes the residual being selected. That would be a
different, joint operator and needs its own byte budget and controls.

## No-duplication anchors

Read `probes/E38_IS_THERE_ANY_SELECTOR_AT_ALL.md` for the actual mass result;
`probes/D0C_GRANULARITY.md` for the partition/null; `probes/E31_WHAT_A_GATHERED_BYTE_COSTS.md`
and E33 for locality limits; H2I/H5 for why local score gains do not imply
generation. Reuse their pinned data and labels; do not rerun E38's 26 BPB arms.
