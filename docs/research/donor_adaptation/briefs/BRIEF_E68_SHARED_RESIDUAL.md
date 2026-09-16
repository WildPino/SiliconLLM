# E68 — can a cheap shared low-rank path carry what three FFN groups omit?

**Status: preregistered design, 2026-09-16; no runner or result yet.**
CPU-only; no T4, model conversion, engine change or timing authorized by this
brief. Freeze before the instrument exists, then preserve the result regardless
of sign. This tests one *linear* shared path on a real 1.5B donor; it does not
stand in for a trained nonlinear shared network or a 10B deployment.

## Decision and relation to earlier cells

E38's mass-ranked carve at `k=3/256` is unusable in end-to-end BPB; E67's
output-aware greedy criterion reduces the frozen local FFN-output SSE by only
`4.2285%`. These reject two ways of choosing **the same 105 surviving
neurons**, not a different operator with a shared bypass. The new question is:

> Is the contribution of omitted FFN neurons predictable from the layer input
> by a sufficiently cheap, always-on rank-`r` linear map?

For each layer, let `x` be the input to its MLP, `z=SiLU(W_gate x)*W_up x`,
`y=W_down z`, and `a(x)` the same E67 float64-mass top-three groups. Let
`y_sparse=sum_{g in a(x)} W_down[:,g] z_g`. Fit a rank-`r` map `S_r(x)=x B_r`
to **the calibration residual** `y-y_sparse`, then evaluate
`yhat=y_sparse+S_r(x)` on a disjoint held-out slice. It is a diagnostic of
the proposed *shared plus block-sparse residual* idea, not an executable
router: obtaining `a(x)` still uses the dense `z` oracle. A rank-`r` map
always reads both factors per token unless they reside in cache; no "read
once" discount is assumed.

## Frozen object, split and ranks

- Donor Qwen2.5-1.5B, revision
  `8faed761d45a263340a0528343f099c05c9a4323`, fp32/eager, untouched.
  Layers `[1,7,14,21,27]`; D0c's existing 256 equal groups of 35.
- Fit: existing `common.get_slice(tok,"calib",8,512,424242)`, **4,096
  positions**, `ids_sha256=92fc41840c8feb0595758e540a048c1661bf3f618b3355d637ecf418b6c00157`.
  This is disjoint from the held-out corpus; store the fit hashes, never its
  activations in the result JSON.
- Score: **exact E67 positions** — sequences 0–3, positions 384–511 from
  `common.get_slice(tok,"heldout",24,512,1234)`, 512 positions/layer, full
  slice `ids_sha256=a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`.
  E67's rank-zero local SSE values are pinned anchors, not tuning targets.
- Primary `r=64`; secondary `r=32,128` from **the same one fit**, reported
  as a curve without checkpoint/rank selection. Rank zero is the sparse-only
  control. A separate rank-64 **shared-only** arm fits `y` (not the residual)
  on the same calibration split and omits all selected groups at score time.

## Frozen fit — no optimizer or hyperparameter search

Represent calibration inputs and targets as row matrices `X∈R^{N×D}` and
`R∈R^{N×D}`. Fit the ridge reduced-rank regression

```text
min_{rank(B)≤r} ||R-XB||_F² + λ||B||_F²
λ = 0.001 × trace(XᵀX)/D
C = XᵀX + λI = LLᵀ                 (Cholesky)
M = L⁻¹ XᵀR = UΣVᵀ
B_r = L⁻ᵀ U[:, :r] Σ[:r] V[:, :r]ᵀ
```

For inference-accounting purposes the two factors are
`A_r=L⁻ᵀ U_r Σ_r` and `V_rᵀ`, so `S_r(x)=(x A_r) V_rᵀ`.
Fit the shared-only comparator with the same `X`, `λ`, algorithm and rank,
targeting `y` instead of `R`. Use float64 accumulations/linear algebra for
the fit, fp32 donor forward, and explicitly record any conversion applied
to factors at score time. No bias or intercept, no per-layer λ search, no
gradient steps, no held-out selection of ranks.

At the 1.5B diagnostic width `D=1536`, rank 64 has `2Dr=196,608`
factor weights/layer. Hypothetically at half a byte each this is **98,304
bytes/layer**, before scales/layout, versus `3×D×105/2=241,920` selected
FFN bytes/layer; the shared path adds 40.6% to the selected FFN payload.
At E36's synthetic A10B geometry (`D=4096`, `L=16`) the same rank would be
`8,388,608` factor weights/token, 4,194,304 packed bytes/token, **0.81%**
of E63's measured mixed `517,210,112` charged bytes/token if all else stayed
unchanged. This is arithmetic **only**: no 10B quality, format compatibility,
cache residency, kernel parity or tok/s is inherited.

## Required controls and outputs

1. Run E67's planted mass/tie and reconstruction controls (or equivalent
   independently checked ones) before interpreting any donor score. Every
   sparse arm selects 3×35 neurons, and the `r=0` held-out SSE at each of the
   five layers must reproduce E67's full JSON to relative `1e-5`.
2. Verify fit/score slice hashes, model revision and labels hash. No fit
   activation or residual may come from the held-out split. Record source
   hashes and factor shapes, finite values, rank, λ and Cholesky/SVD success.
3. Independently verify `B_r=A_r V_rᵀ` on a planted small matrix and that
   ridge objective is non-increasing as rank rises on calibration data
   (floating tolerance declared in the runner). Check `r=0` is exactly
   sparse-only.
4. For each layer/rank/arm report held-out
   `sum||y-yhat||²/sum||y||²`, per-sequence values and per-token median, plus
   calibration error as an overfit diagnostic. Report layer results and
   aggregate SSE ratios; do not average layer percentages to make a verdict.
5. Keep full-precision outputs in JSON and report peak RSS/time only as
   operational observations, not model rate.

## Predeclared verdict (primary rank 64 only)

- `COMPLEMENTARY_SHARED_SIGNAL` iff the combined arm cuts **aggregate SSE by
  ≥50%** against E67 mass-only, cuts it by **≥20%** against the separately
  fitted rank-64 shared-only arm, improves mass-only by ≥30% in at least
  four of five layers, and is no worse than mass-only by >5% in any layer.
- `SHARED_ONLY_SIGNAL` iff the shared-only arm cuts aggregate SSE by ≥50%
  against mass-only but `COMPLEMENTARY_SHARED_SIGNAL` does not fire. This
  points to a low-rank substitution, not sparse complementarity.
- `PARTIAL_LOCAL_SIGNAL` iff combined rank-64 cuts aggregate SSE by ≥20%
  against mass-only but neither stronger band fires.
- Otherwise `NO_LOCAL_SIGNAL` for **this linear residual fit**.

Even a positive is *local* and only licenses a separately frozen full-model
BPB/rank and exact byte/engine protocol. A negative cannot refute a jointly
trained nonlinear shared path, output-aware residual selection, different
rank/partition or target-width behavior. There is no automatic T4 launch.
