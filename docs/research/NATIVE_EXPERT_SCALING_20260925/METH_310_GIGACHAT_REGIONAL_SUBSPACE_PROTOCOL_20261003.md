# METH-310: input regions and full nonlinear-output subspace bound

Prospective protocol, frozen before scientific observations, 2026-10-03.

## Question, reused evidence and changed variable

Can ten input-selected regions per source parent support rank128 full nonlinear-output representations? This tests the proposed narrow child's output column-space restriction, using actual GigaChat captured FFN states rather than synthetic weights or original-channel omission. Unlike298 global linear projection rank192, partition by input regions before fitting full nonlinear-output subspaces. Unlike299, every original channel contributes to the reference; no channel subset is omitted. No training or promotion of failed CPU profiles303–309 is licensed.

Reuse every qualified298 fit/test row for layers1/13/25, source experts0/32/63, 106 English/code/technical fit chunks and125 Cyrillic test chunks. Both domains were previously consumed; this is a local diagnosis, not new independent full-model quality. Pin298 capture records/raw hashes and180 tensor/config/index bindings; require all27 streams, finite coordinates/footer, gate/up input equality and gate/up/down coordinate alignment. Actual BF16 down tensor SHA must match180. Full output Y=post-SwiGLU Z times original down transpose in FP64. Split-640-column/full-source closure <=1e-12 each row. This mathematical reference does not assert source-HF or native bit equivalence.

## Fixed routing and rank

- Shared query per layer: top16 eigenvectors of uncentered sum Xfit.T Xfit, pooling all three parents, in FP64. No centering/intercept. Canonical eigenvector signs by positive largest absolute coefficient. Fit-only data; query is unquantized and not fitted to held-out outputs.
- Normalize each16D projected row to unit length; stop on zero/nonfinite query. Each parent has ten spherical kmeans keys. Seed310000+layer*100+expert, default_rng first row then deterministic farthest-point initialization (minimum maximum dot, lowest row ties). At most100 Lloyd iterations; argmax dot uses lowest child ties; normalized centroid update, empty/zero-centroid keeps prior key. Fit assignment stability stops iterations. Reassign final keys; held-out uses only these keys/query. No restarts or test-dependent selection.
- Each region fits an uncentered FP64 output SVD subspace of width min128/numerical rank. Numerical tolerance eps64*max(shape)*largest singular value. Empty fit region has zero-width basis. Predict using projection of true output into that basis, an optimistic bound: actual child coefficients/function realization are not fitted.
- No basis completion with arbitrary nullspace vectors. Record actual widths/counts, keys/query/basis hashes, every row error and all90 child exposures. Test-empty region remains in exposure gate; no favorable subset.
- Global fit rank128 output basis is the same-parent control. Held-out per-region rank128 SVD is an undeployable diagnostic ceiling, without changing router. Finite test-region counts below128 can trivially give100% oracle energy; never call this population proof.
- Orthonormality max abs error <=1e-10, fit projection/SVD energy difference <=1e-8, routed test energy <= corresponding test oracle+1e-8. Source output norm strictly positive.

## Gates and resources

All nine parents must have test relative-L2 median<=1%, p95<=5% (linear percentile), aggregate retained output energy>=99%, and no regional retained-energy loss >0.5 percentage points versus global fit-rank128. Every one of90 children must have>=32 fit and>=16 test rows. No post-observation threshold/rank/router/precision change or source omission.

Six CPU math threads, no concurrent model/rate job, no GPU/download/new source capture. Stop at20min wall or12GiB RSS, preserve partial failure. Existing captures ~674MB; expected minutes of local SVD algebra. Reuse before expensive collection. Script checks budget between regions; an individual BLAS operation is not interruptible by that check.

If gates fail, close this exact PCA16/spherical10/fit-rank128 conversion rule, retain global/held-out diagnostics to identify representation versus fit/generalization limits. Do not generalize to every trained regional function or other donors. If all pass, only separately frozen new cost/precision/realized-function checks become justified. Stable complete CPU cost, useful child contributions and independent composed-model quality/rate on the same artifact remain mandatory. No existing native profile becomes qualified by this result.
