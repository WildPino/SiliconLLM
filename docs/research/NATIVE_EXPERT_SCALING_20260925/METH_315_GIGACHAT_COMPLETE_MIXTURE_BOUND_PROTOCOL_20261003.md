# METH-315: complete mixture, shared256 and four regional128 outputs

Frozen before scientific observations, 2026-10-03. Changes310/311 target from independent-parent preservation to COMPLETE source MoE FFN on qualified314 archives. No new source collection/GPU/candidate training. Previous goal turn is progress:314 qualified all six source-target archives after preserved312/313 apparatus failures.

## Binding and all-state source controls

Pin314 raw SHAa46b9395df437442e113651c06d9ee8fc253c6731f9026deaf431a5fb01b30b5, all six actual archive SHA/size/shape/dtypes, coordinates and parent counts. Require all five314 source gates. Inputs/gates/IDs/parent/shared/full outputs fixed; no new route or source subset. Reconstruct every full target from parent outputs/gates/shared in originalFP64 sum/finalF32 and require BIT EXACT full_mixture_output; zero-output normalized error exactly1, full norms positive. Archives are anchor-conditioned, domains consumed, not untouched full-model quality.

## Fit-only fields and complete mixture estimate

- Shared output basis: top256 eigenvectors of uncentered FIT FULL-mixture output second moment per layer. Query: top16 eigenvectors of uncentered all unique FIT input second moment. Canonical signs as310. No centering, rescaling, test fitting or quantization. Orthonormality<=1e-10; fit projection/eigenvalue energy closure<=1e-8.
- For each64 source parent, retain every actual selected FIT input (known minimum10). Normalize16D query, ten spherical keys with fixed seed315000+layer*100+expert and310 deterministic farthest/Lloyd/tie/empty-key rules. No restarts. FIT keys assign test inputs. Stop on zero/nonfinite query or changed fit count<10; never pad copied keys. Include all ten children even empty.
- Remove common output projection from each original unweighted parent output. Fit one residual output SVD basis per region, width min128/numerical rank, tolerance eps64*max(shape)*largest singular value, no nullspace completion; empty/allzero region zero-width. Shared/residual orthogonality<=1e-8. Match each parent-global fit residual128 control under the SAME shared basis.
- Estimated COMPLETE target: optimistic shared projection of TRUE full output + SUM of four original positive source gates times each selected region's projection of TRUE parent residual output. Evaluate against full source mixture, not sum/median of parent errors. This may lose original shared output outside shared space; it is an initializer/projection rule, not optimal joint training. Coefficients require true outputs and are undeployable.

## Best joint span diagnostic, every state

For each state, take the UNION of its four selected FIT regional residual bases. Positive source gates do not change the column span when coefficients are unrestricted. Compute thin FP64 SVD (gesdd) with the same numerical-rank rule; common/residual span orthogonality<=1e-8, internal orthonormality<=1e-10. Project TRUE full-output residual into that span, add common projection implicitly, record minimum achievable row error. Require oracle residual norm<=contained estimate residual norm+1e-10*full norm EVERY state. This oracle uses true per-state coefficients; it does not fit bases on test or implement an input function. It bounds this fixed chosen field's expressiveness, not every joint-trained or supervised conditional model. LRU64 caches spans by sorted child IDs, preserving exact fields; no row sampling.

## Gates, costs and decisions

For each of SIX layer/domain cases: regional estimate median relativeL2<=1%, p95<=5% (linear percentile), aggregate retained energy>=99%; no>.5pp energy loss versus matched parent-global control. All1920 children require>=32fit and>=16test states. Known rare-parent/region exposure deficiencies remain FAILED; no silent waiver or synthetic function count. Record the same1%/5%/99% diagnostic oracle gates separately; never use them to override primary or exposure gates.

All37,381 rows, all selected parent/child/actual basis widths/hashes and all per-state primary/control/oracle errors retained. Six CPU math threads, no concurrent model/rate job, no GPU/source collection.20min wall/12GiB RSS checked between parent fits/each512 oracle states; maximum checked RSS not sampled peak. Expected minutes of algebra; individual BLAS/SVD not interruptible by that check. Preserve failure/partial results, no overwrite or post-count threshold/row change.

Failure closes this precise complete-mixture projection rule before student training. If fixed-field joint oracle also fails, coefficient-function training alone with unchanged fields cannot meet these local gates; output bases/router/coverage must change prospectively. A passing diagnostic does not license sparse/poorly covered children or failed native cost profiles. Realized nonlinear/shared functions, precision/causal whole quality/rate, useful RAM-scale n and cross-family/100B evidence remain missing. No general architecture rejection follows a local bound.
