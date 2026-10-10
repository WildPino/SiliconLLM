# Structured routing: reusable evidence and the conversion condition

10 October2026. Existing-state investigation during the actual51 stored audit.
No new timing, predictions, fitting, compiled binary or inference. Goal INCOMPLETE.
This maps the [paper follow-up](LITERATURE_CONVERSION_FOLLOWUP_20261010.md) to
existing project implementations so a product-key prototype is not rebuilt.

## Existing work that actually addresses parts of the requirement

| Existing record | Historical measured result | Scope that remains missing |
|---|---|---|
| [METH-53](METH_53_PRODUCT_KEY_ROUTER_RESULT_20260927.md) | Exact factorized top4 oracle;273408 synthetic pairs,2.7223ms/six threads;10x ratio4.800 fails its frozen scaling gate | Learned-router fidelity, useful experts, normalized engine mass, full model, DRAM |
| [METH-54](METH_54_R64_PRODUCT_KEY_RESULT_20260927.md) | Rank64 query/factorized keys;273408 synthetic pairs,0.440490625ms/six threads;10x ratio1.39210259 passes its own cost gates | Same missing learned/useful/engine/DRAM scope; key pool fits a small working set |
| [METH-58](METH_58_PRODUCT_KEY_NATIVE_COMPONENT_RESULT_20260927.md) | Actual learned Qwen adapter E128;96 fixtures, exact unordered top4 sets and BF16 gates/residuals | Original compact core/LUT integration, ordered tie rules/current F32 mass, large useful n and own-history whole quality |
| [METH-104](METH_104_HIERARCHICAL_CPU_ROUTE_RESULT_20260927.md) | Learned hierarchical E1280;96 fixtures, parent/child IDs and BF16 gates exact;1.476525ms vs0.979333ms parent route | Hot-input FP32 route only; no selected factors/DRAM/full model; METH-105 semantic failure prevents promotion |

Current METH-53 source raw SHA is
56bff10df1260ea70ebdcf211e8745c830ca62534e429f149697aa2e4708f322;
METH-54 is
c73f955b95e9f1870092d873b374ac12450392ef72f72d52ff07f461f7d5791d.
Both match their retained report/summary. METH-58 current source SHA is
ff3311298bf4b1cd53240971ed5bebbf98f82718868e7e11bee2722097f44238;
METH-104 is
c657da1ffd963c611951e555f38d55247fdc4736097103abbb5a25197124b373.
The current METH-54 JSON confirms its raw timing and gate arithmetic. This
review does not rerun historical binaries or independently requalify every old
fixture/bank/runtime extent. Historical component timings remain separate
artifacts; they cannot be added to the actual51 decode rate as a measured model.

## Exact conversion requires additive structure, not just a fast search

In column convention, reshape an existing flat router into n=A*B rows
w_ij with bias b_ij. An exact two-axis additive representation exists iff
all rectangular differences vanish:

    w_ij - w_i0 - w_0j + w_00 = 0
    b_ij - b_i0 - b_0j + b_00 = 0.

Necessity follows by substituting w_ij=u_i+v_j. Sufficiency follows by choosing
u_i=w_i0 and v_j=w_0j-w_00, with the same construction for biases. A shared
query projection of rank r further restricts the key differences to its
reachable row span. The current learned original flat router has not been
shown to satisfy either condition. Exact search within a product-key family
therefore does not establish exact replacement of that flat router.

For a fixed grid and uniform row weights, the least-squares additive fit is
the row mean plus column mean minus grand mean. Its residual is the interaction
component of this grid. This fit optimizes router coefficients under that
declared weighting, not chatbot loss, activated-expert usefulness or arbitrary
hidden-state score error. Reassigning expert IDs changes the grid and is an
additional search/learning problem. No such fit or rearrangement is run here.

For a particular input x, suppose every fitted score differs from the original
by at most epsilon. If the original kth versus(k+1)th score gap is strictly
greater than2epsilon, the topk set is preserved. Ordered IDs need the relevant
pairwise gaps too; exact ties require the same declared lexicographic rule.
Without measured margins this sufficient condition grants no current parity.

On the same selected set, normalize the selected scores with softmax. If the
score perturbations are bounded by epsilon, each fitted/original mass ratio
lies in [exp(-2epsilon),exp(2epsilon)]. Preserving IDs alone does not preserve
the mixture coefficients. A temperature tau changes epsilon to epsilon/tau.
The current contract requires both IDs and normalized mass, with finite
precision and actual engine accumulation assessed separately.

## What is reusable, and what changes the next action

Reuse the existing C search/key-scan implementations, rank64 cost evidence and
learned component exporters as starting points for a qualified future route.
Do not repeat a synthetic product-key demonstration as if it supplies useful
large-n conversion. The scientific missing step is a useful function bank
whose decoder/core and structured scores were coupled during construction or
adaptation, followed by original-engine numerical/own-history quality and
selected-factor DRAM/cost measurement on the same artifact.

The literature and current review select no immediate router replacement or
new training run. Close actual51 adjudication and run the fixed target/readout
control first. Its result determines which representation/function path can
benefit from structured retrieval. METH-58/104's donor-sized Qwen core is not
being reactivated as a substitute for the original compact-engine objective.
