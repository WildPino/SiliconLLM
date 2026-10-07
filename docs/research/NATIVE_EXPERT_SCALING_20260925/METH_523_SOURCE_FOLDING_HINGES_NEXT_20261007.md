# Selected next523: source folding with explicit ReLU hinges

7 October2026. PROPOSED; algebra and cost ledger below are deductions.
No523 protocol, binding, numerical observation, export, fit or process.
Goal ACTIVE/INCOMPLETE; [522](METH_522_REDUNDANCY_LOWER_BOUND_RESULT_20261007.md)
is terminal/admitted. This note supersedes the operational next522.

## Problem to change

522's width obstruction counts original neurons, although many neurons can
contribute through the same linear map inside a ReLU region. Carry the full
effect of those neurons in a folded matrix, and keep a bounded set of original
ReLU hinges explicit. Copying these source-derived functions trades stored RAM
for fixed active work; it does not add donor intelligence by duplication.

This changes the FEATURES and source transformation from499/521's shared
rank512 dictionary and constrained readout. It also changes500..503's functions:
their omitted atoms contribute zero, whereas this proposal folds their locally
linear contribution. Preserve those closed recipes and thresholds.

## Exact real-arithmetic identity

For the qualified Switch ReLU expert, let I=diag(s_I) WIcode, O=diag(s_O) WOcode,
and xhat=alpha_x q_x. Define the ideal continuous source f(xhat)=O ReLU(I xhat).
Choose a reference sign vector s and an explicit hinge set U of at most512 rows:

    L = sum(j outside U with s_j=1) O[:,j] I[j,:]
    g(xhat) = L xhat + O[:,U] ReLU(I[U,:] xhat)
    f(xhat)-g(xhat)
        = sum(j outside U) O[:,j] (ReLU(z_j)-s_j z_j), z=I xhat.

If every omitted row keeps its reference sign, the residual is exactly zero,
regardless of how many positive source neurons are folded. There is no512
limit on the number of source neurons contributing to L. For a sign change,
the residual term is determined by the crossed margin and original WO column;
neuron count or Hamming distance alone does not determine output error.

At U empty this is a full-input region matrix; at U all rows it is the full
continuous source. The512 hinge proposal retains nonlinear degrees of freedom
between these extremes. It is an exact identity on the stated region, not an
exact identity with the original F32/A16 native function or outside that region.
Constant and zero-input cases need explicit arithmetic controls.

## Three independent errors to account for

For selected branch b, physical candidate G_b and original native F_src:

    G_b-F_src = (G_b-g_b) + (g_b-f) + (f-F_src).

1. Stored matrix, input/hidden codecs and accumulation give G_b-g_b.
2. Omitted sign changes under the actual selector give g_b-f.
3. Original WI F32 casts, the global hidden maximum/A16 round and final WO
   cast give f-F_src. Folding removes that global hidden quantizer from the
   folded part, so source BYTE identity cannot be inherited.

Keep vector terms and their cross inner products; their squared energies are
not additive. Bound roundoff before comparisons as in521's metric repair.
Separate the continuous reference, decoded candidate and physical evaluator.
Use saved original unweighted outputs directly; never divide rounded pF by p.
An original hidden q_j=0 is not a negative-sign certificate: a small positive
preactivation can round to zero. True reference signs require integer WI dots.

## Prior evidence and restrictions

- [222](METH_222_CONDITIONAL_FUNCTION_RESULT_20261001.md): fitted affine/PCA64
  hierarchy worsened validation when count grew. Full source folding and private
  original hinges replace that fitted low-dimensional residual mechanism.
- [226](METH_226_DONOR_TANGENT_NATIVE_RESULT_20261001.md)/[227](METH_227_CONDITIONAL_DONOR_TANGENT_RESULT_20261001.md): Qwen smooth tangents
  have qualified local C arithmetic but failed absolute function quality.
  Switch ReLU has exact linear cells; neither Qwen's cost nor accuracy carries.
- [231](METH_231_SOURCE_DERIVATIVE_CHILD_RESULT_20261001.md): source derivatives
  with inherited nonlinear features still failed. New private source hinges
  must actually remove the omitted sign residual, not merely match a center.
- [252](METH_252_PRIVATE_FEATURE_NATIVE_RESULT_20261001.md)/[270](METH_270_PRIVATE_SOURCE_ROWS_RESULT_20261002.md): source features can improve
  fidelity, but those Qwen selectors/functions and native cost recipes closed.
  No generic source-feature success or transferred timing is inferred.
- [504](METH_504_ANGULAR_REGION_RESULT_20261006.md): isotropic source cones had
  insufficient coverage. A cone around the same centers with fewer uncertain
  negative rows cannot be assumed larger. Do not repeat a cone-size grid.
- [521](METH_521_SOURCE_INFORMATION_TRANSFER_RESULT_20261007.md): unique global
  counterfactual interpolation failed even in F64. No additional bits, old
  labels or constraints repair that stopped recipe. Source folding is different.

## Price before another function experiment

At D768/H3072 and B512, ignoring scales/indices/headers/core/head/router:

| Active representation | Coefficient bytes | MACs |
| --- | ---: | ---: |
| Original I8 WI and WO |4718592 |4718592 |
| Folded I16 L only |1179648 |589824 |
| Folded I16 L plus512 I8 source hinges |1966080 |1376256 |
| Folded F32 L plus512 I8 source hinges |3145728 |1376256 |

The I16 hybrid has5/12 source coefficient bytes and7/24 source MACs. I16 dots
need qualified wide accumulation, so MAC ratios are not latency ratios. A full
I16 row can sum768*32767^2, aboveI32; AVX lane/pair overflow must be handled.
Stored row scales, original source scales, hinge IDs, gathers, output blending,
all active router bytes and uncached DRAM reads must be charged separately.
No native kernel, memory measurement or token rate exists for this hybrid.

ONE hybrid per127 exposed parents costs249692160 coefficient bytes. Four per
parent cost998768640, before controls or unexposed-parent treatment. A one-bank
count is not a whole-model RAM estimate. If every3072 atom is explicit somewhere,
that coverage/storage requirement is additional; folded source knowledge is
represented in different coordinates and does not inherit522's incidence bound.

Full development WI negative-margin recovery costs11721*3072*768 integer MACs;
all17540 UID recovery costs41382051840. Do not recompute old native source WO
outputs. The NEW discarded negative preactivations, continuous shadows and
new candidate functions must each be counted explicitly if acquired. Fold
construction costs up toD^2*H per reference pattern; no per-token free folding.

## First action and proposed stop sequence

Recover exact380 tensor extents/scales and admitted499 input/source-output,
500 hidden/region and504 integer-dot contracts. Price fresh payload verification,
one development reference per parent, signed-margin storage, fold construction,
I16 encoding, candidate evaluation and a genuinely independent audit. Source
protocols must commit before a binding and before any523 observation.

The first numerical question should be the SINGLE-PARENT-FUNCTION hybrid,
ALL127 exposed parents, B512, one development-derived reference per parent.
Select explicit hinges using development sign-residual contribution energy
times original WO-column energy; ties by original neuron ID. This differs from
old magnitude/hidden-energy row selection. Reference rule, exact codec and all
thresholds still require a complete protocol; no data-dependent selection yet.

The intended controls are full folded map with U empty, continuous512-hinge
function, encoded512-hinge function, and physical arithmetic. Keep original
parent ID and normalized source p as declared controls, with their full cost.
ALL dev/consumed/teacher/natural/rare/book denominators retained. Consumed
diagnostics are not fresh quality. No minority expert or failing row excluded.

First screen must establish source information in the candidate, separate sign/
codec/source-quantizer residuals and meet the existing1% local RMS standard.
Failing encoded fidelity closes that frozen hybrid, before count growth or C
whole-model trials. Passing only permits a separately frozen matched one-versus-
multiple-branch test with equal active shape, rotated-choice control and original
mass accounting. Source-parent selection alone does not solve CPU LUT at large n.

Proposed finite envelope to refine from exact contracts: main<=15min/1.5GiB,
independent audit<=20min/1.5GiB, combined new outputs<=2GiB, CPU10/one numerical
thread, no model/corpus/T4. These are proposed ceilings, not measured costs or
authorization to run before a complete feasible protocol. Preserve first faults
and do not replay old completed source/main/audit responses for convenience.

If the proof/price cannot justify this screen, record the obstruction and select
another representation. No spherical-region, row-count, precision or seed grid
is licensed by this note. Whole fresh donor-relative quality AND SAME>=50/s,
compact core, useful RAM-dependent much larger n, winner AND mass, physical DRAM
and actual family/scale variants remain subsequent independent requirements.
