# Next: transfer donor history functions into the original engine envelope

9 October 2026. Goal ACTIVE/INCOMPLETE. All jobs terminal; no T4 allocation.
[Complete original-operator recovery](ORIGINAL_FALCON_WHOLE_RECOVERY_RESULT_20261009.md)
supersedes the previous proposed24-step pass. Five of six native distribution
gates FAIL; original all-history numerical FAIL is retained.

## Actual resumption, not a fresh training restart

`results/native_expert_scaling/original_falcon_whole_recovery_20261009/candidate_25.pt`:
actual model/Adam25/CPU+CUDA RNG/ledger/completed FIT IDs,8,614,638,714B, SHA
`1a6f366e5df0d913f4fcffd205f2f32c3da844e4c0c4c08b66b8c29accdc8790`.
Actual packed507,505,920B, SHA
`98835350c753253b1e0a06e5525bf046153dae12aa51ff893a940fa1729b7969`.
717,877,248 masters/679,477,248 bank coefficients, D256/L6/H128/k8/n1152/V65537.
Do not reset Adam, replay completed updates/baselines or regenerate cached labels.
A different initialization/geometry is a separately frozen candidate with explicit
ancestry, not a continuation of Adam25.

Native48 before/after complete;24 new FIT updates actual1->25. FIT caseKL12.14269
->8.14321, DEV12.09313->9.01572 (25.4476% recovery), DEV disagreement96.5741%.
Both final mean KLs beat matched uniform; five absolute/domain/relative-disagreement
criteria fail. The readout free-feature mean feasible loss~.66174 is optimistic
and sampled: no history function/general D256 capacity conclusion. Actual-direction
sqrt8 scaling hurt the previous model. No further oracle/scale/same-recipe dose
selected merely because FIT drops. Finite dose failure is not an impossibility.

## Look at the whole compiler: the information losses are distinct

| Stage | Current approximation | Question required before expensive adaptation |
|---|---|---|
| Residual representation | source D2048 -> P256 | Which observable donor distinctions disappear under the projection? |
| History core | fresh five Mamba1 + one SWA; donor24 parallel SSM/attention blocks | Can pretrained recurrent/read-write modes be transported, and what history remains observable? |
| Composition | four donor FFNs represented in each target site | Can the four successive nonlinear state updates be approximated by one conditional update at this cost? |
| FFN function | source SiLU gate -> sign quadrants/mean-gate proxy, then ternary/AQ63 | What output-weighted residual remains on real reached states? |
| Conditional selection | normalized top8 over n1152 | Which omitted function residual depends on history or region and is recoverable by routing? |
| Readout | projected head and learned gamma | Can actual generated states reach the useful directions identified by the oracle? |

These effects are confounded by the whole recovery. Bank size/hash diversity does
not identify retained knowledge, and no single omission is experimentally proven
to be the dominant cause. The goal is a converter into the original LUT/ternary/
SSM machinery with bounded active cost; donor-sized runtime is not presently priced.

## Selected next investigation: source recurrent transport, with composition explicit

First complete algebra/operator/cost accounting against the locally pinned source
and original code. Then freeze one finite NEW internal-state capture only if it
supplies missing information for selecting an initializer or a geometry. Reuse
existing canonical FIT/DEV IDs and teacher output packets; no new RESERVED queries
or repeated donor answers. No source/model/native runs have yet been executed for
this next investigation. A protocol/binding/caps must precede any capture.

Primary source code already inspected: pinned local
`results/native_expert_scaling/chatbot_source_runtime/site/transformers/models/falcon_h1/modeling_falcon_h1.py`,
`FalconH1Mixer.torch_forward`, `FalconH1RMSNormGated`, decoder and MuP vector;
original `benchmarks/phase60/engine.c` recurrent loop and tensor learner match.
This code comparison supports algebraic possibilities, not empirical quality.

### A concrete state map, not a parameter-count argument

For donor head h/channel c, scalar a_h<0 is repeated across its256 state coordinates:

```text
s_t = exp(a_h * delta_t) s_(t-1) + delta_t * x_t * B_t
y_t = C_t^T s_t + D_h x_t
```

For orthonormal R in R^(256 x r), set z_t=R^T s_t, B'_t=R^T B_t,
C'_t=R^T C_t. Scalar decay commutes exactly with this projection. The retained
recurrence is the same algebra as original engine.c, with A[c,j]=a_h. The output
residual is `C_t^T (I-RR^T) s_t`, bounded by the product of omitted C/state norms.
This identity is conditional on exact x/B/C/delta inputs and initial-state mapping;
it does not make their nonlinear generators or whole chatbot exact. An arbitrary
projection across heads does not commute when head decays differ.

Need NEW reached B/C/state/output-energy information to pick R and donor channels,
not just SVD of coefficient matrices. Rank96 may be useful or insufficient; measure
output-weighted omitted modes, per-domain tails and propagated history error.
Do not transfer/reset a source state at each step and call that target free-running.

### Original B/C/delta generators have an identifiable representation cost

Donor B/C are direct in_proj -> depthwise conv4 -> SiLU channels. Original B/C
are linears of the original convolved/SiLU x channels. Auxiliary internal channels
can carry donor B/C features, while zero out_proj columns prevent direct residual
contribution. A coordinate subset of96 B and96 C needs192 such channels. A dense
R projection of all256 B/C needs512 source feature channels; projecting before
SiLU does not commute with SiLU. Count them before claiming a compact warm map.

Donor delta is softplus of an unconvolved affine projection plus bias. Real identity
`SiLU(t)-SiLU(-t)=t` permits paired auxiliary channels with identity conv, x_proj
recombination and original dt_proj/bias. Four selected heads need8 delta channels.
The original DT rank16 can represent four distinct delta functions; it cannot
represent arbitrary48 independent donor functions without compression. MuP/input/
output scalars, time-step clamp, biases and F32/approximate-sigmoid arithmetic must
be included. This is an untested construction, not numerical qualification.

Illustrative channel counts, not selected architectures:

- Original DN512:256 retained x channels (four donor heads),96+96 B/C coordinate
  channels,8 paired delta channels =456. Only a state coordinate subset is represented.
- DN1024:256 x, full256+256 B/C and8 delta =776; now a general rank96 state map
  is representable by x_proj. Wider core has real per-token compute/storage cost.

Unused/auxiliary scan channels still cost recurrence work in the unchanged engine.
Original auxiliary output columns can be zero; compute cannot be counted as zero
unless an actual qualified implementation skips it. These counts do not solve the
D2048->D256 residual projection, omitted x channels or omitted source layers.

### Two operator omissions cannot be hidden in constant weights

Falcon uses post-gate RMSNorm: the denominator depends on the current full3072-vector.
Original gate -> out_proj has no such norm. A constant folded scalar is not exact;
normalizing a retained subset does not recover the full denominator. Either measure
approximation error or price a small norm extension and its missing-energy estimator.

Every donor block runs SSM and RoPE attention in parallel on the same normalized
input, then FFN. Original alternates five SSM blocks/one non-RoPE SWA, and the present
four-to-one FFN grouping discards sequential composition. State transport alone
cannot fix this. Compare composed projected donor residuals and one-site residuals
on real states before selecting L6, more layers or a parallel mixer. Preserve
source order when a deeper variant is selected; do not substitute an average for
function composition. n can shrink per site as depth grows, but active work grows.

## Finite capture and decision requirements

Before source inference, specify exact NEW tensors/times/layers, canonical case IDs,
full histories, dtype, output bytes, source/internal code hooks and pre/post hashes.
Use a fixed FIT subset with DEV held for evaluating the chosen map; no map selection
on DEV. Streaming sufficient statistics can reduce storage, but output-weighted
cross-moments/tails and direct residual checks need an auditable definition.
Existing whole teacher logits supply the same global labels without regeneration.

A meaningful first comparison is scalar-decay projected recurrent outputs versus
original-state target outputs with independently reconstructed coefficients and
accumulated state, including gate/norm/output multipliers. Report how much of the
source block survives the representation before fitting. Local error informs a
candidate; it is not a necessary whole-chatbot theorem or a substitute for native
DEV/own-history quality. If composition/projection loss dominates, revise that map
before spending long compute on warm SSM alone.

Price original DN512 versus any wider/deeper candidate in complete matrix products,
scan states/exponentials, normalized routing, selected ternary bytes, fullV head,
packed bytes and training residency. Current fullV head alone16,777,472 products;
original small-V701.7/s cannot be inherited. Width/depth changes need an actual
native variant and useful same-artifact batch1 accepted-speed measurements.

Keep Adam25 and all failed measurements. Implement/freeze at most one candidate
justified by new source evidence; evaluate original native distributions and new
all-history numerical parity separately. Fresh canonical own-history generation/
tasks and useful>=50 accepted IDs/s remain required before quality/speed admission.

## RAM-driven n, routing and longer compute

Flat O(n) router remains. In exact real arithmetic, renormalizing selected softmax
weights cancels the global denominator; selected masses equal softmax over selected
logits. This can remove unnecessary all-n normalization, but cannot find top8
without a correct winner search. Exact or measured approximate structured CPU
selection and finite-precision mass parity must be qualified as n grows.
LUT integer arithmetic and active ternary work depend on selected expert shapes;
router, memory locality and physical DRAM can still depend on total n. Copied
experts do not qualify useful added capacity. Measure quality and latency with
actual newly useful functions, unchanged active budget and accepted tokens.

Current full-bank F32 masters/gradients/Adam require10.872GB on CPU. For10B/100B,
select a measured block/streamed optimization schedule or another justified
precision/residency strategy; current prototype is not that optimizer. T4 month+
is authorized in principle, but no allocation selected. Communicate concrete
reason, measured T4-compatible memory/throughput, budget/checkpoints/plateau stops
once the source-informed candidate earns longer adaptation. Additional donor
families/~10B/~100B remain actual variants to demonstrate; frozen Giga/Qwen
analysis is reusable, donor-adaptation work remains operationally frozen.
