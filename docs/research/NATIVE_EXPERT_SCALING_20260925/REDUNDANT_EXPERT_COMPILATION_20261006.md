# Redundant conditional compilation: decision and algebra

6 October 2026. Proposal/deductions before METH500; no empirical claim here.
User steering permits more stored parameters to preserve a donor function with
less active work. This agrees with the goal's capacity/cost separation.

## Decision

499 is terminal and independently admitted: encoding improves, fitted consumed
quality and winner/mass fail. Its completed work remains retained. The proposed
pooled-rank/source-query500 had no implementation, binding, observations or job.
Defer that potentially large label acquisition and use the next free500 namespace
for ONE source-anchored overlap feasibility screen. This changes representation;
it is not another optimizer run on499's fixed function class.

## Exact ideal condition and accounting

For bias-free ReLU, F_e(x)=sum_j v_ej ReLU(w_ej^T x). Child c stores S_ec and
computes that same sum over its stored atoms. If every nonzero required term is
included in the selected child, the real output is exact. Otherwise omitted
r_ec(x)=sum_(j notin S_ec) v_ej ReLU(w_ej^T x) must be measured as a VECTOR:
signed output terms can cancel. Coverage of the union of all children alone
does not establish coverage of the selected child.

Stored copies / distinct FFN parameters =rho=sum_c|S_ec|/H. Selected FFN MAC
fraction =a=|S_ec(x)|/H. Four children ofH/2 give rho2/a.5; eight ofH/4 give
rho2/a.25 only if suitable coverage and cheap selection are possible. For
D768/H3072, the four-child example is4,718,592->2,359,296 selected FFN MACs.
These exclude selection, quantizers, accesses, LUT and the rest of the model.
P_new=P_fixed+rho*P_FFN: twice the FFN does not imply twice the whole model.
Duplicated parameters add no distinct donor knowledge; conditional access can
still become useful. Parameters, encoded bytes and physical DRAM are different.

In the actual I8/A16 source, original WI rows and WO column codes/scales must
retain their numerical contracts. Keeping all nonzero original hidden A16 codes
also keeps a positive maximum whenever hidden activity is nonzero. Original
hidden scale can then be reproduced, and removed integer WO terms are zero.
500 checks actual output bytes; it also measures branches missing the maximum,
where a new A16 scale introduces another error. Literal copies do not yet
establish a convenient ternary/LUT representation.

## Parent probability and geometry

Preserve the original contribution p_e F_e as p_e Fhat_e,c(e,x). The child is
an internal representation selector. Adding copies as ordinary flat-softmax
competitors changes Z to sum_e m_e exp(z_e), so it changes parent mass.
Conditional child masses q_ec summing1 can preserve parent mass with logits
z_e+log(q_ec), but choosing just one branch with weight p_e*q_ec attenuates
its output unless q is one-hot or the construction compensates explicitly.
Original-parent choice/normalization remains open;500 holds it fixed locally.

On each fixed ReLU sign cell the real function is V diag(s) W x. Overlap or
private regional representations can avoid asking one compact global dictionary
to approximate every cell. Region count and selector boundaries can themselves
be expensive.499's53,943 unidentified coefficient directions apply to its fixed
global feature class; they are not a bound against a changed representation.

## Evidence and next test

453 proves zero-only WO omission locally, after approximate compact WI; its
7.2401% hidden density is not an original-source/all-domain overlap guarantee.
431's additive extra-parent recipe and former rank/format failures retain their
defined closures.477-R1 requires returning to head/composed/fresh quality.

Related primary work: [MoEfication](https://arxiv.org/abs/2110.01786) exploits
pretrained FFN activation sparsity and partitions/router learning.
[ToMoE](https://arxiv.org/html/2501.15316v1) learns subset masks, coverage and
active budgets while freezing source weights; its mask formulation permits
overlap (inference from the formulation). [Sparse Upcycling](https://arxiv.org/html/2212.05055v2)
initializes larger expert pools with copied MLPs and continues training.
These motivate investigation, without proving immediate donor-equivalent
quality or our CPU/LUT/rate target.

500 separates source support, best-child oracle, cheap input-only selection,
mask omission and native quantization. A local pass would justify a separately
priced native/fresh inquiry. A failure closes this four-half-width recipe and
identifies the next uncertainty. No large grids, full-model reruns or new GPU.

[Frozen500 protocol](METH_500_OVERLAPPING_SOURCE_PROTOCOL_20261006.md).
