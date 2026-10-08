# First coupled-affine design: prospective budget and numerical screen

8 October2026. New implementation of the FIRST prerequisite in
[coupled-jet NEXT](CHATBOT_COUPLED_JET_NEXT_20261008.md). Freeze this protocol,
actual code and used-input/runtime binding BEFORE the first new design values.
This is a changed function class; no learned bank or chatbot admission is implied.

## Uncertainty, reused information, decision and stops

Can shared512 plus top4 full896x896 affine functions meet the necessary COMPLETE
cost budget, and can the fixed16 original FIT jet constraints determine their
coefficients without severe numerical instability? Prior value+J learner fails
response fidelity; old227 independent hard-nearest tangents remain closed.
The new class preserves full input/output rank and jointly couples selected mass.

Reuse original dimension ledger/spec, byte-qualified original16 FIT anchors,
P/centers, original F64 shadow selected mass and qualified Q. No new source/model/
shared value, J, choice, initializer, PCA, fitted prediction, control or optimizer.
Development anchors/labels never enter the design. The old original plan's
revision-label error/correction remains explicit and pinned, no plan replay.

Exact integer cost gates FIRST: for E16 and E160 storage/cost projection,
5*new complete matrix MAC <=3*source MAC AND 5*new logical coefficient bytes
<=3*source logical bytes. Source head, attention projections/context and F32
cache retained. All proposed BF16 matrices, F32 active/stored affine biases,
centroid norm scalars and router matrix products priced. Scalar work remains
additional work, not a throughput claim. E160 support FAIL is not reopened.

If costs pass, form mu=mean16 x; s_j=RMS_i((x_i-mu)^T Q_j). Require all s>1e-12.
Use centred z=(x-mu)^T Q/s, and only original selected ids and F64 shadow mass.
For each selected e at i, grad w_e=2*w_e*(c_e-sum_f w_f*c_f)^T P,
gamma_e=s*(grad w_e Q). In original parent order W[i,e]=w_e, zero otherwise.
Each 33x33 K[i,e] block is:

    row0:      W_ie * [1,z_i^T]
    rows1..32: gamma_ie * [1,z_i^T] + W_ie*[0,I32]

K is528x528. This is the algebra of the proposed coupled values/chart derivatives;
no RHS, source/core residual, coefficient solve or affine function is evaluated.

F64 NumPy SVD, full U/s/Vt: numerical rank at eps64*max(shape)*sigma_max,
require rank W16/K528 and condition2(W),condition2(K)<=1e6. These are numerical
screens, not exact-rank or error-preservation theorems. Procedure requires
finite C-order F64 arrays, relative SVD reconstruction<=1e-12 and U/V Frobenius
orthogonality errors<=1e-10. Save W/K, mu/s/z, selected full-input gradients/
chart gradients and all SVD factors for independent assembly/certificate audit.
Only four gradient entries/anchor are stored; missing weights/gradients are zero.

Any failed eligibility screen closes THIS fixed16-anchor construction before
coefficient bank. No post-observation temperature/chart/regularizer/rank/anchor
retry. A pass permits ONE separately frozen compiler; it does not admit it.
Keep original novel1%/ALL-category3%, actual native parity, FRESH chatbot quality
AND accepted50 SAME artifact/useful n/LUT winner+mass/DRAM/family/scale gates.

## Actual implementation and resource envelope

Worker `benchmarks/native_expert_scaling/chatbot_coupled_design.py`; binder
`chatbot_coupled_binding.py`; reused held-handle `chatbot_directional_launch.py`.
Python3.12.10 isolated/no site/no pycache, NumPy2.4.6/psutil7.2.2 through five
restricted roots; no Torch/GPU. OpenBLAS1, worker CPU0..10, launcher CPU11.
Worker60s/family120s, conservative summed OS peaks512MiB through actual worker
exit, all saved arrays8MiB/log2MiB. Inputs rehashed before/after; foreign files
and empty pycache preserved. No model download/install/T4. Actual costs retained.

New independent audit must reconstruct whole cost and K assembly through a
different formula, verify saved SVD factors and numerical decisions, and close
actual instances/OS faults. Retain first partials/faults; no scientific restart
merely because a observation wait returns. Output namespace
`results/native_expert_scaling/chatbot_coupled_design_20261008`; raw/binding/
terminal/log in this document directory. No current result exists at freeze.
