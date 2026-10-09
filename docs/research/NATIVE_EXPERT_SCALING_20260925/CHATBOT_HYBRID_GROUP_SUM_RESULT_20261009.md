# Initial projected FFN groups: omission cannot be repaired by one scalar

9 October2026. TERMINAL/exit0,controlled local result. Goal ACTIVE/INCOMPLETE.
Freeze425d9c56edc09571090208f82a7e355a22733f41;
[26-input binding](chatbot_hybrid_group_sum_binding_20261009.json)
SHA6010c2df6be9730b0cb34b8be865ebff9b97b4466f194f8bf6f909c7d7c45ffb.
[Protocol](CHATBOT_HYBRID_GROUP_SUM_PROTOCOL_20261009.md),
[raw result](chatbot_hybrid_group_sum_result_20261009.json),
[terminal](chatbot_hybrid_group_sum_result_20261009.terminal.json).

## New variable and exact scope

Use all3132 stored old C ff_input vectors and INITIAL source-informed projected
FFN masters. This differs from the completed trained/ternary common-bank probe.
Evaluate all72 groups in F64 at each common operand;compare fullsum F,selected8
sum S and selected normalized mixture M under the initialized router. The two
36-group source halves see the same input here,not their original two-block
composition. No source/student recurrence,old whole C,optimizer or generation
replay. Six old prefixes,170 FIT/91 DEV operands per site;12 sites.

For any one nonzero M,the best unconstrained amplitude is

\[
c_*={\langle F,M\rangle\over\lVert M\rVert^2},\qquad
\min_c{\lVert F-cM\rVert^2\over\lVert F\rVert^2}
=1-{\langle F,M\rangle^2\over\lVert F\rVert^2\lVert M\rVert^2}.
\]

It is the component of F orthogonal to M. An input-dependent oracle amplitude
is more permissive than any fixed positive rescaling. It does not change the
direction of the selected mixture.

## Actual geometry

ALL1092 DEV responses fail the preregistered1% component RMS even under c_*.
The minimum error across all DEV/sites is49.7946%;site medians88.2472–94.2755%.
Decision AMPLITUDE_ONLY_REPAIR_INSUFFICIENT_ON_TESTED_OPERANDS.

| Site | FIT-optimal fixed scalar | DEV median RMS under per-operand oracle |
|---|---:|---:|
|0|8.871691|88.2472%|
|1|8.640628|92.1426%|
|2|8.006310|91.6768%|
|3|5.351717|91.8986%|
|4|8.051080|91.9919%|
|5|6.257749|91.9851%|
|6|4.490268|91.3645%|
|7|6.649034|91.4867%|
|8|4.632948|92.9368%|
|9|4.286053|94.2248%|
|10|4.666469|94.2755%|
|11|3.673240|93.1322%|

Unscaled mixture has about98% median relative error at each site. Multiplying
by72 gives269–372% site-median DEV errors. Fixed FIT scalars also fail every
DEV row. All vectors,metrics and group energies finite/positive. Per-split
72x72 Gram identity sum(C)=total||F||² passes fixed relative1e-10. Raw F/S/M
vectors and Gram arrays retained with hashes and row/selection metadata.

These results rule out amplitude-only reconciliation of this initialized
selection on these operands. They do not rule out better routing,recombining
atoms,wider/overlapping learned functions,shared-plus-private decomposition or
whole adaptation. They do not measure the new trained checkpoint286 FFN error,
source hidden-state distribution,full source trajectory or chatbot quality.

## Resources and implication for conversion

Family12.391s,worker9.484s,OS peak946,044,928B through exit,
GPU allocated428,482,560B/reserved488,636,416B. All resource/input gates PASS,
no faults/T4/whole source or student calls. Session87486 closed exit0. Args:

```
--binding docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_hybrid_group_sum_binding_20261009.json
--binding-sha 6010c2df6be9730b0cb34b8be865ebff9b97b4466f194f8bf6f909c7d7c45ffb
--freeze 425d9c56edc09571090208f82a7e355a22733f41
--directory results/native_expert_scaling/chatbot_hybrid_group_sum_20261009
--out docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_hybrid_group_sum_result_20261009.json
```

The initialized branch contents/selection need a functional adaptation,not just
larger output scales. Next construction separates common response from private
selected functions,with explicit source-function/data and whole-output recovery
stages. Overlap must cover required contributions at bounded active cost;simple
duplicates remain insufficient. [Next](CHATBOT_HYBRID_SHARED_PRIVATE_NEXT_20261009.md).
No native threshold relaxation or useful large-model capacity/rate promotion.
