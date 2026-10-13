# Next capacity hypothesis: full-size affine experts for nonlinear source FFNs

13 October2026. **AFFINE BANK PROPOSED/UNIMPLEMENTED;OPERAND PRODUCER COMPLETE/AUDIT RUNNING.**
[Bound collector/state](FFN_OPERAND_CAPTURE_STATE_20261013.md) now observes
all24 actual pre-FFN input/up/raw+scaled gate/product/output with strict parent
continuity and full audit pending. Source6e3f80e/binding c14c2a8,32GiB/2h local
RTX3060 budget. Original proposal below retained;do not restart allocation.
This changes the function family;
it is not another low-rank output bank or a token-mean/shared-rank retry.
Known-corpus all24 native backbone is qualified. Token-only/common linear
generator correction is rejected with necessary ranks1408/1857/1857.
G is linear:its Jacobian is G in every cell. Move the conditional investigation
to the actual nonlinear source FFN,retain every2048 output coordinate.

## Algebra and the user's redundancy idea

Source bias-free smooth FFN proxy with canonical coefficient values:

F(h)=beta D[(U h) * SiLU(alpha G h)].

For a cell center mu,an expert stores F(mu) and a full2048x2048 matrix:

J(mu)=beta D[diag(SiLU(alpha G mu))U
            +diag((U mu)*alpha*SiLU'(alpha G mu))G].

Evaluate F(mu)+J(mu)(h-mu). Here Jacobians change between cells. More cells
store different local facets of the SAME pretrained nonlinear function;larger
storage does not establish additional knowledge/intelligence. The operator
keeps room for every output direction;individual J need not have full rank.
Derivative is of the real-valued proxy,not the BF16 step function. Measure
proxy/rounding/remainder separately using actual source BF16 nodes.

Dense source FFN has3*2048*4608=28,311,552 logical coefficients per invocation.
One affine expert has2048^2=4,194,304 plus center/bias. This6.75 ratio is
algebraic active coefficient count,NOT a latency/DRAM/speed prediction. Routing,
mass handling,correction,quantization and cell coverage add real costs. Source
generators/attention/head and native readout remain costly;they still require
CPU implementation and measurement before any>=50 usefulTPS claim.

This is a credible place to spend RAM on redundancy:constant expert shape,
more distinct source-derived operators. Whether enough cells preserve quality
within practical RAM/conversion budgets is entirely unproven. No claim that
100B redundant coefficients conserve a100B donor's knowledge.

## Immediate concrete next step

Implement a frozen operand collector on the exact qualified source/native
artifact. Capture ALL24 actual pre-FFN RMSNorm outputs,up projection,scaled
gate,product feature and FFN output on the same known FIT24/DEV24 histories.
Existing H traces are BEFORE SSM/attention and are not these FFN inputs.
Do not substitute them or run FFN assays on arbitrary embedding vectors.

Require every old native H/preconv/output word,final state/ring/attention-cache
witness and every requested label logit to match the qualified parent during
the collector run. Keep OWN persistent states,no teacher injection. All411
parameters/actual MuP/Python scalars verified before/after. Original FFN hooks
observe nodes;no learned substitution or new RESERVED/source answers.
Separate audit recreates actual source FFN nodes and every comparison receipt.

Compute exact storage before launch:5 BF16 fields widths2048/4608/4608/4608/
2048 =>19,394,027,520B for22547 histories/all24 sites,plus native snapshots/
metadata. Proposed cap32GiB in unique ordered E family,producer7200s/reserve
180s,OS8GiB/GPU allocated6GiB/reserved10GiB;full CPU audit separately bounded.
These are proposed budgets,not an allocated job. Freeze collector/audit/binding,
announce local RTX3060 purpose/budget,then owned bounded launch and goal pause
under the human's standing hours-long-run instruction. No Luna/subagents/T4.

After data audit,fit centers/routing/precision choices on FIT only;quantify
Taylor remainder,local source discrepancy and dense-control fidelity. Keep
full output coordinates. Test actual own-model substitution on canonical chat/
followup/DEV using the frozen whole-model gates;local error cannot admit it.
Compare single/few routed cells with structured CPU IDs AND masses,actual byte
ledger/DRAM and worst-case fallback. Increasing n must improve useful coverage
while active expert count stays bounded. Report any conversion master/gradient
cost that grows with n. Finish full audits even on catastrophic quality failure.

If this fails,retain the data for nonlinear remainder/other representations;
do not claim conditional conversion impossible. Reserved/generalization/other
model families remain subject to the existing policy,not opened for cell tuning.
