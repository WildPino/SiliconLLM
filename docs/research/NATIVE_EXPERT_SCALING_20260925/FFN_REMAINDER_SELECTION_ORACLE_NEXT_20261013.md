# Separate fixed-cell selection loss from token-dependent remainder coverage

13 October2026. PROPOSED/NOT EXECUTED. Only after the complete exact-subset
audit and closure. No new quality admission or physical implementation.

The64-cell FIT energy rule fixes one neuron subset for every token in a cell.
The remaining error can have at least two causes:the important neurons change
within the cell,or the nonlinear information is distributed across many
neurons even at one input. Current fixed-subset results alone do not separate
them. DEV does not select another budget after FIT selection fails.

## Defined diagnostic

Keep current64 centers/routes/full2048 outputs,actual48 histories/sites0,12,23
and the combined exact remainder r_i. For EACH token form

score_i(x)=r_i(x)^2*||beta D[:,i]||^2.

Select32/128/512/2048 by descending score,ties by source neuron index. Add
their exact projected remainders to the same full-output affine function.
All4608 is the identity control. Compare every prediction with the smooth
source and actual BF output,and with the existing fixed-cell subset curves.
Save neuron IDs as well as predictions;measure both per-case means/worst
and domains. No model/count/centers trained on DEV,RESERVED untouched.

This is a per-token importance heuristic with oracle access to ALL source
neurons. It is not an optimal subset or a lower bound on sparse expressivity.
Its scores ignore cross covariance,so observed cancellation remains relevant.
It is also not a cheap router:calculating these IDs already evaluates the
full source input projections/nonlinearity. Never report the oracle as a
CPU candidate,as useful expert-n or as retained whole-chatbot quality.

## Decision

If a moderate budget greatly improves the complete known-corpus curves,
fixed-cell aggregation is demonstrably costly for this rule. Investigate
source-informed partitioning or a cheap predictor of these conditional IDs
while holding active width fixed;increasing available choices then has a
concrete purpose. That predictor must have FIT-only training and actual
CPU IDs AND masses/cost/DRAM and own-model quality checks.

If the per-token heuristic also fails,its failure is preserved. Inspect
projected covariance/sign-crossing geometry and joint selection before
asserting that more expert cells or a different nonlinear representation
cannot work. No universal lower bound is supplied by either heuristic.

Freeze producer/auditor/binding before observations. Complete trajectories
and selected IDs require an exact byte calculation;use ordered E family,
proposed CPU6/1800s each/OS8GiB/output4GiB. Full independent replay and all
hashes/terminal ownership remain required even on catastrophic failure.
This is one diagnostic to choose the next representation,not the final
pipeline. Compact shared core/ternary export/scalable CPU routing/multiple
donors and same useful50 remain open.
