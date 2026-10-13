# Full-width native recurrent chatbot: known-corpus quality qualified

13 October2026. **COMPLETE / INDEPENDENT_STORED_QUALITY_PASS. Goal ACTIVE/INCOMPLETE.**

## What changed and what the evidence proves

Retained trained source hidden2048/FFN/attention/head/tokenizer,substituted all24
mixer branches with own persistent state96/3072channels/own postgateRMS/p6 I16
readout. First generator from complete source-only token lookup;later23 from
actual candidate contextual source in_proj. No teacher state injection,random
core,D256 residual reduction,optimizer,RESERVED or T4. This full composition
passes known-corpus quality;the previous failed D256 shadow is not revised.
Source GPU FFN/attention/head remain producer/source-identity witnesses.

| Measure | Observed | Frozen gate |
|---|---:|---:|
| Tasks |14/16|>=12,each category>=2|
| Own-answer followup |Milo,correct|correct|
| DEV case-mean KL |.02155979257|<=1|
| DEV case-mean disagreement |4.884743%|<=20%|
| Worst domain case-mean KL |.05139347345|<=2|
| Worst domain disagreement |11.104442%|<=35%|
| Prefix/continuation controls |16/16 bit exact|all exact|
| Output/I64/rawF64/BF16 mismatches |0|0|
| Final state/ring mismatches |0|0|

All24 sites/all97 trajectories/580056 native positions audited;1,187,954,688
each output and integer words. Actual source-generator residual maxrelative
8.94934e-5;B/C projection max1.72017e-7/2.94360e-7;scan1.36530e-6;
final state1.90924e-6;causal fields/own norm0. No silent numerical defect found
within the declared checks. Strict arithmetic_div expected4/actual2 and
history_color expectedgreen/actualGreen retained;no semantic/case-fold waiver.
Matching donor task COUNT14/16 is not a claim of identical donor case outputs.

## Resources and custody

Producer3576.031s,holder3592.765s,peakOS4,026,753,024B,GPU allocated
3,187,002,368B/reserved3,265,265,664B. CPU independent audit2658.329s,
peakOS1,065,394,176B,no GPU. All6 owned preparation/producer/audit command
instances terminal0 and absent at closure. E family4805files/17,235,894,707B.
All declared input/output bytes hashed before/after by full audit;current
source/protected/extra inputs and output extent set verified again at closure.
No final accepted TPS measurement. Shared desktop timings are job costs.

Producer188 selected pins omitted separate bodies for transitive ABI helper
and tokenizer/Jinja runtime. Supplemental current496 audit pins do not
retroactively establish exhaustive historical runtime custody. Independent
stored math/output/tokenization replay passes within this explicit limitation.
Teacher logit custody inherits known repaired corpus's original failed family.

Frozen source19ab57b,binding e3c91ba;producer/audit binding30d9ed4;
audit launcher16ae694. [Protocol](FULL_WIDTH_NATIVE_CHAT_PROTOCOL_20261013.md),
[producer](full_width_native_chat_result_20261013.json),
[independent audit](full_width_native_chat_adjudication_20261013.json),
[closure](full_width_native_chat_session_closure_20261013.json).
Audit SHA af08c1f0800339ad689d0c5101f111577f91e97033b8b94096200ba472453744;
closure SHA2e5ec100a7d580818cc4edc757999ddce9792bd3de84e037a3eafe868cdf3129.

## Next mathematical constraint

Do not treat the CUDA bridge as the final engine. Dense source generators/
FFN/attention/head still incur source active cost. Global low-rank generator
and fixed conditional output subspaces previously failed;unchanged retries
are closed. New variable:subtract a token-conditioned prototype from ACTUAL
candidate contextual generator inputs,then measure the output-aware residual
rank on full FIT/DEV histories. Existing traces suffice: no GPU/source rerun.

Exact real-arithmetic identity G H = G mu(token) + G(H-mu(token)). The first
term is reusable RAM lookup;only the second must be evaluated per token.
Whether its required rank is small enough is an empirical question. Fit
prototypes/basis on FIT only,include ALL DEV and unknown-token fallback;
retain global-centered control,all5 field errors and fixed quality/cost gates.
Any oracle/prototype result remains a component hypothesis until causal fields,
full chatbot quality,CPU cost/DRAM and same artifact>=50 usefulTPS are measured.
Conditional useful-n,other source families/scales,conversion O(n) costs and
full final engine remain open.
