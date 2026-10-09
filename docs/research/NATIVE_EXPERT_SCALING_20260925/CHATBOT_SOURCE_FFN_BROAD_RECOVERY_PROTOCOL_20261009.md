# Fixed broader function recovery under unchanged engine arithmetic

9 October 2026. Drafted during broad source operand acquisition, before local
initial or optimizer observations. Freeze exact code after source capture passes.
This changes FIT coverage/schedule; forward, STE, loss and optimizer stay fixed.
Reuse earlier selected source-derived sectors and four before responses. Previous
actual 256-step states remain controls, not initialization or resumed moments.

## Uncertainty and controlled variables

Can more input coverage across 12 domains recover input-varying FFN responses,
especially at source site0, where two-prefix fitting scarcely generalizes?
Keep original D2048/FFN4608/sites0+23, same three matrices and positive row scales,
MUP/SiLU, AQ63, nearest-even trits/int-dot/F32 scales/final BF16, shared input and
28,322,816 trainable elements/site. No new deployed operator/active row/reference.
Extract and compile the unmodified Student AST from frozen local recovery24c8635;
record full AST/source SHA. This reuses exactly the prior finite forward/STE,
not a new reimplementation. Original selected code/scales must match all pairs.

All48 source x/y cases:24FIT/4422 labels,24DEV/4386,12 domains. New44 initial
function responses, reused4; save every output with hash. Compute mean/variation
metrics from actual source y and both before/after responses. Initial inference
and STE output bits must match on each NEW case before fitting. Old four initial
bit proofs are retained; current sector and original Student are identical.
No source model calls, replies, native or RESERVED. Consumed DEV is not final
fresh interaction quality. Acquisition itself does not admit learned functions.

## Fixed fitting schedule and cost comparisons

Exactly256 updates/site, site0 then23; fresh original selected masters/scales,
fresh Adam moments. Sort12 FIT domain names; within each sort its two case IDs.
For step1..256, domain=(step-1)%12; domain_visit=(step-1)//12;
case=domain_visit%2; case_visit=domain_visit//2. Batch at most64 contiguous rows:
offset0 if N<=64, otherwise(case_visit*64)%N, stop=min(offset+64,N).
No shuffle/reply cropping. Record actual per-domain/case/row/offset exposure.
This fixes update count, not equal row count, elapsed time or total training
compute versus earlier two-prefix fitting; report all of them separately.
The binder materializes the full256-step schedule from this rule and labels;
the worker consumes it and checks exact actual case/row/unique-position exposure.

Unchanged normalized squared total output error, sourceBF16 y castF32:
sum((student-y)^2)/max(sum(y^2),1e-30). Mean/variation remain diagnostics, not
training weights. AdamWlr5e-5/betas.9,.999/eps1e-8/wd0/foreachFalse/global clip1;
row scales>=1e-8 afterstep. Allsix finite positive gradient norms, allmasters/
moments finite, Adamstep exact. Save states2/256 with model/moments/RNG/history/
bindings. No checkpoint or DEV selection, early quality stop or lost-step relabel.
Faults attempt a separate atomic valid state and preserve original failure.

## Criteria frozen before observations

Retain prior whole-output gates on all24 DEV cases/site:
case-mean error ratio<=.90, everycase ratio<=1.05, everycase absolute normalized
L2<=.10 and cosine>=.99. NEW safeguard: everycase centered output error<=.10,
normalized by the actual centered source response energy. This is an additional
prospective function-fidelity condition, not a change to earlier failed gates.
Bothsites/allgates required. Report each gate even if others fail.

For r=prediction-source, norm(r)^2=N*norm(mean(r))^2+norm(r-mean(r))^2.
Verify F64 decomposition to1e-12 relative source energy. Report source mean
energy, normalized mean/varying squared errors, centered error, per-domain
whole/centered means and exact stored before/after outputs. Do not equate mean
recovery, matrix energy or stored parameter count with conditional knowledge.
Final pair layout/scales must roundtrip every trit before retaining packed fields.
Record changed trit count and F64 master/scale relative change against the exact
initial sectors, to distinguish nominal gradients from effective feature changes.

If broader recovery meets these gates, price all-site conversion and NEW whole
source own-history/chat controls before selected-function/core compression.
If it fails, inspect coverage and mean/varying loss weighting as separate future
variables; no automatic extension or capacity ceiling. Neither outcome licenses
final C promotion, fresh useful quality+50, useful n/CPU mass/DRAM or family claims.

## Fixed resource envelope and price

Previous two-site actual learning including failed family92.031s/GPU.894GB/
OS2.029GB. New full per-site x/y F32 storage144.31MB; new before/after payloads
plus four durable states and packed fields remain under2GiB by shape accounting.
Same local family600s/60reserve,OS8GiB/GPU10/11GiB/output2GiB/log4MiB/sixcores/
no overlap. Measure firsttwo optimizersteps/site, preserve state2 and extrapolate
remaining steps with their maximum plus90s allowance for broader before/after,
checkpoint and verification. Stop if projected work exceeds cap-reserve; no
cap increase/replay. No model worker starts while capture is live.

Bindings must verify successful broad capture receipt/all x/y/source/selected
sectors/old before packets/original Student/new schedule/helper/runtime/foreign
hashes. Shared launcher already admits this new schema, preserving earlier
four-case validation. No T4 allocation or full-model/month price is inferred.
