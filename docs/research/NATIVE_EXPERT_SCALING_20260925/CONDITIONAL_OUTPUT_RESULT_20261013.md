# Conditional output functions: quality rejection and coverage diagnosis

13 October2026. **BOTH COMPLETE / BOTH INDEPENDENT_STORED_PASS / QUALITY FAIL.**
[Bank protocol](CONDITIONAL_OUTPUT_FUNCTION_PROTOCOL_20261013.md),
[result](conditional_output_function_result_20261013.json),
[full audit](conditional_output_function_adjudication_20261013.json).
[Coverage protocol](CONDITIONAL_OUTPUT_COVERAGE_PROTOCOL_20261013.md),
[exhaustive result](conditional_output_coverage_result_20261013.json),
[full coverage audit](conditional_output_coverage_adjudication_20261013.json),
[joint closure](conditional_output_session_closure_20261013.json).
All six scientific commands terminal0/matching processes absent;goal incomplete.

## Precisely tested redundancy idea

Both trained-source affine banks use actual own-normalized3072 BF16 input Z
and original O2048x3072 output coefficients. Six FIT-only trees: sparse16-
feature/64-coordinate ternary sketch, deterministic weighted input variance/
median splits,one hard uint32 ID/F32 mass1. All3 sites' FIT objects frozen
before DEV. Fit leaf affine planes and input-driven factors, all source
weights/recurrence/maps unchanged; no original-model forward or new answers.
Tree16/r128 and tree64/r32 expand output coefficients ~1.67/1.69x while
reducing active coefficients to657,408/165,888 versus6,291,456. Costs are
arithmetic; prototype banks are F64, each about84.94/87.69MB including extra
means/spectra/metadata, not ternary expert banks or physical DRAM timings.

## Error of executed real affine factors

Per-case mean relative RMS against frozen native raw output scores; FIT and
DEV at each site, equal case weighting. These are different aggregate metrics
from the prior covariance's global balanced energy tail. Same global128
control is executed from actual input under identical metric definitions.

|Site|Global128 FIT / DEV|Tree16/r128 FIT / DEV|Tree64/r32 FIT / DEV|
|---:|---:|---:|---:|
|0|65.938% /74.202%|35.462% /71.962%|43.209% /80.907%|
|12|67.116% /77.046%|45.775% /79.846%|55.560% /90.405%|
|23|44.509% /47.479%|30.420% /48.440%|38.214% /58.399%|

All288 case/arm combinations FAIL BOTH extra1% raw and extra1% BF16 budgets.
All donor-relative mean/worst/centered/domain gates FAIL at each site/arm/
split. DEV donor rounded-output means16/r128 .719777/.800459/.484870,
64/r32 .808786/.905138/.584399. Full results retain all per-case/labels/
domains/bit counts and controls; no gate loosened/no native kernel deployed.

Conditional FIT projection improves versus global rank128, but improvement
largely does not generalize. Smaller active rank with more leaves is worse
here despite similar total coefficient budgets; that is evidence for these
fixed banks/route/corpus, not a general expert-count scaling law. Actual FIT
leaf counts16-arm611--845 and64-arm122--231, all above r+1; no support rule
or partition failure occurred. Weighted conditional means explain only
~10.82/3.65/23.75% of FIT energy for16 leaves and13.37/4.86/27.38% for64.
Most response variation remains inside leaves under this input-only route.

## Strong diagnostic: exhaustive fixed-bank coverage

Each expert output lies in affine image mu_j+span(U_j). Given true target Y,
project onto EVERY frozen image and choose minimum squared error for every
token. This is optimistic finite numerical coverage for ANY selector of these
unchanged images; real input-driven maps may be more constrained. Oracle uses
the response AND all experts and is an inadmissible inference algorithm.

Fully independently qualified per-case mean DEV results:

|Site|16/r128 actual / oracle DEV|64/r32 actual / oracle DEV|
|---:|---:|---:|
|0|71.962% /67.326%|80.907% /74.137%|
|12|79.846% /75.868%|90.405% /84.826%|
|23|48.440% /44.900%|58.399% /51.914%|

Oracle each-case1% passed0/24 at every site/arm/split, including FIT. The full
audit confirms these fixed finite-space errors: router-only optimization
cannot rescue these unchanged banks to the frozen1% budget. Assignment is
imperfect (DEV oracle IDs differ from input IDs ~64--89% depending arm/site),
but coverage itself is insufficient. This does not exclude other source-
informed/learned/nonlinear banks or more/bigger specialists. No causal or
categorical information lower bound follows from this affine-response class.

## Full independent bank audit

Auditor does not import producer. Recursive FIT tree rebuilt from actual
inputs/weights with all ID/mass agreement exact. All240 leaf SVDs/full66,888
singular values,means/eigenbasis/coefficients/bias/decompositions verified.
All66,668,082 bank float words/73,512 bank integer words and277,057,536
main F64 prediction words qualified. All135,282 routed IDs/67,641 masses and
277,057,536 independently rounded BF16 words: mismatchesZERO.

Max relative numeric errors: meansZ1.46e-15/meansY1.13e-15, local squared
spectrum5.21e-15, orthogonality4.00e-15, eigen residual5.76e-15, bias equation
7.08e-16, full factors3.83e-16, metadata1.46e-15. Factor-versus-ideal-response
projection mean errors only ~2.9e-8--1.0e-7 across arms/sites/splits; they
cannot account for the measured0.30--0.90 response loss. Numeric implementation
is qualified; trained coverage/generalization remains the observed defect.
Original dense real versus native controls PASS frozen1e-6 on every history.

Coverage auditor independently expands the full residual norm, including
U.TU finite-roundoff correction. ALL5,411,280 token/expert errors and270,564
actual/oracle IDs verified;max scaled absolute error4.4119e-15,metadata4.8330e-15.
Zero negative expanded energies,zero independent argmin differences. All
source/data/runtime hashes unchanged before/after,all minima/statistics/gates
qualified. These are numerical finite-space bounds, not rigorous interval
certificates or a generic nonlinear information impossibility theorem.

## Resources, provenance and remaining pipeline

Affine source freeze9d911d8/launch7a320b6/audit launch444abf2/admissiondfe15aa.
Binding6b56efe0ed686847d833d26490bc22a405147f6257838fd4e11d997436db633b;
result fbb9fc0b1095fa2200d09e6e2577112f6934e2a354e847c8b849658ce458a77a;
audit b6ff4b0863b45e8cc3bdea6176ec321ef0299435d5ae769a8ed55db27c03c560.
Producer188.297s/OS963,424,256B;full audit117.36s/OS1,139,048,448B.
New153 NPZ files2,751,110,214B under3GiB. Coverage sourceebce734/launchc60cd7c,
producer193.297s/OS235,044,864B/new144 NPZ44,614,992B under64MiB;
full audit141.546s/OS181,665,792B,implementationebce734/launch0679be1/admissioned92487.
Coverage result6b60c6b47485498c41566a349535da8d19f73b2b59b7c7723485af33afc5375d,
audit d09df60123831eede21b788750662da89cbd91614d9e224492e8b2c758eab2db.
No source/native/GPU/T4/optimizer/RESERVED, all actual code/runtime/data/source
pins and before/after hashes; original engine/foreign work preserved.
Joint closure checks all297 NPZ extents/2,795,725,206B,report/adjudication/
binding links and all6 PID+creation identities/terminal0. Protected engine/
foreign hashes preserved;foreign publisher untouched,no owned job live.
No new mechanical scientific-command fault. A read-only rg named a missing
src directory during primitive inspection; no experiment/output was changed.

Selected next [full-rank multi-plane ternary LUT readout](TERNARY_FULL_RANK_READOUT_NEXT_20261013.md)
is PROPOSED/UNIMPLEMENTED/UNBOUND. Preserve source response dimensions while
testing weight precision/lookup layout instead of another unchanged small
affine bank. First accuracy, then actual native LUT/DRAM/batch1 cost; smaller
bytes do not guarantee speed. This is an enabling shared-core function path.
Conditional knowledge/compact causal state/source FFN/attention/head/tokenizer/
all-layer OWN-history chatbot/multiple donors/scales/useful n/conversion
masters and gradients/SAME useful>=50 tokens/s remain open. Goal active.
[Two primary upcycling sources](CONDITIONAL_REDUNDANCY_LITERATURE_NOTE_20261013.md)
reviewed at abstract level; authors' continued-pretraining results do not
constitute a no-training engine.c capacity transfer or CPU rate proof.
