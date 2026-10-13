# FIT source-function rank: qualified global linear compression limits

13 October 2026. **COMPLETE / INDEPENDENT_STORED_PASS / GOAL INCOMPLETE.**
[Protocol](SOURCE_FUNCTION_RANK_PROTOCOL_20261013.md),
[producer](source_function_rank_result_20261013.json),
[full V2 audit](source_function_rank_adjudication_v2_20261013.json),
[custody closure](source_function_rank_session_closure_20261013.json).

## Decision toward the pipeline

The qualified trained3072/state96 scan is comparatively cheap (~59us in the
prior separate native timing). Dense input generation (~2515us separately)
and normalization/output (~1175us) dominate those operator costs. This probe
asks how far globally shared low-rank LINEAR functions can reduce the expensive
maps while preserving their observed responses. All24 FIT cases/site, actual
captured BF16 H at0/12/23 and actual-input native own-normalized raw F32 output
responses; equal-case weights, uncentered moments. DEV arrays never consumed.

Measured small-rank residuals are large already on FIT. Do not launch an
unchanged global small-rank generator/readout fit as the next main conversion
route. Select [observable-route redundant conditional output functions](CONDITIONAL_OUTPUT_FUNCTION_NEXT_20261013.md)
as the next proposed controlled experiment. No conditional bank has been built
or qualified yet. Initial-layer word->generator compilation remains a separate
unimplemented exact-input alternative.

## Algebra and precise scope

Input C_H=mean_i(H_i.T H_i/n_i); positive source raw LINEAR section energies
E_s=trace(W_s C_H W_s.T), S_s=1/sqrt(E_s) for gate/x/B/C/delta. L=sqrt(C_H),
J=S W L, K=J.T J. For any globally linear rank<=r matrix M, the declared
FIT squared response loss is ||S(W-M)L||_F^2. Its optimum is the discarded
squared singular-value sum of J. Input percentages below use
total balanced energy5, not per-case mean relative error or a nonlinear field
budget. A minimizing candidate can project S W onto retained left response
directions, without inverting C_H; factors were not built here.

Output C_Y=mean_i(Y_i.T Y_i/n_i), with actual stored raw F32 Y promoted F64.
Its tail is the optimum finite output-space rank-r projection loss. Every
real rank-r linear output factorization produces responses in one rank-r
subspace, so this is also an optimistic necessary envelope for that class.
New rounded contractions could change the precise numerical response; the
envelope is not an exact certificate for a new F32/BF16 kernel. Native
normalization, source output multiplier and BF16 final output remain required.

Full eigenspectra/matrices stored and audited. Tables are sqrt(discarded
energy/total energy), as percentages. They describe this observed FIT proxy,
not own-history chatbot errors or a general information impossibility result.

### Balanced input-generator linear response error

| Rank | Site0 | Site12 | Site23 |
|---:|---:|---:|---:|
|128|33.245%|42.713%|39.454%|
|256|24.871%|33.636%|31.131%|
|512|15.785%|23.739%|21.108%|
|1024|6.647%|13.073%|11.185%|
|2048|0%|0%|0%|

### Stored own-normalized output response projection error

| Rank | Site0 | Site12 | Site23 |
|---:|---:|---:|---:|
|128|66.442%|67.295%|42.424%|
|256|53.269%|54.399%|34.147%|
|512|35.802%|38.013%|24.102%|
|1024|16.108%|18.940%|12.220%|
|2048|0%|0%|0%|

The zero tail at2048 is the full output-space/dimension control, not lossless
rounded factor-kernel qualification. Actual C_H, K and C_Y have zero negative
and zero exact-zero modes at all3 sites; the declared negative-root correction
was therefore not applied. Site0 C_H minimum3.1509e-7 versus maximum31.467
also cautions against unqualified covariance inversion; this probe used none.

At rank256, input2,240,512 coefficients versus13,729,792, output1,310,720
versus6,291,456; considerable arithmetic savings but poor finite responses.
At rank1024, input8,962,048/output5,242,880 coefficients; savings shrink while
the measured proxy losses remain substantial. Counts include both factors,
not routing/workspace/scales/conv/state/norm/head. No new latency measurement.
These observations favor studying conditional residual geometry rather than
assuming the donor response lies in one small global subspace.

## Complete independent numerical evidence

Auditor did not import producer. All72 FIT histories: full H/Y moments from
32-row blocks, all88,118,943 stored F64 array words, all136,986,624 input/
score data words. All roots/eigenvectors/eigenvalues/scales/energies/tails/
metadata/gates verified; independent 128-row J Gram blocks, direct rectangular
SVD of6704x2048 J AND11148x2048 weighted Y stack for every site, all2048 modes.

Maximum relative disagreements: H covariance1.24e-16, Y covariance2.79e-16,
root square3.33e-15, section energies1.52e-15, scales7.66e-16, K6.41e-16,
direct squared input spectrum1.35e-15/output2.04e-15, metadata1.49e-15.
All original frozen gates PASS; all hashes exact before/after. These are
F64 numerical checks, not rigorous interval-certified exact-real lower bounds.

## Reproducibility, failures and resource closure

Source/protocol/storage freeze9b5b6e1, producer launch e58f0d5,
binding_v2 SHA c496a060d1098f1bea0ecf34b86d21e534803f29573a39132138b2c7b4d90115.
Producer result SHA0d4e97138c8b4ad72b569ea8710427c8a5a8863d4cd902006056b2e922cc022e.
V2 audit implementation859b306/launch926e814, admitted result e5edddd,
adjudication SHA bd69bfb02bdd5ab76387ea261856c16ca1f64dfda6403a0d7a577652a0387cc3.

Producer CLOSED0/54.594s/OS642,674,688B; complete V2 audit CLOSED0/149.813s/
OS1,380,171,776B. Three NPZ files/new ordered E family704,960,778B under1GiB.
Preparations and both audits have PID+creation/terminal receipts; every owned
matching process absent. Same preparation PID3992 was recycled with different
creation times; closure qualifies identities by PID AND creation.

Two mechanical faults fully preserved: initial preparation KeyError on old
capture's binding schema BEFORE data computation; corrected explicit capture
binding SHA link. Audit V1 completed all computations then failed final JSON
serialization of NumPy bool; partial JSON invalid, terminal1. V2 separate code
casts ordinary bool and repeats the ENTIRE independent audit. No source data,
producer arrays, numerical threshold or rank changed. Partial JSON trailing
space preserved under scoped Git whitespace check; bytes never repaired.

Closure independently verifies all output extents, report/adjudication links,
5 terminal statuses, protected engine/foreign tracked file hashes and retained
partial-report hash. Source/checkpoint/runtime pinned to actual resolved NumPy
ABI/linalg/BLAS/psutil/Python. Foreign publisher processes preserved. No source
forward, GPU/T4, native rerun, optimizer, new teacher answers or RESERVED.

## Still required for the goal

The proposed conditional output pilot expands total representation at similar
bank coefficient budgets while reducing active rank. It must use input-derived
structured IDs/masses and frozen held-out gates; ideal response projections
cannot substitute for actual input-driven factors or native DRAM/latency.
Input generator/ternary LUT transfer, compact causal shared core, all-layer
OWN-history chatbot quality, useful n/~10B/~100B, multiple donor families/scales,
conversion master/gradient memory and SAME useful >=50 tokens/s remain open.
No owned job is live and the overall goal remains ACTIVE/INCOMPLETE.
