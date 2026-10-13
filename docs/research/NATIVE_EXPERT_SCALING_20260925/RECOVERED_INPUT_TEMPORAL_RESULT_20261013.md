# Recovered inputs through native recurrence: failures isolated,full audits complete

13 October2026. **COMPLETE / TEMPORAL CALIBRATION FAIL / WHOLE OPERATOR NUMERIC_FAILURE RETAINED.**
Goal ACTIVE/INCOMPLETE. No cheap generator/readout or autonomous chatbot admitted.
All144 actual native histories and the complete independent audit finished;
no failed subset or gate was discarded. Two DIFFERENT counterexamples remain.

## Fixed experiment and decision

Use the SAME previously qualified full3072/state96 DLL,FIT Ws/Vs,actual source
parameters,full D skip/F64 gate-RMS/BF16 norm/AVX F32 output. Change ONLY the
supplied x/B/C/delta/gate to fixed recovered-input replay arrays. Source
inverse RMS diagnostic uses ORIGINAL source y/gate;own denominator uses new
latent/regenerated gate. All48 complete histories/sites0/12/23,zero state per
case,no source-model forward/refit/rank/DEV choice/recovery update/timing.

Fixed primary gates:EVERY case own-norm BF16 physical output versus qualified
original-operand native output<=1%;EVERY full final recurrent state<=1%.
Original donor-relative local DEV gates mean/centered<=.1,worst/domain<=.2.
Original numeric controls unchanged:projection1e-6,scan/state/normalized/raw/
BF16 smooth1e-4,scalar1e-8,raw F32 scalar-reference bit mismatch0.

- Own-output approximation:144/144 PASS.
- Final-state approximation:143/144 PASS,one DEV site23 failure.
- Donor-relative own-norm local budgets:all3 sites/all4 gates PASS.
- Full numeric audit:one fixed-source-denominator BF16 smooth1e-4 false flag;
  global INDEPENDENT_STORED_NUMERIC_FAILURE retained. All state/projection and
  primary own-output controls PASS. Approximate data not admitted for temporal
  calibration;current output success does not establish future state fidelity.

| Site | DEV own-output vs baseline mean | DEV worst | DEV final-state mean | DEV worst state | Failed state cases | DEV own-output vs donor mean |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.452295% | 0.469179% | 0.510066% | 0.609093% | 0 | 7.641556% |
| 12 | 0.637608% | 0.667679% | 0.452104% | 0.617362% | 0 | 8.735070% |
| 23 | 0.655870% | 0.785487% | 0.444019% | 1.125798% | 1 | 2.568276% |

All FIT/DEV/mode/source/baseline/whole/centered/label/domain/projected/latent/
normalized/raw-score/state metrics and flags are retained in the
[result](recovered_input_temporal_result_20261013.json). Fixed-source-denom
source budgets FAIL0/12 as before;site23 passes. No alternative norm selection.

## Full independent numerical evidence

[Complete audit](recovered_input_temporal_adjudication_20261013.json):
12,987,072 projected B/C values,207,793,152 latent
scan values (all3072 channels at every timestep),42,467,328
full FINAL state values,415,586,304 normalized BF16 values,
277,057,536 raw F32 scores and as many BF16 outputs verified.
Every actual raw F32 score bit equals the separate scalar8-FMA reference.
The auditor recomputes each state timestep in F64 but compares saved latents
and full final states;all intermediate state-coordinate bits are not stored.

Maximum projection errors B/C 1.41487173189e-07/
1.55637169972e-07;scan 1.37016890857e-06,
final-state 1.57903784402e-06. Normalized BF16 differences0 for
BOTH modes;raw-score bit mismatches0;all scalar metrics exact0. Own-norm BF16
smooth max 7.12813468175e-05<1e-4.
Before-after bound input/current output hashes exact,including resolved
NumPy ABI/BLAS/psutil/Python and reused native/ref DLL/maps/source parameters.

### Separate numeric counterexample:FIT Smol rewrite/site23

`23/broad_fit_smol_rewrite_004`,control BF16_smooth_full_source_denominator:
0.000140981165685>1e-4. F32 smooth discrepancy is only
1.91825885359e-07;actual scalar F32 scores and their BF16 integer
RNE output are bit exact. The failure is closeness to the separately rounded
F64 reference,not a discovered scalar/AVX mismatch. Do not relabel it PASS.

The [coordinate-complete witness](recovered_input_temporal_bf16_witness_v2_20261013.json)
records ALL121 differing BF16 words:111 adjacent-bin
midpoint crossings and10 multi-bin/sign score discrepancies near cancellation.
Complete interval/value/bit checks show each actual/reference F32 value belongs
to its BF16 rounding interval. Adjacent crossings contribute squared relative
energy 1.9875689077e-08;the other
10 contribute 8.64319947868e-19. BF16 is discontinuous
at a midpoint;small raw arithmetic differences can move an output by a whole
BF16 bin. Largest-energy coordinate position161,output421:
actual F32 -323.000061035,F64-cast F32 -322.999938965,
BF16 outputs -324/-322.

The V1 adjacent-only explanation failed on a non-adjacent word;its source/
launch/terminal retained. V2 completes all coordinates and separates mechanisms.
Original1e-4 threshold and whole-operator failure are unchanged. No same native
history rerun was used to localize this failure;one scalar-reference case call
per witness version is charged explicitly.

## Temporal counterexample:DEV Numina/site23 and its algebra

`broad_dev_numina_cot_100k_035`,387 positions. Own output versus baseline
0.785487% PASS;
full final-state 1.125798% FAIL.
Latent difference 1.275109%,own normalized
input difference 1.091109%.
Regenerated delta itself has whole-history relative error
0.283775% against the original source.
Neither field RMS nor current readout establishes the state gate.

Ordered real-arithmetic decomposition,with fixed native projected B:

    Uhat-U = (dhat-d)*x*B + dhat*(xhat-x)*B + dhat*xhat*(Bhat-B)
    Edecay_t = ahat_t Edecay_prev + (ahat_t-a_t) S_original_prev
    Eterm_t = ahat_t Eterm_prev + corresponding forcing term.

F64 sequential original/regenerated states and all four final components are
stored. The [independent audit](temporal_error_decomposition_adjudication_v2_20261013.json)
uses final survival-weight sums,not the producer recurrence loop:

    w_t = product_(k>t) a_k
    Edecay_final = sum_t (w_hat_t-w_original_t)*U_original_t
    Eterm_final = sum_t w_hat_t*corresponding forcing_t.

All2,064,384 final F64 state/component words,identity/Gram/scalars/hash controls
PASS;max separate final-vector discrepancy 2.03373032018e-13<1e-10.
Component-sum identity residual 9.36298952839e-16
in implementation original-state units;native paired error disagreement
4.8920508563e-07<1e-4. Original/new
native-state errors ~6.6e-7. This is far smaller than the1.125798% state failure.

| Ordered final-state contribution | Norm / original native state norm |
|---|---:|
| decay | 0.375417% |
| delta | 1.215686% |
| x | 0.274723% |
| B | 0.384667% |

Decay+delta together: 0.997330%;x+B together:
0.471822%;total error 1.125798%.
The largest individual ordered contribution is delta forcing;decay partially
cancels it (cross Gram decay/delta -3.12080984536e-05,negative).
The full4x4 Gram/cross terms are retained. These norms are not additive or
unique causal percentages;the forcing attribution depends on the fixed
algebraic order. Gate,C and output norm can attenuate current emission error,
but do not enter this state update. Unlimited future instability is not proved.

**Units reconciliation:**decomposition producer/auditor normalize F64 vector
norms by a F32-computed norm of the actual native state (143.46295166);
the temporal metric uses its F64 norm (143.463205197). The common factor
is 0.99999823274. The table uses canonical F64-state units,
reconciled exactly with the original native failure;original arrays/results
were not overwritten. Complete conversion and Gram are in the closure.

V1 decomposition preparation refused global numeric failure and produced no
binding/arrays. V2 explicitly requires all1152 state/projection/own-output
controls PASS plus the completed rounding witness;this supports only STATE
mechanism diagnosis,not whole-operator numeric or temporal calibration admission.
All original false flags/gates remain. No native history rerun/refit/new inverse.

## Resources, revisions and closure

Native source/auditor/protocol freeze fd2f8a0;binding SHA b8d9c09a...,
producer launch4442738,audit launch4fd621f. WitnessV1 freeze48e82b7/launchdf72349;
V2 freeze8bc4b9d/launch095f5e5. DecompositionV1 freezee594018 (guarded prep fault),
V2 freezeeaeccfd/binding cd317a4e.../launchd4f7385/audit launchfbd46dc.
Exact argv/versions/hashes/caps in bindings,launches and terminals.

| Terminal | PID | Elapsed | Peak OS working set | Exit |
|---|---:|---:|---:|---:|
| recovered_input_temporal_binding_20261013.preparation_terminal.json | 8212 | 11.799604s | 41,168,896B | 0 |
| recovered_input_temporal_result_20261013.terminal.json | 8224 | 355.874206s | 644,218,880B | 0 |
| recovered_input_temporal_adjudication_20261013.terminal.json | 10356 | 508.131868s | 985,927,680B | 0 |
| temporal_error_decomposition_binding_20261013.prepare_terminal.json | 2404 | 0.141671s | 36,401,152B | 1 |
| recovered_input_temporal_bf16_witness_20261013.terminal.json | 13676 | 1.016000s | 196,722,688B | 1 |
| recovered_input_temporal_bf16_witness_v2_20261013.terminal.json | 19560 | 1.188000s | 196,870,144B | 0 |
| temporal_error_decomposition_binding_v2_20261013.prepare_terminal.json | 10204 | 0.631081s | 60,243,968B | 0 |
| temporal_error_decomposition_result_v2_20261013.run_terminal.json | 19988 | 5.849116s | 225,243,136B | 0 |
| temporal_error_decomposition_adjudication_v2_20261013.audit_terminal.json | 21892 | 1.221748s | 225,001,472B | 0 |

Nine owned commands terminal;all PID/creation pairs absent. CPU native1 thread/
affinity0;full audit sixBLAS/CPU0..5. Producer355.87s/full audit508.13s,
within900s/1200s/OS6GiB. E native family144NPZ/3,546,915,264B;
decomposition1NPZ/16,517,166B within128MiB. Two explicit
failed commands retained;no timeout-induced restart or redundant closed poll.
No source-model/GPU/T4/RESERVED/optimizer/new teacher reply. Engine/foreign
tracked hashes unchanged;related foreign Python processes preserved.
[Closure](recovered_input_temporal_session_closure_20261013.json) records all.

## Decision toward the actual pipeline

Do not fit a cheaper generator against these inferred inputs as if they were
true supervision. The original whole-field consistency passed but the fixed
state gate did not. No general impossibility or catastrophic chatbot conclusion
follows from this marginal local counterexample. The mechanism is measured
regenerated forcing;true input error remains unobserved.

Next:[TRUE generator-input capture](TRUE_GENERATOR_INPUT_CAPTURE_NEXT_20261013.md),
SELECTED/PROPOSED/UNIMPLEMENTED/UNBOUND. Same48 source prefills,actual BF16 in_proj
inputs at ALL24 layers,plus full7 continuity fields at sites0/12/23. Capturing
all input layers in the same forward supports later all-layer conversion.
Proposed local RTX3060 family1800s/OS12GiB/GPU10GiB allocated11GiB reserved/
new ordered E6GiB,payload4,231,620,960B;no job/GPU/T4 allocated now. Announce
purpose,budget/stops,bind actual runtime/source/GPU availability before launch.
Then qualify true-input replay and [FIT source-function response-rank envelope](SOURCE_FUNCTION_RANK_NEXT_20261013.md),
followed by cheaper shared plus conditional ternary functions/OWN causal layers.
Useful n,CPU IDs AND masses,real expert DRAM,conversionO(n) masters/gradients,
multi-family/scales,donor-relative chatbot quality and>=50 useful batch1 token/s
on the SAME complete artifact remain unresolved. No goal completion admission.
