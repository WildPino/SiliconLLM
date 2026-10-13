# Full-size ternary readout: numerical preservation qualified, native cost pending

13 October2026. **COMPLETE / INDEPENDENT_STORED_PASS / FIT p6 SELECTED / DEV PASS.**
[Fixed protocol](TERNARY_READOUT_PRECISION_PROTOCOL_20261013.md),
[producer](ternary_readout_precision_result_20261013.json),
[full independent audit](ternary_readout_precision_adjudication_20261013.json),
[custody closure](ternary_readout_precision_session_closure_20261013.json).

## What changed toward the goal

Global small-rank and input-routed affine output compression discarded too
much actual response;even exhaustive fixed-bank ideal selection could not
meet1%. This experiment keeps the full2048x3072 readout and reduces source
weight precision via multiple balanced ternary planes. It yields an accurate
source-derived packed function representation for native execution testing.
The LUT backend itself,cache/DRAM/cost,whole chatbot and useful n are not built
or admitted here. A full-rank matrix is not a general capacity certificate.

Source-trained O BF16 coefficients,actual native state96/3072-channel own-
normalized BF16 inputs Z,all48 FIT/DEV histories/sites0/12/23. Source
weights and all upstream generator/state/maps unchanged. Quantizer uses
source-only row maxima/F64 scales and actual F64 nearest-even ratio;ALL9
source parameter artifacts saved before any response. p4/5/6,balanced
integers q=sum3^d*t_d,each trit{-1,0,1},five-coordinate codes uint8[0,242],
final3 zero pads. Reference response=(q promotedF64 dot Z)*F64 scale,then
F32/BF16 RNE. No activation quantization or new source forward/native LUT.

## FIT-only choice and every held-out case

Rule frozen before response observation:smallest p passing BOTH raw/BF16
extra1% EACH case and original donor FIT gates at ALL3 sites. p4 fails all
extra cases,p5 passes sites0 and23 but fails every site12 case. FIT selects6. DEV
validates6 without upgrading based on DEV;all diagnostic p arms retained.

DEV means/worst relative RMS percentages against unquantized original native
output boundary (raw F32 scores and rounded BF16 outputs):

|Site|p4 raw mean|p5 raw mean|p6 raw mean /worst|p6 BF16 mean /worst|
|---:|---:|---:|---:|---:|
|0|2.628%|.868%|.290% /.296%|.361% /.365%|
|12|3.189%|1.050%|.350% /.367%|.415% /.430%|
|23|2.712%|.888%|.296% /.310%|.366% /.380%|

All144 p6 case/site raw and BF16 gates PASS,including FIT. All-case worst
raw.367007%/BF16.431457% (site12 FIT),below original1% budget. p4 extra
pass0/144,p5 pass96/144 (raw AND BF16),p6 pass144/144. All original donor
mean/worst/centered/domain gates pass for SELECTED p6 at all3 sites/FIT+DEV.
Selected DEV donor BF16 mean.0763946657654/.0872459267091/.0251148531471
versus unquantized actual-input baseline.07634113382/.08717654814/.02490538199.
These are local supplied-input mixer outputs,not a donor-relative whole model.
Full per-case/centered/labels/domains/worst/bit counts/other-arm gates retained.

## Coefficient geometry and redundancy meaning

RMS coefficient error p4 roughly2.99/3.24/3.03%,p5 .989/1.071/1.003%,
p6 .329/.356/.334% at0/12/23. All9 matrices have robust numerical rank2048;
full2048 direct source/quantized singular values and independent Gram spectra
qualified. Source sigma_min.147544/.144000/.119861,p6.147586/.144066/.119841.
No zero rows,no clipping,no positive-mode flooring/inversion. Half-step
coefficient bound max~1+7.1e-14 fits frozen1+1e-12 allowance. Actual F64 ratio
ties retained;quantizer makes no exact rational-RNE claim for unrounded scales.

All digit/carry/pattern/padding identities exact over integers. The extra
planes encode precision of the SAME trained function;they are representational
redundancy,not extra donor intelligence. No imposed low response rank,while
full numerical rank alone does not imply accuracy (p4 is a counterexample).
Conservative coefficient-induced output bounds and actual amplification
metrics are retained for cancellation analysis,not used to waive measured gates.

p6 prospective packed codes7,557,120B+F64 row scales16,384B per site,versus
source BF16 O12,582,912B/nativeF32 O25,165,824B. Packed+scales ~3.32x smaller
than nativeF32 coefficients,not a latency claim. Each archived parameter
packet also carries redundant int16 q and spectra for audit;these increase
stored artifact bytes and are NOT hidden inside proposed deployed-code counts.
Existing engine2-trit int8/pshufb/nibble differs from this proposed five-trit
floating LUT. Full F32 table597,780B,multiple-plane gathers/accumulations and
blocking/DRAM need real measurements;source BF16 preservation comes first.

## Full independent audit scope and numerical evidence

Auditor does not import producer. Independently decodes original source
coefficients/native coefficient bit equality,manual sign/floor/parity RNE,
all56,623,104 q coordinates,56,678,400 packed codes,18,432 scales and36,864
singular values. Every digit/padding/histogram exact. Independent column-block
Gram spectra max source/quantized squared discrepancies5.74e-15/5.66e-15;
no negative Gram eigenvalues. Scale discrepancy0.

All415,586,304 archived BF16 output words compared with separate32-row
integer-q reconstruction/contraction/low-bit RNE:ZERO bit differences,
independent BF16 smooth error0. All346,321,920 actual Z/Y values consumed;
every raw-score metric/bound/flag/aggregate/FIT choice/DEV decision recomputed.
Metadata max error3.47e-18. Raw F64 score arrays were NOT archived;do not claim
all their producer bits are stored/audited. Original dense control vs native
passes1e-6 every history. No original AVX reduction bit-equivalence claim for
the new ternary reference. All before/after hashes exact.

## Resources, custody and prior cost evidence

Source freeze4d759bb/code3f7d61d/producer launchffd585c/audit launch02b96e1/
admissiona3e5b1a. Binding
cee77ca52d0bf65d4f4e03f3051b4d51d63c0a139e7716fee806671e5c8355f7;
result213cc23053c6a9c9be24742f0c90166f8cc4707cfa2c9a1168323907de467664;
audit1da534e497ac53c331793c70166a60946f9e6ec8277a99e496b1b407d64e4da9.

Preparation14.281s/OS40,906,752B,producer162.5s/OS528,916,480B,
full audit116.828s/OS988,246,016B;allterminal0/PID+creation matching processes
absent. New153 NPZ/1,001,656,242B under1GiB,orderedE family. Closure verifies
every new extent/report/adjudication/binding link,three scientific terminals,
protected engine/foreign hashes and current external publisher snapshot.
No new mechanical scientific-command fault. One read-only summary command
had a PowerShell brace error;corrected multiline extraction,experiment unchanged.

Actual old M398/399/400 reports inspected/pinned,not rerun. M398 full-width
input-ready synthetic-code decoder~.138--.165s fixture medians rejected;
M399 builder and rows individually>14ms;M400 blocked~.119--.125s likewise
rejected. Distinct Q6/synthetic-code apparatus and partly fingerprint checks,
not learned Falcon readout or cold expert-DRAM proof. New numeric success does
not overturn those cost failures or imply fast arbitrary lookup layouts.

## Next and complete objective

[Native selected-p6 LUT execution](TERNARY_READOUT_NATIVE_NEXT_20261013.md)
SELECTED/PROPOSED/UNIMPLEMENTED/UNBOUND:accurate SAME source-derived artifact,
new floating backend,original dot primitive control,then warm/cold cost with
explicit cache state/DRAM limitations. No GPU/T4/RESERVED/optimizer/newteacher.
Useful conditional ternary experts/RAM-driven n/10B/100B/compact causal core/
generator/attention/FFN/tokenizer/head/all-layer OWN-history chatbot/multiple
donors/conversion masters and gradients/SAME useful>=50 token/s remain open.
Goal ACTIVE/INCOMPLETE,no owned run live;this is a qualified compiler component.
