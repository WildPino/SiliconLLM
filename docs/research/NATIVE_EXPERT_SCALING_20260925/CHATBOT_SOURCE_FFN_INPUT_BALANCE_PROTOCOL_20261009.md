# Corrected input-balance orientation

9 October2026. Freeze before the new mirrored responses. Original grid completed
faithfully under its protocol:site0 best isidentity/noimprovement;site23 alpha0/
beta1 reducesDEV error23.54%. Original result andbindings remain immutable.

The proposed positive input-alpha was the wrong orientation for balancing:
x'=Sx,W'=W/S implies RMS(x')=A*S,RMS(columnW')=B/S. Equality requiresS^2=B/A,
not A/B. Positive input-alpha amplified input outliers and reduced their weight
columns. This is a correction to the proposal,not an implementation fault or
a post-observation threshold change. HiddenR already had the matching direction.

Use SAME captured x/y/sourceweights/MUP/trit rule/native arithmetic/F64 metrics/
identity1e-5/DEV<=.90 andeverycase<=1.05 gates. Only input-alpha changes:
ordered alpha{0,-.5,-1},beta{0,.5,1}. The prior alpha0 trials/outputs andidentical
selectedDEV responses are reused. Exactly12 newFITtrials and6 oldFITtrials
reused;no repeated decomposition/source/model/teacher generation. Select by
FIT only,first strict minimum inthat order. Compare original F32 responses
from saved rawoutputs forfinite unquantized identity checks. Preserve allnew
identity/quantized outputs andselected trit/scales/S/R. Selected negativealpha
DEV is newly computed;selectedalpha0 reuses its unchanged oldDEV packet.

Same300s/30s reserve,OS8GiB,GPU10/11GiB/output512MiB/log4MiB,sixcores/nooverlap.
Actual original calibrationworker25.656s supplies the price,not a training
estimate. Keep originalbaseline errors,criteria andfirstresult. If corrected
calibration stillfails,price a finite function-aware learnedweight/scale
recovery. No optimization/native/T4/freshquality/rate/large-n/DRAM admission.
