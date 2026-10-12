# Restricted D256 trained operator compiler: prospective protocol

12 October2026, full goal ACTIVE/INCOMPLETE. New residual-coordinate transform,
not a replay of the prior full-D generator control or linear384 channel assays.

## Question and decision

Compile actual trained Falcon S6 generators into D256 input/output coordinates
with EACH source gamma folded into input coefficients. Preserve actual source
nonlinear generators/state dimensions initially. Does this restricted operator
identity hold, and how much does F32 coefficient export alter it? A local PASS
makes a warm compiler primitive available; composed donor fidelity, conditional
FFN/routing and native performance still need separate experiments.

Reuses frozen auxiliary compiler and actual P256 (SHA3dd96e0f...), same existing
FIT13 embeddings and actual source sites0/12/23. Let z=embedding@P; source input
is Pz, explicitly INJECTED and restricted, not a true source layer residual.
G=P^T P measured, rho2048/256=8:

```text
target_input = z / sqrt(z^T G z/256 + 8*epsilon_source)
input_weight = sqrt(8) * source_lift_input_weight * diag(gamma_site) * P
output_weight = P^T * source_lift_output_weight
```

Source-lift output weight already contains learned postgate gamma. Keep3072-only
postgate RMS and its unchanged source epsilon. Do not silently substitute the
diagonal gamma approximation or assume F32 P is perfectly orthogonal.

## Gates fixed before observation

All8 fields x/B/C/delta/gate/y/denominator/projected output at each actual site:
source restricted operator vs compiled target relative RMS<=1e-10, independent
scalar recurrence<=1e-10. F32 exported coefficients plus F32 G/epsilon are
promoted back to F64 for prediction; all8 coefficient-roundtrip fields relative
RMS<=1e-5. This isolates coefficient rounding, NOT F32 runtime arithmetic.
Save all exact/reference/roundtrip operand frames and all actual F32 coefficients.

Three separately reported mechanical controls, no selection or quality gates:
use identity instead of G; use unscaled source epsilon in target input norm;
use projected diagonal gamma and source input projections instead of gamma folding.
Every case/control is completed and all negative scientific outcomes retained;
only resource bound or execution fault stops processing. No training/new teacher
data/source model/chatbot/GPU/native/T4 call. Original engine unchanged; dimensions
and required normalization/metric extensions are not current packed ABI support.

## Cost, custody and next

300s CPU-only including hashes,1 thread/OS4GiB/outputs128MiB. Full bound source
weight file, P and all consumed inputs before/after exact. Principal NumPy/Python
binaries bound, not full DLL tree. Foreign processes preserved, no timing isolation
or throughput admission. Exclusive new output namespace; exact source/protocol,
binding and invocation commits before run. Stage/fault terminal always retained.
Independent stored all-frame/parameter audit120s, unchanged numerical gates.

Report dense coefficient count/bytes, linear work/state sizes from actual arrays.
Source-state width/cost remains provisional even when residual D becomes256.
Next build one coherent24-layer pretrained projected shadow chatbot with actual
source attention/FFN composition; qualify every precision/operator boundary and
held-out/own-history behavior before any long adaptation or native adoption.
Full goal still includes useful original LUT/ternary selective capacity, useful
large n/RAM/realDRAM/structured CPU IDs AND masses, >=50 useful same-artifact tokens/s
and more source families/scales. This component cannot replace that endpoint.
