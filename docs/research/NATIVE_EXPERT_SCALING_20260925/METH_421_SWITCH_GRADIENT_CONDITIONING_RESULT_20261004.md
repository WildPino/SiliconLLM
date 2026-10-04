# M421 result: loss-subtraction cancellation explains the retained real FD fail

Frozen8c68486, first/only diagnostic exited0, ALL5 apparatus gatesPASS.
Raw SHA2d9f26ed199f6cc73817af613fd24a94f46ef3eda602319b28e29a6449d16111.
Fresh384 complete420 baseline archives/whole22.36GB sources/manifests/selected
native418 first captures and immutable helper identities. No native forward
rerun or fitting; same two real points/rank8/seed/directions/central step1e-4.

| Source/factor | Old autograd directional projection | Original central FD | Centered central FD |
| --- | ---: | ---: | ---: |
|256 A |-1.626265957366484e-10 |-1.5987211554602254e-10 |-1.6262659257511064e-10 |
|256 C |8.259657298111917e-10 |8.260059303211165e-10 |8.259657521073568e-10 |
|128 A |-1.0312779389498957e-10 |-1.0658141036401503e-10 |-1.0312779381264472e-10 |
|128 C |1.8452563285793397e-10 |1.865174681370263e-10 |1.8452562821522895e-10 |

Original256 A normalized error0.0002754480190625858 exactly reproduced420;
128 A/C also fail same1e-5 at0.00034536164690254615/0.0001991835279092332.
Reported normalized error denominator=max(abs(projection),1e-8), SAME420;
do not confuse these normalized values with unfloored slope-relative error.

Equivalent centered loss uses max-ID-relative logits and log1p of non-max
exponential tail, without subtracting the leading full logit. It preserves
scalar objective within1e-12. ALL eight fixed factor directional checks satisfy
SAME1e-5: centered FD max normalized error2.22961651680103e-9; independent
imaginary-step centered derivative max2.2274032503155373e-9 against OLD gradient.
Unfloored nonzero-slope relative differences are also below3e-8. B/D directional
gradients and all methods exactly0 (A/C initialization0). Old/centered Torch
full factor gradients relative0..6.676782545976555e-8. ZERO ReLU branch crossings;
no rank/seed/epsilon sweep or new chosen point. Analytic complex arithmetic
control passed; local real ReLU mask held fixed, no derivative claim at kink.

This establishes numerical conditioning of the original FD loss difference,
not a wrong smooth backward or a recovered420 qualification. Original420 remains
FAIL.422 must separately freeze the stable-loss numerical form, verify all
tiny/native-STE/real-directional/negative checks and preserve exact forward;
no fitting or retroactive threshold relaxation licensed by this diagnostic.

21.890s/max1414979584B RSS/peak working set/23211512498B hashed/2215092B output,
within2min/3GiB/16MiB. Two complete gradient/direction/logit archives retained
with hashes in raw. CPU only; no GPU/T4/network/new corpus/native artifact/
quality/rate/function-benefit/LUT/physicalDRAM or useful larger-n conclusion.
Goal active/incomplete, no additional-functions model yet.
