# M430 result: first mixture target nearly stationary, probe qualifies

Frozen12e8fb5, first/only run exit0, ALL4 diagnostic gates PASS.
Raw7ccc823a396707490871da6817a506cc978bf929c9d95aac9cd389836c1fd903.
Original429 real base gradient byte-exact reproduced; same actual mixture loss
1.3925414969931053e-7. Adapter live A norm2.736804879943655e-11 and C
3.0740639200832476e-11: nonzero but BOTH below unchanged1e-10 guard. Base norm
8.184370715869083e-12 for both arms. The nearly stationary target does not
exercise this intended derivative-live numeric guard adequately.429 still FAIL.

Fixed opposite one-hot numeric probe on SAME operators/first inputs/base/scores/
rank8/zero factors/seed25772 passes FULL immutable427 qualifier in both arms.
ALL7 fields/native literal primals/full original logits, F64 continuation bounds,
live/zero gradient guards, independent SAME-step directional FD, fixed offsets,
zero ReLU crossings and dropped-head-offset control pass. No live-gradient,
derivative/FD or capacity threshold changed. This numerical probe is distinct
from the actual mixture loss and earlier opposite-source probability targets.

20.281s/max1095327744B/22373808870B hashes/4504408B output. Four complete
archives retain actual-mixture gradients and both full probe qualifications.
Fresh full sources/manifests/descriptors exact; CPU0/Torch1/BLAS1. No fit/C
changes/quality/rate/GPU/T4/network/new corpus. Full sizes/SHAs in raw inventory.

Next NEW431: replace ONLY the REAL numeric qualification target with this
immutable fixed one-hot rule in429 pilot; tiny precise checker from428 retained.
Training objective remains equal native-teacher mixture on SAME1008/336 points,
all factors/init/rank/optimizer/order/6240 updates/classifier/gates and ALL9
capacity criteria unchanged. No numeric probe labels in training. Freeze before
outcomes. Complete capacity outcome decides this specified additive recipe;
numeric qualification alone cannot establish useful added functions.
