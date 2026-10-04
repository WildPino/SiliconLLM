# M429 result: first real adapter live-gradient guard failure, no learning

Frozen aa68f41; first/only run exit1, ZERO optimizer updates. Raw
7d82c7cdd7197df42c14a179301deb72743848c6d7db39a30c45b506e57b6a55.
Both tiny arms qualify using immutable428 checker; full archives are exactly
428 hashes. Real added-function arm ALL7 derivative/FD fields qualifies;
zero-added post/logits exactly original, C native norm2.9323572459105662e-9,
all other factor gradients zero as specified. Real native/reference max
6.778066e-8. Then adapter real live A/C >1e-10 guard fails before its archive.

First actual target is equal native-teacher probability mixture. The saved real
base gradient norm8.184370715869083e-12 makes numerical probe adequacy uncertain;
this is not evidence of an implementation defect or an additive capacity result.
Do not loosen the live-gradient guard or start fitting from this stopped run.
SAME427 forward/backward/real qualifier, one tiny checker precision change only.

35.156s/max1992216576B checked memory/25661094013B hashed. Three partial archives
retained: tiny real29066B and adapter25166B, real composition1829402B
SHAdd1e637da954ec525777861ce9ed644481f2fd5f3d800f6faed34e275027cc24.
Source manifests/capture/forward/previous qualification archives and original
374/389 binaries freshly exact. CPU only, no GPU/T4/network/new corpus.

Next NEW430 diagnostic: measure the first-point two-arm actual-mixture gradients
and independently qualify SAME operators/factors/inputs with a fixed opposite
one-hot numerical probe (as used in earlier numeric qualification). Training
objective remains the equal teacher mixture, not this artificial numeric probe.
Freeze before outcomes; retain failure429 unchanged. A successful probe can
license a NEW pilot prerequisite change only, with SAME live-gradient/derivative
and ALL9 capacity thresholds, no loss/optimizer/data/selector tuning.
