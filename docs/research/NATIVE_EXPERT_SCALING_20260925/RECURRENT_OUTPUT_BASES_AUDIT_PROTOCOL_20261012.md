# Full recurrent-basis and sequential SSD audit

Frozen before audit. Producer tolerances unchanged. Recompute FIT-only per-head
GY_rec from every x/y operand with independent NumPy contractions;verify GM
identical to frozen old packet. Independent eigh/SVD:all288 head/layout master
maps,biorthogonality,balanced weighted kernel/tail,adaptive rank selection and
offsets. Source root/singular observations are not a bit-exact SVD replay.

Every saved projected input independently checked. Independent CPU recurrence
with all384 channels/state256 EVERY timestep,source A/delta/B/C/D;verify all
51,948,288 actualF32 scan words and corresponding projected input words.
This is a complete temporal check,not a few sampled scalar sequences.

Independently reconstruct both skip arms,combined gated RMS,source denominator
oracle and all1,108,230,144 storedBF16 outputs with integer RNE/blocked source
F64 out projection. All energy/cross terms/label metrics/aggregates/domain
gates/decisions and input/output hashes before/after. Old-control deltas are
bound observations against independently audited prior outputs;do not claim
exact GPU replay or erase oldF32 bridge failure. Numerical pass is separate
from local quality/native/chatbot/n/50.1800s/OS6GiB,six CPU threads,no GPU calls.
