# METH-337 result: bounded readonly recovery interrupted

Freeze83b0d6c. Authoritative exec94660 exit1. Raw SHA256 `f50d585ddc73f6e24dd4ff4c894d056908b4b6b7473931626a4405f82ada93a6`.
MAIN1200.093s AFTER imports; checkedRSS
10,322,083,840B <16GiB; time20min guard trips at
`source_tensor_reconstruction_and_readonly_payload_verification`. Verified1722 source and
existing target segments before stop; first1073-name shard completed152.657s.
Original335 payload immutable, no new payload bytes written. Source unchanged.
Quantizer exact335-equivalence/scalar/tie/subnormal controls PASS; no whole
artifact/tuple/native/quality/rate promotion from this partial check.

Complete original archive rehash precedes tensor verification; previous335
LFS whole-hash pass and327 canonical ALL tensor hashes already available.
A snapshot during the run showed57.414GB reads and58.625CPU seconds, with
65.764GB available RAM in separate host snapshot; read elapsed far exceeds
CPU time. This is a diagnostic snapshot, not a disk benchmark or causal proof
for335 slowdown. The full-container checksum and later ALLtensor identity
passes duplicate source reads. Sequential target reread also duplicates the
verifying stream. Fixed337 failure retained; no gate/budget waiver.

Next NEW policy/implementation: require canonical ALL6392 source tensor byte
identity plus pinned config/tokenizer metadata; do not require another fresh
ZIP envelope hash (original326/335 envelope evidence remains). Reconstruct
original335 offsets before reading, load14.818GB immutable target into bounded
RAM, source tensors read in physical data-pointer order; validate EVERY actual
code/F32/scale array against expected source quantization. Hash full payload
from those RAM bytes; no second disk pass. Exact same source precision/target
values, tuple/finite/alias gates; full-container freshness is explicitly a
separate claim, not implied. This resolves I/O budget without changing the
pretrained/target functional identity estimand or goal.

Historical stop:336 was unexecuted pending complete artifact validation.
[338](METH_338_SWITCH_TENSOR_RECOVERY_RESULT_20261003.md) subsequently verifies
the entire unchanged payload under the explicit canonical-identity policy.
Original-donor NEW whole quality,50 accepted/s, useful RAM-n/LUT/generalization
remain missing. Budget stages are converter costs, not inference rate.
