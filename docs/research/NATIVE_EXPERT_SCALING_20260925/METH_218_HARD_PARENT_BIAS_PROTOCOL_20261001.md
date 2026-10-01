# METH-218: calibrate five local parents against deployed hard counts

Freeze before execution. METH-217 diagnoses five saved raw parent vectors
at (layer,parent)=(12,359),(16,600),(16,889),(17,158),(20,786): soft
counts balanced, hard counts failing, exact-state floor <=1.949%.
Use its hash-bound four-layer fit capture and METH-216 router/result;
no new fit inputs, key changes, projections, B or source core changes.
Only those five raw-mode FP32 bias vectors may change. All ChatML and
other raw vectors, keys and projection must remain byte-identical.

Compute original local scores with the same Torch FP32 bmm batch path.
For each offending parent require NumPy FP32-add/argmax counts to equal
the saved Torch counts exactly. Fit bias directly by cyclic coordinates,
children 0..8, <=200 sweeps. Integer target counts split N into nine
equal groups (floor plus remainder assigned to lowest IDs). For child c,
sort max-other-biased-score minus its raw score in FP64; choose midpoint
at target rank, cast FP32. Of that bias and adjacent FP32 values, choose
the closest actual child count to target, ties choosing lowest bias.
After each sweep recenter the nine biases to zero mean and cast FP32.
Stop when max hard share <=15% and min >=5%; otherwise preserve terminal
fit and reject calibration. No temperature or hyperparameter retry.

Re-evaluate every metric of the four changed raw layers from saved q.
Reuse original passing metrics only for byte-identical other raw layers
and ChatML. Require all original METH-216 fit gates (max-load <=1.25,
hot share <=25%, coverage >=4000, standardized advantage >=.05,
unbiased content argmax agreement >=15%) plus successful calibration,
finite biases and exact artifact readback. Stop before model/heldout
inference if any fails. An internal 15% fitting stop is not a relaxed
replacement for the original heldout 25% gate.

On fit pass, bind original BF16 E1280 factors and verify exact eight-prompt
teacher/control logits again. Evaluate old 256-draw raw/chat reserve in
all 24 layers; only on both passing evaluate new 512-draw raw/chat reserve;
only then METH-121 source documents at windows 128 and 512. All original
layer/cell gates apply without change. No candidate selection from reserves,
no B training, quality or native promotion on failed route gates.
Every nonfit corpus remains consumed development evidence, not fresh quality.

Store a new router and report changed vectors/maximum bias changes,
all stage metrics and original result identities. Bytes remain METH-216's
layout (hypothetical 114048 extra selected bytes/token); actual native
feature reuse, FP32 route fidelity and CPU cost still require verification.
Local RTX3060/six host threads, <=25 minutes, 12 GiB RSS, 10.5 GiB allocated
GPU, <1 GB new disk. Preserve failed calibration and partial cells. No T4.
