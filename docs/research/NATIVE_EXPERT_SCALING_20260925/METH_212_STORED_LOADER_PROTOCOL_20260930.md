# METH-212: load the saved core before consuming fresh quality sources

Bind METH-211 core/export, METH-210 result, donor revision and centered
E1280 checkpoints. Construct a BF16 model from config, overwrite all
290 core parameters from the stored core, and verify every copied
parameter. No original pretrained weight may supply a missing candidate
tensor. Consume all 364 artifact tensors: exact tied BF16 matrix,
72 Q8 FFN code/scale pairs, attention, controls and proposal buffers.
Verify shape/dtype, complete parameter/tensor coverage and tied pointer.
Attach the hash-bound centered E1280 factors after core loading.

On the already consumed 24 METH-121 sources, require exact per-source
document NLL and donor prompt-match count reconciliation with METH-210's
exact-tied arm. Donor choices may use a separate reference model;
candidate core weights must come only from METH-211. Verify proposal
code/scale-bit hashes too. Stop on any failure before new sources.
Passing licenses a separately frozen fresh-quality selection and scorer,
not native rate or useful large-n specialists.

Local RTX 3060, six host threads, <=15 minutes, <=20 GiB RSS,
<=10.5 GiB allocated GPU, <1 GB result. No T4 or training.
Runner: `benchmarks/native_expert_scaling/meth212_stored_core_loader.py`.
