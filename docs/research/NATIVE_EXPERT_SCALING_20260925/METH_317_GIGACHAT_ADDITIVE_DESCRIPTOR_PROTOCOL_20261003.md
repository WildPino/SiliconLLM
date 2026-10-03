# METH-317: full-width additive-weight descriptor

Frozen before observations,2026-10-03. New variable: preserve original
32-head MLA/dense8960/routed1280/shared1280 widths while encoding each eight
weights by TWO U8 indices into two256x8 signed-I8 palettes, entries[-63,63].
Sum reconstructs[-126,126]; one positiveF32 scale per output row. No output
rank cap. Head originalQ6_K, F32router/norm and BF16embedding retained.

This is inspired by learned additive codebooks in [AQLM](https://arxiv.org/abs/2401.06118)
and its [official implementation](https://github.com/Vahe1994/AQLM).
Our proposed bounded-I8 books/S7 activations and native kernel are different,
untrained and unvalidated. Published quality/rates cannot qualify this candidate.
Old IQ2/Q2 maps failed quality;301/302 activation-LUT kernels failed cost.
The new representation changes both fitting and native decoding; no rerun
of those unchanged maps. Learned nonlinear/full-block targets would use314
assets only after a complete native cost gate; current code does not train.

## Source, accounting, stop

Pin physical300 raw SHA9458aec3f6baafbe00eaa266a3ac7da84c43040153cbfee3fc24ea9fd64dcd75
and300 descriptor reader SHA986abfee67cc7137bed3aecaf6a32c469d38c61d481b16859275476ce6e92ea4.
Freshly parse source/Q4 GGUF metadata and require prior header hashes, size,
all catalogue shapes/types. Read NO tensor payload. Original active/storage
geometry applies to ALL organs, not only routed experts. Codes2bits/weight,
4KiB palettes PER independent tensor bank, scales4bytes/output row; count
selected MLA-head banks, four routed parents, shared organs and full head.
Report logical palette vector loads/coefficient products, not DRAM traffic.

For actual64 parents gate complete addressed descriptor<=560,000,000bytes.
That is only the previous14ms/40GB/s yardstick, not observed physical speed.
If failed, stop before training/native work; if passed, only freeze a separate
complete native preflight. No source/student quality or50tok/s promotion.

Also calculate64/640/6400 banks, fixed top4. Larger two cases contain no actual
new donor weights/functions. Flat F32 router grows linearly. Explicitly
HYPOTHETICAL coarse64+per-parent n/64 key scan uses16D BF16 keys/four coarse
parents, original coarse router bytes. This is a cost scenario, no learned
hierarchy/exposure/route-quality evidence and no substitution in scientific data.
Stored code/scale/palettes/router/key bytes grow with n; separate them from
active bytes. Do not call analytical100B capacity transferred knowledge.

Model-free controls:2bits/group accounting,4KiB palette, known full MLA/routed
coefficient totals, safe AVX2 signed16 pair bound2*254*63<=32767 and signed32
dot8960*126*63<2^31. Hypothetical native decoder adds128 to reconstructed
I8 weights then subtracts128*sum(activation); must independently verify this
before timing. No activation-dependent palette table build proposed.

CPU stdlib/header algebra only,120s/1GiB stop, no performance/inference/GPU job.
Command: `.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth317_gigachat_additive_descriptor.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth317_gigachat_additive_descriptor_result.json`.
Preserve failure; do not overwrite/reduce banks/change precision after results.
