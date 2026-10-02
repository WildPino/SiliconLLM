# METH-284: actual complete archive binding/operators inside phase60 C

## Uncertainty,evidence and consequence

276's unchanged725-field archive passes Python-reference277/280 prediction,
281 generation health/K64,282 anonymous semantics and283 full PIQA. Its
actual complete C binding and unique-B lookup have never been executed.
Reuse274's exact source operator (verbatim prefix copied into a new header;
original source/helper hashes stay unchanged),124's scalar conditional
equations and276's own loader/6144 source vectors. This changes execution
implementation,not weights/routing/data. It is necessary work toward the
same-artifact complete native path;component success is not final success.

The new `SILICON_COMPLETE_I16` compile personality in phase60 `engine.c`
binds the original safetensors file directly using a generated fixed-artifact
offset/type/shape catalog. All725 tensors bind exactly once;whole-file SHA256,
length,range,finite values,private/escape indices and alias surjectivity
are checked. No other weights/checkpoint are read. BF16 source weights,
unique banks/maps,LUT and proposal remain in stored representations.
RMSNorm/bias F32 archive values are cast as Python276 does. Complete
attention/RoPE/residual/full-head/greedy routines are implemented but NOT
executed or qualified by this operator phase. Bank-off mode is the stored
compact-core ablation,not an original donor. Head proposal is loaded but
not yet executed;no CPU K64 or efficient decoding claim.

Actual archive SHA `4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`,
1,329,447,260bytes.283 terminal pass SHA
`2ed1e54df9722c32287445eeb2b8d40590742b1893656c374cff6490912a7725`.
Vectors SHA `f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`;
274 actual CPU source outputs SHA
`0336e15c46e80f8d4d45816525e16cce8f34de44e635c04fac584f12257ca5ba`.
Catalog SHA `15dc10453baf8f928aae8b5a20a1521272cc44cda385184d44a5e34b7da2226c`.
Source-only catalog generation and compilation completed before numerical
observations. One development preflight typo called integer st_size;
fixed before this freeze,zero model/native numerical observations. The
subsequent catalog/phase60 build exits0. No post-observation repairs yet.

## Prospective numerical/causal checks

Freeze this protocol/source/catalog BEFORE computing new reference/native
outputs. Load the same276 archive with its own725-field GPU loader and
verify all76 archived project implementation hashes. TF32 off,deterministic
Torch,CuBLAS4096:8,six threads/localRTX3060. On all256 original states per
24layers (6144,not new held-out data),retain GPU parent/child/alias IDs,
BF16 softmax gates and isolated BF16 conditional contributions. Save/hash
reference arrays,then synchronize and finish the GPU phase BEFORE CPU work.

The phase60 C binary recomputes all6144 input quantizers/source FFNs and
routes/selected conditional contributions from the original archive. No
fixture-provided quantized inputs,selected routes or outputs feed execution.
Retain every native output. Fixed gates:

- All725 fields bound/no weight fallback and pinned archive SHA/length.
- All6144 source FP32 output bytes exactly equal original274 actual CPU
  outputs. This detects a changed operator/binding;original numeric limits
  already qualify that operator against its source equation.
- Parent/child/alias IDs exactly match GPU on every state,matching slots by
  parent ID so harmless ordering is not a route change.
- BF16 gate max absolute difference<=.002 (approximately one BF16 step
  near a quarter). Describe actual values;no bitwise gate claim unless exact.
- Isolated BF16 conditional relativeL2 median<=.001,max<=.01 over6144
  complete896-dimensional outputs,finite everywhere. These small numeric
  tolerances recognize CPU/CUDA reductions;they do not inherit whole quality.
  Report all per-state errors and exact BF16-value mismatch count.
- Causal negative: preserve routes/IDs/gates/source outputs but read unique
  B at `(resolved_alias+1)%count` inside the operator. The unchanged
  contribution checker must reject median OR maximum numerical limit.
  Returned alias metadata stays correct deliberately,so ID checking alone
  cannot detect this fault. This diagnostic mode cannot be used for quality.

Any positive guard failure stops before whole-prefix quality/rate;do not
regrade,tune weights,select states or enlarge thresholds. Preserve raw
failure/outputs. A changed execution recipe needs its own prospective record.
Pass licenses separate whole-native numeric/cache/quality qualification;
all285 acceptance rules must freeze before those observations. Original259
semantic/264 cached/274-275 cost stops are retained. No50tok/s,n/RAM/DRAM
or other-family conclusion is drawn from this component phase.

## Resources and reproduction

GPU reference hard12minutes after imports,20GiB process RSS/10.5GiB CUDA.
CPU compile/qualification/negative hard180seconds after GPU completion,
no GPU/model overlap or performance measurements. Native correct/negative
outputs each44,433,432bytes;reference<48MiB,combined new binaries/references/
reports<160MiB. No downloads/T4/new dataset/weight fitting. Preserve terminal
session/failure;observation timeout never licenses duplicate launch.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth284_native_archive_qualification.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth284_native_archive_qualification_result.json
```

Catalog reproduction on absent output:
`python benchmarks/native_expert_scaling/meth284_archive_catalog.py --out benchmarks/native_expert_scaling/meth284_archive_catalog.h`.
Native compile:
`clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -DSILICON_COMPLETE_I16 benchmarks/phase60/engine.c -o results/native_expert_scaling/meth284_complete_core_cpu.exe -lm -lpsapi -lbcrypt`.
Compilation/hash values and raw outputs are retained in the result. Catalog
depends on one exact archive;it is a reproducible case-specific loader,
not universal safetensors or multi-family support.
