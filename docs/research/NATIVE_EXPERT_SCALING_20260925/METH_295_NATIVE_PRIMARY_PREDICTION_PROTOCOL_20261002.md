# METH-295: actual native-primary held-out prediction

## Question, reuse and decision

Measure actual original285 native complete-bank model quality against
original BF16 donor and BF16 learned E1280 on ALL24 new292 sources,8 per
category. This implements292's explicit evaluation-order revision, not a
repair/regrade of failed5% arithmetic fidelity. Reuse original276 archive,
284725-field/source/router/alias controls,285 cache/negative controls,
291 exact actual-CPU hybrid closure,293 committed source-only findings and
294 exact native scorer bridge. No weights/routes/data fitting or filtering.

Identity: archive SHA
`4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`;
manifest SHA
`99fdb09fe13e8abf8c401718911d08137851aa90c043415dd92eeb7d619f3847`;
answerability SHA
`a6b34f60541efd02d21ae3e458c17c0b52050c192b22c6aad4e526ca3fe29109`;
policy SHA
`0a3aab8f4733523b3af0b9d6bf919de406b1e94da249d5a3125176a246c57753`;
scorer bridge SHA
`fd94fe0fa4568090e22f2e07ee84755bc556ccb3750654c933765dcc507a7da3`.
Full annotation/helper/source/checkpoint bindings are asserted by the script.

Passing prediction licenses freezing native generation; failing any quality
gate closes this fixed native profile. Apparatus failure remains distinct
and must be retained before narrow prospectively frozen repair. No numerical
threshold change, source exclusions after scoring, failed-variant adoption,
native promotion or final-goal claim.

## Geometry and execution

Actual entry `benchmarks/phase60/engine.c`, define
`SILICON_COMPLETE_I16_NATIVE_PRIMARY`; clang21.1.8 O3/AVX2/SSSE3/FMA/OpenMP,
six threads/RNE/ISA/integer-edge guards. Every weight is loaded from the
original archive; no GPU state/logits/hidden injection or weight fallback.
New295 C entry is mechanically verified as294 with only complete-arm
selection, output arm count and progress/flush plumbing changes. Original
forward equation and full-head scoring are unchanged.

Document metric remains M17 stride512/context512: prefix=[EOS]+document;
first=0,512,...,end=min(first+512,length),lo=max(0,first-512); input
prefix[lo:end], relative first-scored=first-lo, RoPE positions=lo..end-1,
targets=document[first:end]. Only actual scored tokens contribute NLL.
Windows are independently rebuilt. Native cache positions are relative;
global position offset enters only RoPE. Prospective1200-token index fixture
checks the nonzero-offset1024..1199 window, and actual bundle bytes are read
back and checked against ALL IDs/targets/window metadata before execution.
No assertion of nonzero-offset CUDA numerical equality is made.

All prompt IDs are scored, including last top1; next-token targets are
shifted prompt IDs plus last=-1 sentinel. Native head computes all151936
logits in original BF16-return profile, lowest-ID argmax, float64 stable
logsumexp NLL. Binary output retains first full-head row of EVERY case plus
position/top1/target/NLL of EVERY scored row. Per-case first-row top1 must be
exact and NLL within1e-9 of independent NumPy float64 logits oracle. All
positions/targets/finite/ranges/counts/EOF must pass. No K64 approximation.

Fresh original donor and BF16 E1280 controls use original pinned loaders,
BF16/deterministic CUDA MATH SDPA/no TF32/no cache/local RTX3060/six threads.
Each document's captured per-token loss sum must exactly reproduce a
separate call to original M17 score_doc. GPU276 same-archive arm is only
descriptive. Save all per-source/per-token NLL/top1 rows and hashes.
CUDA models/tensors are released and memory allocation must be0 BEFORE
actual CPU assay. Optional native bank-off ablation is omitted:294 already
qualified its scorer;292 makes that model arm descriptive only.

## Fixed acceptance gates (unchanged280/292)

- Native pooled BPB minus EACH donor/E1280 <=.01.
- EACH category BPB difference against EACH <=.02.
- Pooled native agreement with donor top1 minus E1280 agreement >=-.01.
- EACH category agreement difference >=-.02.
- All binding/native loader/forward/window/scorer/control checks above pass.

BPB=nats/(ln2 * actual UTF8 bytes). GPU canonical scalars use original
float32 logsoftmax/sum; native sum is float64 over actual BF16-return logits.
Report this arithmetic difference without pretending CUDA equivalence.
Source bootstrap10,000 draws/seed295295/p05,p95 is descriptive, not a gate.
Frozen full-native generation/health, anonymous semantics, full1838 PIQA,
actual CPU K64 and accepted>=50 on SAME artifact remain later requirements.
No new useful capacity/large-n/RAM/DRAM/10B/100B evidence from this screen.

## Budget, stops and observation policy

Expected20-45minutes native CPU assay, hard75minutes native subprocess.
Compilation hard60s; preceding local CUDA controls hard10minutes including
binding after Python imports; GPU allocated<=10.5GiB/PythonRSS<=20GiB.
CPU child RSS<=20GiB checked every5s; kill/wait on resource stop and retain
failure and completed-case binary/logs. No downloads/T4/new resources.
Native progress one JSON line per completed case, not per token. No CPU
performance job overlaps model work. Assay wall time is conversion/evaluation
cost only; full-head teacher-forced document scoring is not accepted decode
rate and cannot establish the50tok/s gate.

```powershell
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth295_native_primary_prediction.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth295_native_primary_prediction_result.json
```

Keep authoritative live session; do not restart on observation timeout.
