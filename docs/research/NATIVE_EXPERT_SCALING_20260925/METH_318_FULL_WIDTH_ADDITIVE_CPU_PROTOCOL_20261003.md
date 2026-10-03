# METH-318: full-width two-book native decoder cost

Frozen before native observations,2026-10-03. Previous goal turn is PROGRESS:
316 closes fixed compact output fields;317 passes actual64 source descriptor
accounting. This new variable replaces activation-dependent vector LUTs by
direct small I8 palette decoding, preserving full source widths. It is neither
an unchanged301/302 kernel nor a rescue of failed IQ2/Q2 source maps.

## Fixed source and geometry

Pin317 physical raw SHA93e778d724490138ce1b187f9f887c8e86d78778d6d0c67a15d1ace71edc2e83
and immutable303 source extractor/controller SHA3b0300ada9b645fbf4576dff301cfa93bc0635ab8254e66d080de83570ef4765.
Retain all303 source/header pins. Before execution require committed physical
NEW controller/C/protocol/engine bytes. Legacy default body must be byte exact
to0ff9705. Compile engine.c opt-in SILICON_ADDITIVE_I8_FULL_PREFLIGHT with
MinGW clang/O3/AVX2/SSSE3/FMA/OpenMP/ffp-contract=off/stdc11. Record compiler,
runtime, executable, source/control and protocol hashes.

Actual26 layers/D1536,32 MLA heads: q1536x6144,kv1536x576,kb32x128x512,
vb32x512x192,wo6144x1536. First dense8960, other25 layers64 distinct banks
per routed projection with top4 and hidden1280, shared1280. No child query,
copied-region labels, low-rank output or8-head substitution.283 matrices,
1,428,619,264 encoded active coefficients/357,154,816 code bytes,
5,327,360 active scale bytes/8,683,520 independently banked palette bytes.
317 raw catalogue determines ALL geometry/counts/allocated bytes for checks.

Each eight synthetic coefficients: twoU8 indices, two256x8 I8 palettes per
matrix bank, palette entries[-63,63], reconstructed weights[-126,126], one
positiveF32 row scale. Dynamic S7 activation[-63,63], ties-even/zero controls.
Fixed mixer seeds(l+1)*10,000,000,000+organ1..11; palette offset1,234,567.
Fully allocate/fill/touch all source-sized64 banks. No generated bank is donor
knowledge; original64 is a geometry anchor, not640 useful learned functions.

Fresh Q4 header/source components: full original161,602,560byteQ6_K head,
F32 router/correction biases/attention,FFN,KV,final norms. Same source payload
hash recipe as303. Fresh BF16 header plus full394,002,432byte embedding hash.
Both native processes reread and verify specified payloads (three processes
in this protocol); full GGUF SHA is previously verified, not freshly reread.
M318SPC1 2048byte spec adds BF16 embedding offset/length/hash after source303
2000byte components and changed full-width header dimensions.

## Decoder and independent apparatus controls

Decode four groups/eight coefficients into32 signed bytes by two8-byte book
loads/group; sum I8, add128 through XOR, AVX2 maddubs then madd to signed32;
subtract128*sum(activation) before original row scale. Row tiles4, six threads,
serial threshold32768 coefficients. Pair upperbound2*254*63=32004<32768;
largest true dot8960*126*63=71,124,480<2^31. No per-input large table build.

Post-freeze selftest in EVERY fresh process:

- Exact scalar decoded-weight dot/scaledF32 output for eight fixed spread
  rows of EVERY invoked matrix/selected bank, including all32 MLA head banks.
  Integer sums exact; F32 vector aggregate versusFP64 product<=1e-6.
- Every stored code bank FIRST/LAST8bytes regenerated; independently check
  every stored palette bank FIRST/LAST8entries. No bank omission.
- All253 decoded uniform values x127 S7 input values:32,131 cases and128,524
  tile-lane comparisons.11x11 mixed decoded-pair values x11x11 activation
  extrema/cancellation values:14,641 pair controls.8960-width extrema.
- Independent scalar source-router dot/probabilities/sorted top4/gates, lowest
  ID exact ties, all25 layers. This qualifies OWN fixture routing, not original
  GGML captured-route fidelity. Selected gate/up/down bank IDs equal router.
- Independent full Q6 decode for64 fixed head rows, relativeL2<=1e-5;
  real packed-head bit mutation changes scalar AND AVX output, then restored.
- Exact1536 BF16 embedding values, zero/ties-even quantizer and nonfinite
  rejection. Bad-bank rejection, wrong bias correction, palette/index faults,
  unsupported S8 saturation are detected. All faults restored before timing.

## Fixed timed scope, repetitions and gates

Three FRESH sequential processes, each three repetitions of ten fixed input
fixtures0..9. First two fixtures EACH repetition are warmups; eight measured
fixtures remain. Fixtures/source rows are predetermined, not selected after
timing. Prepare per-layer independent residuals/attention-ready latent context
OUTSIDE timing. INSIDE: actual BF16 embedding row read/decode, all norm/
activation quantization/projections/router/top4/SwiGLU/weighted shared+routed
mixture/final normalization/full128256-logit originalQ6 head. Hash ALL organ
outputs/route IDs after timer, require exact repeatability across90 executions.
Source lookup/embedding enters first layer; subsequent layers remain independent
fixtures. No causal attention/RoPE/KV cache or layer-to-layer residual composition.

Environment fixed before observation: OMP_NUM_THREADS6,DYNAMIC FALSE,
WAIT_POLICY PASSIVE,PROC_BIND FALSE,KMP_AFFINITY none. Remove inherited
OMP_PLACES/SCHEDULE/GOMP_CPU_AFFINITY/KMP_HW_SUBSET. No affinity/wait retuning.
No other model/rate job. Record all90 latencies, nine repetition medians,
24 fixed-input medians, native OS peakRSS, logs/source hashes and route union.

Prospective gates: complete addressed weights<=560,000,000bytes (542,987,008
from317), every numeric/source/capacity/hash control passes, EACH of nine
repetition medians AND24 fixed-input medians<=14ms, within-process and pooled
cross-process repetition-median max/min<=1.10. All failure/variation retained;
no fastest subset, increased warmup or14ms relaxation. Ratio failure means
unstable profile; any median failure closes unchanged decoder before learning.

TOTAL20min wall from controller startup,12GiB child RSS, available RAM>=12GiB
before each process; native individual wall20min/OS peak12GiB guard also.
Controller observes child every<=5s; native guards after initialization/rep.
Preserve partial logs/result before any repair. Source/header algebra/compile
expected seconds, native initialization/fixed runs expected minutes. No GPU.

Command: `.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth318_additive_i8_preflight.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth318_additive_i8_preflight_result.json`.

A PASS licenses ONLY separately frozen large-bank cost/source-aware book
fitting, never50tok/s promotion. Any cost FAIL stops unchanged decoder BEFORE
book training or640-bank heap work. No cache/physical DRAM claim from palette
size/descriptors. Complete same-artifact causal quality/accepted rate, genuine
additional source capacity/learned hierarchy and another family/100B remain
missing. This enabling test can reject cost, not validate transferred quality.
