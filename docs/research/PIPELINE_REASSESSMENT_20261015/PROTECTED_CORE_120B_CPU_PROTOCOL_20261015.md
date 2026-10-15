# GPT-OSS-120B protected projections: favorable CPU rejection screen

This is a **synthetic-weight, source-geometry cost screen** before acquiring or integrating a large model. It cannot pass full-core speed, end-to-end accepted-token rate, or quality. It can reject a proposed format if the measured subset alone consumes the entire 20 ms/token budget for 50 batch-1 tokens/s. The output is separate from every prior donor audit.

## Frozen provenance and geometry

Use the locally pinned [official config](large_family_metadata/config120.json) (SHA-256 `933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566`) and [official reference implementation](large_family_metadata/reference_model.py) (SHA-256 `319ba3c6e4d827eb3115fe4838942af448376b513c02cea4b6b471bf6c21a346`). They came from the pinned [model revision](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json) and [reference revision](https://github.com/openai/gpt-oss/blob/5681a6b6552635e7b29cde94a61ca6a7aee90ebd/gpt_oss/torch/model.py). No source tensor bytes are downloaded or used. The controller verifies these hashes, the config fields, the frozen Git blobs of source/protocol/header/config/reference, and before/after seals of its inputs.

The configuration has 36 layers, width 2,880, 64 query heads and eight KV heads at head dimension 64, top four of 128 experts, and alternating sliding-128/full attention. The official reference's combined QKV is 5,120 × 2,880 (Q 4,096; K/V 512 each), and its O projection is 2,880 × 4,096. This screen physically allocates **557,383,680 Q4_K projection bytes** across 36 layers. Q4_K has 256-element, 144-byte blocks; rows with input 2,880 are zero-padded to 3,072 and **the padding is charged**. It also physically allocates 53,084,160 F32 router weight bytes, plus seeded QKV/O/router biases, two RMS-norm vectors and sink parameters per layer. QKV/O bias additions are included in the timed operator section. This does not use the source's MXFP4 expert representation; it tests a proposed Q4_K protected-core variant.

The CPU executes the existing pinned AVX2/FMA Q4_K×Q8_K kernel over every Q/K/V/O output row, with valid deterministic seeded Q4 blocks and Q8_K activation quantization. Each layer executes an F32 router 128×2,880, top-four selection and selected-logit softmax. One untimed fixture compares **every QKV and O output value in every layer** to the *same header's* scalar reference within a fixed `2e-4 + 2e-4×|reference|` floating-point tolerance. This is a scalar reduction cross-check, not an independent format decoder. A separate scalar router loop and sorting path checks all top-four IDs, normalized masses and finite unit sum. Every timed fixture emits an FNV-1a 64-bit check hash of all layer output vectors and routing decisions. Numerical parity across the three independent processes is required.

**Important omission:** The O projection consumes a scaled Q vector instead of an actual attention result. RoPE, attention score/value work, causal/sliding masks, F16 KV reads/writes, sink-logit softmax, and full residual semantics are not measured. Allocated sinks are a geometry charge only. Experts and the LM head are artificially free. Therefore even a very fast result remains an incomplete favorable lower screen. The logical 2,048-context F16 KV traffic (~80.22 MB/token in the preliminary ledger) is excluded from timings. The C fixture supplies neither source-value numerical parity nor semantic quality.

## Execution and decision rule

After freezing `protected_core_120b_cpu.c`, `.py`, this protocol, the pinned config/reference, and `strat01_q4k_q8k.h` in a Git revision, run:

`python benchmarks/native_expert_scaling/protected_core_120b_cpu.py --freeze <revision>`

The controller compiles with the pinned local LLVM-MinGW Clang and records compiler executable bytes/version, compile argv/log, binary bytes, all input seals, and owner/worker PID-create identities. It pins the `psutil` Python/native modules and copies the compiler's `libomp.dll` into the exclusive output directory, checking its digest. The runtime environment is the sealed M374 profile after removing inherited `OMP_`, `KMP_`, `GOMP_` and worker-binding variables. It launches **three independent processes**, each using three threads on physical-core logical IDs `[0,2,4]`, with Windows thread-affinity readbacks at control and end through `meth388_switch_thread_binding.h`. Each process performs one untimed numerical-control/warmup fixture and records eight fixed fixtures × three repeats. The operator timer sums the 36 layer sections that execute norms, Q8 activation quantization, QKV/O projections and biases, source-style residual additions, and router/top-four/softmax; it excludes finite assertions and output hashing. A separate full wall timer includes those checks, and both times are reported for every cell. No GPU, model loader, learner, or source weights are used. The aggregate deadline is 600 seconds; combined controller/worker OS peak must stay below 4 GiB and output below 128 MiB. A failed worker leaves its PID/create-time, exit code and log descriptor in `first_child_failure.json`, plus the terminal receipt; it cannot produce a passing result. Review CPU topology before launch if this machine's physical-core mapping has changed.

Predeclared classification uses the median of the three independent **process medians**, each taken over its 24 operator-time cells. All fixture and repetition medians, p95, and wall times are retained:

| Three-core median | Consequence |
|---|---|
| ≥20 ms | Reject this Q4_K protected-projection kernel/format variant: even the favorable incomplete subset consumes the 50/s budget. Other core compressions remain untested. |
| 14–<20 ms | A new core transform is needed; omitted work and head/experts have no demonstrated budget. |
| 10–<14 ms | Unresolved incomplete screen; measure full protected core before a larger pilot. |
| <10 ms | Only enabling margin; still no full-core, rate, quality or capacity admission. |

All measurements report the 24 cell times, median and p95, not only a selected fastest cell. Rejected-token time, quality, source precision, and production routing remain separate future gates.
