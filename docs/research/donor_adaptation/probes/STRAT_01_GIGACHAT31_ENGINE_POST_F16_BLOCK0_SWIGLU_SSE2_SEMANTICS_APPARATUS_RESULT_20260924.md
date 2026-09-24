# STRAT-01 post-F16 block-0 SwiGLU SSE2-semantics apparatus result

**Verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The model-free apparatus required by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_SWIGLU_SSE2_SEMANTICS_PROTOCOL_20260924.md)
is qualified. No scientific diagnostic, GGUF access, donor graph, reference
graph, or timing/rate measurement ran.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_block0_swiglu_sse2_semantics_apparatus_20260924/`

- `adjudication.json` SHA-256:
  `98e7acb80f4b54a9e2ca7dc99f38906ba383a68c735efb75d82c8ce0a98eb641`;
- observed pre-apparatus commit:
  `9cc612615c69a5761fd901c54fda81703eb64f21`;
- compiled binary SHA-256:
  `0cca1ab4461ec23ccdb5e047ff4a1614576e14cac8740eead5db1e595bb040ce`;
- orchestration / Python-suite durations: `169.770` / `155.050` seconds.

Those durations qualify apparatus only and are not emitted-token throughput.

## Qualified controls

All eleven source controls pass:

- CLI dispatch is registered;
- reference revision and both source hashes are pinned;
- the transcribed SSE2 polynomial contains no FMA intrinsic;
- four-lane execution refuses a scalar tail;
- scalar-libm replay delegates to the unchanged production helper;
- Q6 and complete post-F16 layer 1 delegate to the already-qualified helpers;
- helper accounting and zero-graph reporting are present;
- the protocol predates implementation.

The captured reference evidence also passes independently: clean
`llama.cpp@5b335f413e4f73b0809c4fe39af894efbcc6a0d2`, exact `vec.cpp`/`vec.h`,
exact `compile_commands.json`/`build.ninja`, and one `vec.cpp` compile command
with `GGML_CPU_GENERIC` but no AVX, FMA, or fast-math flag.

The complete regression suite reports 165 passing Python tests. All 21 C
self-tests pass, including the new ten-check SSE2-semantics test. The latter
proves that the frozen synthetic vector distinguishes scalar `expf` from the
no-FMA SSE2 polynomial, rejects a changed lane, and refuses non-multiple-of-4
counts.

Source hashes at qualification include:

| source | SHA-256 |
|---|---|
| runner | `a689a0def590df5447821d7a5348ee464ed14bb296cdec590f64178b0a6e7dc1` |
| tests | `956fbfba11c72faec13760c9a8375a0d5fc74f31ac59c47455eb76df8a3b0c78` |
| protocol | `c34ebd9925c832f1b974312a9fc8a4d24724f638584521b64d543486b9540a87` |
| `engine.c` | `b68baea8825f8bdd92508e13e46590d0df59d4d5dfdaa58794dbe99934a35781` |
| SSE2 diagnostic header | `a36586e966b319182cd597a0858c0ccb472c4b96a3e9bd44beb12a618cd982d3` |

## Authorization and stop rule

After these exact sources are committed, the protocol authorizes one
scientific diagnostic. It must compare the exact scalar-libm replay against
the pinned no-FMA SSE2 candidate on immutable reference gate/up tensors, then
propagate both through unchanged Q6 and complete post-F16 layer 1.

Do not rerun any predecessor or graph. Do not broaden to AVX2/FMA, production
integration, later layers, tokenizer, quality, RAM, or rate before this cell
is adjudicated.
