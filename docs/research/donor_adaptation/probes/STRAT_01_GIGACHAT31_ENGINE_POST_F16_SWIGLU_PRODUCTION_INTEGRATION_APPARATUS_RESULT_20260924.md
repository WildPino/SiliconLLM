# STRAT-01 post-F16 SwiGLU production-integration apparatus result

**Verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The model-free apparatus required by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION_PROTOCOL_20260924.md)
is qualified. It did not open the accepted GGUF, invoke the standard Rung-2C
producer, or execute a donor/reference graph.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_swiglu_production_integration_apparatus_repair1_20260924/`

- `adjudication.json` SHA-256:
  `60070499fa933d0554fda66d1fc8aa2007c28f1db640f95e92968912f7ba2100`;
- observed pre-apparatus commit:
  `bfbc48890810fefa0aaba0df36f570b9a245856c`;
- compiled binary SHA-256:
  `981ebb70c10b901f67f7519bf380cd2975582a7ec073c908c7c4a5649f8b9222`;
- orchestration / Python-suite durations: `125.331` / `116.672` seconds.

Those durations qualify apparatus only and are not emitted-token throughput.

## Qualified implementation

The exact pinned no-FMA SSE2 polynomial now has one shared implementation in
`strat01_swiglu_sse2.h`. The include precedes Rung 2B; both
`strat01_r2b_run` and the closed diagnostic delegate to the same
`strat01_sse2_swiglu_compute` function. The diagnostic no longer owns or
duplicates the polynomial.

The block-0 Rung-2B path declares and calls the shared four-lane primitive,
including non-multiple-of-four refusal. Rung 2C declares the installed
block-0 semantics. Its two layer-1 SwiGLU sites remain unchanged and scalar,
as frozen. The already closed reference-generic Q4 default and
pinned-generic-F64 F16 reduction remain selected.

All eleven source controls pass:

- shared-header ordering and single-definition ownership;
- explicit add/multiply ordering with no FMA intrinsic;
- four-lane execution and scalar-tail refusal;
- production and diagnostic delegation;
- unchanged layer-1 coordinate;
- unchanged Q4/F16 defaults;
- explicit production configuration;
- protocol-before-implementation ordering.

All three immutable reference manifests and both predecessor adjudications
match their frozen sizes and hashes. The runner also binds every imported
validation/self-test module; `repair1` supersedes the first otherwise-passing
apparatus record, whose source inventory omitted those imported modules. The
complete regression reports 172 passing Python tests. All 21 registered C
self-tests pass, including the synthetic scalar-versus-SSE2 distinguisher and
tail refusal.

Selected source hashes at qualification are:

| source | SHA-256 |
|---|---|
| runner | `9cb933b0611e958c18cb00a7d0d5f3d6a07a9857307a22eac907309408b510a1` |
| tests | `3f029f5ce12279d95e5de29188812559ac31fbf1fb2f08a288279715b0d0908b` |
| protocol | `bdc9bf913ba7844d4510a6df593af5e51772912ec19690640592c1daa50768e2` |
| `engine.c` | `827306d116363746878778c97c89ff0224e1e33bec41b00d8ab890edca7e28bf` |
| shared SwiGLU | `c87a7f9456807d5cca82a644cfe87a1d6b0e8c0dc08dcb7c9fc1c33e06ec42d1` |
| Rung 2B | `d04cd205dcc820417116f635071bf93ddb04a7752f41fad861214bea5fa35655` |
| Rung 2C | `f18df8b69c9f5d550171147ae12760303816333256370c507300146259c8a3dc` |

## Authorization and stop rule

After these exact sources and the updated control documents are committed,
the protocol authorizes one standard accepted-artifact invocation of
`--strat01-gguf-rung2c`. It must produce exactly the two frozen schedules and
be adjudicated against the immutable 32-checkpoint, six-cache reference.

Do not execute a reference producer, repeat any predecessor, alter layer 1,
or broaden to quality, RAM, rate, later layers, AVX2, or FMA before this cell
is adjudicated.
