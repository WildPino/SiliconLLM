# STRAT-01 block-0 Q6_K×Q8_K reference-generic compile parity result

**Verdict: `BLOCK0_Q6_REFERENCE_GENERIC_EXACT_REPAIR`.**

The sole scientific invocation authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_PROTOCOL_20260924.md)
completed at source commit `6ba8420b941c1481d764689b4e7e6010012822fa`.
With the stored Q6_K matrix, generated Q8_K population, input, and downstream
amplifier held fixed, the isolated `no-avx,no-avx2,no-fma` candidate is
byte-exact to the independently compiled pinned `GGML_CPU_GENERIC` oracle and
to immutable reference output.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q6k_q8k_reference_generic_parity_20260924/`

- `adjudication.json` SHA-256:
  `cb811b19c58d3a31688967c18dc9119451ed01faee0e56098166efd4f46ff066`;
- accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- qualified repair-2 apparatus SHA-256:
  `60675f36971f9fd67982ecfcdd1eac8aa5ffa9a3f0c174cd63efc8f7b867e838`;
- engine binary SHA-256:
  `fbd6a4f0870dd4e8d79c0a4ef777706dfcb272e0c392934a083f617cdee8ee01`;
- independent oracle binary SHA-256:
  `b15f858ffca92ab7d4a28432b834d7e3f5054c6f5ebd66ce4fad4b3325c51b4f`.

## Exact outcome

| comparison | changed floats | NRMSE | normalized max | maximum absolute delta |
|---|---:|---:|---:|---:|
| candidate vs immutable `ffn_out-0` | 0 / 12,288 | `0` | `0` | `0` |
| oracle vs immutable `ffn_out-0` | 0 / 12,288 | `0` | `0` | `0` |
| candidate `l_out-0` vs reference | 0 / 12,288 | `0` | `0` | `0` |
| complete downstream candidate vs reference | 0 / 49,152 | `0` | `0` | `0` |

Candidate and oracle also emit the same Q8_K population. Their shared
`ffn_out-0` SHA-256 is
`f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4`;
the downstream SHA-256 is
`d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e`.

The historical AVX-translation-unit generic arm and the closed active-AVX2
arm retain their distinct known hashes, respectively
`f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce`
and
`3d0da87b33d531458bd0ca6529e1be8a715b67ff88b03ab82541f4d47fa445ec`.
All three arms are pairwise distinct on the live fixture. Q8 equality,
candidate/oracle equality, historical replay, one-byte Q6 mutation, schedule
twins, identities, and zero-graph controls all pass.

Execution accounting is exact: two model reads, two Q8 populations, two Q6
matrix reads, one diagnostic execution, and zero donor/reference graph
executions.

## Interpretation and next boundary

The remaining block-0 Q6 residual was caused by compiler target/lowering, not
Q8 quantization, packed-Q6 interpretation, matrix layout, or the active AVX2
transcription. The reference producer's isolated baseline-generic operation
order is the exact numerical repair.

Close this compile-parity cell permanently. Do not rerun any of its three
arms, oracle, matrix, Q8 population, mutation, or downstream amplifier. The
only authorized successor is a separately frozen production integration that
makes the normal accepted-artifact Q6 path delegate to this exact helper and
then checks the standard graph path once. No quality, generation, RAM,
throughput, or full-model claim follows, and no `SPEED_LEDGER.md` update is
due.
