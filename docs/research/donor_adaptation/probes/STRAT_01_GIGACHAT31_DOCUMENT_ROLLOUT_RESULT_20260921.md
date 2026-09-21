# STRAT-01 GigaChat 3.1 paired document rollout result

**Status: `PASS_DOCUMENT_ROLLOUT` — measured, frozen scope only.** The Q4
candidate produced two newly degenerate documents without corresponding BF16
degeneration. The frozen stop gate fails at three or more; the candidate passes
by one document.

## Frozen estimand

This is the preregistered 96-document paired greedy rollout in the
[task/rollout protocol](STRAT_01_GIGACHAT31_TASK_ROLLOUT_PROTOCOL_20260919.md),
executed under the
[frozen brief](../briefs/BRIEF_STRAT_01_GIGACHAT31_DOCUMENT_ROLLOUT.md).
It measures free-running trajectory degeneration, not continuation accuracy.
For each document, the prompt was `[BOS=1] + first 256 source payload IDs`;
generation was greedy with lowest-ID tie-breaking, stopped on EOS ID 2 or at
256 new IDs, with EOS excluded from output. The frozen predicates were `run32`,
`loop8x3`, and `empty`.

## Outcome

| Measure | Q4 candidate | BF16 teacher |
|---|---:|---:|
| Documents | 96 | 96 |
| Degenerate documents | 2 | 1 |
| Newly degenerate on Q4 | 2 | — |
| EOS-terminated documents | 0 | 0 |
| Mean generated tokens | 256 | 256 |
| Runtime total | 8,515.6000655 s | 18,218.2622363 s |

Both Q4-only failures are `loop8x3` and belong to the `code` category:
`file:data/external/the_stack_python/cpython/Lib/test/test_generators.py`
and
`file:data/external/the_stack_python/cpython/Lib/test/test_tools/test_msgfmt.py`.
Q4 has two degenerate code documents; BF16 has none in code or prose. BF16's
one degenerate document is in `technical_general`. Neither arm triggered
`run32` or `empty`.

The generated ID sequences match exactly on 1/96 documents. Their mean common
prefix is 15.104166666666666 tokens. These are descriptive diagnostics only;
they are not accuracy or a separate promotion gate.

## Artifact and input bindings

| Binding | SHA-256 |
|---|---|
| Paired adjudication | `3535a262909731b7e38ed2389e01f9c5a371c552d27533ea2d0a3c0b578638c3` |
| Q4 rollout output | `becc5576164b9b56202ce1e6673419b035bd0eb11a6217ed93ae1f3568251c37` |
| Q4 rollout manifest | `9ebf59269b9835fc04942cc1e5db22d9faf047d55c0579c7bc7e4baccd477e47` |
| BF16 rollout output | `9e285804bd5df0302836610c00a1576431e04590336056e3dbf1ecf0e1308768` |
| BF16 rollout manifest | `a5bbe0898479d1fb20e705397a38c2c5410c7258db3d773c90932bf2dfcd8818` |
| Fresh-heldout corpus | `04687034c1054e0985e24efa87ba37a3fb11031de5b2880e5b8ac1742f80ec6e` |
| Source tokenizer | `b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe` |
| Bound prompt/token-ID set | `1fd25287cda1158becb6685cfe46651828fd13ca2076f04a251b8fa9d6856919` |

The paired model files are the pinned GigaChat 3.1 Q4_K_M candidate
(`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`) and
producer BF16 teacher
(`e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e`). Both
arms used the same clean `llama.cpp` revision
`5b335f413e4f73b0809c4fe39af894efbcc6a0d2`, source-token IDs, greedy policy,
`n_batch=128`, six CPU threads, and a per-document KV reset. Runtime totals
are durations of these reference-runtime quality runs. They are **not**
accepted-token throughput measurements and must not be entered in
`SPEED_LEDGER.md`.

## Interpretation and next gate

The result passes only the frozen document-rollout gate. It does not establish
HumanEval functionality (`PENDING_SANDBOX`), MTP quality, tokenizer/operator
parity in `benchmarks/phase60/engine.c`, execution of this same artifact in that
engine, or an accepted-token rate with lower CI95 of at least 50 tok/s (100 is
the stretch target). The rollout is a reference-runtime quality result, not an
engine or speed result.

The next strategic step is an engine compatibility and parity plan for this
accepted Q4 base artifact: specify the tokenizer and operator/layout gaps in
`phase60/engine.c`, then define the smallest informative parity gate. Do not
repeat the llama.cpp quality rollout. Any later task or engine result must be
recorded as a new scoped result and linked here rather than rewriting this one.

## Raw evidence

Canonical machine-readable adjudication:
`benchmarks/donor_adaptation/density/results/strat01_gigachat_rollout_v1/rollout_adjudication.json`.
The same directory contains `q4_rollout.jsonl` and
`bf16_rollout.jsonl`, their `.manifest.json` and `.runtime.json` sidecars, and
the source-token binding sidecars. The manifests record 96 records per arm,
corpus/model/tokenizer/runtime/scorer identities, and the protocol settings.
