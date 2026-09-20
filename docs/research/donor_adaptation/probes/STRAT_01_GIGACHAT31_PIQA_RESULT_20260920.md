# STRAT-01 GigaChat 3.1: paired PIQA result

Status, 20 September 2026: **MEASURED — `PASS_PIQA`**. This result closes the
first task stop gate in the preregistered
[task/rollout protocol](STRAT_01_GIGACHAT31_TASK_ROLLOUT_PROTOCOL_20260919.md).
It authorizes the frozen document rollout; it does not establish rollout,
HumanEval, C-engine parity, or the final throughput target.

## Cell and anti-duplication control

- Nearest prior cell: the same BF16/Q4 pair passed the fresh document BPB gate.
- Changed coordinate: zero-shot PIQA decision accuracy, not document
  likelihood, on the pinned 1,838-item validation set.
- Falsifiable hypothesis: Q4 must retain at least 98% of the teacher's number
  of correct answers, while the teacher itself must exceed 50% accuracy.
- Paired controls: identical source token IDs, BOS 1, prefix and answer
  suffixes, item order, context, scorer binary, and choice/tie rules.
- Stop rule: do not run rollout if the teacher is at or below 50% or if Q4
  misses `ceil(0.98 * correct_BF16)`.
- Raw destination:
  `benchmarks/donor_adaptation/density/results/strat01_gigachat_piqa_v1/`.

This is not a repeat of the internal or fresh BPB cells: it measures a new,
preregistered task estimand. It is also separate from the MTP pilot because
both PIQA arms are base-model GGUFs without MTP.

## Frozen bindings

| Object | Binding |
|---|---|
| BF16 teacher | `GigaChat3.1-10B-A1.8B-bf16.gguf`, SHA-256 `e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e` |
| Q4 candidate | `GigaChat3.1-10B-A1.8B-q4_K_M.gguf`, SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| Source tokenizer | SHA-256 `b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe` |
| PIQA rows | 1,838, SHA-256 `93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d` |
| PIQA labels | 1,838, SHA-256 `b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb` |
| Token-ID binding | 1,838 records; aggregate item hash `75f3c95875602903a8a8e936209ba5c5ed425dfabaf9b7f89efa649e92d23285` |
| Runtime source | clean `llama.cpp` commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2` |
| Scorer | binary SHA-256 `98037c5db19db24ae2455000c589a1091facc9282a24fa166a0d18783f35942b`; source SHA-256 `74201a252fb964b02de19d8098a87da7c8b517a534dadcc9eb92cc7f5c356bfe` |

The Q4 and BF16 manifests record different repository HEADs because the
English documentation commit landed between the two long runs. The scorer
binary, scorer source, clean runtime revision, token IDs, and protocol are
identical, so this documentation-only change does not alter the paired cell.

## Apparatus controls

The Python scorer tests passed 4/4 and the C++ scorer self-test passed before
the full runs. On planted item 0, shared-prefix scoring at `n_batch=128`,
prefix replay at `n_batch=128`, and shared-prefix scoring at `n_batch=64`
produced byte-identical score files, SHA-256
`3a6884d3e6745a7dc7ddd4678905e4d1af3774e461408a8bd544a13d2c4163fb`.
The option NLLs were exactly 87.548791196021284 and 97.307019395876893, with
choice 0 correct. This closes the preregistered replay and chunk-size parity
controls.

An exploratory `n_batch=1` setting exited before producing a score. That
setting is **VOID_APPARATUS** and is not evidence about the model. It does not
invalidate the registered `n_batch=64/128` parity or either complete run.

Both complete arms used source-token-ID mode, shared prefix KV, a reset per
item, logical context 4,096, physical context 8,192, `n_batch=128`, and six
threads. Each arm consumed 24,119 prefix tokens and 87,617 answer-suffix
tokens.

## Measured result

Primary choice is minimum mean suffix NLL, with exact ties assigned to option
0. The paired bootstrap is descriptive and does not replace the count gate.

| Arm | Correct | Accuracy |
|---|---:|---:|
| BF16 teacher | 1,466 / 1,838 | 0.7976060935799782 |
| Q4 candidate | 1,456 / 1,838 | 0.7921653971708379 |

The teacher exceeds 50%. The registered Q4 requirement is
`ceil(0.98 * 1466) = 1437`; Q4 exceeds it by 19 correct answers. Therefore the
verdict is **`PASS_PIQA`**. The point paired accuracy difference is
`-0.00544069640914037`; the descriptive two-sided 95% paired-bootstrap
interval is `[-0.016866158868335146, +0.005984766050054407]`, from 20,000
draws with seed `20260916` and linear quantiles.

The secondary, non-gating total-NLL choices were 1,443 correct for BF16 and
1,431 for Q4.

## Raw evidence

| Artifact | SHA-256 |
|---|---|
| `bf16_scores.jsonl` | `6e61a17e676b1d56ca2f812c5ad682401bf3a44d1be10962b2897d595ca3c863` |
| BF16 manifest | `94de4d9ee7aedb40e8054cd9d03e0eba4d6747b22523efd5424230ffea0d010f` |
| BF16 token IDs | `2ea6ab4182970b13e02fff9f856cdfc7777de7b0f489719fabdd88b1510113cb` |
| BF16 runtime record | `f777332f797e752657a590c8f177441e1bbedd06281920e9c6dc2b7996a47d30` |
| `q4_scores.jsonl` | `c9901c03d1eee644449f818beb834dd33c0406cb6e6633a132752f2d975d21eb` |
| Q4 manifest | `9dbd7af741b6273b6e6a0d1f207ddcfea208572855eaf8f53592893eb3da8264` |
| `piqa_adjudication.json` | `1ce2c53cd86b9a383279c91cd35cfc276dfd71a5d8a0a4b820f79d672507a51d` |

## Scope and next gate

**MEASURED:** the pinned base Q4 artifact passes the paired PIQA retention
gate against its exact BF16 teacher. **Not measured:** document rollout,
HumanEval functional correctness, MTP task quality, `phase60/engine.c`
semantics, clean-box accepted-token rate, or the final >=50 tok/s lower-CI
gate.

The only automatic continuation authorized by this result is the frozen
96-document greedy rollout in the task protocol. HumanEval completions may be
generated later, but functional execution remains `PENDING_SANDBOX` on this
Windows host.
