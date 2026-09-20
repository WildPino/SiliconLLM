# STRAT-01 GigaChat 3.1 document rollout — frozen execution brief

Status, 20 September 2026: **APPARATUS PASS / FULL PAIRED ARMS AUTHORIZED**.
Implementation was authorized by `PASS_PIQA`. This brief narrows the already-preregistered
[task/rollout protocol](../probes/STRAT_01_GIGACHAT31_TASK_ROLLOUT_PROTOCOL_20260919.md)
to its next executable cell. It does not change the corpus, thresholds, or
artifact pair.

## Mandatory anti-duplication card

1. **Nearest prior cell.** The paired GigaChat PIQA result is
   [`PASS_PIQA`](../probes/STRAT_01_GIGACHAT31_PIQA_RESULT_20260920.md).
   STRAT-02 contains the original rollout primitives, but not a GigaChat
   GGUF/source-token-ID execution.
2. **Changed coordinate.** Measure 256-token greedy trajectory degeneration
   on the frozen 96 fresh-heldout documents. PIQA measured answer ranking;
   fresh BPB measured likelihood. Neither measures free-running trajectories.
3. **Falsifiable hypothesis.** Fewer than three documents may be degenerate
   in Q4 without corresponding BF16 degeneration under the frozen
   `run32`, `loop8x3`, and `empty` predicates.
4. **Frozen bindings and controls.** Teacher BF16 SHA-256
   `e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e`;
   candidate Q4 SHA-256
   `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
   source tokenizer SHA-256
   `b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe`;
   fresh-heldout JSONL SHA-256
   `04687034c1054e0985e24efa87ba37a3fb11031de5b2880e5b8ac1742f80ec6e`.
   Each prompt is `[BOS=1] + first 256 payload IDs`; generation is greedy,
   lowest-ID tie-break, EOS 2 excluded from output, maximum 256 new IDs.
5. **Separate gates.** This cell adjudicates rollout only. Existing source
   binding, fresh BPB, and PIQA remain separate. C-engine parity and the
   accepted-token rate lower-CI gate remain open even on rollout pass.
6. **Claim labels.** Apparatus and unexecuted arms are `PROPOSED`; complete
   paired outputs are `MEASURED`; a control or arm that fails before its
   estimand is `VOID`; corpus/model/tokenizer metadata are `SOURCE-DERIVED`.
7. **Void and stop rules.** Before full arms, require no-model self-tests,
   source/token binding, planted degeneration predicates, and a small paired
   smoke with exact repeatability across two batch sizes. Repair only the
   failing apparatus condition and preserve its VOID record. Stop after Q4
   only if apparatus validity can already be rejected; otherwise both arms
   are mandatory. After both arms, stop promotion if at least three Q4-only
   catastrophic documents exist.
8. **Artifacts and adjudication.** Apparatus controls go under
   `benchmarks/donor_adaptation/density/results/strat01_gigachat_rollout_apparatus_v1/`;
   full outputs go under
   `benchmarks/donor_adaptation/density/results/strat01_gigachat_rollout_v1/`.
   The canonical result will be
   `docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_DOCUMENT_ROLLOUT_RESULT_20260920.md`.

## Apparatus reuse and implementation boundary

Reuse the exact degeneration definitions and artifact validation from
`strat02_task_rollout.py`, but execute the pinned GGUFs through the same clean
`llama.cpp` revision used for source-bound PIQA. The wrapper, not the C++
runtime, reads heldout UTF-8 text with the pinned source tokenizer, constructs
source IDs, decodes generated IDs, binds hashes, and performs paired
adjudication. The C++ core receives IDs only; this avoids the known
`llama.cpp` GigaChat text-tokenizer mismatch.

No generated text is executed. This brief does not authorize HumanEval,
MTP, a speed sweep, T4 use, or a `phase60/engine.c` success claim.

## Apparatus control outcome

The no-model tests, real 96-document source-token binding, and C++ self-test
pass. A one-document paired smoke was then run for both BF16 and Q4 at
`n_batch=128` and `n_batch=64`, with six explicit CPU threads. Within each
arm the generated 256-ID trajectory, decoded text, stop reason, and all three
degeneration flags are exactly identical across batch sizes. Both arms reach
the 256-token cap without `run32`, `loop8x3`, or `empty`; there is no Q4-only
degeneration in the planted smoke. BF16 and Q4 share five initial generated
IDs before diverging, which is diagnostic only and is not an apparatus
failure.

The machine-readable adjudication is
`benchmarks/donor_adaptation/density/results/strat01_gigachat_rollout_apparatus_v1/apparatus_adjudication.json`,
SHA-256 `8eb9f981f5df35ba8189ccf50c0192d5482d9b5603a56ea2bd55e9a5acb119a9`.
It binds the four rollout files and their manifests, the frozen corpus and
source tokenizer, the exact prompt-ID binding, the scorer source, and the
pinned `llama.cpp` revision. This promotes only the apparatus from
`PROPOSED` to `PASS_APPARATUS`; the 96-document estimand remains unmeasured.
