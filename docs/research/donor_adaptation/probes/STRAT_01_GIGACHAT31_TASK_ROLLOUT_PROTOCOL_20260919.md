# STRAT-01 GigaChat 3.1: paired task and rollout

Status, September 19, 2026: **PREREGISTERED, no GigaChat task/rollout executed**. Prerequisites closed: `PASS_SOURCE_BINDING` and `PASS_FRESH_BPB`. Teacher and candidate are the producer BF16 and Q4_K_M, respectively, pinned in the other protocols.

## Tokenizer and format

Use the source `tokenizer.json`, SHA-256 `b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe`, with exact round-trip, BOS ID 1, and no automatic EOS. Source IDs are passed directly to the runtime; the known `llama.cpp` text tokenizer does not construct the inputs.

The GGUF chat template, SHA-256 `e0b8ec1e172a124fe688e705a6e48fc30c575c23dae7388e43007f150d8343a3`, adds 1.041 tokens to a minimal question because of its embedded system prompt. It is not used: it would increase PIQA cost by about an order of magnitude and make the gate depend on a long product prompt. The task format deliberately uses the same plain completion as the STRAT-02 protocol, adapted to GigaChat IDs.

## Order and stop rule

1. **PIQA first.** If the gate or apparatus fails, do not run HumanEval/rollout to promote the candidate.
2. Document rollout only after PIQA passes.
3. HumanEval generations may be saved, but `pass@1` remains `PENDING_SANDBOX` until a genuinely isolated executor exists. Do not execute generated code on the Windows host.

## PIQA

Sources pinned identically to the previous protocol: `valid.jsonl` SHA-256 `93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d`, `valid-labels.lst` SHA-256 `b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb`, 1.838 items, original order.

For each goal, use the exact text prefix `Question: {goal}\nAnswer:`. Tokenize the prefix and each suffix ` {sol}` separately with `add_special_tokens=False`; logical input `[BOS] + prefix_ids + suffix_ids`, NLL on suffix tokens only, no EOS. Both arms receive the same IDs. Primary choice: lower mean NLL per token; exact tie → option 0. Also record the choice by total NLL, NLL per option, and item ID/hash.

Gate: `correct_Q4 >= ceil(0.98 * correct_BF16)`. If BF16 does not exceed 50% accuracy, outcome is `INCONCLUSIVE_NO_BASELINE_ABILITY`, never pass. Descriptive paired bootstrap: 20.000 draws, seed `20260916`, two-sided 95% CI for the accuracy difference; it does not replace the count gate. Before the real run: known synthetic test for the prefix/suffix boundary and off-by-one, score parity across chunk sizes on a planted item, and rejection of invalid IDs/context.

## Document rollout

Use the 96 IDs from fresh-heldout v2, SHA-256 `04687034c1054e0985e24efa87ba37a3fb11031de5b2880e5b8ac1742f80ec6e`. Prompt `[BOS] + first 256 payload tokens`; greedy generation until EOS 2 or 256 new tokens, tie-break on the lowest ID. Record all IDs and text decoded with the source tokenizer. Reuse the frozen controls: `run32`, `loop8x3`, `empty`; catastrophic failure if at least three Q4 documents degenerate without corresponding BF16 degeneration. This is paired drift, not accuracy against the actual continuation.

## HumanEval

Pinned OpenAI source: `HumanEval.jsonl.gz` SHA-256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`, 164 tasks. Prompt `[BOS] + encode(prompt)`, maximum 512 greedy tokens, stop only on EOS 2. Preserve raw completions without repairs. The functional gate remains the previous one (`successes_Q4 >= ceil(0.98 * successes_BF16)`), but no `success` can be computed on the current host: without a verified sandbox, status is `PENDING_SANDBOX`, neither pass nor fail.

PIQA/rollout/HumanEval do not measure `phase60/engine.c` or ≥50 tok/s. The reference runtime may be `llama.cpp`; the final artifact must reproduce tokenization, decisions, and quality in the C engine.
