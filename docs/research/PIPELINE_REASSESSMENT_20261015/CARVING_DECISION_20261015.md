# Carving research decision after the training-budget clarification

15 October 2026. Architectural review complete; no new numerical run or
checkpoint acquisition. The overall pipeline goal remains active/incomplete.

## Selected research direction

Prioritize a **pretrained compact recipient, a frozen conditional memory
extracted from a very-large donor, and prolonged joint recovery of the recipient
and memory adapters**. The [review's final section](CARVING_STRATEGY_REVIEW_20261015.md#revised-priority-with-months-of-single-t4-conversion-available)
supersedes its earlier codebook-first recommendation. The
[accepted training horizon](TRAINING_RESOURCE_CLARIFICATION_20261015.md)
includes months of T4 usage; short local fitting is not the only allowed method.

The intended transformation moves donor computation offline. Online execution
obtains a memory ID from the available history, reads only selected vectors,
computes contextual masses through small projections/gates, and runs the
recipient. The donor's large always-on matrices are absent from that online
path. Lookup, projection, recipient, head and state work all remain charged.

This is a research choice, not a transferred model. In particular, short-ngram
donor features may fail to preserve broad knowledge, reasoning and instruction
following. Bank bytes, distinct row IDs and a smaller recipient do not establish
useful large-model capacity. The selected large donor must contribute causal,
held-out useful knowledge to the same eventual CPU artifact.

Longer-trained source-MoE carving remains second: finer children plus learned
IDs/masses, attention reorganization and a cheaper head require joint recovery
and a new complete cost envelope. Additive activation-code tables remain an
optional third mechanism. Their extreme code rate is not a prerequisite for
the main program. No parallel numerical campaigns are launched.

## Next concrete work

1. Select and bind a pretrained recipient whose **complete** executable C path
   can plausibly leave room inside20ms for memory fusion. Qualify the actual
   recipient artifact, causal attention/KV and head at declared contexts; do
   not compose timings from different models. The existing
   `chatbot_compact_entry.c` is an unexecuted conditional-Qwen profile tied to
   a failed fit/export prerequisite, not an already qualified recipient or a
   reason to export that failed checkpoint.
2. Freeze the donor-memory function and a bounded acquisition/extraction pilot.
   Separate canonical short-context vector construction from existing Falcon
   own-history supervision. Confirm donor/recipient tokenization, source
   revision, stored vector identity, hit/miss rules, all active maps and real
   T4/host-memory constraints. The first feasibility pilot does not require
   acquiring an entire120B checkpoint.
3. Before learning, freeze a donor-gap panel, general chat/held-out formulations,
   no-hit strata and matched-frequency bank shuffling/removal/subset controls.
   Jointly recover the recipient and adapters; an adapter-only negative result
   does not settle this architecture. Prespecify a substantial first milestone
   and extend to months only with useful bank contribution and a viable
   measured schedule. The review proposes1B tokens or14 T4-days as a first
   investment cap, not a universal training requirement or a launched protocol.

For comparative claims against ordinary continued training, fund a separately
trained compute-matched no-memory control. Inference-time bank permutations
establish causal use but do not replace that training comparison. Increasing
distinct memory entries must improve useful behavior at matched active work.
Neither a small pilot nor a donor-gap subpanel completes the final very-large
family, donor-relative quality, DRAM and same-artifact useful50 requirements.

The briefing, metadata screen and advisory are complete. Recipient admission,
extraction, protocol/code and numerical evidence for this new route are not.
No model has been promoted. No long job is waiting and no goal pause is needed.
