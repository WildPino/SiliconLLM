# Training resource clarification

15 October 2026. Owner clarification in the project conversation: **months of
T4 usage are acceptable**. This changes architectural prioritization, not the
state of an experiment. No remote job or resource acquisition is launched here.

## Consequences

Do not reject a promising carving, upcycling, learned-code, routing or
conditional-memory transfer mechanism solely because it needs long adaptation.
The local RTX3060 and short pilot caps are not the total research training
budget. A bounded cheap falsifier can still precede long training; failure of
a short calibration recipe does not prove that longer learning cannot work.

For any selected longer path, separately account for T4 memory feasibility,
trainable versus frozen parameters, gradients/master weights/optimizer state,
activation checkpointing, table or expert offload, donor-state acquisition,
data volume, storage, measured token throughput and resumable scheduling.
Published multi-A100 training is not automatically feasible on a16GiB T4;
duration alone is no longer the reason to exclude it. Concurrent GPU count,
service quotas and uninterrupted availability are not assumed by this note.

The final inference requirement is unchanged: useful very-large-family
conditional capacity and at least50 useful accepted batch-one CPU tokens/s
on the same quality-qualified C artifact. Long training cannot substitute for
a credible complete active-work representation. Native MoE and one family
remain accepted under the [scope clarification](../NATIVE_EXPERT_SCALING_20260925/SCOPE_CLARIFICATION_20261015.md).

Historical budgets, failures and receipts remain unchanged. The final section
of the [carving review](CARVING_STRATEGY_REVIEW_20261015.md) incorporates
this broader adaptation budget. Before a new long run, freeze a concrete
protocol and resource plan; runs expected to require at least one hour follow
the owner's manual-wakeup rule. Shorter jobs do not pause the goal.
