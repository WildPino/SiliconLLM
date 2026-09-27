# METH-67: diagnose the failed METH-66 E128 recipe on viewed prompts

Use the already viewed METH-66 24-prompt development manifest and
fixed held-out raw slice to score the previously frozen METH-55
update-16 and METH-56 update-512 E128 checkpoints against the same
BF16 Instruct donor. Bind checkpoint/source/manifest hashes. Report
top-1, raw ΔBPB and route-load coverage without training or changing
any checkpoint. Compare them with the new METH-66 E128/E1280 arms.

This is a mechanism diagnostic on viewed prompts. It cannot promote a
checkpoint, override the METH-66 stopped gates, or replace an external
source-grounded audit. The result can indicate whether a stronger
retention schedule is a plausible next hypothesis for a *new*
E1280 training protocol and disjoint development set. Local RTX 3060,
≤5 GiB allocated GPU, ≤8 GiB RSS, ≤8 minutes; no T4.
