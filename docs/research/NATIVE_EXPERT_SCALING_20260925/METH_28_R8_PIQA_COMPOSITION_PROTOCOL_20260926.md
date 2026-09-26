# METH-28: PIQA task retention on the stored R8 composition

METH-21 established BF16 donor 70.620% and BF16-core adapter 69.967%
accuracy on all 1,838 PIQA validation items. METH-25 established
new-document BPB/top-1 quality for the stored R8 core plus that
adapter. This cell asks whether compression changes the task result.
The PIQA labels were previously inspected for METH-21, so this is a
fixed *composition* audit, not a fresh task-selection gate.

Bind donor safetensors SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
tokenizer fingerprint `4efeeb9382a77a06`, packed R8 core SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`,
and factor-0.50 E128 adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.
Reconstruct and verify the METH-21 PIQA token manifest SHA-256
`58acfd5882ae5d7e155bdd0ae826728aba510dc6ef4f599b7b807161a498531d`.
Its source JSONL and labels are fixed by their hashes in the METH-21
protocol. Score every item; do not filter by length or outcome.

Use one Qwen2.5-0.5B BF16/SDPA model instance. First score original
BF16 donor. Load all 169 R8 matrices from stored signed-int8 codes
and fp32 scales, reconstruct to BF16, and verify 121 control vectors,
metadata, shapes and tied head. Score R8 donor and then R8+adapter
with the exact exported factors/router. Never re-quantize donor
weights during this audit. The PIQA prompt, EOS, separate suffix
tokenization, mean suffix NLL primary choice, total NLL sensitivity,
and exact-tie option 0 follow METH-21. Save every choice and option
NLL, label and paired discordance.

Report each arm's accuracy and differences versus original donor and
R8 donor. For both R8 donor and R8+adapter versus original donor,
compute 20,000 item-paired bootstrap draws (seed 282828), with the
5th percentile as the one-sided 95% lower bound. The composition
task gate passes only if each point accuracy difference is ≥−0.02
and each lower bound is ≥−0.05. This preserves METH-21's task
retention tolerances and tests the new core precision. A pass does
not establish broad skills, good open generation or native rate.

Run once on local RTX 3060. Stop after 25 minutes wall time,
10.5 GiB allocated GPU memory or 20 GiB RSS. No T4 requested.
