# METH-21: full PIQA task retention for the exported half adapter

**Prospective paired task test.** METH-19 passes a donor-relative
document-loss gate and METH-20 passes only a relative repetition
screen, with high absolute repetition in both arms. This test asks
whether the exact adapter preserves donor performance on a task with
held-out answer labels. PIQA measures one commonsense multiple-choice
skill, not broad usefulness or code generation. The dataset was
previously bound for a different sparse-donor program, but no row was
used to train or choose this Qwen adapter or its output factor.

## Fixed data and scoring

The [runner](../../../benchmarks/donor_adaptation/s1/meth21_half_adapter_piqa.py)
ran in `--prepare` mode with no model forward and bound **all 1,838**
PIQA validation rows and labels. The [token manifest](meth21_half_adapter_piqa_manifest.json)
has SHA-256
`58acfd5882ae5d7e155bdd0ae826728aba510dc6ef4f599b7b807161a498531d`.
Input JSONL SHA-256 is
`93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d`;
label-file SHA-256 is
`b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb`.
Each manifest row binds source content, label, prompt IDs and both
option-ID hashes. The longest EOS+prompt+answer sequence is 245
tokens, below the fixed 4,096-token limit.

Use the same local Qwen2.5-0.5B BF16/SDPA donor and its
factor-0.50 E128/top-4 safetensors adapter as METH-20. Check donor
weight SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
tokenizer fingerprint `4efeeb9382a77a06`, adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`,
and adapter source-checkpoint/model/factor metadata. Score donor and
student in the **same** model instance, changing only whether the
loaded experts are enabled.

Separately tokenize fixed prefix `Question: {goal}\nAnswer:` and
each suffix ` {solution}` without special tokens. Conceptual input
is donor EOS + prefix + suffix. Sum next-token NLL for every suffix
token, without scoring an appended EOS. The primary choice is the
option with lower **mean suffix NLL**, with exact ties assigned to
option 0. Also record total-NLL choice as a sensitivity analysis,
per-option total/mean NLL, item label and correctness. This follows
the existing STRAT-02 PIQA prompt/choice convention. No task row is
discarded or sampled after scoring.

Compute paired mean-NLL-choice accuracy difference as student minus
donor. Bootstrap item-paired accuracy deltas in 20,000 draws with
NumPy seed 212121; the one-sided 95% lower bound is their 5th
percentile. The **task-retention gate** passes only if the point
difference is at least **−0.02** and the lower bound at least
**−0.05**. Report absolute accuracies and discordant item counts
even on failure. A pass supports one task-specific relative-retention
claim, not useful generation or native speed. A failure halts
promotion of this adapter pending a repair validated on new task
data; no factor retune on these labels counts as fresh validation.

One RTX 3060 run; stop at 25 minutes wall time, 10.5 GiB peak
allocated GPU memory or 20 GiB process RSS. No T4 requested.
