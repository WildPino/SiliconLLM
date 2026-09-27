# METH-57: external quality and blinded semantic audit of product-key E128

**Decision:** the fixed METH-56 update-512 checkpoint passes every frozen
automatic and arm-blind semantic gate. It is eligible for E128 native export,
parity and rate work. This is a donor-relative result on one 0.5B family, not
evidence for trained quality at E27k/E273k, a packed LUT model, or a complete
native artifact.

## Bound inputs and execution

The [protocol](METH_57_PRODUCT_KEY_EXTERNAL_PROTOCOL_20260927.md) was
committed before inference, including its two source-pool amendments. The
[manifest](meth57_product_key_external_manifest.json) SHA-256 is
`9571f5d61c29e34b27c05ec531232a08b9eb6cd81bdf85aa5648ff74f1db3d75`:
8 code, 8 PG19 prose and 8 technical/general documents, selected with source
ID and fragment non-overlap checks. The 24/24 [answerability screen](meth57_answerability_screen.json)
was saved before either arm generated. The METH-56 result and checkpoint
SHA-256 are respectively
`06aa8934d849fc3bbe2ae2fb85c5ab4d302c9f3f1c5922138604bf1347273a39`
and `8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`.

Run the [automatic evaluator](../../../benchmarks/donor_adaptation/s1/meth57_product_key_external_audit.py)
with the bound training result and checkpoint. The
[paired raw result](meth57_product_key_external_audit_result.json), SHA-256
`b88e9208db0999a8b3b8550a8e5a1bd4c265910a0b202dc451c1914e2823b996`,
contains per-document nats, per-prompt route/top-1 counts, all 24 paired
continuations, 1,838 PIQA outcomes and timing. A separate readback recomputed
the pooled BPB difference, top-1 count, PIQA accuracy and EOS count from raw
rows. RTX 3060 runtime was 737.797 seconds; peak allocated GPU memory
2,433,962,496 bytes and end RSS 3,057,516,544 bytes, below the registered
budgets. No T4 was used.

## Frozen automatic gates

| Measure | Donor | Student | Gate / outcome |
|---|---:|---:|---|
| Pooled document BPB, 24 sources | 1.198750 | 1.193016 | Δ−0.005734; pass |
| Category ΔBPB: code / prose / technical | — | −0.003441 / −0.008500 / −0.005260 | each pass |
| Prompt-position top-1 | — | 3,997/4,125 = 96.897% | pooled and each category pass |
| Greedy EOS / repetition | 24/24 / 0 | 24/24 / 0 | pass |
| Full PIQA correct | 1,291/1,838 | 1,292/1,838 | +0.0544 accuracy points; paired fifth percentile −0.3808 points; pass |
| Trained pair utility | — | +0.011531 BPB when trained factor pairs are permuted | ≥+0.002; pass |
| Changed B slots / layer | — | 128/128 in all 24 layers | ≥64; pass |

The rank-64 product-key route's 16-pair top-four oracle matches exhaustive
E128 pair scoring at every external prompt position. That exactness is for
the factorized score, not a learned large-E route-quality proof.

## Blind semantic review

The [blind response file](meth57_blind_semantic_review.json) was committed
before reading A/B outputs. A single reviewer inspected each 384-character
excerpt and both continuations. The [verdict](meth57_blind_semantic_verdict.json)
and its [transcription script](../../../benchmarks/donor_adaptation/s1/meth57_blind_verdict_build.py)
were committed at `f0858bb` **before** running the unblinding scorer. Verdict
SHA-256 is
`7e1739a6adc69a50a128df9b8a050f1eeda7daab131abbc3888d0651c8dc6ab8`.
Each counted claim has an excerpt citation; ambiguous judgments are separate.
The [unblinded score](meth57_semantic_score.json), SHA-256
`61b4f4525659aed9f45fdac2474a800e629031415bac3b163e7263f4d841fb5b`,
was produced only after that commit.

| Excerpt-grounded finding | Donor | Student | Registered student ≤ donor |
|---|---:|---:|---|
| Unsupported factual claims | 30 | 22 | pass |
| Severe unsupported named-entity, numerical, comparative or causal claims | 18 | 14 | pass |
| Missing requested specific detail | 1 | 0 | pass |
| Ambiguous, excluded from failures | 6 | 6 | descriptive |

Both arms make many unsupported claims, and this is one agent's audit of 24
short, source-grounded responses. The pass says the student did not regress
against this donor on the registered counts; it does not establish absolute
truthfulness, a population error rate, or multi-family semantic transfer.

## Reproduce and next decision

```powershell
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth57_product_key_external_audit.py --training-result docs/research/NATIVE_EXPERT_SCALING_20260925/meth56_product_key_retention_result.json --training-result-sha256 06aa8934d849fc3bbe2ae2fb85c5ab4d302c9f3f1c5922138604bf1347273a39 --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth57_product_key_external_audit_result.json
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth57_blind_verdict_build.py
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth57_blind_semantic_review.py score --blind docs/research/NATIVE_EXPERT_SCALING_20260925/meth57_blind_semantic_review.json --verdict docs/research/NATIVE_EXPERT_SCALING_20260925/meth57_blind_semantic_verdict.json --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth57_semantic_score.json
```

Before re-running blind scoring, preserve the frozen verdict and verify its
SHA; changing it after seeing arm labels would invalidate this semantic gate.
The next engineering gate is native export and exact E128 product-key/factor
parity, then full-model accepted token rate on that same artifact. In parallel,
a trained distinct-expert count ladder must test quality and route load as E
grows; replicated banks and synthetic router timing are cost diagnostics only.
