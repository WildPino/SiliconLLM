# METH-131: Q7 E1280 passes automatic quality, fails blinded grounding

**Decision.** Do not promote the METH-130 Q7 child-B bank as a
quality-conserving replacement for the exact BF16 E1280 bank. The
precommitted document, prompt, generation and PIQA gates all pass on
24 new source/fragment-disjoint excerpts, but the separately frozen
arm-blind excerpt-grounding gate fails: Q7 has **34 versus 33
unsupported claims** and **14 versus 12 severe claims**. Both arms
miss a requested supported detail twice. The margin is small and
comes from one reviewer's 24-pair screen; the frozen criterion is
binding. No native Q7 full-model speed or 10B/100B transfer claim follows.

The [protocol](METH_131_Q7_FULL_MODEL_QUALITY_PROTOCOL_20260928.md)
was committed at `8d86746` before selecting new sources. The
[selector](../../../benchmarks/donor_adaptation/s1/meth131_q7_external_manifest.py)
excludes source IDs and three 256-byte overlap marks from every prior
manifest through METH-121. Its [manifest](meth131_q7_external_manifest.json)
SHA-256 is `84d8dbb400be133b6d1e440d55da5d09afd8ebbe0bd659474bdad8e708868ba9`:
eight pinned Git code excerpts, eight separate PG19 shard-00004 books,
and eight pinned Git technical excerpts, each 4,095 bytes. Git sources
remain in a shared broad domain with prior tests. The
[answerability screen](meth131_answerability_screen.json) fixed one
excerpt-contained detail per row before inference, SHA-256
`95480c4e6dd4a621502264f59ca9dc565134e776c18014025535fba2033cf911`.
Both were committed at `ae818fe` before the model run.

The [automatic evaluator](../../../benchmarks/donor_adaptation/s1/meth131_q7_full_model_audit.py)
was committed at `2d1955e`. It binds the BF16 Qwen2.5-0.5B-Instruct
donor, METH-56 parent, METH-107 child, exact METH-126 bank and Q7
METH-130 bank by SHA-256. Before inference it checks every router,
shared-A and BF16 child-B row against the centered model and checks
the Q7 bank's dequantized B rows. Across the whole bank, dequantized B
has relative weight L2 error 0.13216. In PyTorch, Q7 codes and FP32
scales are dequantized and the resulting B is cast to BF16 for the
model's matmul. This is a **quality proxy for the stored Q7 weights**;
the C pair-LUT operation has its own accumulation order and still
requires full-model numeric parity.

| Fresh automatic endpoint | Exact BF16 E1280 | Q7 E1280 | Frozen gate |
|---|---:|---:|---|
| Pooled document BPB, 98,280 bytes | 1.177796 | 1.177881 | Δ+0.000085 ≤+0.005: pass |
| Code / prose / technical ΔBPB | — | −0.000005 / +0.000185 / +0.000075 | each ≤+0.02: pass |
| Donor prompt top-1 agreement, 4,195 positions | 96.281% | 96.687% | loss ≤1 point: pass |
| Greedy EOS / 24 | 23 | 23 | at most two fewer: pass |
| Three-times repeated 8-gram / early non-EOS | 0 / 0 | 0 / 0 | at most one extra: pass |
| PIQA correct / 1,838 | 1,291 | 1,288 | Δ−0.163 point: pass |

The PIQA paired bootstrap lower fifth percentile is −0.544 point,
above the frozen −5-point floor; 10 exact-correct items are lost and
seven gained. PIQA is a reused regression task, not new independent
task evidence. The [raw automatic result](meth131_q7_full_model_audit_result.json),
SHA-256 `6053a6d1f5977d6f1adfdee97daf45d3d376796ce4c41fd7d2737067a8375db8`,
contains every document, prompt, response and PIQA choice. The local
RTX 3060 run took 854.844 seconds, peaked at 4.259 GB allocated GPU
memory and ended at 2.543 GB process RSS. No T4 was used.

After the automatic pass, the
[blinder](../../../benchmarks/donor_adaptation/s1/meth131_q7_blind_semantic.py)
created [anonymous pairs](meth131_q7_blind_pairs.json), SHA-256
`29e0e65e9da56bbb13fe70727a0e4d5dd4d27afc9b596655e763265d8eb6f3b6`,
with a separate hidden mapping. Thirteen pairs have identical text.
The [verdict builder](../../../benchmarks/donor_adaptation/s1/meth131_q7_blind_verdict_build.py)
and [24-row verdict](meth131_q7_blind_verdict.json), SHA-256
`33f0dc9a56d19d72e9daf40abb422be7a40612f3e8e4d431f8c66943a7f42e02`,
were committed at `e15afde` before unblinding. Each counted claim has
excerpt evidence and a severe flag. The [revealed mapping](meth131_q7_hidden_map_revealed.json),
SHA-256 `51b310454660b7c54b03ad29c271d104ebaaa0f614daf150421ec160ef4c914f`,
and [unblinded score](meth131_q7_unblinded_score.json), SHA-256
`1e62fd1dc91bd2bd264028884611337ac24eaf293ba0a5b3a85d2d33b230a8a4`,
give:

| Excerpt-only finding | Exact E1280 | Q7 E1280 | Gate |
|---|---:|---:|---|
| Unsupported claims | 33 | 34 | fail |
| Severe unsupported claims | 12 | 14 | fail |
| Answers lacking a supported requested detail | 2 | 2 | pass |

The difference is concentrated in five non-identical rows: Q7 adds
severe errors in two code responses, adds one unsupported claim in a
prose response and one in a technical response; exact B has one
additional unsupported claim in another code response. Thirteen exact
matching responses contribute the same findings to both arms. Many
errors are shared, and some excerpt starts or ends mid-sentence;
the review is a narrow paired screen, not an estimate of general
truthfulness. No verdict was edited after seeing the mapping.

Q7's component RAM and CPU savings from METH-130 remain real, but this
run rejects semantic promotion. A higher-resolution nibble code could
use the same eight-byte row footprint and pair-LUT layout; that is a
new candidate, requiring a precommitted component protocol and fresh
quality sources. The current 24 responses are now viewed. Useful
learned capacity beyond E1280, native full-path parity and >=50
accepted tokens/s on one artifact remain open.
