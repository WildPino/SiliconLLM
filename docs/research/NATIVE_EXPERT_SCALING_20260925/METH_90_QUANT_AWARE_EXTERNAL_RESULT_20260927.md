# METH-90: automatic pass, blind semantic failure

**Decision:** the exact METH-85 grouped-R8 core plus METH-88 adapted E128 bank is **not promoted to native integration**. It passes the frozen document, generation and full PIQA gates on new sources, but fails the preregistered blind unsupported-claim gate. The [protocol](METH_90_QUANT_AWARE_EXTERNAL_PROTOCOL_20260927.md), manifest, answerability screen and evaluator were committed before model inference at `4b9298c`. The blind findings were committed at `7e78937` before unblinding.

The bound Instruct donor source SHA256 is `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`, grouped core SHA256 is `c484a1130e495342d1b8644ff156cc002d0fd68fe12230604e59f35d3af6433a`, and adapted E128 checkpoint SHA256 is `3e39c6557afcd072c8a0f232880f9bd3061baa937b43bc88d48ad68f709a36b5`. The [METH-90 source manifest](meth90_quant_aware_external_manifest.json), SHA256 `70d48b30de55c41283374223ba2b61edaacad396e6f3410c45c166bbaa885937`, fixes eight code, eight PG19 prose and eight technical sources with prior ID/fragment exclusions. All 24 excerpts passed the pre-inference [answerability screen](meth90_answerability_screen.json).

The [automatic audit](meth90_quant_aware_external_audit_result.json), SHA256 `2d3a05a92b3fe28600eebcea08b9079846b2284bfcd51d84403e0af50d49301b`, records every document, prompt, generation and 1,838 PIQA pair. It took 635.45 s on one local RTX 3060, peaked at 2.434 GB allocated GPU memory, and ended at 3.745 GB process RSS.

| Frozen automatic endpoint | Donor | Adapted stored-core student | Decision |
|---|---:|---:|---|
| Pooled document BPB, 98,280 bytes | 1.167631 | 1.163857 | Δ−0.003774; pass |
| Code / prose / technical ΔBPB | — | −0.001932 / −0.004935 / −0.004455 | pass each |
| Greedy EOS, 24 prompts | 23 | 22 | pass |
| Repeated 8-gram 3× / short non-EOS | 0 / 0 | 0 / 0 | pass |
| Full PIQA correct, 1,838 items | 1,291 | 1,287 | −0.218 point; bootstrap lower fifth −0.762 point; pass |
| Donor prompt top-1, 4,318 positions | — | 4,085 = 94.604% | diagnostic; no direct gate |

The [blind A/B file](meth90_blind_semantic_review.json), SHA256 `30b8614e65582ebf5cab24d887332cd44f500e99fba00231ed55d4749164358d`, contains the same 384-character excerpts and both responses without arm labels. The [24-row verdict](meth90_blind_semantic_verdict.json), SHA256 `a1d69e7f15843995be2971e0e68c8ca2a71aa1248618c09dd43371948cc974e4`, records each counted claim and limiting excerpt evidence. It was committed before the [unblinded score](meth90_semantic_score.json), SHA256 `e73086723a0ddca20d98364cb16c344ebf8bb19d50b4fb6d3f0232117eab0ea7`:

| Blind finding | Donor | Student | Frozen gate |
|---|---:|---:|---|
| Unsupported factual claims | 37 | **43** | fail |
| Severe unsupported named-entity, numerical, comparative or causal claims | 15 | 15 | pass |
| Missing requested specific detail | 0 | 0 | pass |
| Ambiguous, excluded | 0 | 1 | descriptive |

The unsupported excess is +4 in code and +2 in technical/general; prose is equal. Both arms make many excerpt-grounding errors on short, often clipped inputs. This is one reviewer's judgment of 24 items, so the exact margin has sampling and rubric sensitivity; under the frozen rule it is a failure, and the verdict is not edited after unblinding. The original BF16+E128 bank passed its separate METH-57 blind audit on different sources. That comparison suggests the stored-core interaction remains important, but it does not causally isolate quantization, adaptation or source difficulty.

The artifact still has no full `engine.c` inference, measured CPU LUT/core DRAM traffic, or ≥50 accepted tok/s result, and it does not validate larger learned expert counts. The next compact-core candidate must specifically address excerpt-grounded response fidelity and pass new independent sources before native promotion. The METH-90 sources and responses are now viewed.
