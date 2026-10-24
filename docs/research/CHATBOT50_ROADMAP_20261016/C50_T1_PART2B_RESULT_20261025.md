# C50-T1 part 2b result: the anchored LoRA keeps the gain and loses the same facts

25 October 2026. Protocol: [C50_T1_PART2B_PROTOCOL_20261024](C50_T1_PART2B_PROTOCOL_20261024.md) (`342a05c`, pushed
before any T4 session). Verdict record: `results/c50_t1_part2b.json` (sha256 `69616e1da0c68e33…`), written by
`c50_t1_verdict.py part2b` from the records only. Panel run: `results/c50_t1_a1_run.json` (`88204bd4…`). The
artifacts are under `E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/t1/`.

## Verdict

**FAIL on G2 and G3. The line stops for a design review.**
- **The gain holds:** R_llama 0.671, above the 0.559 threshold (G1).
- **The facts do not:** A1 is still worse than T0 on nq and tqa (G2).
- **Two more panels fail:** mmlu and arc, which c48's T1 passed, now fail the same rule (G3).

| Gate | Rule | A1 | c48's T1 (MVE-0) | Held |
| --- | --- | --- | --- | --- |
| L | P0's layout, only the 100 merged tensors differ | 100 of 278 differ | 100 of 278 | **yes** |
| G1 | R_llama ≥ 0.559 | **0.6711** (CI95 0.569..0.791) | 0.6970 (0.589..0.824) | **yes** |
| G2 | nq not worse than T0 | 174 / 155 (b 33, c 14): b − c 19 > 13.7 | 174 / 147 (33, 6) | **no** |
| G2 | tqa not worse than T0 | 292 / 272 (25, 5): 20 > 11.0 | 292 / 269 (30, 7) | **no** |
| G3 | mmlu | 281 / 267 (24, 10): 14 > 11.7 | 281 / 272 (24, 15), ok | **no** |
| G3 | arc | 319 / 305 (30, 16): 14 > 13.6 | 319 / 315 (28, 24), ok | **no** |
| G3 | gsm8k | 81 / 81 (2, 2) | 81 / 78 (5, 2), ok | yes |
| G3 | chat | 28 / 27 (3, 2) | 28 / 28, ok | yes |

Counts are T0 right / A1 right (b = T0 right and A1 wrong, c = the reverse). "Not worse" is A4's paired rule: b − c ≤
max(2√(b + c), 1). Every general panel leaves out both halves of the donor-gap items.

**Reported, not gated:**
- **The 206 S4 donor-gap items:** A1 gets 42 right, T0 none. c48's T1 got 46.
- **A1 against c48's T1**, same rule:
  - nq: A1 is not worse (b 7, c 15);
  - tqa: A1 is not worse (4, 7);
  - mmlu: not worse (8, 3);
  - arc: **worse** (15, 5).

## What the anchor changed, and what it did not

**It did not change which facts break.** The post-hoc record `results/c50_t1_part2b_posthoc.json` (not
registered) lists, per panel, the items T0 gets right and each merge gets wrong:

| Panel | T0 right | lost by A1 | lost by c48's T1 | lost by both | expected if independent |
| --- | ---: | ---: | ---: | ---: | ---: |
| nq | 174 | 33 | 33 | **27** | 6.3 |
| tqa | 292 | 25 | 30 | **23** | 2.6 |
| mmlu | 281 | 24 | 24 | **23** | 2.0 |
| arc | 319 | 30 | 28 | **23** | 2.6 |

**What it changed is the other side: the items T0 gets wrong and the merge gets right.**
- **mmlu and arc:** A1 wins fewer of these than c48 did, 10 against 15 on mmlu and 16 against 24 on arc. Its losses
  are the same as c48's, so with fewer wins to offset them it now fails the rule.
- **nq:** A1 wins more, 14 against 6. That is the whole of its improvement over c48 there. It still loses the same 33
  items.

**The lost items are T0's fragile ones.** On mmlu and arc, the likelihood panels, a margin can be read for each
item: the runner-up choice's mean NLL per token minus the label's. Here it is for T0, on the items T0 gets right:

| Panel | margin of the items A1 loses (q25 / median / q75) | margin of the items A1 keeps | kept items below the lost median |
| --- | --- | --- | ---: |
| mmlu | 0.165 / **0.251** / 0.491 | 0.541 / **1.272** / 2.063 | 11.7% |
| arc | 0.095 / **0.307** / 0.530 | 0.610 / **1.240** / 2.026 | 12.8% |

The items that flip are the ones T0 barely gets right.
- **Overlap is the expected outcome.** Any change of the weights of this size flips the same items first, so the
  overlap between c48, A1 and part 2a's two halves is what fragility predicts.
- **Part 2a's reading must be weakened.** There, the overlap was read as "item-level and systematic, not noise from
  moving the weights". On mmlu and arc that does not follow. The margin was not measured on nq and tqa, whose items
  are generated, not scored.
- **The direction is still real.** On nq and tqa, b exceeds c clearly for every merge built so far.

**This is what the anchor did not reach.** It acts on the training documents, at positions where T0 is sure. The
facts that break are closed-book answers that T0 holds by a thin margin. Nothing in the training target touches
those items directly.

## Checks

| Check | Value |
| --- | --- |
| Training | 1,240 of 1,240 steps, two sessions (acct1 then acct2), 48,132 s; step-0 identity max \|Δlogit\| 0 |
| Checkpoint moved between accounts | sha256 `cbe0ad17…` on both sides (printed by the kernel on arrival) |
| `final.pt` | sha256 `05e185aed7f6b752…` |
| BF16 merge, ‖bf16(W + ΔW) − W‖ / ‖ΔW‖ and cosine, medians | 1.005, 0.995 |
| Q4_K_M sha256 | `e490f80351e336c1…` |
| L (`layout`): header and layout as P0, data differs only in the merged tensors | **PASS**, 100 of 278 differ |
| After Q4_K_M against P0, ‖ΔQ‖ / ‖ΔW‖ and cosine, medians | 4.42, 0.226 (product 1.00, as in MVE-0) |
| Panels | all seven COMPLETE (nq 539 s, tqa 437 s, gsm8k 1,444 s, arc 1,439 s, mmlu 1,949 s, docs 2,411 s, chat 173 s) |
| A1 session 2, first push | VOID at startup: the moved checkpoint was not under `train/`. The guard stopped it before any training, and it was fixed and pushed again (§12) |

## Registered predictions, scored

| Prediction | Measured | Score |
| --- | --- | --- |
| K-none, K-all, K-switch all pass | all pass | **RIGHT** |
| Step-0 anchored share 0.27-0.32 | 0.327 | **WRONG** (0.007 above) |
| Step time +25-40%, 13-14.5 h, two sessions | +29%, 13.4 h, two sessions | **RIGHT** |
| G1 holds, R_llama 0.60-0.72 | 0.671 | **RIGHT** |
| A1 loses about half as many of T0's nq and tqa items as c48 (33, 30); G2 still fails on at least one | 33 and 25 lost; G2 fails on both | **WRONG** on the size (none fewer on nq, 17% fewer on tqa); right that G2 fails |
| G3 holds | mmlu and arc fail | **WRONG** |

Three RIGHT, three WRONG. The wrong ones share one cause: I assumed the lost facts came from following the teacher
where T0 was sure. They do not move with that.

## Design review: what this leaves open (nothing chosen here)

The registered outcome is "G1 holds, G2 fails: the anchor is not enough", and G3 fails as well. The protocol named
three options for that case: a lower τ, anchoring every position where T0 beats the teacher, and adding T0's own
answers on closed-book prompts. The margin result changes how they rank.

1. **A null control first, local, about 4 h, no GPU.** Merge a random ΔW with A1's per-tensor norms into the same
   layout, and run the four failing panels.
   - **If the null alone fails G2 or G3**, then a trained update of this size cannot pass the gates as written. The
     gates need a baseline that includes perturbation, for example the null's own b and c.
   - **If the null passes**, the loss is specific to the training, and the gates stand.
   - It is the planted control the general-panel gates never had.
2. **T0's own answers as an anchor**, on closed-book prompts outside every panel. This targets the fragile items
   directly. It is one T4 arm, about 13 h.
3. **A lower τ, or anchoring every position where T0 ≥ F0.** The table above says the losses do not move with the
   anchor on the training documents. I expect little from these.
4. **Accept the trade, as a product decision.** c48's or A1's T1 recovers 0.67-0.70 of the document gap and costs
   a net 19-27 closed-book facts on nq (of 174) and 20-23 on tqa (of 292). That decision is the user's, not a gate's.

My recommendation is 1, then 2 if the null passes.

## What this cannot say

- One arm, one seed, one τ.
- nq and tqa's general items have now been read four times, mmlu's and arc's three times. A PASS by a later arm on
  these items needs fresh items to say anything general.
- The margin analysis is post-hoc, on mmlu and arc only.
