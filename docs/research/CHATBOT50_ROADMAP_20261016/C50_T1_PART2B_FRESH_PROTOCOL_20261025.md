# C50-T1 part 2b review, addendum 2: G2 and G3 against a matched null, confirmed on fresh items

25 October 2026. Registered before any model is run on the fresh items. This follows
[addendum 1's result](C50_T1_PART2B_NULL_RESULT_20261025.md), outcome 2: an update of A1's size with no learned map
already fails G2.

**It was decided to amend G2 and G3 (option 1).** The test is now relative to a matched null, and it runs on items
no C50 cell has read.

## Why this cell sits where it does

**The thesis of Phase 1** is capacity added to T0 at fixed active bytes (ROADMAP §0, §5), so the speed stays at v0
while the stored knowledge grows.
- **S1** showed the speed half.
- **S4's capacity arms lost to the LoRA control:**
  - arm (a) rewrote T0's routing and broke 5 of 6 panels;
  - arm (b) broke nothing, but gained little.
- **Decision C** (§12, 24 October) took the LoRA to v0 first (step 1). Capacity on the residual comes after it, with
  new experts that act only where T0 is uncertain and leave T0 intact elsewhere (step 2).

**Addendum 1 showed the fact cost belongs to the edit, not to what is learned.** Any edit of T0's own weights of this
size costs facts, whether learned or random.

**This cell closes step 1 or sends it back:**
- if A1 is no worse than a matched null, the LoRA's cost is the price of editing T0's weights, and the line moves to
  step 2;
- the amended rule is also the one step 2's arms will need.

## The amended gates

**G2′ and G3′:** on each of nq, tqa, mmlu and arc, **A1 is not worse than N-rot** by A4's paired rule.
- **The pairing:** b = N-rot right and A1 wrong, c = the reverse; "not worse" iff b − c ≤ max(2√(b + c), 1).
- **Why N-rot:** it keeps A1's per-tensor norm and its input and output subspaces, and scrambles only the learned
  pairing. It moves 0.66 × as many items as A1, against N-rand's 0.53 ×, so it is the closer match in effect.
- **The other gates are unchanged.** G1 (R 0.671 ≥ 0.559) and the gsm8k and chat parts of G3 stand from part 2b,
  where they held against T0 itself. L stands from part 2b.

## The fresh items

`c50_t1_fresh_tasks.py` builds them, with the same pinned sources, prompts and formats, on rows no panel has used:

| Panel | Rows | Candidates | Dropped: in corpus v1 / duplicate of an old line | Items | tsv sha256 |
| --- | --- | ---: | --- | ---: | --- |
| nq | NQ-open validation 600..1199 | 600 | 0 / 0 | 600 | `2373dc8c…` |
| tqa | TriviaQA rc.nocontext validation 600..1199 | 600 | 0 / 0 | 600 | `61af685f…` |
| mmlu | items 11..20 of each of the 57 subjects | 570 | 2 / 2 | 566 | `30c6eb1a…` |
| arc | ARC-Challenge test 600..1171 | 572 | 0 / 0 | 572 | `8845ad51…` |

- **Contamination check:** an item is dropped if any of its input lines shares a 13-token window holding at least 8
  letter tokens with S3's corpus v1 (31,324,189 distinct windows), the text every T1 arm trained on. This is S3's own
  rule, turned the other way.
- **Its planted control,** run inside the builder: a 64-token span of the corpus is flagged, and a shuffled copy of
  it is not.
- **Where they live:** `E:/.../t1/fresh/inputs/`, with `manifest.json` holding every hash.
- **No donor-gap exclusion is needed.** These are plain draws, not selected on F0 against T0.

## Run, local

`c50_t1_run.py --inputs t1/fresh/inputs --panels nq,tqa,arc,mmlu`, with S2's executables and settings (threads 6,
ubatch 512, no flash attention), on three models in this order:

| Arm | Model | sha256 |
| --- | --- | --- |
| N-rot | `t1/null/t1_null_rot_q4_k_m.gguf` | `793e5dd4…` |
| A1 | `t1/t1_a1_q4_k_m.gguf` | `e490f803…` |
| T0 | P0, `gguf/ling-mini-2.0-a810f64-q4_k_m.gguf` | `44de7315…` |

About 1.5 h per model, 4.5 h in all (ESTIMATE, from addendum 1's walls). It measures quality only, so it is not a
rule-7 run.

**Reader:** `c50_t1_verdict.py fresh --inputs … --t0 … --a1 … --nrot … --json results/c50_t1_fresh.json`, from the
records only.
- **Its checks:** each model's sha256 equals its pin, each run used the fresh input set (manifest hash), and all four
  panels are COMPLETE.
- **Its planted controls,** every call, per panel: the rule passes a copy of N-rot, and fails N-rot with 30 of its
  right items broken.
- **It was checked before this protocol on the old records,** passed off as fresh, and matches an independent
  computation on every panel.

## Rule (applied by the reader)

| Verdict | Condition | Next |
| --- | --- | --- |
| **PASS** | A1 is not worse than N-rot on all four panels | Step 1 of decision C closes. Step 2 (capacity on the residual: added experts that leave T0 intact where they are off) gets its own protocol, pushed before any session |
| **FAIL** | A1 is worse than N-rot on any panel | The loss on that panel goes beyond what an edit of this size costs. The line stops for a review, and option B (T0's own answers as anchor) is back on the table |

Any failed check or planted control is a STOP.

**Reported, not gated:** A1 against T0 and N-rot against T0 on each panel. The second tests whether addendum 1's
finding replicates on fresh items.

## Registered predictions (scored, not gated)

| Prediction | Value |
| --- | --- |
| Verdict | **PASS** |
| N-rot against T0 | worse on at least one of nq and tqa (addendum 1 replicates) |
| A1 against T0 | worse on nq and on tqa (the original G2 failure replicates) |
| A1 against T0, mmlu and arc | worse on at least one |
| Per-model wall | 1.2-1.8 h |

## What this cannot say

- **One null, one seed.** N-rot's own spread is not measured, so a near-threshold result is weak in both directions.
- **This rule is weaker than the original.** "No worse than a random edit of the same size" is a floor, not a claim
  that the facts are kept. The net cost against T0 stays reported.
- **What a PASS clears.** It clears step 1's fact gate for this LoRA at this size. It says nothing about step 2's
  experts, which will be gated the same way against their own matched null.
