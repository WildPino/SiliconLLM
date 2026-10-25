# C50-T1 part 2b review, addendum 2 result: PASS; on fresh items A1 is not worse than N-rot, nor than T0

25 October 2026. Protocol: [C50_T1_PART2B_FRESH_PROTOCOL_20261025](C50_T1_PART2B_FRESH_PROTOCOL_20261025.md)
(`4e7c2ab`, pushed before any model ran on the fresh items). Verdict record: `results/c50_t1_fresh.json` (sha256
`0fd0f013495887aa…`), written by `c50_t1_verdict.py fresh` from the records only. Panel runs:
`results/c50_t1_fresh_run_nrot.json` (`8c498ac4…`), `results/c50_t1_fresh_run_a1.json` (`7faddbd3…`) and
`results/c50_t1_fresh_run_t0.json` (`5c00104f…`). Inputs manifest `2a57f821…`.

## Verdict

**PASS.** On nq, tqa, mmlu and arc, A1 is not worse than N-rot by A4's paired rule.
- **Step 1 of decision C closes.** A1 stands as v0's LoRA: R_llama 0.671 on S4's documents (part 2b's G1), and its
  facts hold on fresh items.
- **Step 2 is next:** auxiliary experts from F0, attached to T0, with T0 left intact where they are off. It gets its
  own protocol, pushed before any session.

| Panel | n | right: T0 / A1 / N-rot | A1 vs N-rot (gate): b, c; b − c vs limit | A1 vs T0 (reported) | N-rot vs T0 (reported) |
| --- | ---: | --- | --- | --- | --- |
| nq | 600 | 175 / 166 / 168 | 26, 24; 2 ≤ 14.1, **ok** | 28, 19; 9 ≤ 13.7, ok | 16, 9; 7 ≤ 10.0, ok |
| tqa | 600 | 361 / 374 / 364 | 23, 33; −10 ≤ 15.0, **ok** | 22, 35; −13 ≤ 15.1, ok | 20, 23; −3 ≤ 13.1, ok |
| mmlu | 566 | 266 / 283 / 271 | 26, 38; −12 ≤ 16.0, **ok** | 22, 39; −17 ≤ 15.6, ok | 13, 18; −5 ≤ 11.1, ok |
| arc | 572 | 322 / 334 / 319 | 37, 52; −15 ≤ 18.9, **ok** | 34, 46; −12 ≤ 17.9, ok | 21, 18; 3 ≤ 12.5, ok |

b = the reference right and the arm wrong, c = the reverse; "not worse" iff b − c ≤ max(2√(b + c), 1).

**On fresh items neither A1 nor N-rot is worse than T0 on any panel.**
- A1 has 9 fewer right than T0 on nq, inside the rule. It has 13, 17 and 12 more on tqa, mmlu and arc.
- **Addendum 1's finding, that an edit of A1's size fails nq and tqa, does not replicate.** Neither does part 2b's
  G2 failure. The post-hoc below finds why.

## Checks

| Check | Result |
| --- | --- |
| Model sha256 equal to the pins (T0 `44de7315…`, A1 `e490f803…`, N-rot `793e5dd4…`) | all three |
| Each run used the fresh inputs (manifest hash) | all three |
| Panels COMPLETE | all twelve |
| Planted controls, per panel: a copy of N-rot passes; N-rot with 30 right items broken fails | fired on all four |
| Walls, sum of the four panels | N-rot 4,971 s, A1 4,494 s, T0 4,592 s (chain 16:43 → 20:42) |

## Registered predictions, scored

| Prediction | Measured | Score |
| --- | --- | --- |
| Verdict PASS | PASS | **RIGHT** |
| N-rot worse than T0 on at least one of nq and tqa | ok on both | **WRONG** |
| A1 worse than T0 on nq and on tqa | ok on both, and 13 more right on tqa | **WRONG** |
| A1 worse than T0 on at least one of mmlu and arc | ok on both, 17 and 12 more right | **WRONG** |
| Per-model wall 1.2-1.8 h | 1.38, 1.25, 1.28 h | **RIGHT** |

**Two right, three wrong.** All three wrong ones came from expecting the old panels' losses to replicate.

## Post-hoc, not registered: the old panels censored the wins

`results/c50_t1_fresh_posthoc.json`, from `c50_t1_fresh_posthoc.py`. It was written after the verdict, and it decides
nothing.

**What the old general panels left out.** Every general panel of S4, T1 MVE-0, part 2a, part 2b and addendum 1
excludes S2's donor-gap items: the 409 items F0 gets right and llama.cpp's T0 gets wrong.
- **These are the items where an update toward F0 can win against T0.**
- **Leaving them out is a selection on T0's outcome.** It removes part of c (arm right, T0 wrong) and none of b
  (T0 right, arm wrong).
- **So the paired rule against T0 was biased toward "worse".** Any edit that flips items in both directions loses
  on such a panel, learned or random.

The same old records, read twice: with the donor-gap items out, as the gates read them, and back in, which gives
the original unselected 600-item draw.

| Panel | Donor-gap items | T0 right | A1 (b, c): out → in | N-rand (b, c): out → in | N-rot (b, c): out → in |
| --- | ---: | ---: | --- | --- | --- |
| nq | 98 | 174 | (33, 14) **worse** → (33, 28) ok | (18, 3) **worse** → (18, 13) ok | (20, 9) **worse** → (20, 15) ok |
| tqa | 117 | 292 | (25, 5) **worse** → (25, 23) ok | (17, 3) **worse** → (17, 17) ok | (18, 6) **worse** → (18, 13) ok |
| mmlu | 79 | 281 | (24, 10) **worse** → (24, 30) ok | (21, 5) **worse** → (21, 12) ok | (19, 4) **worse** → (19, 10) ok |
| arc | 105 | 319 | (30, 16) **worse** → (30, 58) ok | (13, 3) **worse** → (13, 14) ok | (16, 11) ok → (16, 21) ok |

- **The "out" column reproduces part 2b's and addendum 1's numbers exactly.** That shows the reread uses the same
  records and the same scoring.
- **Putting the items back changes only c, as it must.** On those items T0 is wrong, so b cannot grow.
- **On the full draw, every arm is not worse than T0 on every panel.** That matches the fresh items, which are a plain
  draw with no such exclusion.

**What this corrects:**
- **Part 2b's reading,** "A1 loses facts on nq, tqa (G2) and mmlu, arc (G3)": it came from the panel's construction.
  On a plain draw, old or fresh, A1 is not worse than T0.
- **Addendum 1's reading,** "the fact cost belongs to the edit": also a property of the censored panel. A random edit
  of A1's size is not worse than T0 on a plain draw either.
- **The records and verdicts stay as written.** Part 2b's FAIL was correct under its registered rule. The rule's panel
  was the flaw. This result supersedes those two readings only.

**S4's arms, partly uncensored.** S4's evaluator produced only S4's half of the donor-gap items, so the "in" column
here holds 40-59 of them per panel instead of 79-117. T0 is the PyTorch T0, which gets some of these items right.

| Panel | c48 out → S4 half in | c496 | arm (a) | arm (b) |
| --- | --- | --- | --- | --- |
| nq | (35, 9) **worse** → (37, 14) **worse** | (37, 12) **worse** → (39, 16) **worse** | (56, 16) **worse** → (56, 20) **worse** | (11, 8) ok → (11, 11) ok |
| tqa | (25, 11) **worse** → (27, 18) ok | (32, 12) **worse** → (34, 20) ok | (54, 9) **worse** → (56, 15) **worse** | (15, 8) ok → (20, 9) **worse** |
| mmlu | (23, 20) ok → (24, 28) ok | (20, 20) ok → (21, 30) ok | (40, 18) **worse** → (40, 28) ok | (17, 12) ok → (19, 18) ok |
| arc | (32, 32) ok → (34, 50) ok | (32, 30) ok → (34, 50) ok | (43, 26) **worse** → (44, 40) ok | (23, 15) ok → (25, 26) ok |
| gsm8k | (4, 4) ok → (4, 4) ok | (2, 6) ok → (2, 6) ok | (19, 1) **worse** → (20, 2) **worse** | (2, 4) ok → (3, 7) ok |

- **Arm (a)'s regression survives on nq, tqa and gsm8k.** It loses 54-56 of T0's items on nq and tqa, two to three
  times the LoRA controls. So S4's conclusion stands in direction: arm (a)'s new expert slots competing for T0's top-8
  broke things.
- **Its mmlu and arc failures do not survive.** "5 of 6 panels" overstates it, so the count is not to be quoted.
- **c48 and c496, S4's LoRA controls, stay worse on nq** with S4's half back in. That panel is still half
  censored, so it is not read further here.

## What this means for the thesis

**Step 1 is closed.** v0 is T0 plus the A1 LoRA:
- it recovers 0.671 of the document gap to F0;
- its facts hold on fresh items;
- and, as a LoRA merge, it reads the same bytes per token as T0.

**For step 2, two things change:**
- **The fact panels must be plain draws:** fresh items, with no exclusion selected on T0's or F0's outcome. The
  donor-gap items are reported apart, from a disjoint set. With that, the gate can again be "not worse than v0"
  directly. A random-donor control stays mandatory, for the gain rather than for the facts.
- **The argument for an additive branch changes.** It is no longer "editing T0's weights costs facts", since this
  result removes that. What remains is the thesis itself, stored capacity beyond what a rank-48 LoRA holds, plus
  S4's arm (a), whose competing slots did break facts.

## What this cannot say

- **One null, one seed, one LoRA.** A1's 9-item deficit on nq is inside the rule, but the rule is wide at these
  counts.
- **The post-hoc tables are chosen after the numbers.** They explain the old verdicts; the fresh items, registered
  before the run, are the test.
- **The fresh items were not run on F0,** so how many donor-gap items they hold is not known. They are a plain draw
  in the protocol's sense only.
