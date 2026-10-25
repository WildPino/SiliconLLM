# C50-T1 part 2b review, addendum 1 result: a random update of A1's size already fails the fact gates

25 October 2026. Protocol: [C50_T1_PART2B_NULL_PROTOCOL_20261025](C50_T1_PART2B_NULL_PROTOCOL_20261025.md)
(`cec2dda`, pushed before either null was built). Verdict record: `results/c50_t1_null.json` (sha256
`9ae37c80af8fb682…`), written by `c50_t1_verdict.py null` from the records only. Panel runs:
`results/c50_t1_null_run_rand.json` (`0cec2625…`) and `results/c50_t1_null_run_rot.json` (`ac886337…`).

## Verdict

**Outcome 2.** Both nulls are worse than T0 on nq and on tqa by A4's paired rule.
- **What G2 measures:** as written, it measures perturbation. Any update of A1's size fails it, with or without a
  learned map.
- **What follows:** the gates need a null baseline. That is an amendment, and the choice is the user's. Nothing is
  amended here.

| Panel | T0 right | A1: right (b, c), b − c vs limit | N-rand | N-rot |
| --- | ---: | --- | --- | --- |
| nq | 174 | 155 (33, 14): 19 > 13.7, **worse** | 159 (18, 3): 15 > 9.2, **worse** | 163 (20, 9): 11 > 10.8, **worse** |
| tqa | 292 | 272 (25, 5): 20 > 11.0, **worse** | 278 (17, 3): 14 > 8.9, **worse** | 280 (18, 6): 12 > 9.8, **worse** |
| mmlu | 281 | 267 (24, 10): 14 > 11.7, **worse** | 265 (21, 5): 16 > 10.2, **worse** | 266 (19, 4): 15 > 9.6, **worse** |
| arc | 319 | 305 (30, 16): 14 > 13.6, **worse** | 309 (13, 3): 10 > 8.0, **worse** | 314 (16, 11): 5 ≤ 10.4, ok |
| b + c over the four | | 157 | 83 (0.53 × A1) | 103 (0.66 × A1) |
| R_llama (CI95) | | 0.671 (0.569..0.791) | 0.011 (−0.004..0.030) | 0.077 (0.058..0.103) |

b = T0 right and the merge wrong, c = the reverse; "not worse" iff b − c ≤ max(2√(b + c), 1). Both halves of the
donor-gap items stay out of every panel.

**What the nulls show:**
- **N-rand** keeps only A1's rank and per-tensor norm, and fails all four panels. It recovers none of the document gap.
- **N-rot** keeps A1's subspaces and scrambles the learned pairing. It fails three panels. It recovers 0.077 of the
  gap, about a ninth of A1's 0.671, so the subspaces alone carry a little of the gain.
- **A1's net loss on nq and tqa** (19 and 20) is close to the nulls' (11-15 and 12-14).

## Post-hoc, not registered: A1 against the nulls

`results/c50_t1_null_posthoc.json`, from `c50_t1_null_posthoc.py`. It was written after the verdict, and it decides
nothing.

| Panel | lost by A1 / N-rand / N-rot | won by A1 / N-rand / N-rot | lost by A1 and N-rand (expected if independent) | A1 vs N-rand (b, c) | A1 vs N-rot (b, c) |
| --- | --- | --- | --- | --- | --- |
| nq | 33 / 18 / 20 | 14 / 3 / 9 | 11 (3.4) | 23, 19: ok | 25, 17: ok |
| tqa | 25 / 17 / 18 | 5 / 3 / 6 | 9 (1.5) | 18, 12: ok | 20, 12: ok |
| mmlu | 24 / 21 / 19 | 10 / 5 / 4 | 7 (1.8) | 21, 23: ok | 19, 20: ok |
| arc | 30 / 13 / 16 | 16 / 3 / 11 | 5 (1.2) | 26, 22: ok | 30, 21: ok |

Here b = the null right and A1 wrong, c = the reverse.

**A1 is not worse than either null on any of the four panels.**
- **A1 moves more items in both directions.** On nq it loses 33 of T0's items against the nulls' 18-20, and it wins
  14 against 3-9.
- **The nulls partly lose the same items as A1.** On nq they share 10-11 of A1's 33 lost items, against 3-4 if the
  losses were independent. That is consistent with the fragile-item reading of part 2b.
- **This is a comparison chosen after the numbers.** It is on items that have now been read five times (nq and tqa)
  and four times (mmlu and arc). It is a reason to consider an amendment, not a pass.

## Checks

| Check | N-rand | N-rot |
| --- | --- | --- |
| Null construction: largest relative norm error; median \|cos\| with A1's ΔW | 1.3e-9; 0.0004 | 9.1e-10; 0.044 |
| Built from A1's `final.pt` (`05e185ae…`); the merge used the null file | yes | yes |
| BF16 merge, ‖bf16(W + ΔW) − W‖ / ‖ΔW‖ and cosine, medians | 1.005, 0.995 | 1.005, 0.995 |
| Q4_K_M sha256 | `5ca9afe50e0b43d3…` | `793e5dd459a6f3cf…` |
| L against P0 | **PASS**, 100 of 278 differ | **PASS**, 100 of 278 differ |
| After Q4_K_M, ‖ΔQ‖ / ‖ΔW‖ and cosine, medians | 4.43, 0.225 | 4.40, 0.227 |
| Panels, all COMPLETE | nq 850 s, tqa 514 s, arc 1,507 s, mmlu 2,127 s, docs 2,881 s | 685 s, 506 s, 1,836 s, 2,344 s, 2,968 s |
| Planted control (`self`: the merge's ΔW equals A1's) | bit-exact | |

The reader's first call was **VOID**, before it printed any number.
- **What happened:** it read gsm8k, which the null run does not record, and stopped on FileNotFoundError.
- **The fix** (`3db1132`, pushed before the reader ran again): `correctness()` takes the panels to compute.
- **The self-check, repeated** on A1's records passed off as both nulls, reproduces part 2b's numbers and gives
  outcome 2.

## Registered predictions, scored

| Prediction | Measured | Score |
| --- | --- | --- |
| L passes for both, exactly 100 tensors differ | both, 100 | **RIGHT** |
| N-rand R in −0.05..0.05 | 0.011 | **RIGHT** |
| N-rand b + c below 0.5 × A1's | 0.53 × | **WRONG** (just above) |
| N-rand passes all four panels | fails all four | **WRONG** |
| N-rot R in −0.10..0.10 | 0.077 | **RIGHT** (but its CI, 0.058..0.103, excludes 0) |
| N-rot b + c at 0.5-1.0 × A1's | 0.66 × | **RIGHT** |
| N-rot worse than T0 on at least one of nq and tqa | both | **RIGHT** |
| Outcome 2 | 2 | **RIGHT** |

Six RIGHT, two WRONG. The two wrong ones are both on N-rand, and they share one cause.
- **My assumption:** a null would be roughly symmetric on mmlu and arc, flipping fragile right items and fragile
  wrong items alike.
- **What happened:** it is not symmetric. On mmlu, N-rand loses 21 of T0's items and wins 5.
- **The reason I missed:** T0 sits near an optimum of its own loss, so a random step of this size makes it worse on
  average, on every panel.

## What this leaves open: the choice is the user's

The registered next step for outcome 2 is an amendment of G2 and G3, and that choice is the user's.

1. **Amend the gates to a null baseline, and confirm on fresh items** (my recommendation).
   - **The rule:** "not worse than a matched null by A4's paired rule", with N-rot as the null, since it is the
     closer match in effect.
   - **Why fresh items:** A1 passes that rule here, but only post-hoc and on items read many times. A pass needs
     items not read before. Those would be a new draw from the same sources, run on A1 and on N-rot, about 2 × 2.6 h
     locally, no GPU.
2. **Keep the gates as written.** Then no update of this size can pass. The only route left is a much smaller ΔW,
   which would also give up most of R. I expect nothing from this.
3. **Accept the trade as a product decision.**
   - **The gain:** A1 recovers 0.67 of the document gap.
   - **The cost:** a net 19 nq facts of 174 and 20 tqa facts of 292. Roughly 60-80% of that is what a random update
     of the same size costs.
4. **Option B anyway**: T0's own closed-book answers as an anchor, one T4 arm of about 13 h. It aims to beat even the
   null's losses. After this result, it is a refinement and not a fix.

## What this cannot say

- One seed per null and one null of each kind. The null's own b and c have a spread that was not measured.
- "A1's size" is the per-tensor Frobenius norm. N-rot moves 0.66 × as many items as A1, so even N-rot is not fully
  matched in effect.
- Every number here is on items read many times. A general claim needs fresh items.
