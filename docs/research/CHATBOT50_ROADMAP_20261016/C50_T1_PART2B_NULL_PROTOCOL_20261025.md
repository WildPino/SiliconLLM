# C50-T1 part 2b review, addendum 1: can an update of A1's size without its learned map pass the gates?

25 October 2026. Registered before either null model is built. This follows
[part 2b's result](C50_T1_PART2B_RESULT_20261025.md). That result was FAIL on G2 and G3, and the design review
chose option A: a null control first, then option B if the null passes.

## Question

A1 and c48's T1 lose almost the same items, and on mmlu and arc those are the items T0 holds by a thin margin. So
part of what G2 and G3 count may be what any change of the weights does to fragile items, not what the training
taught.

**If the 100 tensors get a ΔW of A1's size with no learned map, are nq, tqa, mmlu and arc already worse than T0 by
A4's paired rule?**

## The nulls

`c50_t1_null.py` writes, from A1's `final.pt` (sha256 `05e185ae…`), a file with the same keys and shapes. So
`c50_t1_merge.py` merges it exactly as it merged A1. Seed 20261025.

| Null | ΔW per tensor | What it keeps of A1 |
| --- | --- | --- |
| **N-rand** | B′A′, both Gaussian, scaled so that ‖B′A′‖_F = ‖BA‖_F | rank 48 and the Frobenius norm |
| **N-rot** | c·B·Q·A, Q a random 48 × 48 orthogonal matrix, c so that the norm matches | the norm, and A1's input and output subspaces; the learned pairing between them is scrambled |

**Built before this protocol, construction checks only (no panel run):**

| Check | Result |
| --- | --- |
| `self` (B′ = B, A′ = A), the planted control: the merge's ΔW must equal A1's bit for bit | **bit-exact** |
| N-rand: largest relative norm error; median \|cos(ΔW′, ΔW_A1)\| | 1.3e-9; 0.0004 |
| N-rot: the same | 9.1e-10; 0.044 |

The files are under `E:/.../t1/null/`, with sha256 `27a38284…` (rand) and `5b918db1…` (rot).

## Run, local

Per null:
1. `c50_t1_merge.py merge`, then `quantize` with P0's recipe, then `layout` against P0. L must pass with exactly 100
   tensors differing.
2. `c50_t1_run.py --panels nq,tqa,mmlu,arc,docs`, with S2's settings: about 2.6 h per null, 5-6 h in all. The docs
   panel gives each null's R, which shows whether the null carries any of the gain.

**Reader:** `c50_t1_verdict.py null`, from the records only. Both halves of the donor-gap items stay out. It was
checked on A1's own records, passed off as both nulls: it reproduces part 2b's numbers and gives outcome 2. With the
`self` file it gives STOP.

## Outcome rule (applied by the reader)

| Outcome | Condition | Next |
| --- | --- | --- |
| **2** | either null is worse than T0 on nq or on tqa | G2 as written measures perturbation. The gates need a null baseline. That is an amendment, and the choice goes to the user |
| **1** | both nulls pass all four panels, and N-rot moves at least half as many items as A1 (b + c over the four panels ≥ 0.5 × A1's 157) | the loss is specific to the learned map, and the gates stand. Next is option B, with its own protocol pushed before any session |
| **1w** | both pass all four, but N-rot moves fewer than half as many items as A1 | the null is not matched in effect, so a pass says little. Review |
| **3** | both pass nq and tqa, and at least one fails mmlu or arc | G3's failure is reachable by perturbation, G2's is not. Review |

Any failed check (null construction, merge identity, L, panels COMPLETE, model = layout) is a STOP.

## Registered predictions (scored, not gated)

| Prediction | Value |
| --- | --- |
| L passes for both, exactly 100 tensors differ | yes |
| N-rand: R; flips; panels | R in −0.05..0.05; b + c below 0.5 × A1's; passes all four |
| N-rot: R; flips; panels | R in −0.10..0.10; b + c at 0.5-1.0 × A1's; **worse than T0 on at least one of nq and tqa** |
| Outcome | **2** |

The reason for the N-rot prediction:
- **On mmlu and arc a null should be roughly symmetric.** Fragile right items and fragile wrong items both flip.
- **On nq and tqa it should not be.** These are generated, and T0 is right on only 35% and 60% of the items. An item
  T0 gets wrong is rarely one token away from the right answer, so a random change of the weights loses more than it
  gains.

## What this cannot say

- One seed per null.
- "A1's size" means the Frobenius norm per tensor. A trained ΔW and a random one of the same norm need not move
  the outputs equally. N-rot keeps A1's subspaces to come closer, and the flip ratio is reported for that reason.
- The four panels' items have now been read once more (nq and tqa five times). Whatever is concluded about the gates
  is about these items.
