# C50 cell M part 2: result. R_acc separates v0 from T0 on 72 unseen documents

**Status:** RESULT, 28 October 2026. Protocol: [C50_M_PART2_ADDENDUM_20261028.md](C50_M_PART2_ADDENDUM_20261028.md),
pushed before the run (`2de73b7`). A metric test: no outcome moves the thesis.

- **Reading record:** `results/c50_m_part2_verdict.json` (sha256 `d8270d2c…`), from `c50_m_verdict.py --part2`.
- **Records:** `E:/.../m2/` (9 arms, F0, `MANIFEST.json`).

## Verdict

**R_acc SEPARATES v0 FROM T0.** R_acc(v0) is +0.463, CI95 [+0.421, +0.506]. **All 78 STOP checks hold:**
- every arm's weights and temperatures equal part 1's;
- the arms, F0 and llama T0 score the same 151,739 targets;
- torch T0 against llama T0 is +0.0017 BPB, inside the 0.010 bound;
- F0's dump equals its records exactly.

**The denominator exists.** F0 − T0 top-1 is +0.0265, CI95 [+0.0239, +0.0293].

**In raw points,** v0 − T0 top-1 is +0.0123, CI95 [+0.0107, +0.0139]. So v0 recovers 1.2 of the 2.65 top-1 points
that F0 has over T0 on documents none of them saw.

**R_acc stays the primary for R-type readings.** The injection cell's docs top-1 guard stands as written.

## The arms (72 documents, 151,739 positions)

| Arm | τ_nll | R | R_cal (F0 at its τ) | R_cal (F0 at 1) | R_acc | R_acc without F0's top decile | top-1 | top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | 1.325 | | | | | | 0.661 | 0.830 |
| T0 | 1.425 | 0 | 0 | 0 | 0 | 0 | 0.634 | 0.806 |
| c48 | 1.100 | 0.988 | 0.653 | 1.136 | 0.451 | 0.448 | 0.646 | 0.816 |
| c496 | 1.100 | 0.968 | 0.629 | 1.093 | 0.424 | 0.412 | 0.646 | 0.816 |
| a | 1.100 | 0.720 | 0.245 | 0.426 | −0.094 | −0.176 | 0.632 | 0.806 |
| b | 1.200 | 0.666 | 0.347 | 0.604 | 0.225 | 0.218 | 0.640 | 0.812 |
| v0 | 1.150 | 0.933 | 0.642 | 1.116 | 0.463 | 0.458 | 0.647 | 0.817 |
| E2 | 1.125 | 0.958 | 0.649 | 1.129 | 0.458 | 0.456 | 0.646 | 0.817 |
| N | 1.125 | 0.955 | 0.647 | 1.124 | 0.452 | 0.450 | 0.646 | 0.817 |
| N-shuffle | 1.150 | 0.947 | 0.647 | 1.125 | 0.457 | 0.455 | 0.646 | 0.817 |

## The readings (paired document bootstrap, CI95)

| Reading | R | R_cal | R_acc | Change |
| --- | --- | --- | --- | --- |
| S4: a − c496 | −0.248 [−0.282, −0.219] | −0.383 [−0.421, −0.351] | −0.517 [−0.589, −0.456] | holds |
| S4: b − c48 | −0.322 [−0.369, −0.281] | −0.306 [−0.340, −0.278] | −0.226 [−0.271, −0.183] | holds |
| T1 2b: v0 − T0 | +0.933 [+0.899, +0.968] | +0.642 [+0.609, +0.674] | +0.463 [+0.421, +0.506] | holds |
| N − v0 | +0.021 [+0.019, +0.024] | +0.005 [+0.003, +0.007] | −0.011 [−0.025, +0.004] | loses CI under R_acc |
| N − E2 | −0.003 [−0.004, −0.001] | −0.003 [−0.004, −0.001] | −0.006 [−0.025, +0.012] | loses CI under R_acc |
| E2 − v0 (the data-bound reading) | +0.024 [+0.022, +0.027] | +0.007 [+0.005, +0.010] | −0.005 [−0.024, +0.017] | loses CI under R_acc |
| N-shuffle − v0 | +0.013 [+0.012, +0.015] | +0.005 [+0.003, +0.007] | −0.006 [−0.020, +0.009] | loses CI under R_acc |
| N-shuffle − N (the null) | −0.008 [−0.010, −0.007] | +0.000 [−0.002, +0.003] | +0.005 [−0.011, +0.022] | — |

The original rules, with their margins rescaled by v0, pass under none of the three metrics, as in part 1.

## What it says

- **R_acc measures something on unseen documents.**
  - v0's 1.2 top-1 points over T0 are about 8 times the half-width of their CI.
  - The null (N-shuffle − N) stays within ±0.02 of R_acc.
- **R is demoted again on unseen documents.** T0 at one temperature takes 0.66 of v0's R; on part 1's fit documents it
  took 0.64. Under R, N differs from its own shuffle (−0.008, with a CI that excludes 0); under R_cal and R_acc it does
  not.
- **F0 is overconfident too (τ = 1.325), and that changes R_cal's scale.** With F0 at τ = 1, every trained arm's R_cal
  exceeds 1 (v0 1.116). Part 1's R_cal, which kept F0 at 1, was outside the scale, not just above it. With F0 at its own
  τ, R_cal(v0) is 0.642.
- **The gains of N and E2 over v0 exist only in likelihood.**
  - Under R they are +0.021 and +0.024, CI > 0; under R_acc, −0.011 and −0.005, with CIs that include 0.
  - N-shuffle − v0 behaves the same way (+0.013 under R), so R credits even the shuffled branch.
  - **To be reread:** "the binding limit is block 1's teacher data" rested on E2 − v0 under R. On unseen documents, by
    accuracy, E2 does not beat v0. More teacher data on this route is not supported by accuracy.

## Predictions, scored

| Prediction | Measured | Score |
| --- | --- | --- |
| SEPARATES | SEPARATES | hit |
| R_acc(v0) between 0.10 and 0.30 | 0.463 | miss: higher, because F0's top-1 margin over T0 shrank from about 0.06 on the fit documents to 0.027 |
| F0 − T0 top-1 smaller than on part 1's documents, CI excluding 0 | 0.027 [0.024, 0.029] | hit |

## Conduct

- **Four of the nine Kaggle sessions ran twice.** The first runs on acct2 and acct3 mounted code staged days earlier: each
  account stages its own folder, and only acct1's had been restaged. They failed in about 10 minutes on the missing
  documents file, and nothing from them was read.
- **A guard now prevents this.** `c50_s4_kaggle.py verify` compares each account's version with the canonical staging.
  It was tested on both a stale account and a current one.

## What this cannot say

- **A single panel:** 72 documents from one collection, and F0's τ is fit on wiki, edu and math text, not on code.
- **A gain smaller than about 0.02 in R_acc cannot be read** (the null's CI).

## Next (one recommendation, CLAUDE.md rule 7)

**Go on with the injection protocol, from step 1** (F0 on the fresh nq and tqa panel). Its primary is accuracy on facts,
and its docs top-1 guard is now an instrument that discriminates. Nothing here changes that protocol.

**What would contradict this recommendation:** the injection cell's v0 calibration rows failing for a reason this panel
shares. No reason of that kind is known.
