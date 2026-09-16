# E67 — output-aware selection does not rescue the frozen 1.17% FFN carve locally

**Result:** `NO_LOCAL_SIGNAL`, 2026-09-16. The frozen
[brief](../briefs/BRIEF_E67_OUTPUT_AWARE_SELECTION.md) and its float64 addendum
were committed as `03667e2` / `93b0025` before any donor run. The
[runner](../../../../benchmarks/donor_adaptation/engine/e67_output_aware_selection.py)
was committed as `c42b4e2` before the one-layer smoke or full measurement.
Neither a T4 nor a speed measurement was used.

## The distinct question

E38's `k=3/256` "oracle" chooses groups by `sum(z_i²)` for the FFN's SwiGLU
intermediate. That is not an output- or loss-optimal selector. E67 asked
whether a selector that sees every group's *actual* contribution
`v_g = W_down[:,g] z_g` could reduce the local FFN reconstruction error at the
**same 105 selected neurons/token/layer**. It tested a greedy reduction of
`||W_down z - sum_selected v_g||²`, not an exact subset optimum or an
executable low-byte router. A pure-torch planted counterexample proves the
mass-to-output inference is invalid in general; this real-donor measurement
tests whether the difference is large *here*.

## Identity and controls

Pinned Qwen2.5-1.5B revision `8faed761d45a263340a0528343f099c05c9a4323`,
fp32/eager, intact weights. The existing 24×512 held-out slice reproduces
`ids_sha256=a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`;
sequences 0–3, positions 384–511, make **512** local observations at each
of D0's layers 1, 7, 14, 21 and 27. D0c's `E=256` coactivation labels and
stored random equal partition are unchanged. The JSON records the selected-ID,
label, runner and dependency hashes; the result file itself is SHA-256
`98d1b63ec6b259dbf1902d01e696fc78199e4709e244cbbae4e4ca2c4b4a33aa`.

All mandatory controls fired. The planted unequal-column, cancellation and
float64 tie tests passed before the donor loaded. For every layer, the 256
contributions reconstruct `W_down z`; maximum relative vector error was
`4.0211e-7`, far inside `1e-5`. Every sparse arm selected exactly three
distinct 35-neuron groups (105 neurons); each dense-identity arm covered all
256. The float64 mass selector had **zero boundary ties** on the sampled
tokens. Independent re-summation of the five layers' SSE reproduced the
runner's aggregate reduction exactly. The runner and D0c-label SHA-256s in
the result JSON match the files in this worktree.

## Measured local reconstruction

The table reports `sum ||y-yhat||² / sum ||y||²` for each layer, **not** BPB
and not the square-root `relerr` used elsewhere in the programme.

| layer | mass top-3 | output-norm top-3 | output-greedy top-3 | random-label output-greedy | greedy SSE reduction vs mass |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.555243 | 0.546102 | 0.539110 | 0.575200 | 2.906% |
| 7 | 0.824307 | 0.823499 | 0.812891 | 0.841695 | 1.385% |
| 14 | 0.892013 | 0.884344 | 0.854483 | 0.881594 | 4.207% |
| 21 | 0.739830 | 0.737637 | 0.730527 | 0.776032 | 1.257% |
| 27 | 0.294530 | 0.293609 | 0.281343 | 0.650673 | 4.477% |

No verdict depends on the output-norm arm; full precision for every arm is
in the machine-readable result.
The actual summed SSEs are **7,988,513.931826** for mass and
**7,650,716.937602** for output-greedy: ratio `0.9577146642`, or
**4.22853358%** reduction. Per-layer reductions are only **1.26–4.48%**;
none reaches the registered 20% partial-signal bar. Output-greedy improves
many individual positions but the *amount* recovered is small. The frozen
band is therefore **`NO_LOCAL_SIGNAL`**, not a borderline call.

## What this rules out, and what it does not

This rules out the **particular post-hoc, output-greedy selector** as a large
local reconstruction rescue at `k=3` on the frozen D0c partition, these five
layers and this held-out sample. In this comparison, E38's flawed use of the
word "oracle" was corrected, but switching its ranking target from activation
mass to greedy FFN-output error still leaves most local error intact. There
is no basis to spend a full-model BPB/rank run or T4 on *this selector alone*
under the registered stopping rule.

It is **not** an exact optimal `k`-subset search, a loss-aware selector, a
trained router, a test of joint training, or the proposed shared low-rank
`S(x)` plus block-sparse residual. In particular, adding a shared path changes
the residual that block selection must explain; E67 does not bound that
architecture. Local error is not end-to-end BPB or generation. The donor is
1.5B, not a pretrained 10B, and no `engine.c` artifact or tok/s follows.

Operationally, the full CPU run took `38.33 s` with six intra-op threads and
peaked at `9.21 GiB` process memory. These are not performance measurements of
an inference architecture: every selector computed the entire dense FFN first.

Canonical artifacts:

- [full result JSON](../../../../benchmarks/donor_adaptation/engine/results/e67_output_aware_selection.json)
- [one-layer smoke JSON](../../../../benchmarks/donor_adaptation/engine/results/e67_output_aware_selection_smoke.json)
- [frozen brief](../briefs/BRIEF_E67_OUTPUT_AWARE_SELECTION.md)
