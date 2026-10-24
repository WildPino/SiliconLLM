# C50-T1 protocol, part 2b: the anchored LoRA, part 2's first T4 arm

24 October 2026. Registered before any session of it runs on Kaggle and before its trainer has run on a GPU. It
follows [part 2a](C50_T1_PART2A_RESULT_20261024.md) and its registered rule (addendum 1 of
[part 1's protocol](C50_T1_PROTOCOL_20261024.md)): no partial merge of c48 was clean, so the first arm is c48's
LoRA with its loss anchored to T0 where T0 is sure.

## Question

c48 recovers 0.70 of the T0 → F0 document gap and survives the merge into P0's 4-bit layout (MVE-0). But it
breaks closed-book facts T0 had: against T0, nq 147/174 and tqa 269/292 (MVE-0, corrected). Part 2a showed that
the broken items are largely the same whichever half of the LoRA is merged. So the loss looks like it comes from
the training target, not from the modules trained.

**Can the same LoRA keep the gain and stop breaking those facts, if the teacher is not followed where T0 was
already sure of the right token and the teacher was no better?**

## What trains: arm A1, "c48-anchored"

**S4's c48 recipe, unchanged:**
- rank 48 on the same 100 modules: QKV, attention output, the shared expert in layers 1-19, layer 0's dense FFN;
- the same data (block 1, shards 0-469, one epoch in stored order, 8,192 tokens per step, 1,240 steps);
- the same optimizer, schedule, seed and fp16 forward;
- the same identity check at step 0, sessions, checkpoints and memory guard.

The code is `c50_t1_train.py`. It imports S4's trainer and student and changes only the target.

**The anchor.** At each scored position with target token y:
- **anchored** iff p_T0(y) ≥ τ **and** p_T0(y) ≥ p_F0(y). There T0 is sure of the right token and the teacher is no
  better. The target is T0's own distribution: its top-32 and the rest, from the frozen base (LoRA off, no
  gradient) in the same step.
- **free** otherwise: the teacher's top-32 and the rest, exactly as in c48.

The loss is S4's 33-bucket KL, summed over both kinds of position. p_F0(y) is the teacher shard's `lpt`. p_T0(y)
comes from the same PyTorch base the student starts from.

**τ = 0.5**, fixed by MVE-A below.

## MVE-A, measured before this protocol: how much the anchor takes

`c50_t1_anchor_share.py`, record `results/c50_t1_mve_a.json`. Setup:
- the data: S4's validation shards 470-471 (96 documents, 41,073 scored positions), which are never trained on;
- the instrument: T0 = P0 in llama.cpp, with S2's score tool and settings (6 threads, ubatch 512, ctx 8,192,
  flash attention off; 601 s);
- the teacher: p_F0(y), from the shards.

The tool's targets line up with the shards' at every position (asserted).

| τ | anchored share | target-bit gap on anchored positions | on free positions |
| --- | ---: | ---: | ---: |
| 0.3 | 31.5% | −5,280 bits | +30,667 bits |
| **0.5** | **29.4%** | **−3,948 bits** | **+29,335 bits** |
| 0.7 | 27.2% | −2,875 bits | +28,262 bits |
| 0.9 | 24.1% | −1,514 bits | +26,900 bits |

The target-bit gap is the sum over positions of log2 p_F0(y) − log2 p_T0(y): the bits T0 would save on the right
token by matching F0 there. In total it is +25,387 bits. T0 is no worse than F0 on y at 41.6% of positions;
the mean p(y) is 0.606 for T0 and 0.653 for F0.

- **At τ = 0.5 the anchor takes 29.4% of positions.** On those positions, following the teacher would cost T0
  3,948 bits on the right token.
- **The free positions hold all of the positive gap.** By the target bits alone, the anchor gives up none of the
  document gain. The KL also moves the rest of the distribution, so this is a bound on the target, not a
  forecast of R.
- **τ = 0.5 is fixed:** "sure" as the plain majority of the probability. Most anchored positions are much surer:
  24.1% of all positions have p_T0(y) ≥ 0.9.

## Planted controls, before any T4 session (CPU, `c50_t1_train.py --controls`)

At step 0 the student is T0:
- **K-none:** with τ = 2 nothing can be anchored, and the loss must equal S4's loss bit for bit;
- **K-all:** with every position anchored, the loss is KL(T0 ‖ T0) and must be below 1e-5 per position.
- **K-switch:** at step 0, B = 0, so neither K-none nor K-all can tell a LoRA-off switch from a broken one. So B is
  set to random values. The anchor's p_T0(y) must still equal T0's forward (< 1e-6, in fp32, as the anchor computes
  it), and the student must differ from T0 (> 1e-2).

All three must pass, or the trainer is wrong and nothing is pushed. **Run before this protocol was pushed** (`results/c50_t1_controls.json`): on 127 positions,
K-none 68.8702 against S4's 68.8702, bit for bit; K-all 0.0 per position; K-switch 0.0 against T0, with the student
8.13 away. K-switch first failed (1.06e-4) because it compared the anchor's fp32 log-softmax with an fp64 one. That
record is kept (`results/c50_t1_controls_fail_precision.json`), and the rerun compares in fp32.
 On T4, the trainer also repeats S4's step-0
identity check: max |Δlogit| against T0 must be 0, or the session is VOID.

## Run

**Where:** Kaggle T4 × 2, kernels `c50-s4-t1-a1-s1`, `-s2` (task `t1train-all-0.5`, then `t1train-all-0.5+resume`
with `--sources` on the previous session), with the code dataset
re-versioned to hold `c50_t1_train.py` (its manifest is printed in every session). Sessions stop at 9.5 h and
resume from `ckpt.pt` in a `+resume` kernel, as S4's trainings did.

**Time:** c48 took 1,240 steps in 10.3 h. The anchor adds one forward without gradient per step. I expect 25-40%
more, so 13-14.5 h, in two sessions. The first session's log measures it (`s_anchor`, `s`).

**Budget, announced:** ≤ 20 session-hours of T4 × 2.
- Each account has 30 GPU-h per week. The quota cannot be read from the API.
- The account is chosen at push time. A push refused for quota is not a failure: it waits for that account's weekly
  reset, or goes to another account that holds the datasets.
- **Session 1 runs on acct1 (wildpino).** By the S4 kernels' own logs, acct1 used about 18 h since 22 October,
  acct2 about 22 h and acct3 about 24 h. So acct1 has about 12 h left even if its week has not reset, enough for one
  9.5 h session.
- **Session 2** (about 4-5 h) runs on acct1 if its quota allows. Otherwise `ckpt.pt` moves to another account as a
  private dataset, and its sha256 is checked on arrival.
- A session cut short by the quota loses at most the 100 steps since the last checkpoint. If its output is lost
  with it, it is a VOID and is reviewed, as S4's container deaths were.

## Evaluation, local, as in MVE-0

1. Download `final.pt` and record its sha256.
2. `c50_t1_merge.py merge`, then `quantize` with P0's recipe, then `layout` against P0. L must pass with exactly
   100 tensors differing.
3. S2's seven panels with `c50_t1_run.py` (~2.7 h).
4. `c50_t1_verdict.py part2b`, which reads only the records. Every general panel leaves out both halves of the
   donor-gap items (the reader's `both_halves`).

## Gates (A1 is PASS iff all hold)

| Gate | Rule |
| --- | --- |
| **L** | layout as P0, with only the 100 merged tensors differing: v0 by layout |
| **G1, the gain** | R_llama on S4's 36 documents ≥ 0.559, MVE-0's threshold |
| **G2, the facts** | nq and tqa not worse than T0 by A4's paired rule |
| **G3, the rest** | mmlu, arc, gsm8k and chat not worse than T0 by the same rule |

**Reported, not gated:**
- the 206 S4 donor-gap items;
- A1 against c48 (MVE-0's T1) on nq and tqa, and how many of c48's lost items A1 keeps;
- the validation curves: KL to F0, comparable with c48's, plus the anchored KL and the anchored share;
- the step time, with and without the anchor.

## Outcomes

- **PASS:** the merged A1 is T1, an artifact at v0's layout.
  - Next: the timed S0 session on T1 (needs the user's go, rule 7), then step 2's protocol.
  - The cheaper anchored attention-only variant (`--only attn`) becomes optional, not needed.
- **G1 holds, G2 fails:** the anchor is not enough. The line stops for a design review.
  - Options, none chosen now: a lower τ, anchoring every position where T0 beats the teacher, or adding T0's own
    answers on closed-book prompts to the anchor.
- **G2 holds, G1 fails:** the anchor costs the gain. Review.
- **Both fail, or L fails:** review.
- **A VOID session** (identity, non-finite loss, container death) is recorded and reviewed before any rerun.

## Registered predictions (scored, not gated)

| Prediction | Value |
| --- | --- |
| K-none, K-all, K-switch | all pass |
| Step-0 anchored share (PyTorch T0, validation) | 0.27-0.32, close to MVE-A's 0.294 |
| Step time | 25-40% above c48's: 13-14.5 h in all, two sessions |
| G1 | holds: R_llama 0.60-0.72 |
| nq and tqa | A1 loses fewer of T0's items than c48 (33 and 30), about half as many; G2 still fails on at least one of the two |
| G3 | holds |

## What this cannot say

- One arm, one seed, one τ.
- nq and tqa's general items have now been read three times: in MVE-0, in part 2a and here. No training sees them,
  and the only design choice they informed is the registered rule that picked this arm. But a PASS here says that
  A1 keeps these facts. It does not say that the anchor protects closed-book facts in general. That needs fresh
  items.
- The anchor's mask comes from PyTorch T0. MVE-A reads it with llama.cpp T0. MVE-5 puts the two within 0.010 BPB.
