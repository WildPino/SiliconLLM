# BRIEF H4D — do the trained fp32 masters hide quality behind ternarization?

**Preregistered descriptive CPU diagnostic, before reading the H4D arm.** H4 Stage A's
terminal ternary rank-48 q/o model is already adjudicated and may not be rerun to a
pass. This brief changes exactly one inference-time operation on its *same trained
checkpoint*: replace `ternarize(A) · diag(s) · ternarize(B)` with the saved fp32
masters `A · diag(s) · B`. No weights, rank, donor, slice, optimizer state or
evaluation metric change. No training or GPU is authorized by this brief.

## Question and anti-duplication

H4 terminal reads 1.15373752359353 BPB, 66/160 teacher-forced and 6/160 free.
E20 established that first argmax errors can explain the free-running collapse;
with the crude per-position survival estimate `p=66/160`, its five-prompt
32-token geometric model predicts about 8.51 free tokens, not far from six.
This is an interpretation, **not** a new gate or proof of independence.

The unmeasured, decision-relevant cell is the *latent fp32 masters in H4's
terminal checkpoint*. E65's QO-48 fp32 result used post-hoc factors, not the
same trained masters; H0 used rank 512. This diagnostic distinguishes whether
the present checkpoint loses most of its remaining quality at the ternary
projection, versus already being limited in its learned low-rank geometry.
It does **not** estimate what a separately trained fp32 model would do.

## Frozen inputs and apparatus

- Donor `Qwen/Qwen2.5-1.5B`, revision `8faed761d45a263340a0528343f099c05c9a4323`.
- Stage A terminal `h4_trained.npz` SHA-256
  `11a483419e6b8795164e4651547f6b3765fc325adaa3283153d9cd5276d4cefb`;
  metadata must say `TIME_CAP`, `terminal_checkpoint:true`, step 1148,
  1145 applied and origin factors SHA-256
  `7ae0d2ceeff628ff9b076b5df5806def54340f193e6cae5eb8e8013817959371`.
- Canonical terminal ternary CPU evaluation SHA-256
  `23c3792f4a04efedf90ec0617654a529f0e2a115419f55cd71ad0d55d48e48f2`;
  init CPU evaluation SHA-256
  `140a8968ccc49ebe4bdf8518bd6f901b0b5c16e76ed5df87a069bc2f7c1e461c`.
- Frozen `h4_eval.py` SHA-256
  `0b536b46a23e2290172824d139c8564378331e98eb2cb93a1eccbf9a3d9e59b3`;
  `h4_qat.py` SHA-256
  `8a3b5c7be26e3b21f9c223d69706fb36c5ed7d41b2d260dfed70b8e1778b7bb7`.

The separate `h4_latent_diag.py` runner reuses the unchanged frozen CPU evaluator,
substituting only `TernaryLowRank.forward` so `A` and `B` bypass `_quant()`.
It must assert committed-runner identity before loading the donor. The original
H4 evaluator still checks the terminal metadata, checkpoint hash, model/revision,
origin init hash and frozen slice. The diagnostic output must be distinct from
both H4 canonical controls and the ternary terminal result. No best-checkpoint
selection, interim factors or quantized-activation kernel is involved.

## Readout and interpretation fixed before the arm

Report BPB, free and teacher-forced exact counts of 160, mean/median rank,
rank≤5, per-prompt scores, and the signed differences from the fixed terminal
ternary result. `latent − ternary`: negative BPB and positive fidelity indicate
that ternarization of these saved masters costs quality *on this model*;
the opposite signs show the latent masters are not a usable hidden fp32 rescue.
Differences are descriptive; **no H4 or H4D pass/fail gate** and no new Stage B
license are created. The existing H4 bands must not be recomputed or presented
as if this fp32 arm were the trained ternary deliverable.

Prediction, fixed now: fp32 latent will lower BPB by at least 0.05 and raise
teacher-forced fidelity by at least 10 positions, but will remain below
`GENERATOR_PARTIAL` (>14 free). Score each clause separately. A free score above
14 would be informative but **not** an H4 gate: the fp32 latent model has a
different weight format and has not been timed on `engine.c`.

No quality result here transfers to a 10B pretrained model, its conversion,
its memory footprint or its token rate. A later training or speed study needs
its own preregistration and resource decision.
