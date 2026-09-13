# H1 — two T4 sessions, and the second is not optional

**Budget: 2 x 2.8 GPU-h = 5.6 GPU-h.** Brief:
`docs/research/donor_adaptation/briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md`.

## The question

H0 answered *can gradients move this object at all* — yes, on attention. H1 asks the question
that decides whether the FFN carve is a real format or only a post-hoc approximation: **is the
carve TRAINABLE, or merely APPLIED?** Every carve measured so far (E23, E37, E38, E40) was
applied to a donor that never trained under it. H1 trains 8 layers under the carve and compares
against the matched post-hoc control on the very same 8 layers.

## Commands — session 1, then session 2

```
python h1_qat.py \
    --factors h0_trained3.npz \
    --labels  labels_E256.npz \
    --stats   h1_actstats.npz \
    --train   h1_train.npz \
    --heldout h1_heldout.npz \
    --calib   h1_calib.npz \
    --out     h1_trained_s1.npz \
    --router-lr 0.0003 --aux 0.01 \
    --steps 4000 --bs 2 --accum 8 --lr 2e-4 --every 250 \
    --seed 1717 --max-hours 2.8
```

Then, **from the bundle session 1 produced**:

```
python h1_qat.py \
    --factors h0_trained3.npz \
    --resume  h1_trained_s1.npz \
    --labels  labels_E256.npz \
    --stats   h1_actstats.npz \
    --train   h1_train.npz \
    --heldout h1_heldout.npz \
    --calib   h1_calib.npz \
    --out     h1_trained_s2.npz \
    --router-lr 0.0003 --aux 0.01 \
    --steps 4000 --bs 2 --accum 8 --lr 2e-4 --every 250 \
    --seed 2718 --max-hours 2.8
```

**`--factors` stays H0's bundle in BOTH sessions and `--resume` is what continues the run.**
They are different things: `--factors` installs the frozen q/o attention organs, `--resume`
continues the FFN masters and the router. Passing an H1 bundle to `--factors` would install
zero attention organs; the trainer now refuses outright rather than doing it quietly.

**The seed differs on purpose** (1717 then 2718). Batches come from a fixed rng, so a
continuation on the same seed would retrain the same draws. Adam moments are not checkpointed;
the restart is recorded in the output json.

## Why two sessions, and why the second is not padding

H0's run 1 looked flat and run 3 — same object, more steps — removed 97.9% of the damage. A
single 2.8 h session would have declared H0 dead. Brief §7.

## Stop conditions — stop early and say so if any of these happens

1. **The script refuses to start** with `--router-lr and --aux are NOT SET`. It means the flags
   above were dropped from the command. They are not optional and must not be guessed.
2. **`G-H1c` aborts** — `a master did not move after an update the optimizer APPLIED`. The
   trainer is not training; any "no effect" reading would be an artefact. Send the log.
3. **`G-H1b CANNOT BE READ`** — the GradScaler declined all 40 first steps. That is a
   numerical-scale failure of the trainer and **not** H1's answer. Send the log; the fallback is
   bf16, which this card reports as supported.
   Individual `step N DECLINED by the GradScaler` lines at warm-up are **normal**, not errors.
4. **`nonfinite` climbs above ~1% of microbatches.** fp16 loss scaling is not holding.
5. **The job dies on attention.** The script asserts `sdpa` at load. Do not work around it:
   `eager` fp16 on a T4 goes non-finite at `layers.0.self_attn.o_proj`.

Otherwise let each session run to its time cap.

## What comes back

**`h1_trained_s1.npz` + `.json`, and `h1_trained_s2.npz` + `.json`.** Bring all four (~353 MB
each npz).

**DO NOT PICK A CHECKPOINT.** The trainer overwrites its output every 250 steps so an
interrupted session is still usable, and it prints a held-out BPB so the run is watchable —
**but that BPB is measured on the gate's own frozen slice.** Choosing the best checkpoint by it
would be selection on the evaluation set and would void `G-H1`. Bring back the last file each
session wrote, whatever its printed number says.

**The gate is NOT decided on the GPU.** `h1_eval.py` re-measures it on the CPU box in fp32 with
the instrument every published number in this programme used.

## The gate, fixed before the run

| | held-out BPB, CPU fp32, frozen 24x512 slice |
|---|---|
| donor, untouched | 0.767595 |
| H1's start line — H0 run 3 | 0.810022 |
| the 8 layers ternarised, NOT carved | 0.947851 |
| **`applied-8L` — the matched post-hoc control** | **1.096636** |
| **`G-H1` passes iff trained-8L** | **< 1.096636** |

Ordinal, no tolerance. The bands below it are `TRAINING-HELPS` [0.947851, 1.096636),
`CARVE-IS-TRAINABLE` (0.810022, 0.947851), `CARVE-IS-FREE` <= 0.810022.

## What is already known to be against us, stated before the hours are spent

The CPU router smoke (11.7 h, all 8 layers, addendum E) shipped `--router-lr 0.0003 --aux 0.01`
**with a stated shortfall**: it beats a STATIC selection on **2 of 8 layers**, and in total
error mass it is **2.27% worse** than STATIC. `G-H1e` is therefore **expected to fail**, and
that expectation is registered. It does not block H1, because H1's primary gate is `G-H1` — the
experts — and the smoke is a frozen-expert proxy that is structurally blind to the one thing
joint training changes: which experts receive gradient.

## What H1 is NOT

- **Not the 10 B goal.** This is a 1.5 B donor and 8 of its 28 layers.
- **Not additive to 28 layers.** Measured: ternarisation costs +2.708 BPB over 28 layers and
  +0.138 over these 8. Nothing measured here may be multiplied up.
- **Not a speed measurement.** Training computes the full FFN and masks it, so it buys quality
  information and not speed. No timing is taken and none may be quoted from it.
