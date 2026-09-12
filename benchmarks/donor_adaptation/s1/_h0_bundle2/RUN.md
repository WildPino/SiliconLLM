# H0 run 3 — the continuation. One T4 session, ≤ 2.8 GPU-h.

**This is NOT a fresh H0.** Run 2 already passed the gate: teacher-forced **111/160** on the CPU
box in fp32, against a registered bar of **48**, with BPB `2.812226 → 0.825358` where the intact
donor reads `0.767595`. Plan: `T4_HEALING_PROPOSAL.md` §11.

## The question this one session answers

Run 2 stopped itself at its 2.8 h wall at **step 500 of 4000**. Finishing the schedule is
~15 GPU-h — **six chained sessions** — and I am not asking for that, because **the watchable
curve was flat across the only interval we can see it**: `tf 110` at step 250, `tf 112` at step
500.

> **Is the curve actually flat, or is 500 steps just early?**

One session takes it to roughly **step 1100**. If tf on the CPU box comes back near 111, the
remaining 3500 steps are not worth six sessions and H0 is done at 111. If it climbs, the six
sessions are worth asking for and I will ask.

## Command

```
python h0_qat.py \
    --factors h0_start.npz \
    --train   h0_train.npz \
    --probe   h0_probe.json \
    --out     h0_trained3.npz \
    --steps 4000 --bs 2 --accum 8 --lr 2e-4 --every 250 --max-hours 2.8 \
    --seed 909091
```

**`--seed 909091` is load-bearing and new.** Batches are drawn i.i.d. from a fixed rng; run 2
used the default `1717`. Without a different seed this run would redraw **the exact sequence run
2 already trained on**, and any gain would be memorisation of a replayed slice. The flag defaults
to `1717` so every earlier run still reproduces bit for bit.

**`--factors h0_start.npz` is run 2's output**, not the original init bundle. Same 280-key
layout, so the loader takes it unchanged — verified by a CPU smoke from this exact file: step-0
watchable tf read **111/160**, matching the CPU fp32 gate, and `G-H0e` fired on the resumed run.

## Two things that are different from run 2, and neither is hidden

1. **Adam moments are not checkpointed.** This restarts them. The first few steps will look
   worse than a true continuation would; that is expected and is not a fault.
2. **`--steps 4000` with `--max-hours 2.8`** means it will again stop on the clock, around step
   ~600 of its own count (≈ step 1100 overall). That is intended — the wall is the budget.

## Stop conditions — stop early and say so

Same as run 2's `RUN.md`, unchanged:

* **`G-H0e FAILS: a master did not move after an update the optimizer APPLIED.`** The trainer is
  not training; send the log.
* **`G-H0e CANNOT BE READ: the GradScaler declined all 25 of the first steps.`** A numerical
  scale problem, not H0's answer; send the log.
* Lines reading `step N DECLINED by the GradScaler` are **normal at warm-up**. Run 2 declined 4
  before applying at step 5, with the scale walking 65536 → 4096, and that is the expected
  behaviour of a cold fp16 scaler.
* The job asserts `sdpa` at load. On a T4, EFFICIENT/FLASH do not exist (Turing) and `eager`
  goes non-finite at `layers.0.self_attn.o_proj`.

## What comes back

`h0_trained3.npz` + `h0_trained3.json`. **Do not run the gate on the GPU.** The printed
teacher-forced count is fp16 and watchable only; the gate is re-measured on the CPU box in fp32
with `h0_eval.py`, which is the same instrument every published number in this programme used.

## Files

`MANIFEST.json` carries the sha256 and size of all four. `h0_start.npz` is
`6d3fd3e3d0a15f5c…`, 352,967,686 bytes — **the only copy of run 2's weights outside this
machine, and they cannot be regenerated.**
