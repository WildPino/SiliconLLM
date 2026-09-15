# H2I Phase A — one byte is inert; the matched carve hole is trainable-sized and rank-visible

**Brief:** `briefs/BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md` + addendum A  
**Runner:** `s1/h2i_applied.py`, commit `c4a156f`  
**Result:** `s1/results/h2i/h2i_applied_8L.json`, sha256
`c367acb45113e9ce68fb898de0b0833db61b6a1b84990b7ccc6467f82d2ef8cc`  
**Donor:** Qwen2.5-1.5B rev `8faed761`; H0 run-3 q/o; layers 3,6,9,12,15,18,21,24;
E=256, hard k=16, E37 router; CPU fp32  
**Cost:** 971 s CPU; no GPU and no timing claim

## Verdict

> **`PHASE_B_ELIGIBLE`.** On the exact H1 eight-layer scope, replacing only the FFN forward
> quantizer by one-byte R8 changes H0 by `−0.000016893 BPB`. Applying the hard carve then costs
> `+0.209070870 BPB`. The byte conversion is inert; selection is the live trainable object.

| arm | BPB |
|---|---:|
| intact control | 0.767594964 |
| H0 run 3 | 0.810022488 |
| R8 8L, k=E | **0.810005595** |
| R8 8L, E37 hard k=16 | **1.019076465** |

Both registered launch gates fire: one-byte absolute delta `0.000016893 <= 0.01`; carve delta
`0.209070870 > 0.05` with the required ordinal sign. The result is write-once and the published
H1 rows were hash-validated rather than remeasured.

## The rank check changes how the win must be judged

The applied R8 arm agrees with H0's own freshly captured continuations at only **9/160**
free-running positions, per prompt `[6,0,0,0,3]`. Under H0's fixed context it gives **99/160**
teacher-forced top-1, mean target rank **3.675**, and target rank <=5 at **143/160** positions.

So the target often remains nearby, but four trajectories diverge at their first token. This
is why the Phase B result must improve both score and rank; lowering BPB alone cannot be promoted
as a functioning carve.

## What changed relative to earlier evidence

E64 found int8 carve dearer than ternary over all 28 layers. On this matched eight-layer scope,
applied R8 is instead **0.077559667 BPB better** than H1's applied ternary `1.096636133`. The
registered E64-direction prediction is falsified. The two statements coexist because accumulated
post-hoc damage is not linear in layer count; neither scope licenses extrapolation to the other.

The old H2T route remains retired. Training ternary-format damage would attack an axis this test
and E66 show can be avoided. The next GPU object is R8 weights with the carve/router present
during training, starting from H0 run 3.

## Do not repeat

- Do not rerun these Phase A rows or the rank partner; their result hash is frozen.
- Do not use H1's ternary applied number as the H2I training threshold. The matched threshold is
  `1.019076465262247`.
- Do not infer rate: this PyTorch path computes the dense FFN before masking.
- Do not promote the 8-layer ordering against E64 into an all-layer law.
- Do not infer 10 B quality, or activation-int8/LUT behavior, from weight-only R8.
