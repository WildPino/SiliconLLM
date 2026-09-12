# BRIEF E35 — the 50 tok/s envelope: measure the shape that fits, do not derive it

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

---

## 0. Why this probe: E34 answered "no" and this one has to answer "then what"

E34 measured that **T10's attention + head floor alone reads 20.03 tok/s** — 2.50× short of the
goal with the FFN deleted — and that a *perfect* kernel on that floor still reads only 33.65
tok/s. That closes the question "can the goal's literal shape be reached by shrinking the FFN".
It does not answer the question the goal actually asks, which is **"far girare un modello grande a
50 token/s"**: *what shape does run at 50, and how big can it be?*

E34 §4.1 produced a desk answer — 31 of 48 attention layers at the ceiling — and this programme's
rule is that a desk table does not decide anything. **E34's own §7 item 1 named this probe as the
main line.** Here the envelope gets measured.

## 1. The question

> **What is the largest shape that actually reads 50 tok/s on this box, and does the linear
> active-weight model predict where the crossing happens before it is measured?**

## 2. The model this probe exists to test, stated before the measurement

From E34's measured numerator, **46.74 G-w/s** (packed, operative after E32), and the exporter's
own accounting at `D=4096, NH=32, NKV=8, HD=128, V=32768`:

```
charged(L) = 41,943,040 · L  +  134,217,728          (attention·L + head, FFN at carve k=1)
tok/s(L)   = 46.74e9 / charged(L)
```

| `L` | charged/token | predicted tok/s | note |
|---|---|---|---|
| 48 | 2.1475 G | **21.8** | T10's own depth; E34 measured **20.03** at `k=1` (which charges 3.9% more) |
| 32 | 1.4764 G | **31.7** | |
| 24 | 1.1408 G | **41.0** | |
| **19** | **0.9311 G** | **50.2** | **the predicted crossing** |
| 16 | 0.8053 G | **58.0** | |
| 12 | 0.6376 G | **73.3** | leaves ~0.30 G of the 50 tok/s budget for an activated FFN |

**The crossing is predicted at `L = 19` with the FFN at zero.** Every number in this table is a
prediction of a measurement that has not been taken, from a model fitted to nothing in it.

## 3. Arms

New synthetic shapes, **added to `synth_export.py`'s table without touching a single existing
entry** (every past probe's shape must keep its bytes): `T10L32`, `T10L24`, `T10L16`, `T10L12` —
identical to T10 in every field except `L`. Each exported with `--carve 256 --carve-k 1`, so the
FFN is at the floor and **depth is the only thing that moves**.

| arm | `L` | why |
|---|---|---|
| `T10-L48` | 48 | E34's own artifact, re-benched this session — the planted control |
| `T10-L32` | 32 | above the crossing |
| `T10-L24` | 24 | just above |
| `T10-L16` | 16 | just below |
| `T10-L12` | 12 | well below, and the one with FFN budget left over |

Five interleaved reps, reps outermost, idle box, operator idle (E33 §6). One file per arm; the
`k=1` carve is in the file, so no arm depends on a flag another arm does not have — the defect
that voided E34 run 1.

## 4. The gates

**`G-E35A` — the planted control.** `T10-L48` must reproduce E34's `T10-FLOOR = 20.03 tok/s`
within ±10%, same box, same file, same flags. Below that bar the session is not comparable to the
probe it extends and E35 is VOID.

**`G-E35B` — the exporter must agree with the model, exactly.** For every arm, the exporter's
`active_weights_per_token` must equal the runner's independently computed `charged(L) + carve
residue`, at **zero tolerance**. If they differ, the runner wins and §2's table is reported as
wrong, with the delta.

**`G-E35C` — every new shape must pass `GATE V3`**, the exporter's own independent layout
recomputation. A shape I invented for this probe is exactly where a layout bug would hide.

## 5. The verdict cell, named before the run

**The measured `L` at which the interpolated curve crosses 50 tok/s.** Reported as a real number
by linear interpolation in `charged` between the two bracketing arms, because tok/s is linear in
`1/charged` and `charged` is linear in `L`.

| band | name | what it would mean |
|---|---|---|
| **`L* ≥ 24`** | `DEPTH-IS-AFFORDABLE` | half of T10's depth fits at 50 tok/s; the architecture keeps most of its layers |
| **`16 ≤ L* < 24`** | `DEPTH-IS-HALVED` | the model predicted this; roughly a third of T10's depth, FFN at zero |
| **`L* < 16`** | `DEPTH-COLLAPSES` | attention is even more expensive than E34's floor implied and the shape must change on another axis too (width, rank, KV) |

## 6. Predictions — fixed here, before the run

1. **`G-E35A`, `G-E35B`, `G-E35C` all fire.**
2. **`L*` lands in `[17, 21]`** — the model says 19. Wider than a point because E34's own floor
   arm came in 8% under its model prediction (20.03 measured against 21.8), and if that bias is
   real `L*` moves down.
3. **The bias is real and `L*` < 19**, i.e. inside `[17, 19)`. E34's `T10-FLOOR` charged 3.882%
   more than the pure floor and still read 8% slow, so something beyond charged weights costs
   time at depth — most likely the per-layer fixed cost the carve machinery adds (E26's
   `−6.36%`).
4. **The curve is linear in `1/charged` to within 5%** across the five arms. If it is not, the
   numerator is not a constant across depth and **that** is the finding, not `L*`.
5. **Registered as the honest negative**: `L* ≈ 19` at `D=4096` with **zero FFN** is **not a 10 B
   model** — it is ~1.6 B of attention. **This probe cannot produce a 10 B at 50 tok/s and will
   not claim to.** What it produces is the *active-weight envelope*: at 50 tok/s this box affords
   **≤0.9348 G active charged weights per token**, and any 10 B must fit its *active* half inside
   that, which is an MoE/sparsity statement and belongs to `SCALEUP_ARCHITECTURE.md`.
6. **The FFN budget left at each depth is reported** — `0.9348 G − charged(L)` — because that, not
   `L*`, is the number the architecture needs: how much activated FFN a given depth can afford at
   50 tok/s.

## 7. What E35 will NOT be able to claim

- **Nothing about quality.** Synthetic noise weights, no BPB, and a 16-layer donor is not a
  trained model. Whether a shallower model is *good* is E24/E27/E29's axis and is not touched
  here.
- **Nothing about width, rank or KV heads.** Depth only, one axis at a time, deliberately. The
  other three axes are separate probes and their costs are **not** additive without measurement.
- **Nothing about the KV cache.** 40-token benches; weight traffic only. Long-context KV is still
  owed (E30 §8 item 2) and gets worse with depth, so `L*` here is an **upper** bound on depth.
- **Nothing about `--lutblk`.** Revoked by E32; every arm runs packed, the operative path.
