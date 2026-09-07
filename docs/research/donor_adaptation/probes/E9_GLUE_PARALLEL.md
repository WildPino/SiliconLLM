# E9 — the SwiGLU glue ran on one thread. It doesn't now.

**Verdict: `GLUE-CONFIRMED`.** Pre-registered and pushed as
`briefs/BRIEF_E9_GLUE_PARALLEL.md` (`82dbb54`) before any measurement was taken.

Donor: `Qwen2.5-Coder-7B` packed, 7.072 B active weights/token.

---

## 1. The result

| Coder-7B, `--bench 300`, 10 interleaved pairs, idle, witness on | before | after |
|---|---|---|
| **rate** | **6.45 tok/s** (spread 4.0%) | **6.79 tok/s** (spread 4.0%) |
| `ffn~` witness | 123.812 ms | 116.205 ms |
| **paired median ratio** | — | **1.0557** |

Per-pair: 1.0785 1.0442 1.0683 1.0385 1.0669 1.0603 1.0512 1.0651 1.0415 1.0510.
Band fixed before the run: ≥1.035 **CONFIRMED**. **Predicted 1.055–1.066 — the measurement landed
inside its own band.**

| `--profile --bench 100` | before | after |
|---|---|---|
| **glue(silu)** | **11.346 ms** | **2.938 ms** (**3.86×**) |
| ffn | 123.752 | 112.980 |
| gate+up | 75.078 | 73.597 |
| down | 37.286 | 36.403 |
| `sum/ffn` | 0.9999 | 0.9999 |
| rate | 6.59 | **7.12 tok/s** |

## 2. What it was

```c
for(int i=0;i<F;i++) s->hb[i]=silu(g[i])*u[i];
```

**No `#pragma omp parallel for`.** Every `matvec` on both sides of this loop was parallelised
across `--threads 6`; the glue between them ran on **one core** — 530,432 `expf` calls per token
(F 18,944 × L 28), 11.346 ms, **7.5% of the whole token**, for **zero weight traffic**.

It had never been charged because nothing had ever decomposed the FFN. E8's sub-timers, built to
answer a bandwidth question, found it as a side effect.

## 3. This one is bit-exact, and that is the point

E8 re-partitioned a float sum and could never be bit-exact; it had to be carried by an end-to-end
parity gate. **An elementwise map with no reduction cannot change a bit when the index range is
split** — each `i` is written once and reads only its own inputs.

| **G-G1** | `--logits` sha256, before vs after | **PASS — byte-identical, `b94b56d002880d84…548765`** |
|---|---|---|
| **G-G2** | 10 interleaved pairs at target scale | **CONFIRMED, 1.0557** |
| **G-G3** | does `glue` itself fall, `sum/ffn` in 0.98–1.02? | **PASS — 11.346 → 2.938, `sum/ffn` 0.9999** |

**+5.6% at zero numeric cost.** The gate is sha256, not "small" — the standard §12's rope hoist
met and E8 could not.

## 4. A prediction that landed, and why

Predicted **1.055–1.066**; measured **1.0557**. Predicted glue **2.0–3.5 ms**; measured **2.938**.

Both bands were derived the way E8 §3 said the next one had to be: **from the measured 11.346
divided by a structural factor (the thread count), plus §12.4's measured 2.7 µs per OpenMP region
× 28 layers**, *not* from first principles. The two preceding predictions in this programme —
the witness cost (150× off) and E8's magnitude (1.5–1.7× predicted, 1.349 measured) — were both
built from first principles and both missed. **This is the corollary paying for itself.**

## 5. What was deliberately left alone

**`rmsnorm` stays serial**, and the brief said so before the run rather than after: `norm+glue`
totals 0.330 ms/token over 56 calls, and at 2.7 µs per OpenMP region parallelising it would spend
0.151 ms to save at most 0.275 ms. **A remedy has to clear its own overhead.** Measured after:
`norm+glue` 0.327 → 0.329 ms, untouched, as intended.

**The `expf` itself is untouched.** At 2.938 ms it is still 2.6% of the FFN, and a vectorised
polynomial exp would be worth more than threading was — but it forfeits the sha256 gate and puts a
BPB question on the table. That is a different experiment with a different gate set. **The free,
bit-exact win is banked first.**

## 6. Where the day leaves the goal

| Coder-7B, 7.072 B active/token | tok/s | short of 50 |
|---|---|---|
| E7 as published this morning | 4.460 | 11.2× |
| after E8 (dependency chain) | 6.37 | 7.8× |
| **after E9 (glue)** | **6.79** | **7.4×** |

**1.52× in one day, all of it parity- or bit-gated.** Delivered weight rate **48.0 G-weights/s,
24.0 GB/s moved**, against measured streaming ceilings of 28 / 37.0 / 42.

At 0.5 B the same two changes take the engine from §12.2's 56 tok/s to **79.12**.

**Ledger §19.3 is untouched by both.** Engine work had 1.7–2.5× in it; E8 and E9 have now taken
**1.52×** of that, leaving roughly 1.15–1.65× to the measured ceilings. **The missing 7.4× is still
a property of the model, not of the code.**

## 7. Owed

1. **The vectorised `expf`** — §5. Bigger than E9 was, but not free.
2. **The next kernel binder at 22.5 GB/s** — E8 §9 item 2, unchanged. Still the largest engine
   item on the list.
3. **The non-packed ternary path still has a single accumulator** — E8 §9 item 3.
4. **A witness plateau per shape, published as an absolute** — E8 §9 item 4. The 7 B plateau is now
   observed twice: `ffn~` 122.9–126.8 at `--mvacc 4` pre-E9, **114.5–117.0 after**. Recorded here
   so the next session can detect a contended sweep from outside it.
