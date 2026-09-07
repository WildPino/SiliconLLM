# E9 — the SwiGLU glue runs on one thread

**Pre-registered. Pushed before any measurement was taken.** Opened by E8 §9 owed item 1.

---

## 1. What E8 found by accident

E8 decomposed the FFN organ to answer a bandwidth question. The decomposition also charged, for
the first time in this programme, a component that moves **no weights at all**:

| Coder-7B, `--profile --bench 100`, `--mvacc 4` | ms/token | share |
|---|---|---|
| gate+up | 76.661 | 49.4% of the token |
| down | 37.973 | 24.5% |
| **glue(silu)** | **11.628** | **7.5% of the whole token** |
| residual | 0.035 | 0.0% |

`glue(silu)` is `s->hb[i] = silu(g[i]) * u[i]` over `F` elements per layer — **530,432 `expf` calls
per token** at this donor (F 18,944 × L 28). §12's rope hoist was the same shape of finding and
was worth **+10.3%**.

## 2. The defect

```c
{ TICF;
  if(g_fuse){ const float* g=s->gubuf; const float* u=s->gubuf+F;
              for(int i=0;i<F;i++) s->hb[i]=silu(g[i])*u[i]; }
  else      { for(int i=0;i<F;i++) s->hb[i]=silu(s->hb[i])*s->hb2[i]; }
  TOCF(F_GLUE); }
```

**There is no `#pragma omp parallel for` on either branch.** Every `matvec` on both sides of this
loop is parallelised across `--threads 6`; the glue between them runs on **one**.

11.628 ms across 530,432 elements is **21.9 ns**, about **83 cycles**, per element — consistent
with a scalar CRT `expf` and with nothing else being wrong.

## 3. Why this one is free, and E8 was not

E8 re-partitioned a float sum and could therefore never be bit-exact; it had to be carried by an
end-to-end parity gate. **This is an elementwise map with no reduction**: each `i` is written once
and reads only its own inputs. Splitting the index range across threads **cannot** change a single
bit.

So E9's gate is the strong one this programme prefers and E8 could not have: **sha256.**

## 4. Gates — fixed here, before anything was run

| gate | test | pass |
|---|---|---|
| **G-G1** | `--logits` sha256, 0.5 B packed, before vs after | **byte-identical.** Not "small", identical. FAIL ⇒ reverted, because a parallel elementwise map that changes bits means the loop was not what it looks like |
| **G-G2** | Coder-7B packed, `--bench 300`, 10 interleaved pairs, witness on | see bands |
| **G-G3** | `--profile`: does `glue(silu)` itself fall, and does `sum/ffn` stay in 0.98–1.02? | descriptive |

**Verdict bands, fixed now:**

| paired median ratio | verdict |
|---|---|
| **≥ 1.035** | **CONFIRMED** |
| 1.015 – 1.035 | **UNDECIDED** |
| **≤ 1.015** | **REFUTED** |

G-Z4 measured a paired-ratio spread of **1.8%** across ten pairs on this exact bench, so 1.035 is
roughly twice the instrument's own dispersion — **E4's law, that a threshold inside the
instrument's dispersion cannot decide.**

**Predicted band, written before the run: 1.055–1.066.** From 11.628 ms on one thread → about
11.628/6 = 1.94 ms plus one OpenMP region per layer at §12.4's measured **2.7 µs** (28 × 2.7 µs =
0.076 ms), so ~2.0–3.5 ms allowing for imperfect scaling on a loop that touches 6.4 MB of
activations per token. That saves 8.1–9.6 ms of a 155.0 ms token.

**Per E8 §3, this band is derived from the measured 11.628 divided by a structural factor (thread
count), not from first principles.** That is the corollary the last two missed predictions bought.

## 5. What is deliberately NOT done

**`rmsnorm` is also serial, and it stays serial.** `norm+glue` totals 0.330 ms/token across 56
calls; at §12.4's 2.7 µs per OpenMP region, parallelising it would cost 0.151 ms to save at most
0.275 ms. **The remedy has to clear its own overhead**, and at that size it barely does. Recorded
so nobody reads the omission as an oversight.

**The `expf` itself is not touched.** A vectorised polynomial exp would be worth more than
threading, and it would forfeit the sha256 gate and put a quality question (BPB) on the table.
That is a different experiment with a different gate set, and it should be judged after the free
win is banked, not mixed into it.

## 6. Honest ceiling

Even a perfect result here is **+6%**, and Coder-7B is **7.8× short of 50 tok/s**. This is banked
because it is free and bit-exact, not because it changes the strategic picture — ledger §19.3 and
§21.3 are untouched by it.
