# BRIEF E49 — THE ATTENTION KERNEL WAS NEVER SWITCHED ON

**Registered before any E49 cell exists.** No speed claim in this brief may be quoted until
`G-E49a` (parity) passes; the Phase 60 law is that a kernel change is not a result until the
system is shown to compute the same thing.

## 1. The finding

`benchmarks/donor_adaptation/engine/donor_engine.c:202`:

    static int g_attn=ATTN_SERIAL;

The default attention kernel is the **scalar serial dot product**. `--sweep6` sets every one of
its ten arms to `ATTN_AVX4` (line ~1336). And **no runner from E26 onward passes `--attn`**:

* `bench()` — `e28_kernel_transfer.py:132` — builds `[engine, --weights, w, --threads, n] +
  flags + [--bench, N]`, with no `--attn`.
* `one_rep()` — `e44_interval.py:171` — the same.
* `e40_levers_exhausted.py:213`, the origin of the 112.7 tok/s headline, calls
  `bench(..., ["--carve-k", str(k)])`.

Only the E3/E4/E5 family ever set the kernel. **Therefore E36, E37, E39, E40, E43, E44, E45, E46,
E47 and E48's own pure-`none` control all ran scalar attention.**

Smoked on the target arm (`e40_r128.bin`, `--carve-k 3`, 6 threads, window 1280 = mean position
640, two reps each):

| kernel | wall | tok/s |
|---|---|---|
| `serial` — the default | 29.189 / 29.394 s | 43.85 / 43.55 |
| `avx4` | **18.801 / 18.345 s** | **68.08 / 69.77** |

**+56% at realistic context.** Pure `avx4` is also *faster* than `--sweep6` (19.64–20.50 s), which
runs `avx4` **plus** six doubled arms — so the whole effect is the kernel and there is no residual
neighbour-warming term. These are smoke numbers, two reps, and they are not the measurement.

## 2. How it was found, which is the part worth keeping

Not by reading the source. The chain was:

1. E48 addendum A.4 registered a **drift line** comparing E48's slope to E46's, and explicitly
   **demoted it to "not a gate"** so that E48 could not overrule E46.
2. That demoted line showed a **2.5× disagreement** on a quantity whose intercepts agreed to 2.6%.
3. That forced B.5's adjudication, which fired **THE SWEEP INSTRUMENT IS BIASED**.
4. The branch was right. **Its canned reason was wrong** — B.5 wrote "a regime the engine never
   runs in", and it is a regime the engine *can* run in and should.

**A closure gate could not have caught this, and mine didn't.** `G-E48b2` compared the whole-token
slope against the attention-organ slope and **fired at 6.4%** — both measured under the same
kernel, so both carried the same defect. **A closure test between two quantities that share a bias
is blind to that bias.** That is a new law and it is going in the record next to
`feedback_gate_vs_measured_dispersion`.

And it hid for twenty-two experiments for the **same reason E46's context cliff hid**: at
`NTOK = 40` attention is a sliver of the token, so the kernel is nearly invisible there. One blind
spot, two symptoms.

## 3. What reconciles, and what does not change

| | kernel | `b` (ms/pos) | `C50` |
|---|---|---|---|
| E46 phase B | `serial` | 0.02031 | 575 |
| E48 under `--sweep6` | `avx4` | 0.00818 | 1454 |

**Neither measurement was wrong.** They measured different kernels. `C50 = 575` is the correct
`serial` number and E48's B.5 control reproduced it four times over (515 / 540 / 544 / 575).

Consequently **E48 addendum B.1 is re-scoped, not withdrawn**: its shares are a valid
decomposition *of the `avx4` attention organ*, validated within that kernel by `G-E48b1`'s 3× test
(6/6 at the gated windows). And the softmax rising to 33.1% there is exactly what the kernel
change predicts — `avx4` vectorises the Q·K and A·V dot products and does **nothing** for the
scalar `expf` softmax. B.2 (S15 void) and B.3 (`G-E48d` void) stand unchanged.

## 4. The gates

### `G-E49a` — PARITY, and nothing else counts until it passes

`avx4` is **not** bit-identical to `serial`: E4 published `NATS_TOTAL` **166667.2449386003**
(`serial`) against **166667.1361128952** (`avx4`), a difference in summation order. So bit-identity
is the wrong test and **end-to-end parity is the right one** — this is Phase 60's law, that
kernel-level agreement does not compose to system correctness.

* Run `--bpb` on **S15** (`e37_carved_nf.bin`, real weights, real corpus) under both kernels.
* **`|ΔBPB| < 1e-4`**, i.e. fifty times inside `σ_seed = 0.005`.
* **PLANTED CONTROL, and the null does not count without it.** The same harness, unchanged, must
  be shown to **FIRE** on a difference that is really there. Control: the same comparison run
  against a deliberately different artifact (`e37_dense_nf.bin`), which must return a `ΔBPB` orders
  of magnitude larger. An instrument that cannot see a real difference has not shown that this one
  is absent.

If `G-E49a` fails, **every speed number in E49 is void** and `serial` stays the kernel of record.

### `G-E49b` — the speed, measured rather than smoked

Both kernels, **pure** (no `--sweep6`, no `--profile`), windows **{40, 160, 640, 1280, 2560}**,
**5 reps**, round robin with reps outermost. Refit `ms/tok = a + b·pos` per kernel and recompute
`C50` per kernel. Reported with dispersion; every absolute carries the standing ±5%, the ratio does
not.

### `G-E49c` — does the published headline move? (two-sided, and registered as such)

At **`NTOK = 40`** — the context at which *every* published number in this programme was taken —
the two kernels must differ by **less than ±5%**, the band the ledger already carries, **or the
112.7 tok/s headline is restated in E49 rather than defended.** This gate can go either way and is
written to be able to.

### `G-E49d` — the ladder, so that "`avx4` is the best available" is measured

`serial`, `serial2`, `serial3`, `ilp4`, `avx1`, `avx4` at window 1280, 3 reps. Cheap, and it stops
`avx4` becoming the answer merely because it is the one I happened to smoke.

## 5. My prediction, registered

| quantity | prediction |
|---|---|
| `G-E49a` parity | **passes**, `ΔBPB` below 1e-5 — it is a summation-order change in a 128-element dot product |
| `G-E49a` planted control | **fires**, `ΔBPB` above 0.1 against the dense artifact |
| `G-E49b` `b(avx4)` | 0.008–0.010 ms/pos |
| `G-E49b` `C50(avx4)` | **1100–1500** |
| `G-E49c` headline at `NTOK = 40` | moves **less than 5%** — the headline survives, scoped as it already is |
| `G-E49d` | `avx4` fastest; `avx1` and `ilp4` between it and `serial` |

My last four registered predictions scored half, one-of-three, zero-of-three and two-of-six. This
one is recorded on the same terms.

## 6. What E49 does not claim

It does not claim a new headline, it does not revise `C50` until `G-E49b` measures it, and it does
not touch quality. It also does **not** change the engine's default — changing `g_attn`'s
initialiser is a separate, gated change, and E49's job is to establish first whether it deserves to
be made.
