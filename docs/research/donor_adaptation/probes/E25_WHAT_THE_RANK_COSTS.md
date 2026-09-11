# E25 — what the rank actually costs, in the engine

**Brief**: `briefs/BRIEF_E25_WHAT_THE_RANK_COSTS_IN_THE_ENGINE.md`, pushed at `e506c7b` before
the runner was run.
**Code**: `engine/donor_engine.c` (the factored kind), `engine/qwen_export.py` (`--quant tagged`,
`--factors`), `engine/synth_export.py` (`--rank`), `engine/e1_bpb_through_engine.py`
(`nbytes_tagged`), `engine/e25_parity_factored.py`, `engine/e25_rank_cost.py`.
**Results**: `engine/results/e25_parity_factored.json`, `engine/results/e25_rank_cost.json`.

## VERDICT — `RANK-PAYS-WHAT-IT-WEIGHS`

**The engine executes a factored matvec, it reproduces PyTorch on the real factors, and at the
goal's shape the rank lever delivers exactly the weights it removes and nothing is lost to the
extra call.** At `T10` (10.60 G active, the goal's "es 10B" dimensions) rank-512 `q/o` reads
**5.36 tok/s against 4.70**, `+14.12%`, against a byte prediction of `+12.86%`. The planted
byte-neutral control reads `−0.43%`, i.e. **the factored path's own overhead is below this box's
resolution**.

**And the number that matters more than the ratio: the engine is a constant-throughput machine.**
Across four `T10` arms whose rates differ by 15%, charged throughput sits in a **1.54% band**:

| `T10` arm | active weights/token | tok/s | **G active weights/s** |
|---|---|---|---|
| `T10-TAG-R0` | 10.6032 G | 4.70 | **49.80** |
| `T10-R2048` (byte-neutral) | 10.6032 G | 4.68 | **49.59** |
| `T10-R512` | 9.3952 G | 5.36 | **50.36** |
| `T10-R256` | 9.1939 G | 5.43 | **49.95** |

**So the rank axis buys speed through exactly one mechanism — fewer weights — and the engine
converts weights into time at a rate that does not care how they are arranged.** Every budget
table in this programme that charges a rank-`r` projection at `2·D·r` is, at this shape,
charging the right thing.

**The distance to the goal, measured rather than derived, for the first time at the goal's
shape: 4.70 tok/s. The target is 50. That is 10.6× away, and rank-512 `q/o` on its own closes it
to 9.3×.**

---

## 0. What was owed, and why nothing could be said about speed until now

E21 §8, E22 §8, E23 §9 and `decisions/T4_HEALING_PROPOSAL.md` §6 all end with the same item:
**`engine.c` had no factored matvec.** E21 and E22 measured the rank cut by computing `A·B` and
installing a **dense** matrix, which is the correct way to price its **quality** and says nothing
at all about its **cost**. Every consequence of the rank axis for tok/s — E18 §31's budget line,
E23 §7's `k = 139…156` re-derivation, the whole reason `q/o` at rank 512 is in the plan — rested
on `2·D·r < D²`, an arithmetic statement that nothing had executed.

## 1. What was built

### 1.1 The factored kind

`mat_t` gains `rank`, `fa`, `fb`, `fs`. When `rank != 0`, `matvec` runs

```
y = A · ( s ⊙ ( B · x ) ) + bias
```

as **two calls to the same kernels** with an intermediate of size `r` in its own buffer — it
needs its own, because the LUT path reuses `g_i32b`/`g_f32b` *inside* a single call. `rank == 0`
is every matrix written before this kind existed and that path is untouched.

`s` multiplies the **intermediate**, never a factor. Folding it into `B` would change `B`'s
per-row ternarization and folding it into `A` would change `A`'s; the point of carrying it is
that H0's trained scale is fp32 while the factors are not.

### 1.2 `quant == 3`, the tagged container

Every matrix carries its own `int32` kind — `0` packed, `1` factored, `2` fp32. This is not a
convenience. **The configurations this programme measures are mixed**: E21, E22 and E23 all left
`k_proj`/`v_proj` fp32 and cut only `q_proj`/`o_proj`, and H0 trains exactly that object. A
single global `quant` flag cannot express it, so until `quant == 3` existed **every rank result
was a PyTorch number with no runnable artifact behind it.** `quant` 0/1/2 stay untagged and
byte-identical.

`--fuse` refuses a factored matrix — its two factors share no output row space — and now keys
its kind off the source matrices rather than the global flag. `--lut` already requires
`quant == 2` and therefore declines. Both refuse at load.

## 2. Gates on the construction — all four fire

| gate | result |
|---|---|
| **`G-E25a`** — legacy paths cannot have moved | patched vs unpatched engine on `qwen25-15b_tq.bin`, 10 × 151,936 logits: sha256 `e09b30c847f3956142fb3bc670214cb13aba327f` **on both sides**. Bit-identical. |
| **`G-E25b`** — the writer agrees with an independent size model | `GATE V3` on all 9 synthetic arms and `GATE E25-L` on the real export: `5,690,990,052` bytes, matching `e1_bpb_through_engine.layout_bytes_tagged`, which is derived from the format and knows nothing about the writer. |
| **`G-E25c`** — the reader agrees too | the engine's own `layout OK: consumed exactly 5690990052 bytes`. |
| **`G-E25P`** — **end-to-end parity** | worst relative l2 **`6.445e-04`** (bar `2e-3`), top-1 agreement **`1.0000`** on 10/10 positions. **FIRES.** |

`G-E25P` is the one that matters, and it is Phase 60's law applied: *kernel-bit-exact does not
compose to system-correctness*, so a new kernel is gated end to end or not at all. The artifact
is `Qwen2.5-1.5B` at the pinned revision, fp32 everywhere except `q_proj`/`o_proj` on all 28
layers, which carry the factors from `h0_factors.npz` — **E22's `QO512-TB`, the arm that reads
BPB 2.812226 / tf 28 / free 1, and the state H0 starts from.** The reference installs
`h0_qat.TernaryLowRank`, i.e. the same module the T4 trains and the same module `h0_eval.py`
measured that start state with. The exporter calls `h0_qat`'s own `r3_actsearch` — `A` against
`rms_A·|s|`, `B` against `rms_in` — so the two sides cannot drift without this gate saying so.

**One bug was caught on the way, by the readers rather than by reasoning.** The first tagged
writer emitted `A` and `B` without their own kind tags; the file parsed as far as the first
factor and then died on a byte that was a weight *code* being read as a *kind*. `GATE E25-L` now
catches that class in the writer, before a 3-minute write.

## 3. Part B — the arms, and the two controls

Timing uses `synth_export.py`, whose weights are **noise**. That is the honest way to price a
shape and not a shortcut: E3 §4's Gate V1 planted `--codes zero` against `--codes dense` on one
shape and found no timing difference. **No number in this section says anything about quality.**

Two shapes: **`S15`** = Qwen2.5-1.5B's dimensions, where the real factors exist, and **`T10`** =
`(4096, 14336, 48, 32/8, 128, V=32768)`, the goal's "es 10B". `--head ternary`, `--fuse` off in
every arm including the dense controls — a factored `q_proj` cannot be fused, so leaving it on
for the dense arms would compare 97 OpenMP regions against 225.

Idle box, `--threads 6`, **3 repetitions per arm, interleaved by repetition** so a thermal drift
hits every arm equally, dispersion reported with every rate.

| arm | active/token | mean tok/s | spread | vs its `R0` | byte prediction |
|---|---|---|---|---|---|
| `S15-PACKED` (untagged) | 1.5436 G | 29.70 | 2.1% | +0.67% | 0 |
| `S15-TAG-R0` | 1.5436 G | 29.50 | 1.3% | — | 0 |
| **`S15-R768`** (byte-neutral) | 1.5436 G | 28.78 | **6.4%** | **−2.46%** | 0 |
| `S15-R512` | 1.4995 G | 30.40 | 3.2% | +3.03% | +2.94% |
| `S15-R256` | 1.4555 G | 30.51 | 3.3% | +3.42% | +6.05% |
| `T10-TAG-R0` | 10.6032 G | 4.70 | 6.6% | — | 0 |
| **`T10-R2048`** (byte-neutral) | 10.6032 G | 4.68 | 7.3% | **−0.43%** | 0 |
| `T10-R512` | 9.3952 G | 5.36 | 3.7% | **+14.12%** | +12.86% |
| `T10-R256` | 9.1939 G | 5.43 | 6.8% | **+15.68%** | +15.33% |

### 3.1 The container is free

`S15-PACKED` vs `S15-TAG-R0` is `+0.67%` with spreads of 2.1% and 1.3%. The `int32` kind is read
once per matrix at load and costs nothing at run. **Every tagged number below therefore carries
no container offset**, which had to be established before any of them could be read as a rank
effect.

### 3.2 The planted control — and it says the overhead is not there

For a square `[D,D]` projection the factored form moves **exactly** as many weights as the dense
one at `r = D/2`: `2·D·(D/2) = D²`. `S15-R768` and `T10-R2048` are byte-neutral **to the last
weight** — `1,543,569,408` and `10,603,200,512`, identical to their `R0` controls — so whatever
they lose IS the factored path's own cost: the second matvec call, the extra OpenMP region, the
intermediate vector.

They read **`−2.46%`** and **`−0.43%`**, both slower, both with spreads of 6.4% and 7.3%.
**Neither is resolvable from zero.** The registered prediction was "0–2% slower"; `T10` lands
inside it and `S15` lands 0.46 points outside — but with that dispersion the only defensible
statement is the stronger one: **at both shapes the extra 56 / 96 OpenMP regions per token cost
less than this box can measure.**

**Disclosure.** `S15-R768`'s `−2.46%` is carried by a single repetition: 27.55, then 29.39 and
29.39. Reps 2–3 alone give `−0.74%` against the same two reps of `S15-TAG-R0`. The outlier is
**kept** — dropping the rep that makes a control look worse is how a control stops being one —
and the 6.4% spread is precisely the reason the conclusion is "below resolution" rather than a
number.

### 3.3 The rank lever at the goal's shape

`T10-R512` reads **`+14.12%`** against a byte prediction of `+12.86%`; `T10-R256` reads
`+15.68%` against `+15.33%`. **The registered alternative — "fails to beat `T10-TAG-R0` by at
least 8%", which would have corrected every budget table that charges `2·D·r` — DOES NOT FIRE.**
The measured gains sit 1.3 and 0.4 points above their arithmetic, differences well inside the
6–7% dispersion, so the correct reading is *the lever delivers its arithmetic*, not *the lever
beats its arithmetic*.

### 3.4 The invariant, which is worth more than any of the ratios

| shape | `G active weights/s`, across all arms | band |
|---|---|---|
| `S15` | 44.41 … 45.84 (mean 45.16) | **3.17%** |
| `T10` | 49.59 … 50.36 (mean 49.92) | **1.54%** |

At `T10`, four arms whose rates differ by 15% deliver charged throughput inside 1.54%. **The
engine converts active weights into time at a rate that does not care how those weights are
arranged** — dense or factored, one call or two. That is why the rank axis is a pure weight
saving here, and it is a far more useful statement than any single ratio because it predicts
arms that were never run.

Charged bytes, at the packed format's exactly 0.500000 B/weight: `24.90 GB/s` at `T10`,
`22.77 GB/s` at `S15`. **These are charged, not moved** — the byte convention law applies and
the two must never meet in a fraction. `T10` streams 9% better than `S15` per weight; the
candidate reasons (fewer per-layer fixed costs against more work, and `T10`'s 32,768-entry
vocabulary against Qwen's 151,936) are **not** separated by this run.

### 3.5 E18 §31's budget line, corroborated at the goal's shape

E18 §31 derived that 50 tok/s needs **`0.982–1.060 G` active ternary weights per token**. At
`T10`'s measured `49.80 G weights/s` that band maps to **47.0 – 50.7 tok/s**. The budget was
derived from smaller shapes; **this is the first time it has been checked at the goal's
dimensions, and it brackets 50 where it was built to.**

## 4. Predictions — scored

| # | registered | measured | verdict |
|---|---|---|---|
| 1 | container free, within dispersion | +0.67%, spreads 1.3–2.1% | **HELD** |
| 2 | byte-neutral arms 0–2% slower | `T10` −0.43%, `S15` −2.46% | **HELD at `T10`, MISSED at `S15` by 0.46 pt** — and neither is resolvable, which is the stronger reading |
| 3 | `T10-R512` faster by 10–13% | **+14.12%** | **MISSED — above the band**, in the favourable direction, and inside dispersion of its own byte prediction |
| 4 | `S15-R512` not resolvable | +3.03%, spreads 1.3–3.2%, prediction +2.94% | **HELD** — the point estimate sits on the arithmetic but does not separate from zero at 3 reps |
| 5 | `R256 > R512 > R0` at `T10`, unresolved at `S15` | `T10` 5.43 > 5.36 > 4.70; `S15` 30.51 > 30.40 > 29.50 | **PARTIAL** — the gap to `R0` holds at `T10`; **`R256` vs `R512` is inside noise at both shapes** |

**Two held, one partial, two missed.** Prediction 3's miss is the one that matters and it is the
good kind: I registered a band below the byte arithmetic because I expected the second call to
cost something, and it does not.

**One cell where the byte ledger looks optimistic and is not being called a finding.**
`S15-R256` reads `+3.42%` against a `+6.05%` prediction — a 2.6-point shortfall at a 3.3%
spread. `T10-R256` shows nothing of the kind (`+15.68%` vs `+15.33%`). The candidate is that at
`S15`, `r = 256` makes each factor a 196 KB matrix spread over 6 threads and the fixed per-call
cost stops being negligible. **It is one cell, at one shape, inside the noise, and it is
recorded as a question, not an answer.**

## 5. What E25 does NOT claim

- **Nothing about quality.** Part B's weights are noise. The rank that is *good* remains E21's
  question, and `r/D = 1/3` is validated at `D = 1536` only (E21 §8). `T10-R512` is `r/D = 1/8`
  and **no one has measured whether a model survives that**.
- **`T10` is a shape, not a model.** It has the goal's dimensions and random weights. A 10 B
  worth running does not exist here, and `4.70 tok/s` is the rate of a shape.
- **`T10`'s vocabulary is Mistral's 32,768**, giving it the second-smallest head on the disk
  (134 M). A Qwen-vocabulary 10 B carries a 622 M head, and `INDEX.md`'s head table already
  records that vocabulary is a speed variable that is chosen rather than earned. **The 4.70 tok/s
  is the friendly case.**
- **`6.79 tok/s` is untouched.** That is the real 7.072 B packed donor — a different artifact, a
  different shape, and a real one.
- **`k/v` and the head are dense in every arm**, as in E21/E22/E23.
- **One box**, the reference floor: Ryzen 5 3600X, Zen 2, 6c/12t, L3 32 MB, `--threads 6`.
- **Three repetitions.** Enough for a 14% effect at a 7% spread; **not** enough for the 3% ones,
  which is why §3.1 and §4 row 4 say "not resolvable" instead of quoting them.

## 6. What this changes, and what it leaves standing

**Discharged**: the item owed by E21 §8, E22 §8, E23 §9 and the T4 proposal §6. The engine has a
factored matvec, it is correct end to end, and the rank axis now has a measured price.

**Unchanged and now sharper**: the goal is 50 tok/s at ~10 B and the measured starting point at
that shape is **4.70 tok/s**. The rank lever on `q/o` is worth `+14%`. E18 §31's budget says the
whole model must come down to `0.982–1.060 G` active weights per token — from 10.60 G. **Rank on
`q/o` alone is 1.2 G of that 9.6 G gap.** The FFN is 8.45 G of `T10`, and that is where the
remaining nine tenths of the problem live; E19's carve and E23/E24's router are aimed at exactly
it, and E24 is still unrun.

**H0 is untouched either way.** It asks whether gradients move a ternary low-rank object, and
this probe measured the time that object takes. The gate is still `tf ≥ 48` on CPU in fp32.
