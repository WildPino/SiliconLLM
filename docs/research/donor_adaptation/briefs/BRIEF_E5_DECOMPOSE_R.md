# BRIEF E5 — decompose `R`: what is the other 81% of the attention organ?

Pre-registered 2026-09-06, **pushed before the arms exist**. Successor to
`probes/E4_ATTENTION_ACCUMULATORS.md` §7 item 1, which named this experiment, and to `INDEX.md`
§4 item 0, which E4 installed in first place as it closed the item that used to hold it.

---

## 1. What this asks

E4 split the attention organ in two with a planted control and then made one half 6.53× faster:

| | `T10` @800, `--attn avx4`, 6 threads | share of the organ |
|---|---|---|
| `X` — the `Q·K` dot loop | **2.242 ms** | 18.6% |
| `R` — *everything else in the per-head body* | **9.816 ms** | **81.4%** |
| organ | 12.058 ms | 100% |

`R` was obtained by subtraction (`serial2` = the dot loop twice, bit-identically; `X = serial2 −
serial`; `R = serial − X`), and the split was checked against a third point it was not fitted to:
`serial3` = 3× landed **−0.56%** from `3X + R`. So `R` is a measured quantity, not a leftover.
**Nothing has ever measured what is inside it.**

By construction `R` contains at least three things:

```c
float sum=0.0f;
for(int t=0;t<=pos;t++){ a[t]=expf(a[t]-mx); sum+=a[t]; }        //  S  -- the softmax pass
float rs=1.0f/sum;
for(int i=0;i<HD;i++) out[i]=0.0f;
for(int t=0;t<=pos;t++){                                          //  Y  -- the A.V loop
    const float* vt=s->vcache+((size_t)l*s->maxseq+t)*KVO+(size_t)kvh*HD;
    float w=a[t]*rs;
    for(int i=0;i<HD;i++) out[i]+=w*vt[i];
}
```

plus **`P`** — everything that is neither loop: the `#pragma omp parallel for` fork/join that runs
**once per layer, 48 times per token** at `T10`, the load imbalance of 32 heads over 6 threads, the
per-head address arithmetic, and the `out[]` init. `P` has never been named before this brief and
is not a rounding term: §3 shows that my own a-priori accounting of `S` and `Y` **does not reach
`R`**, and the gap is 3–7 ms.

**Why it matters, in the goal's own units.** At `T10` @800 after E4, `f = 12.735 ms`, the organ is
**94.7% of `f`**, and `R` alone is **77.1% of `f`**. The ceiling `1000/f` is **78.5 tok/s**; with
`R` removed entirely it would be `1000/2.919` = **342.6 tok/s**. Nothing else on the non-weight side
is within an order of magnitude of this term. E4 moved the ceiling 40.5 → 78.5 by fixing 18.6% of
the organ; `R` is the other 81.4%, and it is unexamined.

**What this is NOT.** It is not an optimisation. Every arm below makes the engine *slower* on
purpose. E5 produces a decomposition and a named next lever; it adopts nothing. As in E4, anyone
reading this for a tok/s headline is reading the wrong document — at `T10` @300 the whole organ is
4.540 ms of a 309 ms token.

---

## 2. Arms — a new flag, orthogonal to E4's

`--attnr {none,sm2,sm3,av2,av3,fork2}`, composed with `--attn avx4` on every arm, so the `Q·K` half
is held at its fastest measured form and `R` is the largest share of the organ it can be. The
dispatch is **outside the `t` loop**, per head, exactly as E4's was.

| arm | what it does | isolates |
|---|---|---|
| `none` (= plain `--attn avx4`) | unchanged | baseline: `X + S + Y + P` |
| `sm2` | the softmax pass run **twice** over the same unmodified `a[]`, second result kept | `X + 2S + Y + P` |
| `sm3` | the softmax pass run **three times**, third result kept | `X + 3S + Y + P` |
| `av2` | the `A·V` loop run **twice**, second result kept | `X + S + 2Y + P` |
| `av3` | the `A·V` loop run **three times**, third result kept | `X + S + 3Y + P` |
| `fork2` | one **extra empty `#pragma omp parallel for` over the heads**, per layer | one extra fork/join |

**Value preservation is by construction, not by tolerance.** E4's `serial3` returned
`d1+(d2-d1)+(d3-d1)` where the three accumulators are bitwise equal, so the two corrections are
exactly `+0` and the arm is *bit-identical*. The same shape is used here: the discarded passes are
folded back in as exact zeros, so **every arm must produce a bit-identical `NATS_TOTAL`** (gate G2).
`sm2`/`sm3` read `a[]` before the kept pass overwrites it, so all passes see the same input; `av2`/
`av3` accumulate the discarded passes into a `float tmp[128]` on the stack (`HD ≤ 128` in every
measured shape) — that is `2·HD·4 = 1 KB` of extra L1 traffic per head against the `pos·HD·4` the
loop already moves, i.e. **1/400 at `pos=400`**, and it is stated here so it is not discovered later.

`fork2`'s extra region writes one float per head into a per-head slot that is summed into a
`volatile` sink, so it cannot be elided, and it does nothing else.

---

## 3. Predictions, with their arithmetic, before any arm is built

`T10` @800: `L=48, NH=32, HD=128, NKV=8`, mean position 400.5, 6 threads, ~3.6 GHz.

**`S` — the softmax pass.** `L·NH·pos = 48·32·400.5 =` **615,168 scalar `expf` calls per token**,
102,528 per thread. The CRT's `expf` is not vectorised and is not inlined; 25–60 cycles is the
plausible range, giving 0.7–1.7 ms. The `sum +=` chain is a serial FP reduction of the same length,
4 cycles per element ⇒ 0.11 ms, and it is *not* hidden behind the `expf` because it depends on it.
**Predicted `S` = 0.8–2.0 ms.**

**`Y` — the `A·V` loop.** `L·NH·HD·pos = 78.7 M` FMAs per token, exactly as many as `Q·K`. Unlike
`Q·K` this is **not a reduction**: `out[i] += w*vt[i]` accumulates `HD` independent chains over
contiguous `i`, so clang may vectorise it without reassociation and without `-ffast-math` — it should
*already* be AVX2. It reads `V` at exactly the byte rate `Q·K` reads `K`: **78.7 MB unique, 315.0 MB
touched**, and E4 measured that same path at 2.242 ms (35.1 GB/s unique).
**Predicted `Y` = 2.2–3.5 ms.**

**`P` — the residual.** `S + Y` predicted = **3.0–5.5 ms** against a **measured `R` = 9.816 ms**.
So either one of the two estimates above is badly wrong, or **4–7 ms of the attention organ — a
third of `f` — is in neither loop.** It is pre-registered that I expect the second, that I cannot
today say what it is, and that `fork2` exists to price the largest single candidate.

**`fork2`.** 48 extra parallel regions per token; an OpenMP fork/join on 6 threads costs ~2–20 µs.
**Predicted 0.1–1.0 ms**, and this arm prices *one* extra region against the *one* already there, so
the baseline's own fork/join cost is read directly off it.

**A named candidate that E5 does not test.** `NH=32` heads over 6 threads with `schedule(static)`
gives two threads 6 heads and four threads 5 — the region runs at `6/5.33 = 1.125×` its mean, a
**12.5% tax on the whole organ**, `X` included. If `P` comes back large, this is the first thing to
look at, and it is cheap. It is written here so that it counts as predicted rather than as a
discovery made after the fact.

---

## 4. Gates, fixed before the arms exist

**G1 — the instrument must fire, and the model must be able to miss.**
`S` and `Y` are solved from the 1× and 2× points *by construction*, so those two points prove
nothing. The 3× arms are the test:

```
S = sm2 − none            predict sm3 = none + 2S       required |error| <= 3%
Y = av2 − none            predict av3 = none + 2Y       required |error| <= 3%
```

3% is E4's own tolerance, chosen there before its 3× point landed at −0.56%. **A component whose 2×
arm does not move the organ by more than 0.30 ms at `T10` @800 is reported `INCONCLUSIVE`, not
zero** — a doubling that costs nothing is equally consistent with a small component and with a
compiler that deleted the extra pass, and E5 will not claim to tell those apart. (E4's `serial2`
establishes that clang does not delete an identical duplicated loop *in this file, at this
optimisation level*; that is evidence, not proof, and it does not transfer for free to a loop with
a `libm` call in it.)

**G2 — parity must be BIT-IDENTICAL, not merely within σ_seed.**
Every arm is value-preserving by construction (§2), so `--bpb` on `qwen25-05b_tqh.bin` over the
pinned 24×512 slice must return a `NATS_TOTAL` **bitwise equal** to `--attn avx4 --attnr none`.
Any difference at all means the exact-zero construction is not exact and **that arm is void**.
This is a stronger gate than E4's G2 and it costs nothing.

**G3 — one sweep, and drift is named as drift.**
E4's closing law: **between-sweep dispersion on this machine is 5–10%, within-sweep 0.000–0.015
tok/s.** All six arms at all four points run in a **single sequential sweep** under one binary. The
`none` arm's organ is additionally compared to E4's 12.058 ms at `T10` @800; any gap is reported as
machine drift and **is not allowed to enter a ratio** — every number E5 concludes from is a
difference between arms measured minutes apart in the same sweep.

**G4 — no threshold inside the instrument's dispersion.**
If the §5 label depends on a share that lands within 3 percentage points of a boundary, the label is
withheld and an interleaved A/B/A/B/A/B run decides it, exactly as E4's did (there the baseline arm
itself drifted 10.55% across the pairs while the ratio held to 2.98%).

---

## 5. The decision rule, fixed now

Report `R = S + Y + P` with all three numbers, their shares of the organ and of `f`, at all four
points. The label is set at `T10` @800:

| condition | label |
|---|---|
| `S >= 0.50·R` | **SOFTMAX-DOMINATED** |
| `Y >= 0.50·R` | **AV-DOMINATED** |
| `P >= 0.50·R` | **OVERHEAD-DOMINATED** |
| none of the above | **SPLIT** |

and E5 must, for whichever term wins, state `1000/f` if that term went to zero — the ceiling that a
follow-up could buy — using the same `f` arithmetic E3 and E4 used. Terms are *not* rounded into
each other: if two are within 3 points of 0.50 the label is `SPLIT` and both are named.

**What each label costs us next**, so the conclusion is not free to be reinterpreted afterwards:
`SOFTMAX-DOMINATED` points at a vectorised `expf` and at fusing `S` into `Y` (one pass over `a[]`
instead of two — the decode-side of what flash-attention does); `AV-DOMINATED` points at the same
place `X` was, and at the KV layout, since `Y` reads `V` exactly as `X` read `K`; `OVERHEAD-DOMINATED`
points at the OpenMP region — 48 forks per token, the 32-over-6 imbalance — and would mean the
attention organ has been mis-attributed to arithmetic for two probes running.

---

## 6. What E5 does not do

- It does not adopt anything. No arm here is a candidate kernel; all six are slower on purpose.
- It does not touch the `Q·K` loop, the weight path, the KV cache format, or any exporter.
- It says nothing about quality: G2 requires bit-identity, so there is no BPB result to report
  beyond "unchanged, bitwise".
- It does not measure `f` past 800 tokens of context (E4 §7 item 3, still open) and does not extend
  to the six shapes E4's last sweep covers — `T10` and `S05`, @300 and @800, as in E4's run 2.
- It does not re-open E4's numbers. `X = 2.242` and `R = 9.816` are inputs here, and if the `none`
  arm disagrees with them by more than G3's band that is a finding about the machine, reported in
  E5 and not silently absorbed.

---

## 7. Reproduction

```
cd benchmarks/donor_adaptation/engine
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp donor_engine.c -o donor_engine.exe -lm
bash e5_run.sh                      # sequential; 6 arms x {T10,S05} x {300,800} + parity
python e5_analyse.py ../results/e5/arms.json
```

Never `-ffast-math` (Phase 35): it would let the compiler reassociate the very reduction whose
latency E4 measured, and every number in both probes would stop being about the code that ships.

---

## 8. AMENDMENT after run 1 — `VOID`, and the gate that was missing

Pushed **before run 2 exists**. Run 1 (`results/e5/arms.json`, analysis at
`results/e5/analysis.txt`, both committed) is **VOID at `T10`**, which is the point §5 judges.

### 8.1 What run 1 did

Five of the eight G1 predictions missed, two of them producing numbers that cannot be true:

| point | solved | what it says |
|---|---|---|
| `T10` @300 | `S` = **−0.544 ms** | running the softmax pass **twice** made the organ *faster* |
| `T10` @300 | `Y` = **−1.012 ms** | so did running the `A·V` loop twice |

A negative component is not a small component; it is an instrument saying the difference it was
asked for is smaller than the noise it is sitting in. The model was not wrong — it was never given
a measurement to be wrong about.

### 8.2 The defect, found in an organ no arm touches

`qkv_proj`, `o_proj`, `ffn` and `head` are on the weight path. **No E5 arm modifies any of them**,
so their sum `W` must be constant across the arms of a cell. It was not:

| cell | `W` min | `W` max | spread |
|---|---|---|---|
| `S05` @300 | 16.8 | 17.2 | +2.6% |
| `S05` @800 | 16.4 | 16.5 | +0.7% |
| **`T10` @300** | 304.3 | 363.4 | **+19.4%** |
| **`T10` @800** | 311.2 | 405.8 | **+30.4%** |

At `T10` @800 the three arms that produced the impossible numbers — `sm2`, `sm3`, `av2` — are
exactly the three that read `W ≈ 405` while every other arm read `W ≈ 311`. **The machine was
contended during part of the `T10` block**, run 1 took all three repetitions of an arm back to back,
so the episode landed inside single arms and presented itself as an arm effect. This is
`feedback_perf_parallelization`'s law — *a contended timing is not a timing* — arriving in a form
the existing gates could not see, because every gate E5 had was a comparison between arms and the
contamination was **in** the arms.

**Retroactive audit of E4 under the same check** (`results/e4/{arms2,shapes_avx4}.json`):

| sweep | worst cell spread on `W` |
|---|---|
| E4 run 2, all four cells | **0.2% – 0.7%** |
| E4 six-shape sweep, all twelve cells | **0.0%** |

So this is a fact about E5's run 1, not about E4's numbers, and nothing published from E4 moves.

### 8.3 G0 — pre-registered here, before run 2

> **G0.** In every cell, `W = qkv_proj + o_proj + ffn + head` is invariant by construction. Any
> single measurement whose `W` exceeds the **minimum `W` of its cell** by more than **5%** is
> discarded as contended, before any arm difference is taken. If fewer than **2** measurements
> survive for any arm in a cell, that cell is **VOID** and no share, label or ceiling is reported
> from it. The number discarded is reported per cell.

5% is not chosen from run 1's numbers: it is the between-sweep band E4 §2.4 fixed, and the claim
G0 makes is that anything above that band **inside one sweep** is the machine, not the code.
E4's sweeps clear it by an order of magnitude.

### 8.4 The one design change: rep-major interleaving

Run 1 was arm-major — three repetitions of `sm2`, then three of `sm3`, and so on — which is what let
one contention episode become one arm's result. **Run 2 is rep-major**: one repetition of every arm,
three times over. An episode is then spread across arms instead of concentrated in one, and G0 can
drop the individual measurements it touched instead of voiding the cell. The reported value per arm
is the median of its three interleaved single-rep measurements. This is the same remedy E4 §9 used
when its own label sat inside the instrument's dispersion, applied to eight arms instead of two.

### 8.5 A departure of the code from §2, recorded rather than back-edited

§2 said `av2`/`av3` would accumulate the discarded passes into a `float tmp[128]` on the stack and
fold them back as exact zeros, and it priced the 1 KB of extra L1 traffic that costs. **The arms as
built (`00f4538`) do not do that.** They run the same loop from zero 2× or 3× and reset between
passes with `out[i] -= out[i]`, which is exactly `+0` for any finite `out[i]` and — because it
*reads* `out[i]` — keeps the previous pass from being dead-coded. That is strictly better: no
scratch, **no extra traffic at all**, and `avrep==1` is a byte-for-byte copy of the baseline loop so
`none` is unchanged. §2 is left as written; this paragraph is the correction, and the 1 KB it
budgeted for is not spent.

The `clang -S` output confirms both constructions survive `-O3`
(`results/e5/asm_notes.txt`, taken before any arm was timed): four distinct `callq exp` sites for
the softmax arms, and a vectorised `vsubps %ymm0,%ymm0,%ymm0` for the `A·V` reset.

### 8.6 What run 2 does NOT change

The arms, the model, the 3% tolerance, the 0.30 ms `INCONCLUSIVE` floor, the label rule of §5, and
**the numeric predictions of §3** — `S` 0.8–2.0, `Y` 2.2–3.5, `P` 4.0–7.0, `fork2` 0.1–1.0 ms at
`T10` @800. Those ranges are **not re-fitted** now that run 1 has printed numbers against them:
run 1 is void, so its scoring of §3 is void with it, and a prediction revised after seeing even a
void measurement is not a prediction. They stand exactly as pushed at `c1bdd70`.

---

## 9. AMENDMENT after run 2 — the remedy worked, the gate still bites, and G0 is NOT being relaxed

Pushed **before run 3 exists**. Run 2: `results/e5/arms2.json`, analysis `results/e5/analysis2.txt`.

### 9.1 §8.4's remedy did what it was designed to do

Rep-major interleaving separated the machine from the arms. The `T10` @800 cell, measurement by
measurement, against its own cell minimum on the untouched weight path `W`:

| pass | `serial` | `serial2` | `none` | `sm2` | `sm3` | `av2` | `av3` | `fork2` |
|---|---|---|---|---|---|---|---|---|
| 1 | +0.0% | +22.9% | +34.5% | +36.5% | +31.7% | +20.9% | +6.1% | +4.5% |
| **2** | **+0.3%** | **+0.2%** | **+0.3%** | **+0.0%** | **+0.3%** | **+0.6%** | **+3.2%** | **+3.4%** |
| 3 | +3.9% | +4.3% | +11.9% | +21.9% | +13.6% | +3.6% | +3.1% | +5.8% |

**Pass 2 is a complete, clean set of all eight arms** measured inside one contiguous window — the
contention is bursty in time, not attached to any arm, which is exactly what run 1 could not show
and what §8.4 predicted.

### 9.2 What G0 says, and what is not being done about it

| cell | worst `W` excess | G0 |
|---|---|---|
| `T10` @300 | +4.8% | **PASS** |
| `S05` @800 | +6.1% | **PASS** |
| `S05` @300 | +16.0% | **VOID** (`sm3` has no clean measurement) |
| **`T10` @800** | +36.5% | **VOID** (`none`, `sm2`, `sm3` have none) |

G0 requires **two** clean measurements per arm. Pass 2 alone supplies one. **The gate is not being
relaxed to one** — it was fixed at two in §8.3 before run 2 ran, and a threshold moved after seeing
which side of it the data fell on is not a threshold. The cell needs **more samples, not a looser
gate**.

Run 2 did establish one cell under every gate it has: **`T10` @300 passes G0 and both of its 3×
predictions** — `sm3` at **+2.12%** and `av3` at **+0.63%** against a 3% tolerance fixed before any
arm existed. `S05` @800 passes G0 but its softmax 3× missed at **+3.24%**, so it is `FAIL` under
G1 and reported as such.

### 9.3 Run 3

Identical in every respect to run 2 except **six passes instead of three**. Same arms, same order,
same tolerances, same label rule, same §3 predictions — still not re-fitted.

**A second, independent witness to contention**, recorded per measurement from this run on:
`cores_busy` = `(kernel + user − idle)` CPU-seconds from `GetSystemTimes`, divided by the
measurement's wall time — the mean number of cores burning on the whole machine, by any process.
A clean 6-thread run sits a little under 6 (the single-threaded model load pulls the mean down);
anything materially above it is another process. **It is a witness, not a gate.** G0 decides what is
discarded; `cores_busy` either corroborates G0's verdict or contradicts it, and **a disagreement
between the two is reported rather than resolved in favour of whichever is convenient.** No second
discarding rule is being added after the fact.

### 9.4 The fallback, fixed now so it cannot be chosen afterwards

If `T10` @800 fails G0 again at six passes, E5 **reports the decomposition at whichever cells pass
and does not issue the §5 label at all.** It does not migrate the label to `T10` @300 or to `S05`,
and it does not report a share from a voided cell with a caveat attached. §5 names one judging point
and that point either produces a gated measurement or produces nothing.

---

## 10. AMENDMENT after run 3 — G1 did its job, and what it caught was my arms

Pushed **before run 4 exists**. Run 3: `results/e5/arms3.json`, analysis `results/e5/analysis3.txt`,
produced by `e5_analyse3.py`, which is frozen byte-for-byte as the file that issued this verdict.

### 10.1 Run 3's verdict

**G0 passes in all four cells** at six passes — the extra samples did what §9.3 said they would.
**G1 fails at the judging cell**: `T10` @800 `sm3` lands at **−0.99%** (PASS) and `av3` at
**+3.40%** (FAIL) against the 3% fixed before any arm existed. `S05` @300 `av3` misses at +7.04%
and `S05` @800 `sm3` at +4.83%.

Per §4, a component whose 3× prediction misses is **not established**, and `P` is a residual of
`R − S − Y`, so a failed `Y` takes `P` with it. **§5's label is therefore not issued from run 3.**
For completeness the arithmetic that would have produced one is in `analysis3.txt`; it is not
quoted here as a result, because a number from a failed gate is not a result.

Independently, §5's own G4 would have blocked it anyway: the leading share came out at **51.0%**,
one point from the 0.50 boundary, and G4 withholds a label that lands within three.

And a defect in my own analyser, found while writing this section and fixed here rather than after
run 4 could benefit from it: `e5_analyse.py` implemented G4 as `share < 0.50 and near`, so it
withheld a label only on the LOW side of the boundary and printed OVERHEAD-DOMINATED at 51.0%.
G4 is symmetric by construction -- a share inside the instrument's dispersion decides nothing in
either direction -- and the code now reads `abs(share - 0.50) <= 0.03`. The frozen `e5_analyse3.py`
keeps the bug, because it is the file that issued run 3's verdict and is not editable after the
fact; run 3's label is void for the reasons above regardless.

### 10.2 The cause, and it is not the machine

Both `av2 − none` and `av3 − av2` are, by construction, **one extra `A·V` pass**. They must be equal.
Measured against `none` as the 1× point:

| cell | `av2 − none` | `av3 − av2` | ratio |
|---|---|---|---|
| `T10` @300 | 0.672 | 0.681 | 1.01× |
| **`T10` @800** | **1.905** | **2.439** | **1.28×** |
| `S05` @300 | 0.079 | 0.135 | 1.71× |
| `S05` @800 | 0.148 | 0.190 | 1.28× |

`none` runs the **byte-for-byte baseline block**; `av2`/`av3` run the wrapped loop. Keeping `none`
byte-identical was a deliberate choice made in §2 so that the baseline would still be E4's baseline
— and it made the 1× point a *different piece of code* from the 2× and 3× points. `av2 − none` is
therefore "one extra pass **plus** the difference between two code shapes", and the 3× test, which
is blind to that offset, caught it. **This is the gate working exactly as designed: the prediction
that could miss, missed, and it missed for a reason that is mine and not the model's.**

### 10.3 Run 4 — the arms

`--attnr sm1` and `--attnr av1`: the **1× point inside the wrapped path**. `smrep`/`avrep` are now
runtime counts of 1, 2 or 3 taking one code shape, so:

```
S = sm2 − sm1        predict sm3 = sm1 + 2S        Y = av2 − av1        predict av3 = av1 + 2Y
```

and `sm1 − none`, `av1 − none` are reported as **the price of the code shape itself**, in their own
table, instead of being absorbed into a component. Both increments (`2×−1×` and `3×−2×`) are printed
side by side; if they disagree again the arms are still wrong and E5 will say so. `none` stays
byte-identical to E4's baseline and remains what `R` and `X` are measured against.

This is the same lesson E4 learned with `serial_e3` — a baseline that differs from the arms by
anything other than the thing under test will be charged for that difference.

### 10.4 Run 4 — the machine

Run 3 passed G0 **at 5%** in every cell and still could not resolve a component, because 5% of the
weight path (≈15 ms at `T10`) is larger than the components being estimated (`Y` ≈ 1.9 ms). Three
changes, all fixed here before the run:

- **G0 tightens to 1%.** Affordable: run 3's `T10` @800 keeps ≥2 measurements per arm even at 1%.
  Cells that cannot meet it are VOID, including `S05` if it comes to that.
- **The `cores_busy` witness of §9.3 is promoted to a gate**, announced in advance: a measurement
  more than **+0.30 cores** above its cell's minimum is discarded. Run 3 showed the two instruments
  already agree — kept measurements averaged 6.33–6.42 cores busy, discarded ones 6.99–7.59 — so
  this is a second view of the same contamination, not a new licence to drop inconvenient points.
- **Every arm runs at `HIGH_PRIORITY_CLASS`**, uniformly. It changes nothing about the code under
  test; it changes how often a desktop's background work preempts it. Because every E5 conclusion is
  a within-sweep difference, a uniform priority shift cancels — and it is recorded here because it
  makes this sweep's **absolute** numbers incomparable to E4's, which G3's drift line must now say.

### 10.5 What has still not changed

The §3 predictions — `S` 0.8–2.0, `Y` 2.2–3.5, `P` 4.0–7.0, `fork2` 0.1–1.0 ms — are **still not
re-fitted**, now against two runs that have printed numbers at them. §5's label rule, §4's 3%
tolerance and 0.30 ms `INCONCLUSIVE` floor, and §9.4's fallback all stand as written: if `T10` @800
cannot produce a gated decomposition at run 4, **E5 reports the cells that pass and issues no label**.

### 10.6 A note on the machine, for the record

The contention is real, external and persistent: the cleanest measurement in run 3 still read
**6.06 cores busy** for a 6-thread engine, with bursts to **9.54**. `MsMpEng` (Defender) is a
protected process and its CPU time is not readable, so it cannot be confirmed or cleared as the
cause; what is measurable is that a 5 GB weights file is re-opened by a new process for every one of
the 48 measurements in a cell. **This measurement wants a quiet machine** — the standing law that
speed needs an idle box, met head-on. Elevating priority is what can be done from inside the
experiment; closing the desktop's background work, or excluding `D:/_ktmp` from real-time scanning,
is not mine to decide and is not done here.

---

## 11. AMENDMENT after run 4 - the between-process design cannot resolve these components

Pushed **before run 5 exists**. Run 4: `results/e5/arms4.json` (240 records, 10 arms x 6 passes x 4
cells, every measurement at `HIGH_PRIORITY_CLASS`), analysis `results/e5/analysis4.txt`.

### 11.1 Run 4's verdict: VOID, at the gate fixed in section 10.4

| cell | measurements | kept at G0=1% | arms with <2 survivors |
|---|---|---|---|
| `T10` @800 | 60 | 18 | **2** - `av3`, `fork2` |
| `T10` @300 | 60 | 15 | 5 |
| `S05` @800 | 60 | 13 | 5 |
| `S05` @300 | 60 | 10 | 8 |

**All four cells are VOID.** G0 was tightened to 1% in section 10.4, before run 4 ran, and the rule
is that a cell with fewer than two clean measurements on any arm is void. The judging cell missed by
two arms. **It is not loosened now.** A gate moved after seeing which side of it the data fell on is
not a gate, and the whole point of fixing 1% in advance was to make this outcome cost something.

G2 passed: all eight new arms print `NATS_TOTAL 166667.1361128952`, bit-identical.

### 11.2 The witness that was promoted to a gate did nothing

Section 10.4 promoted `cores_busy` to a gate at +0.30 cores. Removing it changes the survivor count
in **none** of the four cells - every measurement it would have discarded, the 1% `W` gate had
already discarded. It is not a second, independent view of the contamination at this tolerance; it
is a coarser view of the same one. Reported because it was announced as a gate and earned nothing.

### 11.3 What run 4 actually establishes

The `W` excess over the cell minimum at `T10` @800, across all 60 measurements: **median 1.98%, p75
2.28%, max 8.54%** - against run 3's 29.6% max, so `HIGH_PRIORITY_CLASS` bought a real improvement,
and it is still not enough. The reason is arithmetic and does not wait for a quieter machine:

> The weight path at `T10` @800 is **301.2 ms**. One percent of it is **3.0 ms**. The component
> being estimated, `Y`, is about **1.9 ms**. A gate loose enough for measurements minutes apart to
> pass admits an excursion larger than the thing being measured; a gate tight enough to protect the
> measurement rejects two thirds of it.

**This is a property of the design, not of the run.** Three sweeps, each cleaner than the last, hit
the same wall: E5 compares arms that live in *different processes*, minutes apart, each re-reading a
5 GB weights file, and charges the difference between them to a 2 ms loop.

### 11.4 Run 5 - one process, arms interleaved per token

The fix is the one this project already wrote down after the 0.503-vs-0.50 case: when a difference
is smaller than the instrument's drift, **interleave** it. Run 5 puts every arm inside a **single
process** and rotates the arm **per token**:

- An arm is a **pair** `(--attn, --attnr)`, so the ten arms of run 4 - `serial/none`,
  `serial2/none`, `avx4/none`, then `sm1`, `sm2`, `sm3`, `av1`, `av2`, `av3`, `fork2` on `avx4` -
  all live in one run. `X`, `R`, `S`, `Y` and `P` then come out of a single model load, with no
  between-process term anywhere in the arithmetic.
- The rotation is a **palindrome of period 2n**: `idx = pos % 2n`, then `arm = idx < n ? idx :
  2n-1-idx`. Attention cost grows with position, so a plain round-robin would hand arm 0
  systematically shorter contexts than arm n-1; under the palindrome each arm holds positions `b+i`
  and `b+2n-1-i`, whose sum is the same for every `i`. **Every arm's mean position is exactly equal
  by construction, not by averaging.** At n=10 with `--bench 800`, each arm gets 80 tokens and each
  residue appears 40 times.
- A contention episode now lands inside a few tokens and is shared by every arm holding a token in
  that window, instead of landing on whichever arm happened to own the next four minutes.

### 11.5 The gates for run 5, fixed here

- **G2 (parity).** The rotation keys off `pos` inside `forward()`, so it is active in `--bpb` as
  well as `--bench`: a single interleaved run must print the **same** `NATS_TOTAL` as `none`. This
  is a strictly stronger G2 than run 4's - it tests every arm's value-preservation *while they are
  mixed*, which is the configuration the timings come from.
- **G0, now within-process.** `W` is still untouched by every arm, and the arms now share one
  process, so `W` per arm must agree **within 1% across the arms of a single run**. Same number as
  section 10.4, but now a test the design can pass: it asks whether the interleave distributed the
  machine evenly, not whether the machine held still for seven hours. A run whose arms disagree by
  more than 1% on `W` is discarded whole.
- **G1 unchanged**: `S = sm2 - sm1`, `Y = av2 - av1`, predicted at 3x and tested at **3%**, with
  both increments printed side by side. The `INCONCLUSIVE` floor stays at 0.30 ms.
- **G4 unchanged**: a leading share within 3 points of 0.50 is withheld, in either direction.
- **Repetitions**: **5 independent processes** per cell. The within-process arm differences are the
  measurement; the spread of those differences *across* the five runs is the reproducibility
  interval, and it is reported with every component. Fewer than 3 runs surviving G0 and G2 voids
  the cell.

### 11.6 The limitation this design introduces, stated before it is used

An interleaved arm is measured **while the other arms run around it**. If interleaving itself has a
cost - an instruction cache that no longer holds a single loop, a branch that is no longer predicted
- that cost sits in every arm. So: **`none` inside the sweep is the baseline for everything in run
5, and E4's `none` is never subtracted from a run-5 number.** The `1x - none` column of section 10.3
keeps pricing the code-shape difference, and if the sweep's `none` organ differs materially from run
4's `none` organ, that difference is reported as the price of interleaving rather than absorbed.

Section 3's predictions are still **not re-fitted**. Section 9.4's fallback still stands.
