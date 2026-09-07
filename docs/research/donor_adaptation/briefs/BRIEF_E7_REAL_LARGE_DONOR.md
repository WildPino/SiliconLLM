# E7 — a real large donor, end to end: the largest trained weights on this disk

**Pre-registered. Pushed before the export was run.** Opened by INDEX item **0-bis**.

---

## 1. Why this exists

E6 showed a donor generating text and could only do it at **0.5 B and 1.5 B**. Every measurement
at target scale in E3, E4 and E5 was taken on **`T10`, a synthetic shape file** — correct
dimensions, untrained weights. So the programme's headline currently reads:

> "10 B at 50 tok/s" has a **measured speed** and **no measured model**.

That is the gap this probe attacks, with the largest **real** weights available offline.

## 2. The donor

`Qwen/Qwen2.5-Coder-7B`, already in the HF cache (14.5 GB of bf16 safetensors). Nothing else
locally is bigger: `Qwen3-8B`, `Qwen3-30B-A3B` and `Qwen3-Next-80B-A3B` are **config stubs of 1 MB**,
not weights.

| | Coder-7B | `T10` (synthetic) |
|---|---|---|
| `d_model` / `d_ffn` | 3584 / 18944 | 4096 / 14336 |
| layers | 28 | 48 |
| heads (q/kv) | 28 / 4 | 32 / 8 |
| head_dim | 128 | 128 |
| vocab | 152064 | 32768 |
| tied embeddings | **no** | no |
| **active weights/token** | **≈ 7.07 B** | 10.603 B |

**7.07 B is 67% of the target size.** E7 does not claim 10 B. It converts "no measured model at
scale" into "a measured model at 7.07 B", which is the largest step available without a download.

## 3. What gets built

| artefact | export | why |
|---|---|---|
| `qwen25-coder7b_f32.bin` (~30 GB) | `--quant fp32 --fold layers` | the arm that can be checked against PyTorch. The largest file this runtime has ever been handed — E1 found a 32-bit `ftell` bug that made everything over 2 GB unloadable, and 30 GB is a new size class |
| `qwen25-coder7b_p.bin` (~5.7 GB) | `--quant packed --rule R0 --head-ternary --fold layers` | the **deliverable format**, and the arm the speed number comes from |

**`--rule R0`, not E1's R3.** R3 needs 32 calibration sequences through a 7.6 B fp32 model on CPU,
which is an hour of arithmetic that cannot change a **speed** result: the packed format is a fixed
2 trits/byte and the kernel scans it densely, so throughput is rule-independent. The quality of the
ternary conversion is not what E7 measures — E6 §2 already showed what it is.

## 4. The gates

| gate | test | threshold, fixed now |
|---|---|---|
| **G-L** | the engine consumes **exactly** the file, and its header matches `config.json` | exact byte count; every one of D, F, L, NH, NKV, HD, V |
| **G-P** | `--logits` from the fp32 arm vs PyTorch fp32, first 8 positions of a prompt | **rel L2 ≤ 1e-4** and **top-1 agreement 1.0000** — E1's circle at 5× the size it was gated at (E1 read 2.8e-06 / 1.5e-05 at 0.5–1.5 B) |
| **G-G** | `--generate` on the fp32 arm vs PyTorch greedy, E6's five frozen prompts, 32 tokens | **≥ 90% of 160 positions**, any divergence at a REF top-2 gap < 1e-2 |
| **G-C** | planted control: the packed arm must **not** pass G-G | else G-G is measuring prompt predictability and E7 is void |
| **G-S** | `--bench` on the packed arm, contexts 300 and 800, **≥ 3 repetitions**, dispersion reported | see §5 — a **pre-registered band**, not a free reading |

## 5. The prediction, written before the run

E3 measured `T10` at **3.090 tok/s** @300: 313.880 ms of weight path + 10.576 ms of `f`. Its
delivered weight rate is **32.8 G-weights/s**. Carrying that rate to Coder-7B's 7.07 B:

    weights   7.07 / 10.603 x 313.880  =  209.3 ms
    f         10.576 x (28x28)/(48x32) =    5.4 ms      (layers x heads, E4's engine is faster still)
    total                                 214.7 ms  ->  4.66 tok/s

**Pre-registered band at 300 tokens of context: 4.0 – 5.4 tok/s.** Outside it, either the
weight-rate model or the measurement is wrong, and E7 says which before it says anything else.

**This band is also the point.** If it holds, a *real* 7 B donor runs about **11× short of
50 tok/s**, on an engine whose attention organ has already been optimised 6.5× (E4) and taken apart
(E5). It would say, in measured units on trained weights, that the wall is the **weight path** and
nothing in the attention organ can move it.

## 6. What E7 will not claim

- **Not 10 B.** 7.07 B active is 67% of the target and the brief says so in §2. The remaining step
  needs a download the disk has not got.
- **Not a quality result.** R0 ternary at 7 B will not write language; E6 §2 established what the
  ternary conversion costs and E7 adds nothing to it.
- **Not that the fp32 arm is a deliverable.** A 30 GB fp32 file is a measuring instrument.

---

## 7. AMENDMENT — the first export attempt was killed for memory, and what that costs

The packed export ran, loaded all four checkpoint shards (7 min 28 s) and was then **killed by the
system for low memory**. `AutoModelForCausalLM(dtype=float32)` on a bf16 checkpoint materialises
**~30 GB** of parameters before a single byte is written, and this machine carries a large resident
background set.

**The fix, and its risk.** `qwen_export.py` gains `--load-dtype {float32,bfloat16}`. Under
`bfloat16` the donor is held at its **own checkpoint precision** (~15 GB) and every tensor is
widened to fp32 at write time — inside `w_fp32` and inside `quantize`, one place each, so the
arithmetic of the rules is unchanged. bf16 -> fp32 only pads the mantissa, so the artifact should be
**byte-identical**. "Should be" is exactly the kind of claim this programme does not accept.

**Control, threshold fixed before it ran:** export `Qwen2.5-0.5B --quant fp32 --fold none` twice,
once per `--load-dtype`, and compare **sha256**. Anything but equality means the low-memory path is
a different exporter and E7's artifacts cannot inherit E1's parity.

**What it costs.** `--load-dtype bfloat16` **refuses** `--fold` and `--rule R3`: folding multiplies
a gain into every row and R3 runs calibration forwards, and both would then be bf16 *arithmetic*
rather than bf16 *storage*. So E7's two arms are exported **`--fold none`**, unlike E1's. That is
sound for what E7 measures — the fold is a **quality** optimisation (E2, −0.220001 BPB), it changes
neither the file layout nor the weight count, and therefore cannot move a speed number. It does
mean **E7's packed arm is not E1's `tqh` operating point** and its BPB is not comparable to one.
E7 does not report a BPB.

---

## 8. VERDICT — `REAL-WEIGHTS-CONFIRMED` (added after the run; nothing above was edited)

Written up in `probes/E7_REAL_LARGE_DONOR.md`. All five gates passed.

| gate | threshold from §4 | result |
|---|---|---|
| G-L | exact byte consumption, header == config | **PASS** — `consumed exactly 30462466100 bytes` |
| G-P | rel L2 <= 1e-4, top-1 == 1.0000 | **PASS** — 1.070e-05, 1.0000 |
| G-G | >= 90% of 160 greedy positions | **PASS — 160/160** |
| G-C | the packed arm must NOT pass | **PASS — 0/160** |
| G-S | median inside 4.0-5.4 tok/s at 300 context | **PASS — 4.460**, spread 0.45% over 3 reps |

**The prediction in §5 was 4.66 tok/s and the measurement is 4.460, off by -4.3%.** What passed is
E3's delivered weight rate carried onto a shape it had never seen, on trained weights.

**The unplanned result is §4 of the probe.** Delivered weight rate: `T10` synthetic 10.6 B gives
32.8 G-weights/s, real Coder-7B gives **31.5** -- **-4.0%, inside this programme's own +/-5% band
for an absolute rate.** The synthetic shape files E3, E4 and E5 all relied on are a **sound speed
proxy**, checked rather than argued for the first time.

**And the goal is unmoved.** A real 7 B donor decodes at 4.460 tok/s: **11.2x short of 50**, on an
engine whose attention organ is already 6.5x faster (E4) and fully decomposed (E5). The gap is
entirely the weight path, and that is now a statement about trained weights rather than a shape.

**The exporter control over-delivered**: the 0.5 B sha256 pair matched each other AND matched E1's
own `qwen25-05b_f32.bin` byte for byte, so `--load-dtype bfloat16` is provably the same exporter.

---

## 9. EXTENSION, pre-registered before it ran — is there a fixed cost inside the timed region?

§8 left one thing unresolved and it is not cosmetic. At this shape the **800-context cell is faster
than the 300-context cell** (4.580 vs 4.460), the opposite of `T10`. Two-point arithmetic on the
wall times (67.276 s, 174.723 s) fits a marginal **214.894 ms/token** plus a **fixed 2.808 s**
inside the timed region. If that fixed term is real, **every short `--bench` in the SPEED_LEDGER
reads low**, including the 300-context cells §13 and §16 quote.

A two-point fit cannot tell a fixed cost from a slowly-growing one. A third point can.

**G-X1 — the discriminator.** `--bench 1600` on the packed arm, 3 repetitions.

| model | prediction at 1600 |
|---|---|
| fixed cost + constant marginal (the fit) | `1600 x 0.214894 + 2.808 = 346.6 s` -> **4.616 tok/s** |
| no fixed cost, `f` growing with position | **below 4.580** |

**The call, fixed now: if the 1600 median is ABOVE 4.580 the fixed-cost reading survives; if it is
BELOW 4.580, it is refuted and `f` growth is what the 300/800 pair was showing.** Reported band
for the fit: **4.50 – 4.68** (the linear extrapolation minus whatever `f` adds between 800 and
1600, which the fit cannot see).

**G-X2 — `f` beyond 800 tokens, which is E4's owed item 3.** `--bench {300, 800, 1600} --profile`,
one repetition each, reporting `rope + attention + norm/glue` in ms/token. Profiling perturbs
absolute timings (`SPEED_LEDGER` §12: the profiler once manufactured a 9.6% anomaly), so **these
runs are used for the organ SPLIT only and never for a rate.** Nothing bounds `f` above 800 today;
this is the cheapest place it has ever been measurable, because the weight path here is ~215 ms
and `f` is a few, so a 2x change in `f` is visible without being confounded by the weights.

---

## 9-bis. VERDICT on the extension (added after G-X1/G-X2 ran; nothing above was edited)

**G-X1 — the fixed-cost reading is REFUTED.** `--bench 1600`, packed arm, 3 reps:

    4.05 / 4.23 / 4.31 tok/s   ->  median 4.230, spread 6.1%

§9's call was fixed before the run: *above 4.580 the fixed-cost reading survives, below 4.580 it is
refuted.* **4.230 is below**, and it is below the fit's own reported band (4.50–4.68) by more than
the band's width. The fit predicted **4.616** and missed by **−8.4%** — well outside the ±5% that
governs an absolute rate here.

So there is **no fixed cost inside the timed region**, and none of the SPEED_LEDGER's short
`--bench` cells reads low. What the 300/800 pair was showing is `f` growth, sampled too coarsely
to see: between 300 and 800 the attention organ costs less than the run-to-run noise, so the pair
inverted on noise, not on a mechanism. **My §7 two-point fit was fitting noise** — the 300→800
inversion is 2.7%, inside this programme's own ±5% band, and I should not have modelled it. It cost
nothing only because §7 labelled it "a two-point fit, not a measurement" and §9 tested it.

The 6.1% spread at 1600 is itself the reason a third point was needed rather than a third opinion.

**G-X2 — `f` beyond 800 is measured, and it is linear.** E4's owed item 3, closed. Profiled runs
are admissible only where the **`ffn`-invariance witness** holds: the FFN organ cannot depend on
context length, so any cell whose `ffn` sits above the uncontended ~175–180 ms plateau was taken
under load and its attention reading is thrown away. Six of nine profiled cells survived:

| context | `ffn` (witness) | **attention ms/token** |
|---|---|---|
| 300 | 176.935 | **4.932** |
| 800 | 177.832 / 175.730 | **13.092 / 12.713** |
| 1600 | 176.193 / 179.518 / 180.119 | **25.462 / 25.636 / 25.951** |

Discarded by the witness: `ffn` 237.124, 228.602, 213.422 — the cells that would have read
attention as 8.031, 6.428 and 16.067 and manufactured a knee that is not there.

Slope, on the surviving cells:

| interval | Δ attention | per token of `--bench N` | **per token of actual context** |
|---|---|---|---|
| 300 → 800 | 7.97 ms | 0.01594 | **0.0319 ms** |
| 800 → 1600 | 12.78 ms | 0.01597 | **0.0319 ms** |

(`--bench N` averages the organ over positions 0…N−1, so mean context is N/2; the third column is
the physical quantity.) **The two intervals agree to 0.2% over a 5.3× range in context. `f` is
linear to 1600 tokens with no knee**, which is what a KV-cache-streaming organ should do and is
now measured rather than assumed.

**Correction to a claim I was about to make.** Comparing E7's attention (12.90 ms @800) against
E4's `T10` figure (12.058 @800) looked like a ~2.2× anomaly: Coder-7B has 28 layers × 28 heads =
**784 head-layers** against `T10`'s 48 × 32 = **1536**, so it should cost **0.510×**, not the same.
The comparison was wrong, not the engine — E4's 12.058 is the **`avx4`** arm and E7 was running the
**default**, which §10 shows is `serial`. Against E4's own `serial` row, 24.463 ms:

    predicted   24.463 x (784/1536) = 12.49 ms
    measured    12.90 ms  (median of the two admissible 800 cells)   ->  +3.3%

**Inside the band.** A cross-model, cross-shape prediction of an organ cost from head-layer count
alone, correct to 3.3%. There was never an anomaly.

---

## 10. AMENDMENT, pre-registered before the run it governs — the engine's default attention kernel is not the one E4 won with

**The finding.** `donor_engine.c:116`:

    static int g_attn=ATTN_SERIAL;

The default attention kernel is `serial` — E4's **second-slowest** arm (`serial` 24.463 ms at
`T10` @800, against `serial_e3` 23.978 and `avx4` **12.058**). E4's 6.53× win on the `Q·K` dot loop
is reachable **only** via an explicit `--attn avx4`, and was never wired in as the default. E4's
own runners pass `--attn` on every point (`e4_g4prime.sh`, `e4_interleave.sh`), so E4's table is
sound; but **`e3_bench.py` defaults `--attn` to the empty string and does not pass it**, and
neither did `e7_real7b.py`. Every E7 number above, and E3's `T10` baseline, is on `serial`.

**Already in hand when this section was written** (declared, not hidden): one profiled `avx4` cell
at 300 context, `ffn` 178.898 — admissible under the witness — reading attention **2.336 ms/token**
against `serial`'s 4.932. That is **0.474×**, which matches E4's `T10` ratio of 0.493× to within
4%: the arm transfers to a real donor at a different head count.

What is **not** in hand, and is what this section pre-registers:

**G-Y1 — does the rate move?** `--bench {300, 1600} --attn avx4` on the packed arm, **3 reps each,
un-profiled**, dispersion reported, machine otherwise idle.

The prediction is arithmetic on organ costs already measured, and it is deliberately unflattering:

| cell | token now | attention saved | predicted token | **predicted tok/s** | measured now |
|---|---|---|---|---|---|
| 300 | 224.2 ms | 4.932 − 2.336 = **2.60 ms** | 221.6 ms | **4.512** (+1.2%) | 4.460 |
| 1600 | 236.4 ms | 25.68 − ~12.2 = **13.5 ms** | 222.9 ms | **4.487** (+5.7%) | 4.230 |

**The call, fixed now: at 300 the gain is ~1% and therefore BELOW this programme's own ±5%
resolution — G-Y1 at 300 is expected to be a NULL, and a null there does not refute the kernel.
At 1600 the gain is ~5.7% and sits right at the edge; that is the cell that can actually speak.**
Bands: 300 → **4.35–4.70**; 1600 → **4.30–4.70**. A 1600 median at or below 4.230 (today's serial
figure) refutes the claim that the kernel reaches the rate at all.

**G-Y2 — what this does and does not cost the ledger.** Whatever G-Y1 reads, the honest accounting
is fixed here before seeing it: attention is **2.2% of the token at 300 and 10.5% at 1600**, and
the FFN is **73–80%**. Halving attention cannot move an 11.2× gap. **§10 is a correctness note
about which kernel ran, not a speed result**, and E7's verdict, its 4.460 tok/s, its ±4.3% against
the pre-registered prediction and its 31.5 G-weights/s proxy check all stand unless G-Y1 moves the
300 cell by more than ±5% — which the table above predicts it will not.

**G-Y3 — the ledger's `T10` figures.** E3's 3.090 tok/s at `T10` @300 was taken on the `serial`
default. E4 measured `avx4` at `T10` @800 as **3.230 tok/s against `serial`'s 3.110 — +3.9%**,
already published, already inside ±5%. So the ledger does not understate the engine by more than
its own resolution, and **§13–§16 need no numeric correction**; they need the sentence that says
which kernel produced them. That sentence is owed regardless of how G-Y1 reads.

**What is NOT proposed here.** Changing the default. `serial` is the arm every prior probe's
baseline was taken on; flipping `g_attn` would silently re-base E1–E7. The default is a
**documentation** defect, and the fix is a runner that passes `--attn` explicitly and a ledger line
that names the kernel — not an edit that makes old numbers unreproducible.

---

## 10-bis. VERDICT on G-Y1 (added after the run; nothing above was edited) — and the instrument was not good enough

`--bench {300, 1600} --attn avx4`, packed arm, 3 reps each, un-profiled, machine idle.

| cell | `serial` (E7 §3 / §9) | **`avx4`** | Δ | §10 predicted | pre-registered band |
|---|---|---|---|---|---|
| 300 | **4.460** (spread 0.45%) | **4.450** (2.5%) | **−0.22%** | +1.2% | 4.35–4.70 ✓ |
| 1600 | **4.230** (spread **6.1%**) | **4.290** (1.2%) | **+1.42%** | +5.7% | 4.30–4.70 ✗ by 0.23% |

**The 300 cell is the NULL §10 called in advance, and it lands exactly where §10 said it would.**
Attention is 2.2% of the token there; a 0.474× kernel on 2.2% is ~1%, and this programme cannot
resolve 1%. Nothing is learned and nothing was expected to be.

**The 1600 cell did not decide, and the reason is the instrument, not the engine.** §10 predicted
+5.7% from an organ saving of 13.5 ms. The saving is real and was measured directly and
profiled-admissible: attention **25.64 → 11.315 ms/token, a saving of 14.33 ms**. But the *rate*
moved by only **3.31 ms/token** — **4.3× less than the organ says it should have.**

Before that gap gets a mechanism, note what §9-bis just finished teaching: **the `serial` 1600
baseline has a 6.1% spread** (4.05 / 4.23 / 4.31), and its own maximum, 4.31, sits **above the
`avx4` median of 4.290.** E4's law applies verbatim — *a threshold inside the dispersion of the
instrument cannot decide* — and here the dispersion (6.1%) is larger than the effect (5.7%).
**G-Y1 at 1600 is underpowered by its own baseline. It is not evidence either way**, and the
0.23% band miss is not a refutation of anything.

I am not going to fit a mechanism to a 4.3× discrepancy measured against a baseline that cannot
support it. §11 builds the instrument that can.

## 11. AMENDMENT, pre-registered before the run it governs — pair the arms instead of comparing two sweeps

**G-Y1b.** The two arms at 1600 were measured as **separate sweeps**, minutes apart, and
`SPEED_LEDGER` §13 already fixes the between-sweep dispersion at **5–10%** while stating that
**ratios do not carry it**. So measure a ratio: **interleave the arms per repetition**, the method
E4 built for exactly this (`e4_interleave.sh`).

    for k in 1 2 3:  serial @1600  ->  avx4 @1600      # un-profiled, idle machine

**Three paired ratios**, reported individually and as a median, plus each arm's own spread.

**The call, fixed now.** The organ measurement says `avx4` removes **14.33 ms** from a **236.4 ms**
token at 1600, which is **+6.4%**, so:

| outcome | reading |
|---|---|
| paired median ratio **≥ 1.05** | the organ win reaches the rate; §10's G-Y2 accounting stands and the ledger gains a kernel note plus a real 1600-context figure |
| paired median ratio **1.01 – 1.05** | the win **partly** reaches the rate; the residue is unexplained and becomes an owed item, not a story |
| paired median ratio **≤ 1.01** | **the organ win does NOT reach the rate.** That is the interesting outcome and it would mean the profiler's organ split is not additive to the wall at this shape — a defect in the instrument that reads every table in `SPEED_LEDGER` §12–§14 |

**Band: 1.00 – 1.09.** A paired ratio outside it means the pairing itself failed and the run is VOID.

**What does not change either way**, restated so no later section can quietly forget it: attention
is 2.2% of the token at 300 and 10.5% at 1600, the FFN is 73–80%, and E7 is **11.2× short of 50
tok/s**. §11 can move a 1600-context rate by at most 6.4%. **It cannot move the goal, and it is not
being run as though it could** — it is being run because §10-bis found a 4.3× discrepancy between
an organ and a wall, and a discrepancy that size in the measuring apparatus is worth 40 minutes
whatever it turns out to be.

---

## 11-bis. VERDICT on G-Y1b — **VOID by its own band**, and the defect it exposes is bigger than the question

`--bench 1600`, arms interleaved per repetition, un-profiled:

| pair | `serial` | `avx4` | ratio |
|---|---|---|---|
| 1 | 4.17 | 3.96 | **0.950** |
| 2 | 4.04 | 4.42 | **1.094** |
| 3 | 4.11 | 4.00 | **0.973** |
| | spread **3.2%** | spread **11.5%** | **median 0.973** |

**§11 fixed the band at 1.00–1.09 and said a paired median outside it means the pairing failed and
the run is VOID. 0.973 is outside. G-Y1b is VOID.** No number from it enters anything.

**I am not naming a cause I did not measure.** A CPU-load sample taken immediately after the run
read 46%, but three samples minutes later read 7 / 13 / 14%, and nothing of mine was running in
either case. A single instantaneous sample is not a load measurement, and "the machine was busy"
is exactly the *plausible* artefact the planted-control law says this programme fails on. **What
made the `avx4` arm scatter 11.5% is unknown.**

What is *not* unknown is the consequence: **within-arm dispersion at the 1600 cell reached 11.5%,
roughly double the largest effect attention can produce there (6.4%). The un-profiled 1600 cell
cannot resolve this question with this instrument at all** — G-Y1 was underpowered, and G-Y1b
shows the pairing does not rescue it.

### 11-bis.1 The defect: `--bench` has no contention witness

The `ffn`-invariance witness that made §8.2 trustworthy — the FFN organ cannot depend on context
length, so a cell whose `ffn` leaves the plateau was measured under load — **exists only when
`--profile` is on.** On the plain `--bench` path there is nothing to check. **Every un-profiled
rate this programme has published rests on the operator's belief that the machine was idle, with
no instrument that can contradict it.** That is a general defect, it is new, and it is worth more
than the question §11 was asked.

### 11-bis.2 The question §11 was built for, answered from data already in hand

§11's third and most consequential outcome was: *the profiler's organ split is not additive to the
wall, which reads every table in `SPEED_LEDGER` §12–§14.* **That outcome is excluded**, using the
four witness-admissible 1600-context profiled cells (`ffn` all inside the 176–181 plateau):

| cell | attention | `ffn` | TOTAL | wall | TOTAL vs wall | **TOTAL − attention − `ffn`** |
|---|---|---|---|---|---|---|
| `serial` A | 25.462 | 176.193 | 242.130 | 242.083 | **+0.019%** | 40.475 |
| `serial` B | 25.636 | 179.518 | 247.313 | 247.265 | **+0.019%** | 42.159 |
| `serial` C | 25.951 | 180.119 | 247.713 | 247.668 | **+0.018%** | 41.643 |
| **`avx4` D** | **11.315** | 178.458 | 231.005 | 230.936 | **+0.030%** | **41.232** |

Two things, and neither needs a new run:

1. **The organs sum to the wall to 0.03% in every cell.** The split is additive by construction and
   by measurement.
2. **The remainder after removing the two organs that moved is arm-invariant.** `avx4`'s 41.232
   sits *inside* the `serial` arm's own scatter (40.475–42.159, a 4.1% range). Swapping the
   attention kernel changed the attention organ and nothing else.

Taking the two cells with the closest `ffn` (B, 179.518, and D, 178.458): attention falls
**14.32 ms**, `ffn` differs **−1.06 ms**, so the wall should fall **15.38 ms**; it falls
**16.33 ms**. **Agreement to 0.95 ms — 0.4% of the token.**

**So the `avx4` win does reach the wall, by about 6–7% at 1600 context.** That figure is stated
from profiled walls and is therefore **NOT published as a rate** (`SPEED_LEDGER` §12: the profiler
once manufactured a 9.6% anomaly, and profiled runs are for the organ split only). **E7 publishes
no `avx4` rate.** The honest position is: the organ win is measured, its arithmetic consequence is
consistent to 0.4%, and no un-profiled instrument on this machine can resolve 6–7% at the 1600
cell.

### 11-bis.3 What still stands, unchanged

§10's G-Y2 accounting, fixed before any of this ran, is exactly what happened: attention is 2.2%
of the token at 300 and 10.5% at 1600, the FFN is 73–80%, and **E7 remains 11.2× short of 50
tok/s.** Two runs, ~80 minutes, and the goal moved by nothing — which is what §10 and §11 both
said in advance they would be worth, and is why they were cheap to be wrong in.

### 11-bis.4 Owed, and it is now the first item

**Give `--bench` a contention witness.** The cheapest form: time the `ffn` organ on the plain
`--bench` path and print it on the `BENCH` line, so every rate this programme publishes carries a
number that says whether the machine was quiet. **Gate: the `BENCH` rate must be unchanged** —
adding a timer to the hot path to measure contention would be a fine way to manufacture some.
