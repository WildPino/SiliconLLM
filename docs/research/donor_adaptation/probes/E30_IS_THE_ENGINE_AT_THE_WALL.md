# E30 — is the engine still core-bound, or has the fast kernel put it against the memory wall?

**Verdict: `AT-THE-WALL`.** The registered cell reads **0.909** — `T10-LUTBLK` moves **33.01 GB/s**
against a ceiling of **36.30 GB/s** measured on this box, at this thread count, in this session.
An *infinitely good* kernel at the goal's shape reads **6.83 tok/s**. That is the hard cap on all
future kernel work on this machine, and the engine is already **1.10×** from it.

**Brief**: `briefs/BRIEF_E30_THE_WALL.md`, pushed at `63515b4` before the runner existed.
**Runner**: `benchmarks/donor_adaptation/engine/e30_the_wall.py`.
**Instrument**: `benchmarks/donor_adaptation/engine/e30_bandwidth.c`.
**Result**: `engine/results/e30_the_wall.json`. 94 s, idle box — witness **2.8% mean / 7% peak**
before the run against a 25% bar.

Everything below is in the **moved-byte** convention on both sides of every ratio. Charged weights
appear only where they are labelled as such, and never inside a fraction with a byte.

---

## 1. The gates

| gate | what it demands | reading | |
|---|---|---|---|
| `G-E30A` | **planted control.** The bandwidth bench must show this box's L3 cliff — resident ≥ 3× the DRAM plateau — or it is not measuring memory and reports no ceiling | resident **454.3 GB/s**, plateau **36.30 GB/s**, **12.5×** | **FIRES** |
| `G-E30B` | the byte accounting must reproduce the artifact **to the byte** | accounted `5,849,628,724` vs file `5,849,628,724`, **unaccounted 0** | **FIRES** |
| `G-E30C` | each engine arm must reproduce E28's published rate within ±10%, or the comparison is not licensed | `T10-PACKED` **+0.23%**, `T10-LUTBLK` **+6.88%** | **PASS** |

### 1.1 `G-E30B` caught the brief's own table

The brief's §1 accounting — pre-registered, and mine — **omitted the q/k/v fp32 bias vectors**,
`(QD + 2·KD)·4 = 24,576` bytes per layer. It read **5.3116 GB/token** where the file says
**5.31276**. The error is `+0.022%` and changes no conclusion in this document, but it is exactly
what the gate was written to catch, and the gate demands the file size *to the byte*. It now
reproduces exactly:

| | bytes |
|---|---|
| header (`QWENDON1` + 9 ints + 2 floats) | 52 |
| embedding table, fp32 (read: **one row**, 16,384) | 536,870,912 |
| 48 layers × 109,281,280 | 5,245,501,440 |
| `model.norm` fp32 | 16,384 |
| head (`32768×4096` packed + row scales) | 67,239,936 |
| **file** | **5,849,628,724** ✓ `os.path.getsize` |
| **moved per token** (layers + head + one embedding row) | **5,312,757,760 = 5.31276 GB** |

## 2. The ceiling, measured and not cited

`e30_bandwidth.c` is a read-only streaming sum — four accumulators per thread (E8's law, so the
loop is a byte stream and not a latency chain), a volatile sink so nothing is elided, built with
the engine's own flags and never `-ffast-math`. Read-only on purpose: the engine's weight path
reads megabytes of weights against one activation vector, and a STREAM-style triad would mix in
write bandwidth and flatter the denominator.

| buffer | best GB/s | median GB/s | |
|---|---|---|---|
| 4 MB | 301.8 | 255.8 | |
| 12 MB | **454.3** | 280.2 | L3-resident |
| 16 MB | 448.6 | 233.7 | **Probe-3's cliff edge** |
| 20 MB | 401.8 | 78.7 | the fall |
| 24 MB | 98.7 | 59.5 | |
| 32 MB | 57.5 | 42.3 | |
| 64 MB | 41.5 | 39.5 | |
| 256 MB | 37.4 | 37.2 | |
| 1 GB | 36.6 | **36.30** | **`BW-CEIL`** = median of sizes ≥ 1 GB |
| 2 GB | 35.6 | 34.67 | |
| 4 GB | 39.2 | 38.36 | |
| 8 GB | 35.5 | 34.81 | |

The cliff falls exactly where Probe-3 put it, at **16 MB**, which is why `G-E30A` fires and the
ceiling counts. The resident number (454 GB/s) is far above Probe-3's ~100 GB/s because the
access pattern is different — a pure sequential sum, not a matvec. **That does not matter here:
the gate asks whether the instrument can tell L3 from DRAM, and a 12.5× step answers it.**

## 3. The engine against the ceiling

| arm | tok/s (3 reps, interleaved) | spread | **GB/s moved** | **÷ `BW-CEIL`** |
|---|---|---|---|---|
| `T10-PACKED` | 4.32 / 4.42 / 4.35 → **4.363** | 2.3% | **23.18** | **0.639** |
| `T10-LUTBLK` | 6.21 / 6.23 / 6.20 → **6.213** | 0.5% | **33.01** | **0.909** |
| *ceiling* | *6.832* | | *36.30* | *1.000* |
| *the goal* | *50* | | *265.64* | ***7.318*** |

**`AT-THE-WALL` (≥ 0.85).** And the two rows together say what changed between E10 and now:
E10's "core-bound" verdict was **right for the kernel it tested** — the packed path still sits at
0.639, with a third of the machine's bandwidth on the floor — and what moved is the kernel, not
the machine. E13's blocked tile-major layout spends the third that E10 was wasting, and there is
essentially nothing left to spend.

### 3.1 The band name straddles the boundary, and I am saying so before anyone finds it

The verdict cell landed **0.059 above** the `0.85` boundary, and the noisy side of that fraction
is the **denominator**. An earlier hand-run of the same sweep, same box, same evening, read a
plateau of **39.54 GB/s** rather than 36.30. So after the registered result I ran the instrument
three more times (7 reps each) — **post-hoc, a dispersion bound on the ceiling, NOT a gate and NOT
a re-verdict** (E14 §6 forbids promoting a post-hoc metric to a gate; the registered cell stands
exactly as measured):

| ceiling used | GB/s | ratio | band |
|---|---|---|---|
| **registered** (this run's median ≥ 1 GB) | **36.30** | **0.909** | **`AT-THE-WALL`** |
| pooled median of 16 plateau readings, 4 sweeps | 37.81 | 0.873 | `AT-THE-WALL` |
| **the single best reading ever observed** | 40.81 | 0.809 | `PARTIALLY-BOUND` |

**Against the most generous denominator this box has ever produced, the engine is still using 81%
of the machine's entire read bandwidth, and the headroom bound is 1.24×.** The band name is
`AT-THE-WALL` on the registered cell and on the pooled median, and only falls to the top of
`PARTIALLY-BOUND` against a cherry-picked best-of-28. **Either way §6's disposition is the same**,
which is why the brief wrote one disposition for both bands.

### 3.2 The `--lutblk` anchor read 6.88% high, and that is the box, not a new lever

`T10-LUTBLK` reads **6.213** here against E28's **5.813**. This session's witness was 2.8% mean
where E28's was 5.5%, and the spread here is 0.5% against E28's 12.4%. Charged throughput on the
E28 convention is therefore **65.88 G active weights/s**, not 61.64 — E28's published numerator is
**conservative**, and the shape lever reads **1.424×** here against E28's 1.338×. Both anchors are
inside ±10% so the comparison is licensed; **the ledger keeps E28's numbers and gains this one as
a second reading, it does not replace them.**

## 4. What the wall permits, in the currency the programme budgets in

At the wall, `tok/s` is **linear in bytes per token** — which is the one genuinely useful
consequence, because it turns the programme's budget arithmetic from optimistic into *exact*.

**50 tok/s at T10 permits `36.30 / 50 = 0.726 GB/token`.** In packed ternary that is about
**1.452 G weights/token** against the dense token's **10.6032 G**: a **7.3× cut in MOVED weights**,
with a perfect kernel, on a machine at 100% of its measured bandwidth.

| budget for 50 tok/s at T10 | G weights/token | where it came from |
|---|---|---|
| E10/E25 numerator 49.9 G-w/s | 0.998 | one kernel, one layout |
| E28 numerator 61.64 G-w/s | 1.2328 | E13's layout at the goal's shape |
| **E30 — the physical bound** | **1.452** | **the machine's entire read bandwidth, perfect kernel** |

**E28's widened budget already sits at 85% of the physical bound.** The numerator has `1.178×` left
in it *in total, forever, on this box* — and getting it requires a kernel that is not merely better
than `--lutblk` but perfect.

And the quality side does not reach: E27 measured that **every** configuration inside `0.998 G`
is broken (`FLOOR-MIN` 38/160) and every configuration that holds leaves the budget **5–8× away**.
Widening 0.998 → 1.452 is `1.45×`. It does not close a 5–8× gap.

## 5. What the wall does NOT forbid — and this is the constructive half

The ceiling is on **DRAM-streamed** bytes. The same instrument measures **454 GB/s resident**, a
**12.5×** different machine on the other side of a 16 MB line. Weights that live inside L3 do not
pay the wall at all:

> A 16 MB resident core costs **0.035 ms** of a 20 ms token budget — **0.18%**. It is free.

That is **~32 M ternary weights per token at no bandwidth cost**, which is Probe-3's independently
measured "≤16 MB active slice ≈ 24–40 M ternary params/token" arriving from the bandwidth side.
So the honest specification of the 10 B goal on this machine is:

> **≈ 32 M weights resident and free, plus ≤ 1.45 G streamed — per token, for 50 tok/s.**

That is not a kernel specification. It is `SCALEUP_ARCHITECTURE.md`'s split — a resident
*thinking* core and a streamed *knowing* pool — written as a bandwidth inequality, and E30 is the
first measurement that prices both halves in the same units.

**Three things this does not license.** (i) **Batching and speculative decode amortize weight
reads over several tokens and are outside the goal**, which is single-stream decode. (ii) The
accounting is **weights only**, at the 40-token contexts E28 used; at long context KV traffic adds
to the load and moves the wall **closer**. (iii) The ceiling is **this box's** — the reference
floor this programme designs against. More channels move the wall; they do not move the shape of
the argument.

## 6. What this decides

The brief wrote the disposition for `AT-THE-WALL` and `PARTIALLY-BOUND` identically, before the
run, and it now applies:

> **The numerator is finished as a research direction.** The remaining gap is not available at any
> kernel quality, and the only variable left is bytes per token.

Concretely, and this is a change of plan I am recording rather than a preference:

1. **E28 §8 item 1 — the per-matrix `--lut` guard at `donor_engine.c:1437` — is DEMOTED from "the
   next build" to "worth doing".** Its value is real and measured (`1.34×` to rank and carve
   artifacts, which today cannot load the fast kernel at all) but it is *bounded by this ceiling*:
   it moves a rank arm from packed's 0.639 toward `--lutblk`'s 0.909, and **cannot exceed 1.24× at
   the most generous ceiling this box produces**. It buys byte-efficiency, not bytes.
2. **Every subsequent probe belongs on the denominator**, and it now has a hard target rather than
   a moving one: **1.45 G moved weights/token**, of which ~32 M may be resident and free.
3. **The honest recommendation for the 10 B goal becomes architectural**, which is what
   `SCALEUP_ARCHITECTURE.md` and Phase 64 already describe — and §5 gives the inequality they have
   to satisfy.

## 7. Predictions, scored

Fixed in the brief at `63515b4`, before the runner existed. **4 HIT, 1 MISS.**

| # | prediction | reading | |
|---|---|---|---|
| 1 | `G-E30A` fires, resident ≥ 3× plateau | 12.5× | **HIT** |
| 2 | `BW-CEIL` in 25–45 GB/s | 36.30 | **HIT** |
| 3 | verdict `PARTIALLY-BOUND`, ratio 0.70–0.90 | **0.909 → `AT-THE-WALL`** | **MISS** |
| 4 | `T10-PACKED ÷ BW-CEIL` below 0.75 | 0.639 | **HIT** |
| 5 | *(registered in advance as independent of the band)* no kernel reaches 50 tok/s at this shape on this machine — 265.58 GB/s is ≥ 6× any reading this box can produce | **265.64 GB/s = 7.32×** | **HIT** |

**The miss is narrow and in one direction: I underestimated how far E13's layout had already gone.**
The predicted interval's top edge is 0.90 and the cell read 0.909 — outside by 0.009 — and §3.1's
post-hoc dispersion straddles the boundary it crossed. The reasoning behind the prediction was
sound and the conclusion it was attached to (`--lutblk` "moved the engine most of the way to the
wall but not onto it") is wrong only in the last four words.

## 8. What E30 could not claim, and what it leaves owed

**Could not claim** (brief §7, unchanged): nothing about quality — no BPB, no teacher-forced count,
nothing exported; nothing about other machines; nothing about the KV cache; nothing about
`--lutblk`'s quality cost, which E14 measured as `CHEAP-BUT-NOT-NEUTRAL` at 1.5 B and **has still
never been measured at T10's shape** (E28 §8 item 3 still stands, and is now the *only* engine-side
item that can still change a verdict).

**Owed forward:**

1. **`--lutblk` end-to-end parity vs the packed default at T10.** E30 makes this more urgent, not
   less: the entire engine-side gain of the last two probes rides on a kernel whose quality cost at
   the goal's shape is unmeasured.
2. **The ceiling at long context**, with KV traffic in the accounting. §5(ii) says it moves the wall
   closer; nobody has measured how much.
3. **The per-matrix `--lut` guard**, demoted per §6.1 but still the cheapest remaining engine work.
4. **The resident half of §5 has never been built.** The 454 GB/s figure is a streaming sum, not a
   matvec over a resident keystone. Before any architecture is priced against it, that number needs
   to be re-measured in the shape it will actually be used.
