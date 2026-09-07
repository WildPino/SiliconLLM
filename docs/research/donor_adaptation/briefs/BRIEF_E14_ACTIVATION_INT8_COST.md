# E14 — what does int8 activation quantization cost in BPB?

**Pre-registration.** Written and pushed before any BPB is read. Bands in §4 are fixed here and are
read off mechanically afterwards.

Donor: `Qwen2.5-0.5B` packed (`qwen25-05b_tqh.bin`), standing eval slice
`ids_qwen25-05b_tqh.bin` (12,288 tokens). Machine may be loaded — **BPB is deterministic**, and
this experiment contains no timing.

---

## 1. Why this exists, and why it is now the blocking question

E13 made the LUT path **1.358× faster than the packed default at donor scale** and **2.00× faster
per weight inside L3**. That is the only speed lever this programme has found since E9.

**It cannot be spent.** The LUT path quantizes activations to int8 (`AQ 63`), and the only thing
ever measured about that is the **round-trip error on `x` itself** — E11's `--lut-diag`:
**1.40e-01** relative L2 whole-vector, **3.10e-02** at `--lut-group 32`. Both are input-side
arithmetic. **Nobody has ever carried either through to BPB or to a generated token on a donor.**
`BRIEF_T2` §5 recorded it as untested; E11 §7 and E13 §8 both list it as owed.

So the state is: a 1.358× lever whose price is unknown, sitting next to a goal that is 5.4× away.
**E14 prices it.** Until it does, `SPEED_LEDGER` quotes 6.79 tok/s exact and the packed path stays
the default — which is exactly what E13 §7 committed to.

## 2. What is held fixed

**The weights are identical in every arm.** `build_tm` is a pure transpose of the same packed
bytes (E13 made it a permutation of a permutation), so `--lut`, `--lutblk` and the packed default
hold **bit-identical ternary weights** and differ **only** in how activations are represented.
There is no weight-side confound to separate out, by construction.

The eval slice, the model file, the thread count and the engine binary are the same across arms.

## 3. Arms

| arm | flags | what it is |
|---|---|---|
| **A0** | *(default)* | fp32 activations, ternary weights. **The baseline**, and the engine's shipped path. |
| **A1** | `--lut` | int8 activations, **one scale per vector** — engine.c's convention. |
| **A2** | `--lut --lut-group 32` | int8 activations, **one scale per 32 channels**. |
| **A3** | `--lutblk` | E13's blocked layout. **Must be numerically identical to A1.** |

## 4. Gates and bands, fixed now

| gate | test | band |
|---|---|---|
| **G-N0** — instrument control | `BPB(A3) − BPB(A1)` | **must be exactly `0.000000000`.** E13's G-M0 proved these two arms emit **byte-identical logits** (`7a9b3e04…f48e44`). A non-zero difference means the BPB harness is not deterministic and **every number below is void**. |
| **G-N1** — planted control | `BPB(A2) < BPB(A1)` | **must FIRE.** E11 measured the input-side error at **1.40e-01** whole-vector against **3.10e-02** at G=32 — a **4.5×** difference. **An instrument that cannot see that ordering cannot price either arm**, and its nulls do not count. |
| **G-N2** — the verdict | `ΔBPB = BPB(A1) − BPB(A0)`, and separately for A2 | **≤0.010 `ACTIVATION-CHEAP`** · 0.010–0.020 `ACTIVATION-MARGINAL` · **≥0.020 `ACTIVATION-COSTLY`** |
| **G-N3** — greedy | `--generate`, 160 tokens, the five frozen prompts, A1/A2 vs A0 | **descriptive, not pass/fail** — see §6. |

**G-N2's boundaries are precedent, not invention.**

- **0.010 = 2 σ_seed.** `σ_seed ≈ 0.005` is R1's sealed constant, the number this programme judges
  every delta against. Below 2σ the arms are not distinguishable from seed noise.
- **0.020 is the bottom of the range this programme has already REJECTED.** Phase 61 ternarized the
  SSM projections, measured **+0.018–0.022 BPB**, returned `DOUBLE-FAIL`, and left the projections
  fp32. **A cost at or above 0.020 is therefore one this programme has ruled too expensive before,
  and E14 does not get to re-litigate that on the strength of a speed number.**

## 5. Prediction, on the record

**A1 `ACTIVATION-COSTLY` (≥0.020); A2 materially cheaper than A1, most likely
`ACTIVATION-MARGINAL`.**

**Confidence is low on the magnitude and I am saying so before the run.** The directional part is
well founded — E11 measured a 4.5× gap in input-side error between the two, and the donor's
activations carry a crest factor of **8.3 average / 69.6 max**, which is precisely the condition a
whole-vector amax grid handles worst (it spends its 63 levels covering outliers). **The magnitude
part is not founded**: I have no calibration from activation relative-L2 to BPB, and that missing
calibration *is the thing this experiment measures*. Recording the distinction because three of
this programme's five predictions missed, and the two that landed (E9, E13) were both derived from
a measured quantity over a structural factor — **this one is not, and should be trusted less.**

## 6. Why greedy parity is descriptive here

E6 established that the ternary build **already** does not reproduce PyTorch's greedy trajectory —
it emits tokens, not language, at 1.9% and 6.2% agreement. So a greedy comparison against *PyTorch*
would be measuring the ternarization, not the activations. **G-N3 therefore compares A1/A2 against
A0** — the same ternary weights with fp32 activations — and reports the agreement as a
characterisation. **It is not a gate, and no verdict is read off it**, because there is no
pre-registered band that could be justified from anything measured.

## 7. What this cannot claim

- **It is measured at 0.5 B.** E12 is running precisely because this programme has been burned by
  quality numbers taken at one shape. **E14's number is a 0.5 B number and inherits that
  limitation**; a Coder-7B repeat is owed if the verdict is anything but `ACTIVATION-COSTLY`.
- **It prices the activations, not the LUT path.** A cheap verdict does **not** make `--lutblk`
  shippable on its own — Phase 60's law still applies, and adoption would need the end-to-end
  parity gate the packed default already carries.
- **It says nothing about speed.** No timing is taken and none may be quoted from this run.
- Whatever it returns, **it does not move the goal**: the 1.358× lever, if free, takes Coder-7B to
  9.21 tok/s and **5.4× short of 50**. The gap remains a property of the model (§19.3, E10, E13).

## 8. Sequencing

**E12 has the machine.** E14 is deterministic and would be valid under contention, but E12 is the
decisive experiment and a six-thread competitor would roughly double its wall-clock. **E14 runs
after E12 finishes**, and this brief is pushed now so the pre-registration precedes the run rather
than the convenience.

---

# §10 — AMENDMENT, written after reading A0 and before any treatment arm

**A0 is the baseline, not a treatment.** What it says is a property of the setup, so reading it does
not consume the pre-registration — but it invalidates where §4's verdict was pointed, and the
correction is recorded here, pushed, before `--lut`, `--lut-group 32` or `--lutblk` is run.

## 10.1 The baseline does not predict

`donor_engine --weights qwen25-05b_tqh.bin --bpb ids_qwen25-05b_tqh.bin --threads 6`, 11m48s:

| | value |
|---|---|
| A0 | **13.5645220434 nats/token** = 19.569470 bits/token = **4.629292115 BPB** |
| chance, `log2(151936) / 4.227313` | **4.071878 BPB** (`ln V` = 11.931215 nats/token) |
| **A0 − chance** | **+0.557414 BPB, +1.633307 nats/token** |

The slice is 12,288 tokens, 12,287 predicted, **51,941 scored bytes, 4.227313 bytes/token** —
computed by decoding the slice, and **identical for the 0.5 B and 1.5 B ids files**, so one chance
line serves both cells.

**Cross-checked two ways.** E12's PyTorch simulation of the same conversion reads 13.448727
nats/token on a different slice — **0.86% from the engine's A0**, which is the first end-to-end
corroboration of E12's finding on the real runtime. And E1 §2.1's published `TQH` number, 4.531234
BPB on the shared density slice, sits **+0.461 above that slice's chance line**. **Three
independent measurements agree that the 0.5 B ternary export does not predict.**

## 10.2 What that does to §4's verdict gate

**G-N2 cannot be read at 0.5 B.** `ΔBPB` between A1/A2 and A0 would be a difference between models
that are all worse than guessing, and E12 §2 is the ruling on exactly that: above the chance line
BPB measures how confidently wrong a model is, not how damaged it is. **G-N2's boundaries make it
worse, not better**: 0.010 is 2 σ_seed and 0.020 is the bottom of what Phase 61 rejected, and
**both were drawn from regimes where the model predicts**. A `ACTIVATION-CHEAP` reading here would
mean "the activation quantization changes little about a model that already knows nothing", which
is not the question and cannot license spending E13's 1.358× lever.

## 10.3 The change, and its cost to the pre-registration

**The verdict cell moves to the 1.5 B export**, `qwen25-15b_tqh.bin` with
`ids_qwen25-15b_tqh.bin` — the same rule, the same pipeline, the same slice bytes, and a model E1
measures at **3.475707 BPB, 0.594 BELOW chance**. It is the smallest standing artifact that
predicts.

**0.5 B is retained, demoted to instrument-only.** G-N0 and G-N1 are properties of the harness and
the arms, not of where the model sits, so they are read at both cells:

| gate | cell | unchanged? |
|---|---|---|
| **G-N0** `BPB(A3) − BPB(A1)` exactly 0 | both | yes — pure instrument, and a direct check of E13's bit-identity claim through a second mode |
| **G-N1** `BPB(A2) < BPB(A1)` must fire | both | yes — the harness must resolve a 4.5× input-error difference |
| **G-N2** the verdict, bands `0.010` / `0.020` | **1.5 B only** | **bands unchanged; the cell moved** |
| **G-N3** greedy, descriptive | 1.5 B | unchanged |

**This is a post-registration change and it is worth being explicit about the cost.** §4's bands
were fixed before any run and are **not** touched; what moved is which cell they are read at, and
the reason is a baseline property I should have checked when the brief was written. **§5's
prediction was written for 0.5 B and is not rewritten** — it is carried to the 1.5 B cell as-is, and
if it lands it lands at a shape it was not written for, which is weaker evidence than E9's or
E13's and will be reported that way.

**What this does not rescue.** Even a cheap verdict at 1.5 B does not make `--lutblk` shippable
(§7 still stands), and it does not move the goal: the lever is 1.358× and Coder-7B would reach
9.21 tok/s, still 5.4× short of 50.
