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
