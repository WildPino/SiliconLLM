# E59 — the fastest kernel this programme has is LOSSY and has never met a trained model. What does it cost?

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

---

## 1. The question, and why it is the one E58 leaves

E58 addendum A withdrew my own headline: the ×1.11–1.17 it measured is the **packed** kernel's
remainder, and the same binary ships **`--lutblk`**, which E13 measured **end-to-end** at
**×1.217** (0.5 B, `--bench 300`) and **×1.358** (Coder-7B, `--bench 100`). E58 also measured that
**`Rem` is 3.7% / 2.0% of the token** — there is no glue cost hiding a multiple — so **a faster
weight kernel is the only engine-side lever that exists.**

That lever has a price nobody has paid yet. **The LUT path quantises activations to int8.** E13 §6
says it plainly: *"`--lutblk` is not shippable on the strength of this… That is a quality cost on
the donor and it has never been carried through to BPB or greedy parity end-to-end."* §8 item 2
has owed that measurement since it was written. **E59 takes it, on the trained artifacts.**

## 2. Checked, not assumed — what the record already holds

Every source below was **opened and read**; what it *said* is recorded, per E57 C.7's amended rule
and E58 §8's scope clause.

* **`probes/E13_BLOCKED_TILE_MAJOR.md`** — verdict `LAYOUT-CONFIRMED`. Kernel sweep (`n_in` 3584,
  5 reps, medians): `lutblk` **87.84 / 97.87 / 103.05 / 101.22 / 73.11 / 67.51 / 71.82** G-w/s at
  4 / 8 / 12 / 24 / 48 / 512 / 2048 MB against packed **51.40 … 59.02** — `blk ÷ packed` **1.19 to
  2.00**. Engine end-to-end: 0.5 B **75.03 → 91.31** (×1.217), Coder-7B **6.78 → 9.21** (×1.358).
  `G-M0`: `--lut` and `--lutblk` produce **byte-identical** 311 MB logit dumps (sha256
  `7a9b3e04…`) — *the two LUT layouts agree with each other and with neither packed path.*
  §5 also records a discarded 3-rep pass with a **180% spread**, and warns three reps are not
  enough at this shape. §7 states today's honest speed as *"6.79 tok/s exact, with a 1.358× lever
  available on a lossy path whose cost is the next thing that has to be measured."*
* **`probes/E14_ACTIVATION_INT8_COST.md`** — verdict **`CHEAP-BUT-NOT-NEUTRAL`**, run on the E13
  `--lutblk` build. dBPB from fp32 activations **−0.016961** whole-vector and **−0.005991** at
  group-32 (both *better*), while **greedy agreement against fp32 activations is 45.6% and 64.4%**.
  E14 §0: *"`ACTIVATION-CHEAP` is true of the band and false of the model, and E14's registered
  instrument set contains no gate that could have told the difference."* **This is where E14 §3's
  SCORE/RANK law comes from, and E59 inherits it.**
* **`probes/E11_LUT_KERNEL_CEILING.md`** — verdict `NO-LIFT` on the shipped `--lut` layout
  (0.391 at the 512 MB verdict cell), **not** on the LUT arithmetic; E13 explains rather than
  overturns it. E11 measured the activation cost as **1.40e-01** relative L2 whole-vector and
  **3.10e-02** at `--lut-group 32`.
* **`briefs/BRIEF_E57_…` addendum A and `probes/E58_…`** — the six trained arms' packed rates and
  greedy counts, and the organ tables E59's speed numbers sit beside.
* **The engine source** — `donor_engine.c:1552`: *"`--lut` requires a `--quant packed` model"*, so
  **the fp32 arms cannot use this kernel at all.** `:1531`: `--lut-group` must be even. `:190`:
  `--lutblk` is default OFF *"so every `--lut` number already published reproduces byte for byte."*
* **Capability smoke, already run, NOT a measurement:** `donor_engine_e53.exe --lutblk` on
  `qwen25-05b_tqh.bin` builds a 235.5 MB blocked tile-major replica in 0.13 s and reports
  `activations int8, AQ=63`.

## 3. Design

**Two artifacts** — `qwen25-05b_tqh.bin` and `qwen25-15b_tqh.bin`, the trained ternary arms E58
timed. **Three kernel arms**, one binary, `donor_engine_e53.exe`, `--threads 6`:

| arm | flags |
|---|---|
| `PACKED` (control) | *(none)* |
| `LUTBLK` | `--lutblk` |
| `LUTBLK32` | `--lutblk --lut-group 32` |

**Speed**: `--bench 160`, **k = 9 reps**, a discarded warm-up per arm, **arms interleaved inside
one sweep** (E13 §6: absolutes carry ±5% between sweeps, ratios do not), foreign occupancy /
clock / timestamp per cell via `e44_interval.Split`, quantiles and a bootstrap interval reported
as data. No `G-E55a2` (malformed, E57 A.5).

**Fidelity**: 5 prompts × 32 new tokens, greedy, scored by `E51.score_against_ref` — the same
instrument and the same prompts as E57, so the numbers drop straight into E57's table.

## 4. Gates, registered before the run

### `G-E59a` — the planted control, which must both FIRE and DISCRIMINATE

> **(i)** `--lut` and `--lutblk` must produce **identical token sequences** on the same artifact —
> E13's `G-M0` says they are bit-identical, so a scorer that cannot reproduce that is broken.
> **(ii)** `--lutblk` and `PACKED` must **differ** on at least one token — they quantise
> activations differently by construction, so an instrument reporting them identical is measuring
> nothing. **Both halves must hold or no number below is read.**

This is the `feedback_instrument_must_not_measure_itself` shape: a comparison that can only ever
return "same" passes every artifact.

### `G-E59b` — fidelity, measured against the RIGHT reference

> Report, per artifact and per arm: **(a)** greedy agreement **`arm` vs `PACKED` on the same
> artifact** — 160 positions, the paired quantity that isolates what the *kernel* changes; and
> **(b)** greedy agreement **vs E6's stored HuggingFace references**, with the floor caveat below.

**(b) is reported and may not carry the verdict.** `05b_tqh` and `15b_tqh` already read **3/160**
and **10/160** on the packed path (E57, replicating E17). **A counter already on its floor cannot
show further damage** — `feedback_gate_is_not_a_progress_meter`, and the exact trap that closed H0
prematurely. The load-bearing number is **(a)**.

### `G-E59c` — speed, as a ratio with a dispersion

> Report each arm's median tok/s with a bootstrap 95% interval and interquartile width, and the
> **ratio `LUTBLK ÷ PACKED` computed within the interleaved sweep**, never across sessions.
> A ratio is quoted only if the two arms' intervals do not overlap, or the overlap is stated.

### `G-E59d` — the joint law, inherited from E13 §7

> **A speed ratio from this experiment may never be quoted without the fidelity number beside it.**
> E13 §7 already ruled that `6.79 → 9.21` "is not the same kind of number" as the exact-path
> numbers above it. Nothing is promoted, no flag becomes a default, `donor_engine.c` is untouched.

## 5. Predictions — on the record, before the run

| | prediction |
|---|---|
| `G-E59a` (i) `lut ≡ lutblk` | **identical**, 160/160, both artifacts |
| `G-E59a` (ii) `lutblk ≠ packed` | **differs**, and differs early (within the first 4 tokens) |
| speed ratio, `05b_tqh` | **×1.20 – 1.40** |
| speed ratio, `15b_tqh` | **×1.25 – 1.50** — bigger organs sit further into the region where E13's sweep gives 1.5–2.0× |
| **fidelity (a), whole-vector** | **35 – 60%** agreement with packed — E14's 45.6% is the anchor |
| **fidelity (a), group-32** | **55 – 80%** — E14's 64.4% is the anchor |
| fidelity (b) vs HF | **unchanged or worse than packed's 3/160 and 10/160**, and uninformative either way |

**The claim I am registering, and it is the one that decides what E58's correction means:**
**the lever is not free, and on a HEALED model it would be spent undoing the healing.** If
group-32 agreement comes back **above 90%** I am wrong, the kernel is close to free, and the
engine-side ceiling for a faithful arm rises to E13's ×1.22–1.36. If it comes back near E14's
45–64%, then **the ×1.217/×1.358 cannot be spent on a model whose whole value is that it is
right**, and E58's packed ceiling is the honest one after all — for the wrong reason, which is
still worth knowing.

## 6. What E59 cannot claim

* **Nothing about a healed model.** These artifacts are broken *before* the kernel touches them.
  E59 measures what the kernel changes **relative to the same artifact on the packed path**; the
  transfer to a healed artifact is an inference, and will be labelled one.
* **Nothing about the fp32 arms.** `--lut` requires a packed model
  (`donor_engine.c:1552`), so the only arms that are currently *faithful* cannot use this kernel
  at any price. That is a structural fact, not a result.
* **Nothing about BPB.** E14 already measured the BPB half and found it *improves* while the
  argmax moves. Re-measuring BPB here would re-derive E14's trap, not escape it.
* **No 10 B claim.** Two artifacts, one context (`--bench 160`), one box.
