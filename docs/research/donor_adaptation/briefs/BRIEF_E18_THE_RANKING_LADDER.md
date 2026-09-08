# BRIEF E18 — the floor under every agreement number, and how much ternarization a model that still ranks can survive

**Status: PRE-REGISTERED.** Part A is already computed and is reported below as a measured *input*,
because the bands in §5 are derived from it; it is a pure computation over reference data with no
treatment and no arm. Part B is pushed before any arm ran. Directions are called in §6.

---

## 1. The questions

**A. What is the floor?** Every greedy-agreement number in E6, E14, E16 and E17 is quoted against
an implicit floor of zero. Zero is the floor for a *uniform* guesser. It is not the floor for a
degenerate model that emits high-frequency tokens, and the arms in question emit exactly those.
E17 §7 recorded this as the cheapest open item in the programme.

**B. How much ternarization can a model that still ranks survive?** E17 established that no
converted donor ranks, across three scales, two rules, two folds and both head settings. But every
one of those arms converts **everything** — FFN, attention and (usually) head together. T2b measured
an organ ladder in BPB and **five of its seven rungs have never been generated with**.

## 2. Part A — the floor, computed

`e18_agreement_floor.py`, from `results/e6/ref.json` alone: no engine, no weights, no arm. The
floor is the score of the best possible **constant** predictor — the strongest zero-information
model — over the same 160 positions every agreement in this programme is scored against.

| reference | positions | distinct tokens | **best constant predictor** | token |
|---|---|---|---|---|
| Qwen2.5-0.5B | 160 | 91 | **11/160 = 6.88%** | `'\n'` |
| Qwen2.5-1.5B | 160 | 82 | **12/160 = 7.50%** | `'\n'` |

Against which every ternary arm ever measured:

| arm | scored | floor | excess | what it actually emits |
|---|---|---|---|---|
| 0.5 B fp32 (`A1`) | `160/160` | 11 | **+149** | coherent text |
| 0.5 B `TQH` (`A2`) | `3/160` | 11 | **−8** | `','` 45 times of 160 |
| 1.5 B `TQH` (`A3`) | `10/160` | 12 | **−2** | `'\n'` **77 times of 160** |
| 1.5 B `TQ` (E17 `H1`) | `12/160` | 12 | **0** | — |
| 7 B `B1`/`B2`/`B3` | `0/160` | — | below | — |

**E17's best ternary arm scores exactly the floor.** Not one ternary artifact this programme has
built carries ranking information; the `0`–`12/160` spread E17 tabulated is dispersion around a
degenerate baseline. E17 §4's "best is 12/160" must be read as **"at the floor"**, and this brief
records that as a correction to how that table should be quoted, not to the numbers in it.

## 3. Part B — the ladder

Seven arms at 1.5 B, `R3`, on the same heldout slice and the same five frozen prompts. **Run in
PyTorch, not the engine**, because the `QWENDON1` format carries a single global `quant` field:
"head fp32" works only by keeping the head *tied* to the fp32 embedding, so a mixed-precision organ
arm is not expressible in the format without changing it. The question is about the **model**, not
the runtime, and E1 §2.1 measured engine against PyTorch at `+1.5347e-05` BPB on an identical arm.

**Nothing is re-derived**: `capture()` and `apply_arm()` are imported from `t2b_organs.py`, the same
definitions that produced the BPB column below; `PROMPTS`/`N_NEW` from `e6_generate`; the reference
from `results/e6/ref.json`.

| arm | organs converted | BPB (T2b) | vs chance `4.069819` | greedy |
|---|---|---|---|---|
| `base` | none | `0.767595` | `−3.302224` | **harness control**, must reproduce `ref.json` |
| `I` | identity through the same code path | `0.767595` | `−3.302224` | **instrument control**, must be bit-exact |
| `H` | `lm_head` only | `1.106584` | `−2.963235` | **empty** |
| `A` | `q,k,v,o` only | `1.903569` | `−2.166251` | **empty** |
| `F` | `gate,up,down` only | `2.476967` | `−1.592852` | **empty** |
| `FA` | `F+A` (= E1 `TQ`) | `3.484251` | `−0.585568` | `12/160` on the engine |
| `FAH` | `FA+H` (= `TQH`) | `3.475706` | `−0.594113` | `10/160` on the engine |

`FA` and `FAH` are **cross-instrument replications**: if PyTorch reproduces the engine's `12/160`
and `10/160`, the harness is validated by two independent known values before any empty rung is
read. That is the whole reason they are in the run.

## 4. Instruments

- Greedy argmax, 32 new tokens, the five frozen `e6_generate` prompts, denominator **160**, E6's
  counting convention (every position counted, including after a divergence).
- Reference: `results/e6/ref.json`, `Qwen/Qwen2.5-1.5B`, PyTorch fp32 eager — the same file the
  floor in §2 was computed from and the same one E6/E17 scored against.
- BPB is **not** re-measured. T2b's column is quoted.

## 5. Gates and bands — registered before part B ran

**Floor `12/160`** (§2, measured). **Ceiling `160/160`** (three independent readings: E6 `A1`,
E17 `H0c`, E7's 7 B fp32).

**The margin that may not be interpreted, derived rather than chosen.** E17 measured that removing
the ternary head — a change that rewrites **49%** of the output at 1.5 B — moved agreement by **2
tokens**, and at 0.5 B by **0**. So near the floor, a structural change of enormous magnitude moves
this metric by ≤ 2 tokens. **Margins of ≤ 2 tokens are therefore not interpreted**, and that number
comes from a measurement rather than a preference.

- **`G-L0`** `base` reproduces `ref.json` at **160/160**, or **E18-B is VOID** — the harness must
  fire on the known-positive.
- **`G-L1`** `I` is **token-identical** to `base`, or **VOID** — the substitution code path must be
  an identity when it substitutes nothing. (T2b gated this in BPB at `+0.000e+00`; it has never
  been gated in generation.)
- **`G-L2`** `FA` = **12/160** and `FAH` = **10/160**, ± the 2-token uninterpretable margin. A miss
  here is **not** a void: it is a finding that the engine and PyTorch diverge in *ranking* even
  though E1 matched them in *score*, which would be a result in its own right and would be reported
  as one.
- **`G-L3`** each of `H`, `A`, `F` gets one of three labels:
  - `matched ≤ 14` (floor + the uninterpretable margin) → **`AT-FLOOR`**
  - `15 ≤ matched ≤ 79` → **`ABOVE-FLOOR-DOES-NOT-RANK`**
  - `matched ≥ 80` (half the positions) → **`RANKS`**
- **`G-L4`** descriptive, no label: the ladder as (BPB vs chance) against (agreement), so the
  question "where does ranking collapse" is answered with a curve rather than a threshold.

## 6. Predictions — directions called

1. **`G-L0` and `G-L1` fire.** High confidence.
2. **`G-L2` replicates**: `FA` at `12/160`, `FAH` at `10/160`.
3. **`H` (`−2.963`) → `RANKS`.** Ternarizing only the readout, with the whole body exact. **This is
   the least confident call in this brief** and it cuts against E17, which found a ternary head
   rewrites half the output — but there the body was ternary too, and here the representation
   reaching the head is exact.
4. **`A` (`−2.166`) → `RANKS`**, and below `H`.
5. **`F` (`−1.593`) → `ABOVE-FLOOR-DOES-NOT-RANK`.** The key call. FFN is 88% of the body's
   parameters at this width, and `F` is the rung nearest the collapse.
6. **The ladder is NOT linear in BPB.** Between `F` (`−1.593`, predicted above floor) and `FA`
   (`−0.586`, measured at the floor) lies `0.89` BPB and the entire collapse. Agreement will fall
   much faster than BPB across that step. **If instead `H`, `A` and `F` all come back at the floor,
   the collapse is at the very first rung and the readable conclusion is far stronger — that
   ternarizing any single organ of this donor destroys ranking.**

## 7. What E18 cannot claim

- **Part A does not rescue E14's `45.6%`.** The floor bounds a *degenerate* model from below; it
  says nothing about what a genuinely different but equally good model produces. **The intermediate
  ranking band (E14 §5 item 3) is still owed and is still not supplied.**
- **`RANKS` at 80/160 is a registered convention, not a discovered threshold.** Nothing measured
  says half the positions is where usefulness begins.
- **No speed number moves and none is taken.** `6.79 tok/s` exact stands; §19.3 unchanged. Part B
  runs in PyTorch and is not a runtime measurement of anything.
- **A PyTorch result is not an engine result** (Phase 60's law). If a rung ranks, an engine arm for
  it does not exist and could not be built without changing the `QWENDON1` format, and E18 does not
  claim otherwise — it says what the *model* can survive, which is what determines whether building
  that format is worth it.
- **One donor, one scale, one rule, one corpus, 160 positions, 5 prompts.** The ladder is 1.5 B and
  `R3` only.
- **`R1` and `R2` are deliberately NOT run here**, and this is a change of plan from E17 §9 item 3,
  recorded with its reason: T2 measured them at FFN-only as `3.851979` and `3.390467` against R3's
  `2.476967`, i.e. **both are worse than R3 at the same coverage**, and R3 at full coverage already
  sits at the floor. A full-coverage `R1`/`R2` arm is therefore predicted to land at or below the
  floor and would complete a table rather than open a route. They stay owed as completeness, not as
  a candidate.

## 8. Cost and protocol

T2b's whole seven-arm BPB run took `1,646 s`. E18-B replaces the scoring pass with 160 greedy
forwards per arm and reuses the same calibration capture (`16,384` calib tokens). Estimated 1.5–2.5 h,
single job, no concurrent exporter or trainer. Nothing is timed and no rate is quotable from it.

## 9. Checked, not assumed

- The `QWENDON1` header carries **one** global `quant` field, and the head path writes through the
  same `W()`; "head fp32" is the *tied* case (`tied: 1` on E1's `tq`, `tied: 0` on `tqh`). Read in
  `qwen_export.py`, which is why part B is in PyTorch — not assumed from the flag's name.
- `t2b_organs.py` exposes `capture()` and `apply_arm()` and defines arms `base/I/F/A/H/FA/FAH`;
  E18 imports them rather than re-deriving the substitution.
- The floor is computed from the reference continuations only, so it cannot be contaminated by the
  arms it judges.
- T2b's `F` reproduces T2's `R3` at `2.476967`, and its `FA`/`FAH` reproduce E1's `TQ`/`TQH` engine
  arms to 6 decimals — so the BPB column and the engine artifacts are the same ladder, checked
  before being placed in one table.
