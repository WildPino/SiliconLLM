# T3 — Rotate the basis before ternarizing

**Pre-registration: `briefs/BRIEF_T3_ROTATION.md`, pushed at `7cdeca8` before the run.**
**Runner: `benchmarks/donor_adaptation/ternary/t3_rotation.py` @ `045d4d8`. Log: `ternary/t3_run.log`.**
**Results: `density/results/t3_rotation.json` + `density/results/t3_arms/*.npy` (10 arms × 24 sequences).**
**Run: 2026-09-05, 6 threads, CPU only, 2656 s. Donor Qwen2.5-1.5B rev `8faed761…`.**

---

## 0. Verdict

| | |
|---|---|
| **Mechanical label, brief §4 verbatim** | **`VOID`** |
| **Why** | arm Q missed the brief's replication constant by 1.007 — **because the constant is wrong in the brief**, not because the harness is |
| **Label once the constant is corrected** | **`NULL`** — *kurtosis does not predict ternarization damage on this donor; this line of attack closes* |
| **What actually happened** | rotation did not fail to help. It **hurt**, by `+0.634 ± 0.039` (dense orth) and `+1.138 ± 0.072` (Hadamard) against its own fold-matched control — 127 and 227 σ_seed |
| **The one keeper** | folding the RMSNorm gains before quantizing is worth **−0.220 ± 0.053 BPB free** (44 σ_seed, 8.1% of the damage), no format change, no kernel change |

The `VOID` stands as the label. It is reported first, before any result, because that is what
the brief says to do. Everything below §3 is therefore **read under a VOID**, and §1.3 argues why
it may be read at all. The direction of the finding is negative in every arm, so nothing here
can be a favourable result laundered through a broken gate — but the reader should check that
claim rather than take it.

---

## 1. Gates, before any result

### 1.1 Arm X — the control the brief said it lives or dies by

Rotate and fold, quantize **nothing**. A change of basis is exact, so these must reproduce base.

| arm | basis | folded | rotated | BPB | Δ vs base | `‖RᵀR − I‖∞` |
|---|---|---|---|---|---|---|
| base | — | 0 | 0 | 0.767594958417 | — | — |
| **XN** | none (fold only) | 57 | 0 | 0.767594965208 | `+6.79e-09` | — |
| **XP** | permutation | 57 | 198 | 0.767594965208 | `+6.79e-09` | `0.0` |
| **XH** | `hadamard_block` | 57 | 198 | 0.767594966056 | `+7.64e-09` | `1.11e-16` |
| **XO** | `dense_random_orth` | 57 | 198 | 0.767594955022 | `−3.40e-09` | `3.55e-15` |

Tolerance `1e-4`. Worst arm is **`7.6e-09`, four orders of magnitude inside it**, and that
residue is fp32 accumulation order, not algebra: the orthogonality deviations above are float64
round-off on bases built natively in float64. **PASS.** The untie of `lm_head` from the
embedding, the 57 norm folds and the 198 rotated tensors are all exact.

### 1.2 Arm P — the planted null

`P − N = +0.000137 ± 0.000115`, ci95 `[−0.000095, +0.000353]`, tolerance `1e-3`. **PASS**
(0.027 σ_seed).

Runner departure (a), pre-registered in the runner's docstring at `d604d07`: a residual-stream
**permutation** permutes a reader's columns and a writer's rows, and R3 searches one scale per
output row over `(weight, act_rms)` pairs — both invariant. P is therefore an **analytic** null
in the weights, not merely a weak one, and the brief's `PERMUTATION-ARTEFACT` branch is
unreachable by construction. It is not fp-exact only because calibration sums squares over
permuted channels, so `act_rms` differs in its last bits and a near-tie in the scale search can
flip a code. The measured `1.4e-04` against an unquantized permuted arm (XP) exact to `6.8e-09`
is exactly that signature.

### 1.3 Arm Q — the replication gate, and the reason the label is `VOID`

| | |
|---|---|
| measured `Δ(Q)` | **`+2.716656`** |
| brief §4 constant | `+1.709372`, tolerance `0.01` |
| gate | **FAIL** → `VOID` |

**The brief contradicts itself, and the runner obeyed both halves.** §3 fixes the organ set as
"FFN + attention organs" — seven tensors per layer, 196 in total. `+1.709372` is T2's **FFN-only**
Δ (`t2_rules.json`, three tensors per layer, 84 in total). The two cannot both be arm Q.

The runner took the organ set from §3 and the constant from §4's table, which is the only
reading that does not silently rewrite a pre-registration. The gate then fired on the mismatch.

**The harness is not broken, and this is checkable rather than asserted.** T2b already ran the
identical arm — R3, no fold, no rotation, FFN + attention, same slice, same calibration — as its
arm `FA`, from a different runner (`t2b_organs.py`), on a different day:

| | T3 arm Q | T2b arm FA |
|---|---|---|
| BPB | `3.484251267084465` | `3.484251267084465` |
| Δ vs base | `+2.716656` | `+2.716656` |
| mean zero fraction | `0.4714445757622622` | `0.4714445757622622` |
| tensors substituted | 196 | 196 |

**Bit-identical to the last stored digit, across two independent runners.** That is a strictly
stronger replication than the one the brief asked for. The gate that should have been written is
`Δ(Q) = +2.716656 ± 0.01` — and it passes with a residue of zero.

**What is *not* being done here:** the label is not being changed. `VOID` is what the
pre-registered rule returns and `VOID` is what is recorded. §2 computes what the rule would have
returned with the corrected constant, and says so in those words.

---

## 2. The label the corrected gate reaches

Brief §4, verbatim, with `q = Δ(Q) = 2.716656` and `best = min(Δ(H), Δ(O)) = 3.131144` (arm O):

| outcome | condition | evaluates to |
|---|---|---|
| ROTATION-WINS | `best ≤ q − 0.30` and `Δ(P) ≥ best + 0.20` | `3.131 ≤ 2.417` → **false** |
| ROTATION-MARGINAL | `q − 0.30 < best ≤ q − 0.05` | `3.131 ≤ 2.667` → **false** |
| PERMUTATION-ARTEFACT | `best ≤ q − 0.30` but `Δ(P) < best + 0.20` | first clause false |
| **NULL** | `best > q − 0.05` | `3.131 > 2.667` → **TRUE** |

> **NULL** — "kurtosis does not predict ternarization damage on this donor. D2's table is then
> evidence about sparsity only, and this line of attack closes."

The P arm never has to arbitrate: nothing improved, so there is no gain whose mechanism could be
in dispute.

`NULL`'s condition does not distinguish "no effect" from "much worse". Here it is much worse, and
the size is in §4.

---

## 3. All ten arms

Shared eval slice `heldout` 24×512 seed 1234, `ids_sha256 = a1a48dc9fc5a6dc1…`, **51,870 scored
bytes**. Calibration `calib` 32×512 seed 42424, `ids_sha256 = c5509846cdc3aa44…`, T = 16,384
tokens, disjoint corpus half — **one capture per basis**, because R3 weights by the activation
RMS of its actual input and the rotated stream has different per-channel RMS. Paired
sequence-bootstrap, 2000 resamples, seed 7, byte-weighted. σ_seed = 0.005.

| arm | fold | basis | quant | BPB | Δ vs base | paired SE | ci95 | σ_seed | zero frac | s |
|---|---|---|---|---|---|---|---|---|---|---|
| base | — | — | — | 0.767594958 | — | — | — | — | — | 115 |
| XN | ✓ | — | — | 0.767594965 | `+0.000000` | `1.4e-08` | — | 0.0 | — | 113 |
| XP | ✓ | perm | — | 0.767594965 | `+0.000000` | `1.7e-08` | — | 0.0 | — | 142 |
| XH | ✓ | hadamard | — | 0.767594966 | `+0.000000` | `1.9e-08` | — | 0.0 | — | 140 |
| XO | ✓ | orth | — | 0.767594955 | `−0.000000` | `1.8e-08` | — | 0.0 | — | 142 |
| **Q** | — | — | R3 | 3.484251267 | **`+2.716656`** | 0.129001 | `[+2.478, +2.980]` | 543 | 0.4714 | 386 |
| **N** | ✓ | — | R3 | 3.264250478 | **`+2.496656`** | 0.111607 | `[+2.290, +2.723]` | 499 | 0.5084 | 385 |
| **P** | ✓ | perm | R3 | 3.264387047 | `+2.496792` | 0.111625 | `[+2.290, +2.723]` | 499 | 0.5084 | 415 |
| **H** | ✓ | hadamard | R3 | 4.401868011 | **`+3.634273`** | 0.072082 | `[+3.498, +3.784]` | 727 | 0.4610 | 412 |
| **O** | ✓ | orth | R3 | 3.898739344 | **`+3.131144`** | 0.115508 | `[+2.927, +3.374]` | 626 | 0.4607 | 400 |

Every quantized arm converts the same 196 tensors (28 layers × `gate/up/down/q/k/v/o`). The head
is **not** converted in T3 — that is T2b's question, and one variable at a time.

### 3.1 Paired contrasts

T2 §4 is where a Δ-against-base was shown to be unable to compare two arms, so every comparison
below is a paired between-arm bootstrap on the same 24-sequence index draws.

| contrast | reads as | Δ | paired SE | ci95 | σ_seed |
|---|---|---|---|---|---|
| `H − P` | Hadamard vs planted null | `+1.137481` | 0.072395 | `[+0.998, +1.279]` | 227 |
| `O − P` | dense orth vs planted null | `+0.634352` | 0.038909 | `[+0.560, +0.712]` | 127 |
| **`H − N`** | **Hadamard vs fold-only** | **`+1.137618`** | 0.072398 | `[+0.998, +1.279]` | **227** |
| **`O − N`** | **dense orth vs fold-only** | **`+0.634489`** | 0.038971 | `[+0.560, +0.712]` | **127** |
| `P − N` | planted null | `+0.000137` | 0.000115 | `[−0.0001, +0.0004]` | 0.03 |
| **`N − Q`** | **the fold, alone** | **`−0.220001`** | 0.052861 | `[−0.325, −0.120]` | **−44** |
| `H − Q` | Hadamard vs brief's reference | `+0.917617` | 0.091270 | `[+0.740, +1.096]` | 184 |
| `O − Q` | dense orth vs brief's reference | `+0.414488` | 0.047061 | `[+0.325, +0.507]` | 83 |

`H − P` and `O − P` are the contrasts brief §7 named. They are within `1.4e-04` of `H − N` and
`O − N`, which is the planted null's own residue — as departure (a) predicted they must be.
**`H − N` and `O − N` are the ones to read**, because they hold the fold fixed and vary only the
rotation, and the fold is not free (`N − Q`).

---

## 4. What the numbers say

### 4.1 Rotation loses, in both bases, decisively

`+0.634 ± 0.039` and `+1.138 ± 0.072` against the fold-matched control. These are not marginal
misses of a 0.30 bar in the good direction — they are 2× and 4× that bar in the **bad** one, at
16× their own SE.

### 4.2 Kurtosis is not merely non-predictive — it points two ways at once

D2's table, medians over the same 12 real tensors (`d2_basis.json`):

| basis | median kurtosis | worst | median `frac90` | measured T3 damage vs N |
|---|---|---|---|---|
| identity / permutation | 4.808 | 22.111 | 0.4066 | **0** (that is arm N/P) |
| `hadamard_block` | 3.095 | 4.242 | 0.4422 | `+1.138` |
| `dense_random_orth` | 3.012 | 3.139 | 0.4440 | `+0.634` |

Read down the first comparison: flattening from 4.81 to ~3.0 **costs** BPB. Read between the two
rotations: `dense_random_orth` flattens *more* than `hadamard_block` (3.012 vs 3.095; worst 3.14
vs 4.24) and **costs less**. A single monotone function of kurtosis cannot produce both. Kurtosis
does not order these arms consistently in either direction — it is not a weak predictor here, it
is not a predictor.

### 4.3 The sign is consistent with D2's *own* reading, which the brief inverted

D2 recorded rotation as a negative because it makes the weights **less sparse** (`frac90` 0.4066
→ 0.4422/0.4440). The brief argued that "quantization wants the opposite of sparsity". At ternary
width that is backwards: a ternary code's only free parameter is **where the zero band ends**, so
near-zero mass is what the format represents for free. The runner's measured zero fractions move
with `frac90` and against the damage:

| arm | mean zero fraction | Δ vs N |
|---|---|---|
| N (fold, no rotation) | **0.5084** | — |
| H (Hadamard) | 0.4610 | `+1.138` |
| O (dense orth) | 0.4607 | `+0.634` |
| Q (no fold, no rotation) | 0.4714 | `+0.220` |

Rotation destroyed 4.7 pp of exact zeros and paid for it. **But this explains the sign and not
the magnitude:** H and O have effectively the same zero fraction (0.4610 vs 0.4607, 0.03 pp
apart) and effectively the same kurtosis, yet differ by `+0.503` BPB. Neither variable measured
in D2 ranks the two rotations. **Why block-Hadamard is 1.8× worse than a dense orthogonal at
matched flatness and matched sparsity is open, and nothing here should be built on a guess about
it.**

### 4.4 The fold is the one keeper, and it is free

Arm N differs from arm Q in exactly one thing: the RMSNorm gains are folded into the linears that
read them **before** R3 runs. Arm XN proves the fold is exact as a re-parameterization
(`+6.8e-09`). It buys **`−0.220001 ± 0.052861`**, 44 σ_seed, **8.1% of the `+2.716656` that R3
leaves on the runnable organ set**.

Same format, same shapes, one fp32 scale per output row, no new kernel, no runtime cost — the
change is entirely in the exporter (the folded model's norm gains become 1.0). This is **the
third time in this programme that the parameterization, not the format, was where the damage
was** — after T2's `RULE-HELPS` and T2's finding that R0 is indistinguishable from random signs.

**It does not gate anything yet.** It was measured as a control inside a `VOID` run, and it has
never been measured through `donor_engine.c`. It goes on the open list as a candidate, with its
own confirmation owed.

### 4.5 The smoke inverted every sign it was asked about

The 4-layer / 2-sequence smoke (`t3_rotation_smoke.json`, `ternary/t3_smoke.log`) and the run:

| contrast | smoke (4 of 28 layers) | run (28 layers) |
|---|---|---|
| `N − Q` (the fold) | `+0.185073` | **`−0.220001`** |
| `H − N` (Hadamard) | `−0.596376` | **`+1.137618`** |
| `O − N` (dense orth) | `−0.532612` | **`+0.634489`** |
| mechanical label | `VOID` (same Q constant) | `VOID` (same Q constant) |
| label with the corrected Q gate | **`ROTATION-WINS`** | **`NULL`** |

**Three contrasts, three sign flips, and the smoke's label is the opposite of the run's.** (Both
runs return `VOID` mechanically, for the §1.3 reason; the row that matters is the corrected one.)
The
smoke's X controls passed at `6.4e-07`, its planted null passed, its algebra was correct and its
conclusion was wrong. `truncate(model, k)` was added precisely so the smoke would cut the model
and not the transformation, and it did its job — the transformation was right. A truncated model
is simply a different model for this question.

> **LAW.** *A smoke run establishes that the apparatus runs. It does not establish the sign of
> the effect.* A truncated model is not a small model: cutting 24 of 28 layers reversed the sign
> of every contrast here, not merely their size.

---

## 5. What this closes and what it costs

**Closed.** The residual-stream rotation line, on this donor at ternary width, with rule R3.
Brief §5's follow-ons are closed with it and should not be run:

- the **online** Hadamard (a runtime multiply between `up`/`down` and `v`/`o`) — its cheaper
  offline sibling costs `+0.634` at best; there is no gain here for a runtime cost to protect.
- the **activation-side** rotation for the LUT path. That is a separate measurement and the brief
  correctly said it would not be inferred from these numbers — but it would now be inferring a
  win from a `+1.138` loss on the weight side, and the crest-factor argument it rests on is
  untested independently. If it is ever run it needs its own pre-registration and its own reason.

**Cost.** 2656 s of one CPU, no GPU, no gradients. The brief budgeted "six arms, under an hour";
it ran ten arms (four X controls and arm N being the two documented additive departures) in 44
minutes.

**Unchanged.** `donor_engine.c` was not modified by this probe, exactly as brief §2.3 said. The
runtime numbers in `SPEED_LEDGER.md` §12 are untouched by T3.

**Standing after T3.** The whole runnable model at ternary width still costs **`+2.708`**
(T2b arm FAH), or **`+2.497`** with the fold on the FFN + attention organs. Healing still has to
come from somewhere else, and T2b §6 still says the place to aim it is **attention** — 4.98× the
damage per weight at 10% of the weights.

---

## 6. What the brief got wrong, recorded so the next one does not

1. **A pre-registration can contradict itself, and the gate will fire on the contradiction.**
   §3 fixed the organs at FFN + attention; §4 pinned the replication constant to an FFN-only
   number. Both were written the same afternoon. **A replication constant must name the arm it
   came from and the file it is in** — `+1.709372` should have read "T2 arm R3, FFN only, 84
   tensors, `t2_rules.json`", at which point the mismatch with §3 is visible on the page.
2. **The enabling measurement was read with its sign inverted.** D2 measured rotation as
   de-sparsifying and recorded it as a negative. The brief re-read the same table as a positive
   by asserting "quantization wants the opposite of sparsity" — an unmeasured premise, stated in
   bold, with no arm testing it. The thing T3 falsified hardest is that sentence.
3. **The smoke's label was quoted before the run.** It was quoted with "smoke is not the run"
   attached, which was the right hedge, but the label was `ROTATION-WINS` and the run was `NULL`.
   §4.5 is now a law.

---

## 7. Reproduction

```
cd benchmarks/donor_adaptation/ternary
D_THREADS=6 python t3_rotation.py                 # 2656 s, 10 arms
D_THREADS=6 T3_SMOKE=1 python t3_rotation.py      # 237 s, 4 layers -- APPARATUS ONLY (see 4.5)
T3_ONLY=base,Q,N python t3_rotation.py            # the fold contrast alone
```

Outputs `density/results/t3_rotation.json` and `density/results/t3_arms/<arm>.npy` (24
per-sequence BPB values per arm; every contrast in §3.1 is recomputable from those files and the
byte weights, with `numpy.random.default_rng(7)`).
