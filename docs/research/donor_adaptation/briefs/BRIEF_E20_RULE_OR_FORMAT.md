# E20 — is it the rule, or the format? A data-optimal ternary head

**Pre-registration. Nothing below has been measured.**

---

## 0. The question, and the qualifier it pays off

E16 concluded that the binding constraint had passed **from the rule to the format**, and recorded
its own qualifier in the same breath: *only 2 of 4 rules were ever tested.* E17, E18 and E19 have
since eliminated the head, the organ ladder, and structural sparsity. But **every ternarization this
programme has ever run is a weight-space rule that never looks at a single token**:

```
t1_ternarize.py:97   scale = w.abs().mean(dim=1, keepdim=True)   # per output row
t1_ternarize.py:98   wq    = (w / scale).round().clamp(-1, 1)     # {-1, 0, +1}
```

That is BitLinear-1.58 round-to-nearest. It minimises nothing that the model is evaluated on. The
literature's post-training quantizers (GPTQ, AWQ, SparseGPT — and this programme's own **D4**, whose
Hessian ablation fired cleanly: `real_H` `0.483` vs `identity_H` `0.0` vs `shuffled_H` `−1.595`)
all choose codes using the **input second moment** instead.

**So "the format cannot hold this donor" has never actually been tested. What has been tested is
"the cheapest possible rule cannot hold this donor."** E20 separates them.

**The format is held fixed and is exactly what the engine stores**: ternary codes `{−1, 0, +1}` with
**one fp32 scale per output row** (`QWENDON1`'s `rows_head = V` scale slots). Same bytes, same
kernel, same `0.500000` B/weight. **Only the choice of codes and scales changes.**

## 1. Why the head, and why 1.5 B

The head is the one tensor whose job is ranking (E17), it is the cheapest damage in the programme
that is still fully broken (T2b `H`: **`+0.338989` BPB, `9/160`**, below the `12` floor), and
ternarizing it alone leaves **every other weight bit-exact** — so the measurement is a clean
approximation question with no confounds:

> **Can a ternary 1536×151936 matrix with per-row scales, given the donor's exact hidden states,
> reproduce the donor's argmax?**

If a data-optimal ternary head **ranks**, the rule was the problem and E15–E19 have been measuring
bad rules, not a bad format — which reopens the conversion route. If it does **not**, that is
capacity evidence about the format itself, and it is the strongest form of the conclusion this
programme has been circling.

1.5 B because every ranking result in E17/E18/E19 is at 1.5 B, and because at 0.5 B the shipped
ternary arm sits **above** the chance line (E14 §10), so there is no signal to recover.

## 2. Arms — three points on one axis: how much does the quantizer know?

All arms ternarize **`lm_head` only**. `tie_word_embeddings = True` on this donor, so the head is
untied and cloned first, exactly as `t2b_organs.py:149` does — the embedding stays fp32.

| tag | scale | codes | knows the data? | role |
|---|---|---|---|---|
| `base` | — | — | — | **control**, must reproduce `results/e6/ref.json` |
| `ID` | — | identity quantizer through the same code path | — | **planted control for the machinery** |
| `R3H` | `mean|w|` per row | RTN | **no** | **known-negative**, must replicate T2b/E18 |
| `OPTH` | per row, searched to minimise `‖w − s·q‖²` | RTN at that scale | **no** | isolates *scale choice* from *data* |
| **`GPTQH`** | per row, searched to minimise `(w−ŵ)ᵀH(w−ŵ)` | **Hessian error compensation across input columns** | **yes** | **the verdict arm** |

`H = E[h hᵀ]` over the frozen calibration slice (32×512, seed 42424), `h` = the exact hidden state
entering the head. `H` is 1536×1536; damping `λ = 0.01·mean(diag H)`, stated here, not tuned after.

`OPTH` exists so that a `GPTQH` win cannot be attributed to "we finally picked a sensible scale".

## 3. Gates

- **`G-Q0`** — `base` reproduces E6's reference at `160/160`. Otherwise **VOID**.
- **`G-Q1`** — `ID` is **token-identical** to `base` and its BPB differs by `0`. This proves the
  untie-clone-requantize path is lossless when the quantizer is the identity. Otherwise **VOID**.
- **`G-Q2`** — `R3H` replicates T2b/E18: BPB `1.106584` to `< 1e-5` **and** `9/160`. Ties this
  harness to the published numbers before any new arm is read. Otherwise **VOID**.
- **`G-Q3`** — the verdict: `GPTQH` greedy agreement against the bands in §4.
- **`G-Q4`** — sanity, **not** a verdict: `BPB(GPTQH) < BPB(OPTH) ≤ BPB(R3H)`. If the data-aware
  arm does not beat the weight-space arms **in BPB**, the implementation is suspect and that is
  **reported as a finding, not silently accepted** — a quantizer that cannot win on the objective
  it optimises has not been demonstrated to work.

## 4. Bands — fixed here, before the run

Unchanged from E18/E19, and derived from measured quantities:

- **floor = 12/160** — best constant-token predictor on this reference (E18 part A).
- **margin = 2** — E17's own.
- **`AT-FLOOR` ≤ 14**; **`RANKS` ≥ 80**; between them, **`PARTIAL`**.

## 5. Predictions — and the alternative, registered

1. `G-Q0`, `G-Q1`, `G-Q2` all fire. *(High confidence: E18 and E19 both did the equivalent.)*
2. `G-Q4` holds — the Hessian arm wins on BPB, by a visible margin (`≥ 0.05` BPB over `R3H`).
3. **`GPTQH` → `AT-FLOOR`.** Reasoning: E19's `V52` cost only `+0.141846` BPB — less than half
   `R3H`'s `+0.338989` — and still scored exactly the floor. A quantizer that recovered, say, half
   of `R3H`'s BPB damage would land near `V52`'s damage level, and `V52` did not rank.
4. `OPTH` lands between `R3H` and `GPTQH` in BPB, and `AT-FLOOR` in ranking.

**THE REGISTERED ALTERNATIVE.** *If `GPTQH` comes back `RANKS` or `PARTIAL`, then the format is not
the constraint and E16's promotion of "format over rule" — carried forward through E17, E18 and E19
and into the ledger's §§29–32 — was premature. In that case every one of those eliminations must be
re-read as a statement about round-to-nearest, the conversion route reopens, and the next experiment
is `GPTQH` applied to every organ, not just the head. This outcome would be the most consequential
single result in the programme and it is written here before the data exists.*

**And the third outcome:** if `G-Q4` fails — the data-aware arm does **not** win on BPB — then E20
decides nothing about the format and is a report on a broken or mis-specified quantizer. That is a
real possible outcome and is registered as such rather than being available as an excuse afterwards.

## 6. What E20 will not be able to claim

- **One tensor, one donor, one scale.** A head-only result does not transfer to the FFN, which is
  80.7% of the weights and has different conditioning.
- **GPTQ is not the optimum.** It is a strong greedy method, not a proof of the best ternary head.
  A null is therefore *evidence* about capacity, not a theorem — the honest statement is "the best
  method this programme can build does not rank", never "no ternary head can rank".
- **No healing.** E20 *fits* weights; it does not train them. SGD healing remains untested and
  remains E18 §9 item 1.
- **No speed claim.** Same format, same bytes, same kernel: **`6.79 tok/s` is untouched** and no
  timing will be taken. A `RANKS` result would change what can be *quoted*, not what runs.
- **The head is 15% of this donor's active weights.** Even a perfect ternary head does not by itself
  produce a fast model; E19 part A already showed where the budget goes.

## 7. Cost

`H` capture on 16,384 calibration tokens (one forward pass), then five arms × (BPB on the frozen
24×512 heldout + 160 greedy positions). GPTQ over 1536 input columns on a 151936-row matrix is
`~1.8e11` element-updates, minutes on 6 threads. **Estimated 60–90 minutes, CPU only, one job.**
