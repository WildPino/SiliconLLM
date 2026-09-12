
# BRIEF E37 ADDENDUM — two of my own gates, as written, are wrong

**Pushed before any E37 artifact is exported and before any number exists.** Both corrections
come from reading `qwen_export.py`, `donor_engine.c` and `e26_parity_carve.py`, not from seeing
a result. This is the same discipline as E33's addendum (`c6090f4`): a gate that cannot be
satisfied, or that is satisfied for the wrong reason, is worse than no gate.

---

## 1. `G-E37A`'s tolerance ignores that the carved path reorders a sum

**As registered:** *"`S15-K256`'s BPB must equal `S15-DENSE`'s to **< 1e-6**."*

**Why that is wrong.** At `k = E` the carved FFN keeps every group, so it computes the same
function — but not by the same additions. `donor_engine.c`'s `ffn_carved` builds `g_rows` in
**router-selection order** and accumulates `down` with `matvec_colacc` over those rows, while
the dense path walks `F = 8960` columns in index order. **Same trits, same scales, different
summation order in fp32.** A `1e-6` bar is a statement about floating-point associativity, not
about my plumbing, and it would fail for a reason that has nothing to do with the carve.

**`G-E37A′`, which adopts a number this programme already uses for the identical comparison
rather than inventing one.** `e26_parity_carve.py:49` sets `TOL_P = 1e-4` with the comment
*"G-E26P: same trits, a different summation order"* — E26 part A ran exactly `carved --carve-k E`
against `dense` and read `4.481e-06` worst relative L2 with top-1 `1.0000`. `G-E37A′` therefore
scores parity as E26 scored it:

* **worst relative L2 on logits `< 1e-4` AND top-1 agreement `== 1.0000`** (E26's bar, adopted);
* **BPB agreement `< 1e-4`**, the same scale, over 12,264 predictions.

If `G-E37A′` fires, damage at lower `k` is the sparsity. If it does not, E37 reports no quality
number at all.

## 2. `G-E37B` compares against a reference the carve is forbidden to match

**As registered:** *"`S15-DENSE` through the engine must reproduce E1/E16's published
**3.475706372** to within `0.001`."*

**Why that is wrong.** The published anchor artifact (`D:\_ktmp\e1\qwen25-15b_tqh.bin`) was
exported at `--rule R3 --head-ternary` with the **default `--fold layers`**. `qwen_export.py:390`
**refuses `--quant carved` unless `--fold none`**, because the labels were fitted on the unfolded
donor and a fold rescales the very FFN rows the groups name. So the carved arms **cannot** be
built fold-matched to the anchor, and a dense control that *is* fold-matched to the carve is a
different object from the one the record publishes. As written, `G-E37B` asks one artifact to be
two things.

**`G-E37B′`, split into the two questions it was conflating:**

* **`G-E37B1` — the box and the protocol.** Run the **existing published artifact** unchanged
  and require **3.475706372 ± 0.001**. This checks that this session reproduces the record. It
  is a gate.
* **`G-E37B2` — the reference for the carve.** Build `S15-DENSE-NF` at `--fold none`, same rule,
  same head treatment. **This** is the parity reference for `G-E37A′` and the dense baseline
  every carved arm is scored against.

**The difference between the two is reported as a MEASURED FOLD COST and is not scored as a
gate.** It is a fact about what folding does to a ternarized donor — interesting, previously
unmeasured on this cell, and not this probe's question.

**Registered now, before the build: I am not predicting its sign or size.** Whatever it is, the
verdict cell is scored against `S15-DENSE-NF`, because that is the object the carved arms differ
from by exactly one thing.

## 3. What does not change

The verdict cell (`S15-K3` BPB), the bands (3.70 and the 4.069819 chance line), `G-E37C`
(charged accounting, zero tolerance), `G-E37D` (the fitted router must beat a seed-matched
random one), and **every prediction in §6 of the brief, including prediction 2's registered
expectation of `SPARSITY-DESTROYS`.** None of them is touched by either correction.
