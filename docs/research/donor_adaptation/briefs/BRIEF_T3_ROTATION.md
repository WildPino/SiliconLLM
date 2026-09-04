# BRIEF T3 — Rotate the basis before ternarizing

**Status: PRE-REGISTERED. Written and pushed before the run, per standing practice.**
**Author: the Adapter / Principal. Date: 2026-09-04.**
**Depends on: `probes/T2_TERNARIZATION_RULE.md`, `results/d2_basis.json`, `SPEED_LEDGER.md` §11.**

---

## 1. Three separate findings that turn out to be the same finding

**T2 §4:** the whole of the ternarization damage that any rule could remove was in *where the
grid sits*. Searching the per-row scale bought −0.686 BPB and weighting that search by activation
RMS bought a further −0.914 — the largest single component. Nothing about *which* weights get
which sign mattered: R0 is statistically indistinguishable from random signs (−0.064 ± 0.126).

**The LUT path (`donor_engine.c --lut-diag`):** int8 activations cost `rel l2 1.40e-01` because
the donor's activation **crest factor** (`amax/rms`) is 8.3 on average and **69.6** at worst, so a
63-step grid leaves a typical element 4–8 usable levels. Per-group scales recover 4.5× of that by
confining outliers, not by removing them.

**D2 (`results/d2_basis.json`), already measured on this donor and read for a different question:**

| basis | median kurtosis over 12 real tensors | worst |
|---|---|---|
| identity | **4.81** | **22.11** (L27.o_proj) |
| `hadamard_block` | 3.09 | 4.24 |
| `block_random_orth256` | 3.05 | 5.03 |
| `dense_random_orth` | 3.01 | 3.14 |

A Gaussian has kurtosis 3. **An orthogonal rotation reliably flattens this donor's weights to
Gaussian, and it flattens the worst tensor by 5×.**

> D2 recorded that table as a **negative**, and correctly so for its own question: rotation makes
> the weights *less sparse* (`frac90` 0.31 → 0.44), and D2 was hunting a basis in which they are
> concentrated. **Quantization wants the opposite of sparsity.** The same numbers that killed D2's
> hypothesis are the evidence for this one, and no new measurement was needed to find them.

> **The question this brief pre-registers:**
> **Does rotating the residual basis before ternarizing reduce the damage T2 left on the table?**
> Outliers are what force a per-row scale to be wrong for most of its row. If a rotation removes
> them from the weights *and* from the activations, both of this programme's two open numeric
> problems are the same problem.

## 2. Why this is legal, and why it costs nothing at runtime

For any orthogonal `R`, `y = W x = (W R)(Rᵀ x)`. Rotation is **exact in fp arithmetic terms** — it
is a change of basis, not an approximation — and it is free at inference **if `Rᵀ x` is what the
previous organ already produces.** That is arranged offline:

1. **Fold the RMSNorm gain `γ` into the following linear.** After folding, RMSNorm is a pure
   rescale by `‖x‖`, and `‖Rx‖ = ‖x‖`, so the norm commutes with `R`.
2. **Rotate the residual stream once:** `embed → embed·R`, every organ reading the stream gets
   `W·R`, every organ writing to it gets `Rᵀ·W`, and the head gets `Rᵀ` folded in.
3. The engine sees **the same shapes, the same format, one fp32 scale per output row, and no new
   kernel.** `donor_engine.c` is not modified by this brief at all.

**This is the QuaRot/SpinQuant construction and it is not novel.** It is proposed here because
this programme has never applied it, its enabling measurement is already on disk, and it is the
only lever left that attacks the weight side and the activation side with one change.

## 3. Arms

Fixed: donor Qwen2.5-1.5B rev `8faed761…`, shared eval slice (`heldout` 24×512 seed 1234,
`ids_sha256 = a1a48dc9…`), FFN + attention organs, calibration 32×512 seed 42424, paired
sequence-bootstrap SEs, **rule = R3** (T2's pre-registered winner, so T3 and T2 differ in exactly
one thing).

| arm | what | purpose |
|---|---|---|
| **base** | nothing | replication gate: `0.767594958` |
| **X** | rotate + fold, **no quantization at all** | ⚠ **THE CONTROL THIS BRIEF LIVES OR DIES BY.** A change of basis is exact, so this must reproduce base to fp round-off. If it does not, the fold is wrong and every other arm is measuring my algebra, not rotation |
| **Q** | R3, no rotation | replication gate: must reproduce T2's `+1.709372` |
| **H** | R3 after `hadamard_block` | the cheap rotation, block-diagonal |
| **O** | R3 after `dense_random_orth` | the flattest basis in D2's table (kurtosis 3.01) |
| **P** | R3 after a **permutation** | ⚠ **PLANTED NULL.** A permutation is orthogonal and leaves kurtosis *exactly* unchanged (D2 measured `permutation` identical to `identity` to 16 digits). If P improves things, the gain is not coming from flattening and the mechanism claim is dead |

Arm **P** is the arm that makes the others mean something. Any rotation reshuffles which weight
lands in which row and therefore perturbs the per-row scale; P has that property and *only* that
property, so `H − P` and `O − P` are the parts attributable to flattening.

## 4. Pre-registered decision rule — fixed before any result

Let `Δ(arm)` = BPB(arm) − BPB(base). Let `q = Δ(Q)`, expected ≈ `+1.709`. Let
`best = min(Δ(H), Δ(O))`.

| outcome | condition | what it means |
|---|---|---|
| **ROTATION-WINS** | `best ≤ q − 0.30` **and** `Δ(P) ≥ best + 0.20` | flattening is the mechanism and it is worth ≥0.30 BPB free. Compose with R5, re-run T2b, and apply the same rotation to the LUT path's activations |
| **ROTATION-MARGINAL** | `q − 0.30 < best ≤ q − 0.05`, P as above | real but small; report and do not build on it |
| **PERMUTATION-ARTEFACT** | `best ≤ q − 0.30` **but** `Δ(P) < best + 0.20` | the gain is row reshuffling, not flattening. **The mechanism claim is withdrawn** and what is really being measured is that R3's per-row scale is sensitive to row composition — itself a finding, and a cheaper one to exploit |
| **NULL** | `best > q − 0.05` | kurtosis does not predict ternarization damage on this donor. D2's table is then evidence about sparsity only, and this line of attack closes |
| **VOID** | arm X does not reproduce base to `< 1e-4` BPB, or arm Q misses T2's Δ by `> 0.01` | the fold or the harness is wrong; report nothing else |

`0.30` is the bar for ROTATION-WINS because it is 60 σ_seed and ~18% of the residual damage R3
leaves — large enough to change what the healing brief must do, and small enough to be reachable.
**These thresholds are fixed here, before the run.**

## 5. What this brief does NOT test

- It does **not** test rotations that need a runtime multiply (the "online" Hadamard inside the
  FFN between `up` and `down`, and inside attention between `v` and `o`). Only the residual-stream
  rotation, which folds entirely offline. If T3 wins, that is the next question.
- It does **not** test the effect on the LUT path's **activation** crest. That is the same
  rotation and the obvious follow-on, but it is a separate measurement in `donor_engine.c` and
  will not be inferred from these numbers.
- It does **not** change the engine format or `donor_engine.c`.
- It does **not** compose with R5, GPTQ, or T2b's organ sweep. One variable at a time.
- It does **not** sweep the calibration budget. `D4b` is still unrun; every R3-family number in
  this programme remains a **floor**.

## 6. Cost

Rotation is offline weight surgery plus one BPB pass per arm (~140 s at 6 threads on 1.5B,
measured in T1/T2/T2b). Six arms plus one calibration pass. **CPU only, no gradients, under an
hour.**

## 7. Reporting

`probes/T3_ROTATION.md`. Report Δ and paired SE for every arm, the arm-X and arm-Q controls, the
planted null P, the §4 label verbatim, and the paired contrasts `H − P` and `O − P` — because
T2 §4 is where a Δ-against-base was shown to be unable to compare two arms.
