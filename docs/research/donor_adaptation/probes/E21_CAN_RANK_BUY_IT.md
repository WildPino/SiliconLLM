# E21 — can rank buy what precision and sparsity could not? `ATTENTION-YES-HEAD-NO`

**Brief**: `briefs/BRIEF_E21_CAN_RANK_BUY_IT.md` @ `24b8832`, pushed before any arm ran.
**Runner**: `benchmarks/donor_adaptation/ternary/e21_rank.py`. **Results**:
`engine/results/e21_rank.json`. Run 3445 s, `VOID: none`. **No timing taken; `6.79 tok/s` exact.**

---

## 0. The finding

**Attention's `q/o` is nearly free to compress by rank. The output head is not.** Both were
measured the same way, on the same donor, in the same run, with the same construction.

| | BPB cost | teacher-forced top-1 | mean rank of donor token |
|---|---|---|---|
| `QO-ACT-512` — attention `q`+`o`, all 28 layers | **`+0.052689`** | **144/160 (90%)** | **`1.16`** |
| `H-ACT-512` — `lm_head`, same rank, same method | `+0.278096` | 101/160 (63%) | `4.83` |

`QO-ACT-512` is **the least damaging structural modification this programme has ever measured** —
cheaper than the best ternarization (E20 `GPTQH`, `+0.170414`) and cheaper than the best carve
(E19 `V52`, `+0.141846`) — and it is the **only arm across E18, E19, E20 and E21 to read
`RANK-IS-CHEAPER`**, the band fixed in the brief before the run.

**But the configuration the brief was built to test failed.** `BOTH-ACT-256` — head and `q/o`
together at rank 256, the §2 budget configuration — reads `+0.998577` BPB and **48/160**
teacher-forced, worse than either organ alone (68 and 93) -- **while its BPB is `0.342498` BETTER than the
additive prediction.** The composition is sub-additive in BPB and below both parts in ranking; see §4a.

---

## 1. The construction, and why the control is not a straw man

The format question does not arise here: every arm is fp32 and nothing is exported. The axis is
**rank**, which this programme had never measured end to end — D2 computed spectra for `q/o` and
the FFN on three layers and stopped, with no BPB, no ranking and no `lm_head`.

Arms minimise **`‖(W − Wᵣ)X‖_F`**, not `‖W − Wᵣ‖_F`:

```
Wr = W H^(1/2) Br Brᵀ H^(-1/2),   Br = top-r eigenvectors of H^(1/2) WᵀW H^(1/2)
```

the exact rank-`r` minimiser of the activation-weighted error, reducing to plain SVD at `H = I`.
Computed through the 1536×1536 Gram rather than a 151936×1536 SVD — same answer, far cheaper.
`H = XᵀX` over the frozen 32×512 seed-42424 calibration slice (16,384 tokens), 57 organs in one
pass; damping `λ = 0.01·mean(diag H)`, E20's, unchanged.

**The brief predicted this would matter and stated the unfavourable prior first.** D2's measured
spectra say these matrices are not low-rank in weight space: `q_proj` needs rank 425–547 for 90%
of Frobenius energy, `o_proj` 596–937. `G-R2` measures whether the data weighting rescues that:

| pair | plain SVD | activation-weighted | Δ |
|---|---|---|---|
| head, r=256 | `3.826871` | `1.330385` | **`−2.496486`** |
| head, r=512 | `3.480711` | `1.045691` | **`−2.435020`** |
| `q/o`, r=256 | `4.686939` | `1.545880` | **`−3.141059`** |

**`G-R2` holds on every pair and by an enormous margin.** Plain SVD at r=256 puts `q/o` **above
the chance line** (`4.686939` against `4.069819`) — a model worse than guessing uniformly — while
the same rank, weighted, reads `1.545880`. The energy columns say why: plain SVD at r=512 retains
`0.5319` of the head's Frobenius energy, while the weighted construction at r=256 retains `0.8937`
of the energy **in the directions the data occupies**. **A matrix that is not low-rank in weight
space can be low-rank where it is used**, and D4's Hessian ablation (`real_H` `0.483`,
`identity_H` `0.0`) is confirmed on a second axis.

---

## 2. Gates

| gate | demands | result |
|---|---|---|
| `G-R0` | `base` reproduces E6's reference on **both** metrics | **FIRES** — `160/160` free-running **and** `160/160` teacher-forced, BPB `0.767595` |
| `G-R1` | `FULL` (factorise at full rank through the same path) token-identical to `base`, BPB diff `< 1e-9` | **FIRES** — identical, diff `0` |
| `G-R2` | activation weighting beats plain SVD in BPB at equal rank | **HOLDS**, all three pairs, `−2.44` to `−3.14` |
| `G-R3` | verdict: `BOTH-ACT-256` against the bands | `RANK-IS-WORSE` (48/160) |
| `G-R4` | teacher-forced for every arm, known-positive first | fired first at `160/160` |

`VOID: none`.

---

## 3. The table

Free-running bands unchanged since E17/E18: floor `12`, `AT-FLOOR ≤ 14`, `RANKS ≥ 80`.
Teacher-forced bands fixed in brief §6 from E20 part B's measured `107`–`119` for eight ternary
heads: **`> 119` = `RANK-IS-CHEAPER`**, `107`–`119` = `RANK-IS-COMPARABLE`, `< 107` = `RANK-IS-WORSE`.

| arm | BPB | Δ vs base | free-running | **teacher-forced** | mean rank | rank ≤ 5 | weighted energy kept | band |
|---|---|---|---|---|---|---|---|---|
| `base` | `0.767595` | — | 160/160 | 160/160 | `1.00` | 160 | — | — |
| `FULL` | `0.767595` | `0.000000` | 160/160 | 160/160 | `1.00` | 160 | `1.0000` | — |
| `H-SVD-256` | `3.826871` | `+3.059276` | 1/160 | 12/160 | `39258` | 25 | `0.3638` | WORSE |
| `H-SVD-512` | `3.480711` | `+2.713116` | 0/160 | 27/160 | `32055` | 40 | `0.5319` | WORSE |
| `H-ACT-256` | `1.330385` | `+0.562790` | 7/160 | 68/160 | `22.38` | 101 | `0.8937` | WORSE |
| `H-ACT-512` | `1.045691` | `+0.278096` | 9/160 | 101/160 | `4.83` | 135 | `0.9375` | WORSE |
| `QO-SVD-256` | `4.686939` | `+3.919344` | 1/160 | 2/160 | `28329` | 8 | `0.6816` | WORSE |
| `QO-ACT-256` | `1.545880` | `+0.778285` | 7/160 | 93/160 | `43.31` | 130 | `0.9287` | WORSE |
| **`QO-ACT-512`** | **`0.820284`** | **`+0.052689`** | **26/160** | **144/160** | **`1.16`** | **160** | `0.9796` | **CHEAPER** |
| `BOTH-ACT-256` | `1.766172` | `+0.998577` | 4/160 | 48/160 | `103.96` | 84 | `0.9281` | WORSE |

---

## 4. `QO-ACT-512`, read carefully

E20 §4 established that an aggregate score can be one lucky prompt, so the split is checked first:

| arm | teacher-forced per prompt | free-running per prompt |
|---|---|---|
| **`QO-ACT-512`** | **`[30, 26, 30, 29, 29]`** | `[7, 0, 6, 5, 8]` |
| E20 `OPTH` (for contrast) | — | `[2, 2, 32, 3, 3]` |

**`QO-ACT-512`'s competence is uniform**, 26–30 of 32 on every one of the five prompts. It is not
E20's one-prompt artefact, and its free-running `26/160` — above the `12` floor, `PARTIAL` — is
spread across four of the five prompts. `rank ≤ 5` is **160 of 160**: the donor's token is in the
top five at *every single position*, and first on 144.

**And it is the first modified donor in this programme that writes sensible text.** Against the
donor's own continuations:

| | text |
|---|---|
| `base`, p0 | `" Paris. The capital of France is also the capital of the European Union. The c"` |
| `QO-ACT-512`, p0 | `" Paris. The capital of France is Paris. The capital of France is Paris. The ca"` |
| `base`, p2 | `" 212 °F or 100 °C and ice melts at 32 °F or 0 °C. If the temperature of"` |
| `QO-ACT-512`, p2 | `" 212 °F and water freezes at 32 °F. If the temperature of a pot of water is 20"` |

Correct Paris, correct 212/32 °F, a working Fibonacci branch on p1 and plausible `matplotlib`
imports on p4. Every ternary and carved arm from E15 onward produced either the donor's tokens or
noise; this produces **different, correct** tokens. That is what a `+0.052689` BPB cost and a
90% per-step argmax rate look like from the outside.

**Why attention and not the head.** `QO-ACT-512` retains `0.9796` of the activation-weighted energy
at rank 512 of 1536; `H-ACT-512` retains `0.9375`. The head maps a 1536-dimensional state to
**151,936** logits whose top-2 differences decide the answer — it is a near-full-rank object *by
construction*, and E17 already found it is the tensor whose job is ranking. Attention's `q/o`, by
contrast, feed a residual stream that sums 28 layers, so per-layer error has somewhere to go. That
is also brief §7's prediction 4, which **held**: `QO-ACT-256` (93) beats `H-ACT-256` (68) at equal
rank.

---

## 4a. Composition: sub-additive in BPB, below both parts in ranking

Define `excess = BPB(A+B) − [BPB(A) + BPB(B) − BPB(base)]`, zero if the two damages simply add.

| | BPB | teacher-forced |
|---|---|---|
| `H-ACT-256` alone | `1.330385` | 68/160 |
| `QO-ACT-256` alone | `1.545880` | 93/160 |
| additive prediction | `2.108670` | — |
| **`BOTH-ACT-256` measured** | **`1.766172`** | **48/160** |
| **excess** | **`−0.342498`** | **below both parts** |

**The two metrics disagree about the same composition, in opposite directions.** In BPB the pair
is `0.34` *better* than adding the parts; in per-step argmax it is `20` tokens *worse than the
better half and 20 worse than the worse half*. So "damage compounds" is true of ranking and false
of BPB, and the unqualified sentence should not be used.

> **Corrected 2026-09-11 after E22.** The sentence "BPB does not order interventions across
> axes", as stated in §6's third asymmetry, is too general. It was derived from one pair
> (`H-ACT-256` vs E20's `R0H`: within `0.011` BPB, 68 vs 107 teacher-forced). E22 measured a
> second pair and the ordering held to the token — E19's `V52` costs `+0.141846`, less than E20's
> best ternary head at `+0.170414`, and reads `117` teacher-forced against that head's `117`.
> **The claim that survives both is weaker: BPB ordering *can* fail across axes, so it must be
> checked rather than assumed.** E22 §7 carries the measurement.

The mechanism is §6's third asymmetry again. Weighted low-rank preserves logit *geometry* --
which is what BPB scores, and two geometry-preserving perturbations overlap rather than stack --
while argmax lives on the top-1/top-2 boundary, where two independent perturbations each get a
fresh chance to flip the order. **Sub-additivity in a score metric is not evidence of tolerance in
a rank metric**, which is E14 §3's law arriving on a third axis.

## 5. What this does *not* buy, priced honestly

**The §2 budget configuration failed, and the saving does not transfer as written.** `r = 512` on
this donor (`D = 1536`) is **one third** of full rank and saves only **33.3%** of `q/o`. The
brief's 7B table assumed `r = 256`, which at `D = 3584` is one *fourteenth* of full rank —
**a far more aggressive truncation than anything measured here**, and `BOTH-ACT-256` shows that at
r=256 even this donor breaks (`48/160`).

Redone at the rank fraction actually measured (`r/D ≈ 1/3`), charging every weight as ternary
(`0.500000` B) and using E18's `0.982–1.060 G` budget for 50 tok/s:

| Coder-7B | dense | with `q/o` at `r/D = 1/3` |
|---|---|---|
| `q + o` | `719.3 M` | `479.5 M` |
| `k + v` | `102.8 M` | `102.8 M` |
| `lm_head` | `545.0 M` | `545.0 M` (resists rank) |
| **`attn + head`** | `1.367 G` | **`1.127 G`** |

**`1.127 G` is still 1.06–1.15× the entire 50 tok/s budget with the FFN at zero.** Compressing
attention by the fraction this probe validated is a large win — E19's `1.367 G` floor drops by
`240 M` — **and it is not sufficient, because the head is `545 M` and the head is the organ that
does not tolerate rank.** E19's conclusion that a carve must cut attention *and* the head stands;
E21 shows the attention half is achievable and the head half is not, by this lever.

**No speed claim and no timing.** Nothing was exported. A low-rank tensor is two GEMVs with an
intermediate of size `r`; the weight *bytes* fall in proportion to the parameter count, which is
what a memory-bound engine charges (E10, E13: the weight path runs at 97% of demonstrated stream),
but **`engine.c` does not implement a factored tensor and `QWENDON1` has no kind for one.**
That work is owed and unpriced. **`6.79 tok/s` is untouched.**

---

## 6. Predictions — two held, two missed, one gate held hard

| # | brief §7 | outcome |
|---|---|---|
| 1 | `G-R0`, `G-R1` fire; `FULL` exactly lossless | **held** — token-identical, BPB diff `0` |
| 2 | `G-R2` holds by a large margin | **held** — `−2.44` to `−3.14` BPB, larger than the quantization analogue |
| 3 | `H-ACT-256` `AT-FLOOR` free-running **and** `107`–`119` teacher-forced | **half missed** — `7/160` free-running is `AT-FLOOR`, but teacher-forced is `68`, far below the ternary band |
| 4 | `QO-ACT-256` less damaging than `H-ACT-256` | **held** — 93 vs 68 |
| 5 | `BOTH-ACT-256` → `RANK-IS-COMPARABLE` | **missed** — `48/160`, `RANK-IS-WORSE` |

**The registered alternative did not fire on its own terms.** It was keyed to `BOTH-ACT-256`, which
read `RANK-IS-WORSE`. `QO-ACT-512` reading `RANK-IS-CHEAPER` is a **pre-registered arm against a
pre-registered band** (§4 names `QO-ACT` the verdict arm for attention, §6 fixes the band), so it
stands as a result — but it is not the configuration the brief was designed around, and the
alternative's consequence for the healing target is claimed only for **attention**, not for the
§2 configuration.

**What the prediction record keeps showing**: the direction was called right twice, wrong twice,
and what made the run readable was `G-R2` — a control registered because the prior was *against*
the method, not because it was expected to pass.

---

## 7. What changes

**A structural cut that is nearly free now exists.** Every experiment from E15 to E20 closed a
route. This one opens a narrow one: attention `q/o` at `r/D ≈ 1/3`, activation-weighted, costs
`+0.052689` BPB and keeps 90% per-step argmax with uniform per-prompt competence. **It is the
first intervention here that is cheap enough to compose with others.**

**The healing target for the T4 sessions changes.** E18 §9 item 1 said "heal a ternary donor",
E19 sharpened it to `V52`, E20 sharpened it again to "a head already right 70% per step". E21
says the best-conditioned starting point available is **low-rank attention**, at 90% per step and
`+0.053` BPB — a far better-posed optimisation than anything previously on offer.

**The head is now the named binding constraint.** On a 7B it is `545 M` = 51–56% of the whole
50 tok/s budget on its own, it resists rank (`H-ACT-512`: `+0.278096` BPB, 101/160), and E20
showed its cheapest known treatment is ternary (`+0.170414`, 117/160). **No lever measured in this
programme makes the output head small.** That is the single most specific open problem the goal
now has.

---

## 8. Owed

1. **Compose the two cheapest cuts.** `QO-ACT-512` (`+0.052689`) and E19's `V52` FFN carve
   (`+0.141846`) together land the **1.5 B** at ≈`0.98 G` active — inside the 50 tok/s budget.
   Is the damage additive? §4a says the two metrics answer that differently for the same pair,
   so it must be measured on BOTH, not assumed from either. **Cheap, CPU, and directly on-goal.**
2. **Rank × precision.** Every arm here is fp32. The engine ships ternary. A ternary low-rank
   `q/o` is the artefact that would actually run, and the two damages have never been composed.
3. **The rank fraction at scale.** `r/D = 1/3` was validated at `D = 1536` only. The 7B claim in
   §5 is a projection at the measured fraction, not a measurement, and E16's non-monotonicity in
   scale applies.
4. **A factored tensor in `QWENDON1` and `engine.c`** — §5. Without it no rank result can be
   converted into a tok/s number.
5. **The head.** §7. It resists rank, costs `+0.170414` ternary, and is 51–56% of a 7B's budget.
   Vocabulary-side factorisation and tied output clusters are untouched here.
6. **Unchanged from E20**: healing/QAT (GPU, user launches); re-read published ranking numbers as
   compounds; `R5` in the exporter.

---

## 9. Appended 2026-09-11 after E22 — both owed items 1 and 2 are discharged, oppositely

**§8 item 1 — compose `QO512` with E19's `V52`.** Done, and it is the best result the donor
programme has: **`0.9441 G` active (inside E18's 50 tok/s budget), BPB `1.005039`, 126/160
teacher-forced, above every ternary head E20 measured**, per-prompt `[27, 24, 22, 28, 25]`.
§8 item 1 warned not to assume additivity, and it was right to: the pair is **super-additive in
BPB** (`+0.042909`) while ranking **above its worse half** (126 ≥ 117) — the exact mirror of §4a's
reading of `BOTH-ACT-256`. Four compositions now exist and no two behave alike; the operative rule
is that a stack must be measured.

**§8 item 2 — rank × precision.** Done, and it closes the axis. Ternarizing both factors of the
rank-512 form costs `2.812226` BPB / 28 teacher-forced, against T2b's **dense** ternary attention
at `1.903569` on a harder organ set (`k/v` included): **`+0.908657` worse.** Layer 0 `q_proj`,
relative weight error: dense ternary `0.8084`, fp32 rank-512 `0.3515`, two ternary factors
`0.9874`. **The factors' errors multiply.** `QO-ACT-512` is `+0.052689` BPB *in fp32 only*, and
that qualifier belongs on every future quotation of it.

**§0's headline therefore needs its second clause.** "The least damaging structural modification
ever measured here" is true and stays — **and it cannot currently be shipped**, because the format
it would ship in costs more than the modification saves. What E22 leaves standing is the target
for healing, not a runnable artefact.

`probes/E22_DOES_CHEAP_COMPOSE.md` carries both.
