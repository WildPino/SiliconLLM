# Brief S1 — which bar predicts BPB: a published, calibration-free sparsity rule measured against our own metric

**Author:** the Adapter / Principal · **Date:** 2026-08-29 · **Status:** pre-registered, not yet dispatched.

---

## 0. Why this exists — F1 left exactly one door open and the Researcher found the key

`BRIEF_F1` Amendment 1 §A1.4 registered the single route to a different answer:

> *"Nothing here connects a per-layer ε to end-to-end quality, and no BPB was run. Every number is Frobenius
> energy of one FFN block's output. If the model tolerates far more than ε = 0.05 end-to-end, the ladder
> stops too early."*

Two independent results now make that door cheap to walk through.

**(a) A published measurement on our donor's exact geometry disagrees with ours — but on a different bar.**
arXiv:2509.00454 **[T] Table 1** reports, on Qwen2.5-1.5B, `S_inter` = **50.49%** droppable at ≥99% of the
unpruned model's own average zero-shot accuracy. F1 measured, on the same geometry, that an oracle seeing
both `W_g` and `W_u` reaches only **16.4%** at 1% FFN-output Frobenius error, and **4.9%** at ε = 0.01.

**These are not in conflict.** The Researcher established the rules select on the same axis (`S_inter` and
our oracle both act on the intermediate vector) but hold different things fixed: **1% reconstruction error
versus 1% relative zero-shot accuracy.** This very session measured how far those can diverge — ShortGPT
**[T] Table 4**: perplexity `8.03 → 40.78` while MMLU ends **higher** (`43.17 → 43.35`).

> **So the question is no longer who is right. It is which bar predicts BPB.**
> BPB is a language-modelling loss, i.e. on the *loss* side of that decoupling — so **their bar is the looser
> and less relevant one, and ours may be stricter than BPB actually needs.** The truth lies somewhere in
> `[4.9%, 50.49%]` and **neither number locates it.** That interval is the difference between the FFN lever
> being dead and being the whole plan.

**(b) Their rule is fully specified, training-free, and calibration-free.** Per-token, per-layer:

```
    m_p  =  argmin ||m||_0   s.t.   ||m ⊙ v||_1  ≥  p · ||v||_1
```

— keep the largest-magnitude entries holding fraction `p` of the vector's **L1 norm**; sparsity is the
fraction zeroed. **No training, no teacher, no calibration set.** It sits inside every sealed constraint,
and their own §4 argues sparsification *"should be truly data-free"* because calibrated patterns overfit —
which cuts at TEAL/CATS **and mildly at F1's own fitted θ.**

**Applying their rule to our pinned checkpoint and measuring BPB is cheap, and it is the measurement that
decides the FFN lever.**

---

## 1. ⚠ THE SCALE TRAP — read before designing anything

The same **[T] Table 1** reports `S_inter` across three sizes:

| donor | `S_inter` |
|---|---|
| 0.5B | 46.54 |
| **1.5B (ours)** | **50.49** |
| 14B | **71.66** |

> **Sparsity RISES with model size, and our donor sits near the bottom of the trend.**
>
> **Every negative this programme has produced on FFN sparsity — F1 and D0 both — was measured at 1.5B.**
> The plan's target donors are 26B and above. **A negative measured at the bottom of a rising trend does not
> transfer upward**, and treating it as if it did would be the `h = 128` error repeated with the sign
> flipped: a quantity validated at one width and assumed scale-free.

**Mandatory:**

1. **State the largest donor that fits on this machine**, and run the sweep at **at least two sizes** if two
   are obtainable within the SKU and the disk. Qwen2.5-0.5B and -1.5B are both small; a third point above
   3B is worth real effort.
2. **If only one size is feasible, say so in the report's first paragraph** and mark every conclusion as
   **size-local**. Do not write a sentence that generalises across scale from one point.
3. Report the **trend**, not the endpoint, if you get two or more points.

---

## 2. What to measure

Per layer, per token, sweep the **global** `p` (their rule uses one `p` uniformly across all layers, and the
mean induced sparsity is what they report — **do not silently substitute our per-layer fitting**, that is a
different method).

| arm | rule | why |
|---|---|---|
| **A — published** | top-`p` L1 on the intermediate `i = (W_u x) ⊙ σ(W_g x)` | the thing being tested |
| **B — ours** | F1's ε-threshold on pre-activation `W_g x` | our own rule at **matched achieved sparsity**, so the comparison is of rules, not of budgets |
| **C — NULL** | random mask at **matched achieved sparsity** | the floor. **Any arm not beating this is measuring nothing** |
| **D — oracle** | top-k by true `\|h_i\|` at matched sparsity | the ceiling F1 already established at 16.4%; here it gets a BPB, not a Frobenius number |

**The deliverable is one curve: BPB against ACHIEVED FFN sparsity, all four arms on the same axes.**

Report **achieved** sparsity throughout, read off the masks, never the requested `p`. Report BPB against the
unmodified model's BPB on the same held-out slice, and give `σ_seed` context — the project's constant is
**0.005 BPB**, and a delta below it is not a delta.

---

## 3. Controls, pre-registered

| # | control | must |
|---|---|---|
| **C1 IDENTITY** | `p = 1.0` (nothing masked) | **BPB identical to the unmodified model, to full precision.** If it is not, the harness is wrong and nothing else in the run counts |
| **C2 PLANTED** | mask a known-critical set (e.g. the top-`k` by `\|h\|`) | **BPB must degrade sharply.** An instrument that cannot detect damage cannot certify its absence |
| **C3 NULL FIRES** | arm C at high sparsity | **must be clearly worse than arms A/B.** If random matches the method, the method is not a method |
| **C4 ACHIEVED** | achieved sparsity printed from the masks, per layer and in aggregate | reported next to nominal `p` everywhere |

**C1 and C2 are the pair that make this instrument trustworthy: one shows it does not lie when nothing
changes, the other shows it notices when something breaks.**

---

## 4. What this probe does NOT settle, stated so it cannot be over-read

**It measures whether the sparsity EXISTS. It says nothing about whether we can exploit it.**

D0 has already established, on this donor, that **scattered sparsity is worth approximately zero at our 48 KB
block granularity**: identity, random and co-activation orderings all give `0.0000` block-skippability on 27
of 28 layers, `expert_active(full)` sits at 0.998–1.000, and no permutation tested makes the firing pattern
contiguous.

> **So a strong result here does NOT revive the MoE-carve route.** It would mean the sparsity is real and
> our exploitation strategy is what fails — which is a different, and more hopeful, problem than the one we
> currently think we have. **Keep the two questions separate in the report.** Do not let a good BPB curve be
> read as a speed result; there is no speed result in this brief.

**A qualitative check the metrics cannot give:** at the best surviving `p`, **sample generated text and read
it.** This session found five of eight depth-pruning papers with no free-form generation benchmark at all,
and one method reporting "85.10% retention" while XSum fell from 20.82 to **0.04**. BPB is on the right side
of that decoupling, but a few paragraphs of output cost nothing and have caught what aggregates hide.

---

## 5. Standing rules

> Report **ACHIEVED**, printed from the objects — here, from the masks.
> An instrument must **fire on a known positive** before its passes count. C2 is that control.
> A result that flatters the hypothesis earns more scrutiny — **and a large tolerable sparsity would flatter
> the whole programme**, since it reopens a lever we spent today closing.
> **Write no number from a stage still running.**
> `σ_seed = 0.005 BPB`. **MMLU never inside an average**, and it is not this probe's metric in any case.

---

## AMENDMENT 1 — 2026-09-03, the Adapter / Principal — the scale arm becomes runnable

**Appended, not edited in place. Pre-registered before the arm exists.**

§1 of this brief declared the scale question mandatory and then conceded it might be infeasible:
*"If only one size is feasible, say so in the report's first paragraph and mark every conclusion
size-local."* **The Owner has since made Kaggle T4×2 available, and the concession is withdrawn.** The scale
arm is now required, not optional.

### A1.1 Why this arm outranks everything else in the brief

Every FFN-sparsity result this programme holds was measured at **Qwen2.5-1.5B**:

| result | value | measured at |
|---|---|---|
| F1 oracle bound on `|h_i|` | `a` = **0.836** | 1.5B |
| F1 threshold ladder | `a` = 0.951 at ε=0.01 | 1.5B |
| D0 carved-model BPB, 28 layers, 25% active, oracle router | **+1.09 BPB** (0.7676 → 1.8582) | 1.5B |

And the one published measurement on the size axis runs the other way — arXiv:2509.00454 **[T] Table 1**:
`S_inter` **46.54%** at 0.5B, **50.49%** at 1.5B, **71.66%** at 14B.

> **Our donor sits near the bottom of a rising trend, and the plan's targets are 26B and above.**
> A negative measured there does not transfer upward. Treating it as if it did is the `h = 128` error with
> the sign flipped — a quantity validated at one width and assumed scale-free. **This programme has made
> that error before and it is the reason this arm exists.**

### A1.2 ⚠ THE CONTROL THAT MAKES THE TREND READABLE — non-negotiable

1.5B was measured **on this machine, on CPU, in fp32**. 7B and 14B would be measured **on Kaggle, on T4
GPUs, in fp16**. Those differ in platform, in device, and in numerical precision.

> **A trend computed across those two conditions confounds SCALE with PLATFORM, and would be worthless.**
>
> **Mandatory: reproduce the 1.5B point on the Kaggle GPU path, with the same code, and show it matches the
> local CPU result.** That anchor is the control. **If the 1.5B GPU and CPU numbers disagree by more than
> `σ_seed = 0.005 BPB`, STOP and report the discrepancy — do not proceed to the larger sizes**, because the
> instrument would then be measuring the platform.

Report fp16-vs-fp32 as a stated, quantified difference at 1.5B, not as an assumption.

### A1.3 Sizes, and the memory arithmetic — read before choosing

Kaggle T4×2 is **2 × 16 GB**, no NVLink, fp16 native (Turing — **no bf16**, see the project's training-infra
note).

| donor | fp16 weights | fits? |
|---|---|---|
| Qwen2.5-0.5B | ~1.0 GB | trivially |
| **Qwen2.5-1.5B** | ~3.1 GB | trivially — **this is the anchor, run it first** |
| **Qwen2.5-7B** | ~15.2 GB | **yes**, `device_map` across both cards |
| Qwen2.5-14B | ~29.6 GB | tight across both cards; **see the warning below** |

> **Do NOT reach for 8-bit or 4-bit quantisation to make 14B fit.** This probe measures **activation
> statistics**, and quantising the weights changes the activations we are measuring. That would confound the
> result with the very axis under test. **If 14B does not fit in fp16, run 7B and say 14B did not fit** —
> three points (0.5B, 1.5B, 7B) already establish a trend, and an honest gap beats a contaminated point.

**S1 scales far more easily than F1 does, and this is worth knowing:** S1's published rule is a **pure
per-token threshold with no fitting**, so it needs **no covariance accumulator**. F1's `H` is `L × [D,D]`
fp64 — 0.53 GB at 1.5B, but **~10 GB at 14B** (`D`=5120, `L`=48), which is what would actually break. Arms A,
C and D need only the intermediate vector; **only arm B carries a fitted θ**, and it fits a scalar per layer.

### A1.4 What to report

**The deliverable becomes a family of curves: BPB against ACHIEVED FFN sparsity, one curve per donor size,
on one set of axes.** All four arms at every size, at matched achieved sparsity, with paired standard errors
as already required.

Then the two questions that decide the programme:

1. **Does the achievable sparsity at a fixed BPB cost rise with donor size?** If it does, at what rate, and
   does an extrapolation to 26B clear the `α = 0.125` target the budget needs?
2. **Does the oracle bound move?** F1's `a` = 0.836 is currently the ceiling that closes the lever.
   **Compute the same oracle-on-`|h_i|` bound at every size.** If it falls materially with scale, F1's
   closure is size-local and must be reopened — **and that is the outcome most worth finding.**

### A1.5 Operational constraints

- **Kaggle budget is 30 GPU-h/week per account across 3 accounts**, and this is measurement, not training —
  no gradients, no optimiser, no distillation. **It touches no sealed constraint.**
- **The Owner launches long runs.** Per standing practice: the Builder smokes it, verifies it end-to-end at
  0.5B, **STOPs with a ready-to-run command and a stated GPU-hour estimate**, and does not start the real
  run itself.
- Existing plumbing: `scripts/kaggle_ops.py` and `kaggle_run.py`. **Reuse them; do not invent a new
  launcher.** Note the known trap recorded in the project's Kaggle memory: **a global OAuth access token
  overrides `KAGGLE_CONFIG_DIR`**, which has cost this project a night before.
- Kaggle duplicates subprocess stdout 2–3×. **Do not count log lines as evidence; use logical invariants.**

### A1.6 What does not change

Everything in §2 through §5: the four arms, matched achieved sparsity, the four controls (C1 identity to
full precision, C2 planted, C3 null fires, C4 achieved-from-masks), `σ_seed = 0.005`, ACHIEVED never
requested, and §4's warning that **this probe measures whether the sparsity EXISTS, not whether our 48 KB
block granularity can exploit it** — D0 has already answered the second question, and a good curve here does
not revive the carve.
