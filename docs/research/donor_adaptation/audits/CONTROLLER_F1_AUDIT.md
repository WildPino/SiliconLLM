# Controller audit — F1, the gate-sign predictor

**Controller** · 2026-09-03 · branch `research/donor-adaptation` @ `e79f548`
**Target:** `docs/research/donor_adaptation/probes/F1_GATE_PREDICTOR.md`;
`docs/research/donor_adaptation/briefs/BRIEF_F1_GATE_SIGN_PREDICTOR.md` incl. Amendment 1;
`benchmarks/donor_adaptation/f1/f1_gate_predictor.py`, `f1_tables.py`;
`benchmarks/donor_adaptation/f1/results/*`; cache `D:\_THINGS\_scratch_f1` (84 files, 1,585,457,664 B).

> **Verdict: the closure survives, but four of the report's own summary statistics are wrong and all
> four err in the same direction — the direction that widens the negative.**
>
> The apparatus is clean. The randomised-RNG defect that destroyed D0's error bars is **not** present
> here: F1 uses no `svd_lowrank`, no randomised SVD, no torch RNG at all, and a double re-run is
> bit-identical. Every per-layer table I checked reproduces exactly from the artefacts.
> **What is present is the prose signature — now seen in D0, in P1, and here, three probes running.**
>
> **7 BLOCK · 9 FLAG · 6 PASS.** Nothing found reopens the lever. Everything found changes what may be
> said about it — which matters more now that a multi-size sweep will be scored against F1's bound.

**Scratch scripts** referenced below live in this session's scratchpad
(`.../1f268f4b-.../scratchpad/`): `chk_json.py`, `chk_means.py`, `chk_oracle.py`, `chk_eps.py`,
`chk_eps_ladder.py`, `chk_ann.py`, `chk_repro.py`, `chk_rest.py`. Each is self-contained and reads only
the pinned safetensors snapshot plus `D:\_THINGS\_scratch_f1`.

---

## 1. BLOCK-1 — **`ε` is referenced to the FFN sublayer's own output. On 23 of 28 layers that object is 4–14% of the residual stream, so the reported "1% budget" is a 0.04–0.14% budget on what the next layer sees.**

*Claim 3, and the finding the coordinator asked me to lead with.*

**What the code does.** `theta_rule` (`f1_gate_predictor.py:258`):

```python
h = (silu(g) * u).astype(np.float32)
Y = h @ Wd.T
denom = float(np.linalg.norm(Y))
def rel_err(th):
    m = (g > th)
    drop = np.where(m, np.float32(0.0), h)
    return float(np.linalg.norm(drop @ Wd.T) / denom), float(m.mean())
```

So `ε` is `‖ΔY‖_F / ‖Y‖_F` with `Y = h W_dᵀ` — **post-`down_proj`, PRE-residual**, pooled Frobenius over
the 1024 calibration tokens. That is the narrowest object in the layer.

**The report describes this accurately** (§1.1). It is not mis-stated. It is **un-interrogated, and it
was never pre-registered**: the brief asks only *"State how that threshold is defined and print it as
ACHIEVED"* (§3.1), and defines no `ε` at all. **The entire `a(ε)` ladder rests on a discretionary Builder
choice, and it is the choice that sets `a = 0.95`.**

**I measured the reference object.** One forward pass, same donor, same 2048 calibration tokens, hooking
`layers[l].post_attention_layernorm` (forward-pre) for the residual stream the FFN writes into, and
`layers[l].mlp` (forward) for `Y`:

| layers | `‖Y‖_F / ‖residual‖_F` | ε = 0.01 on `Y` is, of the residual stream |
|---|---|---|
| **3 – 25 (23 layers)** | **0.041 – 0.139** | **0.00041 – 0.00139** |
| 0 | 1.015 | 0.0101 |
| **1** | **6.941** | 0.0694 |
| 2 | 1.174 | 0.0117 |
| 26 / 27 | 0.805 / 0.703 | 0.0081 / 0.0070 |

**On 23 of 28 layers the bar is 7–25× stricter than "1% of the layer" reads.**

**And I measured what a residual-referenced budget gives.** Same instrument, `ε_ffn = 0.01 / ratio_l`
("cost at most 1% of the residual stream"), 256 calibration tokens, 20 bisection steps:

| layer | `a` at ε = 0.01 on `Y` (published bar) | **`a` at 1% of the residual stream** |
|---|---|---|
| 0 | 0.9995 | 0.9995 |
| **3** | 0.9190 | **0.3257** |
| 6 | 0.9986 | 0.9267 |
| 12 | 0.9988 | 0.9391 |
| 18 | 0.9985 | 0.9452 |
| 24 | 0.9962 | 0.9554 |
| 26 | 0.9294 | 0.9023 |
| 27 | 0.9454 | 0.9243 |
| **mean (these 8)** | **0.9732** | **0.8648** |

> **The bar is genuinely stricter than it reads, and correcting it moves `a` from ≈0.97 to ≈0.86 on this
> sample, with layer 3 collapsing from 0.919 to 0.326.** That is a real move in the direction that
> reopens the lever. **It is not remotely enough: the target is 0.02.**

**Two things the Adapter must not conclude from this.**

1. **The ratio does not explain the ladder's depth structure — it runs the other way.** The coordinator
   asked specifically whether depth-varying ratio could account for the ladder on its own. It cannot.
   The two genuinely sparse layers (1, 2) are precisely the two where the FFN *dominates* the residual
   (ratio 6.94 and 1.17), i.e. where the reference object is **least** generous; the saturated mid-stack
   layers (ratio 0.04–0.14) are the ones the correction helps. **So "small `a` at layers 1–2" is a real
   property of those layers, not a measurement artefact**, and the correction moves the mid-stack, not
   the ends.
2. **1% of the residual stream, per layer, compounded over 28 layers, is not a demonstrated-safe bar
   either.** No `ε` on any reference object is validated against end-to-end quality. The report says so
   itself (§9.3) and is right that this is its largest gap. This finding narrows *which* `ε` to argue
   about; it does not settle it.

**Relevance to the multi-size sweep:** if the sweep scores `a` against F1's `ε`, it will be scoring
against a per-layer FFN-output budget whose severity relative to the residual stream **varies 170× across
depth in this donor alone** (0.041 to 6.94). A sweep across model sizes should fix the reference object
first, or report both.

*Repro:* `python <scratch>/chk_eps.py` (ratios and per-token energy; one forward pass, ~3 min), then
`python <scratch>/chk_eps_ladder.py` (residual-referenced ladder, 8 layers, ~2 min).

---

## 2. BLOCK-2 — **on layers 1, 2 and 26 the `ε` criterion is set by about eight tokens.**

`rel_err` is a **pooled** Frobenius ratio over tokens, so it is energy-weighted. From the same forward
pass, the share of `‖Y‖²_F` carried by the largest-norm tokens:

| layer | top-1 token | top-8 tokens | top-1% of tokens |
|---|---|---|---|
| **1** | **0.5061** | **0.9956** | 0.9958 |
| **2** | **0.5157** | **0.9938** | 0.9940 |
| **26** | **0.4957** | **0.9571** | 0.9581 |
| all others | 0.0012 – 0.0932 | 0.0089 – 0.2005 | 0.0210 – 0.2217 |

**One token carries half the FFN output energy on layers 1, 2 and 26; eight tokens carry over 95%.**
These are the massive-activation / attention-sink tokens the report itself invokes in §3 — and **they are
exactly the layers the report designates as the only ones with measurement room** (§2.2 "the seven layers
where the range exceeds 0.01"; §2.3 "there the instrument genuinely discriminates").

So on those layers `θ(ε)`, `a`, and every `S_frac` read against them are effectively **n ≈ 8**, not
n = 1024. The report presents them as load-bearing without disclosing that their tolerance is set by a
handful of outlier tokens. A per-token (or trimmed) relative error, or the same ladder with the top-8
tokens excluded, would settle whether layers 1–2's sparsity is a property of the layer or of the sinks.
**That question is directly on the critical path for the multi-size sweep**, because attention-sink
structure is itself known to change with scale.

*Repro:* `python <scratch>/chk_eps.py` — the `top1 / top8 / top1%` columns.

---

## 3. BLOCK-3 — **the oracle headline `a = 0.836` is layer 0's value, not the mean. The mean is 0.7984.**

*Claim 2 — the number closing the lever.*

`f1_analyse.json → layers[L].threshold_rule.oracle_magnitude["0.01"].oracle_active_frac`, all 28 layers:

```
0.8359 0.1328 0.1953 0.8164 0.8750 0.8125 0.8906 0.8828 0.8828 0.8789 0.8828 0.8828 0.8789 0.8750
0.8672 0.8711 0.8750 0.8750 0.8711 0.8633 0.8750 0.8789 0.8672 0.8672 0.8555 0.8555 0.5234 0.6875
```

**mean = 0.79841 · median = 0.8711 · layer 0 = 0.8359.**

The report states `0.836` as a mean **three times**: §0 ("still needs `a = 0.836` mean"), §1.2 ("Mean over
28 layers: … oracle-|h| `a(.01)` = 0.836"), §1.3 ("cannot get `FFN_active` below ≈ 0.84"). The per-layer
**table column is exact**; only the mean is wrong, and the wrong value is a verbatim copy of that table's
first data cell. Error **+0.0376**, in the direction that makes the ceiling look higher and the negative
stronger.

*Repro:* `python <scratch>/chk_means.py` — prints the column and its mean beside the report's claim.

---

## 4. BLOCK-4 — **three more `a`-ladder means are wrong. Four of four errors share a sign.**

| report §1.2 states | JSON gives | error | direction |
|---|---|---|---|
| `a(.001)` = **0.993** | **0.99207** | +0.0009 | widens the negative |
| `a(.01)` = **0.951** "(calib)" | **0.94713** | +0.0039 | widens the negative |
| `a(.05)` = **0.909** | **0.87881** | **+0.0302** | widens the negative |
| oracle `a(.01)` = **0.836** | **0.79841** | **+0.0376** | widens the negative |
| `a_eval(.01)` = 0.9506 | 0.950618 | exact | — |

**Four of the five summary means are wrong and all four err the same way.** The `a(.01)` "(calib)" row is
the *held-out* number reused under a calibration label; the true calibration mean is 0.9471.

The consequence for the prose is direct. §1.3 argues *"Loosening ε does not rescue the construction. At
ε = 0.05 — five times looser … `a` is still 0.909 mean."* **The true figure is 0.879**, and the same
sentence's "≥ 0.97 on 20 of 28 layers" is **21 of 28**. The argument stands; the numbers supporting it do
not.

This is the D0 and P1 failure signature reproduced a third time: **tables exact, prose over-stated, every
over-statement leaning the same way.** Per `feedback_exact_tables_overstating_prose`, every sentence in
this report should name the row it comes from before it is quoted again.

*Repro:* `python <scratch>/chk_means.py`.

---

## 5. BLOCK-5 — **`(1+3a)/3 = 1.284 > 1` double-charges `W_g`. The dense-gate baseline is 0.967, and the "pessimisation" headline is false.**

*Claim 4.*

`f1_gate_predictor.py:824`:

```python
rec["baseline_dense_gate_FFN_active"] = (1.0 + 3.0 * a_eval) / 3.0
```

The dense-gate path computes `g = W_g x` **at full width** — that is the entire content of the 1/3 floor
— and therefore **already holds `g` on `S`**. It does not re-read `W_g[S]`. Its cost is

```
FFN_active_dense_gate = (1·FD + a·FD + a·FD) / 3FD = (1 + 2a)/3
```

which is the identical convention this programme banked in `docs/ENGINE_PLAN.md:56` and quoted in the
brief's own §0: **gate 100% + up 21.5% + down 12.3% = 44.6%** — gate charged **once**. Both forms give
1/3 at `a = 0`, so the floor claim is untouched. At the measured `a`:

| | report | correct |
|---|---|---|
| dense-gate baseline at `a_eval = 0.950618` | **1.2840** | **0.9671** |

So §0 / §5.2 / §8.2's *"at the measured activation fraction the sparse path is a pessimisation — it costs
more than simply running the FFN dense"* is **false**: 0.967 < 1.

**This is the same defect the Builder correctly caught in the brief** (§5.3: the brief's `a = 0.02` row is
`n=2` while the rows above are `n=3`). F1 found the brief mixing `n`, then applied `n=3` to a baseline
whose gate term *is* the dense read. Same error, one fence over. It is also exactly the failure named in
`feedback_charged_vs_moved_bytes` — the wrong unit hiding in a denominator.

**The corrected statement is still negative and is sufficient:** dense-gate = 0.9671; `fit_actw` at r=64
conv-A = 0.9788. **The rank-64 predictor is 0.0117 worse than simply computing the gate densely.** Under
convention B the predictor path is 1.0022 > 1, so a pessimisation claim survives *there* — but only under
the ternary/fp16 convention, and the report attaches it to the baseline row instead.

*Repro:* `python -c "a=0.950618; print((1+2*a)/3, (1+3*a)/3)"` → `0.96708 1.28406`; cross-read
`docs/ENGINE_PLAN.md:56` and `BRIEF_F1_GATE_SIGN_PREDICTOR.md:12`.

---

## 6. BLOCK-6 — **C4 is a tautology of the same kind as C1, is not labelled one, and establishes no minimum detectable effect.**

*Claim 7.*

`f1_gate_predictor.py:882–898`: `Pp = qr(randn(F,32))[0]` (orthonormal `[F,32]`), `Wg_p = Pp @ randn(32,D)`
— **exactly rank 32**. Then `Gp = Xe @ Wg_p.T`, whose rows lie in `colspace(Pp)`, a 32-dimensional
subspace of `R^F`. `Up4 = left_subspace(Wg_p @ Hh, 256)` — its top-32 columns span that **same** subspace.
So `Gp Up4 Up4ᵀ ≡ Gp` at `r ≥ 32` **by the identical argument the report uses to label C1 a tautology**
(`U_D` spans `colspace(W_g)`). The reported `rel_dev` of 2.4e−07 – 4.7e−07 is fp32 round-off; `recall =
1.0000` at r = 32 on all 28 layers is forced, not observed.

C4 therefore demonstrates that `eigh(MᵀM)` recovers an exact column space — a numerical check worth
having, and it does traverse `recall_curve`. **It does not demonstrate detection sensitivity.** §6's
verdict *"the instrument demonstrably detects a genuinely low-rank gate. It did not detect one in the
donor because the donor does not have one"* claims more than the control supports. The donor is not
exactly rank-anything; the live question is whether the instrument would detect a gate that is
**approximately** rank-64 with a few-percent residual. **No control in F1 addresses that, so the probe
publishes no MDE.** Per `feedback_planted_controls`, a tautological positive does not discharge the
requirement that an instrument fire on a known positive before its nulls mean anything.

**And "recovers at the planted rank and NOT before" fails on the metric the science actually uses.**
`recall_at_tau_eq_theta` at **r = 16** — half the planted rank — is **≥ 0.9993 on 21 of 28 layers**
(max 1.0000); at **r = 8** it is ≥ 0.9996 on those same layers. The report quotes "r=16: 0.8495–0.9999",
which is the minimum and a rounded maximum. The "broken one rank below" claim holds only in `rel_dev`, an
L∞ reconstruction metric by which no arm is judged.

*Repro:*
```
python -c "import json;a=json.load(open('benchmarks/donor_adaptation/f1/results/f1_analyse.json'))['layers'];
print(sum(1 for l in range(28) if a[str(l)]['arms']['C4_planted']['by_rank']['16']['recall_at_tau_eq_theta']>0.999))"
```
→ `21`.

---

## 7. BLOCK-7 — **the ANN arm ran none of its registered controls, and the omission is not disclosed.**

*Claim 8.*

Amendment A1.4 is explicit: the ANN arm's curve is to be measured *"against the same **C2**
random-projection floor, **C3** wrong-layer, and **C4** planted-positive controls"*. `stage_ann()`
(`:948–1090`) implements **none of the three**; `f1_ann.json` contains no control key. §9's gap list
records that `nlist` was not swept (§9.6) but **does not record that the arm has no null and no positive
control.**

Two conclusions are nevertheless drawn from it: §8.4 *"Amendment 1's arm is the weaker one, not the
stronger one"* and §8.5 *"`nlist ∝ √N` … does not transfer to `W_g`'s rows. This is a second instance of a
transplant failing."* **Neither is supported by a controlled measurement.** An arm with no floor cannot be
said to sit below one.

*Repro:* read `f1_gate_predictor.py:948–1090`; `python -c "import json;d=json.load(open('benchmarks/donor_adaptation/f1/results/f1_ann.json'));print(sorted(d['layers']['0'].keys()))"`
— no C2/C3/C4 key exists.

---

## 8. PASS-1 — **the randomised-algorithm check (the START HERE item). F1 does not have D0's defect.**

D0's B3 — a rank-64 `torch.svd_lowrank` sketch drawing from an unseeded torch global RNG, with the
treatment sd since measured at 1.3× to 868× the null's — **has no analogue here.**

| check | finding |
|---|---|
| any `svd_lowrank` / randomised SVD in the fit path? | **No.** `left_subspace()` (`:216`) is deterministic: `np.linalg.eigh(MᵀM)` on a `[D,D]` fp64 Gram, then `U = M V / σ`. No sketch anywhere in F1. |
| any torch RNG? | **No.** `torch` is imported only to read safetensors and to run the single `capture` forward pass. `torch.manual_seed` is never called and is never needed. |
| all randomness seeded? | **Yes, and pre-registered.** Every draw is `np.random.default_rng(<constant>)`: `RNG_C2=20260901`, `RNG_C2B=20260902`, `RNG_C4=20260903`, `RNG_ANN=20260904`. C2/C2b are drawn **fresh per (layer, rank)** as `seed + 1000*l + r` (`:855`, `:864`); C4 as `RNG_C4 + l` (`:884`). All seeds are emitted into `f1_analyse.json["pre_registered"]["seeds"]`. |
| re-run twice — do the statistics move? | **No.** I re-ran layer 1's entire path (θ rule → `a_eval` → `fit_actw` r=64 → recall curve) twice in one process: `θ = −5.3926794210`, `a_eval = 0.3578738076`, `S_frac = 0.8293170929`, `oracle = 0.1328125000` — **bit-identical across trials.** |
| re-run vs. the archived JSON | agrees to **2.7 × 10⁻⁸** in `S_frac`, which is **exactly one boolean entry** of 36,700,160. BLAS reduction-order non-determinism, not RNG. The JSON's `Wg_sha16`, `H_sha16` and `X_eval_sha16` match the on-disk arrays exactly. |

**Conclusion: every F1 margin is quoted against the right dispersion, insofar as any dispersion is quoted
at all — and none is (FLAG-4).** One minor reproducibility defect is recorded as FLAG-7.

*Repro:* `python <scratch>/chk_repro.py`.

---

## 9. PASS-2 — **the oracle *is* an oracle. I tried to beat it and could not, by much.** (Claim 2's construction survives; only its arithmetic does not.)

The task asked whether it truly sees `W_u`, whether it is per-token, and whether a selection rule that
could beat it was left untried.

- **Sees `W_u`: yes.** It selects on `h = SiLU(g) ⊙ u`, with `u = X W_uᵀ` (`:283`, `:308`) — strictly
  stronger than anything the construction under test can read.
- **Per-token: yes.** `np.argsort(np.abs(ho), axis=1)` — each token drops *its own* smallest-`|h|`
  neurons. **The budget, however, is a fixed count `k = round(mid·F)` shared by every token.**
- **A rule it omits, and the omission is principled, not cosmetic.** Dropping neuron `i` costs
  `|h_i| · ‖W_d[:,i]‖`, not `|h_i|`. The column norms are **not** uniform: they span **2.4× (L0) to
  5.6× (L1)**. The oracle ignores them.

I tested both gaps directly:

| layer | F1 oracle (8-step) | oracle, 14-step | **`|h_i|·‖W_d[:,i]‖`** | **global threshold (variable per-token budget)** |
|---|---|---|---|---|
| 0 | 0.8359 | 0.8342 | 0.8341 | **0.8252** |
| 1 | 0.1328 | 0.1301 | 0.1282 | **0.1112** |
| 2 | 0.1953 | 0.1915 | 0.1867 | **0.1708** |
| 13 | 0.8750 | 0.8746 | 0.8737 | **0.8714** |
| 26 | 0.5234 | 0.5231 | 0.5219 | **0.5083** |
| 27 | 0.6875 | 0.6839 | 0.6860 | **0.6842** |

The `W_d`-norm correction buys **0.001 – 0.005**. Letting the per-token budget float buys
**0.011 – 0.020**. **Together they move the corrected mean ceiling from 0.798 to roughly 0.78.**

> **Net on Claim 2: the construction is sound, the reported number is not. The ceiling is ≈ 0.78–0.80,
> not 0.836. The lever does not reopen through the oracle** — 0.78 against a target of 0.02. The oracle
> remains a legitimate ceiling on any gate-only rule, to within about 0.02.

*Repro:* `python <scratch>/chk_oracle.py`.

---

## 10. PASS-3 — **Claim 1's held-out structure is real. The easiest way for the ladder to be wrong is not how it is wrong.**

Verified in code, not in prose:

- `Xt1 = Xt[:1024]` (calibration, seed 20260822, part `calib`, `corpus_sha256` `10d4d281…`) fits θ at
  `:812`.
- `G = (Xe @ Wg.T)` and `a_eval = (G > θ).mean()` come from `Xe` (part **`heldout`**, seed 20260828,
  `corpus_sha256` `f46b0310…`, `n_rejected: 0`) at `:818–821`.
- The two slices are **different corpus parts**, not two draws from one pool. `T_fit = 16384`,
  `T_eval = 4096`, `T_theta = 2048` stored / 1024 used — all as the report states.
- Bisection convergence checks out: `rel_err(.01)` lands in **0.009906 – 0.010000** across all 28 layers,
  so the report's "converged everywhere; a near-1.0 `a` is not a refinement failure" is correct.
- The 13-point quantile grid then `n_refine + 8 = 22` bisection steps is exactly what the code does
  (`:290`, `:340`).

**A θ fitted and reported on the same tokens would have inflated `a` in the safe direction. That did not
happen.** Claim 1's structure is the strongest part of the probe.

*Repro:* `f1_gate_predictor.py:812–821`; `python <scratch>/chk_json.py`.

---

## 11. PASS-4 — **Claim 5 (`eff_rank`) is correct, and the two quantities are computed the same way.**

Both are the participation ratio of a **singular**-value vector, `PR(σ) = (Σσ²)² / Σσ⁴`:

- `eff_rank_actw` = PR(σ(`W_g H^{1/2}`)) and `eff_rank_plain` = PR(σ(`W_g`)), both from
  `spectrum_facts()` on the `sig_all` returned by `left_subspace` (`:745`).
- `eff_rank(H^{1/2})` = PR(√eig(`H`)) — the singular values of `H^{1/2}`. **Same functional, same object
  class.** I re-derived it independently from the cached Gram matrices: L0 20.12, L1 1.97, L13 21.86,
  L27 6.92 — matches §3 exactly, as do `eig₁/eig₂` and `eig₁/tr(H)`.

Every §3 column reproduces: `W_g` 99%-energy rank **mean 1452.5**, range 1288–1471 (report: 1452);
`r=64` plain energy captured **mean 0.1428**, range 0.1172–0.1787 (report: 0.1428, 0.117–0.179); `actw`
99%-energy rank **342–1148** (report: 342–1148); condition number **16.1–65.7** (report: 16–66).

**The verdict — that `eff_rank_actw ≈ 2` is inherited from `H` and must not be quoted as evidence that a
rank-2 gate predictor exists — is correct and well-supported, and is the strongest reasoning in the
report.** It is also the one section whose conclusion I could not weaken.

*Repro:* `python <scratch>/chk_repro.py` (the `eff_rank` block).

---

## 12. PASS-5 — **which controls can actually fail.** (Claim 7, the rest.)

| control | can it fail? | verdict |
|---|---|---|
| **C1** full-rank | **No.** `U_D` spans `colspace(W_g)`. | Tautology, **correctly labelled** in both the JSON (`IS_A_TAUTOLOGY`) and §6. Reproduces: max rel dev 1.44e−06, recall 1.000000, `S_frac@1.0` = 0.950618 = `a_eval` to 6 d.p. |
| **C2** random `B`, `A` solved | **Yes** — a different construction (a rank-`r` read of `x`, not a projection of `g`), fresh per (layer, rank). It discriminated: beats `fit_plain`, loses to `fit_actw` by 0.08–0.12 on layers 1–2. | **Real control, fired.** |
| **C2b** random output subspace | **Yes** — same family and cost, unfitted. Sits at ρ = 0.99 and is **flat in `r`** (0.9875–0.9894 across all six ranks on layers 1–2) — what a null should do, and not guaranteed. | **Real control, fired.** Builder addition, and the right one: without it a win over C2 could have been a win of family rather than of fit. |
| **C3** wrong layer, both weightings | **Yes.** Indistinguishable from C2b (0.9899 vs 0.9898 mean at r=64). | **Real control, fired.** |
| **C4** planted | **No** — see BLOCK-6. | **Tautology, not labelled.** |

The `selfcheck` algebra is genuinely strong and I confirm all of it: Eckart–Young rel err 0.0 (k=8) /
7.88e−15 (k=32); orthonormality 2.00e−15 / 2.22e−15; `H^{1/2}` roundtrip 1.68e−15; direct-SVD alignment
3.92e−09 (synthetic) and **1.62e−09** (layer 0, real data); **200/200 random rank-`k` factorisations
worse at both k = 8 and k = 32**. Refusing shuffled-`W_g` nulls, per the brief, is correct.

---

## 13. PASS-6 — **the two byte conventions are internally consistent and applied identically to every arm.** (Claim 4, the part that holds.)

Checked against `feedback_charged_vs_moved_bytes`:

- **Convention A:** `FFN_active = pred/3 + s`, `pred = r(D+F)/(D·F)`. Reproduced at all six ranks —
  `0.9809 / 0.9801 / 0.9793 / 0.9788 / 0.9790 / 0.9809` — matching §5.2 exactly.
- **Convention B:** `W_*` at 0.25 B/elem ternary, `A,B` at fp16 ⇒ `(8/3)·pred + s`. The algebra
  `2/(3·0.25) = 8/3` is correct. Reproduced — `0.9838 / 0.9860 / 0.9911 / 1.0022 / 1.0258 / 1.0746` —
  matching §5.2 exactly.
- **Applied identically:** `ffn_active()` (`:456`) is the single call site for every arm, and
  `FFN_active_n2` is emitted beside `FFN_active` for every arm at every rank and every recall target.
- §5.1's ANN claim is also right: `f1_ann.json`'s `bytes_frac` uses the same 0.25 / fp16 / 1 B convention
  (`index_cost`, `:1000`), so §4's BYTES columns and §5.2's conv-B column are the same convention.
- Every §2.4 and §5.2 cell reproduces from the JSON to 4 d.p.

**The one accounting error is BLOCK-5 — the baseline row, not the arms.**

*Minor, recorded here:* §5's "best `FFN_active` reached anywhere (any arm/rank/layer)" = 0.7384
**excludes C1**, which is an arm in §2.1's own table; including it, layer 1 gives **0.4347**
(`r = D = 1536` charged at target width). Excluding a tautology is defensible; writing "any arm" is not.

---

## 14. FLAG-1 — the ANN failure has **two** independent uncorrected defects; the Euclidean-k-means error alone does **not** account for it.

The report names the MIPS/L2 mismatch (§4 reason 2) and it is real: `cent, assign = kmeans(K, …)` is
Euclidean on un-normalised `W_g` rows (`:933`, `:1027`) while the query is `coarse = Xe64 @ cent.T`
(`:1055`), an inner product. That defect governs **reachability** at low `nprobe`.

**But at `nprobe = nlist = 95` every list is probed, the router is bypassed entirely, and the arm still
fails.** Layer 1, where `a = 0.3579` and the perfect floor is 0.354:

| `nprobe` | reach | recall | `S_frac` |
|---|---|---|---|
| 64 | 0.9451 | 0.9720 | 0.9451 |
| **95 (= full scan)** | **1.0000** | 0.9900 | **0.9229** |

With every key reachable, hitting recall 0.99 still requires keeping **92%** of neurons on a layer where
only 36% are active. That residual is **pure PQ score error** (`pq_reconstruction_rel_err` mean 0.8257,
range 0.8162–0.8299) — 48 × 8-bit subquantizers over a 1536-dim fp32 key is 128× compression on a
near-isotropic point set. Layer 2 behaves identically (0.9437 at `a` = 0.4796).

> **A correctly-built MIPS coarse quantizer would fix reachability and would not fix the ranking. Both
> defects must be corrected before the arm says anything about IVF-PQ over `W_g`. Neither was tested
> corrected, and no MIPS index was ever built.** The arm's negative is a statement about *this index*,
> not about the construction Amendment 1 registered.

In the arm's partial defence, the diagnosis is right for the right reason at one point: with the
partition collapsed (largest list mean 4494 / 8960, max 8842; smallest 1.0), the PQ residual
`K − cent[assign]` is nearly `K − mean`, so the router defect **feeds** the PQ defect. They are coupled,
not independent — one more reason a corrected run is needed before the transplant is declared dead.

**What would settle it:** rebuild with normalised keys (or an anisotropic/ScaNN-style quantizer) plus a
larger `m` or OPQ rotation, and re-run `nprobe`. Cheap — `ann` cost 1442 s for all 28 layers.

*Repro:* `python <scratch>/chk_ann.py` — per-layer `by_nprobe` for layers 0, 1, 2, 26, 27.

**Sub-finding, cosmetic, no published number affected:** `stage_ann` calls
`recall_curve(Gh, true_mask, 0, D, F)` with `r = 0`, then overwrites the cost fields **only** on
`at_recall_target` entries (`:1063–1071`). The `curve` list therefore keeps `FFN_active` computed with a
zero-cost predictor. Nothing in the report reads `curve`, but a future reader of `f1_ann.json` will find
free-looking index costs there.

---

## 15. FLAG-2 — the Spearman −0.836 is **definitionally coupled**. It reproduces exactly; it is not independent evidence. The depth confound *was* tested and is ruled out.

*Claim 6.*

`skill = (uninformative − fit_actw) / (uninformative − 0.99a)` and `RANGE = uninformative − 0.99a`. The
statistic correlates `X/Y` against `Y`, with `range` in the denominator of `skill`. A negative
coefficient is close to forced by the definition.

I reproduced it, and tested the confound the task named:

| statistic | value |
|---|---|
| Spearman(range, skill) | **−0.83580, p = 3.100e−08** — matches the report exactly |
| Spearman(depth, range) | +0.117, p = 0.55 |
| Spearman(depth, skill) | +0.049, p = 0.80 |
| partial Spearman(range, skill \| depth) | −0.848, p = 1.2e−08 |

**So it is not a depth artefact** — the report's reading survives that specific challenge, and the
alternative explanation the task asked me to test is refuted. But the p-value must not be quoted as if
`range` and `skill` were separately measured. The substantive point — that a skill of 0.97 across a range
of 0.001 moves `S_frac` from 0.9900 to 0.9895 and is worth nothing — is carried entirely by the table,
and the table is exact. **Drop the statistic, keep the table.**

*Repro:* `python <scratch>/chk_json.py`.

---

## 16. FLAG-3 — the oracle is not held out, and its bisection is quantised and biased upward.

1. **The oracle is computed on 512 *calibration* tokens** (`n_tok_oracle = 512`, `:257`), while `a_eval`
   is 4096 held-out tokens from a different corpus part. §1.2 prints them as adjacent columns and §1.3
   compares them directly ("the gap between 0.836 and 0.951") without stating that they sit on different
   token sets, of different sizes, from different corpus parts.
2. **The oracle bisects only 8 steps over [0,1]**, so `oracle_active_frac` is quantised to 1/256 — every
   published value is an exact multiple of 0.00390625 (0.8359375, 0.1328125, 0.875, …).
3. **The bisection returns the last *feasible* `lo`**, so the drop fraction is under-estimated and the
   active fraction **over-estimated**. Re-running at 14 steps moves every layer down: L0 0.8359 → 0.8342,
   L1 0.1328 → 0.1301, L2 0.1953 → 0.1915, L26 0.5234 → 0.5231, L27 0.6875 → 0.6839. Small, and in the
   same direction as every other error in this report.

*Repro:* `python <scratch>/chk_oracle.py`.

---

## 17. FLAG-4 — **every number in F1 is n = 1, and `a` is sensitive at 10–40× the scale the report argues over.**

There is **no replication anywhere**: one calibration slice (16 × 1024, seed 20260822), one held-out
slice (4 × 1024, seed 20260828), one token count, one run per layer. No error bar on `a`, on `S_frac`, or
on `skill` exists in any artefact. §2.2 correctly identifies *quantisation* (2.7e−08) and *dynamic range*
as two resolutions — but **never estimates sampling dispersion**, which is the one that binds.

I re-derived `a(.01)` with 256 calibration tokens instead of 1024, same instrument, same θ rule:

| layer | `a(.01)` @ 1024 tok (published) | @ 256 tok | Δ |
|---|---|---|---|
| 26 | 0.9737 | 0.9294 | **0.0443** |
| 3 | 0.9393 | 0.9190 | **0.0203** |
| 27 | 0.9559 | 0.9454 | 0.0105 |
| 24 | 0.9970 | 0.9962 | 0.0008 |
| 0, 6, 12, 18 | — | — | < 0.0003 |

**On exactly the layers with dynamic range, `a` moves 0.01–0.04 with the calibration token count** —
10–40× the 0.001 separations §2.2 spends its argument on, and comparable to the 0.030 arithmetic error in
BLOCK-4. Part of this is BLOCK-2's few-token energy concentration.

**No margin in this probe should be read as significant until a dispersion for `a` exists** — and this
becomes load-bearing the moment F1's bound is used to score a multi-size sweep, because a cross-size
difference smaller than 0.04 would be unreadable. **What would settle it:** two more calibration slices
at different seeds. One `capture` re-run, ~380 s each.

*Repro:* `python <scratch>/chk_eps_ladder.py` (the `a(.01)` column) against `f1_analyse.json`.

---

## 18. FLAG-5 — the operating point `τ` is tuned **on the eval set**. Comparisons are fair; absolute numbers are optimistic.

`recall_curve` (`:439`) picks `τ` from the `(1−ρ)` quantile of the predictor's scores **at the true
actives on the eval tokens** — knowledge no deployed engine has. It is applied identically to every arm
and every null, so §2's *comparisons* are sound. But every absolute `FFN_active` in the report (0.9788,
0.7384, the whole §5.2 column) is measured at an oracle-selected threshold.

**Direction: this flatters the construction, so it makes the negative conservative.** Recorded for
completeness, not as a defect. A calibrated `τ` (fitted on `X_theta`, applied to `X_eval`) would be the
honest operating point and would only move `S_frac` up.

---

## 19. FLAG-6 — three further prose-vs-table over-statements, checked the way the P1 audit checked them.

The coordinator asked specifically for F1's prose against F1's own tables. Beyond BLOCK-3 and BLOCK-4:

1. **§1.3, θ = 0:** *"It gives `a` = 0.13–0.30 … at a relative output error of 0.41 to 0.88 on most
   layers. That is not a tolerance; that is destroying the layer."* The true `a` range at θ = 0 is
   **0.0015 – 0.2992**, and on layers 1 and 2 the sign rule gives **`a` = 0.0015 and 0.0069 at a relative
   error of 0.052 and 0.052 — 0.15% of neurons for 5% error.** The sentence is true of 22 layers and
   **false of the two layers the report itself nominates as the sparse ones**; the quoted range silently
   excludes six layers (1, 2, 3, 4, 5 at the bottom, 27 at the top). Both the range and the verdict are
   contradicted by the report's own §1.2 table, columns `a at θ=0` and `rel-err at θ=0`.
2. **§0 and §1.2, "On 24 of 28 layers `a ≥ 0.95`."** That count is on the **calibration** `a(.01)`; the
   sentence's subject is the held-out 0.9506, on which it is **26 of 28**. This is the single
   over-statement that leans *against* the negative — worth noting precisely because it is the exception
   that shows the others are not random.
3. **§6, C4:** "recall at τ=θ: r=16: 0.8495–0.9999" — the maximum is 1.0000, and 21 of 28 layers exceed
   0.9993 at r = 16. See BLOCK-6.

*Repro:* `python <scratch>/chk_means.py` and `python <scratch>/chk_rest.py`.

---

## 20. FLAG-7 — the manifest publishes the reassuring floor count and omits the achieved one.

§11 states *"`H_ridge_rel` 1e−10 (**0 eigenvalues floored on the selfcheck `H`**)"*. That is the
**synthetic** `H` from `stage_selfcheck`. On the **real** per-layer `H`,
`f1_analyse.json → spectra[L].H_diag.n_eigs_floored_achieved` is:

```
17  1  2  2  2  1  2  1  1  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1  2  2  0  2
```

— 1 to 17 eigenvalues clipped per layer (17 at layer 0), with `cond_after_floor_achieved` = **exactly
1e10 on all 28 layers**, i.e. the floor binds everywhere and `H` is numerically rank-deficient at every
layer. This changes no conclusion — `H^{1/2}` is used only to *choose* a subspace and is never inverted,
which is a genuinely good design decision the report is right to highlight — but in a document otherwise
scrupulous about "ACHIEVED, not requested", the manifest prints the number that reassures and omits the
number that was achieved.

*Repro:* `python <scratch>/chk_rest.py` (the `n_eigs floored` block).

---

## 21. FLAG-8 — reproducibility and provenance.

**RNG, minor defect:** `stage_ann` creates `rng = np.random.default_rng(RNG_ANN)` **once, outside the
layer loop** (`:959`) and consumes it sequentially across 28 layers × (1 coarse + 48 PQ) k-means runs. The
full run is deterministic, but **`python f1_gate_predictor.py ann 5` does not reproduce layer 5 of the
full run.** `stage_analyse` does not have this problem — its C2/C2b/C4 seeds are `base + f(l, r)`.

**What the manifest pins, and it is above this programme's bar:** `repo_id`, `revision`, `config_sha256`,
`safetensors_bytes`, achieved `n_layers` / `D` / `F` / `stored_dtype` (read off tensor headers, not config
prose), both slices' `ids_sha256` + `offsets_sha256` + `corpus_sha256` with `n_rejected: 0`, every
pre-registered constant, every seed, a full env block per stage, and per-layer `Wg_sha16` / `H_sha16` /
`X_eval_sha16`. I verified those three against the on-disk arrays for layer 1 — all match.

**What it does not pin:**

- **`f1_analyse.json` is untracked** (11.7 MB, `??` in git status) while `f1_ann.json` (2.2 MB) **is**
  tracked. Every number in §1, §2, §3, §5 and §6 lives only in the untracked file. **Nothing in the
  report's core is reproducible from the repo alone today.**
- The **§3 `H`-spectrum columns were produced by a snippet pasted into §11**, not by the instrument. The
  snippet is not in the repo, not hashed, and not re-run by any stage. (I re-derived it independently and
  it is correct — PASS-4 — but it sits outside the artefact chain.)
- `safetensors` is pinned by **byte length only**, not by content hash.
- The 1.5 GB cache lives on `D:\` outside the repo. Correct decision; but `analyse` cannot be re-run from
  a clone without re-running `capture`.

**Machine contamination:** `capture` and `analyse` ran with the Owner's `llama-server.exe` resident at
~18 GB; `ann` ran clear. **This affects the elapsed times in §11 and nothing else** — no F1 number is
timing-derived, and the report already refuses to draw any performance reading from them. Peak RSS is
not instrumented (report's own §9.9); the allocation inventory is verified by shapes and dtypes only.
**PASS on contamination.** My own re-derivations ran alongside other work; none of my findings is
timing-based, and all numeric findings were re-derived from the pinned cache and pinned weights.

---

## 22. What this audit does **not** overturn

- **Claim 1's held-out structure.** Verified in code (PASS-3).
- **The `a(ε)` per-layer table** — all 28 × 3 cells, plus `a_eval`, `θ`, `rel_err`, the `a at θ=0` and
  `rel-err at θ=0` columns, and the oracle column. Exact.
- **§2.2, §2.3, §2.4, §3, §4, §5.2, §6 — every table.** Reproduced from `f1_analyse.json` /
  `f1_ann.json` to the last digit, including the full ANN table (reach / ceiling / recall / `S_frac` /
  MACs / BYTES at all eight `nprobe`), the layer-1 and layer-2 rank sweeps, the per-token recall minima
  (L1 0.9479, L2 0.9503, L15 0.9783, L27 0.9300), the recall-0.999 row, and the C1/C2/C2b/C3 verdicts.
- **The `eff_rank` verdict** (PASS-4) — the report's strongest reasoning, and I could not weaken it.
- **The Builder's finding on the brief's `n=2` / `n=3` mixing** — correct; I reproduced all three rows
  (0.10335 / 0.05335 / 0.02335 under `n=3`; the brief's 0.0167 row is `n=2`).
- **The core negative.** With every correction in this audit applied — oracle 0.798, or ≈0.78 under a
  stronger selection rule; `a` from 0.951 to ≈0.865 at a residual-referenced 1% budget; dense-gate
  baseline from 1.284 to 0.967 — **the target is 0.02 and the closest measured number anywhere is
  0.7384.** Nothing here reopens the lever at 1.5B.

**One scope note the multi-size sweep depends on.** F1's §9.4 already states that `a` is measured on
Qwen2.5-1.5B only and that every "target width" column is geometry substituted into a formula rather
than a measurement. That disclosure is honest and it is now the operative one: **this audit finds no
reason to trust or distrust F1's bound at other scales, because F1 measured one donor.** What the audit
does establish is that the bound's *definition* (BLOCK-1), its *token base* (BLOCK-2), and its
*dispersion* (FLAG-4) must be fixed before a cross-size comparison can be read at all.

---

## 23. What must change before F1's numbers are quoted again

| # | severity | required change |
|---|---|---|
| 1 | **BLOCK** | State that `ε` is referenced to the FFN sublayer output, which is 4–14% of the residual stream on 23 of 28 layers, and report the residual-referenced ladder (`a` ≈ 0.865). Mark the reference object as a Builder choice, not a registered one. Fix the reference object before the multi-size sweep, or report both. |
| 2 | **BLOCK** | Disclose that on layers 1, 2 and 26 — the "layers with measurement room" — eight tokens carry over 95% of the `ε` denominator. |
| 3 | **BLOCK** | Correct the oracle mean to **0.7984** (or ≈0.78 with a stronger rule) in all three places. It is not 0.836; 0.836 is layer 0. |
| 4 | **BLOCK** | Correct `a(.05)` → **0.879**, `a(.01)` calib → **0.9471**, `a(.001)` → **0.992**; stop labelling `a_eval` as calib; "20 of 28" → 21. |
| 5 | **BLOCK** | Withdraw `(1+3a)/3 = 1.284` and the "pessimisation vs dense" headline. Dense-gate is **(1+2a)/3 = 0.967**, per `ENGINE_PLAN.md:56`. The surviving statement — *the r=64 predictor is 0.0117 worse than computing the gate densely* — is sufficient. |
| 6 | **BLOCK** | Label C4 a tautology, withdraw "the instrument demonstrably detects a genuinely low-rank gate", and state that F1 publishes **no MDE**. |
| 7 | **BLOCK** | Disclose in §9 that the ANN arm ran none of A1.4's three registered controls, and withdraw §8.4 / §8.5's two conclusions until it does. |
| 8 | FLAG | Say that the Euclidean-k-means error does not account for the ANN failure — the PQ error survives a full scan — and that no MIPS index was ever built. |
| 9 | FLAG | Drop the Spearman p-value (definitionally coupled); keep the table. Record that the depth confound was tested and ruled out. |
| 10 | FLAG | Publish a dispersion for `a`. Two more calibration slices; one `capture` re-run each. Load-bearing for the multi-size sweep. |
| 11 | FLAG | State the oracle's token base (512 calib, not 4096 held-out) and its 1/256 quantisation. |
| 12 | FLAG | Fix §1.3's θ=0 range and verdict, the "24 of 28" count, and the C4 r=16 range. Report the real per-layer `H`-floor counts. Track `f1_analyse.json` or say why not. |

**STOP.** No fixes implemented, nothing committed, nothing pushed.
