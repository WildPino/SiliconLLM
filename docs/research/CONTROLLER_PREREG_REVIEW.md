# Controller review — stage −1 pre-registration (PRE-RUN pass, §7.2)

**Target:** `docs/research/DONOR_ADAPTATION_FIRST_RESPONSE.md` §3 (stage −1 PTQ ternary kill gate),
with §5 read for context.
**Governing mandate:** `docs/prompts/master_prompts/DONOR_MODEL_ADAPTATION.md` §6 (working laws),
§9 (stage −1), S1, S4, §8.L, §8.N.
**Also read:** `docs/CANONICAL_EVAL.md` (local standard for pinning a slice),
`docs/research/DONOR_PRIOR_ART_SURVEY.md` (the literature the method slot rests on).
**Reviewer status:** read-only. No code inspected because none exists yet — this is a protocol audit.
**Verdict: NO. This pre-registration may not proceed to the Owner for execution approval.**
7 BLOCKs, 6 FLAGs, 3 PASSes.

---

## BLOCKS

### B1 — The eval slice does not exist, and the gate's *width* is a free parameter of it

> §3.6: "Metric: **BPB** on a held-out general-text slice"

`grep -rn "held-out general" docs/` returns exactly one hit: that line. There is no general-text slice
anywhere in this repo. `CANONICAL_EVAL.md` pins the project's own slice with token-stream md5, n,
split rule, explicit val offset (29,451,460), window count, token count, metric formula, dtype and
harness, and states the reason: **"the val slice choice alone moves absolute BPB by ~0.04 — 20× the
typical ±0.002 gate width."** Stage −1 pins none of it.

This is not merely untidy. The gate is `converted ≤ donor × 1.10`, so **the absolute headroom is
`0.10 × donor_BPB`, and `donor_BPB` is set by the slice.** A slice on which the donor scores 0.75
grants 0.075 BPB of damage; a harder or more out-of-domain slice on which it scores 1.5 grants 0.150 —
twice the permitted degradation, for free, chosen after the apparatus exists. **A motivated person
satisfies this gate by choosing the eval corpus, without touching the method.** That is the exact
post-hoc reinterpretation law §6.1 forbids ("A gate interpreted after the result is not a gate").

**Required before the run**, committed and pushed (§7.2 push-before-run), to CANONICAL_EVAL.md's
standard: corpus name + version/revision + licence; the exact document/byte range or record ids;
sha256 of the raw byte stream; total evaluated bytes and total evaluated tokens; UTF-8 declared;
tokenizer repo **and revision SHA**; context length and **stride** (non-overlapping vs sliding — this
alone moves BPB materially and is unstated); BOS/special-token handling; whether the final partial
window is scored; the loss reduction (sum-CE-bits / sum-target-bytes, as CANONICAL_EVAL does);
and the exact decontamination procedure against the calibration set (n-gram order, threshold,
direction). Additionally, **pin `donor_BPB` itself as a number in the pre-registration before the
converted arms run** — the gate must be a fixed absolute BPB, not a formula evaluated later.

### B2 — A multiplicative gate on a log-scale metric is the wrong shape, and it is ~3x looser than the frontier it is implicitly benchmarked against

> §3.6: "**GATE: stage −1 PASSES at a given bit-width iff converted BPB ≤ donor BPB × 1.10.**"

BPB is already `−log2 p` per byte. The honest scale-free quantity is the **additive excess**,
`ΔBPB = BPB_conv − BPB_donor`, which *is* the extra bits per byte the conversion costs — a direct
information-theoretic price, invariant to how hard the slice is. A **ratio** on a log-scale metric
means "the permitted damage scales with how badly the donor already does", which is backwards: the
worse the baseline, the more damage is licensed. It also makes B1's slice-choice lever operate.

Convert to the units the literature actually reports, which the Principal's own §2.7 quotes
(PT²-LLM 1.04–1.08× **perplexity**) [D from L?]:

- Gate at donor BPB 0.75 ⇒ `ΔBPB = 0.075` bits/byte.
- With `B` bytes/token, `Δ per token = 0.075·B` bits ⇒ `PPL ratio = 2^(0.075·B)`.
- At `B = 3.5 … 4.5` (English, Qwen2.5-class BPE) [?] ⇒ **PPL ratio 1.20 … 1.26×**.
- Inverting: PT²-LLM's claimed **1.08× PPL** corresponds to `ΔBPB ≈ log2(1.08)/4.0 = 0.0278` ⇒
  **a BPB ratio of ≈ 1.037×**.

**The gate is ~2.7× more permissive than the published claim of the very method it intends to run.**
A method that underperforms its own paper by nearly threefold still PASSES. §3.6 asserts "if it FAILS,
the path is very probably dead" — but with this width the converse is nearly vacuous: the gate cannot
distinguish "rotation-PTQ ternary works" from "rotation-PTQ ternary half-works", and stage −1's stated
purpose is exactly that distinction.

**Required:** restate the gate additively and absolutely — `BPB_conv ≤ BPB_donor + Δ*`, with `Δ*` a
fixed number in bits/byte, **derived and shown**, together with (a) the ratio it implies at the pinned
`BPB_donor`, and (b) the PPL ratio it implies at the measured bytes/token, so it can be compared to
the literature it will be read against. If the Architect still wants a permissive kill gate, that is
legitimate — but the permissiveness must be stated in bits and defended, not smuggled in by a
percentage on a logarithm.

### B3 — Control 1 has no anchor for the sealed donor. As written it cannot be run at all.

> §3.7.1: "**Reproduce a known-positive.** At 4-bit, land within a stated tolerance of the published
> figure for this method/model."

Three defects, any one fatal:

1. **The tolerance is not stated.** An unstated tolerance is decided after the number appears. §6.1.
2. **The published figure does not exist for this pairing.** `DONOR_PRIOR_ART_SURVEY.md:57-61` records
   PT²-LLM's table as **LLaMA-2-7B / 13B / 70B, WikiText2 perplexity, at ternary** — there is no
   Qwen2.5-1.5B row and no 4-bit row. QuaRot has 4-bit numbers, but QuaRot is a different method and
   `:28` notes it reports **no ternary result at all**. §3.2 seals the donor as `Qwen/Qwen2.5-1.5B`
   because it is "the most heavily represented model class in the rotation-PTQ literature" — the
   survey does not support that for this method family, which is LLaMA-2-heavy.
   **§3.2 and §3.7.1 are in direct conflict and one of them must move.**
3. **Even if an anchor existed, it is on the wrong harness.** The published number is WikiText2 PPL at
   a fixed seqlen; the gate number is BPB on an unnamed slice. Control 1 would then validate a
   measurement path that **is not the one producing the gate number** — it can pass while the BPB
   harness is silently broken. This is precisely §6.8 ("parity is end-to-end or it is nothing").

**Required:** (a) reproduce the anchor on the anchor's own model, harness, dataset, seqlen and
calibration-sample count, with a numeric tolerance pre-stated; **and** (b) a separate bridge check
that the project's BPB harness and the published PPL harness agree on the *unquantized* donor, so the
known-positive actually certifies the instrument that reads the gate. If (a) is impossible for the
sealed donor, either the donor or the method must change — before the run, as a stage-0 finding.

### B4 — No σ. The gate has no INCONCLUSIVE band, and the one σ it cites is imported across apparatus.

> §3.6: "A donor near 0.75 BPB may inflate by +0.075 = **15 σ_seed**."
> §8.N.4 (mandate): "**Statistical resolution:** … Without it, a retention delta is uninterpretable …
> **do the equivalent for every metric you intend to decide with.**"

Stage −1 proposes **one BPB number per arm**. There is no repetition, no variance estimate, no
resampling.

The σ_seed citation is a category error and it is *the same error the Principal withdrew in §5.3*.
σ_seed = 0.005 was measured on **from-scratch training seeds at 8.3M**; cross-script reproducibility
0.004 was measured across **two training scripts**. Stage −1 trains nothing. Its actual variance
sources are entirely different and all unmeasured:

- the PTQ method's own stochasticity (rotation/Hadamard seed, GPTQ-style column ordering, clipping
  search) — known to grow sharply as bit-width drops, and **largest exactly at 2-bit and ternary**,
  the points that decide;
- the calibration draw (which sequences land in the 512);
- finite-slice sampling error on the eval slice.

Quoting σ_seed here reads as reassurance that the number cannot supply. §5.3 says a criterion built on
a non-transferable anchor is "an undemonstrated bridge across scale, architecture, objective *and* the
QAT/PTQ regime — precisely the transfer this document forbids elsewhere." **§3.6 does it again.**

**The arithmetic asked for.** Is the 1.10× gate wide enough that σ does not matter? Only away from the
line. With per-arm sd `σ_r`, the difference `BPB_conv − BPB_donor` has `SE = σ_r·√2`.

| σ_r | SE of the difference | 2·SE band around Δ* = 0.075 |
|---|---|---|
| 0.005 (optimistic — σ_seed borrowed) | 0.0071 | 0.061 … 0.089 |
| 0.010 | 0.0141 | 0.047 … 0.103 |
| 0.020 (plausible at ternary) [?] | 0.0283 | 0.019 … 0.131 |

A ternary result landing at ΔBPB = 0.070 or 0.080 is **within 1 SE of the threshold even under the
most optimistic σ**, and a permissive gate is precisely the design that parks borderline methods near
its own line. The 15σ framing is true only of the *distance from zero*, which nobody is testing.

**Required:** (a) ≥3 replicates at the ternary point over the two live nuisance axes
(PTQ seed × calibration draw), reported as mean ± sd; (b) a slice-level SE (bootstrap over eval
windows) for the donor baseline; (c) **a pre-registered INCONCLUSIVE verdict**: a result within 2·SE
of Δ* is neither PASS nor FAIL and buys another replicate, not a decision. As written the gate is
binary and therefore forced to manufacture a verdict out of noise.

Related dead guard: §3.5's sensitivity arm reads "If they differ by more than **harness σ**" — a
quantity this pre-registration never defines and never measures. That guard cannot fire in either
direction as written.

### B5 — Control 2's lesion is plausibly a known-*negative*, and there is no ladder

> §3.7.2: "Zero the output projection of **one attention head in one layer** — the smallest structural
> corruption available … BPB must rise by ≫ the gate width."

Three problems.

1. **Attention heads are the most redundant structure in a transformer.** The head-pruning literature
   is built on the observation that many individual heads can be removed with near-zero loss. Zeroing
   one head in one layer of a ~28-layer GQA donor is a corruption with a **substantial prior
   probability of producing no measurable BPB change at all** — i.e. it may be a *known-negative*
   dressed as a known-positive. Law §6.3 requires the instrument to be shown to fire on a
   known-positive; a lesion whose sign is not established cannot serve.
2. **"BPB must rise by ≫ the gate width" is a prediction, not a protocol.** There is no
   pre-registered action if it doesn't. That is the moment §6 was written for, and it is unhandled.
   Whatever happens will be rationalized.
3. **It is the wrong calibration target.** §6.3's minimal-significant-corruption rule wants the
   instrument's **detection floor**, and a floor is measured *at* the decision boundary, not far above
   it. A lesion that rises "≫ the gate width" proves the detector notices catastrophes; it says
   nothing about resolution at Δ*, which is where every real verdict will be read.

**Better-calibrated lesion — and the Principal already designed it, in §5.3.** Use the same monotone
perturbation ladder proposed for stage 2a: scale one head's `o_proj` block by `(1−ε)` for a geometric
sweep of ε (or inject Gaussian noise at declining SNR), measure BPB at each rung, and **report the
lesion magnitude at which ΔBPB equals the gate width Δ\*.** That yields a *detection floor in donor
units*, fires by construction, is monotone (so a non-monotone curve is itself a caught apparatus bug),
and is reusable. Additionally pick at least one rung in a structure with a **known** non-redundant
effect (an MLP `down_proj` column block, or an RMSNorm scale) so the ladder is anchored to a lesion
whose sign is not in doubt. It is an internal inconsistency that §5.3 invented this instrument and §3.7
did not use it.

### B6 — No control proves the calibration pipeline reaches the output

§3.7 has four controls and none of them is a calibration-influence control. §3.5's sensitivity arm
(second disjoint calibration set) **does not substitute, and the reason is structural**: two
similar-composition general-text corpora may legitimately produce near-identical results, so a null
there is *ambiguous* — it is consistent both with "calibration is not load-bearing" and with
"calibration is not connected". Only a control whose expected effect is **large and known-positive**
can separate them.

This is the project's named characteristic failure — §6.3's "a measurement tool that silently returns
zeros" — and calibration is the classic site for it: hooks registered on the wrong module, statistics
collected under `no_grad` into a buffer nothing reads, a scale computed and then overwritten by a
default. Every one of those produces a beautifully plausible ternary number.

**Required — new control 5:** run the ternary point with calibration replaced by **random tokens
uniformly sampled from the vocabulary** (not shuffled text — text-shuffled activations may still carry
usable marginal statistics). Pre-register the expectation: BPB must differ from the real-calibration
run by ≫ the harness SE. **If it is bit-identical, the calibration path is dead and every number in
stage −1 is void.** Log both directions per §6.4.

### B7 — The bit-width sweep is not one variable, and §3.4's deliverable overclaims what a curve can answer

> §3.4: "fp16 baseline → 4-bit → 3-bit → 2-bit → ternary (1.58-bit), **identical apparatus,
> calibration and eval. One variable, five points.**"

"Identical apparatus" is asserted, not verified, and in this method family it is very unlikely to hold:

- **Group size** typically shrinks as bits drop (128 at 4-bit → 64 or 32 at 2/3-bit). That changes the
  scale-parameter budget, i.e. the *effective* bits/weight, so the x-axis is not the axis it is
  labelled.
- **Quantizer identity changes.** "Ternary (1.58-bit)" is not the 1.58-bit point of a uniform grid; it
  is a 3-level codebook, usually with an absmean/threshold rule (λ) that has no analogue at 2/3/4-bit.
  Between 2-bit (4 levels, uniform, affine) and ternary (3 levels, symmetric, thresholded) **the
  quantizer changes class**, not resolution. That is the single most important point on the curve and
  it is the one most likely to be a different algorithm.
- **Outlier / salient-channel escape hatches** (keeping a fraction of channels at higher precision) are
  standard in this family at 2-bit and below and typically absent at 4-bit. If the implementation
  enables one silently, the low-bit points are **mixed-precision results reported as pure-bit
  results** — a hidden default that flatters exactly the arm the project cares about.
- **Clipping/search ranges** are commonly bit-dependent defaults.

If any of these move, the "curve" superimposes a quality axis on a configuration axis and is not a
curve. §6.2: "If you change two things and the result moves, you have learned nothing."

**Required:** before the run, commit a **per-point configuration table** — quantizer class, levels,
group size, symmetric/affine, scale dtype and count, clipping/search procedure, outlier or
mixed-precision policy and its fraction, calibration sample count, fitted parameters per layer, RNG
seed — read **from the implementation, not from the paper** (law §6.5: the artefact is the authority).
Then run the sweep with **group size and outlier policy held constant across all five points** as the
primary arm; if the method's own defaults differ, run the defaults as a clearly labelled **second,
two-variable arm** and never mix the two on one plot. Also report **effective bits/weight including
scales**, not nominal bit-width. Finally, label the fp16 point as the reference, not a curve point —
it is "no quantizer", a different arm by definition.

**And the overclaim:** §3.4/§3.9 say the curve answers "*which bit-width would the engine have to grow
to?*". It cannot. That question needs the engine's **rate and footprint at each bit-width**, which is
not measured here and which §2.9(c) shows is counter-intuitive on this silicon (nibble-packing is ~10%
*slower* on a compute-bound path; bytes halve, time does not). The curve gives quality-vs-bits only.
State that.

---

## FLAGS

### F1 — Calibration size deviates from the method's published configuration, breaking control-1 comparability

§3.5 seals 512 × 2048 ≈ 1.05M tokens. The survey (`:61`) records PT²-LLM as using **128 samples**.
Running 4× the samples is a second variable relative to the published anchor: control 1 must use the
**published** sample count or it is not reproducing the published number. State the split explicitly —
anchor runs at the paper's configuration, gate runs at the sealed 512 — and report both.
Separately: the choice of 512 is unmotivated. If calibration size matters (§8.L.1 says measure it,
don't pick a default), it is a free variable nobody has priced.

### F2 — "Composition mixed, not code-only" is not a specification

§3.5 pins the corpus "by content hash" but never says what is in it. A hash pins *a* corpus; it does
not make the composition reviewable or reproducible. Required: the source datasets, their proportions
by token, the sampling script, the seed, and the shuffle procedure — the same standard §8.L.4 applies
to licence. Also state whether the **second disjoint calibration set** (the sensitivity arm) is itself
decontaminated against the eval slice; §3.5 only claims decontamination for the first.

### F3 — Dtype is a hidden default, and the project has already been bitten by it

§3.4 says "fp16 baseline". On CPU, `transformers` will not silently give you fp16: unless
`torch_dtype` is passed explicitly it loads in fp32 regardless of what `config.json` declares, and
CPU fp16 matmul is emulated where it exists at all. So "the fp16 baseline" is plausibly an fp32
baseline — a different number. `CANONICAL_EVAL.md` exists in part because **absolute BPB is
dtype-sensitive** and quantifies the bf16-autocast eval noise at ~1e-4. Required: declare the
`torch_dtype` actually used for the baseline and for the dequantized path, declare thread count and
`torch.set_num_threads`, and record the resolved dtype from the loaded module rather than from the
argument (law §6.5).

### F4 — Control 4 is a guard, not an exercised guard

§3.7.4: "**Named refusal** when config and loaded weights disagree (§6.5, §6.6 — `strict=True` does
not verify architecture)." Law §6.4 requires **both directions recorded**: a must-pass case (correct
artefact loads and runs) *and* a must-refuse case (a deliberately mismatched config is refused, with
the refusal logged). The pre-registration names the guard and never says it will be fired. Required:
a scripted must-refuse case — mutate one architectural field (`num_key_value_heads`,
`num_hidden_layers`, or `tie_word_embeddings`) in a copied config and assert the named refusal —
plus, per §6.6, an **architecture-sensitive numerical probe**: a fixed prompt whose fp32 logits hash
is recorded in the manifest before the sweep and asserted at every arm's load. Note for this donor
specifically: Qwen2.5 uses tied embeddings at small scale [?] — a silent untie or a head loaded from
the wrong tensor is the exact §6.5 failure shape and would move BPB while `strict=True` stays happy.
Read the config from the artefact and assert against the manifest; never the reverse.

### F5 — The FAIL side of the scope statement is over-read; the caveat is one-sided

§3.6 handles the PASS side well (see P1). The FAIL side does not get the same discipline:

> §3.1: "a clear failure makes the ternary-primary route **moot**."
> §3.6: "If it FAILS, the path is **very probably dead** — that is the whole informational content."

The mandate's own §9 is deliberately narrower: "This is evidence against **the tested, bounded PTQ
path** — not a claim that all possible PTQ is impossible." §3.9 restates the narrow version correctly,
which makes §3.1 and §3.6 internally inconsistent with §3.9. One donor (dense, 1.5B, Qwen2.5) under
one method, one calibration composition and one precision map is a narrow experiment. Required: bring
§3.1 and §3.6 to §3.9's wording. Both the PASS and the FAIL must be scoped, or the scoping is
advocacy rather than method.

### F6 — Donor/target mismatch is unstated, and expert calibration coverage is a named mechanism against transfer

§4.2 places the live donor region at "**sparse-MoE donors, ~1–2B absolute active params, ≲24B total**".
The stage −1 donor is **dense, 1.5B total**. Nothing in §3 says how a ternary-PTQ result on a small
dense transformer transfers to a large MoE, and there is a concrete mechanism for it *not* to:
**per-expert calibration coverage**. A 1.05M-token calibration set spread over E experts with top-k
routing gives each expert roughly `k/E` of the tokens; at E=64, top-8 that is ~131K tokens per expert,
and the tail experts far less. Ternary quantization statistics for those experts are estimated from
sparse data — a failure mode entirely absent from the dense donor. Required: state this limit
explicitly in §3.6's scope paragraph, and pre-register that a stage −1 PASS does **not** license a
per-expert precision map at the MoE target without its own calibration-coverage measurement.
Related, minor: §2.5 and the §8 decision log (row 8) frame stage −1 as resolving "does P61's
+0.018–0.022 transfer from 8.3M-QAT to donor-PTQ?" It cannot. Stage −1 measures a different model,
scale, organ set and regime; it produces a donor-PTQ number, not a transfer verdict on P61.

---

## PASSES

### P1 — The PASS-side scope disclaimer is correctly and unusually strongly stated

> §3.6: "**If it PASSES, it says NOTHING about S4 retention.** It is not evidence of ≥90% global
> retention, not evidence of ≥80% per critical task, not evidence of S2 fit, not evidence of anything
> at donor scale. A pass licenses proceeding to the next cheap question and nothing more."

This is exactly right and matches S4 and §9's stage −1 gate. It is the strongest paragraph in §3.
(Its one-sidedness is F5, not a defect in this text.)

### P2 — Control 3 (exact comparator) is well-formed and bidirectional

> §3.7.3: "Same weights twice → bit-identical BPB. Different weights → different BPB."

Must-pass and must-refuse in one line, per §6.4, and it is the right primitive for a determinism-gated
project (§6.7). Retain unchanged. One addition worth making: run it **across process restarts**, not
only within one process, so thread-count and reduction-order nondeterminism is in scope.

### P3 — The pre-registration correctly refuses to start, and says so

§3.3 ("**No run may start until this slot is sealed, and it cannot be sealed today**"), §3.8 (install
and CPU budget withheld) and §2.7's recorded catch of a fabricated number are the discipline the
project's laws ask for. The Principal surfaced its own evidence-chain failure rather than burying it.
That is why this review is a protocol audit and not a post-mortem.

---

## Verdict

**May this pre-registration proceed to the Owner for execution approval? NO.**

Seven BLOCKs stand. Note that B1 and B2 compound: an unpinned slice plus a ratio gate means **both the
threshold and the quantity it is compared against are chosen after the apparatus exists**. That
combination is not a weak gate; it is not a gate.

### Minimal fix list that would make this ready

1. **Pin the eval slice** to `CANONICAL_EVAL.md`'s standard — corpus/revision/byte-range/sha256,
   token and byte counts, tokenizer revision, context length **and stride**, BOS and final-window
   handling, loss reduction, decontamination procedure — and **pin `BPB_donor` as a number** before
   any converted arm runs. (B1)
2. **Restate the gate additively**: `BPB_conv ≤ BPB_donor + Δ*`, Δ* a fixed bits/byte value, with the
   implied BPB ratio and the implied PPL ratio at the measured bytes/token both shown, so it is
   comparable to the 1.04–1.08× literature. (B2)
3. **Resolve the §3.2 / §3.7.1 conflict**: either change the donor to one the method actually
   publishes, or change the method to one published on Qwen2.5-1.5B. Then state control 1's numeric
   tolerance, run it at the paper's own harness/dataset/seqlen/sample-count, **and** add the
   BPB-vs-published-harness bridge check on the unquantized donor. (B3)
4. **Add σ**: ≥3 replicates at the ternary point over (PTQ seed × calibration draw), a bootstrap SE on
   the eval slice, and a **pre-registered INCONCLUSIVE band** at 2·SE around Δ*. Delete the σ_seed
   citation — it is imported from an apparatus that does not apply. Define the "harness σ" that
   §3.5's sensitivity arm depends on, or that guard cannot fire. (B4)
5. **Replace control 2 with the §5.3-style monotone lesion ladder**, reporting the detection floor in
   donor units (the ε at which ΔBPB = Δ*), with at least one rung on a structure of known
   non-redundant effect, and a pre-registered response if the ladder fails to fire. (B5)
6. **Add control 5 — noise calibration.** Random-vocabulary calibration must move BPB by ≫ SE.
   Bit-identical output means the calibration path is dead and stage −1 is void. (B6)
7. **Commit the per-point quantizer configuration table read from the implementation**, hold group
   size and outlier policy constant as the primary arm, report effective bits/weight including scales,
   label fp16 as the reference rather than a curve point, and drop the "which bit-width would the
   engine grow to" claim or pair it with engine rates that do not yet exist. (B7)

The six FLAGs may proceed if the Principal records the residual risk per §7.2, **except** F4, which I
would fold into the fix list in practice: an unexercised architecture guard against this project's own
characteristic failure is the cheapest thing on this page to fix and the most expensive to discover
afterwards.

Re-review required after the fixes. This is the pre-run half of §7.2; the post-run half is a separate
pass against the recorded evidence.
