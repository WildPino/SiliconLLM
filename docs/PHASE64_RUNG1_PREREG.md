# Rung-1 (S0) — Pre-Registration

**Sealed by the Architect 2026-07-19. Pushed before any rung-1 gate-bearing run executes** (push-before-run; the GitHub received-time is the witness). Nothing below is amended after numbers exist. Costs may re-price; **gates never move**.

Upstream context, all already pushed: `PHASE64_DECISIONS.md` (D1-D9), `PHASE64_TRAINING_PLAN.md` §12 (Inventor returns) and §13 (MVE gates read + re-priced ladder), `PHASE64_RUNG1_BRIEF.md` §8-§10 (WS1/WS2/WS5/WS6 returns).

> **AMENDMENT LOG (MM).** This document was pushed, then amended before launch. Both facts are on the record because
> the document's whole value is that an auditor can check it against the run — so the amendments must be as visible as
> the seal, without diffing commits.
>
> | version | commit | what |
> |---|---|---|
> | v1 — initial seal | `daad4fe` | as sealed by the Architect |
> | v2 — pre-launch amendments | `8d32ccf` | §2 stage-E extension re-costed from measurement (+34%, was "open, to be measured"); §2 decontamination instrument specified (window-vs-window, 50% stride) after a planted positive control showed document-level Jaccard passes the leak it exists to catch; §6.2 2×2 read per-distance + **anti-escape-hatch clause** + scheduled early read; §8 two prerequisites added (license posture, job liveness/resumability); §9 re-priced 70-150 → 95-200 session-h |
> | v3 — corpus scope resolved | `d83b0d5` | §2 corpus decided from the metadata census: **strict-permissive only** (`no_license` dropped) with an enumerated licence whitelist, and **rung-1 scoped to Python**. Both tighten: the corpus gets smaller and its provenance more explicit, and one undeclared parameter (language mix) is removed from the confound set before any run. Follow-up in the same commit: the whitelist is shown to be *necessary* (BigCode's permissive class admits GPL/MPL/proprietary refs), the resulting margin is 1.0× not 9.5×, and a **pre-registered corpus-size trigger** plus **uniform, monotone fetch selection** are added |
> | v4 — WS3 returns | `d83b0d5` | §2 corpus as built (6.53 GB / 1.38 M files, filter breakdown, **traversal-coverage limitation declared**); §2 **S2 trigger read** (fires at 8 B tokens only, margin warning not wall); §3 **val is temporally held out by construction** + quoting rule (rung-1 BPB not comparable to P62's 1.242) + record-only in-corpus secondary val on a reserved hash band; §4 **budget unit declared** (equal student tokens; V4096 sees ~11% more bytes — stated in the claim, not corrected away); §4.3 logit budget re-measured (ratio 0.637/0.708, 8-18 h, off the critical path). Tightens or declares throughout; no threshold moved |
> | v5 — α declared | `b861153` | §4.3 the span-KD challenger no longer inherits an unnamed α: it runs at **α ∈ {0.5, 0.25}** and is **read as a trend, not a best-of**, so closing D3 permanently now requires a coherent monotone pattern where a single point previously sufficed. Raised by an external commenter (relayed by the MM) in general form; the specific mechanism — a directional gradient blend over ~97% of positions under span mode, not a magnitude effect — is ours. **Tightens**; costs one arm (§9: 33% → 41% of a rung) |
> | v6 — measured-constants refresh + consistency pass | `b861153`, expanded in this commit | *Record note (MM): v6 shipped in two pushes — the constants refresh in `b861153`, the consistency pass here. The row below describes the expanded form; the narrower first form is in `b861153`'s diff. A pushed row is rewritten in place at most once, and only to widen it.* Two figures updated after the ex-band BPE retrain: §4's byte-advantage template **+11.2% → +11.3%** (B/tok **2.571 / 2.861 measured on a 64 MiB corpus sample disjoint from the BPE-training bytes**, ratio-derived cross-check 2.564 / 2.854 agreeing), and §4.1's tie-breaker rationale **"17% shorter student sequence" → ~10%** (10.15% measured) — the 17% did not reproduce from any measured pair and is corrected before launch, not defended. **Consistency pass from MM review, five findings all fixed**: §9 sum recomputed for the fifth arm (**2.1 → 2.2 rung-equivalents, +108% → +116%** — the four-arm arithmetic had survived v5); 0.708 vs 0.709 labeled as pre-/post-retrain in §4.3; "four arms, not six" → five/seven; the pre-re-measure logit budget (6-12 h) labeled superseded where it appears (§4.3, §8); "only arm 4 consumes logits" → arms 4-5. **No gate, threshold, or tie-breaker direction moves**; the trial-slice span statistics in §4.3 are noted as pre-retrain and re-derived from production logs |
>
> **Legitimacy check, stated so it can be contested — and stated in the form that is actually attackable.**
>
> It would be too convenient to write "no numbers exist yet". Numbers do exist, and they drove these amendments: the
> licence census, the corpus build, the seq-2048 throughput bench, the compile screen, the decontamination positive
> control. So the honest claim is narrower and is the one the seal actually makes — **no rung-1 *gate-bearing* number
> exists.** Not one arm of §4 or §6 has run; no code-val BPB, no A/B delta, no gate reading. Six of the seven §8
> prerequisites are still open; the one now closed (licence posture) was closed **by a rule written into v2 before the
> census ran**, and it resolved in the direction the rule pre-declared, which is the opposite of fitting a criterion
> to a result.
>
> The distinction that carries the weight: every measurement above is **apparatus or corpus** — it sizes and specifies
> the instrument. None of it is an outcome the gates are read on. A pre-registration that could not be sharpened by
> measuring its own instrument would just be a pre-registration written in ignorance.
>
> **Direction check, the part that can be audited line by line:** every amendment either **tightens** a criterion or
> **declares additional cost or additional limitation**. None loosens a threshold, softens a criterion, or widens a
> tie-breaker. Three that could easily have gone the other way and did not: §6.2's anti-escape-hatch clause makes an
> uncalibrated instrument read **UNREAD** rather than letting recall pass on a null *or* be demoted on one; §2's
> traversal-coverage limitation **narrows the scope of every absolute claim** rung-1 can make, and was volunteered;
> and §3's quoting rule **forbids** comparing rung-1's BPB to the P62 1.242 already published in the README — giving
> up the flattering comparison before anyone could make it. Verified by MM against the diff before each push.

## 1. What rung-1 is for

S0 (~30 M total / ~11 M active, E32 × h128 top-8) is the first rung of the ladder on **real code data**. It answers three things: does the recipe hold at rung scale on-domain; do the two candidate recipe improvements (vocab, structured x_proj) survive more steps and real data; and does the MoE dial actually deliver capacity (the D2 probe opened by the MVE's D→E finding).

Quality at S0 is **gated** (unlike the MVE, where it was recorded): the per-rung property gates of `PHASE64_DECISIONS.md` §10 apply.

## 2. Recipe and declared deviations from the MVE

Frozen spec v1 (`SCALEUP_ARCHITECTURE.md` §9) with these **declared, arm-symmetric** deviations, each already adjudicated on measured evidence:

| deviation | source | note |
|---|---|---|
| **CE-primary** (KD not in the main run) | §13, D3 FAIL | span-KD returns as a screening arm, §4.3 |
| **α-QAT ON** at stage-D entry, ramp **≤10% of stage-D steps** | §13 gate (iii) fired; brief §8 | end-state bit-identical to hot-swap QAT |
| **`--sparse-moe` ON** | brief §8 | warrant: grad-equivalence 1.1e-06, determinism bit-identical, resume 29-interruption |
| **effective batch 16 via micro-batch 8/rank + accum** | brief §8 | *comparability choice*, not a workaround — keeps the step grid identical to MVE runs 1-3 |
| **fp32 AdamW** (not 8-bit) | brief §10 | measured winner at this scale (−6% throughput for 8-bit); re-opens at S2 |
| **flat LR + `--warmup 200`** | run-3 declared deviation | carried forward, arm-symmetric |
| **stage-E context extension to seq = 2048** at the widest micro-batch the card holds | curriculum §4 "progressive context", made explicit by the §6.2 distance rule | **measured, not assumed**: memory is linear in L as predicted (+1.4% at constant B·L), but throughput is not — the scan is sequential in L, so depth ×4 with width ÷4 starves the GPU on both axes. Spending the memory headroom on width recovers +20.6% (batch 2 → 4), bringing the penalty from 2.69× at seq 2048 down against 1573 tok/s at seq 512. **Applied to stage E only (f = 0.20): +34% on the run.** 9988 MiB of 12 GB says batch 4 is the ceiling on the 3060 — no further free width here; **per-protocol the ratio may transfer to the 16 GB T4, the absolute does not.** Priced into §9 |

**`--compile-scan` is OFF — decided, not conditional: gate (b) failed reproducibly on the local screen** (§6.4). **Data**: pipeline of brief §3 with **P62 decontamination J=0.5 mandatory**; the manifest hash is recorded in §8 at launch and verified at every training start.

**Decontamination instrument — specified precisely, because "J=0.5" alone reconstructs the wrong filter.** The comparison unit is **window-versus-window at equal size on both sides, with 50% stride on the training side**, not document-versus-window. Reason, found by a planted positive control before any rung-1 number existed: Jaccard is symmetric in size, so a 24 KiB training file *containing* a verbatim 4 KiB val window scores J ≈ 0.17 and survives any sensible threshold — the document-level comparison passes exactly the leak it exists to catch. Cutting both sides to equal windows makes Jaccard behave like containment (at equal size, J = 0.5 corresponds to ≈2/3 shingle overlap); the 50% stride ensures a leak straddling a boundary falls wholly inside some window. LSH bands are **derived** from the declared threshold (b=16, r=8 → curve 0.707), not hard-coded, and every LSH candidate is verified against true Jaccard — bands produce false positives by construction, and discarding them unverified would delete real training data.

**Corpus scope — decided 2026-07-19 from the metadata census (no content fetched).** The census measured strict-permissive availability at **821 GB point estimate, 382 GB at the conservative floor**, against a 40 GB target — roughly an order of magnitude of headroom. Two decisions follow, both taken *because* the headroom removes any cost:

1. **Strict-permissive only; `no_license` is dropped.** `no_license` means no licence was *detected* — unknown, not permissive — and at 9.5× margin there is no reason to train on unknown provenance. Tightened further: the filter is an **enumerated licence whitelist** rather than BigCode's permissive *class*, so the corpus composition is stateable and auditable file by file, and the manifest records the licence per file. The claim this supports is precisely *"trained only on files carrying an explicit permissive licence, enumerated and auditable"* — **not** any claim about the licence status of model outputs.
2. **Rung-1 is scoped to Python.** Python alone holds 48.5 GB strict-permissive, above the target on its own. Reasons, in order: the pinned code-val is CPython, so the canonical metric is on-domain; the language mix is otherwise an **undeclared parameter that would silently move the headline BPB**, and it is removed from the confound set rather than chosen; and one variable at a time. Multi-language expansion is a **rung-2+ scope decision with its own gate** (D1 — it changes the tokenizer, hence the recipe). Every rung-1 claim is stated as Python-scoped. Side benefit: the census's one unstable term (JavaScript, 155-1038 GB, a heavy tail almost certainly made of bundles and minified files we would filter anyway) becomes irrelevant to the decision.

**The whitelist was necessary, not merely prudent — and the margin it leaves is 1.0×, not 9.5×.** Measured on 4789 sampled `permissive` rows, BigCode's `license_type == "permissive"` is a *class*, not a per-file guarantee: it admits GPL-1.0-or-later, MPL-2.0, CC-BY-4.0, `LicenseRef-scancode-proprietary-license`, and `LicenseRef-scancode-unknown-license-reference`. The implemented rule — keep a file only if `detected_licenses` is non-empty **and every** detected licence is on the whitelist — drops 10.4% of permissive files but **17.9% of permissive mass** (dropped files run ~1.9× average size). Python then yields **39.8 GB against a 40 GB target**: the two decisions above are each right and **compose to 1.0×**, before dedup, decontamination and quality filters. Neither showed it alone. *The enumerated whitelist itself is recorded in the manifest and is load-bearing twice over — it sets both the corpus size and the licence claim.*

**Why this is not a blocker, stated with the arithmetic rather than asserted:** 40 GB is the *full ladder's* target, not rung-1's. At the measured byte rates, S0's 0.8-1.5 B tokens is ≈2-4 GB and the whole instrumented rung ≈4-6 GB — covered with large margin by the 6.53 GB actually fetched. **Pre-registered trigger, fixed now:** the post-filter corpus size is reported, and **if it falls below 2× S2's single-epoch requirement, the rung-2 multi-language expansion gate fires early** rather than multi-epoch training being adopted silently — D1 makes recipe invariance the ladder contract, so a quiet second epoch is a recipe change and does not happen by default.

**TRIGGER READING (2026-07-19) — it fires, at the top of S2's bracket only.** Clean Python strict-whitelist pool ≈39 GB. S2 at 8 B tokens needs 20.5 GB (V2048) or 22.8 GB (V4096), so 2× is 41 / 45.6 GB — above 39. At 5 B tokens 2× is 25.6 / 28.5 GB, comfortably below. Consequences, taken as pre-registered and **without moving the threshold**: (i) **multi-language expansion becomes a planned rung-2 agenda item, not a contingency** — it is gated there anyway under D1, since it changes the tokenizer and therefore the recipe; (ii) single-epoch at S2-8B remains *feasible* at 1.7-1.9× — this is a margin warning, not a wall, and it is stated that way; (iii) recorded cross-link: **the vocab winner moves this number** — V4096 needs ~11% more bytes for the same token budget, so §4.1's outcome tightens or loosens S2's margin; (iv) growing the corpus is a re-run of the same command with a higher `--target-gb`, and monotone selection keeps rung-1's file set an exact subset. Nothing here is a rung-1 decision.

**Fetch is target-driven, and its selection has two requirements.** Downloading 40 GB for a rung that consumes 4 would put ~13 unnecessary hours on the critical path. But the fetched subset must be (a) a **uniform selection under the declared seed 20260719**, never a prefix of the stream — The Stack is repository-ordered, which is the same bias the census had to defeat — and (b) **monotone**: selection by hash threshold, so raising the target to grow the corpus for rung-2 leaves rung-1's file set an exact subset and earlier manifests stay valid.

**Corpus as actually built (2026-07-19), with its declared limitation.** 6.53 GB, 1,382,190 Python files, `corpus_sha256` in the manifest and verified at every training start. Metadata rejects: licence 8.45 M (the whitelist doing the heavy work), size 1.08 M, not-selected 957 k. Content rejects 1.25%: whitespace-fraction 10,947, long-line 6,082, mean-line 231 — the rewritten whitespace filter is the largest contributor and is catching the data blobs it claims to catch (verified against wrapped base64, which defeats the line-length filters). Network loss 12 of 1.4 M. Then exact-hash −0 (confirming the variant is already deduplicated), MinHash −0.08% (genuine near-duplicates only), decontamination −0.00% (temporal, above).

*Limitation, declared rather than buried:* the selection is uniform **by hash** (key mean 0.4984, max|F−U| 0.0055; subset-monotonicity verified) but the traversal covered 12 M of 47.3 M metadata rows, and The Stack is repository-ordered with only a 10 k shuffle buffer to decorrelate. **The corpus is therefore a uniform sample of roughly the first quarter of the repository-ordered pool, not of the whole pool.** No A/B is affected — every arm trains on the same corpus — but absolute claims are scoped to it, and the monotone hash selection means a later target increase extends coverage without invalidating any earlier manifest.

*Census method, recorded with its limits:* sampling is **not uniform** — The Stack is ordered by repository, so a 10k shuffle buffer decorrelates shards and rows but does not make the draw uniform; the intervals are sampling intervals, not guarantees. Per-language lower bounds were summed to reach the floor, which **understates** the true lower bound of the sum — conservative in the safe direction. At 9.5× margin neither limitation can change the decision; at 1.2× the census would have had to be redone properly.

**Pre-registered scale check on the filter, fixed now:** the removal count and fraction are reported at full corpus scale. Zero collaterals on a smoke corpus does not establish the false-positive rate at 40 GB. The asymmetry is deliberate — over-removing training data is cheap, missing a leak invalidates every rung-1 gate — but **if removal exceeds 5% of documents, the pipeline stops and examples are reviewed before any training starts**, because that would indicate the instrument is over-firing rather than that the corpus is that contaminated.

## 3. Units, baselines, significance

- Canonical metric: **BPB per-domain on the pinned code-val** (P62 CPython byte-slice), never blended. Byte-normalized, so it is comparable across vocabularies.
- **σ_seed = 0.005** (R1 calibration). A single-seed delta **≤ σ_seed is not a claim** — it triggers the pre-declared tie-breaker, never a narrative.
- All A/B arms are **seed-paired** (identical seed across arms of a comparison).
- Eval cap identical at every stage entry and exit (the MVE fix: transition deltas must compare like with like).

**The canonical val is temporally held out by construction — recorded 2026-07-19, with a quoting rule attached.** The P62 val is CPython 2026 (it contains `Lib/_colorize.py`, added in 3.13); the corpus is a 2023 Software Heritage snapshot. Decontamination therefore removed 0.00% not because the filter is blind — a planted leak proves it fires, and max Jaccard against any val window was 0.031, nowhere near threshold — but because **the val postdates the corpus and there is nothing to catch**. This is the strongest possible form of decontamination: structural, not filtered. The filter stays mandatory anyway (defence in depth; the corpus grows at later rungs).

Two consequences, both about *reading* the number, neither about the gate:
1. **The rung-1 BPB is a temporal-generalization measure, not an in-distribution one.** This biases the absolute number in the **conservative** direction (harder val), and it is arguably the right measure for a code model that will be deployed on code written after its training data.
2. **Quoting rule: rung-1 BPB is NOT comparable to P62's 1.242.** That number came from an 8.3 M model whose training data was contemporaneous with its val; comparing across it would conflate model size, corpus, and a temporal shift. The rung-1 baseline is the matched-dense arm (§6.3) and nothing else. External codec baselines (brotli/zstd) remain valid, since they were computed on the same pinned val.

**Secondary val — in-corpus, held out, RECORD-ONLY, never gated.** A slice of the 2023 corpus itself, reserved *before* training by a **permanent hash band** so that no future corpus expansion can ever pull those files into training. Its only job is to decompose: the gap between it and the P62 val separates *temporal shift* from *modelling ability*, which is otherwise unidentifiable if rung-1 BPB comes out disappointing. It is explicitly not a gate and not a target — the canonical metric is unchanged.

*As built (2026-07-19):* 23,396 documents / 112.0 MiB held out, 1,358,794 / 6.42 GB remaining for training; `val_blob_sha256` recorded. The band sits at the **bottom of the hash scale**, so it is inside any selected corpus for any threshold — permanence exercised, not asserted (verified invariant under a hypothetical expansion to p → 1.0). Observed band fraction 1.693% against 1.684% expected (= band 0.01 ÷ select_p 0.5939), i.e. 0.5% relative — the "1.000%" that first read as a failure was a wrong denominator in the assertion, not a non-uniform hash. The rule lives in **one definition** imported by both the fetcher and the split; two copies of a selection rule is one too many when the whole point is that it names the same files forever. Files fetched before the band existed are split retrospectively rather than re-fetched — the rule is deterministic, so the resulting set is exactly what a band-aware fetch would have produced.

*Property worth naming:* because the band is a fixed file set independent of the corpus threshold, **this val stays comparable across every rung even as the training corpus grows** — a second, unplanned benefit beyond non-contamination.

**Prerequisite before the secondary val is quoted:** the BPE tokenizers must be trained on the **training split only**. They were trained from an 8 MiB sample drawn before the band existed, so ~1.7% of that sample can be val-band content, and merges learned on held-out content survive the removal of the documents — the Builder's own reason for training tokenizers after decontamination, applied to the band. Retraining costs minutes. Scope, stated so the fix stays proportionate: this does **not** touch the canonical P62 gate (temporally out of corpus, unreachable), and does **not** bias the vocab A/B (both arms treated identically); it affects only the record-only decomposition, and only slightly.

## 4. The screening block — stage C, one variable at a time

Three comparisons run **sequentially on stage C only**, each arm at a declared **15% of the stage-C token budget**, each arm seed-paired. These are **screenings, not claims**: they select the configuration for the main run and are labeled as such wherever quoted.

**Budget unit — declared, because the vocab arms make it load-bearing (added 2026-07-19).** Each arm's 15% is counted in **student tokens**, i.e. equal optimizer steps and equal compute. Measured consequence: at **2.571 B/tok (V2048) versus 2.861 B/tok (V4096)** — measured on a 64 MiB corpus-level sample disjoint from the BPE-training bytes; the ratio-derived cross-check (0.637 / 0.709 × teacher 4.025 = 2.564 / 2.854) lands on the same +11.3% by an independent route — **the V=4096 arm sees ~11% more bytes for the same step budget**. (The 8 MiB BPE-training sample packs at 2.638 / 2.980, but that is in-sample fit, not corpus packing, and is never quoted; the manifest now carries both fields, labeled.) That is not a nuisance to be corrected — it is part of what a larger vocabulary buys, and the byte-normalized metric makes the arms comparable regardless. But it must be **stated in the claim, and quantified rather than hedged**. Claim template, fixed here: *"at equal token budget — with the V4096 arm seeing **+11.3%** more bytes, which is part of what the larger vocabulary buys — V\<winner\> gives lower code-val BPB by X"*. A qualifier the reader can weigh beats a caution they must take on trust. The alternative unit (equal bytes) answers a different question and is not what the ladder's token-denominated budget matches.

**The whole screening block runs on the logit-covered slice** (§4.3's data specification, generalized here), which buys two things beyond the confound it was introduced to close. First, **chained controls**: each screening's control arm *is* the previous screening's winner, so the block costs **five arms, not the seven** a per-comparison-control design would need (four-not-six before v5 added the second α point). Second, the stored logits are per *teacher* token and therefore vocabulary-independent — the same logit set serves both vocab arms, which is why they were stored that way in the first place.

| arm | what it adds | control |
|---|---|---|
| 1 — V=2048 | vocab candidate | paired with arm 2 |
| 2 — V=4096 | vocab candidate | paired with arm 1 |
| 3 — x_proj r=26 on the vocab winner | §4.2 | the winning arm above |
| 4 — span-KD, α = 0.5, on both winners | §4.3 | the winning arm above (= the α 0 point) |
| 5 — span-KD, α = 0.25 | §4.3 trend | same control; arms 4-5 read together |

Five arms × 15% of stage C = **75% of stage C ≈ 41% of one rung's tokens.**

Decision rule, identical for all three and fixed here: **adopt the lower code-val BPB if the margin > σ_seed; if ≤ σ_seed, adopt the pre-declared tie-breaker below.** Every tie-breaker points in the cheaper direction — the conservative one.

**4.1 Vocab (D3's open half).** V=2048 vs V=4096, per-domain BPE.
Tie-breaker: **V=4096** (records a **~10% shorter student sequence** — 1 − 2.571/2.861 = 10.15% measured on the disjoint corpus sample, 10.2% ratio-derived, two routes agreeing; the "17%" quoted before v6 did not reproduce from any measured pair — compute saved in training and bytes/s at inference).

**4.2 Structured x_proj (§12).** r=26 vs dense, on the vocab winner.
Tie-breaker: **r=26** (17.6% of x_proj bytes; ~1.76 MB fp32 resident freed across the 5 Mamba layers, on the dominant proj-GEMV class).
Pre-registered reading: the sandbox measured r=26 *beating* dense by −0.0095 at 4k steps. **A null result here is the expected outcome of a regularization benefit fading with scale, not a failure** — it is recorded as such, and r=26 is still adopted by the tie-breaker on byte grounds.

**4.3 KD challenger, on-domain (the reopened D3).** span-KD vs CE, on both winners, with teacher logits from Qwen2.5-Coder-1.5B produced on the 3060 (§8 prerequisite).
Tie-breaker: **CE** (no logit cost).

**Data-coverage specification (sealed here, because it is a second variable if left implicit).** The KD arm samples only inside the resident-logit window; the CE arm would otherwise sample the whole corpus — so a naive pairing would measure *KD-on-subset vs CE-on-full-corpus*, penalizing the KD arm for seeing fewer unique tokens, which has nothing to do with KD. Resolution: **the screening block trains on the logit-covered slice, at full coverage (every position carries teacher signal in the KD arm)**. Rationale: this screening exists to settle a question we deliberately reopened, so it is sized for **maximum statistical power and maximum favorability to KD** — if span-KD cannot beat CE on-domain at full coverage, it will not do so at partial coverage, and D3 closes with maximum confidence. Both arms see byte-identical data: one variable.

**The covered slice is a uniform random sample** of the decontaminated training corpus, drawn with **seed = 20260719, fixed here** (the sealing date, so an auditor can verify the seed predates the corpus it draws from) and with the resulting file-list hash written into the manifest. The seed is fixed in this document rather than chosen at sampling time precisely so that "which tokens carry teacher signal" carries no residual degree of freedom, however innocent. This closes — conservatively and for now — the open observation that the "highest-value code" selection criterion for a KD subset is undefined: *which* tokens carry teacher signal is itself a design variable, and it is not stacked onto this comparison. Value-based selection is a separate, later experiment.

**Mixed-batch machinery is still built** (a declared fraction of each batch drawn from the KD window with KD loss, the remainder pure CE from the full corpus): that is the *rung-2 deployment* form specified in `PHASE64_TRAINING_PLAN.md` §2 (KD-on-subset + CE-on-rest), and it is the mechanism that ships if this screening says KD wins. The screening itself runs at full coverage; the deployment fraction is a rung-2 sizing question, not this one.

**Consequence for the 3060 calendar:** logits cover the *screening block* (all five arms train on the covered slice; only the α arms consume the logits), not a full KD curriculum — **0.05-0.10 B teacher tokens ≈ 6-12 h of 3060** rather than 24-36 h, independently recomputed by the Builder from the plan budgets. **[v6: these figures are the pre-re-measure estimate, kept as the record of the precondition this paragraph itself sets; the precondition was executed and the re-measured budget below supersedes them — they are historical and never quoted.]** **Caveat carried to launch:** the teacher/student token ratio used (0.479) is measured on TinyStories with a V=1024 student; on code, against Qwen's code-efficient tokenizer and a V=2048/4096 student, it will differ. **Re-measure the ratio on the real corpus before committing 3060 time** — the figure above is the best available today, not the final one.

**Span mechanism on code — measured on a 256 MiB trial slice, with two consequences.** Boundary coincidence re-measured rather than inherited: **43.3% (V2048) / 45.4% (V4096)**, above T2's 39-44%, and the larger vocabulary produces *more* anchors, not fewer. Segment decomposition **2.31 / 2.20 student tokens**, essentially identical to the MVE's 2.267 — the chain rule has material to work on and does not degrade toward the anchor case. (Both measured with the pre-retrain tokenizers; the ex-band retrain moves corpus packing by ≤0.2%, and the §4.3 stratified diagnostic re-derives these statistics from the production run's own logs, so nothing downstream consumes the stale figures.)

1. **Quoting rule: the span mechanism acts on about two thirds of segments, not all of them.** 32-34% of segments are exactly one student token long, and for those the span reading *is* the anchor reading. This is the tail of the distribution, not degeneration — but it bounds the mechanism's reach and is stated whenever "span beats anchor" is quoted. **Record-only stratified diagnostic, added as a mechanism check:** report the span-vs-CE difference split by segment length. If span wins but the advantage is *not* concentrated on multi-token segments, then whatever is helping is not the chain rule, and the win is recorded as unexplained rather than as confirmation. Same discipline as every other positive control here: verify the mechanism does what it claims, not merely that the number moved.
2. **A code path that was marginal at the MVE is now load-bearing.** Segments cover **1.47-1.57 teacher tokens** on code against 1.086 at the MVE — Qwen packs code into large tokens, so alignment is coarser and multi-teacher-token segments go from ~8% of cases to roughly half. **Prerequisite before the production scoring run: re-run the logit self-check on the code corpus** (at the MVE: 96.8% true-token-in-top-32, mean p_true 0.460 — the positive control that catches an off-by-one). A path exercised 8% of the time and now exercised 50% of the time has not been tested where it now matters.

**α is declared, and the challenger is read as a TREND rather than a point (added 2026-07-19, pre-launch, in response to an external critique relayed by the MM).** The arm previously inherited α = 0.5 from the MVE without naming it — and the MVE's own prereg had already recorded α as "frozen by declaration, not by evidence". That is not tolerable *here*, because §4.3 is the screening that can close D3 permanently: a single-point loss would establish only that span-KD at α = 0.5 loses, not that the teacher signal is worthless.

*Mechanism, corrected from the critique's framing:* the concern is **not** that the KD arm trains at "half CE weight". Under AdamW the update is scale-invariant — scaling a loss by a constant leaves `m/√v` unchanged and decoupled decay untouched — so magnitude is nearly a no-op. The real effect is **directional**: the arm follows a convex blend of the CE and KD gradient directions, and under span mode that blend covers ~97% of positions (against ~45% under anchor, which is why α weighs far more here than it did at the MVE). The confound is real; its mechanism is the blend, not the scale.

*Resolution:* the challenger runs at **α ∈ {0.5, 0.25}**, seed-paired, with CE as the α = 0 point of the same curve. **Read as a trend, never as a best-of** — taking the max over two α values would be exactly the multiple-comparison shopping this document exists to prevent:

| pattern over α ∈ {0, 0.25, 0.5} | reading |
|---|---|
| BPB monotone increasing in α (CE best) | KD hurts, and hurts *more the more of it there is* — **D3 closes for good, with a mechanism rather than a single point** |
| best α beats CE by > 2σ_seed | teacher-domain confound confirmed → **KD-primary re-opens at the rung-2 boundary, with an α sweep inside that gate** |
| CE best but **not** monotone (α = 0.5 better than α = 0.25) | the deficit is about mixing, not about KD → **D3 does NOT close**; carried to rung-2 as unresolved |
| all differences ≤ σ_seed | no trend; D3 does not close on noise |

This **tightens** the screening: closure now requires a coherent trend, where before a single point sufficed. Cost: one additional arm — the block goes from four arms to five, 60% → 75% of stage C (≈33% → ≈41% of a rung), priced in §9. Slice sizing and logit hours are unaffected, since the slice covers one arm's budget.

*Naming, to prevent a future misreading:* this **α is the KD mixing weight** and has nothing to do with the **α-QAT ramp** of §2; the two are unrelated parameters that share a letter.

**This screening does not flip the main run.** Its role is pre-registered as evidence for the *rung-2* default: if span-KD wins by > 2σ_seed on-domain, the teacher-domain confound of the MVE is confirmed and KD-primary is re-opened at the rung-2 boundary (where a recipe change belongs, per D1). If CE wins again, **D3 closes for good** and the sequence-level fallback is retired with it.

## 5. The main run

Full curriculum C→D→E→F on the screening winners, CE-primary. This is the reference arm and the source of the rung-1 quality claim. `--save-stage-ckpt` on: the stage-D artifact (format `mve-stage-2`: model + gstep + kdc.pos + GradScaler, optimizer deliberately absent because it is rebuilt per stage) is the shared branch point for §6.

## 6. Branch A/Bs from the shared stage-D checkpoint

Each costs only E+F (~25% of the curriculum) and forks byte-identically (verified: brief §10).

**Mandatory branch-point assertion, applied to every branch and every resume.** The fork's val BPB at the branch point must equal the parent's val BPB at the save point, automatically asserted — not eyeballed. Rationale: three separate pieces of state that cross a stage boundary have now been found missing from a checkpoint (`kdc.pos`, `GradScaler`, and the QAT α — the last being a plain Python attribute, so absent from `state_dict` by construction, which silently left a branched arm **non-ternary**). The failure mode is common to the whole class: the restored model is wrong in a way that still produces a plausible loss. A BPB discontinuity at the branch point is the general detector for all of it, and it is cheap. **Any branch whose entry BPB does not match its parent's exit BPB is void, and the run does not proceed on it.**

**6.1 Upcycle ε-identity (the D→E investigation).** Baseline upcycle vs ε-identity upcycle — at insertion the router selects one replica per slice at weight 1/8 which, with `down` already magnitude-matched ×k, reconstructs the dense output **exactly**. Implemented as a **ramp** (router logits (1−α)·constructed-identity + α·learned, Switch aux ramped from 0 in step), never a hard identity.

Three outcomes, pre-registered before the arm exists, all informative:

| | transition shock | end-of-E vs baseline arm | replica-divergence telemetry |
|---|---|---|---|
| **H1** — E repays its own insertion cost | ≈ 0.000 | **better by order +0.07** | diverges |
| **H2** — cost is intrinsic to symmetry-breaking | ≈ 0.000 | converges to baseline | diverges |
| **H3** — dead replicas | ≈ 0.000 | not better | **flat** ← the detector |

H1 → the insertion is the target and the fix is adopted at the rung-2 boundary. H2 → the E *budget* is the target, not the insertion. H3 → the ramp schedule is wrong; re-tune before any adoption.

**6.2 No-recall control** (D4 clause 2, second read, now on data that has a real recall task).
**Pre-registered demotion, sealed here before any number exists: if the recall diagnostic on code shows a benefit ≤ σ_seed against this control, recall demotes to declared-v2** and the ladder gates are re-scoped accordingly. This is the clause armed at the MVE and carried forward; it is not re-interpreted.

**The instrument, specified here because the clause is only as good as what reads it.**

*Gate instrument — code-native probes.* Identifiers defined once and used later, drawn from **held-out, decontaminated** code; the measurement is the probability mass on the correct identifier at the use point. **The gate reads the difference between the recall arm and this no-recall control on the same probe set** — a within-probe paired control (matched position and local context) is kept as a variance reducer, not as the gate. **Probe distance must exceed the attention window (win = 128)**, or the single SWA layer carries the reference and the probe says nothing about the recall tier; report a **distance curve** rather than a point estimate — the tier should help more as distance grows, and the shape is worth more than the number.

**Distance-scale rule, corrected before the gate is ever read (2026-07-19).** The gated distances are **bounded by the trained context length**, with the longest gated distance ≤ half of it. Reason, measured not argued: on the run-3 finals the calibration fires at +1.23 nats / 50% top-1 at d=8 — the instrument sees a copy when there is one — while d≥512 reads exactly zero on **both** the recall and no-recall arms, because those checkpoints were trained at seq=512 and d=2048 is 4× beyond it. **The state has no reach there for reasons that have nothing to do with the recall tier**, so a fixed 128/512/2048 scale would decay to null by construction and fire the pre-registered demotion on an extrapolation artifact. This is an *instrument correction made before any rung-1 number exists, justified by a positive control* — not a reinterpretation after a null. Distances beyond the trained context are still measured and reported, **record-only, never gated**.

**Slice sizing — re-derived by the Architect from this document's own budget (2026-07-19).** All five arms train on the same slice (that is what makes them controlled), so the slice must cover **one arm's budget**, not five. Stage C is 55% of a rung; the rung is 0.8-1.5 B student tokens; one arm is 15% of stage C → **0.066-0.124 B student tokens**. At V4096 (worst case, 2.861 B/tok measured) that is 0.19-0.35 GB of text → **0.047-0.088 B teacher tokens → ~7-13 h** at the measured 148.5 h/B. The Builder's 11-22 h correctly inherited the *superseded* 25%-of-stage-C base from before the chained-controls change cut the block from six arms to four; the base moved, so the hours move with it, exactly as he flagged. **Produce to the top of the bracket plus a declared 25% margin ≈ 16 h**, so that no arm can ever wrap the slice and silently take a second pass over it — a quiet second epoch is a recipe deviation. Surplus is simply unconsumed.

**Logit budget, re-measured on the real corpus as required before any 3060 time (2026-07-19).** Teacher/student ratio on Python: **0.637 (V2048) / 0.708 (V4096)**, against 0.479 measured on TinyStories at V=1024 — an under-estimate by a third to a half, exactly why the re-measure was made a precondition. (The 0.708 here is pre-retrain; the ex-band BPE retrain of v6 moves it to **0.709**, a 0.001 shift that changes no hour figure — stated so the two numbers don't read as an error.) Screening logits therefore cost **~8-18 h of 3060**, the exact figure set by the vocab that wins §4.1. Qwen packs Python at 4.025 B/token, which is what drives the ratio up on code relative to prose. **Scheduling consequence: this is off the calendar entirely** — only arms 4-5 consume logits, and they run last (in parallel, sharing the chained control), so the 3060 produces them while Kaggle runs arms 1-3. Size the production to the V4096 (worse) case, or produce the V2048 amount and top up monotonically.

**Consequently, stage E's context extension is declared, not left implicit:** the curriculum already specifies "progressive context" at stage E (`PHASE64_TRAINING_PLAN.md` §4); rung-1 extends to **seq = 2048**, with micro-batch reduced to keep B·L — and therefore scan activation memory and tokens per optimizer step — constant. Gated scale then: **128 / 512 / 1024**, plus d=8 as the standing calibration sanity, plus 2048+ record-only. Without the extension the recall tier would be judged in a regime where the state still reaches, which is not the question the tier exists to answer. Rationale for code-native as the gate: this is associative recall in the form the product actually needs, and it needs no separate synthetic corpus to decontaminate.

*Calibration — synthetic MQAR, run once, NOT gated.* Its role is the **positive control for the detector**: an instrument never shown to fire is an assumption, not a measurement (the same discipline applied to the fork-check, which was validated by deliberately re-introducing the α bug). It establishes that the apparatus can resolve a recall benefit that exists, and it fixes the noise floor.

**The 2×2 is read PER DISTANCE, not once globally** (added 2026-07-19, before any rung-1 number, in response to a measured risk: a model whose long-context exposure is confined to the curriculum's tail may have no reach at 2048 regardless of the recall tier — the same extrapolation artifact that produced this morning's false null). The synthetic calibration is therefore run **at every gated distance and on both arms**. This also yields the decomposition for free: the no-recall arm's curve is the *state's own* reach; the difference between arms is the tier's contribution.

- At distances where the calibration fires, the code reading is **gated** as written below.
- At distances where the calibration is null, that distance is **untested** and contributes nothing to the demotion decision — neither for nor against.
- **Anti-escape-hatch clause, sealed with the rest:** if the calibration fires at **no** distance beyond the attention window (win = 128), the rung-1 recall gate is declared **UNREAD** — not passed. Recall's status then carries to rung-2 unchanged, with the instrument or the context schedule fixed first. Recall does not get to survive on "everything was uncalibrated", and it does not get demoted on it either.

**Scheduled early read — protects the spend, not just the inference (added 2026-07-19).** The rule above keeps a null at an uncalibrated distance from being misread, but it does not stop us from *paying* +34% for a context extension that may not transfer. Tail extension giving the state real reach at 2048 is an open empirical question that nobody has measured. It becomes measurable at near-zero cost on **the first stage-E checkpoint that exists with the extension applied**: run the calibration there, before the remaining arms pay for it. Pre-registered decision tree:

| calibration at that checkpoint | action |
|---|---|
| 2048 fires | proceed as declared; gated scale 128 / 512 / 1024 |
| 2048 flat, 512 fires | tail extension transfers only partially → **cut the gated scale to what is demonstrated and drop the extension to that level on the remaining arms** (recovers most of the +34%) |
| both flat | tail extension does not transfer at all → **stop paying it entirely**, gate stays inside the trained window, and the long-range recall question is recorded as **untested at rung-1** — it carries to rung-2 with a genuine progressive schedule rather than a tail patch |

**Why the extension is worth +34% at all:** the alternative is a recall gate that reads UNREAD, which pushes an undecided architectural component onto S1 (105 M), where settling it costs several times more. Paying to settle it at the cheapest rung is the same economics that put sparse-MoE's first long run at S0.

**Pre-registered reading of the 2×2, fixed now so no cell is interpreted after the fact:**

| synthetic | code | reading |
|---|---|---|
| + | + | recall **stays in v1** |
| + | − | mechanism works, code does not need it → **demote to declared-v2**, cleanly diagnosed |
| − | − | the tier is not being learned or used at all → **demote**, but flagged as a *training* failure (InfoNCE / λnce), and the architectural question is recorded as **untested, not answered** |
| − | + | incoherent → apparatus fault; investigate before the gate is read at all |

This specifies what the gate reads; it does not soften it. A null from a **calibrated** instrument demotes recall exactly as written.

**6.3 Dense-paired baseline** (gate 1 of DECISIONS §10) — **definition sealed: ACTIVE-parameter-matched**, realized as the pre-upcycle QAT-dense checkpoint continued from this same branch point at equal step budget (probe-4 pattern, as `PHASE64_TRAINING_PLAN.md` §4 specifies). The dense checkpoint is ~11.0 M against the MoE's ~11.2 M active, so the pairing is natural and costs one E+F-length branch rather than a separate training run.

Why active and not total: the product thesis is **active ≠ total** — we pay inference cost in active parameters and buy quality with total ones. Active-matched is therefore the claim that matters: *at equal running cost, the MoE beats the dense model*. Total-parameter-matched (a 30 M dense model) answers a different and more academic question, would require a full separate run, and is not our claim.

**Honesty constraint attached to this gate, binding on how the result is ever quoted:** active-matching is favorable to the MoE by construction (2.7× the parameters at equal active cost). The claim is stated as *"at equal active cost, +X BPB"*, never as an unqualified "MoE beats dense", and the total-parameter ratio is reported alongside it every time. **No total-matched control is bought at S0**: the ladder itself is that experiment, spread across rungs — S1 (E128, 105 M total at the same active) is what demonstrates quality scaling with total parameters at fixed running cost, and it is already gated by the D2 reading rule in §7.

**6.4 Compile gate** (measured in the first session, folded in per P61 — a compute-bound microbench does not compose onto a memory-bound engine): (a) two fresh processes, same seed, compiled → bit-identical; (b) resume across a session boundary numerically continuous (`mode="default"` pinned, autotuning not used — already closed in code); (c) compiled-vs-eager loss overlay within the AdamW-8bit tolerance; (d) net-positive over a 12 h session **including** warmup, counting one recompile per stage transition (four). **If (a) or (b) fails, compile is out of gate-bearing runs entirely** — the A≡C bit-identity diagnostic outranks throughput.

**RESOLVED 2026-07-19 — the rule fired: compile is OUT of rung-1's gate-bearing runs; every arm runs eager.** The local screen (WSL, torch 2.6.0+cu124, Triton present) returned (a) PASS 2.0931/2.0931, (c) PASS 0.0001 overlay, and **(b) FAIL**: against an eager control on the identical protocol that resumes *exactly* (2.0932 → 2.0932), the compiled arm gave 2.0885 / 2.0915 / 2.0915 across three repeats — not merely different from uninterrupted, but **mutually inconsistent, with two repeats agreeing despite different session counts**. The result depends on something that is not the training state (hypothesized: Inductor guard specialization against the first tensors a process sees, and/or compilation-cache state — mechanism is hypothesis, the measurement is not). `mode="default"` was already pinned and autotuning already excluded, so the pin does not close the hole.

**Why a screen may fire this rule when a passing screen could not certify:** the asymmetry is real and is banked as a principle — *a screen can falsify but not certify*. The hypothesized mechanism is a property of Inductor, not of a GPU model, so the failure is expected to transfer; and compile is opt-in, so the burden of proof sits with it. We do not spend Kaggle session-hours re-confirming a reproducible failure.

**Cost of the decision, stated plainly: it is the largest optimization measured so far** — 1488 vs 825 tok/s (+80%) at smoke config with a warm cache. It is declined anyway, because the bit-identity diagnostic is load-bearing apparatus and that trade was already adjudicated. **Parked with a concrete re-entry lead** rather than abandoned: if the failure is first-tensor guard specialization, then compiling against a *canonical warm-up batch at process start* — identical for a fresh process and a resumed one — should remove the dependence; that hypothesis is testable locally at zero cost. Second recorded datum for whoever picks it up: the compilation cache persists (3.5 min first run, 0.7 min second), so on Kaggle a cache living in `/kaggle/working` under Persistence would be paid once rather than per session, which changes criterion (d)'s arithmetic. Re-entry is a rung-boundary item, only if calendar bites.

## 7. Gates read at rung-1 (DECISIONS §10 property gates + this rung's additions)

1. **Quality vs matched dense** (§6.3) — the rung gate.
2. **Sparsity bands** 92/79 reproduced.
3. **Router health + i.i.d.-union sanity** (locality at scale = finding, not failure).
4. **Recall** — code-native probes (gate) + synthetic MQAR (calibration), with §6.2's demotion clause and its 2×2 reading.
5. **QAT-gap** recorded vs scale.
6. **ε-identity continuity at every switch** (project law): each transition shock reported; a violation is a finding to explain, not a number to bury.
7. **D2 reading rule** (opened by the MVE, sealed here): the **replica-collapse ablation is a standing instrument at every rung**. S0's value anchors the series. At S1/E128, if the differentiation gain is not materially larger than S0/E32's, **D2 is re-opened at that boundary** — E as the sole capacity dial would not be delivering. This is a reading rule, not a gate change.
8. **Throughput** measured on 2×T4 → the rung-2 cost table (the 3860 floor is checked here, per-protocol).

HumanEval stays **record-only** (no gate — anti-Goodhart). Decode hygiene locked.

## 8. Launch prerequisites (all must be true before the owner launches)

- [ ] HF Read-type token with Stack-v2 terms accepted; AWS S3 credentials (owner; env only, never committed)
- [ ] WS3 pipeline complete, **P62 decontamination J=0.5 executed**, manifest + hash recorded here and verified at training start — manifest hash: `__________` (filled at launch)
- [x] **Licence posture — RESOLVED 2026-07-19 by the pre-set rule** (≥40 GB strict-permissive in absolute terms → take it, Architect closes; below target → owner call). Census returned 821 GB point / 382 GB floor versus 40 GB needed, so the rule fired cleanly: **strict-permissive only, enumerated whitelist, Python-scoped** — see §2. The owner was not troubled with a trade that measurement dissolved.
- [ ] WS4 apparatus smoked and STOPped
- [ ] **Every long-running job emits a periodic liveness signal, and the content fetch is resumable/idempotent.** Measured: the knee is 64 workers (0.74 MiB/s, 121 files/s, 0 loss over 7500 fetches) and ~15.3 h for the 40 GB target — a one-time, $0 cost that overlaps training, so it is **not optimized further**; beyond the knee throughput *degrades* rather than plateaus (candidate causes include GIL serialization of gzip/decode and per-thread connection overhead — hypothesis, unmeasured, and only worth revisiting if calendar bites). A 15 h job that cannot resume is a 15 h job that restarts from zero, and one that only prints at the end is one whose stall is invisible by definition.
- [ ] Challenger teacher logits produced for the screening block — sized to it only (~8-18 h of 3060 at the re-measured ratio, off-Kaggle; the 6-12 h first estimate predated the re-measure), covered slice drawn with seed 20260719, file-list hash in the manifest, teacher/student token ratio re-measured on the real corpus first. **Production launched 2026-07-21**: 120.6 M teacher tokens ≈ 22 h at the sustained 1550 tok/s — a declared +9.7% over the top+25% sizing, produce-side only (arms consume declared budgets; surplus is unconsumed)
- [ ] This document pushed, and the MM's commit visible on origin

Token budgets, per-arm session counts, and the data manifest hash are the only fields filled at launch. Everything above them is sealed now.

## 9. What rung-1 costs, stated openly

Arm inventory: **screening 5 arms × 15% of stage C ≈ 41% of a rung** + **main run 100%** + **three E+F branches ≈ 75%** ≈ **2.2 rung-equivalents**, i.e. roughly **+116% instrumentation overhead** against the +50-80% assumed in `PHASE64_TRAINING_PLAN.md` §5. (v6: sum recomputed for the fifth arm — the 41% was updated at v5 but the total below it still read 2.1 / +108%, the four-arm arithmetic; caught in MM review. The fifth arm adds ~4% to the session-hour brackets below, inside their stated coarseness — declared here rather than re-bracketed.)

This is declared rather than trimmed, and the science is not cut to fit a bracket written earlier — **costs re-price, gates do not**. Two reasons it is the right call anyway: rung-1 is where the recipe itself is validated and three open questions are settled, so heavy instrumentation is exactly what it is for; and **the overhead is front-loaded and does not repeat** — the screening block decides vocabulary and x_proj once for the whole ladder, so S1 and S2 carry only their per-rung property gates and the standing replica-collapse ablation. S0 re-prices to roughly **70-150 session-hours**; against a 600-1000 h ladder this stays inside the $0 path. **Plus the measured +34% for the stage-E context extension** (§2) — carried into the STOP-B table rather than absorbed silently, and subject to the scheduled early read in §6.2 that can recover most of it if tail extension turns out not to transfer. S0 lands around **95-200 session-hours** in the worst case, still inside the $0 path on a 600-1000 h ladder.
