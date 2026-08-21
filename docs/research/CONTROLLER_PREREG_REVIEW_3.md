# Controller re-review #3 — stage −1 pre-registration (§7.2 pre-run)

**Target:** `DONOR_ADAPTATION_FIRST_RESPONSE.md` §3, v4.
**Priors:** `CONTROLLER_PREREG_REVIEW.md` (NO, 7 BLOCKs), `CONTROLLER_PREREG_REVIEW_2.md` (NO, B1/B2 open
plus 2 new). Read-only protocol audit.

> **Verdict: NO — third time, and not because the Principal failed to act.** Most of pass 2 was
> implemented correctly and some of it well. **Two BLOCKs stand: B1 (§3.5b names the fields and fills in
> no values) and the new Δ\* = 0.30 (B2', re-anchored, and falsified by the one row in the table that
> matches the donor).** Plus five closure claims in §3.7c/§3.8 that the sections they point at do not
> contain.

---

## (a) Is Δ\* = 0.30 defensible? **No. It is re-anchoring, and the table refutes it.**

The Principal asked to be attacked here. The attack is short and it is arithmetic.

0.30 is set "just above the verified closed-form ternary frontier (0.27)". **0.27 is not the frontier. It
is one row.** Recomputing every row of the PT²-LLM table the Principal now treats as the authority:

| model | PPL ratio | bits/token | **ΔBPB @ B = 4** |
|---|---|---|---|
| LLaMA-13B | 1.79× | 0.840 | 0.210 |
| LLaMA-65B | 1.88× | 0.911 | 0.228 |
| LLaMA-2-70B | 1.89× | 0.918 | 0.230 |
| LLaMA-7B | 2.01× | 1.007 | 0.252 |
| **LLaMA-2-7B — the row chosen** | 2.113× | 1.079 | **0.270** |
| **Qwen3-14B — the only Qwen row** | 2.58× | 1.367 | **0.342** |
| LLaMA-3-8B | 5.24× | 2.390 | 0.597 |

**The closed-form frontier is a range, 0.21 – 0.60 b/byte, and the donor is a Qwen.** The single most
relevant published row — same family, same method, same bit-width — is **0.342, which FAILS Δ\* = 0.30.**
Now add the two trends the verification states explicitly: degradation *worsens* as models shrink, and
*worsens* as models get more token-saturated. Qwen2.5-1.5B is small **and** modern **and** Qwen. Its
expected value under a flawless implementation is **≥ 0.34, plausibly well above.**

**So Δ\* = 0.30 has a high prior of FAILing when nothing is wrong.** That destroys its stated purpose. It
was meant to separate "our apparatus is broken / this scale is hopeless" from "we reproduced SOTA". It
cannot: a FAIL at 0.30 is unattributable across a broken apparatus, a Qwen-family penalty and a 1.5B
penalty — three causes, one bit of output. This is the v3 error in a new coordinate: **the anchor is again
a single cell, lifted across model, family and scale.**

**A fourth instance of the same failure, and it is live.** §3.6 correctly orders that the *anchors* be
recomputed at measured `B` — and then freezes Δ\* at **0.30, a number computed at B ≈ 4.0**. If the pinned
slice measures B = 3.5 the L2-7B frontier is **0.308**; at B = 3.2 it is **0.337**. **Δ\* drops back below
the frontier and the v3 inversion returns.** Δ\* must be a **formula in measured B**, never a frozen
constant.

**(a), second half — does "reproduce SOTA or stop" confuse apparatus validation with science? Yes, and the
Principal's own suspicion is correct.** "Did we implement it right?" is **control 1a/1b's** job: reproduce
a published number *on the published model and harness*, where a discrepancy is attributable. Δ\*(A)
attempts that same job on a different model at a scale where **no published number exists**. Where (A)
would be valid it is redundant with control 1; where it is not redundant it is unattributable. **Cut (A).**
Its legitimate content is already covered — apparatus by control 1 (with a stated tolerance, still
missing); scale by the curve with the anchor *range* overlaid, plus the ScaleQ-1.58 Qwen3-1.7B datapoint
the project already holds. If a stopping rule is genuinely wanted, make it attributable: *stop if control
1a misses its published anchor by more than its pre-stated tolerance* — a statement about the apparatus
and nothing else.

## (b) Is B1 closed? **No. §3.5b is a schema, not a pin.**

§3.5b lists the required fields — "corpus identity + sha256, byte range / offsets, tokenizer name +
revision, context length AND stride, BOS handling, loss reduction, decontamination" — and supplies **not
one value**. No corpus is named. No sha256 exists. No stride, no context length, no `BPB_donor`, no `B`.
The section closes with *"A missing field here is a BLOCK"*, which is self-adjudicating: **every field is
missing.** v3 denied that pinning was needed; v4 concedes it and still has not done it. Real progress in
reasoning, zero progress in artefact — and this is the cheapest open item on the page.

Minor, and evidence the edit was mechanical: **§3.5b sits physically after §3.6** (file order 3.5, 3.6,
**3.5b**, 3.6b), and the "Asymmetry — PASS says NOTHING about S4" paragraph has been orphaned into §3.5b,
where it does not belong.

## (c) Did the (A)/(B) split introduce inconsistency with §3.3? **Yes — (A) contradicts §3.3 and §3.4.**

§3.3 re-scoped ternary from hypothesis to "the expected-bad end of a curve". §3.4 now says the deliverable
is "the curve and its knee, **not a pass/fail**". **(A) then re-imports a pass/fail at the ternary point
and lets it kill the route.** A route decision keyed to the outcome the document already predicts, at the
point the document already declassified as the hypothesis, is the old gate wearing (B) as cover. (B) alone
is consistent with §3.3 / §3.4; (A) is the residue of the framing §3.3 retired.

## (d) What the rewrite broke, or claims falsely

1. **§3.7b still reads "fp16 → 4 → 3 → 2 → ternary".** §3.4 and §3.8 were corrected to bf16; §3.7b was
   not. Fixing one of two sites re-creates the contradiction somewhere else.
2. **§3.7c mislabels F4 as F6.** The control-4 item is **F4**. **F6 — dense donor versus sparse-MoE
   target, per-expert calibration coverage — is still not closed, and is now recorded as closed under the
   wrong label.** `grep -i expert` over §3 still returns zero hits. A corrupted audit trail is worse than
   the omission it papers over.
3. **§3.7c claims F2 fixed "in §3.5". §3.5 is byte-for-byte unchanged** — still "Composition mixed, not
   code-only… Pinned by content hash", with no sources, proportions, sampling procedure or seed.
4. **§3.7c claims the method repo is pinned by commit SHA. §3.8's install line still reads only
   `pip install datasets accelerate`.** No SHA appears anywhere in §3.
5. **§3.8's dtype paragraph still says "the quantity being measured is ΔBPB ≤ 0.10 with an SE around
   0.007".** Both are dead: 0.10 was withdrawn in §3.6, and 0.007 derives from the σ_seed import §3.6b
   deleted. **τ (control 6) = 0.1 × band half-width would inherit that struck constant.** A retracted
   number surviving downstream is exactly the mechanism §2.7c names.
6. **F4's second half is still absent** — the architecture-sensitive numerical probe (fixed-prompt fp32
   logits hash in the manifest, asserted at every arm's load) and the tied-embeddings note for Qwen2.5.
7. **Unchanged from pass 2, still open:** control 1's numeric tolerance (promised three times, never
   stated); the lesion ladder's ε grid, rung count, named non-redundant structure and failure-to-fire
   response — and it still says "the ε at which ΔBPB reaches **the gate width**", now ambiguous between
   0.30 and the band; control 5's `k·SE` threshold and expected direction; §3.7b's missing quantizer
   levels, symmetric/affine, scale dtype, clipping procedure and RNG seed; the 12 GB VRAM ceiling against
   1b (7B bf16 ≈ 13.5 GB does **not** fit the 3060); LLaMA-2's gated licence.
8. **Cosmetic, but this document has earned the paranoia:** the historical v1 blockquote in §3.6 still
   contains "PT²-LLM claims **1.08×**" with no inline retraction marker. Three fabrications in this project
   came from lifting a cell out of context. Strike it through where it sits.

## What is genuinely closed

**§3.8 is now good work.** The TF32 correction is accepted and carried to its consequence (inert under
bf16); `allow_bf16_reduced_precision_reduction` is identified as the live default; flags are read back
rather than assigned; **control 3 is correctly demoted with the right justification** (blind to
perfectly-repeatable precision loss); control 7 fires in both directions; control 6 compares the delta
against an fp32/fp64 oracle at a derived τ; the device pin is UUID + `PCI_BUS_ID` + `device_count() == 1`
+ materialised-parameter assertion + `device_map="auto"` forbidden. **N6.1, N6.2, N6.3, N6.4 — CLOSED.**
N6.5 closed except for §3.7b's stale "fp16". **B4 — CLOSED**: t(n−1) accepted, n ≥ 10 on the
calibration-draw axis, and the degenerate-seed-axis check is exactly the right instinct. **F5 / §3.1 —
CLOSED**, cleanly. B2's *diagnosis* is fully accepted, and the (A)/(B) separation is the right idea badly
instantiated.

## Verdict

**NO.** Blocking:

1. **Fill §3.5b with values, not field names**, and record `BPB_donor` and measured `B`.
2. **Cut Δ\* = 0.30, or re-derive it as a formula in measured `B`** against the anchor **range**
   0.21 – 0.60 — and state that the Qwen row sits at **0.342, above the proposed line.**
3. **State control 1's numeric tolerance.**
4. **Fix the five false-closure claims** in §3.7c / §3.8, restore **F6** under its own label, and purge the
   stale 0.10 / 0.007.
5. **§3.7b's "fp16" → bf16.**

Everything else is one paragraph each and was already listed in pass 2. **The reasoning is now sound; the
artefact still is not.** I am not softening because the Principal has failed twice: three of the five
blocking items are text the Principal has already agreed to and has simply not yet written down.
