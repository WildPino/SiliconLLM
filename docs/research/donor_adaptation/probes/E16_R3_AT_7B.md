# E16 — is there a ternary 7 B that predicts?

Pre-registered `be65920`, runner `2247923`, both pushed before the export ran. Control result
`34e7ba3`. Follows E15 (`DOES-NOT-PREDICT`) and closes E7 §12 owed item 3.

Slice: density `heldout`, 24×512, `ids_sha256 a1a48dc9fc5a6dc1`, **51,870 scored bytes, 12,264
predicted, 4.229452 bytes/token**, `--seqlen 512` passed explicitly, `--threads 6`, the E13 engine
build. Chance lines: **4.070106** at `V = 152064` (Coder), **4.069819** at `V = 151936` (Qwen2.5);
bands **0.000896** and **0.000609**.

---

## 0. VERDICT — `SCORE-CROSSES-RANK-DOES-NOT`

**The rule was most of what was broken, and fixing it does not produce a working model.**

| arm | rule | fold | BPB | vs chance | greedy vs fp32 |
|---|---|---|---|---|---|
| **C0** Qwen2.5-1.5B, control | R3 | none | **3.475706372** | **−0.594113** | — |
| **B1** Coder-7B, E15's shipped artifact | R0 | layers | 5.299200075 | **+1.229094** | **0/160** |
| **B2** Coder-7B | **R3** | layers | **4.017232598** | **−0.052874** | **0/160** |
| **B3** Coder-7B | R3 | none | 4.168325483 | **+0.098219** | **0/160** |
| — fp32 Coder-7B (E15's B0) | — | — | 0.674026555 | −3.396080 | 160/160 (E7 §2) |

- **G-R0 FIRES.** C0 reproduces E1's `3.475706691632780` to **3.20e-07** against a 0.01 tolerance,
  below chance. Predicted `< 1e-4`. The instrument is verified as *still* firing in the
  configuration that produced these nulls.
- **G-R1 = `RULE-FIXES-IT`** by the registered threshold: B2 is 0.052874 below the line, 59× the
  band.
- **G-R2 = −1.281967477.** Swapping `R0` → `R3` with the fold and every other axis held fixed is
  worth **1.282 BPB** at 7 B and carries the model across the chance line. Because B2 is below the
  line this is a damage figure, not merely a distance (E12 §2) — the first time that condition has
  been met by a ternary 7 B on this engine.
- **G-R3: B3 is +0.098219 ABOVE the line**, and `B3 − C0 = +0.692618791`.
- **G-R4 = −0.151092884.** The fold is worth 0.151 BPB at 7 B and is the *entire* reason B2 sits
  below the line while B3 sits above it.

**And every ternary arm agrees with the fp32 donor on 0 of 160 greedy tokens, diverging at
token 0.** B2's score crossed the chance line; its ranking did not move at all.

---

## 1. The headline, stated as the two metrics disagree

`B2` is **0.155 nats per token** better than a uniform guess (11.777051 against `ln(152064)` =
11.932057). That is 59× the measurement band, so it is not noise: the model carries *some*
information about the next token. It is also **0/160 on greedy top-1**, identical to E7's planted
control — the artifact E15 measured 1.229 BPB *worse* than guessing.

**Both readings are true and they are about different questions.** BPB scores a distribution;
greedy ranks it. A model can be reliably a little better than uniform across 12,264 positions and
still never place the right token first. **`RULE-FIXES-IT` is true of the band and false of the
model** — the same shape of failure E14 recorded one day earlier, and much starker here: E14's
mislabelled arm agreed on 45.6% of tokens, this one agrees on 0%.

**E14's law is what caught it**, and it is worth quoting because it was written before this run and
against a different experiment:

> a gate written as a one-sided band inherits the assumption that the treatment hurts; compare
> distances from the reference, and **pair every scoring metric with a ranking one**.

**E16's brief registered only scoring metrics.** G-R1 through G-R4 are all BPB. The ranking metric
was added afterwards, as `e16_greedy_rank.py`, and it **reports rather than decides** — no band for
it was justified from anything measured before the run, so promoting it to a gate now would be the
move E14 §6 forbade and obeyed. **The registered gate set would have reported "the rule fixes it"
about a model that never once picks the same next token as the donor it was converted from.** That
is the result, and the fact that the instrument set could not see it is part of the result.

The ranking script validates itself before it is read: **B1 reproduces E7's published `0/160`
exactly**, same prompts, same reference, and the reference is E7's own stored fp32 continuation
(`f32_p*.ids.bin`, 2026-09-07) which matched PyTorch 160/160. Each stored file is asserted to begin
with the prompt it claims.

## 2. What the rule is worth, and it is a lot

`B2 − B1 = −1.281967477`, with `fold`, `head-ternary`, quant, revision, thread count and attention
implementation all identical — enforced, not assumed (§5). **`R0` was the broken half of E15's
artifact.** E12 measured `R0` at or above the chance line at 0.5 B, 1.5 B and 3 B; E15 added 7 B and
it was the worst of the four. E16 shows the shipped rule recovers 1.282 of that 1.229-above-chance
position and lands just past the line.

**Two of the exporter's four rules are now measured at 7 B. `R1` and `R2` are not**, so this probe
does **not** establish that no rule in this exporter produces a working 7 B — it establishes that
the two that have been tried do not, including the one the pipeline ships.

## 3. R3's damage grows with scale too — the prediction that broke

The reasoning behind G-R3's band was that R3's only sweep improves with scale. **It reverses.**

| donor | R3, fold none, packed, head-ternary | vs its chance line |
|---|---|---|
| Qwen2.5-0.5B (E1 `TQH`) | 4.531233733626962 | **+0.461415** |
| Qwen2.5-1.5B (E1 `TQH`) | 3.475706691632780 | **−0.594112** |
| **Qwen2.5-Coder-7B (B3)** | **4.168325483** | **+0.098219** |

**R3 is non-monotone: above the line, then well below, then back above.** E12 established that
`R0`'s damage grows with scale; **`R3`'s does too — it simply starts from a better place and takes
one more octave to show it.** The two rules are not different in kind, only in offset.

**Caveat registered in §7 of the brief before the run and held to:** the 1.5 B → 7 B step also
crosses model families (Qwen2.5 → Qwen2.5-Coder), so "scale" here contains the code
specialisation. E15 §1 showed specialisation does not hurt this corpus in fp32 — it helped, and the
fp32 sweep is monotone — but that is an fp32 result and it does not license reading `+0.692619` as
a pure scale term. **A clean axis needs Qwen2.5-7B, which this programme has not downloaded.**

## 4. The fold is load-bearing, and it is the one thing the shipped artifact had right

`G-R4 = B2 − B3 = −0.151092884`. Without it, R3 at 7 B is **above** the chance line; with it,
below. E15's B1 was a *wrong rule* applied over a *correct* fold.

| donor | fold credit under R3 | source |
|---|---|---|
| 0.5 B | −0.529 | E2 |
| 1.5 B | −0.220 | T3 |
| **7 B** | **−0.151** | **E16** |

**Consistent in direction at three scales, shrinking in magnitude.** That is the first fold
measurement above 1.5 B, and it is the reason E15's amended prediction failed: it imported the
0.5 B credit (−0.529) into a 7 B estimate. **The credit at 7 B is 3.5× smaller.** E15 named that
composition as forbidden and made it anyway; E16 now says by how much it was wrong.

## 5. The design gate, which had to pass before any BPB was read

Brief §3 required each new arm's sidecar to differ from its reference on the axis under test and on
`bytes`/`sha256`/`mean_ternary_zero_fraction`, and on nothing else. Both passed:

- **B2 vs B1**: only `rule`, `calib_seqs`, `calib_matches_t2_operating_point` moved.
- **B3 vs B2**: only `fold`, `n_gains_folded` moved.

| artifact | rule | fold | gains | zero-fraction | sha256 | export |
|---|---|---|---|---|---|---|
| B1 | R0 | layers | 56 | 0.3572 | `d31c5047cb331f15` | (E7) |
| B2 | R3 | layers | 56 | **0.5484** | `5840b321cb62cf3d` | 1,835 s |
| B3 | R3 | none | 0 | 0.4930 | `e8f36c811a692360` | 1,672 s |

**One documented allowance, admitted against exactly one value and printed when it fired**: B1's
sidecar predates the `load_dtype` field. Its absence is a documentation gap, not a difference — the
bf16 loader *refuses* `--fold` (`qwen_export.py:240`) and B1 carries `fold: layers` with 56 gains
folded, so it was exported under the `float32` default, which `export_packed.log` confirms.

**R3 zeroes 54.84% of weights against R0's 35.72%.** The rules do materially different things. It
changes no speed number: the packed format stores every code regardless of value.

## 6. Predictions, scored

| gate | registered | measured | outcome |
|---|---|---|---|
| **G-R0** | reproduces to `< 1e-4` | `3.20e-07` | **INSIDE** |
| **G-R1** | **below** the line, band **2.3 – 4.0** | 4.017232598 | **direction RIGHT, band missed by 0.018** |
| **G-R2** | −3.0 to −1.3 | −1.281967477 | missed by 0.018 — **the same single constraint as G-R1**, not a second miss |
| **G-R3** | **below** the line, band **2.5 – 4.0** | 4.168325483 | **DIRECTION WRONG** |
| G-R4 | no band (descriptive) | −0.151092884 | — |
| export cost | 1–3 h, "real uncertainty" | 0.51 h, 0.46 h | **below the band** |

**G-R1 and G-R2 are one prediction, not two.** They constrain the same quantity — both say
`BPB(B2) ≤ 3.999` — and it read 4.017233. Reporting them as two misses would double-count one
error.

**G-R3 is the real failure, and the reasoning failed before the band did.** I extrapolated a trend
from two points in the same configuration, which the brief explicitly defended as legitimate
("one quantity measured in the same configuration at other scales") in contrast to E15's forbidden
cross-configuration composition. **It was legitimate and still wrong: two points cannot establish
curvature, and the brief said so in the same paragraph while relying on them anyway.** The lesson
is narrower and sharper than E15's — *a trend read from two points is a direction, not a law*.

**What did work**: calling a direction at all. E15's post-mortem said the costliest half was
refusing to call one. E16 called two, got one right and one wrong, and both are now on the record
in a form that can be argued with. That is the trade the pre-registration is for.

## 7. What E16 cannot claim

- **Nothing about speed.** No timing was taken. A ternary weight costs the same bandwidth whatever
  rule produced it, so **6.79 tok/s exact stands**, as do E8's 1.349×, E9's 1.056×, E10's
  `CORE-BOUND`, E11's `NO-LIFT`, E13's 1.358× lever and E14's `CHEAP-BUT-NOT-NEUTRAL`. §19.3's
  remaining gap is unchanged.
- **It does not test `R1` or `R2`**, or any calibration budget other than 32 (D4b has never run).
- **The greedy figures are a characterisation, not a gate**, and `0/160` at three arms does not
  establish a *band* for what agreement a working conversion should produce. E14 §5 item 3 owed
  that band before E16 and still does.
- **`+0.692619` is not a pure scale term** (§3).
- **One slice, one corpus mix.**

## 8. Where this leaves the goal

E15 wrote: *"If B2 also lands above the line, no rule in this exporter produces a working 7 B, the
binding constraint moves from the rule to the format."* B2 landed 0.053 **below** the line, so that
conditional did not fire as written — **and the greedy result reaches its conclusion by another
route.** A model that scores 0.155 nats better than uniform and agrees with its own donor on 0 of
160 tokens is not a working model by any reading that matters.

**The honest position after E15 and E16 together:**

1. The **engine** is closed — 97% of this machine's demonstrated streaming rate on the weight path,
   the kernel on its own ceiling at every footprint.
2. The **donor** is excellent and the best this programme has run — 0.674027 in fp32, monotone in
   scale.
3. The **conversion** destroys it, the **rule** was most of the destruction (−1.282), the **fold**
   is load-bearing (−0.151), and **what is left after fixing both is still at chance.**

**So the binding constraint is now the format at this scale, not the rule.** That is what
`SCALEUP_ARCHITECTURE`'s thinking/knowing split was designed around, and it makes the case for
**training into the format rather than converting into it** a measured conclusion rather than a
preference — with the honest qualifier that two of four rules were tested, and a clean scale axis
was not available.

## 9. Owed

1. **A ranking band.** E14 §5 item 3, now twice load-bearing: what greedy agreement do two
   *known-equivalent* models produce on these prompts? Without it, `0/160` and `45.6%` are
   observations that cannot convict.
2. **`R1` and `R2` at 7 B**, if a cheap screen at 1.5 B suggests either beats R3 there.
3. **A clean scale axis** — Qwen2.5-7B, so `+0.692619` can be attributed. ~15 GB download.
4. **The 3 B cell of the R3 sweep** (E12 §26.6 item 1) — now more interesting, not less: it is the
   point between R3's minimum at 1.5 B and its return to the line at 7 B.
5. **A fold sweep at 3 B**, to test whether the shrinking credit (−0.529 / −0.220 / −0.151) is
   smooth.
6. E14's owed items are untouched: the `--lutblk --lut-group 32` speed number, the `--seqlen 512`
   re-run, G-N1 in repaired form.

---

## 10. Appended after E17 — the reference this probe reasoned from was already a planted control

E16 chose `R3` because it "works at 1.5 B": `qwen25-15b_tqh.bin` reads `3.475707`, `0.594` below the
chance line, and §2 of this probe used that as the reason the rule was worth hours of export.

**That file is E6's arm `A3`, which E6 labelled "planted control at 1.5 B" — a known-NEGATIVE — on
2026-09-05.** It agrees with PyTorch on **10/160** greedy tokens, diverges at token 0, and emits
`" the\n\n the\n the the\n the\n the\n the\n the"`. This probe read that artifact's BPB and
never its transcript, three days after the transcript was filed in this same directory.

Nothing in §§0–9 is withdrawn: `C0` was used as a *replication* target for `G-R0`, and it replicated
to `3.20e-07`, which is all `G-R0` ever claimed. What is withdrawn is the **motivation** in §2 — "R3
is the rule that works at 1.5 B" — which was true of the chance line and false of the model, the
same error §0 names at 7 B. **The probe committed at its start the mistake it discovered at its
end.**

E17 (`probes/E17_DOES_THE_HEAD_RANK.md`) also closes §9 item 1 only *partially*: it supplies the
known-positive band (three readings, all exactly `160/160`) and the known-negative population
(seven arms, best `12/160`), which is enough to convict `0/160`. **It does not supply a band for
intermediate agreements**, so E14's `45.6%` stays unbanded, and it adds a new prerequisite — there
is no measured floor for agreement by *frequency coincidence*, so `12/160` is not yet known to beat
zero information.
