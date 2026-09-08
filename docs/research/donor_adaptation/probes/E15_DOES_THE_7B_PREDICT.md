# E15 — does the artifact the goal is quoted on actually predict?

Pre-registered `7c4243f`, amended `c92ffd4` (the fold axis and a re-derived prediction), both
pushed **before any arm ran**. Runner `engine/e15_donor7b_bpb.py`. Engine: the E13 build,
`--threads 6`, **`--seqlen 512` passed explicitly**.

Slice: density `heldout`, 24 × 512, `ids_sha256 a1a48dc9fc5a6dc1`, **51,870 scored bytes, 12,264
predicted, 4.229452 bytes/token** — the same slice E1, E2, T2 and E12 use. Chance line
`log2(152064)/4.229452` = **4.070106 BPB**.

---

## 0. VERDICT — `DOES-NOT-PREDICT`

**`qwen25-coder7b_p.bin` scores 1.229094 BPB WORSE than guessing uniformly over the vocabulary.**
It is the artifact every tok/s this programme has ever quoted on a real donor was measured on.

| arm | what it is | BPB | vs chance |
|---|---|---|---|
| **B0** | fp32 Coder-7B — **the planted control** | **0.674026555** | **−3.396080** |
| **B1** | packed, `rule = R0`, `fold = layers`, `--head-ternary` | **5.299200075** | **+1.229094** |

- **G-Q0 FIRES.** `BPB(B0) = 0.674027 < 1.000`. The control was required to fire or E15 was void;
  it fires by a margin of 3.4 BPB. **The 7 B path through this engine is sound.**
- **G-Q1 = `DOES-NOT-PREDICT`.** `+1.229094` against a band of `0.000896`, which is 1,372× the
  band. No vocabulary convention, slice choice or protocol detail is within three orders of
  magnitude of closing it.
- **G-Q2 = `+4.625174`, descriptive only.** Per E12 §2 that distance is **not** damage: above the
  chance line BPB measures how confidently wrong a model is, not how broken it is.

**The engine is not the problem and the donor is not the problem. The conversion is.**

---

## 1. The donor is excellent, and it is the best one here

B0 is directly comparable to every fp32 baseline already on this slice — same ids, same 51,870
bytes, verified to round-trip byte-identically under the Coder tokenizer (§4).

| donor | fp32 BPB | chance line | vs chance | source |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.871795 | 4.069819 | −3.198 | E12 §1 (PyTorch) |
| Qwen2.5-1.5B | 0.767595 | 4.069819 | −3.302 | E12 §1 (PyTorch) |
| Qwen2.5-3B | 0.724450 | 4.069819 | −3.345 | E12 §1 (PyTorch) |
| **Qwen2.5-Coder-7B** | **0.674027** | 4.070106 | **−3.396** | **E15 (engine)** |

**Two conventions, and they are interchangeable here.** The first three are PyTorch numbers on the
full 24×512 slice; B0 is an engine number. E1 §2.1 measured both on the identical 0.5 B arm and
they agree to `+1.5347e-05` — four orders of magnitude below any gap in this table — so the column
is safe to read down. (E1's 1.5 B engine fp32 arm is a 4-sequence subset and E1 says explicitly it
is not a number to quote; it is not used here.) The chance line differs in the last column because
the Coder config vocab is 152,064 against Qwen2.5's 151,936; the 0.000287 between them changes
nothing.

**Monotone in scale, and the 7 B is the best model this programme has ever run.** The corpus is
40% pg19 / 25% markdown / 25% python / 10% wikitext — prose-majority — and the code-specialised
donor still wins, so the concern recorded in the brief §6 (that specialisation would cost more
than scale buys) is **refuted**. This also bears on the standing open question of whether to pull
a real ~10 B donor: on this evidence bigger donors keep paying, in fp32.

**The smoke agrees in direction.** On the pre-registered 2-sequence smoke (4,396 bytes, chance
4.002055): B0 `0.868184933` (−3.133870), B1 `5.422806770` (+1.420752). Both arms sit on the same
side of the line at both slice sizes, and the smoke's B0 is 0.194 higher than the full slice —
which is why the band in §5 was written for the full slice and the smoke was not read as a result.

## 2. The conversion destroys it

Same weights, same engine, same slice, one difference — the export rule and format:

```
fp32                     0.674027   predicts, and predicts well
packed R0 head-ternary   5.299200   1.229 BPB worse than a uniform guess
```

E12 established that `R0` sits at or above the chance line at 0.5 B, 1.5 B and 3 B. **E15 adds the
7 B cell and it is the worst of the four**: R0's arms read +0.518 / +1.436 / +1.665 above chance at
0.5/1.5/3 B (E12, FFN+ATTN, fold none) and this reads **+1.229** at 7 B with more organs converted
(all attention, all FFN, *and* the head) and the fold applied. **R0 does not become survivable at
scale.**

**What was already on the record and never quantified.** E7 §5 wrote, of this same artifact
answering every prompt with `ERCHANTABILITY` repeated to the end:

> "R0 at 7 B collapses completely, and how much of that is the rule versus the scale is a separate
> experiment nobody has run."

E7 declined to report a BPB deliberately and said why. **E15 is that experiment, and the sentence
was right.** What happened in between is that 4.46 and then 6.79 tok/s went on being quoted as the
rate on the real donor while the sentence stayed true and unmeasured.

## 3. What this does NOT touch — every speed number stands

**No timing was taken in E15 and none may be quoted from it.** More importantly, nothing here
moves a rate that was already measured:

- **A ternary weight costs the same bandwidth whatever scale multiplies it.** The kernel reads the
  same bytes in the same order and does the same work; the per-row scale is a multiply at the end.
  E7/E8/E9/E10/E11/E13 measured the *engine*, and they still do.
- **6.79 tok/s on the real 7.072 B active remains exact**, as does E8's 1.349×, E9's 1.056×,
  E10's `CORE-BOUND`, E11's `NO-LIFT` and E13's 1.358× lever.

**What E15 removes is the right to describe 6.79 tok/s as a rate for a working 7 B model.** It is
a rate for this artifact, and this artifact does not predict.

**The format's speed value, as a ratio.** E7's generate stage ran both arms in the same run under
the same conditions: fp32 **1.382–1.392 tok/s**, packed **4.853–4.888** — a **3.51×** ratio. Those
are generate-stage figures, *not* witnessed `--bench` rates (E7 §11's contention witness postdates
them), so the ratio is the durable part and the absolutes are not quoted. **The honest position:
this programme has a 7 B that predicts and a 7 B that is 3.51× faster, and they are not the same
7 B.**

## 4. Two things checked rather than assumed

- **The tokenizer crossing.** `common.get_slice` caches on `(part, n_seq, seq_len, seed)` and
  **not** on the tokenizer, so a probe changing donor family silently scores the previous family's
  ids. E15 is the first to cross Qwen2.5 → Qwen2.5-Coder. All 24 sequences decode identically under
  both tokenizers, re-encode to byte-identical ids under the Coder tokenizer, and give the same
  per-sequence byte counts — **51,870 bytes, 4.229452 bytes/token, unchanged**. Both tokenizers
  report `len = 151665`; the max id in the slice is 143,296. The cache-key gap is a latent defect
  and is logged in the brief §9.
- **The protocol.** `donor_engine.c:1249` defaults `SL = n`, i.e. the whole ids file as one
  sequence. E15 passes `--seqlen 512` explicitly, so its numbers sit in the same column as E1's,
  E2's, T2's and E12's. (E14's harness does *not*, which is why E14's A0 reads 4.629292 where E1
  reads 4.531234 on an identical file.)

## 5. Predictions, scored

| prediction | registered | measured | outcome |
|---|---|---|---|
| **B0** fp32 control | **0.55 – 0.85** | **0.674027** | **INSIDE** |
| **B1**, first draft (withdrawn) | above chance, **4.6 – 6.5** | 5.299200 | would have been **INSIDE** |
| **B1**, amended (the one that stood) | **3.4 – 5.2**, direction deliberately **not** called | 5.299200 | **MISSED by 0.099**, and the refused direction was correct |

**The amendment made the prediction worse, and the brief said why before the run.** The first draft
reasoned from E12's unfolded R0 arms and landed. The amendment composed E2's fold credit (−0.529,
measured at **0.5 B under R3**), T2's rule penalty (+1.600, measured at **1.5 B, FFN-only, fold
none**) and a scale term — three different configurations — and the brief called that "the exact
move E12 was written to forbid" and did it anyway to set a bound. **It degraded the estimate: the
fold credit measured under R3 did not carry to R0 at 7 B.**

That is the lesson, and it is E12's lesson arriving from the other direction: **composing measured
deltas across configurations is not conservative just because each delta is measured.** Refusing to
call the direction was the part that cost most — the evidence (E12's R0 sweep, E7's 0/160 greedy)
did support the call that was refused.

## 6. What E15 cannot claim

- **It does not price R3 at 7 B.** That is B2, §7, and it is now the most consequential open
  measurement in the programme.
- **It does not separate the rule from the fold.** B1 differs from the standing 0.5 B/1.5 B
  artifacts on both axes. §7's design fixes the fold and moves only the rule, which is why it can
  attribute.
- **It says nothing about speed**, and it does not move the goal's arithmetic: §19.3's remaining
  gap is unchanged.
- **One slice, one corpus mix, one donor.** The fp32 sweep in §1 is four points on one corpus.

## 7. Owed — B2, and it is the experiment that matters now

**Re-export the 7 B with `--rule R3 --calib-seqs 32`, keeping `--fold layers` and
`--head-ternary`, so the RULE is the only variable against B1**, and re-measure on this slice.

**This is E7 §12 owed item 3** — "R3 at 7 B, if anyone wants a clean scale statement about the
ternary damage." E7 filed it as optional. E15 promotes it: it is no longer about a clean scale
statement, it is about whether a ternary 7 B that predicts exists at all.

**Why it is worth the hours.** R3 is the rule the pipeline actually ships for every other standing
artifact (`e1_bpb_through_engine.py:395`), and **it works at 1.5 B**: E1 reads `TQH` at 3.475707,
**0.594 BELOW chance**. So a ternary 7 B that predicts is not ruled out by anything measured — it
has simply never been built. B1 was exported before `--rule` existed as a habit and took the flag
default.

**What it costs.** R3 needs a 32 × 512 calibration forward pass through the 7.6 B model, and it
**refuses the bf16 loader** (`qwen_export.py:243`), so the export runs under a float32 load at
~30 GB resident — which is exactly why E7's low-memory path exists and why that path could not
produce an R3 artifact. Then a 5.7 GB write and a ~31 min BPB run. Estimate 2–3 h, one heavy job.

**Secondary, cheaper, and worth doing in the same pass:** B3, the same export with `--fold none`,
which would separate the two axes E15 could not.

**If B2 also lands above the chance line**, then no rule in this exporter produces a working 7 B,
and the binding constraint moves from the rule to the format — which is the question
`SCALEUP_ARCHITECTURE`'s thinking/knowing split was designed around, and would make the case for
training into the format rather than converting into it.
