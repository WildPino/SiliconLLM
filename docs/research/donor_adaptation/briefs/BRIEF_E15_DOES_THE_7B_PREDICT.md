# BRIEF E15 — does the artifact the goal is quoted on actually predict?

**Status: PRE-REGISTERED. Pushed before any arm ran.**
Runner `benchmarks/donor_adaptation/engine/e15_donor7b_bpb.py`.

---

## 1. The question

`D:/_ktmp/e7/qwen25-coder7b_p.bin` is the packed Qwen2.5-Coder-7B artifact. Every tok/s this
programme has ever quoted on a real donor — E7's 4.46, and the 6.79 that stands today — was
measured on that file. **Its BPB has never been measured.** Not once, at any slice, against
anything.

E15 measures it, against the chance line.

## 2. Why this is being asked now

E12 closed on a rule I had not checked: `--rule R0` is `qwen_export.py`'s flag *default*, but
`e1_bpb_through_engine.py:395` appends `--rule R3` for every non-fp32 arm, so E1's standing 0.5 B
and 1.5 B artifacts are R3 and E12 had measured a rule the pipeline does not ship. Running that
same audit one scale up turns up the reverse:

```
qwen25-coder7b_p.bin.json :  rule = R0   calib_seqs = None   quant = packed   head_ternary = True
qwen25-05b_tqh.bin.json   :  rule = R3   calib_seqs = 32     quant = packed   head_ternary = True
qwen25-15b_tqh.bin.json   :  rule = R3   calib_seqs = 32     quant = packed   head_ternary = True
```

**The 7 B artifact is the only standing one exported with R0** — the rule E12 measured above the
chance line at 0.5 B, 1.5 B *and* 3 B. It was exported by `e7_real7b.py`, which never passes
`--rule`, so it took the default.

Nothing in the programme has ever contradicted this, because nothing ever looked:

- **E7's `160/160 greedy` is the fp32 arm.** `G-G` gates `f32` against the PyTorch reference
  (1.0 agreement, 160/160). The packed arm is `G-C`, the *planted control*, which E7 requires to
  **fail**: `gc = summary["packed"]["agree"] < 0.90`.
- **The packed arm scored `agree = 0.0` — 0 of 160**, and it disagrees with its own fp32
  reference at the **very first generated token on all 5 prompts**, where the reference's top-2
  margins are 1.43 / 1.54 / 1.54 / 1.45 / 2.92 nats. (E7's recorded `worst_gap_at_div = 12.436` is
  the *largest* reference margin over all divergent positions, not the margin at the first one; an
  earlier draft of this brief glossed it as the latter and that was wrong.) E7 recorded the whole
  thing as `G-C PASS`, which is correct as a control: it proves the harness can tell arms apart.

- **E7's parity gate (1.070e-05) is engine-vs-PyTorch on identical weights.** Phase 60's law is
  that kernel-bit-exactness does not compose to system correctness; the companion, which E15 tests,
  is that **parity does not compose to quality**. Agreeing perfectly with a reference built from the
  same export says nothing about whether that export predicts.

**A second axis, found in the export log.** `D:/_ktmp/e7/export_packed.log` reads
`folded 56 RMSNorm gains (--fold layers)`. E2 made `--fold layers` the exporter default, and
`e1_bpb_through_engine.py` pins `--fold none` precisely because otherwise "every E1 arm silently
changes model". **The fold is not neutral under ternarization**: it multiplies a gain into every
row before `sign(w)` and `mean|w|` are taken, so it changes the codes. The 7 B artifact therefore
differs from the standing 0.5 B/1.5 B artifacts on **two** axes, rule *and* fold, and E2 measured
the second one directly (section 3).

So the artifact carrying the headline rate diverges from its own donor on the first generated
token of every prompt, was converted with the rule E12 measured at chance, and carries a fold no
standing artifact carries. **Its BPB is the load-bearing unmeasured quantity in this programme.**

## 3. What is already known, and from which row

| quantity | value | source |
|---|---|---|
| slice: density `heldout`, 24 x 512, sha `a1a48dc9fc5a6dc1` | **51,870 scored bytes, 12,264 predicted, 4.229452 bytes/token** | `common.make_slice`; asserted at `e1_bpb_through_engine.py:353` |
| fp32 BPB, Qwen2.5-0.5B, same slice | 0.871795 (torch) / 0.871810 (engine) | `e1_05b.log:7-8` |
| fp32 BPB, Qwen2.5-1.5B, same slice | 0.767595 | E1 |
| fp32 BPB, Qwen2.5-3B, same slice | 0.724450 | E12 |
| R0 ternary (FFN+ATTN), 0.5/1.5/3 B | +0.518 / +1.436 / +1.665 **above chance** | E12 |
| R3 packed TQH, 0.5 B | 4.531234, **+0.461 above chance** | `e1_05b.log:30` |
| R3 packed TQH, 1.5 B | 3.475707, **0.594 below chance** | E1 |
| R3 **fold none** + ternary head, 0.5 B (`TQH`) | 4.531234, **+0.461 above** | E2 section 3.1 |
| R3 **fold layers** + ternary head, 0.5 B (`NLH`) | **4.001988, 0.068 BELOW** | E2 section 3.1 |
| R3 fold layers, fp32 head, 0.5 B (`NL`) | 4.178296, +0.108 above | E2 section 3.1 |
| R0 vs R3, FFN only, 1.5 B, fold none | 4.076694 vs 2.476967 — **R0 is +1.600 worse** | T2 |
| packed 7 B greedy vs its own fp32 | **0 / 160**; diverges at token 0 on **5/5** prompts, reference margins 1.43–2.92 nats | `results/e7/generate.json` |
| packed 7 B BPB | **never measured** | — |

**Tokenizer crossing, checked not assumed.** `common.get_slice` caches on
`(part, n_seq, seq_len, seed)` and **not** on the tokenizer, so a probe that changes donor family
silently scores the previous family's ids. E15 is the first probe to cross from Qwen2.5 to
Qwen2.5-Coder. Verified before writing this brief: all 24 sequences decode identically under both
tokenizers, re-encode to byte-identical ids under the Coder tokenizer, and yield the same per-
sequence byte counts — **51,870 bytes, 4.229452 bytes/token, unchanged**. Both tokenizers report
`len = 151665`; the maximum id in the slice is 143,296. The crossing is safe, and the cache-key
gap is logged as a defect in section 8.

## 4. The chance line

`log2(V) / bytes_per_token`, the same construction E12 established:

| V | source | chance |
|---|---|---|
| **152,064** | Coder-7B `config.vocab_size` (padded) | **4.070106 BPB** |
| 151,665 | tokens the tokenizer can actually emit | 4.069210 BPB |

**The padded vocab is the primary**, because the claim under test is "the artifact sits *above*
chance" and the larger V sets the higher bar. The gap between the two conventions is
**0.000896 BPB**, and that is the only quantity making the line's position uncertain.

## 5. Bands — fixed here, before any arm runs

**G-Q0 — the planted control. Must fire, or E15 is VOID.**

> `BPB(B0) < 1.000`, where B0 is `qwen25-coder7b_f32.bin` on the same slice.

The fp32 donor must predict, and by a wide margin — the three fp32 baselines already on this slice
sit at 0.87 / 0.77 / 0.72 against a 4.07 line, more than 3 BPB clear. This gate is not a quality
threshold; it asks only whether the 7 B path through this engine, at this shape, on this slice,
produces a model that predicts at all. **If G-Q0 does not fire, no reading of B1 counts** — the
planted-control law, and E12's correction to it: a control must be verified as still sanctioned,
not merely labelled.

**G-Q1 — the reading.** B1 is `qwen25-coder7b_p.bin`, the shipped artifact.

| outcome | condition |
|---|---|
| `PREDICTS` | `BPB(B1) < 4.070106 − 0.000896` |
| `AT-CHANCE` | `abs(BPB(B1) − 4.070106) <= 0.000896` |
| `DOES-NOT-PREDICT` | `BPB(B1) > 4.070106 + 0.000896` |

The band is the vocabulary-convention ambiguity and nothing else. It is derived from a measured
quantity (this slice's 4.229452 bytes/token) over a structural factor (padded vocab / emittable
vocab), per E8 section 3's corollary. It is *not* σ_seed: R0 is deterministic — `sign(w)` with
`alpha = mean|w|`, no draw — so this arm has no seed dispersion to absorb.

**G-Q2 — descriptive, explicitly not a gate.** `BPB(B1) − BPB(B0)` is reported. **If B1 lands at
or above chance, that distance is NOT to be read as damage**, per E12 section 2: above the line,
BPB measures how confidently wrong a model is, not how broken it is.

**Written as distances, deliberately.** E14's `G-N1` was pre-registered as an inequality between
two damaged models (`BPB(A2) < BPB(A1)`) and its direction inverted above the chance line — the
fifth pre-registered rule in this programme aimed at the wrong number. Every gate above is either a
comparison against a fixed line or an explicitly descriptive distance. No gate here orders two
damaged models.

## 6. Prediction, recorded before the run

**Amended before any arm ran, after reading E2 section 3.1.** The gates in section 5 are
untouched; what changed is the calibration claim, because my first draft predicted from the wrong
configuration. Recorded rather than quietly replaced.

**The first draft said "above chance, 4.6 – 6.5", reasoning from E12's unfolded R0 arms. That
reasoning did not apply**: the 7 B is `--fold layers`, and E2 measured the fold as worth
**−0.529 BPB** at 0.5 B with a ternary head (`TQH` 4.531234 -> `NLH` 4.001988). `NLH` is the
*only* 0.5 B ternary artifact on record that lands below the chance line, and it is the fold that
puts it there.

**The revised prediction is that this is too close to call, and I am not making a directional
one.** Three measured forces, two of which point opposite ways:

| force | measured | sign on B1 |
|---|---|---|
| the fold, already in the artifact | E2: `TQH` -> `NLH` = **−0.529** at 0.5 B | **down** (toward predicting) |
| R0 instead of R3 | T2: **+1.600** at 1.5 B, FFN only, fold none | **up** |
| 0.5 B -> 7 B | E1: R3 `TQH` 4.531234 -> 3.475707 = **−1.055** per 3x of scale | **down** |
| — but E12's R0 arms got *worse* with scale | +0.518 / +1.436 / +1.665 at 0.5/1.5/3 B | **up** |

Composed naively from the `NLH` anchor, B1 lands somewhere in **3.4 – 5.2 BPB**, straddling the
4.070106 line. Every one of those adjustments is an extrapolation across a different organ set,
scale or fold from the row it was measured on — the exact move E12 was written to forbid — so the
range is offered as a sanity bound, not a claim.

**That is the finding, before the run: the direction is not derivable from anything on record.**
Which is the argument for measuring it rather than for arguing about it.

**One directional datum does exist and cuts up**: the artifact disagrees with its own fp32
reference on the first generated token of all five prompts, at reference margins of 1.4–2.9 nats.
That is 5/5 independent failures at positions where the donor is reasonably confident. It is
consistent with a model at or above chance, and it is also consistent with a heavily damaged model
that still predicts — ternarization diverges fast even when it works. **It is not enough to call
the direction**, which is why it is recorded here as evidence and not as a prediction.

## 7. Protocol

- Engine: the E13 build (`donor_engine.exe`, has `--lutblk`), `--threads 6`, machine otherwise idle.
- **`--seqlen 512` is passed explicitly on every invocation.** `donor_engine.c:1249` defaults
  `SL = n`, i.e. the whole ids file as one sequence. E1 passes `--seqlen 512`; **E14's harness does
  not, and therefore scored one 12,288-token stream rather than 24 documents** — which is why E14's
  A0 reads 4.629292 on the identical file where E1 reads 4.531234. E15 uses E1's protocol so its
  numbers sit in the same column as every other quality number in this programme.
- Byte convention: `nb = len(tok.decode(ids[1:]).encode("utf-8"))` per sequence, summed = 51,870,
  over `24 x 511 = 12,264` predictions. **Charged bytes, not moved bytes**; the denominator is the
  ceiling and this brief names it once so it cannot drift.
- Order: a 2-sequence smoke of both arms first (cheap: the packed arm is ~2.5 min at the measured
  6.79 tok/s, fp32 ~13 min), to confirm the engine accepts `vocab = 152064` and `tied = 0` before
  spending the full slice.
- Arms are run one at a time. No timing is taken and none may be quoted from this run.

## 8. What E15 cannot claim

- **It does not invalidate a single speed number.** A ternary weight costs the same bandwidth
  whatever scale multiplies it; the kernel does the same work on the same bytes. E7/E8/E9/E10/E11/
  E13 measured the engine, and they still do. What a bad B1 would remove is the right to describe
  6.79 tok/s as a rate *for a working 7 B model*.
- **It does not separate the rule from the fold.** B1 differs from the standing artifacts on both
  axes at once. Separating them needs a `--fold none` 7 B export as a third arm, which is a 5.7 GB
  export and its own run.
- **It does not price R3 at 7 B.** If B1 fails, the immediate owed follow-on is B2: re-export with
  `--rule R3 --calib-seqs 32` and re-measure. That costs a 32 x 512 calibration forward pass
  through a 7.6 B fp32 model on CPU plus a 5.7 GB export, and it is a separate run.
- **It does not locate where the shipped rule crosses the line.** E1 brackets the R3 crossing to
  `(0.5 B, 1.5 B]`; the Qwen2.5 family has no cell in between, so no sweep with this donor can
  narrow it further. E12 section 6 item 1 (R3 at 3 B) remains owed and separate.
- **One slice, one corpus mix.** 24 x 512 on a 40/25/25/10 mix. A code-specialised donor read on a
  majority-prose corpus is a real caveat for B0's absolute value, though not for the chance-line
  comparison, which is per-byte and self-normalising.

## 9. Defect logged in passing

`common.get_slice` caches on `(part, n_seq, seq_len, seed)` and not on the tokenizer or its
identity. Any probe passing a different tokenizer receives the cached ids silently. It has never
bitten, because every donor used so far shares the Qwen2.5 tokenizer — verified for the Coder
crossing in section 3 rather than assumed. The fix is to include a tokenizer fingerprint in the
key; it is not made here because changing the key would invalidate every cached slice and every
`ids_sha256` assertion pinned against them, which is a change that needs its own gate.
