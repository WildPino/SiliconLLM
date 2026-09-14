# BRIEF E66 — one byte at seven billion: the rung between "works but is fp32" and "is destroyed"

**Status: PRE-REGISTERED. Pushed before the runner exists and before one BPB is read.**
**QUALITY AND RANK ONLY. No rate is measured. `G-E63d` is `VOID` and OWED and E66 does not touch it.**

---

## 1. The question, in one sentence

**The largest real donor this programme owns has been converted exactly twice — to fp32, where
it works, and to ternary, where it is destroyed — and the one-byte rung between them has never
been run at 7 B, although it is free at every smaller scale that has been tried.**

## 2. Why this is the next thing, read out of the record

Every number below is transcribed from a published probe or result file, not recomputed here.

**The 7 B, as it stands** (`probes/E15_DOES_THE_7B_PREDICT.md`, `probes/E16_R3_AT_7B.md`),
donor Qwen2.5-Coder-7B, frozen slice 24×512, chance line **4.070106** at `V = 152064`:

| arm | format | B/weight | BPB | vs chance | greedy vs fp32 |
|---|---|---|---|---|---|
| **B0** fp32 | fp32 | 4.0 | **0.674027** | −3.396080 | 160/160 |
| **B2** packed `R3`, `fold=layers` | ternary | 0.5 | 4.017233 | −0.052874 | **0/160** |
| **B3** packed `R3`, `fold=none` | ternary | 0.5 | 4.168325 | +0.098219 | **0/160** |
| **B1** packed `R0`, `fold=layers` | ternary | 0.5 | 5.299200 | +1.229094 | **0/160** |
| **int8** | int8 | **1.0** | **NEVER MEASURED** | | |

E16's verdict is `SCORE-CROSSES-RANK-DOES-NOT`: fixing the rule is worth 1.282 BPB and carries
B2 below the chance line, **and the ranking does not move at all** — 0 of 160 greedy tokens,
diverging at token 0. **BPB alone would have called B2 a success.** That is why §5 registers a
RANK gate that can fail on its own (E14 §3).

**The one-byte rung, everywhere it HAS been run** (`engine/results/e62_third_scale.json`,
arm `I8` = `quant=int8, rule=R8, head_ternary=True, fold=none, calib_seqs=32`):

| donor | fp32 | int8 | damage | share of that donor's dense→chance gap |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.871795 | 0.8718769 | +0.000082 | 0.0026% |
| Qwen2.5-1.5B | 0.767595 | 0.7688588 | +0.001264 | 0.038% |
| Qwen2.5-3B | 0.724450 | 0.7250401 | +0.000590 | 0.018% |

This is §62.12's damage ladder: **half a byte costs 82% of the gap, one byte costs 0.038%.**
The 7 B has only ever been given the half byte.

## 3. What E66 may NOT assume, and it is the same trap E62 itself found

**Three scales agreeing does NOT license the fourth.** E62's own published finding is that
`dBPB` is **not monotone in `N`** — the damage does not rank across donor size, which is why
`feedback_rank_partner_survives_refusal` records *"a fit on two adjacent points has no exponent,
it has an artefact"*. The 0.5/1.5/3 B column above is a **reference class, not a trend line**,
and E66 fits nothing to it.

**And there is direct evidence of a format collapsing exactly at this scale**: ternary is
survivable at 1.5 B (E37's uncarved ternary reads 3.475707, below chance, and ranks) and is
**rank-dead at 7 B**. Whatever breaks there might break here. **That is the experiment.**

## 4. Apparatus, and the provenance discipline this brief exists to apply

`qwen_export.py` already supports `--quant int8 --rule R8`; the engine already runs `quant=5`;
the donor is on disk (`Qwen/Qwen2.5-Coder-7B`, 4 shards, complete); the slice is E1's frozen
24×512, `ids_sha256 a1a48dc9fc5a6dc1`, 51,870 scored bytes, 12,264 predicted. **Nothing new is
built except one export.**

**Every cell asserts its kernel arm from the engine's own `CONFIG` line and is REFUSED
otherwise.** This is §63.3, published today: E64 run 1 compared `attn=serial` numbers against
`attn=avx4` numbers and called the difference a replication failure. E66 is the first experiment
designed after that correction, so it states the arm of every reference it quotes:

| reference | engine it was taken on | arm |
|---|---|---|
| E62's `I8` ladder (0.5/1.5/3 B) | E60-era build | **`avx4`** |
| E15's `B0`, E16's `B1/B2/B3` (7 B) | the E13 build | **`serial`** (pre-E50 default) |

**E66 runs `avx4` and asserts it.** The comparison against E15/E16's 7 B numbers is therefore
**cross-kernel and is labelled as such** — §63.3 bounds that at **~1e-06 on a path with no
selection**, which is three orders below the smallest quantity this brief reads, and the arms
here are dense (no top-`k` anywhere). The comparison against E62's ladder is same-arm and exact.

### 4.1 Arms

| arm | export | why |
|---|---|---|
| **A1** `coder7b_i8_foldlayers` | `--quant int8 --rule R8 --head-ternary --fold layers --calib-seqs 32` | E16 measured the fold worth **0.151 BPB at 7 B** on ternary, and it is the whole reason B2 sits below chance |
| **A2** `coder7b_i8_nofold` | same, `--fold none` | E1's pinned protocol and **the fold E62's int8 ladder used** — the only arm directly comparable to the 0.5/1.5/3 B column |

Both are read. Neither is chosen after the fact: A2 is the one quoted against E62's ladder, A1
is the one quoted against E16's 7 B arms, and §5 says so before either exists.

### 4.2 Controls — planted, and each fires on a KNOWN POSITIVE before any 7 B int8 cell is read

| control | requires | tolerance | what it validates |
|---|---|---|---|
| **`G-E66a`** | re-measure **1.5 B `I8`** and reproduce E62's `0.7688588381536873` | \|d\| ≤ **1e-08** (E62's own `G62A_TOL`) | the **int8 code path**, on today's binary, same arm |
| **`G-E66b`** | re-measure **7 B ternary `R3 fold=layers`** and reproduce E16's `4.017232598` | \|d\| ≤ **1e-04** | the **7 B path end-to-end on today's binary** |
| **`G-E66c`** | every cell's `CONFIG` reports the expected `quant=` **and** `attn=avx4`; otherwise the cell is REFUSED | exact | §63.3 / `feedback_config_must_appear_in_output` **as a gate, not a log line** |

**Why `G-E66b`'s tolerance is 1e-04 and not 1e-09.** E16 ran the **E13 build**; between it and
`e63` sit five commits to `donor_engine.c`, of which §63.3 has measured only one (E50's kernel
default, worth ~5.9e-07 on a dense path). `1e-04` is E37's own standing `G-E37A` tolerance and
is registered here **before** the number is seen. **A miss larger than 1e-03 on a dense path
would not be noise and would be a new finding about a different commit** — in that case E66
stops and the cause is measured, exactly as E64 was made to.

**If `G-E66a` or `G-E66b` does not fire, no 7 B int8 cell may be read.** E64's discipline,
adopted verbatim.

## 5. Gates and predictions — registered, falsifiable, and the RANK gate can fail alone

| gate | question | kind |
|---|---|---|
| **`G-E66d`** SCORE | does `A2` land below the chance line 4.070106? | the weak one |
| **`G-E66e`** SCORE | is `A2` within **0.05 BPB** of fp32 `0.674027`? | the real score question |
| **`G-E66f`** **RANK** | greedy agreement with the fp32 donor over 160 tokens, E16's protocol | **E14 §3 — and the one E16 failed at 0/160** |
| **`G-E66g`** | `\|A1 − A2\|`, the fold's worth on int8 | descriptive |

**Predictions:**

1. **`G-E66a` and `G-E66b` both fire.** If not, E66 stops and nothing is read.
2. **`A2` lands below chance.** Weak; ternary already manages it.
3. **`A2` lands within 0.05 of fp32 (i.e. ≤ 0.724027).** The band is **40× wider** than the
   worst damage in the 0.5/1.5/3 B column (0.001264) *on purpose*: §3 forbids extrapolating that
   column, and E16 is direct evidence that a format can collapse at exactly this scale.
4. **`G-E66f` ≥ 150/160.** **This is the prediction that carries E66.** E16's ternary arms read
   **0/160** while one of them passed on score, so a score without a rank partner cannot
   distinguish "works" from "confidently wrong in a new way".
5. **The fold is worth < 0.01 BPB on int8**, against **0.151** on ternary, because the fold
   compensates a quantisation error that one byte largely does not make. **Mechanism named, and
   falsifiable: if the fold still buys ~0.15 at one byte, my account of what the fold does is
   wrong.**

## 6. What E66 may NOT conclude

1. **NO RATE. Not one tok/s.** E66 is quality and rank only, and **`G-E63d` stays `VOID` and
   OWED** — it is blocked on an idle machine here, not on this run.
2. **Nothing about 50 tok/s.** A dense 7.62 B at one byte moves ~7.6 GB per token; against the
   **int8** band measured in `feedback_charged_vs_moved_bytes` (34.0–34.9 GB/s) that **projects**
   to ~4.4–4.6 tok/s. That is a projection from bytes, in the format's own band (§59.2 respected),
   **not a measurement**, and E66 does not make it one. **Even a perfect result here is a working
   model at roughly a tenth of the target**, and the remaining 10× is traffic, not precision.
3. **Nothing about 10 B.** 7.62 B, `D = 3584`, `L = 28` — not the `A10B-*` shapes, whose rates
   were measured on synthetic weights.
4. **Nothing about a TRAINED conversion.** Post-hoc is a floor (H0, H1, E65).
5. **Nothing about the carve or rank at 7 B.** Those compose on top and are separate cells;
   E64 run 2 is measuring the carve×int8 composition at 1.5 B, and E66 does not anticipate it.
6. **No retraction of E15 or E16.** Their numbers stand for the formats they measured.

---

# ADDENDUM A — apparatus correction, pushed BEFORE the runner runs

**§4.1 specified `--calib-seqs 32` on both int8 arms. That is wrong, and it would have broken
the control it is meant to be compared against.**

Read off the artefacts, not assumed:

| artefact | `quant` | `rule` | `fold` | `head_ternary` | **`calib_seqs`** |
|---|---|---|---|---|---|
| `e60/qwen25-15b_i8h.bin` (the `G-E66a` control) | int8 | R8 | none | True | **None** |
| `e62/qwen25-3b_i8.bin` (the 3 B rung) | int8 | R8 | none | True | **None** |
| `e7/qwen25-coder7b_p_r3.bin` (the `G-E66b` control) | packed | R3 | layers | True | 32 |

**Calibration sequences belong to the PACKED/ternary path, not the int8 one.** E62's `CALIB_SEQS
= 32` applied to its `PACKED` arm; every int8 artefact on disk records `calib_seqs: None`. Had
E66 passed `--calib-seqs 32` on A1/A2, the 7 B int8 cells would have been built by a different
construction from the 0.5/1.5/3 B rungs they are quoted against — a `feedback_control_arm_different_code_path`
defect, found by reading the sidecar instead of trusting my own brief.

**Corrected apparatus:**

```
A1: --model Qwen/Qwen2.5-Coder-7B --revision 0396a76181e127dfc13e5c5ec48a8cee09938b02     --quant int8 --rule R8 --head-ternary --fold layers --load-dtype float32 --threads 6
A2: ... identical, --fold none
```

The revision is **pinned from the local snapshot**, read on disk. `G-E66a` re-measures
`e60/qwen25-15b_i8h.bin` as it stands and exports nothing.

**One consequence for §5, registered now:** A2 (`fold=none`, no calibration) is the arm quoted
against the 0.5/1.5/3 B ladder and it is now construction-identical to it. A1 (`fold=layers`)
differs from that ladder in the fold **only**, which is exactly what `G-E66g` measures.

**And one owed item this turns up:** `e16_r3_at_7b.py` resumes from its own output file
(`if os.path.exists(OUT): prev = json.load(...)`), the same hazard that made E65 run 1's control
tautological. E66 does not resume from its result file. The E16 defect is logged, not fixed here.

---

# ADDENDUM B — controls before export, and committed-apparatus provenance

**Pre-registered 2026-09-14 before this apparatus change and before E66 runs. No threshold,
prediction, arm or estimand changes.**

## B.1 Do not spend two 7 B exports before asking whether the instrument is alive

The first runner orders `all` as **export → controls → cells → greedy**. The controls cost one
existing 1.5 B score and one existing 7 B score; the exports construct two new ~7–8 GB artifacts
and can take hours. If either planted control is dead, the brief already forbids reading the new
cells, so exporting first would spend the largest cost on an experiment known to be void.

The registered `all` order is changed to:

1. run `G-E66a`, `G-E66b` and `G-E66c` on the existing control artifacts;
2. write `results/e66/e66_controls.json` immediately;
3. if either control is dead, stop before export and publish only the control failure;
4. if both fire, export A1/A2, score them, then run the rank partner.

A new `--stage controls` performs steps 1–2 only. It reads **no E66 treatment cell** and changes
no result. `--stage bpb` still runs the controls before the cells when invoked independently.
Partial `controls` and `export` stages may not write the canonical
`e66_one_byte_at_7b.json`; that filename is reserved for an adjudicable score/full run.

## B.2 The runner must prove it was committed before it starts

E64 run 2 exposed a provenance defect during audit: its preregistration preceded the run, but
the exact runner blob was committed 23 minutes after launch although the commit subject claimed
otherwise. E66 turns that lesson into an executable gate.

Before loading any model, the runner must:

* compute the Git blob of its own working-tree file;
* read the blob at `HEAD` for the same path and **REFUSE** unless the two match;
* record the full `HEAD` commit, runner blob and runner sha256 in every stage result;
* record the engine executable sha256 alongside the engine filename.

This does not make Git part of the model measurement. It makes *"pushed before the run"* a
checked property rather than a commit-message claim. The gate may be bypassed only by changing
this brief in a new pre-run addendum; there is no implicit dirty-tree mode.

## B.3 Expected cost and stopping rule

E16's recorded 7 B BPB cells took about 1,964–1,973 seconds each; the 1.5 B control is smaller.
The controls-only preflight is therefore expected to take roughly 40–45 CPU minutes. It is
quality-only: contention changes elapsed time, not BPB. If both controls fire, the expensive
exports are licensed. If either dies, E66 stops and the failure is investigated before any new
7 B artifact is built.
