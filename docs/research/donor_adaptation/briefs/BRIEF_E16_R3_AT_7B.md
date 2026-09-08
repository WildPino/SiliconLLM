# BRIEF E16 — is there a ternary 7 B that predicts?

**Pre-registration. Pushed before the export runs.** Follows E15 (`DOES-NOT-PREDICT`,
`probes/E15_DOES_THE_7B_PREDICT.md`, ledger §28) and closes E7 §12 owed item 3.

---

## 1. The question

E15 measured the artifact every donor tok/s in this programme is quoted on and found it **1.229094
BPB above the chance line** — worse than guessing uniformly — while the same weights in fp32 read
**0.674026555**, the best model this programme has ever run. **The engine is sound, the donor is
sound, and the conversion is what is broken.**

E15 could not say *which part* of the conversion. `qwen25-coder7b_p.bin` differs from every other
standing engine artifact on **two** axes at once:

| | rule | fold | head |
|---|---|---|---|
| `qwen25-coder7b_p.bin` (E15's B1) | **`R0`** | **`layers`** | ternary |
| `qwen25-15b_tqh.bin` (E1's 1.5 B `TQH`) | **`R3`** | **`none`** | ternary |

E16 asks: **is the RULE the problem, and does a ternary 7 B that predicts exist?**

## 2. Why it is not already answered

- **`R3` is the rule the pipeline actually ships.** `e1_bpb_through_engine.py:395` appends
  `--rule R3` for every non-fp32 arm. `R0` is only `qwen_export.py`'s flag *default*, and
  `qwen25-coder7b_p.bin` was exported before `--rule` was a habit.
- **`R3` works at 1.5 B and fails at 0.5 B.** E1 §2.1/§2.2, same slice, same protocol
  (`--seqlen 512`), chance 4.069819:

  | donor | `TQH` (R3, packed, head-ternary, fold none) | vs chance |
  |---|---|---|
  | Qwen2.5-0.5B | 4.531233733626962 | **+0.461 ABOVE** |
  | Qwen2.5-1.5B | **3.475706691632780** | **−0.594 BELOW** |

  **The shipped rule CROSSES the chance line between 0.5 B and 1.5 B, and it has never been
  measured above 1.5 B.** So a ternary 7 B that predicts is **not ruled out by anything measured**.
  It has simply never been built.
- **`R0` gets WORSE with scale** (E12 §1: +0.518 / +1.436 / +1.665 above chance at 0.5/1.5/3 B,
  FFN+ATTN, fold none). `R3`'s only two points move the other way. **Two curves, opposite
  directions, and E15 sits on the wrong one.** Which curve the 7 B lands on is the experiment.

## 3. Design — one variable at a time, and the design is the gate

Three arms. `qwen25-coder7b_p.bin` (E15's B1, **5.299200075**) and `qwen25-15b_tqh.bin` (E1's,
**3.475706691632780**) are the two fixed reference points; every arm below moves exactly one axis
away from one of them.

| arm | model | rule | fold | isolates |
|---|---|---|---|---|
| **C0** | Qwen2.5-1.5B `TQH` (**existing file**) | R3 | none | **planted control / replication** |
| **B2** | Qwen2.5-Coder-7B, new export | **R3** | **layers** | **the RULE at 7 B** — B2 − B1, fold fixed |
| **B3** | Qwen2.5-Coder-7B, new export | **R3** | **none** | **the SCALE** — B3 vs C0, rule and fold fixed |

and `B2 − B3` is **the fold under R3 at 7 B**, which E2 and T3 have only ever measured at 0.5 B and
1.5 B.

**Exact commands** (`--load-dtype` left at its `float32` default; R3 refuses bf16,
`qwen_export.py:243`):

```
python benchmarks/donor_adaptation/engine/qwen_export.py \
  --model Qwen/Qwen2.5-Coder-7B --quant packed --head-ternary \
  --rule R3 --calib-seqs 32 --fold layers --threads 6 \
  --out D:/_ktmp/e7/qwen25-coder7b_p_r3.bin          # B2

python benchmarks/donor_adaptation/engine/qwen_export.py \
  --model Qwen/Qwen2.5-Coder-7B --quant packed --head-ternary \
  --rule R3 --calib-seqs 32 --fold none --threads 6 \
  --out D:/_ktmp/e7/qwen25-coder7b_p_r3_nofold.bin   # B3
```

**A hard gate on the design, checked in the runner before any BPB is read:** B2's sidecar must
differ from B1's on **`rule`, `calib_seqs`, `bytes`, `sha256`, `mean_ternary_zero_fraction` and
nothing else.** Same `model`, `revision: null`, `quant: packed`, `head_ternary: true`, `fold:
layers`, `n_gains_folded: 56`, `torch_threads: 6`, `attn_implementation: eager`, `rule_applied_to`
(8 entries incl. `lm_head`), and identical dimensions. **If any other field moves, more than the
rule moved and G-R2 is not a rule contrast.** B3 must differ from B2 on `fold` /
`n_gains_folded` / `bytes` / `sha256` / `mean_ternary_zero_fraction` and nothing else.

## 4. Instruments

Same engine binary (`D:\_ktmp\e13\donor_engine.exe`), same slice, same protocol as E15:
density `heldout` 24×512, `ids_sha256 a1a48dc9fc5a6dc1`, **51,870 scored bytes, 12,264 predicted,
4.229452 bytes/token**, `--seqlen 512` **passed explicitly**, `--threads 6`.

**Chance lines.** Coder-7B, `V = 152064`: **4.070106**, band **0.000896** (padded-vs-emittable
vocab, derived per E8 §3, and it is not σ_seed — R3 given a fixed calibration slice is
deterministic). Qwen2.5-1.5B, `V = 151936`: **4.069819**.

**No timing is taken in E16, and none may be quoted from it.** A ternary weight costs the same
bandwidth whatever rule produced it, so **no speed number can move**: 6.79 tok/s exact stands
whatever E16 returns. What E16 can change is whether that rate describes a working model.

## 5. Gates and bands, fixed before the run

**G-R0 — PLANTED CONTROL, and it runs FIRST.** C0 through this session's engine, this runner and
this protocol must reproduce E1's `3.475706691632780` to within **0.01** *and* land **below**
4.069819. **If G-R0 does not fire, E16 is VOID and nothing below is read.** E12's law is why this
costs 25 minutes rather than being waved through on E15's control: an instrument must be verified
as *still* firing in the configuration that will produce the nulls, not merely to have fired once.
C0 is also the only known-positive that shares B3's rule, quant and head treatment.

**G-R1 — the primary gate.** `BPB(B2)` against 4.070106 ± 0.000896:
- below the band → **`RULE-FIXES-IT`**
- inside → **`AT-CHANCE`**
- above → **`RULE-IS-NOT-THE-PROBLEM`**

**G-R2 — the rule at 7 B, fold fixed.** `BPB(B2) − BPB(B1)` = `BPB(B2) − 5.299200075`.
**Descriptive if B2 lands at or above the line** (E12 §2: above chance, BPB measures how
confidently wrong a model is, not how damaged it is). Only a B2 *below* the line makes this a
damage figure.

**G-R3 — the scale, rule and fold fixed.** `BPB(B3)` against 4.070106 ± 0.000896, same three
labels; and `BPB(B3) − 3.475706691632780` as the 1.5 B → 7 B step under R3.

**G-R4 — the fold under R3 at 7 B.** `BPB(B2) − BPB(B3)`. Descriptive. E2 measured the fold at
−0.529 (0.5 B, R3) and T3 at −0.220 (1.5 B); **neither is a prediction for this cell and neither is
composed into one below.**

## 6. Predictions — a direction IS called this time

E15's amended prediction missed because it composed measured deltas across three configurations,
and the costlier half was that it **refused to call a direction the evidence supported**. Both
failures are avoided here: each band below rests on **one quantity measured in the same
configuration at other scales**, and each carries a direction.

- **G-R0**: reproduces to `< 1e-4`. It is the same deterministic binary on the same file.
- **G-R3 — B3 lands BELOW the chance line**, band **2.5 – 4.0**.
  Basis: the *only* R3 sweep that exists (4.531234 at 0.5 B → 3.475707 at 1.5 B) improves by 1.056
  BPB over a 3× step and has already crossed the line, and the fp32 baseline keeps improving to 7 B
  (0.674027). Against that, E12 shows ternarization damage *growing* with scale under R0, so the
  band deliberately does **not** commit to beating 3.4757 and is widened downward instead of
  centred on an extrapolation. Two points cannot establish curvature and this band does not pretend
  they can.
- **G-R1 — B2 lands BELOW the chance line**, band **2.3 – 4.0**.
  Basis: B2 differs from B3 only by the fold, which has been measured as a *credit* under R3 at
  both scales where it was measured. The band is B3's shifted down by 0.2 at the bottom and left
  open at the top, because **the fold's magnitude at 7 B is exactly what G-R4 is for and I am not
  going to import it**. This is the weaker of the two predictions and it is the primary gate; that
  is stated here rather than discovered afterwards.
- **G-R2**: `BPB(B2) − 5.299200075` in **−3.0 to −1.3**, i.e. the rule accounts for most of E15's
  gap. If it does not, §7's conclusion is the one that fires.

**If B2 and B3 both land above the line, no rule in this exporter produces a working 7 B**, the
binding constraint moves from the rule to the **format**, and `SCALEUP_ARCHITECTURE`'s premise —
train *into* the format rather than convert into it — stops being a preference and becomes the
measured conclusion. **That outcome is a result, not a failure, and it is worth the same hours.**

## 7. What E16 cannot claim

- **Nothing about speed.** See §4.
- **Nothing about rules R1/R2**, or about any calibration budget other than 32 (D4b was written to
  sweep that knob and has never run; 32 is a registered point, not an optimum).
- **One slice, one corpus mix, one donor family.** The scale contrast in G-R3 crosses two *different
  models* (Qwen2.5-1.5B → Qwen2.5-Coder-7B), not one model at two sizes, so "scale" there includes
  the code-specialisation. E15 §1 showed that specialisation does not hurt this corpus in fp32,
  which is why the contrast is worth reading — but it is not a clean scale axis and is not reported
  as one.
- **It does not touch the head or the activations.** Every arm is `--head-ternary` and every BPB is
  read on the fp32-activation path, so E14's `CHEAP-BUT-NOT-NEUTRAL` and its owed items are
  untouched.

## 8. Cost, and the protocol for running it

**Sequenced so the cheap arm can kill the expensive one.**

1. **G-R0 / C0** — existing file, ~25 min. If it does not fire, stop.
2. **B2** — export then measure. Export = a float32 load of a 7.62 B model (~30 GB resident; 65 GB
   free, measured) + a 32×512 calibration forward pass + a 5.7 GB write. The **calibration pass
   cost at this scale is not known** — the only R3 exports on record are 0.5 B and 1.5 B — so the
   export is estimated at **1–3 h with real uncertainty**, and the estimate is written down here so
   the eventual number can be scored against it. BPB run ≈ 31 min (E15's B1 took 1,860 s).
3. **B3** — same again. The calibration pass is repeated by the exporter rather than shared; that
   is accepted rather than optimised, because a shared capture would be a code change on the path
   under test.

**One heavy job at a time. Never two trainers/exporters at once** (80 GB machine, and a float32
7.6 B load is 30 GB of it).

## 9. Checked before the run, not assumed

- **The calibration slice crosses the tokenizer boundary cleanly.** `common.get_slice` keys its
  disk cache on `(part, n_seq, seq_len, seed)` and **not** on the tokenizer, and
  `qwen_export.py:132` asks it for `calib/32/512/42424` — a file built on 2026-08-22 under a
  Qwen2.5 tokenizer, before Coder-7B existed in this programme. E15 §4 verified the *heldout* slice
  crosses, and that **does not transfer**: `make_slice`'s rejection loop is tokenizer-dependent, so
  a different tokenizer can drop a different candidate and shift every offset after it.
  Re-derived under both tokenizers: **identical ids (`c5509846cdc3aa44`), identical offsets
  (`c9bdfc96dd5addb8`), identical byte counts (67,648 scored bytes, 4.136986 bytes/token), 0
  rejected, max id 99,042** — both tokenizers report `len = 151665`. **The cache is serving the
  right ids and B2/B3 calibrate on the intended text.** The cache-key defect remains real and
  remains logged; it does not bite here.
- **`qwen25-15b_tqh.bin` is the right control.** Its sidecar reads `quant: packed`, `rule: R3`,
  `head_ternary: true`, `calib_seqs: 32`, `calib_matches_t2_operating_point: true`,
  `torch_threads: 6`, `attn_implementation: eager` — identical to B3's intended shape on every axis
  except the donor itself, and E1 measured it through the engine with `--seqlen 512`
  (`e1_bpb_through_engine.py:311`), so it is protocol-comparable without re-derivation.
- **Free RAM measured at 65.6 GB of 83.8 GB total** before the export is scheduled.
