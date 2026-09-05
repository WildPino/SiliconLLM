# E2 — The RMSNorm fold, in the exporter, measured through the engine

**Brief:** `briefs/BRIEF_E2_RMSNORM_FOLD.md`, pre-registered at `338d187`, amended at `edbe956`
(§3.1, arm `NA` is not expressible), `cd65b19` (§3.2, the run plan) and `de7f318` (§3.3, the
mechanical label under a split) — **every amendment written before the run it governs.**

**Runner:** `benchmarks/donor_adaptation/engine/e2_rmsnorm_fold.py`.
**Results:** `benchmarks/donor_adaptation/density/results/e2_rmsnorm_fold_*.json`.

---

## 0. Verdict

*PENDING — run 2 owns the label (brief §3.2). Written from the JSON, verbatim, when it lands.*

---

## 1. What this asks, and what it changes

T3 measured the RMSNorm fold at **`−0.220001 ± 0.052861` BPB** — 44 σ_seed for no format change,
no kernel change and no runtime cost. It measured it as a **control inside a run whose headline
question died**, in a `transformers` module in memory, by the runner that produced it. E2 asks
whether it survives the trip to the artifact: `qwen_export.py --fold`, scored through
`donor_engine.c`, on weights Gate A proves identical.

`fold_norms` pushes each RMSNorm gain into the linears that read it. Per layer that is
`input_layernorm → q,k,v` and `post_attention_layernorm → gate,up` — **2L gains**, reaching
**5 of the 7 converted tensors per layer**. `o_proj` and `down_proj` read an attention output and
an FFN activation, not a norm, so the fold cannot reach them. `--fold all` adds the final
`model.norm → lm_head`, which also **unties** the head.

**One definition of the fold.** The exporter imports `fold_norms` from `t3_rotation`; three probes
now share it. A second copy is how two arms drift apart without anyone noticing.

### 1.1 The arm the brief asked for does not exist

Brief §3 asked for `NA` = *fold all, packed, fp32 head*. A `QWENDON1` file carries a **single**
`quant` flag and `donor_engine.c:397` reads the untied head with it, so **an untied head in a
packed file is necessarily packed**. "Fold everything but keep an fp32 head" is not expressible in
the format. The exporter now refuses the combination rather than silently ternarizing a head
nobody asked to ternarize.

`NA`'s job was an algebra null — with an fp32 head, folding `model.norm` into `lm_head` leaves the
quantized tensors and their calibration untouched — and **that check is sharper at fp32**, where it
must hold exactly rather than approximately. `XA` (fold all, fp32 throughout) replaced it. §3.1 of
the brief, written before any run.

### 1.2 The arms

| arm | `--fold` | `--quant` | head | what it is for |
|---|---|---|---|---|
| `F32` | none | fp32 | fp32 | the reference |
| `XF` | layers | fp32 | fp32 | exactness: the 2L fold is a re-parameterization |
| `XA` | all | fp32 | fp32 | …and so is the final gain, which also unties |
| `TQ` | none | packed | fp32 | the control the fold is measured against |
| `NL` | layers | packed | fp32 | **the fold**, as T3 measured it minus the neutral part |
| `TQH` | none | packed | ternary | the runnable model, unfolded |
| `NLH` | layers | packed | ternary | the fold on the runnable model |
| `NAH` | all | packed | ternary | the only arm that can answer brief §2 |

---

## 2. Gates

**Gate A — the file is the model.** Every converted tensor is read back out of the exported binary
and compared code-for-code and scale-for-scale against the reference built in memory. Bit-identical
on **every arm of every run**, up to **493,961,216 codes**; 0 scales differing, 0.00 ulp.

**Gate F — the fold must be exact where it must be exact.** `|BPB(XF) − BPB(F32)| ≤ 0.002` and
`|BPB(XA) − BPB(F32)| ≤ 0.002` (Gate B's tolerance, 0.4 σ_seed), reported on **both** sides.
Gate F is measured on the 0.5B by run 1 and on the deciding donor by run 3; it is **not inherited
across donors** — "it is model-independent algebra" is exactly the kind of reasoning that has been
wrong here before.

**Gate E — E1 must have closed the loop.** E1's label on the 0.5B is `LOOP-CLOSED`; the engine
scores what PyTorch scores to `1.5e-05` on both donors (`probes/E1_BPB_THROUGH_ENGINE.md`).
Satisfied.

---

## 3. Results

Shared eval slice throughout: `heldout`, 24×512, seed 1234,
`ids_sha256 = a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`,
**51,870 scored bytes**. Calibration: 32×512, seed 42424, rule `R3` (`r3_actsearch`), 6 threads —
the thread count is part of the artifact's identity, not a speed knob (INDEX §5).

### 3.1 Run 1 — Qwen2.5-0.5B, all eight arms, 24 seqs, 5,180 s

Brief §3.2 gave this run **the apparatus and no decision**, before any number existed.

| arm | fold | gains | BPB (PyTorch) | BPB (engine) | engine−torch | Δ vs `F32` (engine) | mean zero frac | Gate A |
|---|---|---|---|---|---|---|---|---|
| `F32` | none | 0 | `0.871795114` | `0.871810461` | `+1.535e-05` | — | — | 290 tensors, exact |
| `XF` | layers | 48 | `0.871795100` | `0.871810461` | `+1.536e-05` | `+0.000000000` | — | 290 tensors, exact |
| `XA` | all | 49 | `0.871795121` | `0.871810463` | `+1.534e-05` | `+0.000000002` | — | 291 tensors, exact |
| `TQ` | none | 0 | `4.509151256` | `4.509163909` | `+1.265e-05` | `+3.637353448` | `0.4913539` | 357,826,560 codes, exact |
| `NL` | layers | 48 | `4.178271368` | `4.178295610` | `+2.424e-05` | `+3.306485149` | `0.5427960` | 357,826,560 codes, exact |
| `TQH` | none | 0 | `4.531219311` | `4.531233734` | `+1.442e-05` | `+3.659423273` | `0.4915176` | 493,961,216 codes, exact |
| `NLH` | layers | 48 | `4.001970241` | `4.001988176` | `+1.794e-05` | `+3.130177715` | `0.5426553` | 493,961,216 codes, exact |
| `NAH` | all | 49 | `4.631919530` | `4.631941061` | `+2.153e-05` | `+3.760130600` | `0.5434361` | 493,961,216 codes, exact |

`gains` is `n_gains_folded` read back from the sidecar and asserted against `2L = 48` and
`2L+1 = 49` (L = 24). The runner refuses an arm whose sidecar does not report the fold it asked for.

**Paired between-arm bootstrap**, 2000 resamples, seed 7, byte-weighted, on the exact per-sequence
PyTorch nats:

| contrast | Δ | paired SE | ci95 | σ_seed | excludes 0 |
|---|---|---|---|---|---|
| `NL − TQ` | `−0.330880` | `0.069328` | `[−0.449554, −0.177475]` | 66.2 | yes |
| `NLH − TQH` | `−0.529249` | `0.073325` | `[−0.663949, −0.380994]` | 105.8 | yes |
| `NAH − NLH` | `+0.629949` | `0.059764` | `[+0.507025, +0.739333]` | 126.0 | yes |
| `XA − XF` | `+2.037e-08` | `9.907e-09` | `[+4.883e-09, +4.213e-08]` | 4.07e-06 | yes |

**Gate F on this donor — PASSES**, four numbers, both sides:

| | diff | tol |
|---|---|---|
| `XF − F32` (PyTorch) | `−1.358e-08` | `0.002` |
| `XF − F32` (engine) | `+2.872e-10` | `0.002` |
| `XA − F32` (PyTorch) | `+6.790e-09` | `0.002` |
| `XA − F32` (engine) | `+1.843e-09` | `0.002` |

**Mechanical label: `FOLD-CONFIRMED`.** Reported verbatim and **apparatus-only**. §3.2 denied this
donor decision authority before any number existed, so a 0.5B label that agrees with run 2 is not
evidence for run 2, and one that disagrees does not overturn it.

### 3.2 Run 2 — Qwen2.5-1.5B, `TQ, NL, TQH, NLH, NAH`, 24 seqs — **owns the label**

`L = 28`, so a layers-fold is `2L = 56` gains and a full fold `2L+1 = 57`. 7,272 s.

| arm | fold | gains | BPB (PyTorch) | BPB (engine) | engine−torch | Δ vs base | mean zero frac | Gate A |
|---|---|---|---|---|---|---|---|---|
| `TQ` | none | 0 | `3.484251281` | `3.484253077` | `+1.797e-06` | `+2.716656322` | `0.4714446` | 1,310,195,712 codes, exact |
| `NL` | layers | 56 | `3.264250451` | `3.264252413` | `+1.962e-06` | `+2.496655492` | `0.5083735` | 1,310,195,712 codes, exact |
| `TQH` | none | 0 | `3.475705979` | `3.475706692` | `+7.128e-07` | `+2.708111021` | `0.4713247` | 1,543,569,408 codes, exact |
| `NLH` | layers | 56 | `3.233373977` | `3.233374462` | `+4.842e-07` | `+2.465779019` | `0.5080662` | 1,543,569,408 codes, exact |
| `NAH` | all | 57 | `3.410356839` | `3.410360248` | `+3.409e-06` | `+2.642761881` | `0.5090935` | 1,543,569,408 codes, exact |

Δ is against the standing 1.5B baseline `0.7675949584171732`. The engine−torch residual spans
`4.8e-07` to `3.4e-06` and does **not** order with the fold — E1's finding again: it is
accumulation order, not a defect the fold would amplify.

**Paired between-arm bootstrap**, 2000 resamples, seed 7, byte-weighted:

| contrast | Δ | paired SE | ci95 | σ_seed | excludes 0 |
|---|---|---|---|---|---|
| **`NL − TQ`** | **`−0.220001`** | `0.052861` | `[−0.324988, −0.119593]` | 44.0 | yes |
| `NLH − TQH` | `−0.242332` | `0.041038` | `[−0.321919, −0.166627]` | 48.5 | yes |
| `NAH − NLH` | `+0.176983` | `0.008303` | `[+0.161821, +0.193895]` | 35.4 | yes |

**Mechanical label, verbatim from the JSON:**

> `FOLD-CONFIRMED / GATE-F-NOT-MEASURED-HERE (no F32/XF/XA arm in this invocation; brief s3.2`
> `gives Gate F on this donor to run 3 -- not final until it passes)`

with `gate_F_measured: false`, `gate_F_ok: false`, `fold_term_label: "FOLD-CONFIRMED"`.

### 3.3 Run 3 — Qwen2.5-1.5B, `F32, XF, XA`, 4 seqs — Gate F on the deciding donor

*PENDING.*

### 3.4 The label, assembled term by term

*PENDING — assembled from runs 2 and 3 the way E1 §2.3 assembled its own, and reported verbatim.*

---

## 4. Replication

The constants are 1.5B constants and belong to run 2. Run 1 reports an empty `replication` block
for exactly this reason: **a replication constant names a donor, an arm, an organ set and a file**,
and run 1's donor is not the one they name.

| arm | measured Δ | standing constant | arm / organ set / file | residue |
|---|---|---|---|---|
| `TQ` | `+2.7166563222481854` | `+2.7166563086672921` | T3 arm `Q` = T2b arm `FA`, 196 tensors, `t2b_organs.json` | **`+1.358e-08`** |
| `NL` | `+2.4966554924531996` | `+2.4966555196149862` | T3 arm `N`, 196 tensors, `t3_rotation.json` | **`−2.716e-08`** |
| `TQH` | BPB `3.475705979` | BPB `3.4757059788577203` | T2b arm `FAH`, 197 tensors, `t2b_organs.json` | at printed precision |

**Brief §4's hedge was unnecessary, and that is itself a result.** It said `NL` would land *near*,
not exactly, on T3's arm `N`, because arm `N` folded `2L+1` gains and `NL` folds `2L`. It lands at
`2.7e-08`. Brief §2 had derived why — with an **fp32 head**, folding `model.norm` into `lm_head` is
a re-parameterization of weights that are never rounded — and `XA` was built to test that algebra
at fp32, where it must hold exactly. The replication tests it **at ternary width**, on 196 rounded
tensors, and it holds there too.

**And the fold reproduces its own dispersion.** T3 measured `−0.220001 ± 0.052861` in a
`transformers` module in memory. E2 measures `−0.220001 ± 0.052861` through the exporter, in the
file format, on weights Gate A proves identical, scored by `donor_engine.c` — point *and* paired
SE at the printed precision. That is not a coincidence and not a leak: the paired bootstrap shares
seed 7, the same 24 sequences and the same byte weights, and the per-sequence nats of the two arms
agree to `1e-07`, so resampling them **must** return the same SE. It is the strongest available
statement that the trip to the artifact moved nothing.

---

## 5. What the numbers say

### 5.0 The two donors disagree about magnitude and agree about sign

| contrast | 0.5B (apparatus) | 1.5B (**decides**) | ratio |
|---|---|---|---|
| `NL − TQ` — the fold, fp32 head | `−0.330880` | **`−0.220001`** | 1.50× |
| `NLH − TQH` — the fold, ternary head | `−0.529249` | `−0.242332` | 2.18× |
| `NAH − NLH` — plus the final gain, ternary head | `+0.629949` | `+0.176983` | 3.56× |
| `NAH − TQH` — fold-all vs no fold | `+0.100721` | **`−0.065349`** | **sign flips** |
| extra the fold gets from a ternary head | `−0.198369` | `−0.022331` | 8.88× |

Every sign that matters replicates. **No magnitude does**, and the fourth row *inverts*: on the
0.5B, `--fold all` is worse than not folding at all; on the 1.5B it is still better than not
folding, just far worse than `--fold layers`.

This is E1's shape again — "the head is free" was a 1.5B-only result — and it is why brief §3.2
took decision authority away from the small donor **before any number existed**. Reading the head
question off the 0.5B would have overstated it 3.6× and got the third comparison backwards.

### 5.1 The fold does not just rescale — it changes what R3 keeps

The mean ternary zero fraction moves with the fold, and the arms are arithmetically consistent with
each other. **On the 0.5B** (168 layer matrices, 169 with the head):

- `TQ → NL`, 168 matrices, all of them re-scaled: `0.4913539 → 0.5427960`, **`+0.0514421`**.
- `TQH → NLH`, 169 matrices, of which **`lm_head` is unchanged** (a layers-fold does not touch it):
  `0.4915176 → 0.5426553`, `+0.0511377`. Scaled to the 168 that did change,
  `+0.0511377269 × 169/168 = +0.0514421181` — **the same number to `1.4e-17`**, i.e. to the last
  bit of a double. That is not a loose agreement: it says `--head-ternary` leaves all 168 layer
  tensors quantized *identically*, so the four packed arms are two clean pairs differing in one
  tensor apiece.
- `NLH → NAH`, 169 matrices, of which **only `lm_head` changes**: `+0.0007808` on the mean, so
  `× 169 = ` **`+0.13195` on the head alone**.

So folding the gains makes R3's per-row threshold search **zero out more weights**, ~5.1 points in
the layers and **13.2 points in the head**. That is a description of the artifact, not a mechanism:
it is derived by exact arithmetic from the reported means, and the per-tensor zero fractions were
not recorded.

The 1.5B behaves the same way and the internal consistency holds to the same precision — over 196
matrices `TQ → NL` is `+0.0369289`, and over 197 `TQH → NLH` is `+0.0367415`, which scaled by
`197/196` is `+0.0369290`.

**And the obvious story is false.** It is tempting to say the fold sparsifies the head and that the
sparsity *is* the damage. The two donors kill it:

| | zeros gained by `lm_head` under the final fold | BPB it costs (`NAH − NLH`) |
|---|---|---|
| 0.5B | `+0.131959` | `+0.629949` |
| 1.5B | **`+0.202378`** | **`+0.176983`** |

The 1.5B head gains **half again as many zeros** and costs **3.6× less**. Zero fraction moves with
the fold, and it does not predict what the fold costs — it does not even keep its sign against BPB,
since in the layers more zeros accompany a *gain* of `−0.220`. Recorded as a dead end, so nobody
re-derives it.

### 5.2 "ci95 excludes 0" is not "the effect matters"

`XA − XF` is `+2.037e-08` BPB with a ci95 of `[+4.9e-09, +4.2e-08]` — **entirely above zero**. The
bootstrap is right: the difference is deterministic, positive, and about the size of one fp32
rounding. It is also **4.07e-06 σ_seed**, which is the column that decides. This is why the
pre-registered rule pairs exclusion with a **magnitude threshold** and never uses exclusion alone.
Kept as an illustration, not as a finding.

### 5.3 The head fold — a first look, answered on both donors

Brief §5 gave `NAH − NLH` **no bar**, because there was no prior to set one from, and said it would
be reported as a first look. It is one, and it has a clear answer:

**Folding `model.norm` into a ternary head costs BPB, on both donors, with the ci95 well clear of
zero** — `+0.629949 [+0.507, +0.739]` on the 0.5B, `+0.176983 [+0.162, +0.194]` on the 1.5B.

The asymmetry with the layer fold is the interesting part. The same operation — push a gain vector
into the columns of the matrix that reads it — **buys** `−0.220` when applied to `q,k,v,gate,up`
and **costs** `+0.177` when applied to `lm_head`. E2 does not say why, and nothing here licenses a
guess: §5.1 rules out the sparsity story, and the per-row structure of R3's search against the
per-column structure of a gain fold was not measured.

**What it settles operationally:** the runtime's configuration is **`--fold layers` with a ternary
head**, and the final gain stays where it is. That is `NLH`, at `+2.465779` — the best of the five
arms that can actually run.

---

## 6. The label branch that could not tell "failed" from "never looked"

Found before run 2, not after. Recorded in full because the failure mode is this programme's own
law turned on its own code.

Gate F is built only from `XF`/`XA` against `F32`. **Run 2 carries none of them.** `gate_f` would
have been empty, `gate_f_ok` `False`, and the run that owns the label would have printed:

> `VOID (Gate F failed: the fold is not exact where it must be)`

— a physical claim about the fold, derived from the fact that nobody looked. E1 §6's law, exactly:
*an error message is a hypothesis, not a diagnosis; a guard that cannot distinguish "measured and
failed" from "not measured" checks nothing.*

The runner now separates `gate_F_measured` from `gate_F_ok`. **All three states were planted and
shown to fire on the 0.5B at 2 seqs, before run 2 was launched** (commit `3aa1bef`):

| state | arms | label |
|---|---|---|
| measured, passed | `F32,XF,TQ,NL` | `FOLD-CONFIRMED` |
| **not measured here** | `TQ,NL` | `FOLD-CONFIRMED / GATE-F-NOT-MEASURED-HERE (…brief §3.2 gives Gate F on this donor to run 3 — not final until it passes)` |
| **measured, failed** | `F32,XF` | `VOID (Gate F failed: the fold is not exact where it must be)` |

The third was planted with a copy of the runner carrying `GATE_F_FP32_TOL = -1.0`, so that a gate
which passes at `2e-09` cannot pass. Without it the patch would only have been shown *not* to say
`VOID` — and a guard that never fires looks identical to one that fires correctly.

The brief also mispredicted its own run 1: §3.2 said runs 1 and 3 would both return `INCOMPLETE`
"because neither holds `NL` and `TQ`". True of run 3; **false of run 1**, which holds all eight.
Run 1's label is therefore real and is reported above rather than suppressed.

---

## 7. Departures from the brief

1. **Arm `NA` → `XA`** (§3.1, before any run). The format cannot express `NA`; `XA` tests the same
   algebra more sharply. §1.1 above.
2. **The decision is split across three runs** (§3.2, before any run), because the 1.5B's fp32 arms
   cost ~40 minutes each and are an exactness test, not a number this programme quotes.
3. **The mechanical label learned to say which term is missing** (§3.3, after run 1, before run 2).
   No threshold, contrast, arm or decision changed. §6 above.

---

## 8. Reproduction

```
cd benchmarks/donor_adaptation/engine

# run 1 -- apparatus, no decision
E1_THREADS=6 python e2_rmsnorm_fold.py --model Qwen/Qwen2.5-0.5B \
  --arms F32,XF,XA,TQ,NL,TQH,NLH,NAH --seqs 24 --calib-seqs 32

# run 2 -- owns the label
E1_THREADS=6 python e2_rmsnorm_fold.py --model Qwen/Qwen2.5-1.5B \
  --revision 8faed761d45a263340a0528343f099c05c9a4323 \
  --arms TQ,NL,TQH,NLH,NAH --seqs 24 --calib-seqs 32

# run 3 -- Gate F on the deciding donor
E1_THREADS=6 python e2_rmsnorm_fold.py --model Qwen/Qwen2.5-1.5B \
  --revision 8faed761d45a263340a0528343f099c05c9a4323 \
  --arms F32,XF,XA --seqs 4
```

The engine binary must be built from `donor_engine.c` at `33f0add` or later — **earlier binaries
cannot load a file over 2 GB** and report `bad magic` against an intact one (INDEX §5).
