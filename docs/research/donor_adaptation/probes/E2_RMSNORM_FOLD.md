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

*PENDING.*

### 3.3 Run 3 — Qwen2.5-1.5B, `F32, XF, XA`, 4 seqs — Gate F on the deciding donor

*PENDING.*

### 3.4 The label, assembled term by term

*PENDING — assembled from runs 2 and 3 the way E1 §2.3 assembled its own, and reported verbatim.*

---

## 4. Replication

*PENDING — the two constants are 1.5B constants and belong to run 2:* arm `TQ` must reproduce
`+2.7166563086672917` (T2b arm `FA`, 196 tensors, `density/results/t2b_organs.json`) and arm `NL`
must land **near** `+2.4966555196149862` (T3 arm `N`, 196 tensors,
`density/results/t3_rotation.json`) — *near*, not exactly, because T3's arm `N` folded the final
gain too and `NL` does not. Run 1 reports an empty `replication` block for exactly this reason:
the constants name a donor, and it is not this one.

---

## 5. What the run-1 numbers say

Everything in this section is a statement about the **0.5B apparatus**. None of it is a decision.

### 5.1 The fold does not just rescale — it changes what R3 keeps

The mean ternary zero fraction moves with the fold, and the three arms are arithmetically
consistent with each other:

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

### 5.2 "ci95 excludes 0" is not "the effect matters"

`XA − XF` is `+2.037e-08` BPB with a ci95 of `[+4.9e-09, +4.2e-08]` — **entirely above zero**. The
bootstrap is right: the difference is deterministic, positive, and about the size of one fp32
rounding. It is also **4.07e-06 σ_seed**, which is the column that decides. This is why the
pre-registered rule pairs exclusion with a **magnitude threshold** and never uses exclusion alone.
Kept as an illustration, not as a finding.

### 5.3 The head fold — a first look, and it points the wrong way

Brief §5 gave `NAH − NLH` **no bar**, because there is no prior to set one from. On this donor it
is `+0.629949`, and `NAH` (`4.631941`) is worse than **`TQH` (`4.531234`)** — folding the final
gain into a **ternary** head is worse than not folding at all, and it costs more than the whole
`−0.529` that the layer fold bought on the same arm.

The brief predicted this arm would matter and did not predict its sign. Whether the sign survives
to the 1.5B is run 2's to say.

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
