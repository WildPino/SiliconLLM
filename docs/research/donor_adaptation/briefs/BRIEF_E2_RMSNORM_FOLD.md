# BRIEF E2 — The RMSNorm fold, in the exporter, measured through the engine

**Status: PRE-REGISTERED. Written and pushed before the run, per standing practice.**
**Author: the Adapter / Principal. Date: 2026-09-05.**
**Depends on: `probes/T3_ROTATION.md` §4.4, `probes/T2B_ORGAN_COVERAGE.md`,
`briefs/BRIEF_E1_BPB_THROUGH_ENGINE.md`, `INDEX.md` §4 item 2.**

---

## 1. What T3 left on the table, and why it is not yet a keeper

T3 measured, as a *control* inside a run whose headline question died:

> arm `N` (fold, then R3) − arm `Q` (R3, no fold) = **`−0.220001 ± 0.052861`**, 44 σ_seed,
> **8.1%** of the `+2.716656` R3 leaves on the runnable organ set.

Same format, same shapes, no new kernel, no runtime cost. Arm `XN` showed the fold is exact as a
re-parameterization (`+6.7904e-09`, BPB `0.7675949652076198` against the baseline `0.7675949584171732`). T3 §4.4 put it on the open list *with its own confirmation
owed*, for two reasons that both still stand:

1. **It was measured inside a `VOID` run.** The run's own gate constant was wrong (T3 §1.3). The
   fold contrast is a paired between-arm contrast that does not depend on that constant, which is
   why it is quotable at all — but it has never been reproduced by a runner that was not the one
   that produced it.
2. **It has never existed in the exporter.** T3 folded a `transformers` module in memory. The
   deliverable is `qwen_export.py` → `donor_engine.c`. Nothing has ever folded a norm into a
   binary this programme can execute.

**This brief does not re-ask whether the fold helps in PyTorch. It asks whether it survives the
trip to the artifact.**

## 2. The thing T3 could not see

T3 converted **196 tensors and not the head** (28 layers × `gate/up/down/q/k/v/o`). Its arm `N`
folded `2L + 1 = 57` gains: two per layer, plus the final `model.norm` gain into `lm_head`.

But with an **fp32, unquantized head**, folding the final norm is a pure re-parameterization of
weights that are never quantized: it changes what number is stored, not which numbers get
rounded to `{−1, 0, +1}`. **Therefore all of T3's `−0.220` is attributable to the `2L = 56`
per-layer gains, and none of it to the final one.**

That stops being true the moment the head is ternary. `--head-ternary` is the arm T2b refused to
retire (the head costs `+0.339` alone and *zero* on top of ternary FFN+attention), and it is the
arm the runtime actually wants, because a tied head is read as **fp32** by `donor_engine.c`
(`donor_engine.c:559` — `packed=0: the tied head reads the fp32 embedding`), which is where the
head-is-the-floor finding comes from. Folding `model.norm` into a *ternary* head changes the
per-channel scale of the head's input and therefore changes R3's own thresholds on 151,936 rows.

**So there are two folds here, not one, and only the first has ever been measured:**

| fold | gains | changes what is quantized? | measured by |
|---|---|---|---|
| **per-layer** | `2L` (input_layernorm → q/k/v; post_attention_layernorm → gate/up) | yes — 5 of the 7 converted tensors per layer | T3 arm `N`, `−0.220` |
| **final** | `1` (model.norm → lm_head) | **only when the head is ternary** | never |

Note also which tensors the fold cannot reach: `o_proj` and `down_proj` read an attention output
and an FFN activation, not a norm. **The fold touches 5 of 7 converted tensors per layer and is
structurally incapable of helping the other 2.**

## 3. Arms — one variable each

`F` = a flag to be added to `qwen_export.py`: `--fold {none,layers,all}`. `all` folds the final
gain and therefore **unties** the head, which the format and the engine already support
(`qwen_export.py:232`, `donor_engine.c:397,693`) because `--head-ternary` already sets `tied=0`.

| arm | `--fold` | `--quant` | `--head-ternary` | converts | what it is for |
|---|---|---|---|---|---|
| **F32** | none | fp32 | — | 0 | E1's arm, re-run in-process as the reference |
| **XF** | layers | fp32 | — | 0 | **exactness control**: the fold through the *runtime's* RMSNorm |
| **TQ** | none | packed | no | 196 | E1's arm — the control the fold is measured against |
| **NL** | layers | packed | no | 196 | **the fold**, as T3 measured it minus the neutral part |
| **NA** | all | packed | no | 196 | **planted null**: with an fp32 head, `NA − NL` must be ≈ 0 |
| **TQH** | none | packed | yes | 197 | E1's arm — the runnable model, unfolded |
| **NLH** | layers | packed | yes | 197 | the fold on the runnable model |
| **NAH** | all | packed | yes | 197 | the fold **including** the final gain, the only arm that can answer §2 |

`NA − NL` is a **planted null with a known answer**: it must be ≈ 0 by the algebra of §2, and if
it is not, the fold implementation is wrong and nothing else in this probe may be read. Per the
planted-control law, an instrument that cannot return a known zero has not earned its non-zeros.

`XF` is the corresponding **known-positive-adjacent** control on the other side: it must return
the F32 arm's BPB, through the engine's own RMSNorm now reading a vector of ones.

## 3.1 AMENDMENT, 2026-09-05, before any E2 run — arm `NA` is not expressible

Written into the brief rather than silently changed, because §3 was pre-registered.

**Arm `NA` as §3 defines it — `--fold all`, `--quant packed`, fp32 head — cannot exist in this
format.** A quantized file carries a *single* `quant` flag and `donor_engine.c:397` reads the
untied head with it, so an untied head in a packed file is necessarily packed. "Fold everything
but keep an fp32 head" is not something the format can say. `qwen_export.py` now **refuses** that
combination rather than silently ternarizing a head nobody asked to ternarize (verified: exit 1,
no file written).

**What replaces it, and why the replacement is stronger.** The null `NA − NL ≈ 0` was an *algebra*
check: with an fp32 head, folding `model.norm` into `lm_head` leaves the 196 quantized tensors and
their calibration untouched, so the two arms must agree. That check does not need the engine, and
it does not need quantization at all — it is sharper at fp32, where it must hold **exactly**
rather than approximately:

| arm | `--fold` | `--quant` | must equal | why |
|---|---|---|---|---|
| **F32** | none | fp32 | — | the reference |
| **XF** | layers | fp32 | `F32` | the `2L` fold is a re-parameterization |
| **XA** | all | fp32 | `F32` and `XF` | the final gain is one too, and it unties |

`XA` is expressible (`quant == 0` writes an fp32 head), goes through the engine, and subsumes the
retired `NA`: if the fold implementation folds the wrong gain, folds into the wrong linears, or
mis-handles the untie, `XF` or `XA` moves. Gate F in §5 is amended to read
`|BPB(XF) − BPB(F32)| ≤ 0.002` **and** `|BPB(XA) − BPB(F32)| ≤ 0.002`.

Measured before this amendment was written, on the 0.5B, as an implementation check only (no BPB,
no decision): `--fold layers` folds **48 = 2L** gains, leaves the final gain and the tie alone;
`--fold all` folds **49 = 2L+1**, sets the final gain to ones and unties. Worst logit change
`2.0e-05` absolute, `9.8e-07` relative — the size T3's arm `XN` residue (`+6.8e-09` BPB) predicts.

The arms that carry the decision — `TQ`, `NL`, `TQH`, `NLH`, `NAH` — are **unchanged**.

## 3.2 RUN PLAN, fixed before the run — which run owns the label

E1 §6.2 cost this programme a law: *a pre-registration that splits a decision across runs must
say which run owns the label.* E2 splits, so it says so here.

| # | donor | arms | seqs | owns |
|---|---|---|---|---|
| 1 | 0.5B | all eight | 24 | the apparatus. Gate F on this donor; **no decision** |
| 2 | 1.5B | `TQ, NL, TQH, NLH, NAH` | 24 | **THE LABEL.** Gate F's fp32 terms come from run 3 |
| 3 | 1.5B | `F32, XF, XA` | 4 | Gate F on the deciding donor, as a pre-registered subset |

Run 3 is a 4-sequence subset for the same reason E1 §3 made the 1.5B fp32 arm one: at 6.2 GB of
weight traffic per token the full slice is ~40 minutes *per arm*, and these arms are an
**exactness** test, not a number this programme will quote. Gate F must be evaluated on the
deciding donor and not inherited from the 0.5B — "it is model-independent algebra" is exactly
the kind of reasoning that has been wrong here before.

Runs 1 and 3 will return `INCOMPLETE` mechanically, because neither holds `NL` and `TQ`. That is
expected and pre-registered; **run 2 carries the label**, and Gate F is assembled from run 3 in
the report, term by term, the way E1 §2.3 assembled its own.

## 4. Fixed before the run

- **Slice:** the shared `heldout` slice, 24×512, seed 1234,
  `ids_sha256 = a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`,
  **51,870 scored bytes**. Scoring convention exactly E1's.
- **Calibration:** `calib` 32×512 seed 42424, `--calib-seqs 32`, T = 16,384 tokens. **One capture
  per fold setting**, not one per run: the fold changes the input distribution of q/k/v/gate/up,
  so a folded arm calibrated on unfolded activations would be a different, worse experiment.
- **Threads pinned at 6** and recorded, per `qwen_export.py`'s `--threads` (the thread count is
  part of the artifact's identity — commit `cdb7119`).
- **Statistics:** paired between-arm bootstrap on the same 24 sequence draws, 2000 resamples,
  seed 7, byte-weighted, from the per-sequence nats E1's runner already recovers. A Δ-against-base
  cannot compare two arms (T2 §4). σ_seed = 0.005.
- **Donors:** **Qwen2.5-1.5B** rev `8faed761…` carries the decision, because that is where T3's
  constant lives and where every quality number in this programme lives. **Qwen2.5-0.5B** runs
  all eight arms first, as the apparatus check and because it is what the runtime is benchmarked
  on — *and its result does not license a conclusion about the 1.5B*, per T3 §4.5's law.
- **Speed is not measured and no timing may be quoted.** The fold is shape-preserving, so the
  only footprint claim allowed is arithmetic: `--fold all` unties the head and adds `V×D` to the
  file. It adds **no weight traffic per token** — a tied head already streams `V×D` — so it is a
  footprint cost, not a bandwidth cost, and that sentence is a derivation, not a measurement.

## 5. Pre-registered decision rule

Let `Δ_arm = BPB(arm) − 0.7675949584171732` (the standing 1.5B baseline), and let the fold's
effect be the **paired** contrasts `NL − TQ` and `NLH − TQH`.

**Gate F — the fold must be exact where it must be exact.** `|BPB(XF) − BPB(F32)| ≤ 0.002`
(Gate B's tolerance, 0.4 σ_seed) **and** `|NA − NL| ≤ 0.01` (the planted null, 2 σ_seed).
If either fails, the fold is implemented wrong and **no other number in this probe is reported**.

**Gate E — E1 must have closed the loop.** If E1's label is not `LOOP-CLOSED`, every number here
is a PyTorch number wearing an engine's clothes, and this probe reports its contrasts *explicitly
labelled as such* rather than as statements about the deliverable.

| outcome | condition | what it means |
|---|---|---|
| **FOLD-CONFIRMED** | Gate F passes and `NL − TQ ≤ −0.10` with ci95 excluding 0 | the fold survives to the artifact and buys at least half of T3's `−0.220`. It goes into the exporter's **default** and every standing ternary number is superseded by a folded one |
| **FOLD-SHRINKS** | Gate F passes, `NL − TQ` ci95 excludes 0 but the point is in `(−0.10, −0.02]` | real but much smaller than T3 measured. The difference between `−0.220` and the measured value is then itself the finding, and it is about the *runner*, not the fold |
| **FOLD-NULL** | Gate F passes and `NL − TQ` ci95 contains 0 | T3's control does not replicate. The fold is retired from the open list and T3 §4.4 is amended |
| **FOLD-HURTS** | Gate F passes and `NL − TQ` ci95 lies above 0 | the fold costs BPB through the engine, and the disagreement with T3 is a bug in one of the two |
| **VOID** | Gate F fails | nothing is read |

The **head question** is reported separately and does not gate: `NAH − NLH` is the first
measurement of whether folding the final gain into a *ternary* head helps, hurts, or is neutral.
It has no pre-registered bar because there is no prior to set one from — it is a first look, and
it will be reported as one.

**Replication constant, naming its arm, its organ set and its file** (T3 §6.1): on the 1.5B, arm
`TQ` must reproduce `+2.7166563086672917` (T2b arm `FA`, 196 tensors,
`density/results/t2b_organs.json`) and arm `NL` must land near
`+2.4966555196149862` (T3 arm `N`, 196 tensors, `density/results/t3_rotation.json`,
BPB `3.2642504780321593`) — *near*, not exactly,
because T3's arm `N` folded the final gain too and this brief's `NL` does not; §2 says that
difference is neutral to ~`1e-08` for an fp32 head, and `NA` is the arm that tests that claim.

## 6. What this brief does NOT do

- No rotation. T3 closed it: `+0.634` and `+1.138` against a fold-matched control.
- No calibration sweep. D4b remains unrun and every R3 number, folded or not, stays a floor.
- No `--lut`, no speed, no change to `donor_engine.c` — **the entire change is in the exporter**,
  and if it were not, that would itself be a finding.
- It does not heal. The fold is a re-parameterization; the `+2.708` that remains after it is
  still the healing brief's problem, and T2b §6 still says to aim that at attention.

## 7. Reporting

`probes/E2_RMSNORM_FOLD.md`. Per donor and per arm: `BPB(engine)`, `BPB(PyTorch)`, `Δ` vs the
engine's own F32 arm, the paired contrasts with SE and ci95, mean zero fraction, Gate F's two
numbers, the §5 label verbatim, and the two replication constants above with the file each is
read from.
