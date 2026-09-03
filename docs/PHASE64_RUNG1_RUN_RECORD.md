# Phase 64 rung-1 — main run record (where the weights are, and what they are)

**Status: TRAINING ON STANDBY as of 2026-09-03, by the owner's decision.** The run completed all four
curriculum stages; the ladder does **not** continue to S1/S2 for now, because the free Kaggle quota is
redirected to the donor-adaptation research branch (`docs/prompts/master_prompts/`,
`docs/research/donor_adaptation/`).

**Why this document exists.** The trained weights live on a local disk, in a directory whose name looks
temporary. This is the record that lets anyone — including us in six months — find them again, know
what they are, and know what was and was not measured on them. Every fact below was verified directly
against the artefacts on 2026-09-03, not transcribed from a report.

---

## 1. Where the artefacts are

**Primary location (machine-local, NOT in git, NOT on any remote):**

```
D:\_ktmp\mainrun_final\
```

| file | bytes | sha256 | what it is |
|---|---|---|---|
| `main_V2048_r26_CE.pt` | 124,449,467 | `e4f4fb1c0e29f02b1ef765e20a601544bdb839069c2b0c5665aa22ebfccb357c` | **THE FINAL MODEL.** gstep 183,104, end of stage F |
| `main_V2048_r26_CE.pt.done` | 5 | — | completion marker, contains `done` |
| `phase64-mainrun.log` | 13,298 | `eb5f11a89c642fa6a4cfa845763b942cbd4f65e4c5b63b233e19a2403485be76` | final-session log (JSON stream), carries the deciding BPB lines |
| `code/benchmarks/` | — | — | the trainer source as it ran |
| `stages_main_V2048_r26_CE/stage_D_ce_anchor_on_m0.pt` | 46,904,468 | `7a2d6e72b242f7c405b6fb2fdd4d8741c2ab2fab04b2bf9a338878fa2c433466` | end of stage D, gstep 137,328 — **10,916,576 params, 115 tensors: this is the model BEFORE the MoE upcycle** |
| `stages_main_V2048_r26_CE/stage_E_ce_anchor_on_m0.pt` | 124,112,249 | `a07224440f7d92fdc0c52fb557df99e81cde7c51688f5146f1c543c5486ca336` | end of stage E, gstep 173,949 — 29,987,809 params |
| `stages_main_V2048_r26_CE/stage_F_ce_anchor_on_m0.pt` | 124,423,609 | `f1e8a4369c98b51ad344daddfff23f14cc5148403c3432bdfde82003a074c599` | end of stage F, gstep 183,104 — same state as the final model |

**Sibling directories on the same disk:**

```
D:\_ktmp\dossier\     three forensic checkpoints from the fp16 divergence investigation (§6)
D:\_ktmp\upload\      resume_main_V2048_r26_CE.pt — the resume state at gstep 173,774
```

> ### ⚠ THE FINAL WEIGHTS EXIST IN EXACTLY ONE PLACE
>
> `main_V2048_r26_CE.pt` is **not** on Kaggle, **not** on GitHub, and **not** backed up. The Kaggle
> dataset `giggio253/phase64-mainrun-resume` holds only the *resume* state at gstep 173,774 — 9,330
> steps short of the end, and before stage F, which is where most of the quality gain happened (§4).
> Corpora and weights are never committed per project policy, so git will not save this. **If this
> directory is lost, the run is lost: ~145 GPU-hours across three accounts.**
>
> The identifying fact if the file ever moves or is renamed: **sha256 `e4f4fb1c…`, 124,449,467 bytes,
> 29,987,809 parameters, 136 tensors, `emb.weight` of shape (2048, 256).**

---

## 2. What the model is

Read from the checkpoint's own `cfg`, not from a plan document:

| | |
|---|---|
| ladder rung | **S0**, the first rung on real code data (`PHASE64_DECISIONS.md` §10) |
| parameters | **29,987,809** (136 tensors) |
| dims | D 256, V 2048, seq 512, micro-batch 8, accum 1 |
| x_proj | **low-rank r = 26** — the Inventor side-lab result, 17.6% of the dense bytes |
| MoE | `--sparse-moe`, E32 × h128 top-8 (introduced at stage E, §3) |
| recall slot | `--recall on` (131.1K params, gate initialised to 0) |
| supervision | **arm = `ce`** — plain cross-entropy, *not* KD. D3's MVE gate failed, so CE became primary and span-KD was demoted to challenger |
| precision | fp16 autocast + GradScaler (T4 is Turing: no bf16) |
| seed | **1** — not 0; the reseed after the first divergence (§6) |
| QAT ramp | `--qat-alpha 3600` (ramp length in steps) |
| hardware | 2 × T4, DDP, `--time-budget-min 660` per session |

**The exact command that produced it**, from the log:

```
python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29555 \
  benchmarks/phase64/mve/mve_train.py \
  --seed 1 --tag m0 --arm ce --recall on --stages CDEF --steps 183105 \
  --seq 512 --batch 8 --accum 1 --fp16 --warmup 200 --max-nonfinite 50 \
  --require-p62 --xproj-rank 26 --chunk-steps 0 --sparse-moe --qat-alpha 3600 \
  --expect-gpus 2 --expect-gpu-name T4 --min-host-ram-gib 20 \
  --data-dir /kaggle/input/datasets/wildpino/phase64-main-data/data \
  --ckpt-dir /kaggle/working --out /kaggle/working/main_V2048_r26_CE.pt \
  --resume-ckpt /kaggle/working/resume_main_V2048_r26_CE.pt \
  --save-stage-ckpt /kaggle/working/stages_main_V2048_r26_CE \
  --resume --time-budget-min 660 --ckpt-min 20
```

**One ambiguity flagged rather than resolved:** the cfg carries `"kd": "anchor"` alongside `"arm": "ce"`.
The arm is CE, so the KD field should be inert — but "should be" is not a measurement, and this project
has been bitten repeatedly by one name carrying two properties. **Confirm from the trainer source before
anyone relies on it.** The filename `stage_D_ce_anchor_on_m0.pt` inherits the same pair of tokens.

---

## 3. The architecture is not constant across the run — it is built by surgery

This is the thing most likely to confuse a future reader, and it is stated first because a checkpoint
loaded with the wrong architecture reports **nothing** (`strict=True` does not verify architecture when
two modules share parameter names). From the log's replay lines:

| entering | surgery |
|---|---|
| **stage D** | QAT switch: **24 MLP linears → `BitLinear158Alpha`** (weights carried over), lr × 0.5, α ramped 0 → 1 over 3,600 steps, ε-identity at α = 0 |
| **stage E** | MoE upcycle: **8 MLPs → E32 × h128 top-8** (magnitude-matched seed, active-only dispatch) **+ recall slot** (131.1K params, gate = 0) |

Consequences that matter:

- `stage_D_*.pt` is **10.9M params / 115 tensors** — dense, pre-upcycle. `stage_E/F` are
  **30.0M / 136 tensors** — MoE. **They are different architectures and are not interchangeable.**
- **The stage-D checkpoint is the natural matched-dense baseline** that the pre-registered property
  gate (`PHASE64_DECISIONS.md` §10, gate 1: *"code-val BPB vs the matched dense arm — the pre-upcycling
  dense checkpoint is the natural baseline"*) asks for. **It already exists on disk.** Nobody has run
  that comparison.
- Rebuild the architecture from the checkpoint's own `cfg` before loading. Never from a table in a tool.

---

## 4. The numbers

### 4.1 The deciding metric — pinned P62 code-val BPB

The only metric that decides anything, per the sealed eval hierarchy (`PHASE64_RUNG1_PREREG.md` v7).
Read from the log, marked `[DECIDING]`, byte invariant **HOLDS** at 1,499,998 scored bytes + 2 unscored
first-token bytes = 1,500,000 declared:

| point | gstep | P62 code-val BPB |
|---|---|---|
| stage C exit | 100,707 | **1.0787** |
| stage D exit | 137,328 | *(measured in an earlier session; not in this log)* |
| stage E exit | 173,949 | **1.0441** |
| **stage F exit — FINAL** | **183,104** | **0.9277** |

**Δ (C → F) = −0.151 = 30.2 σ_seed** at σ_seed = 0.005.

**Where the gain came from, and it is lopsided:** C → E is −0.0346 across 73,242 steps; **E → F is
−0.1164 across 9,155 steps.** The last stage delivered 3.4× the improvement in 12.5% of the steps. That
is an observation about the curriculum's final stage, not an explained mechanism — worth a look, not yet
a claim.

### 4.2 Comparability rulings — these are sealed, do not relitigate them

- **vs 1.242 (the 8.3M Phase-62 code number): NOT COMPARABLE.** Ruled before this number existed; the
  rung-1 validation set is temporally held out. Do not quote the pair, not even as "indicative".
- **vs 1.0787 (stage C exit): COMPARABLE.** Same run, same lineage, same pinned val, same protocol,
  byte invariant holding on both. This is the −0.151 above.
- **vs 1.1376 (screening arm 1, V=2048): DIRECTIONAL ONLY.** Different token budget, different slice.
  Never quote as a controlled comparison.
- **Standing caveat on the C endpoint:** 1.0787 was produced with the GradScaler at 2.0, i.e. under mild
  gradient truncation. Stage F's numerical regime is not established. The C → F delta therefore
  confounds "more training" with "different numerical regime". It does not change the sign.

### 4.3 The per-stage table from the checkpoint's `rows` — and its trap

| stage | steps | bpb_in | bpb_out | delta | tok/s | seconds |
|---|---|---|---|---|---|---|
| C | 100,707 | 4.017175 | 0.952010 | −3.065164 | 3582.47 | 10,736.0 |
| D | 36,621 | 0.952010 | 0.903715 | −0.048296 | 3835.23 | 33,556.3 |
| E | 36,621 | **0.972916** | 0.917704 | −0.055213 | 1251.85 | 1,145.2 |
| F | 9,155 | 0.917704 | 0.808513 | −0.109191 | 3693.84 | 20,303.5 |

> **⚠ TWO TRAPS IN THIS TABLE.**
>
> **(a) `bpb_in`/`bpb_out` is the STREAM-TAIL val, not the deciding metric.** It is apparatus under the
> sealed hierarchy. The tail is per-stage, so the columns are **not comparable across stages** — which
> is why stage E enters at 0.9729 after stage D exits at 0.9037, a +0.069 discontinuity that no delta
> explains. Use §4.1 for anything that matters.
>
> **(b) `seconds` is NOT the stage duration — it is the time since that stage's last resume.** Reading
> it as stage cost is wrong by up to ~20×. Verified: `seconds × tok/s ÷ 8192` gives 4,693 / 15,710 /
> 175 / 9,155 steps, i.e. resumes at ~96,014 (the healthy post-divergence-1 checkpoint), 121,618 (the
> post-divergence-2 restart), 173,774 (the final session), and stage F entire.

**That reconstruction is also the strongest integrity check on the artefact**, and nobody designed it:
the table independently regenerates all three known relay restart points, including both divergence
recoveries, none of which are recorded in it.

### 4.4 Health of the final checkpoint

| check | result |
|---|---|
| non-finite weight elements | **0** over 29,987,809 |
| non-finite entries in `hist` | **0** over 183,104 recorded steps |
| `hist` span | ('C', 1, 7.7701) → ('F', 183,104, 1.7192) |
| step count | ended at 183,104 against a target of 183,105 (off-by-one in the counter; not a truncation) |

**Measured incidentally:** stage E ran at 1251.85 tok/s against stage D's 3835.23 = a **3.06× seq-2048
penalty**, versus the **2.69×** measured in the pre-registration. 14% worse than predicted, same order.

---

## 5. What was NOT measured — the open work, if we come back

The run produced its number. **It did not produce its verdict.** The pre-registered per-rung property
gate set (`PHASE64_DECISIONS.md` §10) is entirely unread. All of it is offline, CPU-only, zero GPU,
from the checkpoints already on disk:

1. **code-val BPB vs the matched dense arm** — the baseline is `stage_D_*.pt` (§3). Not run.
2. **sparsity bands reproduce** (hidden ~92% / gate ~79%, in-place recall 86–92%). Not run.
   → **This is the one with a live dependency on the donor branch**, whose surviving lever after
   D1/D2/D4/D0/F1 all closed is dynamic activation sparsity — supported today only by probe-2 at 8.3M
   on a dense MLP. S0 is the only place to see whether the band holds at 3.6× that scale.
3. **router health + i.i.d.-union sanity** (0 dead experts, max/mean ≤ 1.5). Not run.
4. **recall property** (MQAR-style retrieval diagnostic) — the slot was on with gate initialised to 0.
   Not run.
5. **QAT gap vs scale**, record-only. Not run.

**Export gate:** formally due only at the top rung (S2), not here. If run anyway, note the engine is
compiled for `L 6 / V 1024` and this model is `L 8 / V 2048` — a `-D` recompile at minimum
(`benchmarks/phase60/engine.c:51-75`).

**Without gate 1, `0.9277` has no baseline and is not interpretable as good or bad.** It is a
measurement, not yet a result.

---

## 6. The two fp16 divergences — closed, with a dossier

The run diverged twice under fp16 and was recovered twice. Fully investigated; the mechanism was
localized to a measurement, not a guess.

**Summary:** the forward overflows fp16 at `blocks.7.mix.out_proj`'s input cast, channel 310, margin
1.179×, 11 elements of 2,097,152, the same channel across five independent batches. The state is a
resonance in the dt → dBx → scan chain, **emergent, not written in the weights** — all b7 tensors within
1.2× of a healthy reference while the activation sits at ~3,000×. **The loss scale can never fix it,
because scaling is applied after the forward**; the scale's descent is a clock, not a cause.

**The forensic artefacts, `D:\_ktmp\dossier\`:**

| file | gstep | scaler.scale |
|---|---|---|
| `ckpt_094419_div1_scale2m43.pt` | 94,419 | 2⁻⁴³ — **the decisive one**: its fp16 forward NaNs on essentially any batch |
| `ckpt_096012_healthy_scale4.pt` | 96,012 | 4.0 — healthy contrast |
| `ckpt_121618_postfix_scale16.pt` | 121,618 | 16.0 — post-intervention |

Neither checkpoint is corrupted: 0 NaN / 0 inf across 10,916,576 weight elements and 21,833,267
optimizer-moment elements. The GradScaler contract held throughout. Mirrored on Kaggle as
`wildpino/phase64-numerics-dossier` (CC0). Full narrative: the dossier's own `README.md`.

---

## 7. Kaggle infrastructure — what exists, and how it was run

- **Three accounts**, one real person each per Kaggle ToS: `wildpino`, `giggio253`, `sirwildpino`.
  30 GPU-h/week each. The run consumed roughly 145 GPU-hours in total.
- **Relay**: private repo `WildPino/kaggle-relay`, GitHub Actions, 4-hour cadence, handing the resume
  checkpoint from account to account. Driven by `scripts/kaggle_ops.py` + `kaggle_run.py`.
- **Known trap** (cost a night at launch): the global OAuth access token at `~/.kaggle/access_token`
  **overrides `KAGGLE_CONFIG_DIR`** — the wrong account gets used silently.
- **Datasets** (all private except where noted):

| id | contents |
|---|---|
| `wildpino/phase64-main-data` | the training corpus + tokenized ids |
| `giggio253/phase64-mainrun-resume` | resume state at gstep 173,774 + the stage checkpoints |
| `wildpino/phase64-numerics-dossier` | the three forensic checkpoints (CC0) |

- **Corpus**: 6.53 GB, Python-only, strict licence whitelist. Never committed, never redistributed;
  identified by manifest hashes in `PHASE64_RUNG1_PREREG.md` §"Data manifest"
  (`corpus_sha256 e7faf858…`, `slice_sha256 36aaa2ab…`).

---

## 8. If the ladder restarts

The next rung is **S1 = the same recipe with E128 ≈ 105M params**, then **S2 = E256 ≈ 206M**, the v1
product candidate. Both branch from `stage_F_*.pt` or from the final checkpoint (identical state).
The recipe is invariant by decision D1 — changing it breaks the ladder contract and voids the trend.

**Read the gate set in §5 first.** Climbing a rung without knowing whether the property held on the
rung below is precisely the mistake the ladder was designed to prevent.

---

## 9. Related documents

| what | where |
|---|---|
| sealed pre-registration (amendment window CLOSED at v9) | `docs/PHASE64_RUNG1_PREREG.md` |
| the rung-1 brief | `docs/PHASE64_RUNG1_BRIEF.md` |
| design decisions D1–D9 + the ladder + the gate set | `docs/PHASE64_DECISIONS.md` |
| the sizing model and measured rate curves | `docs/PHASE64_BUDGET.md` |
| architecture spec | `docs/SCALEUP_ARCHITECTURE.md` |
| eval contract | `docs/CANONICAL_EVAL.md` |
| the branch that took the quota | `docs/prompts/master_prompts/`, `docs/research/donor_adaptation/` |
| trainer source | `benchmarks/phase64/mve/mve_train.py` (+ the frozen copy in `D:\_ktmp\mainrun_final\code\`) |
