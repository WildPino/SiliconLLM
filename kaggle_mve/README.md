# Phase 64.4 MVE — Kaggle run pack (3 accounts, 3 arms, in parallel)

> ## ⚠ RUN 3 — this is the current one. Read this section, then follow the per-account steps below.
>
> **What happened in run 2.** The CE control arm (B) diverged at step **24 of 20000** and never recovered, then ran
> ~11 h emitting `nan` and printed `MVE-DONE` anyway. Mechanism: the fp16 forward overflowed → `inf` logits → `NaN`
> loss → `GradScaler` skipped the step → the weights never changed → the same forward overflowed again. A
> self-sustaining deadlock the scaler cannot break, because it rescales *gradients* and the overflow is in the
> *forward*. Arms A and C survived only because `alpha=0.5` halves their CE contribution. **D3 is void for run 2.**
>
> **Two changes, both shipped in the run-3 code:**
> 1. **Divergence guard** (`--max-nonfinite 50`) — aborts with `MVE-DIVERGED`, exit code 2, and **writes no `.done`
>    file**. A diverged run can no longer masquerade as a completed one.
> 2. **`--warmup 200`** — linear LR warmup at each stage entry. The root cause was `lr=3e-3` applied flat from step 1.
>    This changes the recipe, so **all three arms are re-run** to stay comparable. That is the whole point of run 3.
>
> **You do NOT re-upload the 2.6 GB data dataset.** The code fix is 148 KB and rides in a second, tiny dataset
> (`code_v3/`). The notebook picks whichever attached copy actually contains the fix — a *content* check, so
> attaching the two datasets in either order works, and accidentally running run-2 code is impossible: the cell
> stops with `RUN-3 CODE NOT FOUND` instead.
>
> **Checkpoint names are new** (`resume_A3.pt` / `final_A3.pt`, and likewise B3/C3), so run 2's dead checkpoints
> cannot be picked up. You still want the wipe below to reclaim the disk.

<details>
<summary>Run 2 history (superseded — click to expand)</summary>

> ## ⚠ RUN 2 — the first run's KD arm was crippled by a bug. Read this before re-running.
>
> Run 1 (2026-07-16) completed all three arms, but the KD arm only received **~14%** of its intended KD signal
> (visible in the logs as `kd@0/8192`, `kd@490/8192` … where it should read `kd@~7900/8192`). Cause: the sampling
> window was computed with `searchsorted` on an **unsorted** key, so the trainer drew batches from the whole corpus
> while only 2 of 16 logit chunks were resident. Fixed; the smoke's end-of-C BPB improved 2.6082 → 2.4952 on the fix
> alone. **The D3 reading from run 1 is void — this pack re-runs it.**
>
> **BEFORE re-running, on each account, wipe the old outputs ONCE.** Persistence keeps `/kaggle/working`, so the
> buggy run's checkpoints and code would otherwise be silently reused. Run this in a scratch cell, once:
> ```python
> import subprocess
> subprocess.run('rm -rf /kaggle/working/*', shell=True)
> print(subprocess.run('ls -la /kaggle/working', shell=True, capture_output=True, text=True).stdout)
> ```
> (If you skip it, the run will refuse to start with a `STALE checkpoint` error rather than mix two apparatuses.)
>
> **Also new: both T4s.** The notebooks now use `GPU T4 x2` via DDP. Kaggle bills **session-hours, not GPU-hours**,
> so the second GPU is free quota-wise and buys ~1.7× → **~12 h/arm instead of ~22 h**. Effective batch stays 16
> (2 GPUs × micro-batch 8 × accum 1 = 16, same as the old 1 × 8 × accum 2).

</details>

Three independent training arms, one per Kaggle account, run at the same time → the whole MVE finishes in ~1 day
of wall-clock instead of ~3.5 days sequential on the 3060. All three read the SAME precomputed data; only the run
command differs. `$0`.

| folder | Kaggle account | arm | command flags | what it measures |
|---|---|---|---|---|
| `account_1/` | person 1 | **A** (primary) | `--arm kd --kd span --recall on` | — |
| `account_2/` | person 2 | **B** (control) | `--arm ce --recall on` | **A vs B = gate D3** (does span-KD beat plain CE?) |
| `account_3/` | person 3 | **C** (control) | `--arm kd --kd span --recall off` | **A vs C = gate D4 clause 2** (does recall/InfoNCE destabilize?) |

Each `account_N/` contains:
- `mve_data/` — the big data dataset (identical across the three: `code/` + `data/`, ~2.6 GB). **Already uploaded.**
- `code_v3/` — **NEW: the run-3 code dataset to upload** (148 KB). Its `mve_train.py` carries the divergence guard.
- `NOTEBOOK_cell.py` — **the one notebook cell to paste and run** on that account.

---

## Per-account steps (do this on each of the 3 Kaggle accounts)

**1. The data dataset — ALREADY DONE, skip it.** You uploaded this for run 2 and it has not changed. (Only if you
are starting a fresh account: *Create → New Dataset* → drag in the **contents of `account_N/mve_data/`** so the root
has `code/` and `data/` directly under it → Create, ~2.6 GB.)

**1b. Upload the run-3 code dataset — NEW, do this.** *Create → New Dataset* → drag in the **contents of
`account_N/code_v3/`** (the root will have `benchmarks/` directly under it) → title it e.g. `mve-code-v3` → Create.
It is 148 KB, so it processes in seconds.

**2. Create a notebook.** *Create → New Notebook*. Then in the right sidebar:
- **Add Input** → **both** datasets: the big data one *and* `mve-code-v3`. Order does not matter.
- **Accelerator** → **GPU T4 x2** (both GPUs; the cell auto-detects and runs DDP across them).
- **Persistence** → **Files only**  ← the important one: this keeps `/kaggle/working` across sessions so the
  resume works. Without it every session starts from scratch.
- Internet: not needed.

**3. Paste and run.** Open `account_N/NOTEBOOK_cell.py`, copy its whole content into the first (only) cell, Run.
It trains up to ~11 h, then stops cleanly and prints **`MVE-INCOMPLETE`**.

**4. Resume until done.** Next session (the GPU quota refreshes each week; a session caps at 12 h), open the same
notebook and **run the same cell again**. It auto-detects the resume checkpoint in `/kaggle/working` and continues.
Repeat — about **3 sessions per arm** — until the cell prints **`MVE-DONE`** and the file
`final_<A3|B3|C3>.pt.done` exists.

**5. Send back.** When an arm prints `MVE-DONE`: download `/kaggle/working/final_<A3|B3|C3>.pt` and copy the full cell
output text. Send both (the file + the text) to the Builder. The text alone already carries the whole result table
(per-stage BPB in/out, throughputs) — that is what the Architect reads for the gates.

---

## What "done" looks like (the cell output)

Each arm ends with a table like:
```
==== MVE run [full] arm=kd kd=span recall=on ====
  stage   steps    BPB in   BPB out    delta     tok/s     min
  C       11000    5.28xx    x.xxxx   -x.xxxx     xxxx    xxx.x
  D        4000    x.xxxx    x.xxxx   +x.xxxx     xxxx     xx.x
  E        4000    x.xxxx    x.xxxx   +x.xxxx     xxxx     xx.x
  F        1000    x.xxxx    x.xxxx   +x.xxxx     xxxx     xx.x
MVE-DONE: all stages complete.
```
Send this text for all three arms. The comparison is done by the Architect, not on Kaggle.

**If instead you see `MVE-DIVERGED`:** the arm blew up and the guard killed it. That is the guard doing its job —
run 2 lacked it and wasted 11 h. Do **not** re-run the cell (it would resume into the same dead state). Send the
Builder the output text; the arm needs a recipe change, not another session.

---

## Notes / gotchas

- **The three datasets are byte-identical** — they are three copies only because Kaggle datasets are per-account.
- **Config**: `--batch 8 --accum 2` = effective batch 16 (mathematically identical to batch 16; the split is only
  so the stage-E memory peak fits a 12–16 GB card). Do not change it.
- **`--steps 20000`, `--kd span`** are pre-registered (`benchmarks/phase64/MVE_PREREG.md`). Do not change them.
- **`--warmup 200` is a run-3 deviation from the prereg**, adopted deliberately after run 2's CE arm diverged, and
  applied to **all three arms** so they stay comparable. It must be recorded as a deviation when the gates are read.
- **8-bit optimizer**: if `bitsandbytes` is present the cell uses AdamW-8bit; if not it falls back to fp32 AdamW,
  which is fine at this size. Either is acceptable.
- **If a session dies early** (Kaggle hiccup, before the 11 h budget): no problem — a resume checkpoint is written
  every ~20 min, so at most ~20 min is lost. Just re-run the cell.
- **The separate 2×T4 DDP validation** (loss-curve parity + scaling ≥ 1.6×) is a short extra run; the Builder will
  hand you that cell separately if needed. It is not one of these three arms.
