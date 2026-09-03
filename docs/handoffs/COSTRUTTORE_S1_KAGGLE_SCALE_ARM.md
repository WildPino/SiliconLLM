# HANDOFF — COSTRUTTORE — S1 scale arm on Kaggle T4x2
updated: 2026-09-03T20:35+02:00   status: IN PROGRESS (handed to you, nothing done yet by you)
written by: the Adapter/Principal, before spawning you

## 1. The task, in one paragraph

Run the S1 sparsity probe across a **ladder of donor sizes** on Kaggle T4x2, so the programme can
see whether FFN sparsity — and the BPB cost of exploiting it — moves with model size. Everything
this programme knows about FFN sparsity was measured at Qwen2.5-1.5B, and the one published
measurement on the size axis runs the other way. "Done" is: a family of curves, BPB against
**achieved** FFN sparsity, one curve per donor size, all four arms, paired standard errors — plus
the oracle-on-`|h_i|` bound recomputed at every size.

## 2. Pre-registration you are bound by — READ IT FIRST, IT IS SEALED

`docs/research/donor_adaptation/briefs/BRIEF_S1_WHICH_BAR_PREDICTS_BPB.md`, and in particular
**AMENDMENT 1** (lines 144-245), commit `f90ac3d`, already pushed to `origin/research/donor-adaptation`.
You may not edit it. Read at minimum A1.2, A1.3, A1.4, A1.5.

The three constraints that will actually bite you:

- **A1.2 is non-negotiable and is the whole reason the arm is readable.** 1.5B was measured here on
  CPU in fp32. Kaggle is GPU fp16. A trend across those two conditions confounds SCALE with
  PLATFORM. You must reproduce the **1.5B** point on the GPU path with the same code and show it
  matches the local CPU number. **If |BPB_gpu − BPB_cpu| > σ_seed = 0.005 at the same p, STOP. Do
  not run the larger sizes. Report the discrepancy.** The entry script already enforces this; do
  not weaken it.
- **A1.3: fp16 on GPU, fp32 on CPU, and nothing else. No 8-bit, no 4-bit, ever.** This probe measures
  *activation* statistics; quantising the weights would change the object under test. If 14B does not
  fit in fp16 across the two cards, **skip it and report that it did not fit**. Three points
  (0.5B / 1.5B / 7B) already establish a trend, and an honest gap beats a contaminated point.
- **A1.6:** this probe measures whether the sparsity EXISTS, not whether our 48 KB block granularity
  can exploit it. D0 has answered the second question. **A good curve here does not revive the carve.**

## 3. What is DONE — with the artefact that proves it

| # | what | artefact on disk | key fact |
|---|---|---|---|
| 1 | The probe itself, all 5 arms + controls | `benchmarks/donor_adaptation/s1/s1_sparsity_bpb.py` | stages: `selfcheck`, `calib`, `run`, `gen`; `--ladder` selects `P_GRID_LADDER` |
| 2 | Selfcheck passed | `s1/results/s1_selfcheck.json` | — |
| 3 | Calibration at 1.5B | `s1/results/s1_calib_qwen2.5-1.5b.json` | — |
| 4 | The Kaggle driver, already written | `s1/s1_kaggle_entry.py` | it *drives* `s1_sparsity_bpb.py`; it does NOT re-implement it — keep it that way, or A1.2 compares two programs instead of two platforms |
| 5 | Bundle staged | `s1/_kaggle_bundle/` | `common.py`, `s1_sparsity_bpb.py`, `s1_kaggle_entry.py`, `s1_tables.py`, `corpus/manifest.json`, `results/slice_heldout_24x512_s1234.pt`, `dataset-metadata.json` |
| 6 | Kaggle plumbing exists | `scripts/kaggle_ops.py`, `scripts/kaggle_run.py` | **reuse them, do not invent a launcher** (A1.5) |

## 4. What is RUNNING right now — and it is your blocker

| what | PID | log | resumable? |
|---|---|---|---|
| S1 `run` stage, 1.5B, local CPU fp32, full `P_GRID` | **17884** | `s1/results/run_1.5b.part2.log` | **yes** — checkpoints after every `p` into `s1/results/s1_run_qwen2.5-1.5b.json`, reloads on restart |

Started 20:11 local. At handoff time it had finished `p = 1.0`, `0.9999` and was inside `0.999`.
Roughly 8 `p`-levels x 5 arms x ~3.5 min remain: **expect ~22:30-23:00 local.**

**This run produces the CPU anchor your A1.2 control needs.** `s1_kaggle_entry.py` reads it from
`anchor_cpu.json` (see `ANCHOR_CPU_PATH`, line 56) — that file **does not exist yet**. You must
generate it from the completed local artefact. Do not guess it, do not recompute it on GPU, do not
hand-type it.

**Do not kill PID 17884.** Do not start a second local torch job that competes for the 12 cores.

## 5. What is NEXT, in order

1. **Read the brief's Amendment 1.** Then read `s1_kaggle_entry.py` end to end.
2. **While the local run finishes** (it does not block this): verify the bundle is complete and
   self-consistent, and dry-run the entry script locally with `S1_ONLY=qwen2.5-0.5b` on CPU to prove
   the code path executes. This is a *code* smoke, not the science.
3. **Push the dataset / notebook via `scripts/kaggle_ops.py`.** Known trap, recorded in project
   memory and it has cost a night before: **a global OAuth access token overrides
   `KAGGLE_CONFIG_DIR`.** Check that the account you think you are on is the account you are on,
   by reading it back, before you launch anything.
4. **SMOKE on Kaggle at 0.5B** (`S1_ONLY=qwen2.5-0.5b`). End to end. Confirm artefacts come back.
5. **When PID 17884 finishes: build `anchor_cpu.json`** from `s1_run_qwen2.5-1.5b.json`.
6. **Run the 1.5B ANCHOR on Kaggle** and evaluate A1.2. **This gates everything after it.**
7. Only if the anchor passes: 7B, then 14B if and only if it fits in fp16 across both cards.
8. Report per A1.4: the family of curves, and the oracle-on-`|h_i|` bound at every size.

## 6. Traps already hit — do not repeat them

- **A background run launched as a child of an agent's shell dies with the agent.** This exact probe
  was killed twice by session rate-limits that way. Launch anything long with
  `Start-Process ... -PassThru` and write the PID to a file. See `docs/HANDOFF_PROTOCOL.md`.
- **Kaggle duplicates subprocess stdout 2-3x.** Never count log lines as evidence. Read the JSON
  artefacts back and use logical invariants.
- **Kaggle budget: 30 GPU-h/week per account, 3 accounts.** This is measurement, not training — no
  gradients, no optimiser. State a GPU-hour estimate before each real run.
- T4 is Turing: **fp16 native, no bf16.**
- `revision` is pinned per donor and resolves at download time — the script PRINTS achieved revision,
  layer count, `d_model`, `d_ffn`. Report achieved, never requested. This is a standing rule here.

## 7. Everything needed to restart cold

- Repo `D:\_THINGS\Progetti\SiliconLLM`, branch `research/donor-adaptation`.
- Probe dir `benchmarks/donor_adaptation/s1/`. Results land in `s1/results/` (env `S1_RESULTS`
  overrides; on Kaggle it is `/kaggle/working/s1_results`).
- Local run command (already running, for reference only):
  `python s1_sparsity_bpb.py run --model qwen2.5-1.5b`
- Env switches in the entry script: `S1_ONLY` (single size), `S1_LADDER_GRID` (reduced `P_GRID_LADDER`).
- σ_seed = **0.005 BPB**. This is the constant every delta in this project is judged against.
- Eval slice, identical everywhere: `heldout`, 24 x 512, seed 1234,
  `ids_sha256 = a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`,
  12,264 predicted tokens, 51,870 scored bytes. If your slice hash differs, stop — you are not
  measuring the same thing as everyone else.
- Local CPU baseline BPB at 1.5B: **0.767594952**, reproduced **bitwise identical** across two
  separate processes 9 hours apart.

## 8. Update this file

Rewrite sections 3, 4 and 5 after **every** milestone — bundle verified, smoke passed, anchor built,
anchor evaluated, each size done. Assume you will be killed without warning and that your successor
reads only this file.
