# HANDOFF — S1 SCALE ARM ON KAGGLE T4×2
updated: 2026-09-04T14:52+02:00   status: **SMOKE VERIFIED END-TO-END ON KAGGLE. Ready for the Owner to launch the real ladder.**
written by: the Adapter/Principal — smoke executed by the Kaggle operator (fresh session, no
prior context on this thread)

> **Read this file and the brief it names. You need nothing else and no prior conversation.**
> Everything that was blocking this arm has been done; what is left is Kaggle work.

---

## 1. What you are doing, in one paragraph

Every FFN-sparsity result this programme owns was measured on **one** donor, Qwen2.5-1.5B. The one
published measurement on the size axis runs the **other way** — sparsity *rises* with model size — and
our targets are 26B and above. So a negative measured at 1.5B may simply not transfer, and we would be
repeating a mistake this programme has already made once (validating a quantity at one width and
assuming it is scale-free).

**Your job: run the same probe, unchanged, on a ladder of donor sizes on Kaggle T4×2, and report BPB
against ACHIEVED FFN sparsity as a family of curves, one per size.** Plus the oracle-on-`|h_i|` bound
recomputed at every size.

## 2. The pre-registration — sealed, pushed, not editable

    docs/research/donor_adaptation/briefs/BRIEF_S1_WHICH_BAR_PREDICTS_BPB.md

**AMENDMENT 1, lines 144–245** is the operative part. Commit `f90ac3d`, on
`origin/research/donor-adaptation`. Read A1.2, A1.3, A1.4, A1.5 at minimum.

### The three rules that will actually bite you

**A1.2 — the anchor gate. Non-negotiable.**
1.5B was measured here on **CPU, fp32**. Kaggle is **GPU, fp16**. A trend computed across those two
conditions confounds SCALE with PLATFORM and is worthless. So the 1.5B point is re-run on the GPU path
and compared to the local CPU number, arm by arm. **If any comparison differs by more than
σ_seed = 0.005 BPB, STOP — do not run 7B or 14B.** `s1_kaggle_entry.py` enforces this automatically and
aborts the ladder. **Do not weaken it, do not raise the threshold, do not "just check the big ones".**

**A1.3 — fp16 on GPU, fp32 on CPU, and nothing else.**
**No 8-bit. No 4-bit. Not even to make 14B fit.** This probe measures *activation* statistics;
quantising the weights changes the very thing under test. **If 14B does not fit in fp16 across the two
cards, skip it and report that it did not fit.** Three points (0.5B / 1.5B / 7B) already establish a
trend, and an honest gap beats a contaminated point.

**A1.6 — scope.**
This measures whether the sparsity **exists**, not whether our 48 KB block granularity can exploit it.
That second question is answered elsewhere and is negative. **A good curve here does not revive the
carve.** Do not write that it does.

## 3a. Milestones done since the above (2026-09-04, this session)

- **Step 0 (whoami)** — `python scripts/kaggle_ops.py whoami`: all three accounts answer with their
  own server identity (`acct1=wildpino, acct2=giggio253, acct3=sirwildpino ok`). No OAuth-token
  override in effect.
- **Step 1 (local code smoke)** — `S1_ONLY=qwen2.5-0.5b S1_LADDER_GRID=1 python s1_kaggle_entry.py`
  ran on CPU/fp32 (2039 s). **SELFCHECK PASS, C1 IDENTITY PASS, C2 PLANTED FIRES** (+3.808 BPB at the
  full-precision comparison). NOTE for whoever runs this next on this machine: `os.path.isdir(
  "/kaggle/working")` false-positives on this box because `D:\kaggle\working` exists (unrelated old
  phase-56 scratch) — Python resolves a drive-less POSIX absolute path against the CWD's drive. Run
  the smoke from a C:-drive CWD (`cd /c/Users/<x>` first) or `ON_KAGGLE` silently takes the Kaggle
  branch and the local run dies on `glob.glob("/kaggle/input/...")` finding nothing.
- **Bundle rebuilt** — `python s1_kaggle_pack.py build`: `anchor_cpu.json` written from the
  COMPLETE local CPU run, 10/10 p-values, 257,336 B total.
- **Step 2/3 (push + GPU smoke at 0.5B)** — uploaded `wildpino/s1-sparsity-bpb-code` (acct1), pushed
  `wildpino/s1-sparsity-bpb-scale-arm` kernel version 1 with `--only qwen2.5-0.5b --ladder-grid`.
  `_wait_datasets_ready()` cleared with no delay. Kernel reached `KernelWorkerStatus.COMPLETE`.
  **Downloaded and verified from the JSON artefacts (never from log line counts):**
  - `gpu.n_devices_achieved = 2`, both `Tesla T4`, `torch_cuda = "12.8"` — real T4×2.
  - `dtype_achieved = "torch.float16"`, `devices_in_use_achieved = ["cuda:0"]` — fp16 policy held.
  - `selfcheck_pass = true`.
  - `sizes["qwen2.5-0.5b"].status = "OK"`, run `elapsed_s = 171.6`, kernel total `elapsed_s = 187.5`
    → **≈ 0.052 GPU-h consumed on acct1** (2.3× faster than the `estimate` table's 0.12 GPU-h guess).
  - `controls.C1_IDENTITY`: `bitwise_equal = true`, `verdict = "PASS"`.
  - `controls.C2_PLANTED`: `ratio_to_sigma_seed = 763.4`, `verdict = "FIRES"`.
  - `slice.ids_sha256 = a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65` — matches
    §5's required hash exactly; this run scored the pinned span, not a redrawn one.
  - GPU baseline BPB `0.8718624937236495` vs local CPU baseline `0.8717951206093644` at 0.5B — a
    delta of ~0.00007, well inside `σ_seed = 0.005`. (0.5B is not the anchor size; this is a good
    sign, not the A1.2 verdict, which is decided at 1.5B only.)
  - Artefacts downloaded to `s1/results/kaggle_smoke_0.5b/s1_results/`.
  - Cosmetic only: `kernels output` raised `UnicodeEncodeError` on a stray progress-bar glyph while
    ECHOING the CLI's own stdout to this console — the downloaded files themselves are intact and
    were read back successfully; this is the project's known "Kaggle duplicates/mangles stdout"
    trap (§5), not a data problem.

**Per standing practice (A1.5 / the project's "Owner launches long runs" rule), the Kaggle operator
stops here.** The full ladder (1.5B anchor gate → 7B → 14B) is a long unattended run and is the
Owner's to launch, not the operator's.

### Ready-to-run command

```
cd benchmarks/donor_adaptation/s1
python s1_kaggle_pack.py push acct1
```
(no `--only`, no `--ladder-grid` — the full `P_GRID`, the full `LADDER`.) This one kernel push runs
1.5B first, self-evaluates the A1.2 anchor gate from the bundled `anchor_cpu.json`, and only
continues to 7B/14B if it passes — otherwise it stops and reports the discrepancy, per Amendment 1.

**Stated GPU-hour estimate** (from `python s1_kaggle_pack.py estimate`, all figures there are
labelled ESTIMATED except the CPU timings; the one measured GPU data point above — 0.5B in 0.052 h
against the table's 0.12 h guess — suggests the true total will land at or under, not over, this
number): **≈ 5.8 GPU-h for the whole ladder** (0.5B+1.5B+7B+14B), well inside the 30 GPU-h/week
budget on one account. 14B may not fit fp16 across 2×16 GB — if so the script reports that and stops
there, per A1.3; it is not re-tried quantised.

Poll with `python s1_kaggle_pack.py status acct1`, download with
`python s1_kaggle_pack.py output acct1 -p <dir>` when `COMPLETE`. Read `anchor_check` out of
`s1_kaggle_manifest.json`, not the log (§5's stdout-duplication trap).

## 3. State — everything upstream of you is DONE

| # | what | artefact | status |
|---|---|---|---|
| 1 | The probe, all 5 arms + controls | `benchmarks/donor_adaptation/s1/s1_sparsity_bpb.py` | committed |
| 2 | Selfcheck | `s1/results/s1_selfcheck.json` | passed |
| 3 | Calibration at 1.5B | `s1/results/s1_calib_qwen2.5-1.5b.json` | passed |
| 4 | **Local CPU run, full `P_GRID`, COMPLETE** | `s1/results/s1_run_qwen2.5-1.5b.json` | **8128 s. `C1 IDENTITY: PASS`. `C2 PLANTED: +4.533885 → FIRES`** |
| 5 | **The A1.2 anchor — THE THING THAT WAS MISSING** | `s1/anchor_cpu.json` **and** `s1/_kaggle_bundle/anchor_cpu.json` | **built, committed, pushed (`3ba9bbb`)** |
| 6 | The Kaggle driver | `s1/s1_kaggle_entry.py` | written; *drives* the probe, does not re-implement it — **keep it that way or A1.2 compares two programs instead of two platforms** |
| 7 | Bundle | `s1/_kaggle_bundle/` | `common.py`, `s1_sparsity_bpb.py`, `s1_kaggle_entry.py`, `s1_oracle.py`, `s1_tables.py`, `anchor_cpu.json`, `corpus/manifest.json`, `results/slice_heldout_24x512_s1234.pt`, `dataset-metadata.json` |

**The anchor gives the gate real teeth: 10 p-levels × 5 arms = 50 arm comparisons, plus the baseline.**
It was projected *verbatim* from the completed local artefact — nothing recomputed, nothing rounded —
and carries the source file's sha256, the git revision, and the platform block. Local CPU baseline:
**0.7675949524755323**.

**Nothing is running locally that you depend on. There is no blocker.**

## 4. What to do, in order

**Step 0 — verify which account you are on.** This has cost the project a night before:

> **A global OAuth access token silently overrides `KAGGLE_CONFIG_DIR`.** You can be authenticated as
> a different account than the one you configured and never be told.

```
python scripts/kaggle_ops.py whoami
```
Read the answer back. Do not proceed on the assumption that it worked.

**Step 1 — local code smoke, no Kaggle, no science.** Prove the entry path executes:
```
cd benchmarks/donor_adaptation/s1
S1_ONLY=qwen2.5-0.5b S1_LADDER_GRID=1 python s1_kaggle_entry.py
```
(On Windows PowerShell: `$env:S1_ONLY="qwen2.5-0.5b"; $env:S1_LADDER_GRID="1"; python s1_kaggle_entry.py`)

**Step 2 — push the bundle as a dataset, then the notebook.** Use
`python scripts/kaggle_ops.py raw <acct> <kaggle cli args...>` for the CLI.

> ⚠ **Do not push the kernel immediately after creating the dataset.** A freshly created dataset is not
> yet processed and the kernel will fail to attach it. `kaggle_run.py` already solves this — reuse
> **`_wait_datasets_ready()` (`scripts/kaggle_run.py:139`, 1200 s timeout)**. Import it; do not
> re-write it and do not replace it with a fixed `sleep`.

**Step 3 — SMOKE ON KAGGLE at 0.5B.** `S1_ONLY=qwen2.5-0.5b`. End to end. Confirm the artefacts come
back down. **State the GPU-hours it consumed before going further.**

**Step 4 — the 1.5B ANCHOR run. This gates everything after it.**
The entry script loads `anchor_cpu.json`, runs 1.5B on the GPU, and evaluates A1.2 itself. Read
`anchor_check` out of the manifest JSON — **not out of the log** (see the trap below). It reports
`max_abs_bpb_delta_cpu_vs_gpu_ACHIEVED` over all 51 comparisons and a verdict string.

- **PASS** → proceed to 7B.
- **FAIL** → **STOP.** Report the discrepancy table. Do not run 7B. A failure here is not a
  disappointment, it is a finding: it would mean fp16 changes the activation statistics we are
  measuring, which is worth knowing on its own.

**Step 5 — 7B**, `device_map` across both cards (~15.2 GB fp16 in 2 × 16 GB).
**Step 6 — 14B** only if it fits in fp16 (~29.6 GB, tight). Otherwise report that it did not fit.

**Step 7 — report per A1.4**: the family of curves (BPB vs **achieved** sparsity, one per size, all
four arms, paired SEs), and the oracle-on-`|h_i|` bound at every size. Then answer A1.4's two
questions: does achievable sparsity at fixed BPB cost rise with size, and does the oracle bound move.
**A1.4 names the second as the outcome most worth finding** — if the bound falls materially with scale,
a closure elsewhere in the programme is size-local and must be reopened.

## 5. Traps — every one of these has already bitten someone here

- **The OAuth token overrides `KAGGLE_CONFIG_DIR`.** Step 0 exists because of this.
- **Kaggle duplicates subprocess stdout 2–3×.** **Never count log lines as evidence.** Read the JSON
  artefacts back and use logical invariants. Every claim you make must come from a file, not a console.
- **Dataset readiness** — see Step 2.
- **Budget: 30 GPU-h/week per account, three accounts** (`acct1`/`acct2`/`acct3`). This is measurement,
  not training — no gradients, no optimiser, no distillation — so it touches **no sealed constraint**.
  State a GPU-hour estimate before each real run.
- **T4 is Turing: fp16 native, no bf16.** Do not ask for bf16.
- **Report ACHIEVED, never requested.** The script prints achieved revision, layer count, `d_model`,
  `d_ffn`, dtype, devices. Those are what go in the report.
- **Anything long-running must be detached from your shell** (`Start-Process … -PassThru`, PID to a
  file). Two probes in this programme have been killed mid-run because they were children of a shell
  that died. See `docs/HANDOFF_PROTOCOL.md`.
- **If your eval slice hash is not
  `a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`, STOP** — you are not measuring
  the same thing as everyone else.

## 6. Restart-cold facts

- Repo `D:\_THINGS\Progetti\SiliconLLM`, branch `research/donor-adaptation`, at or after `3ba9bbb`.
- Ladder in the driver: `["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-7b", "qwen2.5-14b"]`.
  `BIG = {7b, 14b}` need `device_map` across both cards.
- Env switches: `S1_ONLY` (single size), `S1_LADDER_GRID` (reduced `P_GRID_LADDER`), `S1_RESULTS`
  (output dir; on Kaggle defaults to `/kaggle/working/s1_results`).
- Eval slice: `heldout`, 24 × 512, seed 1234, 12,264 predicted tokens, 51,870 scored bytes.
- σ_seed = **0.005 BPB** — the constant every delta in this project is judged against.
- Local CPU anchor baseline **0.7675949524755323**, reproduced **bitwise identical** across two
  separate processes nine hours apart.
- For reference, what the local run found at achieved sparsity 0.716 — arm A **+0.040984**,
  B **+1.819666**, C **+3.219699**, D **+0.040984**. A and D agree to the last digit at every level,
  as the pre-registered `NOTE_D` predicted. **This is context, not a target: do not tune anything to
  reproduce it beyond what A1.2's gate asks.**

## 7. Update this file

Rewrite sections 3 and 4 after every milestone — whoami verified, bundle pushed, smoke passed, anchor
verdict, each size done. Assume you may be cut off without warning and that whoever picks this up next
starts fresh and reads only this file.
