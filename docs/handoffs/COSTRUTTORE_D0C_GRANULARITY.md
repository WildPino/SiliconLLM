# HANDOFF — COSTRUTTORE — D0c, the carve-granularity arm
updated: 2026-09-03T20:55+02:00   status: IN PROGRESS — script written, partitions building (PID 29376), full run held behind PID 17884
written by: the Adapter/Principal (§1-§2), then maintained by the Costruttore

## 1. The task, in one paragraph

Measure BPB of a co-activation MoE carve of Qwen2.5-1.5B **as a function of expert granularity `E`**,
at matched achieved activation fraction, with a random-partition null at every `E`. D0 Part III
measured exactly one granularity (`E = 32`, 280 neurons/expert). This arm measures three, plus a
conditional fourth, and pairs each with its null. **Done** is: `Δ(E)` and `G(E)` with paired standard
errors for every arm, and the pre-registered outcome label read off §4 of the brief.

## 2. Pre-registration — SEALED AND ALREADY PUSHED. Read it first, in full.

    docs/research/donor_adaptation/briefs/BRIEF_D0C_CARVE_GRANULARITY.md

Commit `4a98c89`, already on `origin/research/donor-adaptation`. **You may not edit it.** It fixes,
before any result exists: the arms (§3), the mandatory null control (§3.1), the achieved-not-requested
rule (§3.2), the decision rule with numeric thresholds (§4), and an explicit ban on extrapolating to
any `E` you do not run (§4, last paragraph).

**Read §4's outcome table carefully. One of its five outcomes — GRANULARITY-INVARIANT — makes D0's
existing negative STRONGER, not weaker. That outcome is as publishable as any other and you must not
treat it as a failure of the arm.** Report whichever row the numbers land on, verbatim.

### The one control that decides whether this arm means anything

§3.1: **every co-activation arm is paired with a random-partition null at the SAME `E`.** A finer
partition tracks any active set better whatever the ordering — smaller blocks waste less budget on
inactive neurons. If the null improves with finer `E` as much as the treatment does, granularity is
not evidence for the mechanism. **The deciding quantity is the coact-minus-null gap `G(E)`, not
`BPB(E)`.** Do not skip the nulls to save time. They are half the run and the half that makes it a
measurement.

## 3. What is DONE — with the artefact that proves it

| # | what | artefact on disk | key number |
|---|---|---|---|
| 1 | Brief and handoff read in full | `briefs/BRIEF_D0C_CARVE_GRANULARITY.md` @ `4a98c89` | 5-outcome rule, thresholds 0.20 / 0.10 |
| 2 | Part III driver **located** — it was NOT lost with its session | `C:/Users/giosa/AppData/Local/Temp/claude/D---THINGS-Progetti-SiliconLLM/1f268f4b-0a3c-40c3-a2f9-63f38f70de0c/scratchpad/carved_bpb_paired.py` | the exact code path to reuse; copied into the tracked script |
| 3 | Standing numbers extracted | `results/d0_carved_bpb_paired.json` | baseline `0.7675949641196624`; `Δ(all_coact)=1.090622885724487`; `Δ(all_null)=1.8111363812608539`; `G(32)=-0.7205134955363668` |
| 4 | Partition arithmetic checked | `d0_layout.py:balanced_labels`, `cap = N//E` | `d_ffn=8960` is divisible by 32/64/128/256 → **every arm's experts are exactly equal-sized, no remainder branch is taken**; nominal activation = k/E exactly. Still MEASURED per §3.2. |
| 5 | **Tracked script written** | `benchmarks/donor_adaptation/density/d0c_granularity.py` | syntax-checked; resumable per arm; replication gate hard-stops on failure |
| 6 | **Partitioner replicates Part III on ALL 28 LAYERS** | `results/d0c_labels/labels_E32.npz` vs `results/d0_carved_labels_E32.npz` | coact mismatching layers `[]`, null mismatching layers `[]` — **exact**. Expert sizes exactly 280/280 at E=32. |
| 7 | Partition cost measured | `results/d0c_partitions.log` | E=32: **240 s** for all 28 layers (8.6 s/layer). All four E expected ~15-20 min total. |
| 8 | **Statistics half replicates Part III BIT-EXACTLY** | inline probe over `results/d0_carved_arms/*.npy` vs `d0_carved_bpb_paired.json` | all six paired comparisons (4 vs-baseline + 2 coact-vs-null) agree to **0.00e+00** on delta, sequence-bootstrap SE, per-token SE and frac_tokens_worse; baseline BPB reproduces to all 16 digits |

**Consequence for diagnosis:** the statistics half and the partition half both replicate Part III
exactly. **The only unverified component left is the forward path**, and Part III's own §13.4
cross-process check put that at max deviation 3.8e-08 — 1e-5 of sigma_seed. If the replication gate
nevertheless fails, look at the forward path and nowhere else.

No BPB measured yet.

## 4. What is RUNNING right now

| what | PID | log path | resumable? | expected finish |
|---|---|---|---|---|
| S1 `run`, Qwen2.5-1.5B, CPU fp32 (**NOT MINE — DO NOT KILL**) | **17884** | — | — | started 20:11:21, expected ~22:30-23:00 local |
| my partition build, E=32/64/128/256 (setup only, no arms) | **29376** | `results/d0c_partitions.log` / `.err`, PID in `results/d0c_partitions.pid` | yes — caches `results/d0c_labels/labels_E<E>.npz` per E | ~21:25 local |
| my D0c arm run | not launched yet | `results/d0c_granularity.log` | yes, per-arm `.npy` in `results/d0c_arms/` | — |

**DO NOT KILL 17884.** Another figure is waiting on its output. 29376 is mine and is safe to kill —
it only builds label caches and each finished `E` is already on disk.

## 5. What is NEXT, in order

1. ~~Write the script.~~ DONE.
2. Wait for 29376 (partitions) → then smoke:
   `$env:D0C_SMOKE="1"; python benchmarks/donor_adaptation/density/d0c_granularity.py`
   (2 sequences, layer 27 only, arms baseline/A0/A2; ~2 min once partitions are cached).
3. Wait for **17884** to exit, then launch the full run detached:
   ```powershell
   $dir = "D:\_THINGS\Progetti\SiliconLLM\benchmarks\donor_adaptation\density"
   $env:D_THREADS="6"; $env:D0C_SECONDARY="0"
   $p = Start-Process python -ArgumentList "$dir\d0c_granularity.py" -WorkingDirectory $dir `
        -RedirectStandardOutput "$dir\results\d0c_granularity.log" `
        -RedirectStandardError  "$dir\results\d0c_granularity.err" -WindowStyle Hidden -PassThru
   $p.Id | Out-File -Encoding utf8 "$dir\results\d0c_granularity.pid"
   ```
4. Arms run in this order: `baseline`, `A0`, `N0` → **replication gate (auto, hard-stops with exit 3)**
   → `A1`,`N1`,`A2`,`N2` → optional `S1`,`S1n` (`D0C_SECONDARY=1`, second invocation, resumes).
5. Report to `docs/research/donor_adaptation/probes/D0C_GRANULARITY.md`. **NOT** into `D0_COACTIVATION.md` (under Controller audit).

## 6. Traps I have already hit / must not hit

- **B3 trap:** `torch.manual_seed(CLUSTER_SEED)` MUST be called immediately before each
  `balanced_labels` call — `torch.svd_lowrank` draws from the global torch RNG.
- **Thread count is part of the harness.** Part III ran at `torch.set_num_threads(6)`. fp32 matmul
  reduction order depends on thread count, so the replication run must also be 6 threads or the
  ~1e-6 agreement is not a fair test. Default `D_THREADS=6`.
- **Null seed formula** is `np.random.default_rng(1000 + E)` — Part III used `1032` for E=32.
  Generalised here to 1064 / 1128 / 1256. **Fixed now, before any result exists.**
- **Achieved, never requested.** §3.2: any arm whose achieved fraction differs from A0's by >0.002 is
  reported unmatched, not compared. The script measures achieved activation inside the hook.
- **Anything long-running must be detached** (`Start-Process ... -PassThru`).
- The oracle router stays an oracle; every Δ is a ceiling no real router can reach.

## 7. Everything needed to restart cold

- Repo `D:\_THINGS\Progetti\SiliconLLM`, branch `research/donor-adaptation`, HEAD `4a98c89`.
- Donor Qwen2.5-1.5B rev `8faed761d45a263340a0528343f099c05c9a4323`, 28 layers, d_model 1536, d_ffn 8960.
- Eval slice `heldout` 24x512 seed 1234, `ids_sha256 = a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`,
  12,264 predicted tokens, 51,870 scored bytes. **If the slice hash differs, stop.**
- Standing numbers to reproduce: baseline **0.7675950**, `Δ(E=32)` **+1.09062**, `Δ(null,E=32)` **+1.81114**, `G(E=32)` **−0.72051**.
- σ_seed = **0.005 BPB**. §4 thresholds: 0.20 BPB on Δ, 0.10 BPB on G.
- ~355-455 s per arm on 6 threads; 7 primary arms ≈ 45 min, +2 secondary ≈ 13 min.
- Resume: the script reloads any `results/d0c_arms/<arm>.npy` already on disk and recomputes only what is missing.
