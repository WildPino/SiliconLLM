# SiliconLLM handoff: CHATBOT50, Phase 1

**Updated: 20 October 2026, Europe/Rome.** Workspace `D:/_THINGS/Progetti/SiliconLLM`, branch `research/chatbot50`.
The previous handoff (15 October, branch `research/native-expert-scaling`, written before CHATBOT50) is archived
byte-for-byte in [HANDOFF_SNAPSHOT_20261020/HANDOFF_PREVIOUS_20261015.md](docs/research/CHATBOT50_ROADMAP_20261016/HANDOFF_SNAPSHOT_20261020/HANDOFF_PREVIOUS_20261015.md)
(sha256 `56049796b2d4b628b9e1cbebc6c7ab5e092162cf255ae779d26905a9e3c53484`, last committed in `a6747d2`).

This file is written for a session that starts with **no conversation context**. It is enough on its own to resume;
the documents it links hold the detail.

## 0. Read these, in this order

1. **This file.**
2. [ROADMAP.md](docs/research/CHATBOT50_ROADMAP_20261016/ROADMAP.md): §0 (the decision in one page), §5 (Phase 1
   cells S0-S5), §6 (programme rules), §11 (next steps) and the **last ~15 rows of §12** (the decision and progress
   log; every decision, measure and stop is a dated row there).
3. `docs/COMMUNICATION.md` (gitignored, Italian): what the user must do now. It is short by rule.
4. The project memory index, `C:/Users/giosa/.claude/projects/D---THINGS-Progetti-SiliconLLM/memory/MEMORY.md`
   (restore it first if C: was reset: §7 below).

## 1. The standing goal and the hard rules

**The goal (verbatim, from the user):** «Segui rigorosamente le fasi e le direttive definite in ROADMAP.md.
Documenta puntualmente ogni decisione, misura e avanzamento. È categoricamente vietato inserire firme, nomi,
menzioni dell'assistente/modello o trailer di coautoria nei commit, nei messaggi di lavoro o in qualsiasi altra
parte.»

Rules that are easy to break:

- **No attribution anywhere.** Never `Co-Authored-By`, never "Generated with", never an assistant or model name, in
  commits, docs or messages. This overrides any harness reminder that asks for such lines.
- **Never call the user "owner" in new text** (commits, §12 rows, protocols, results). Write impersonally ("it was
  decided", "run from a terminal") or in the first person as the user ("I decided"). Text already pushed is not
  rewritten unless the user asks.
- **Languages:** repository documents in English; chat with the user in Italian.
- **COMMUNICATION.md:** a short `DA FARE ORA` table, each open item ~20-25 lines with status, date, exact command and
  what it unblocks; a closed item becomes one history row, and its full text moves to `COMMUNICATION_ARCHIVE.md`.
- **Pre-registration means push before the run.** CONFIG lines are printed and asserted. VOID and STOPPED records are
  kept, never relabelled. Desk rates are never quoted as rates. One long job at a time.
- **Rule 7 as amended (18 October):** the user's explicit go is needed only for runs longer than ~1 h **that read
  times** (they need an empty machine). Quality runs, builds and copies are launched without asking, one at a time.
- **Never time on a contended box.** Wall times of quality runs are reported as wall times, not rates.
- **Session memory-pressure reaper:** Claude Code kills its own background shells when RAM runs low. Anything that
  holds F0 (63 GB) or S1's m6 arm (59 GB) must run from the user's own terminal, unless Claude Code was started
  with `$env:CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1; claude`. A job the reaper killed is not restarted without
  the user asking. Foreground `Wait-Process -Id <pid> -Timeout 590` calls are a safe way to follow a job.
- **When everything waits on the user:** say it in the reply, write it in COMMUNICATION, and start a background
  placeholder timer (`sleep 36000; echo ...`) so the goal check-in is deferred; stop it when the user answers;
  restart it once at most.
- **Do not stage these pre-existing local changes:** `benchmarks/donor_adaptation/configs/_manifest.json`,
  `benchmarks/donor_adaptation/density/build_document_holdout.py`, `docs/navigation/*`,
  `docs/research/RESEARCH_INDEX.md`, `AGENTS.md`, `KAGGLE.md`, `aqtinstall.log`, and the untracked corpora and
  results under `benchmarks/donor_adaptation/` and `benchmarks/chatbot50/corpus/c50_holdout_ext_v1/` (S1's runner
  reads `sel_ids.tsv` from that last folder; it must stay on disk).
- **Files with CRLF:** `ROADMAP.md`, the C50 protocol and result documents, `docs/COMMUNICATION*.md`. Edit them with
  a small Python script that reads and writes bytes (`open(p, "rb")`, append `\r\n`), not with heredocs.
- **Shell quoting:** bash heredocs and `-c` strings mangle backslashes and backticks; backticks inside a bash
  double-quoted string are executed. Write edit scripts to a file first, then run them. Commit messages go through
  `git commit -F -` with a quoted heredoc (`<<'EOF'`).
- **Processes:** never kill by name; read the command line first (`Get-CimInstance Win32_Process`).
- **Untrusted downloads:** run Python with `-I`, with the working directory set to a scratch folder.

## 2. Where Phase 1 stands (20 October)

| Cell | State | Record |
| --- | --- | --- |
| A4 / A4-bis (T0 quality) | **A4 PASS**; T0 = P0 qualified (P0 vs BF16 on 72 post-release documents: dBPB -0.0035, CI95 -0.0063..-0.0007) | [result](docs/research/CHATBOT50_ROADMAP_20261016/C50_A4BIS_POSTRELEASE_RESULT_20261018.md) |
| A1b (timing stability) | PASS, arm B0 (`--mem map`, normal priority) | `results/c50_a1b_verdict_2.json` |
| **S0** (v0 of T0 on the engine) | **Done. v0 = 35.94 tok/s at ctx 2048 (CI95 35.64..36.25), STABLE**; 39.01 at 512, 30.79 at 4096; organ breakdown validated | [result](docs/research/CHATBOT50_ROADMAP_20261016/C50_S0_V0_RESULT_20261019.md) |
| **S2** (Ling-mini -> Ling-flash gap map) | **MIXED, S3-S5 proceed** (RER_K 0.219, RER_R 0.247, D CI95 -0.111..+0.058) | [result](docs/research/CHATBOT50_ROADMAP_20261016/C50_S2_GAP_MAP_RESULT_20261019.md) |
| **S1** (speed invariance with capacity, m = 2, 4, 6) | **PASS in both router formats (20 October, run 3).** ρ at m2/m4: F32 1.011/1.009, BF16 1.003/1.003, every CI95 inside [0.97, 1.03]; raw v(m4)/v(1) F32 0.873, BF16 0.915 (the router's bytes). S4 and S5 use the BF16 router | [result](docs/research/CHATBOT50_ROADMAP_20261016/C50_S1_CAPACITY_INVARIANCE_RESULT_20261020.md) |
| **S3** (teacher signal) | **Option A, corpus (a), 50 M tokens. Protocol pushed; the GPU fidelity check FAILed (+0.0112 BPB, top-1 93.3%; addendum 1 = CPU), then the GPU configuration was chosen by decision, in blocks (addendum 2): block 1 = 10 M positions, COMPLETE 22 October (10,006,142 positions, 32.1 h, 86.5 tok/s; prediction RIGHT).** The user starts and stops it from their own terminal; the teacher is labelled "F0, GPU path" | [protocol + addendum 1](docs/research/CHATBOT50_ROADMAP_20261016/C50_S3_TEACHER_PROTOCOL_20261020.md), [fidelity result](docs/research/CHATBOT50_ROADMAP_20261016/C50_S3_FIDELITY_RESULT_20261020.md) |
| **S4** (the falsifier at m ≈ 2) | **Protocol pushed (22 October), nothing trained yet.** MVE-1: T0 in PyTorch is T0. **MVE-2 PASS** (T4 x2: dequantization bit-identical on 159 tensors; fp16 vs P0 mean \|dlogp\| 0.0782, top-1 97.26%). MVE-4: bank of 1,492,153 rows (m ≈ 1.62). Arm (a) amended before any training (copies start unpicked, so the arm starts as T0). MVE-3 measured (1.56 h; 139-298 positions/s; arm (a) not exact at step 0 on the GPU, router-score fix pushed). Protocol: primary Δ = R(X) − R(matched LoRA) ≥ 0.20 on 36 S4 documents | [design](docs/research/CHATBOT50_ROADMAP_20261016/C50_S4_DESIGN_20261022.md), [protocol](docs/research/CHATBOT50_ROADMAP_20261016/C50_S4_PROTOCOL_20261022.md) |
| S5 | Not started | ROADMAP §5 |

Pins: T0 = P0 = `ling-mini-2.0-a810f64-q4_k_m.gguf` (sha256 `44de7315…`); M0 = Q4_K_M without imatrix (`1f545368…`);
F0 = Ling-flash-2.0 `96f9b7a` Q4_K_M without imatrix (`5348c73f…`, 62,425,484,640 bytes); S1 expansions m2 / m4 / m6 =
`4e7fd2e3…` / `21ad0abe…` / `1d0021ab…` (full hashes in `benchmarks/chatbot50/results/c50_s1_expand_e*.json` and in
`c50_s1_run.py`).

## 3. What runs right now

**22 October, ~22:00: S4 trainings session 1 on Kaggle** (MVE-5 PASS, 28c813c): `wildpino/c50-s4-train-b-s1`, `giggio253/c50-s4-train-a-s1`, `sirwildpino/c50-s4-train-c496-s1`; each stops at 9.5 h with `train/ckpt.pt` (PAUSED) and continues in an `-s2` kernel with `--sources` on the `-s1` one; c48 follows c496 on acct3. Earlier, MVE-5 ran as kernel `wildpino/c50-s4-eval-t0`.
When it ends: `c50_s4_kaggle.py output eval-t0`, then `c50_s4_verdict.py mve5 --t0 <out>/eval --json
results/c50_s4_mve5.json` (a FAIL stops the line for a design review). After a PASS the four trainings start, one per
account at a time: b on acct1 (it reads `wildpino/c50-s4-bank`), a on acct2, c496 then c48 on acct3, e.g.
`c50_s4_kaggle.py --acct acct2 push train-a-s1 --task train-a`, a later session `push train-a-s2 --task train-a
--sources giggio253/c50-s4-train-a-s1` (it resumes from the mounted `train/ckpt.pt`); arm b adds
`--sources wildpino/c50-s4-bank`. Datasets `c50-s4-t0`, `c50-s4-block1`, `c50-s4-code` are on all three accounts
(sizes checked). The driver: `.venv/Scripts/python.exe -I benchmarks/chatbot50/c50_s4_kaggle.py [--acct acctN]
stage|upload|push|status|output`. Datasets on wildpino: `c50-s4-t0` (T0's GGUF), `c50-s4-code`
(code + gguf-py + refs, versioned), `c50-s4-block1` (S3 block 1 and the corpus). Nothing runs locally.

**Before that, 20 October, evening: nothing from the agent session.** The S3 teacher is the user's to start, from their own
terminal (command in COMMUNICATION and below); check its progress with
`python -I benchmarks/chatbot50/c50_s3_teach.py --status`. While it runs it holds F0 (62 GB mapped) and six cores:
no other long job and no rate cell beside it. F0 is staged at `C:/SiliconLLM_stage/f0/` (hash checked). The C: reset
was not needed. The user's `tiktok_publisher` daemon (another project) was suspended during S1 and resumed after it:
for a timed session, ask the user before pausing it again.

## 4. Next steps, in order

1. **Done (20 October): no restore was needed.** ~~After the C: reset, restore the environment~~ (§7): Claude memory, Python 3.12 with the packages, the two
   compilers (only if a tool must be rebuilt), `gh auth login`, the NVIDIA driver.
2. **Done (20 October).** ~~Re-stage S1 on C:~~ `python -I benchmarks/chatbot50/c50_s1_stage.py` (copies T0 and the three expansions from
   E: to `C:/SiliconLLM_stage/s1/`, re-hashes each against its pin; needs ~129 GB free on C:; ~1 h; reads no rate;
   launch it yourself, it is a copy). The S1 runner refuses any file whose hash differs.
3. **Done: S1 PASS (20 October, run 3, in the agent session with the reaper off; runs 1-2 VOID at the pre-flight).** ~~The user runs command E~~ (S1 session) from their own terminal, machine empty (browser closed):

   ```
   cmd /c "cd /d D:\_THINGS\Progetti\SiliconLLM && python -I benchmarks/chatbot50/c50_s1_run.py --out benchmarks/chatbot50/results/c50_s1_run_1.json > E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s1_run_1.log 2>&1"
   ```

   Then: `python -I benchmarks/chatbot50/c50_s1_report.py benchmarks/chatbot50/results/c50_s1_run_1.json --json benchmarks/chatbot50/results/c50_s1_report_1.json`.
   The gate is the residual ρ_f(m) = v_E / v_pred in each router format; PASS in a format iff its CI95 lies inside
   [0.97, 1.03] at both m2 and m4 (protocol, "Gate"). Score the registered predictions, write
   `C50_S1_CAPACITY_INVARIANCE_RESULT_<date>.md`, add a §12 row, update §11 and COMMUNICATION, push.
4. **Done (20 October).** ~~After S1:~~ remove `C:/SiliconLLM_stage/s1/` (the originals stay on E:), copy F0 from
   `E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/gguf/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf` to C:
   (it loads too slowly from the USB disk), hash it against `5348c73f…`.
5. **Done (20 October, §12 row MEASURED).** ~~S3, the F0 GPU study~~ (it needs F0 on C: and a terminal outside the reaper):
   - fidelity: F0 on the 72 FIT documents with the GPU build against F0's CPU run of S2
     (`E:/.../s2/f0/docs.pos.bin`, same record layout as A4's scorer): paired document dBPB, mean |dlogp|, top-1
     agreement. For T0 the GPU path was as close to BF16 as the CPU path (§5); F0 has no BF16 reference that fits.
   - wall throughput at `--ubatch` 512, 1024, 2048 (reported as wall, not rates), CPU vs GPU.
   - the user asked for **no quality loss**: a GPU configuration is adopted only if its fidelity check is fixed in the
     protocol before the run and passes.
6. **Done (20 October).** ~~Write and push the S3 protocol before any teacher data is kept~~: corpus v1 built and
   pinned (`results/c50_s3_corpus_v1_manifest.json`), protocol pushed, fidelity check run: **FAIL** (addendum 1: CPU);
   then **addendum 2: the GPU configuration in blocks, by decision.** The run itself is the user's (block 1 stops by
   itself at 10 M positions):

   ```
   python -I D:\_THINGS\Progetti\SiliconLLM\benchmarks\chatbot50\c50_s3_teach.py
   ```

   Ctrl+C stops it; the same command resumes it (at most the shard in progress is lost). The first start hashes F0
   (~4 min). **Decided 20 October evening: option B** (the GPU in blocks; the alternatives were the CPU stopped at
   S4's volume, or no teacher). T4s cannot hold F0 Q4 (62 GB against 32 GB VRAM + ~30 GB host RAM). A later block
   (`--until` raised, up to 49,892,578) needs its own §12 row: S4's volume or its pilot. Block 1 ended 22 October and
   is scored in §12 (RIGHT). Nothing runs now.
   When the first day has run, score the throughput prediction (14-22 tok/s, addendum 1) in a §12 row.
7. **S4: protocol pushed (22 October), after MVE-1, MVE-2 (PASS), MVE-3 and MVE-4**
   ([design](docs/research/CHATBOT50_ROADMAP_20261016/C50_S4_DESIGN_20261022.md)). Code: `c50_s4_dequant.py` (Q4_K/Q6_K
   in PyTorch), `c50_s4_t0_torch.py` (T0's forward), `c50_s4_t0_gpu.py` (T0 from its quantized blocks on two GPUs,
   MVE-2), `c50_s4_student.py` (arms, block-1 reader, 33-bucket KL, MVE-3), `c50_s4_kaggle.py` + `kaggle/c50_s4_kernel.py`
   (Kaggle), `c50_s4_split.py` (S4/S5 halves of the donor-gap panel). Next, in the protocol's order
   ([protocol](docs/research/CHATBOT50_ROADMAP_20261016/C50_S4_PROTOCOL_20261022.md)): build `c50_s4_bank.py` (CPU),
   `c50_s4_train.py` (checkpoint/resume), `c50_s4_eval.py` (KV cache), `c50_s4_verdict.py --selftest`; upload the
   datasets to the other two accounts; MVE-5 (evaluator fidelity); then the four trainings, each refusing to start
   unless the arm is exactly T0 on the GPU. Budget: MVE-2 ~0.2 h and MVE-3 1.56 h of session; S4 announced at
   <= 75 session-hours. Original plan: arms (a) extra experts,
   (b) memory bank, (c) LoRA control; gate on S2's donor-gap panel; BF16 router (S1). It fixes how much teacher signal
   it needs; any finished prefix of the S3 run is a balanced mix.

## 5. The S3 teacher tool (built 19 October)

- **Source:** `benchmarks/chatbot50/c50_s3_teacher.cpp`. Output in `--out DIR`: `run.json` (run identity; a resume
  with another identity exits 4), `shard_NNNNNN.bin` (per position: int32 doc, int32 pos, int32 target, float
  logp_target, double lse, then K pairs of int32 id and float logp, sorted; 24 + 8K bytes, **280 B per token at K =
  32**, not the 192 B of the sizing note, which assumed f16 values), `manifest.jsonl` (one line per finished shard).
  A stop loses only the shard in progress.
- **CPU build:** `sh benchmarks/chatbot50/c50_build_scorer.sh c50_s3_teacher` → `bin/c50_s3_teacher_avx2.exe`
  (sha256 `2cb54d84…`). On 2 documents its logp_target, log-sum-exp and top-1 are **bit-identical** to A4's P0
  scorer; a deleted shard is rebuilt byte-identical.
- **GPU build:** `sh benchmarks/chatbot50/c50_build_teacher_gpu.sh` links the official llama.cpp release of the
  pinned commit (`b11040` = `5b335f4`, CUDA 12.4, x64; zips checked against the release digests) and writes
  `c50_s3_teacher_cuda.exe` **next to the release DLLs** in
  `E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama-b11040-win-cuda-12.4-x64/bin/` (sha256
  `8a079d53…`); the release loads its backends from the executable's folder. No CUDA toolkit is installed or needed.
  Options: `--ngl N`, `--cpu-moe` (routed experts stay in host memory); with `--ngl 0` llama.cpp still offloads large
  batches (op offload). `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` forces fp32 cuBLAS.
- **T0 evidence (72 FIT documents, against A4's BF16):** CPU dBPB -0.0063, mean |dlogp| 0.2257, top-1 92.65%; GPU
  ngl 0: -0.0028, 0.2233, 92.72%; GPU fp32 + ngl 99 + cpu-moe: -0.0047, 0.2242, 92.71%. GPU vs CPU per token: mean
  |dlogp| ~0.125, top-1 ~95.8%. Wall throughput on a busy box: ~320 tok/s GPU against ~60-70 CPU. Data and scripts:
  `E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s3/gpu_fidelity_t0/` (`compare.txt` has the table).
- **Hardware:** RTX 3060 12 GB, compute 8.6, driver 610.88, PCIe link x8 (host max gen 4). Ryzen 5 3600X, 80 GB RAM
  (~71.7 GB free at idle).

## 6. Where things are

- **Repository:** `D:/_THINGS/Progetti/SiliconLLM`, branch `research/chatbot50`, remote pushed. Campaign code and
  results: `benchmarks/chatbot50/` (binaries in `bin/` are gitignored; their hashes are pinned in the runners and
  protocols).
- **Artifacts (USB disk E:):** `E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/`: `gguf/` (every GGUF:
  BF16s, T0, M0, F0, plants), `s1/` (expansions), `s2/` (S2 panel outputs per arm), `a4/` and `a4bis_post/`
  (inputs and outputs), `s3/gpu_fidelity_t0/`, `tools/llama.cpp-5b335f4/` (pinned source with the static
  `build-avx2` and `build-generic`), `tools/llama-b11040-win-cuda-12.4-x64/` (GPU release), `sources/` (downloaded
  checkpoints), run logs at the top level.
- **Backups made before the C: reset:** `E:/SiliconLLM_Artifacts/backups/c_drive_20261020/` (§7).

## 7. The C: reset (20 October)

**What was on C: for this project, and what happens to it:**

| On C: | Size | Status |
| --- | --- | --- |
| `C:/SiliconLLM_stage/s1/` (T0 + m2, m4, m6) | 120 GB | Copies; originals on E:. Re-stage with `c50_s1_stage.py` |
| `C:/SiliconLLM_stage/s2/` | empty | F0 already moved to E: (hash checked) |
| Claude project memory and transcripts, `C:/Users/giosa/.claude/projects/D---THINGS-Progetti-SiliconLLM/` | 0.3 GB | **Backed up** to `backups/c_drive_20261020/claude_project_D---THINGS-Progetti-SiliconLLM/`; copy it back to the same path after the reset |
| Global `C:/Users/giosa/.claude/CLAUDE.md` and `settings.json` | small | **Backed up** (`claude_global_*`) |
| Session scratch scripts (temp folder) | 5 GB, mostly scratch data | Scripts **backed up** to `scratchpad_e9133806/`; the GPU study data is already on E: |
| Python 3.12.10 (`C:/Users/giosa/AppData/Local/Programs/Python/Python312`) | 2 GB | Package list **saved**: `pip_freeze_python312.txt` (numpy 2.4.6, torch 2.12.0, transformers 4.57.6, tokenizers 0.22.2, safetensors 0.8.0, huggingface_hub 0.36.2) |
| LLVM-MinGW 21.1.8-20251216 (winget `MartinStorsjo.LLVM-MinGW.MSVCRT`) | 0.6 GB | Reinstall that exact version to rebuild the engine (`winget install --id MartinStorsjo.LLVM-MinGW.MSVCRT --version 21.1.8-20251216`) |
| WinLibs GCC 15.2.0-13.0.0-r6 (winget `BrechtSanders.WinLibs.POSIX.MSVCRT`) | 0.9 GB | Reinstall that exact version to rebuild the llama.cpp tools: the pinned static libraries were built with it |
| Hugging Face cache `C:/Users/giosa/.cache/huggingface/` | 106 GB | **Not backed up.** Older-phase models (StdMoE 1b14b 51 GB, Qwen2.5 family, StarCoder2-3B, DeepSeek-Coder-1.3B, Whisper). CHATBOT50 does not use it (its checkpoints are on E: under `sources/`) |
| Kaggle credentials `C:/Users/giosa/.kaggle/` and `C:/Users/giosa/.kaggle_accounts/` (acct1-3) | small | **Credentials: the user saves them**; not copied by the agent. Needed for the Kaggle multi-account playbook |
| `gh` login (`%APPDATA%/GitHub CLI/hosts.yml`) | small | Credentials: `gh auth login` again after the reset |

No directory junction in the repository points to C: (scanned to depth 4). The repository, `D:/_ktmp/` (rung-1
weights) and every CHATBOT50 artifact are on D: or E:.

**22 October, a Windows reset (not planned):** the base Python 3.12.10 folder disappeared and was reinstalled by the
user **without packages** (the MSVC runtimes were restored too). Local work now runs from the project's `.venv`
(`D:/_THINGS/Progetti/SiliconLLM/.venv/Scripts/python.exe`: torch 2.6.0+cu124 with the RTX 3060, numpy 2.4.6,
kaggle); the bare `python` has no torch. MVE-1, MVE-2's fingerprints and the CPU dequantization check ran earlier that
day with torch 2.12.0+cpu. To rebuild the base packages, the freeze file is in the C: backup above.

**After the reset, before any work:** restore the Claude memory folder; install Python 3.12 and the packages from the
freeze file; `gh auth login`; check `nvidia-smi`; check `git -C D:/_THINGS/Progetti/SiliconLLM status`; then §4.

## 8. Pitfalls already met

- The llama.cpp tools' projected-time guard (`--max-projected-s`) reads the first item, which carries the mmap
  page-in; it stopped F0 wrongly once. S2's addendum 1 replaced it for F0 with a 6 h wall limit per process.
- F0 from the USB disk loads very slowly; keep it on C: while it runs.
- `c50_s1_run.py` and `c50_s0_run.py` pin their binaries and sources by hash; a rebuilt binary with a different hash
  is refused, by design. A rebuild needs an addendum.
- The S3 sizing note's hosted prices are unverified aggregator figures; option A does not depend on them.
- **A panel whose sequences share a first token weighs one context many times.** S3's fidelity rule on MMLU (every
  sequence starts with "Question") put 84% of its gap on position 1, where any Q4 build is ~1.5 nats from BF16 and
  the value moves with the batch size. A bootstrap over sequences does not see that the draws are one context.
- **The CUDA release build differs from our AVX2 build even with `--ngl 0`**: its CPU code is not S2's. Only
  `bin/c50_s3_teacher_avx2.exe` reproduces S2's scores bit for bit.
- **The drives:** only C: is NVMe (500 GB); D: and E: are USB. F0 BF16 (206 GB, on E:) runs only on a few tokens
  (~37 min for a 4-token input).
