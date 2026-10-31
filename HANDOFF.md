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
| **S4** (the falsifier at m ≈ 2) | **FAIL (23 October): neither capacity arm beats its matched LoRA control** (R: c48 0.699, c496 0.695, a 0.505, b 0.377; [result](docs/research/CHATBOT50_ROADMAP_20261016/C50_S4_RESULT_20261023.md)). Before it: protocol pushed (22 October). MVE-1: T0 in PyTorch is T0. **MVE-2 PASS** (T4 x2: dequantization bit-identical on 159 tensors; fp16 vs P0 mean \|dlogp\| 0.0782, top-1 97.26%). MVE-4: bank of 1,492,153 rows (m ≈ 1.62). Arm (a) amended before any training (copies start unpicked, so the arm starts as T0). MVE-3 measured (1.56 h; 139-298 positions/s; arm (a) not exact at step 0 on the GPU, router-score fix pushed). Protocol: primary Δ = R(X) − R(matched LoRA) ≥ 0.20 on 36 S4 documents | [design](docs/research/CHATBOT50_ROADMAP_20261016/C50_S4_DESIGN_20261022.md), [protocol](docs/research/CHATBOT50_ROADMAP_20261016/C50_S4_PROTOCOL_20261022.md) |
| S5 | Not run: S4 has no winner to scale; the next step waits on a decision (COMMUNICATION voce 1) | ROADMAP §5 |

Pins: T0 = P0 = `ling-mini-2.0-a810f64-q4_k_m.gguf` (sha256 `44de7315…`); M0 = Q4_K_M without imatrix (`1f545368…`);
F0 = Ling-flash-2.0 `96f9b7a` Q4_K_M without imatrix (`5348c73f…`, 62,425,484,640 bytes); S1 expansions m2 / m4 / m6 =
`4e7fd2e3…` / `21ad0abe…` / `1d0021ab…` (full hashes in `benchmarks/chatbot50/results/c50_s1_expand_e*.json` and in
`c50_s1_run.py`).

## 3. What runs right now

**24 October, ~07:40: C50-T1 part 2b registered; arm A1 (anchored c48) goes to Kaggle.** Protocol: `docs/research/CHATBOT50_ROADMAP_20261016/C50_T1_PART2B_PROTOCOL_20261024.md`. Trainer `benchmarks/chatbot50/c50_t1_train.py` (kernel task `t1train-all-0.5`, then `t1train-all-0.5+resume` with `--sources` on the previous session; kernel names `c50-s4-t1-a1-s1`, `-s2`). After training: download `train/final.pt`, then `c50_t1_merge.py merge/quantize/layout` as in MVE-0, `c50_t1_run.py` (seven panels), `c50_t1_verdict.py part2b --arm a1 --merge --layout --run --lora-sha256 --json`. Correction made today: every T1 reader leaves out both halves of the donor-gap items (S5's half had leaked into MVE-0's and part 2a's general counts).

**24 October, ~17:40: A1 session 1 paused at step 889 of 1,240** (identity exact, 37.7 s/step, +29% over c48). Session 2 runs on **acct2**: kernel `giggio253/c50-s4-t1-a1-s2`, task `t1train-all-0.5+resume`, checkpoint mounted from the private dataset `giggio253/c50-t1-a1-ckpt` (`ckpt.pt` sha256 `cbe0ad17…`; the kernel log prints the sha256 of what it mounted). Output: `c50_s4_kaggle.py --acct acct2 output t1-a1-s2` -> `s4/kaggle_out/giggio253_t1-a1-s2/train/final.pt`.

**25 October, ~01:30: C50-T1 part 2b FAIL (G2, G3); the line is stopped for a design review, waiting on the user (COMMUNICATION item 1).** A1 keeps R 0.671 but loses nq 174 -> 155 and tqa 292 -> 272, and now mmlu and arc too. It loses the same items as c48, which are T0's low-margin ones. Result: `docs/research/CHATBOT50_ROADMAP_20261016/C50_T1_PART2B_RESULT_20261025.md`. Options A (null control with a random ΔW of A1's norms, local, recommended), B (T0's own closed-book answers as anchor, T4), C (lower τ), D (accept the trade). A1's Q4_K_M: `t1/t1_a1_q4_k_m.gguf` (`e490f803…`).

**25 October: option A chosen; null control registered (`cec2dda`) and running.** Protocol `C50_T1_PART2B_NULL_PROTOCOL_20261025.md`. Nulls built by `c50_t1_null.py` in `t1/null/` (rand, rot; `self` = control). Chain: scratchpad `t1_null.sh`, log `t1/null/null.log`, runs `results/c50_t1_null_run_{rand,rot}.json`. Then: `c50_t1_verdict.py null --variant rand t1/null/null_rand.json t1/null/merge_rand.json t1/null/L_rand.json results/c50_t1_null_run_rand.json --variant rot ... --json results/c50_t1_null.json`; the outcome is printed by the reader (2: rebase gates, user's choice; 1: option B).

**25 October, ~16:30: null control, outcome 2; the line waits on the user (COMMUNICATION item 1).** Both nulls (N-rand, N-rot, A1's per-tensor norm, no learned map) are already worse than T0 on nq and tqa by A4's rule; N-rand fails all four panels, N-rot three. R 0.011 and 0.077. So G2 as written measures perturbation. Post-hoc: A1 is not worse than either null on any panel. Result `C50_T1_PART2B_NULL_RESULT_20261025.md`, record `results/c50_t1_null.json` (`9ae37c80…`), post-hoc `results/c50_t1_null_posthoc.json`. The reader's first call was VOID (it read gsm8k; fixed in `3db1132`). Options: 1 amend G2/G3 to "not worse than a matched null (N-rot)" and confirm A1 vs N-rot on fresh items (recommended, ~5 h local, no GPU, needs a new item draw); 2 keep the gates; 3 accept the trade; 4 option B anyway. Null GGUFs: `t1/null/t1_null_{rand,rot}_q4_k_m.gguf` (`5ca9afe5…`, `793e5dd4…`).

**25 October, ~16:45: option 1 chosen; addendum 2 registered (`4e7c2ab`) and running.** G2/G3 amended to "A1 not worse than N-rot" (A4's paired rule) on nq, tqa, mmlu, arc, on fresh items (`c50_t1_fresh_tasks.py` -> `t1/fresh/inputs/`, nq 600, tqa 600, mmlu 566, arc 572, decontaminated against corpus v1). Chain: scratchpad `t1_fresh.sh`, log `t1/fresh/fresh.log`, runs `results/c50_t1_fresh_run_{nrot,a1,t0}.json` (~4.5 h). Then: `c50_t1_verdict.py fresh --inputs t1/fresh/inputs --t0 results/c50_t1_fresh_run_t0.json --a1 results/c50_t1_fresh_run_a1.json --nrot results/c50_t1_fresh_run_nrot.json --json results/c50_t1_fresh.json`. PASS -> step 2 of decision C (added experts on the residual, leaving T0 intact where they are off), its own protocol first; option B skipped. FAIL -> review. The user's reminder, which sets the direction: the point is knowledge from the 100B as auxiliary experts attached to the 10B, keeping its depth and speed.

**25 October, ~21:30: addendum 2 PASS (`1863633`); step 1 closes, v0 = T0 + A1.** On fresh items A1 is not worse than N-rot, nor than T0, on nq, tqa, mmlu and arc. Post-hoc CORRECTION (§12): every general panel since S4 left out the 409 donor-gap items, which censors the wins, so part 2b's and addendum 1's fact losses were the panel's; with them back in, all arms are ok. Arm (a)'s S4 regression survives on nq/tqa/gsm8k. From now on fact panels are plain draws (fresh items), donor-gap items reported apart. Step 2 draft: `C50_T2_DESIGN_20261025.md` (§12 PROPOSED). Waiting on COMMUNICATION voce 1: option X/N/A′ (recommended X, stitching MVE first) and whether the gate may use the "T0 is sure" target. Next after the answer: the stitching MVE protocol (needs a llama.cpp eval-callback tool to dump F0's MoE inputs), pushed before any run.

**25 October, ~22:20: option X chosen ("X, yes"); stitching cell registered (`aff1fbb`, `C50_T2_STITCH_PROTOCOL_20261025.md`).** Inputs `t2/stitch/inputs/` (128 docs of teacher shards 472-493). Tools: `c50_llama_dump.cpp` (bin `c50_llama_dump_avx2.exe` b961d7b2), `c50_t2_moe.py` (numpy F0 block, matches llama.cpp on v0 dev: top-8 1.0, cos 0.99986), `c50_t2_stitch.py` (measure), `c50_t2_stitch_verdict.py` (rule, selftest 8/8). F0 staged at `C:/SiliconLLM_stage/f0/` (sha 5348c73f; drive E: hung once on reads). v0 dump in session -> `C:/SiliconLLM_stage/t2/main/v0`; F0 dump must run from the user's terminal (`t2/stitch/run_f0_dump.cmd`, COMMUNICATION voce 1): the session host reaps F0's 62 GB for low memory. Then: `c50_t2_stitch.py --inputs t2/stitch/inputs --v0 C:/SiliconLLM_stage/t2/main/v0 --f0 C:/SiliconLLM_stage/t2/main/f0 --v0-gguf t1/t1_a1_q4_k_m.gguf --f0-gguf C:/SiliconLLM_stage/f0/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf --json results/c50_t2_stitch.json`, then the verdict script.

**26 October, ~00:10: stitching FAIL (`7883f4f`, `C50_T2_STITCH_RESULT_20261026.md`).** All controls fine. Linear map: about half of F0's routing (overlap 0.48-0.67, chance 0.05), little of the computation (cosine 0.40-0.79; 0.15-0.28 at matched depth, post-hoc). VOID before any number on the way (unescaped JSON path; tool rebuilt 9ae59cba, F0 dump redone into `C:/SiliconLLM_stage/t2/main/f0r`). Dumps stay on C: (v0, f0, f0r; ~85 GB; f0 is the invalid-JSON first copy, removable once nothing needs it). Waiting on COMMUNICATION voce 1: 1 = stitch trained locally on the experts' output (recommended), 2 = N, 3 = A′.

**26 October: choice 1, trained stitch registered (`C50_T2_TSTITCH_PROTOCOL_20261026.md`).** Script `c50_t2_tstitch.py` (bf16 F0 block on the GPU; arms ridge / linear / mlp, shuffled, half-data, PCA positive), verdict `c50_t2_tstitch_verdict.py` (self-test 11/11). Inputs copied to `C:/SiliconLLM_stage/t2/inputs`; F0 dev dump at `C:/SiliconLLM_stage/t2/dev/f0`. Run command in the protocol; ~5-6 h on the local GPU; log `C:/SiliconLLM_stage/t2/tstitch.log`.

**26 October: trained stitch STOP (`C50_T2_TSTITCH_RESULT_20261026.md`).** The planted positive missed the bar at v0 15 -> F0 24 and 16 -> F0 26 (budget-limited, calibrated at 2/16/31 only), so there is no verdict. Numbers: no layer stitches; training +0.06-0.21 cosine over the ridge, still 0.10-0.41 short; routing does not move; affine = MLP; data gain 0.062 per doubling. Waiting on COMMUNICATION voce 1: 1 = freeze X and go to N (recommended), 2 = X data cell, 3 = A′.

**26 October: choice 1 (distillation), option N registered (`C50_T2_N_PROTOCOL_20261026.md`).** X frozen with its state recorded. Code: `c50_t2_n.py` (NStudent, trainer, `--controls`, `--mve`), `c50_t2_n_eval.py`, `c50_t2_n_verdict.py` (self-test 8/8); kernel tasks `t2mve`, `t2n[+resume]`, `t2e2[+resume]`, `t2eval-<v0|n|e2>[_shuffle]` in `kaggle/c50_s4_kernel.py`. v0 is the private dataset `c50-t2-v0` on acct2 and acct3 (`c50_s4_kaggle.py --acct acctN stage|upload v0`); push with `--datasets OWNER/c50-t2-v0`. N trains on acct3 (`t2-n-s1`), E2 on acct2 (`t2-e2-s1`); outputs to `E:/.../s4/kaggle_out/`, then `t2/n/evals` and `t2/n/train` for the verdict.

**27 October: option N FAIL (`C50_T2_N_RESULT_20261027.md`).** N - v0 +0.0103, N - E2 -0.0013, E2 - v0 +0.0117 (R on 36 documents); the shuffle keeps 56%; fire 98%. N and X are both closed; the binding limit is block 1's teacher data. The choice returns: more F0 teacher data (S3 blocks 2+, ~32 h local GPU per block), a new route into F0's own weights, or stop step 2 with v0. Artifacts in `E:/.../t2/n/` (MANIFEST.json).

**27 October: the review directive governs (`C50_DIRETTIVA_REVISIONE_20261027.md`; rules in the local CLAUDE.md).** R rewards flatness (v0: R 0.673, R_acc 0.231). NF and the blocks 1+2 cell are stopped; S3 block 2 runs locally (`c50_s3_teach.py --until 20000000`, log `E:/.../s3/block2_launch.log`). Cell M registered (`C50_M_PROTOCOL_20261027.md`): kernel tasks `m-<s4|t2>-<arm>`, outputs to `E:/.../m/<arm>`, reading `c50_m_verdict.py`. Then directive point 5 (design review, targeted injection).

**27 October: cell M part 1 done (R DEMOTED, `C50_M_RESULT_20261027.md`); injection design reviewed (`C50_INJECTION_REVIEW_20261027.md`, independent review taken).** Next: the injection protocol (gates, MDE, code), its own review, then F0 work after block 2 (donor-gap scoring on the fresh panel, passages, rewordings, a batched-throughput run) and cell M part 2. NF's MVE ran ~40 min on acct3 and was deleted unread.

**27 October: injection protocol registered (`C50_INJECTION_PROTOCOL_20261027.md`; code `c50_inj_data.py`, `c50_inj_train.py`, `c50_inj_eval.py`, `c50_inj_verdict.py` self-test 15/15; kernel tasks `inj-*`).** Nothing of it has run. Its steps 1-6 need F0 locally, after S3 block 2 and cell M part 2; inputs go to `E:/.../inj/`, staged into the code dataset as `refs/inj` (`c50_s4_kaggle.py stage code`). The §12 last row now ends `| PROTOCOL |`.

**27 October: injection engineering smoke PASS (`inj-smoke`, acct1, 36 min; synthetic facts, no data read).** The MVE path, e and l trainings with marks, evaluations with fire (and `--rot`), v0's sampling and the verdict's reader on the real output files all ran; one bug fixed (numpy bools in the MVE record). A step takes ~35 s on T4x2. Not exercised: `inj-general-*` (it reuses `c50_t2_n_eval.py`, which ran in T2).

**28 October: S3 block 2 done (20,021,134 positions, 14:02); cell M part 2 RESULT (`C50_M_PART2_RESULT_20261028.md`): R_acc separates v0 from T0 on 72 unseen documents (+0.463 [+0.421, +0.506]).** F0's records and τ (1.325) in `E:/.../m2/f0` (`c50_m_f0.py`), arms in `E:/.../m2/<arm>`. Before any kernel mounts a code version: `c50_s4_kaggle.py --acct X stage code` for EACH account, upload, then `verify code`. The §12 last row now ends `| RESULT |`. Next: injection protocol step 1 (F0 on the fresh nq/tqa panel).
**28 October, later: injection steps 1-3' done; step 4 STOPped (182 of 424 kept, 76/75 per half).** The extension (protocol amendment, §12 DECISION row): `c50_t1_fresh_tasks.py --injection-extension` wrote `E:/.../inj/fresh2/inputs` (600 nq + 600 tqa, rows 1200-1799); then `c50_inj_data.py gen` T0 and F0 (max-new 32) into `inj/fresh2/{t0,f0}/<p>`; then `pool` (appends from fact 424, writes `qa_new.tsv`, `p1_new.tsv`); F0 on `p1_new.tsv` (max-new 96, as `f0_p1`) into `inj/f0_p1_new`; `inj-sample2` on Kaggle into `inj/v0_sample_new`; then `split`. The §12 last row now ends `| DECISION |`.
**29 October: step 4 PASS** (270 kept; 121 / 117 / 32). `inj-sample2` needs `--datasets wildpino/c50-t2-v0` (the first push without it failed in 12 s). Step 5 runs locally: T0 and F0 `reword`, F0 `type`, F0 `more`. The §12 last row now ends `| MEASURED |`.
**29 October, later: the injection is PAUSED after step 7b** (data kept: `E:/.../inj/`, dev runs in `inj/dev/<arm>-<lr>` with checkpoints in `s4/kaggle_out/*inj-dev-*`; to resume: `c50_inj_verdict.py --pick lr`, then step 8). Current work: the per-organ byte map (`benchmarks/wallmap/`). The §12 last row now ends `| PAUSED |`.

**24 October, ~07:00: C50-T1 part 2a done; nothing runs.** Result: `docs/research/CHATBOT50_ROADMAP_20261016/C50_T1_PART2A_RESULT_20261024.md`. attn-only merge R 0.678, ffn-only 0.533, both regress on tqa: no group is clean, so part 2's first T4 arm is the anchored LoRA (all of c48's modules). The two halves lose the same nq/tqa items (post-hoc), so the loss is in the training target. Partial T1s kept on E: (`t1/t1_c48_{attn,ffn}_q4_k_m.gguf`); the full BF16 merge `t1/t1_c48_bf16.gguf` (32.5 GB) is still there. Next: write and push part 2's T4 protocol before any T4 session.

**24 October, ~03:40: C50-T1 MVE-0 PASS; part 2a runs locally.** MVE-0 ([result](docs/research/CHATBOT50_ROADMAP_20261016/C50_T1_MVE0_RESULT_20261024.md), `c2438b8`): T1-c48 = Q4_K_M of T0's BF16 plus c48's ΔW, P0's recipe (K0 reproduces P0's sha256), P0's layout (only the 100 merged tensors differ), R_llama 0.697 against 0.699 in PyTorch; nq and tqa still regress. File: `E:/.../chatbot50_20261016/t1/t1_c48_q4_k_m.gguf` (sha256 `e644eab3…`). The pinned `llama-quantize` needs the WinLibs GCC 15.2 runtime on PATH (reinstalled 24 October; `t1_part2a.sh` in the scratchpad shows the PATH line); paths go with forward slashes, since the imatrix path is written into the file's header. Running: addendum 1 (part 2a, `1672670`), partial merges `--only attn` then `--only ffn`, each scored on nq, tqa and docs (log `t1/part2a.log`); then `c50_t1_verdict.py part2a --variant attn t1/L_attn.json benchmarks/chatbot50/results/c50_t1_part2a_run_attn.json --variant ffn t1/L_ffn.json benchmarks/chatbot50/results/c50_t1_part2a_run_ffn.json --json benchmarks/chatbot50/results/c50_t1_part2a.json`, whose registered rule picks part 2's first T4 arm. Part 2's protocol comes before any T4 session.

**24 October: option C chosen; C50-T1 part 1 (MVE-0) runs locally.** Decision and corrections: ROADMAP §12 (`c97e3e9`); protocol `docs/research/CHATBOT50_ROADMAP_20261016/C50_T1_PROTOCOL_20261024.md` (`3bb698e`). A background chain (scratchpad `t1_chain.sh`, log `E:/.../chatbot50_20261016/t1/chain.log`) runs K1 (patcher identity), the merge of c48's LoRA into T0's BF16 GGUF, K0 (the unpatched BF16 must re-quantize to P0's sha256), T1's Q4_K_M, L (layout against P0), then S2's seven panels on T1 (`c50_t1_run.py`, ~2.7 h; resumable). Then: `c50_t1_verdict.py mve0 --k0 t1/K0.json --k1 t1/K1.json --merge t1/merge.json --layout t1/L.json --run benchmarks/chatbot50/results/c50_t1_run.json --json benchmarks/chatbot50/results/c50_t1_mve0.json`. Part 2 (nq/tqa fix on T4) needs its own protocol first.

**23 October, ~22:40: S4 is closed, FAIL, and nothing runs.** All four arms trained and all nine evaluations (plus the VOID `eval-a_shuffle`, rerun) were read only after the last one existed; `c50_s4_verdict.py s4` gave FAIL for both capacity arms (Δ_a −0.190, Δ_b −0.322 against their LoRA controls). Records: `benchmarks/chatbot50/results/c50_s4_verdict.json`, the post-hoc `c50_s4_posthoc.json` (not registered: the LoRA controls regress on nq and tqa against T0, arm (b) does not), the inputs and their sha256 manifest under `E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s4/verdict/`, the trained weights in each final Kaggle kernel's output (`giggio253/c50-s4-train-a-s2`, `wildpino/c50-s4-train-b-s1r`, `sirwildpino/c50-s4-train-c48-s2`, `sirwildpino/c50-s4-train-c496-s2`). **Waiting on the user:** A (close capacity, go to S6 and Phase 2), B (S4-bis with S3 blocks 2-5), or C (recommended: a cell on LoRA distillation at v0, fixing the knowledge regression and checking the gain after merging into the engine's 4-bit weights). Budget ~66 of 75 session-hours spent.

**23 October, ~13:30: (c496) and (b) COMPLETE, evaluations running; (a) and (c48) still training.** c496: val KL 0.7196 -> 0.4558 in 11.49 h (ROADMAP §12, a139976); b: 0.7196 -> 0.5473 in one session, 8.96 h (0dd74df). Arm (b)'s evaluations run on acct1, one at a time, with `--sources wildpino/c50-s4-train-b-s1r,wildpino/c50-s4-bank`: `eval-b`, `eval-b-remove` and `eval-b-shuffle` done and downloaded under `s4/kaggle_out/`; `eval-b-mask` (`--task eval-b_mask`) running. Kernel names use hyphens (an underscore becomes a hyphen in the slug); the control goes in `--task`. Running: `giggio253/c50-s4-train-a-s2` and `sirwildpino/c50-s4-train-c48-s1`. Next: a PAUSED arm continues with `push train-<arm>-s<n+1> --task train-<arm>+resume --sources <owner>/c50-s4-train-<arm>-s<n>`; a COMPLETE arm is evaluated on the account that trained it (private outputs): acct2 `eval-a`, `eval-a-shuffle --task eval-a_shuffle`; acct3 `eval-c48`, `eval-c496` (sources `sirwildpino/c50-s4-train-c496-s2`). **Scores are not read until every evaluation exists**; then the evals directory (t0 = MVE-5's `eval-t0`, c48, c496, a, b, b_remove, b_shuffle, b_mask, a_shuffle) goes to `c50_s4_verdict.py s4 --evals DIR --train DIR --json`. Check every push's sources with `kaggle_ops.kaggle(acct, 'kernels', 'pull', ref, '-p', dir, '-m')` (the bare `kaggle` CLI fails on the OAuth token file). Budget ~40 of 75 session-hours spent.

**22 October, ~22:00: S4 trainings session 1 on Kaggle** (MVE-5 PASS, 28c813c): `wildpino/c50-s4-train-b-s1`, `giggio253/c50-s4-train-a-s1`, `sirwildpino/c50-s4-train-c496-s1`; each stops at 9.5 h with `train/ckpt.pt` (PAUSED) and continues in an `-s2` kernel with `--sources` on the `-s1` one; c48 follows c496 on acct3. Earlier, MVE-5 ran as kernel `wildpino/c50-s4-eval-t0`.

**23 October, ~01:40: arm (b) restarted from step 0 as `wildpino/c50-s4-train-b-s1r`** (`--task train-b --sources wildpino/c50-s4-bank`). Session `train-b-s1` was VOID: ERROR after ~2.75 h with an empty log and no files, a container-level death (a known-positive error kernel keeps both; arm (b)'s host path measured flat at 12.0 GiB of the 30 GiB limit); cause not identified (ROADMAP §12, 51a70c3). The trainer now logs cgroup memory per step and ends PAUSED-LOWMEM with `ckpt.pt` within 2 GiB of the limit; the code dataset with it (`c50_s4_train.py` sha e20b5be4) is on all three accounts, so the `-s2` sessions of (a) and (c496) and the c48 run use it. Arm (b)'s next session: `push train-b-s2 --task train-b+resume --sources wildpino/c50-s4-train-b-s1r,wildpino/c50-s4-bank`.

**23 October, ~07:40: session 2 of (a) and (c496) running** (`giggio253/c50-s4-train-a-s2`, `sirwildpino/c50-s4-train-c496-s2`). Session 1 (ROADMAP §12, 4940848): c496 paused at step 1,036 of 1,240, a at 639 (~55 s per step, so a needs a short session 3 too). Every resumed session from now on uses `--task train-<arm>+resume` (fails if no `ckpt.pt` is mounted). When c496 completes: `--acct acct3 push train-c48-s1 --task train-c48`. Budget ~25 h of 75 spent.
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
**29 October, stage 0:** the wall map protocol is pushed (`docs/research/WALL_MAP_20261029/`, code in `benchmarks/wallmap/`). ik_llama.cpp must be the `build-native-ua` build (alignment flag), or its FA crashes. The reference-speed run waits for the user's go (idle box, ~4-5 h). The §12 last row now ends `| PROTOCOL |`.
**29 October, stage 0 result:** `docs/research/WALL_MAP_20261029/WALL_MAP_RESULT_20261029.md` (one recommendation: the ternary kernel at Ling-flash's shapes first). Reference speeds are lower bounds from run 2 (VOID by the replacement rule). The decision on thresholds and next stage is the user's. The §12 last row now ends `| RESULT |`.
**30 October, C: reinstalled:** Python 3.12, Git (portable, `%LOCALAPPDATA%/Programs/PortableGit`) and WinLibs POSIX MSVCRT g++ 16.2 (winget) are reinstalled; the llama.cpp builds on E: need WinLibs' bin on PATH. The Ling-flash Q4_K_M copy staged on C: is gone; the original is on E:. The repo needs `safe.directory` (new user SID).
**30 October, stage 1 (a):** protocol `docs/research/STAGE1_20261030/S1A_FEWER_EXPERTS_PROTOCOL_20261030.md`, code `benchmarks/stage1/` (dry run PASS). Run: `python -I benchmarks/stage1/s1_run.py --work E:/SiliconLLM_Artifacts/stage1/s1a --out benchmarks/stage1/results/s1a_run.json`, then `c50_s2_checks.py` per arm and `s1_verdict.py`. The §12 last row now ends `| PROTOCOL |`.
**31 October, stage 1 (a) running:** launched 08:34 (commit `44dd2b7`). `benchmarks/stage1/s1_autoresume.ps1` (started at logon from the Startup folder entry `s1a_autoresume.cmd`) relaunches the runner after a shutdown, keeps a backup of the report and logs to `E:/SiliconLLM_Artifacts/stage1/s1a/autoresume.log`; it holds on any VOID or STOPPED panel. Remove the Startup entry when the run is DONE.
