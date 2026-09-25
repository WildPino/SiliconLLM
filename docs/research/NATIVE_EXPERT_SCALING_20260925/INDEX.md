# Native expert scaling: research control index

**Date:** 25 September 2026. **Branch:** `research/native-expert-scaling`.
**Status:** NES-01 E128 training runs locally; its quality/routing/C-cost verdict remains open.
Fork point: donor pause checkpoint `90bf966`.

## Goal and current decision

Test the architecture thesis of `benchmarks/phase60/engine.c`: whether more
**independently learned experts** improve useful quality while a compact core,
fixed top-k routing, and affordable total work/DRAM traffic keep per-token cost
contained. [NES-01](E128_EQUAL_TOKEN_PROTOCOL_20260925.md) isolates E32→E128
at small L6/TinyStories scale. Equal tokens do not equal equal exposure per
expert; fixed top-k does not fix dense-router cost. A result at this scale
cannot establish target-scale performance.

The long-term target remains roughly 10B distinct learned parameters using
pretrained knowledge, quality and ≥50 end-to-end tok/s in the **same** C
artifact; 100 tok/s is a stretch goal. Scaling toward 100B is an open
question. [METHOD.md](METHOD.md) tracks a provisional transfer procedure and
its missing gates. The GigaChat arithmetic preflight below informs traffic
constraints but does not replace the native scaling decision.

The historical [donor-adaptation line is paused](../donor_adaptation/PAUSE_20260925.md).
Reuse its evidence and instruments; a direct port alone is a baseline and does
not finish the architecture goal. Read [PRIOR_EVIDENCE.md](PRIOR_EVIDENCE.md)
for native anchors and the [cross-program index](../RESEARCH_INDEX.md) for
scoped donor verdicts. New experiment records live beside this index.

## Latest decisive evidence

[NES-00](NES_00_ASSET_AND_DISPATCH_20260925.md) hash-verifies the native E32
checkpoint/data/export. E128 sparse dispatch agrees with compute-all to
`4.62e-7` maximum relative parameter-gradient difference on RTX 3060 and fits
3,234 MiB allocated at the intended batch/context. The 2-step smoke BPB is
not quality evidence; its apparatus voids are retained in NES-00.

For pretrained transfer, the local GigaChat base Q4 passes fresh paired BPB,
PIQA and document rollout against BF16, but its W4 large-matrix payload is
**calculated** at ~814 MB/token. Its C fidelity port is partial and no
quality-plus-≥50 C artifact exists. Qwen2.5-1.5B H1 shows that training a
carve helps; frozen H4+H2I composition fails. See [METHOD.md](METHOD.md) for
links and scope. No step currently transfers donor knowledge into the native
SSM/SWA target and passes joint quality/rate.

## Running cell and experiment register

| ID | State | Record / raw evidence |
|---|---|---|
| NES-00 | APPARATUS PASS; invalid smokes retained | [asset/dispatch record](NES_00_ASSET_AND_DISPATCH_20260925.md), [RTX 3060 JSON](dispatch_probe_e128_fullstep_rtx3060_20260925.json) |
| NES-01 | RUNNING from `44c7bb1` | [frozen E32→E128 protocol](E128_EQUAL_TOKEN_PROTOCOL_20260925.md); raw `nes01_e128_train.stdout.log`, `.stderr.log`, `nes01_status.json` when complete |
| METH-00 | ARITHMETIC PREFLIGHT | [GigaChat active-organ traffic](METH_00_GIGACHAT_COST_PREFLIGHT_20260925.md); no decoder or quality measurement |

NES-01 started **2026-09-25 11:28:15 UTC** on local RTX 3060 from launcher
`benchmarks/native_expert_scaling/run_nes01.ps1` (PowerShell PID `24620`,
Python child PID `19752`; venv redirector PID `10960`). At **14:31 UTC** its
rotating checkpoint reported **step 1800/4000**, with 117,964,800 layer-0
expert selections. The process was live at that inspection. Checkpoint path:
`results/native_expert_scaling/nes01_e128_resume.pt`; final output:
`results/native_expert_scaling/nes01_e128_final.pt`. The launcher writes
`nes01_status.json` and stops after at most 10 hours, leaving a resume state
if incomplete. Do not infer current progress from this snapshot; inspect the
process, status, log, and checkpoint when resuming. Avoid frequent polling and
CPU benchmarks during training. No T4 work is running or scheduled.

## Decisions and next exact action

- **Open:** E128 joint held-out/routing/generation/C-cost result; whether larger
  E gains useful capacity at equal tokens; how to reduce a pretrained donor's
  active traffic while preserving quality; C export/fidelity of that result;
  generality across families and scales.
- **Closed within their scope:** H5's frozen H4+H2I assembly is adverse;
  STRAT-03's tested shared/router geometry fails. Neither rejects fresh joint
  training. The old donor parity queue is not an automatic next step.
- **While NES-01 runs:** inspect the existing E4 export, reference and C
  measurement path for the exact E128 post-training comparison. Inspection
  found that `e4_export.py --ckpt` already supports an alternate checkpoint,
  but `engine.c` hard-codes `E=32` and `e4_reference.py` hard-codes the E32
  checkpoint/output paths and constructs its model with the E32 default.
  Parameterize these apparatus points before
  E128 C fidelity and timing; verify that E32 behavior remains intact. Avoid
  concurrent GPU or CPU benchmarks. METH-00 supplies a secondary donor traffic
  constraint; defer a donor quality sensitivity run until the native scaling
  result changes the architecture decision. No T4 job is planned.
- **After NES-01 finishes:** verify the final checkpoint/hash and compare with
  E32 under the frozen protocol. If no final checkpoint exists and the process
  is terminal, resume from the rotating optimizer/RNG state using the same
  command. Keep FAIL, VOID and incomplete runs in their records.

Preserve unrelated working-tree changes in `docs/research/RESEARCH_INDEX.md`
and `benchmarks/donor_adaptation/density/build_document_holdout.py`. Update
this index by replacing current state, and put detailed evidence in the
experiment record. Documentation is in English; no model/assistant signatures.
