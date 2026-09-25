# Native expert scaling: research control index

**Date:** 25 September 2026. **Branch:** `research/native-expert-scaling`.
**Status:** NES-01 E128 training running on the local RTX 3060.
Fork point: donor pause checkpoint `90bf966`.

## Latest decisive evidence and active cell

[NES-00 asset/dispatch audit](NES_00_ASSET_AND_DISPATCH_20260925.md) verifies
the local E32 checkpoint, data, tokenizer and E4 export against release hashes.
No trained E128 checkpoint was found in the reviewed native assets. The
existing sparse-slot path agrees with compute-all at E128 within `4.62e-7`
maximum relative parameter gradient difference on RTX 3060, and the full
batch/context apparatus step fits 3,234 MiB peak allocated. The 2-step smoke
is not a quality result; its invalid zero-window BPB and the first failed
resume attempt are retained as apparatus voids in NES-00.

The current cell is [NES-01, fixed-token E32→E128](E128_EQUAL_TOKEN_PROTOCOL_20260925.md).
It holds L6/TinyStories, core, h128, top-8 and 32.768M training tokens fixed.
Its E128 run is estimated at 7–9 local RTX 3060 hours and capped at 10 hours,
with optimizer/RNG checkpoints. The L8/code ladder remains separate and
requires its own matched E32 baseline.

NES-01 started 2026-09-25 11:28:15 UTC from commit `44c7bb1` via
[`run_nes01.ps1`](../../../benchmarks/native_expert_scaling/run_nes01.ps1).
Hidden PowerShell launcher PID 24620, recorded in
`results/native_expert_scaling/nes01_launcher.pid`; Python child PID at launch
10960 (the venv redirector may create another Python process). Expected
outputs: `nes01_e128_train.stdout.log`, `.stderr.log`, `nes01_status.json`
in this directory; rotating optimizer/RNG state and final checkpoint under
`results/native_expert_scaling/`. Check the status JSON or process once after
the expected 7–9 hours; avoid continuous polling. A 10-hour wall cap writes
`RUN-INCOMPLETE` and leaves a resumption state if the run is not finished.

## Decision and objective

Test whether increasing the number of genuinely learned experts improves or
preserves useful quality while routing and per-token cost remain affordable,
using the native `benchmarks/phase60/engine.c` recipe and prior tests.
The independent variable is expert count **E**, not core width, top-k or
expert width. Larger stored capacity must not be mistaken for larger useful
capacity. Copied experts and synthetic weights do not prove the latter.

The donor line is [paused](../donor_adaptation/PAUSE_20260925.md), not deleted or
scientifically refuted. Its port and numerical references remain available.
This pilot does not replace the eventual pretrained-large-model goal with a
small model trained from scratch. A native scaling result and a feasible
pretraining-transfer route are two separate requirements.

## Read first; do not restart from scratch

1. [Prior evidence and reusable assets](PRIOR_EVIDENCE.md): measured scope,
   proposed versus executed rungs, known failures and no-duplication boundaries.
2. [Native architecture](../../SCALEUP_ARCHITECTURE.md), including its later
   two-pool correction: recurrent core and hot shared organs versus DRAM-read
   experts. Nominal L3 bandwidth is not effective LUT throughput.
3. [Frozen Phase-64 decisions](../../PHASE64_DECISIONS.md),
   [hardware budget](../../PHASE64_BUDGET.md) and
   [training plan](../../PHASE64_TRAINING_PLAN.md).
4. [Cross-program index](../RESEARCH_INDEX.md) and
   [donor no-duplication map](../donor_adaptation/audits/AXIS_COVERAGE_AND_NO_DUPLICATION_MAP.md)
   for inherited evidence, not automatic donor execution instructions.

## What stays fixed in the first comparison

Retain the native SSM/SWA and gated-dReLU ternary expert recipe; do not
simultaneously add attention replacements, new activations, low-rank paths or
new routing algorithms. Respect previously retained higher-precision organs.
Pin core dimensions, layer count, expert width, top-k, tokenizer, split,
sequence length, precision/layout, optimizer and training schedule. Existing
Phase-64 E32/E128/E256 rungs are candidates, not newly claimed results or a
final preregistration; confirm actual source/config/checkpoint availability
before selecting the smallest missing comparison.

The historical E32 training anchor uses L6, whereas the proposed code ladder
uses L8 and a different data/tokenizer setup. Do not compare those directly
as an E-only treatment: bind a matched baseline for whichever setup is chosen.

Fixed top-k fixes expert work only: a dense router's score matrix and
selection still grow with E. Report that increase, dispatch, cache misses,
stored RAM, optimizer state and expert exposure rather than assuming constant
total cost. Do not assume cross-token expert locality: the earlier finding
was approximately independent routing.

## Evidence required to decide

| Axis | Record | What must not count as success |
|---|---|---|
| Quality | Held-out BPB on identical text, task/rollout checks, uncertainty and seed variation | Training loss, synthetic speed, more parameters alone |
| Routing | Per-layer loads, dead experts, load imbalance, entropy, top-k margins and stability; actual exposure per expert | Balanced traffic without useful quality |
| Useful capacity | Quality-versus-E at fixed active expert work; separately scoped ablations if needed to establish expert contribution | Cloned unused experts or counting reused weights multiple times |
| Execution | Resident working set, streamed bytes/token, router/selection and expert times, total decode latency, peak RAM; correct C export | A router microbenchmark presented as end-to-end rate |
| Training cost | Tokens, updates, exposure/expert, wall time and resource use | Attributing undertraining from lower exposure solely to architecture |

The first curve uses equal training tokens and the same recipe to test scaling
under a fixed training budget. It does not estimate each size's fully trained
ceiling. If exposure limits the larger arm, label that uncertainty explicitly;
an exposure-matched extension changes the training budget and needs its own
decision. Do not automatically scale training tenfold to rescue a failed arm.

Before any new run, freeze the selected E values, numerical quality/routing/
latency gates, seeds, training cap and stopping rules in one bounded protocol.
Choose limits from the intended use and prior measurement variability, not
after seeing the scaling results. Validate instrumentation for both healthy
and collapsed routing; distinguish a control failure from a scientific FAIL.
The broad 50 tok/s target remains relevant, but high speed at pilot scale is
not evidence of 10B/100B quality or throughput. No 100B extrapolation from a
single small-scale pass.

## Current queue

| Step | Status | Next deliverable / boundary |
|---|---|---|
| Save donor work and suspend old next steps | DONE | Donor checkpoint `90bf966`; no scientific normalization run |
| Consolidate inherited evidence | DOCUMENTED | `PRIOR_EVIDENCE.md`; distinguish unavailable artifacts from failed experiments |
| Bind reusable native training/export assets | DONE | [NES-00](NES_00_ASSET_AND_DISPATCH_20260925.md): hashes, config, E128 apparatus and voids |
| Freeze one scaling protocol | DONE | [NES-01](E128_EQUAL_TOKEN_PROTOCOL_20260925.md): L6 E32/E128, gates and 10-hour cap |
| Train and measure | RUNNING | Local RTX 3060 E128; then held-out, generation, C parity/cost on both arms. No T4 run planned |
| Decide whether to scale further or change routing | NOT STARTED | Require the joint quality/routing/cost result; change one coordinate on failure |

## Experiment register and exact resumption point

| ID | State | Record and raw evidence |
|---|---|---|
| NES-00 | APPARATUS PASS; two smoke voids retained | [Asset/dispatch record](NES_00_ASSET_AND_DISPATCH_20260925.md), [E128 RTX 3060 JSON](dispatch_probe_e128_fullstep_rtx3060_20260925.json) |
| NES-01 | RUNNING from `44c7bb1` | [Equal-token protocol](E128_EQUAL_TOKEN_PROTOCOL_20260925.md), raw stdout/stderr/status above |

Resume by inspecting `nes01_status.json`, the log tail, and the final or
optimizer checkpoint. If the process is gone and the final checkpoint is
absent, use NES-01's same command to resume from the rotating state after
checking no other GPU job is active. Do not treat NES-00 smoke BPB as an E128
result. Preserve the unrelated
working-tree edits in `RESEARCH_INDEX.md` and the donor density script.

## Record-keeping rule

This file is the current-state entry point. Keep its queue short and replace
superseded status, rather than appending long chains of "sole next action".
Each experiment gets one concise record containing question, exact config and
code/checkpoint/data identities, comparison, budget, result, limitations and
decision, with links to raw artifacts. Add one row here and the relevant
prior-evidence/no-duplication pointer. Do not duplicate full results into all
indexes. Global `RESEARCH_INDEX.md` and the strategic roadmap only point here.

Retain FAIL and VOID results without changing historical gates. Repeat only
for a stated missing uncertainty estimate, replication requirement, changed
coordinate or apparatus repair, and state the reason before execution.
All documentation is in English. No model/assistant signatures or trailers.
