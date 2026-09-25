# Native expert scaling: research control index

**Date:** 25 September 2026. **Branch:** `research/native-expert-scaling`.
**Status:** NES-01 E128 trained and measured; BPB/routing/pilot CPU gates pass,
but the predeclared greedy-generation gate fails. Goal remains open.
Fork point: donor pause checkpoint `90bf966`.

## Goal and current decision

Test the architecture thesis of `benchmarks/phase60/engine.c`: whether more
**independently learned experts** improve useful quality while a compact core,
fixed top-k routing, and affordable total work/DRAM traffic keep per-token cost
contained. The user clarified that expert count `E` should eventually scale
with available RAM, potentially by about 10× between 10B and 100B variants;
the CPU LUT/router and model quality must both remain viable as choice grows.
[NES-01](NES_01_E128_RESULT_20260925.md) isolates E32→E128
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

[NES-01](NES_01_E128_RESULT_20260925.md) trained 128 distinct experts for
32.768M token positions. E128 BPB **0.842529** beats E32 **0.858854**, with
no dead experts and C pilot latency **1.198×** E32. Yet greedy repetition
fails the frozen gate: **10/16 E128** versus **3/16 E32** continuations loop
in fp32; LUT results are 9/16 versus 4/16. The failure persists despite
healthy aggregate routing on free generation. Median CPU router+selection
grows **14.4→52.1 µs/token** and selected-expert LUT path **456.4→588.0**;
the latter shows fixed top-k is insufficient to assume constant CPU cost.
This is a single-seed small-model result; do not escalate E256 or treat it
as pretrained transfer.

For pretrained transfer, the local GigaChat base Q4 passes fresh paired BPB,
PIQA and document rollout against BF16. The actual mixed-format GGUF header
prices **1,016 MB of active payload/token**; at 50 tok/s that requires
50.8 GB/s of compressed weight payload before other work. This is calculated,
not a measured DRAM or C rate. The official Granite H Tiny Q4 tensor header
prices **913 MB/token** by the same addressed-payload method; its full weights,
quality and C behavior have not been tested locally. Both direct Q4 paths
exceed the 560 MB/14 ms streaming design allotment at the 40 GB/s yardstick.
GigaChat's C fidelity port is partial and no
quality-plus-≥50 C artifact exists. Qwen2.5-1.5B H1 shows that training a
carve helps; frozen H4+H2I composition fails. See [METHOD.md](METHOD.md) for
links and scope. No step currently transfers donor knowledge into the native
SSM/SWA target and passes joint quality/rate.

## Running cell and experiment register

| ID | State | Record / raw evidence |
|---|---|---|
| NES-00 | APPARATUS PASS; invalid smokes retained | [asset/dispatch record](NES_00_ASSET_AND_DISPATCH_20260925.md), [RTX 3060 JSON](dispatch_probe_e128_fullstep_rtx3060_20260925.json) |
| NES-01 | JOINT FAIL: generation; other pilot gates pass | [result](NES_01_E128_RESULT_20260925.md), [frozen protocol](E128_EQUAL_TOKEN_PROTOCOL_20260925.md), [decoded fp32 samples](nes01_greedy_fp32_samples_20260925.json), [CPU timing](nes01_cpu_timing_20260925.json) |
| METH-00 | ARITHMETIC PREFLIGHT | [GigaChat active-organ traffic](METH_00_GIGACHAT_COST_PREFLIGHT_20260925.md); no decoder or quality measurement |
| METH-01 | HEADER-DERIVED PAYLOAD | [actual Q4_K_M organ ledger](METH_01_GIGACHAT_Q4_ACTIVE_LEDGER_20260925.md), [raw JSON](meth01_gigachat_q4_active_ledger.json); no decoder or quality measurement |
| METH-02 | METADATA SCREEN | [Granite H Tiny](METH_02_GRANITE_H_TINY_METADATA_SCREEN_20260925.md): recurrent/sparse candidate; no local weights or quality/rate result |
| METH-03 | HEADER-DERIVED PAYLOAD | [official Granite Q4 organ ledger](METH_03_GRANITE_Q4_ACTIVE_LEDGER_20260925.md), [raw JSON](meth03_granite_q4_active_ledger.json); 8 MB verified Range prefix, no full local weights or quality/rate result |

NES-01 ran locally **2026-09-25 11:28–18:09 UTC** and exited 0 at step 4000.
Final checkpoint: `results/native_expert_scaling/nes01_e128_final.pt`, SHA-256
`58c84a8730b03c37e98481b8508f5cc84e15757de5513c3e6354f09b975cd462`.
The optimizer/RNG resume state remains for audit. No T4 work is running or
scheduled.

## Decisions and next exact action

- **Open:** cause/remedy of the E128 free-generation loops; CPU router/LUT
  scaling when E grows another 10×; trained quality at that scale; how to
  reduce a pretrained donor's active traffic while preserving quality;
  pretrained-to-native export/fidelity; generality across families and scales.
- **Closed within their scope:** H5's frozen H4+H2I assembly is adverse;
  STRAT-03's tested shared/router geometry fails. Neither rejects fresh joint
  training. The old donor parity queue is not an automatic next step.
- **Next CPU stress:** at E≈1,280, measure dense router, top-k selection and
  selected-expert LUT separately on the same Ryzen 5 3600X; hold top-8 and
  active expert geometry fixed. Synthetic enlarged pools are throughput-only,
  never quality evidence. Record RAM footprint and cache behavior rather
  than assume cache residency. E128 timing proves only the L6 pilot.
- **Next quality design:** choose one changed variable to address E128 greedy
  loops (for example exposure or route training), estimate local training cost
  and freeze BPB, repetition and routing gates before a new run. Do not
  automatically extend E256 on the BPB gain alone.
- **Pretrained transfer:** METH-01 and METH-03 close direct-Q4 cost preflights
  for GigaChat and Granite. Specify a target transformation with active-byte
  margin, step-zero control, paired donor quality gate and resource estimate
  before acquiring another donor or budgeting adaptation. No T4 job is planned.

Preserve unrelated working-tree changes in `docs/research/RESEARCH_INDEX.md`
and `benchmarks/donor_adaptation/density/build_document_holdout.py`. Update
this index by replacing current state, and put detailed evidence in the
experiment record. Documentation is in English; no model/assistant signatures.
