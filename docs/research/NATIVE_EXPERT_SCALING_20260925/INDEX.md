# Native expert scaling: research control index

**Date:** 26 September 2026. **Branch:** `research/native-expert-scaling`.
**Status:** NES-01 E128 improves BPB but fails greedy generation; NES-02
finds a dense-router CPU scaling limit; NES-03 int8 shortlist preserves
tested routes and cuts E1280 router cost. No pretrained-to-native conversion
has passed joint quality and rate. METH-04 identifies a donor-bound low-bit
GigaChat target under the traffic preflight. METH-05 collected a source-ID
BF16 importance matrix but stopped on rare-expert exposure before conversion.
Goal remains open.
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

The end state is a reproducible method that transfers pretrained capability
to a conditional `engine.c` artifact, retains donor-relative quality and
runs at ≥50 accepted batch-1 tok/s on the **same** artifact and declared
hardware/context; 100 tok/s is a stretch goal. It must be tried across
families/scales, including order-10B and, where resources permit, order-100B.
Scaling experts is an enabling question, not a substitute for this transfer.
[METHOD.md](METHOD.md) tracks the provisional procedure and missing gates.

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

[NES-02](NES_02_CPU_EXPERT_COUNT_STRESS_20260925.md) tested E128 against a
throughput-only synthetic E1280 pool with fixed top-8. Median total CPU time
rose **1.637×**, router+selection **10.002×**, and selected-expert LUT time
**1.285×**. Parallel row scoring preserved bit-identical fp32 logits but
missed its predeclared speed gates; it remains opt-in. At the current tiny
geometry, 100B distinct weights would require E169,094 and an exhaustive
fp32 router would address about **1.039 GB/token**. At a favorable 40 GB/s,
that router payload alone needs **25.97 ms**, above the complete 20 ms budget
for 50 tok/s. This is arithmetic, not a 100B model measurement. It motivates
bounded candidate routing and a packed-only expert export.

[NES-03](NES_03_INT8_ROUTER_SHORTLIST_RESULT_20260925.md) uses an int8 router
sketch and exact fp32 rescore of 32 candidates. On frozen held-out inputs,
both E128 and synthetic E1280 C audits miss **zero** exact top-8 IDs across
8,192 positions × 6 layers; E128 BPB/top-1 and greedy output are unchanged.
The parallel E1280 router median is **116.8 µs/token** versus **536.1** in
NES-02; total **1,474.2** versus **1,880.4**. The frozen E128 cost gate passes
by only 0.12 µs/token and machine variation is comparable, so this is an
opt-in provisional router candidate. It does not repair E128 generation or
establish learned large-E quality.

[METH-04](METH_04_GIGACHAT_LOWBITS_PREFLIGHT_RESULT_20260925.md) now names
a concrete GigaChat BF16→mixed low-bit transformation: IQ2_XS for most MLA
and routed/shared experts, Q4_0 for block-incompatible MLA key tensors,
Q3_K head and Q4_K dense first FFN. The bound 414-tensor plan addresses
**534.025 MB/token**, 25.975 MB below the 560 MB design allotment, and
would require a donor-specific importance matrix for **254** tensors.
No transformed GGUF, paired quality, C support or accepted-token speed
exists. At 40 GB/s, payload alone prices to 13.351 ms/token; the margin is
still narrow.

[METH-05](METH_05_SOURCE_ID_IMATRIX_RESULT_20260926.md) repaired the
GigaChat calibration input path to use source-tokenizer IDs. A local
106×512-token BF16 importance matrix contains all 254 required tensors,
but layer 19 expert 29 received 16 activations and layer 20 expert 51
received 39, below the frozen 64-count floor. The 20-minute run used about
21.25 GB resident at the observed peak. Quantization was stopped before
writing weights. A new non-held-out calibration extension and route-coverage
audit are the immediate donor-transfer decision; quality remains unmeasured
for the proposed format.

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
| NES-02 | COST STRESS: dense router 10× cost; parallel speed gates fail | [result and shape ledger](NES_02_CPU_EXPERT_COUNT_STRESS_20260925.md), [serial measurements](nes02_capacity_stress_baseline_20260925.json), [parallel measurements](nes02_parallel_router_result_20260925.json); E1280 is synthetic, no quality inference |
| NES-03 | PROVISIONAL ROUTER COST PASS: E128 quality nonregression; E1280 synthetic | [result](NES_03_INT8_ROUTER_SHORTLIST_RESULT_20260925.md), [frozen protocol](NES_03_INT8_ROUTER_SHORTLIST_PROTOCOL_20260925.md), [probe](nes03_int8_router_probe_20260925.json), [raw result ledger](nes03_int8_router_result_20260925.json) |
| METH-00 | ARITHMETIC PREFLIGHT | [GigaChat active-organ traffic](METH_00_GIGACHAT_COST_PREFLIGHT_20260925.md); no decoder or quality measurement |
| METH-01 | HEADER-DERIVED PAYLOAD | [actual Q4_K_M organ ledger](METH_01_GIGACHAT_Q4_ACTIVE_LEDGER_20260925.md), [raw JSON](meth01_gigachat_q4_active_ledger.json); no decoder or quality measurement |
| METH-02 | METADATA SCREEN | [Granite H Tiny](METH_02_GRANITE_H_TINY_METADATA_SCREEN_20260925.md): recurrent/sparse candidate; no local weights or quality/rate result |
| METH-03 | HEADER-DERIVED PAYLOAD | [official Granite Q4 organ ledger](METH_03_GRANITE_Q4_ACTIVE_LEDGER_20260925.md), [raw JSON](meth03_granite_q4_active_ledger.json); 8 MB verified Range prefix, no full local weights or quality/rate result |
| METH-04 | DONOR TRANSFORMATION PREFLIGHT; quantizer dry run matches map; no weights changed | [result](METH_04_GIGACHAT_LOWBITS_PREFLIGHT_RESULT_20260925.md), [frozen map/gate](METH_04_GIGACHAT_LOWBITS_PROTOCOL_20260925.md), [414 tensor overrides](meth04_gigachat_tensor_types.txt), [byte ledger](meth04_gigachat_lowbit_preflight.json), [quantizer log](meth04_gigachat_quantize_dryrun.log) |
| METH-05 | SOURCE-ID BF16 IMATRIX; STOP on rare expert exposure | [result](METH_05_SOURCE_ID_IMATRIX_RESULT_20260926.md), [frozen protocol](METH_05_SOURCE_ID_IMATRIX_PROTOCOL_20260926.md), [machine audit](meth05_bf16_106chunks_audit.json); no converted weights or quality result |

NES-01 ran locally **2026-09-25 11:28–18:09 UTC** and exited 0 at step 4000.
Final checkpoint: `results/native_expert_scaling/nes01_e128_final.pt`, SHA-256
`58c84a8730b03c37e98481b8508f5cc84e15757de5513c3e6354f09b975cd462`.
The optimizer/RNG resume state remains for audit. No T4 work is running or
scheduled.

## Decisions and next exact action

- **Open:** cause/remedy of the E128 free-generation loops; bounded-candidate
  router quality and CPU cost at large E; trained quality at that scale; how to
  reduce a pretrained donor's active traffic while preserving quality;
  pretrained-to-native export/fidelity; generality across families and scales.
- **Closed within their scope:** H5's frozen H4+H2I assembly is adverse;
  STRAT-03's tested shared/router geometry fails. Neither rejects fresh joint
  training. The old donor parity queue is not an automatic next step.
- **CPU design state:** NES-03 established a 32-candidate int8 path on this
  pilot; the full fp32 router and reference expert copies remain resident.
  A packed-only export, learned large-E routing/quality and accepted-token
  rate remain open. Do not spend the next cell only tuning this pilot router.
- **Next quality design:** choose one changed variable to address E128 greedy
  loops (for example exposure or route training), estimate local training cost
  and freeze BPB, repetition and routing gates before a new run. Do not
  automatically extend E256 on the BPB gain alone.
- **Next exact action: pretrained transfer.** METH-05 fixed source-tokenizer
  calibration, then stopped because two rare GigaChat experts remained below
  the frozen 64-exposure floor after all 106 chunks. Bind distinct
  non-held-out calibration from local Russian/Ukrainian and other source
  files; prove no source-content overlap with the 96-document heldout,
  freeze token IDs and cost, and measure whether the rare experts receive
  enough genuinely new contexts. Do not repeat the same IDs or lower the
  floor retroactively. Only then quantize the BF16 source, verify actual
  tensor types/bytes and run the frozen paired BPB, PIQA and rollout gates.
  Compare this costed route with the Qwen2.5-1.5B dense-source path. The old
  donor port is not an automatic next step. No T4 job is planned.

Preserve unrelated working-tree changes in `docs/research/RESEARCH_INDEX.md`
and `benchmarks/donor_adaptation/density/build_document_holdout.py`. Update
this index by replacing current state, and put detailed evidence in the
experiment record. Documentation is in English; no model/assistant signatures.
