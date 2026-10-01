# Native expert scaling: research control index

**Date:** 1 October 2026. **Branch:** `research/native-expert-scaling`.
**Status:** active research; no artifact meets all final requirements.

## Goal and constraints

Transfer a pretrained LLM into a compact reusable core plus selectively
consulted conditional capacity, execute it in `benchmarks/phase60/engine.c`,
retain useful donor-relative held-out/generative/task quality and measure
>=50 accepted batch-1 tokens/s on the same artifact. Demonstrate which
steps transfer across families/scales, including a real approximately-10B
case; 100B when resources permit. User priority: n can grow with RAM,
without proportional active cost or loss from the more complex routing.

Do not substitute synthetic pools, copied experts, component rate or a
generic donor runtime for transferred capacity. Count resident capacity,
active bytes, route cost and useful distinct functions separately. Do not
assume cache residency, ideal DRAM bandwidth or composition quality.
Freeze decisions before observing results. T4 requires prior explanation
of reason, budget and stop. None has been used.

Procedure and prerequisites: [METHOD.md](METHOD.md).
Initial native inventory: [PRIOR_EVIDENCE.md](PRIOR_EVIDENCE.md).
Detailed experiment register and historical decisions through METH-216:
[archived index](HISTORY_THROUGH_METH216_20261001.md). That snapshot preserves
all prior links/operational history; its old next actions are historical.

## 1. Geometry and useful expert-count scaling

- **Established small rung:** centered BF16 Qwen0.5B-Instruct E1280
  retains scoped fresh donor-relative quality ([METH-121/123](METH_121_123_ZERO_MEAN_CHILD_EXTERNAL_RESULT_20260928.md)).
  Its exact route beats shifted IDs on consumed sources
  ([METH-183](METH_183_E1280_CHILD_ROUTE_ALIGNMENT_RESULT_20260930.md)).
- **Useful tenfold growth unproven:** METH-175 E12800 training passes
  artifact/balance but METH-176 fresh gain fails; METH-178/179/201–203
  find little distinct child function. Do not repeat hash/shared-base B
  training without changing route/function coupling; details in archive.
- **Latest route result:** [METH-216](METH_216_LOCAL_CHILD_KEYS_RESULT_20260930.md)
  fits nine parent-local keys in existing child coordinates. All ChatML
  fit gates pass. Raw passes global load/content/coverage; hot-parent
  share fails in layers 12/16/17/20 (30.163% worst vs <=25%). Old/new
  reserved and source screens remain unopened. No B/native promotion.
- **CPU cost:** selected-factor synthetic component medians are below
  0.7 ms, but METH-198/199/200 fail repeatability for the tenfold pool
  ratio. No reliable large-n ratio or full accepted-token rate follows.
  Parent selection at larger count, real DRAM access, distinct learned
  capacity and complete native route/LUT cost remain required.

## 2. Pretrained-to-compact-core transfer

- The quality-valid BF16 E1280 native reference reaches 16.818 tok/s;
  selected-row head version reaches 20.791 (METH-127/129), below target.
- Q6 FFN corrections fail METH-187/190/191; grouped-Q8 scale refinement
  fails METH-195/196. Do not restart these unchanged recipes.
- [METH-211/212](METH_212_STORED_LOADER_RESULT_20260930.md) produces a real
  820.707 MB Q8-FFN/exact-BF16-tied-head core and loader: all 364 tensors
  supply all 290 config-only parameters, with exact consumed-source parity.
  [METH-214](METH_214_FRESH_PREDICTION_RESULT_20260930.md) rejects its fixed
  fresh quality: pooled donor-top1 loss 1.291 points vs <=1, despite BPB,
  category and finite K64 passes. Generation/task/blind stop; METH-215
  runner is unexecuted. METH-213/214 data are now consumed diagnostic data.
- GigaChat10B Q4 retains scoped quality but costs 1016 MB active/token.
  METH-180/181 global rank-192 variants fail energy proxies. Frozen
  donor-adaptation C fidelity work is reusable evidence, not a resumed
  generic port. StdMoE W4 quality/cost also fail; metadata-only Granite
  is not a transfer result. See METHOD for revisions/assets/variants.
- No compact full composition or multi-family/10B/100B method is validated.
  Core recovery needs a changed train-validated representation/adaptation
  plus new quality; eventual native precision/traffic must be measured.

## Current experiment and exact resume

[METH-217](METH_217_LOCAL_KEY_CONCENTRATION_RESULT_20261001.md) completes
with exact parity and metric reconciliation. Five raw parent failures are
soft-to-hard gaps: soft shares ~11.112%, hard shares 26.706–30.163%,
exact-state floors <=1.949%, rare exact ties. Full four-layer fit captures
are saved locally (795.342 MB, hash bound); no reserved/source inference.
Session 23384 completed. Next freeze direct hard-count calibration of
only those five raw bias vectors, retaining original keys/projection,
ChatML and all other raw biases, then all existing staged route gates.
No B training or native promotion from the diagnostic alone.

## Workspace rules

Preserve unrelated tracked edits in `docs/research/RESEARCH_INDEX.md` and
`benchmarks/donor_adaptation/density/build_document_holdout.py`; stage exact
research paths only. Documentation is English, user updates Italian.
Graphify is optional only for explicit graph work. Routine Git/search
Graphify hooks were removed with adjacent backups; retain Git LFS.
No project inference job preceded METH-217 at this session's process audit.
