# Canonical evaluation — the single source of public absolute BPB numbers

**Status: frozen 2026-07-03.** Every public *absolute* BPB number for the promoted checkpoints comes from
this one slice and one harness. Per-probe *deltas* remain valid on their original harnesses (same-slice,
same-dtype comparisons); this table exists because absolute BPB is slice- and dtype-sensitive (the val
slice choice alone moves absolute BPB by ~0.04 — 20× the typical ±0.002 gate width).

## The anti-Goodhart harness rule (non-negotiable)

A pre-registered gate is read with the **same harness that produced its anchor** — never across two
harnesses. Concretely: Phase 61's gate (BPB ≤ 0.8799 + 0.010 @4k matched) is read on the phase-61
training-script val, the harness of its anchor. The canonical harness below is the *public absolute*
number and anchors **future** gates only. Phase 61 closed (2026-07-03) with a within-script gate FAIL
(see "Phase 61" below); its checkpoints are **not** promoted to this table — folding the training-script
gate numbers into the canonical harness would break exactly the rule stated here.

## The canonical slice (frozen)

| field | value |
|---|---|
| token stream | `results/phase55/ids.u16` (BPE-1024, uint16) — md5 `d6682609e814eb41d8b2d0841636237b`, n = 32,723,845 |
| split | train `[0, int(n*0.9))` · val `[int(n*0.9), n)` (val offset 29,451,460) — the project-wide 90/10 split |
| slice | first **390 windows × 512 tokens** of val = **199,680 evaluated tokens** |
| metric | sum of CE bits over targets / sum of target byte-lengths (BPB) |
| harness | **fp32 weights, CPU, no autocast**, `torch.no_grad`; windows batched on the batch dim only (math identical) |
| tool | `python scripts/canonical_eval.py` → `results/canonical/canonical_bpb.json` (includes per-checkpoint md5) |

## Canonical table (2026-07-03)

| checkpoint | canonical BPB (fp32 CPU) | training-harness BPB | provenance |
|---|---|---|---|
| `archA_5m_fp32` | **0.8104** | 0.8104 | probe-1 fp32 arm (5M Arch-A, plain SiLU MLP, 10,785 steps) |
| `archA_5m_ternary` | **0.8382** | 0.8382 | probe-1 ternary arm (BitLinear158 MLP) |
| `sp_silu` | **0.8807** | 0.8807 | probe-2 SiLU (SwiGLU) arm, 4k matched |
| `sp_drelu` | **0.8813** | 0.8813 | probe-2 dReLU arm, 4k matched |
| `sp58_base` | **0.8799** | 0.8799 | phase-58 base = probe-4 arm A = engine E1–E3.5 target |
| `sp58_reg` | **0.8801** | 0.8801 | phase-58 +coherence arm |
| `moe_dense_big` | **0.8672** | 0.8674 | probe-4 arm B (dense-4096) |
| `moe_gran` | **0.8589** | 0.8589 | probe-4 arm C (E32×h128 top-8) = engine E4 target |
| `moe_coarse` | **0.8636** | 0.8637 | probe-4 arm D (E8×h512 top-2) |

**Readings.** (a) Eight of nine checkpoints match their training-harness value to the 4th decimal —
the training evals were effectively fp32-clean; the two visible gaps (`moe_dense_big` +1.3e-4,
`moe_coarse` +0.8e-4) are the bf16-autocast eval noise of the phase-59 script, now *quantified* and
bounded at ~1e-4 (an order of magnitude below every probe delta read from those runs — no verdict moves).
(b) Every probe-4 arm ordering is preserved on the canonical harness (C 0.8589 < D 0.8636 < B 0.8672 <
A 0.8799), including the non-pre-registered C < B observation (single-seed, 4k steps — its epistemic
status is unchanged by this re-eval).

## Phase 61 — the cross-script reproducibility datum (why single-seed deltas < 0.005 are noise)

Phase 61's A/B is read on its own training-script val (the anchor's harness), **not** the canonical table
above — deliberately, per the rule at the top. It handed the whole project one calibration number:

- **Cross-script reproducibility ≈ 0.004 BPB.** The phase-58 base recipe, refit under the phase-61 script,
  gives **0.8757** vs the **0.8799** anchor — identical recipe, different script, ~0.004 apart. Therefore
  **any single-seed delta below ~0.005 BPB is not resolvable**: e.g. probe-2's dReLU vs SiLU +0.0006 reads
  "zero within noise" (as already labeled). Valid comparisons are **within-script matched**; the absolute
  numbers are the canonical table above.
- The ternary-projection arms (arm A 0.8940, arm B 0.8972 vs gate 0.8899 = base 0.8757 + 0.010) are recorded
  in [`HANDOFF.md`](../HANDOFF.md) and [`ENGINE_PLAN.md`](ENGINE_PLAN.md); by measurement they confirm the
  mixed-precision recipe (ternary MLP only, fp32 backbone + projections). A 3-seed calibration (R1) is queued
  to measure σ_seed directly and retro-price every project delta.

## Notes

- Checkpoints not in the table (`archA_seq512`, intermediate/diagnostic saves) are not promoted.
- The `sp58_*` state dicts carry an auxiliary `pred.*` head from the phase-58 predictability probe;
  it is not part of the forward path (parity to 1e-7 with the anchor confirms this) and is skipped on load.
- Rerun a single row: `python scripts/canonical_eval.py --only <label>` (merges into the JSON).
