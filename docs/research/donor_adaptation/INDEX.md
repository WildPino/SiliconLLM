# Donor Adaptation — what has been tried, what it cost, where to look

**The goal:** run somebody else's pretrained LLM on our architecture (`engine.c`), target **~10B at
50 tok/s** (good) / **100 tok/s** (excellent).
**Last updated: 2026-09-04.**

This is the map. Every row names the artefact that holds the detail; nothing here is a claim that
is not written up somewhere with its controls and its pre-registration.

---

## 0. Where the goal actually stands

| | status |
|---|---|
| **A pretrained donor executes on our runtime** | ✅ **YES** — Qwen2.5-0.5B, parity vs PyTorch `rel l2 2.8e-06`, top-1 `1.0000` |
| **At the target speed** | ❌ **40–46 tok/s at 0.5B**, and 0.5B is 20× smaller than the target |
| **At usable quality** | ❌ **NO** — the conversion costs **+4.738 BPB** on a 0.7676 baseline |
| **The binding constraint** | **quality, not speed** |

**The one-line state:** the road exists end to end — safetensors → export → ternary runtime →
generated tokens, with a parity gate at the seam — and the model that comes out the far end is
destroyed by the numeric conversion. Everything else is downstream of fixing that.

## 1. The runtime (this is the deliverable)

| what | where |
|---|---|
| Export a donor to a flat binary (fp32 / ternary / packed; optional ternary head) | `benchmarks/donor_adaptation/engine/qwen_export.py` |
| The runtime: RMSNorm, GQA + RoPE + KV cache, SwiGLU, ternary matvec, packed `pshufb` path | `benchmarks/donor_adaptation/engine/donor_engine.c` |
| Parity gate vs PyTorch on identical weights | `benchmarks/donor_adaptation/engine/parity_gate.py` |
| Report: profile, the two optimisations, honest position vs the goal | `probes/R1_DONOR_RUNTIME.md` |

**Measured, Qwen2.5-0.5B, 3600X:** 12.45 tok/s (t1) → **40–46 tok/s (t6)**, stable ctx 20→200.
Trajectory: 23.5 (fp32 head) → 38.0 (ternary head) → 40–46 (packed).

## 2. The speed side — what is priced and what is measured

| probe | question | answer | where |
|---|---|---|---|
| **Ledger** | what must a donor cost per token to hit 50/100 tok/s? | **≤680 M active weights/token** for 50 tok/s at the kernel-pure ceiling; 168 M at today's integrated rate | `SPEED_LEDGER.md` |
| **P2** | is the expert path bandwidth- or compute-bound? | **MIXED — ~60% arithmetic.** A denser pack buys ≤1.32× on the FFN, not 2.5× | `probes/P2_EXPERT_PATH_DECOMPOSITION.md` |
| **P3** | what do donor SHAPES cost on the engine's kernels? | 0.5B 81 / 1.5B 30.7 / 3B 15.1 tok/s (matvec only). The ledger was 18% conservative because it priced dense FFNs at a *gather* rate | `probes/P3_DONOR_SHAPE_ON_ENGINE.md` |
| **R1** | what does a real runtime cost? | 40–46 tok/s at 0.5B. **The packing bought nothing, exactly as P2 predicted** | `probes/R1_DONOR_RUNTIME.md` |

**Terms nobody had attacked before the ledger, now priced:** the **output head** (was 40.4% of every
token in fp32 — fixed), the **KV cache** (still fp32, untouched), the **attention projections**.

## 3. The quality side — every structural result, and the conversion that undercuts them all

| probe | what was tried | outcome | where |
|---|---|---|---|
| **D0 / D0c** | carve the FFN into co-activation experts | **+1.09 BPB at E=32; +0.706 at the finest legal E.** Granularity helps, but the random null helps *more* → the mechanism does not strengthen (PARTIAL) | `probes/D0_COACTIVATION.md`, `probes/D0C_GRANULARITY.md` |
| **D1** | magnitude / structured pruning | see report | `results/d1_pruning.json` |
| **D2 / D3** | basis rotation, low-rank | see reports | `results/d2_basis.json` |
| **D4** | solve for thin replacement weights (Hessian) | recovery 0.483 honest vs 0.859 leaked → in-sample optimism; budget never swept | `probes/D4_RECONSTRUCTION.md` |
| **S1** | which bar predicts BPB under sparsity | \|h\| is the bar; A≡D at every digit | `benchmarks/donor_adaptation/s1/` |
| **T1** | **ternarize the donor — the engine's own rule** | **+4.738 BPB = 948 σ_seed. CONVERSION-FAILS** | `probes/T1_DONOR_TERNARIZATION.md` |
| **T2** | was that the FORMAT or one naive RULE? | pre-registered; TWN / α-search / activation-weighted / **GPTQ** | `briefs/BRIEF_T2_TERNARIZATION_RULE.md` |

> ⚠ **Every structural result above was measured on fp32 weights the engine cannot consume**, and
> none was ever composed with a conversion that costs +4.74 BPB on its own. "Does carving a
> *ternarized* donor cost the same as carving an fp32 one?" has never been asked.

## 4. Open, in priority order

1. **T2** — is the ternarization damage a rule artefact? CPU-only, no training. **Running.**
2. **Healing** (QAT / layer-wise distillation) if T2 says the format is the wall. This is training →
   GPU. No brief yet.
3. **The LUT kernel** in `donor_engine.c` — the only identified lever left on the FFN term (P2).
4. **S1's scale arm** — blocked on the fp16 NaN (`eager` attention overflows QK^T; diagnosed, see
   §5). Every sparsity result this programme owns is measured at one size.
5. **D4b** — D4's calibration budget, pre-registered and never run.
6. An already-MoE donor. `SPEED_LEDGER.md` §5 shows the arithmetic does not close for a *dense* 10B.

## 5. Bugs found in our own instruments (all fixed, all with controls added)

| bug | how it presented | fix |
|---|---|---|
| **The A1.2 gate reported PASS over an all-NaN run** | `max(0.0, nan) == 0.0` in Python swallowed all 51 comparisons | hard-fail on any non-finite, minimum comparison count, and **7 planted self-test cases run before any model loads** — the old code fails 3 of them |
| **`l1_keep_count` turned a NaN into a plausible measurement** | `clamp_(1, F)` mapped a NaN row to "keep 1 neuron" → achieved sparsity `8959/8960` | raises `FloatingPointError` on non-finite input |
| **fp16 NaN blamed on the GPU** | my own diagnostic did not pass `attn_implementation` and tested SDPA, not the `eager` path the probe uses | reproduced on CPU with `--attn eager`; cause is HF eager computing QK^T in fp16 (**274,672 vs the 65,504 limit**) before dividing by √head_dim |
| **T1's planted control was mis-specified** | required random signs ≫ ternarization; they are only 1.21× apart, so it returned VOID on sound numbers | identity-substitution control added (bit-exact), reasoning recorded rather than the label quietly overturned |

## 6. Working rules this programme has paid for

- **A negative is only as strong as the sweep behind the setting it was measured at.** (D0c, D4b, T2)
- **A control must itself be shown to fire.** A gate never seen to trip is decoration. (§5, row 1)
- **Profile before optimising.** The head was 40% of every token and nobody had looked. (R1)
- **When reproducing a failure, reproduce the CONFIGURATION, not just the model.** A different
  library default invalidated a conclusion. (§5, row 3)
- **Byte-rate ÷ bytes-per-weight is only valid where the path is bandwidth-bound.** (P2 → R1)
