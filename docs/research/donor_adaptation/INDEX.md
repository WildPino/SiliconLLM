# Donor Adaptation — what has been tried, what it cost, where to look

**The goal:** run somebody else's pretrained LLM on our architecture (`engine.c`), target **~10B at
50 tok/s** (good) / **100 tok/s** (excellent).
**Last updated: 2026-09-04 (T2 closed, LUT kernel built and characterised).**

This is the map. Every row names the artefact that holds the detail; nothing here is a claim that
is not written up somewhere with its controls and its pre-registration.

---

## 0. Where the goal actually stands

| | status |
|---|---|
| **A pretrained donor executes on our runtime** | ✅ **YES** — Qwen2.5-0.5B, parity vs PyTorch `rel l2 2.8e-06`, top-1 `1.0000` |
| **At the target speed** | ❌ **48.3 tok/s at 0.5B** (3 reps, ±0.05, idle machine) = **25.1 G-weights/s delivered**, and 0.5B is 20× smaller than the target |
| **At usable quality** | ❌ **NO**, but the number moved: FFN conversion **+3.309 → +1.260 BPB** (T2), still 252 σ_seed |
| **The binding constraint** | **still quality — but it is the RULE, not the format** (T2, `RULE-HELPS`) |

**The one-line state:** the road exists end to end — safetensors → export → ternary runtime →
generated tokens, with a parity gate at the seam. T2 has now shown the damage at the far end was
**62% a bad map into the format**, not the format itself, and removed that much of it on CPU with
no gradients. What remains is a real quality gap and a 20× speed gap.

**The two numbers that price everything else** (`SPEED_LEDGER.md` **§11**, which supersedes §10):
**25.1 G-weights/s packed, 26.3 with the LUT kernel built** → a 10B donor needs **≤495 M active
weights/token** for 50 tok/s (4.9%) and ≤231 M for 100 (2.3%).
And **the output head alone exceeds that budget on four of twelve donors** — 126% on Qwen3-8B,
27% on Mistral-7B of the same width. Its size is set by the **tokenizer**, not the model (§7).

## 1. The runtime (this is the deliverable)

| what | where |
|---|---|
| Export a donor to a flat binary (fp32 / ternary / packed; optional ternary head) | `benchmarks/donor_adaptation/engine/qwen_export.py` |
| The runtime: RMSNorm, GQA + RoPE + KV cache, SwiGLU, ternary matvec, packed `pshufb` path | `benchmarks/donor_adaptation/engine/donor_engine.c` |
| Parity gate vs PyTorch on identical weights | `benchmarks/donor_adaptation/engine/parity_gate.py` |
| Report: profile, the two optimisations, honest position vs the goal | `probes/R1_DONOR_RUNTIME.md` |

**Measured, Qwen2.5-0.5B, 3600X:** 12.45 tok/s (t1) → **48.3 tok/s (t6, packed)**, three reps
within ±0.05 on an idle machine; **51.0 with `--lut`**. Trajectory: 23.5 (fp32 head) → 38.0
(ternary head) → 48.3 (packed) → 51.0 (LUT). R1's "40-46" and `SPEED_LEDGER.md` §10's 36.1 were
both taken on a contended machine and do not reproduce; §11 records the re-measurement.

## 2. The speed side — what is priced and what is measured

| probe | question | answer | where |
|---|---|---|---|
| **Ledger** | what must a donor cost per token to hit 50/100 tok/s? | **≤495 M active weights/token** for 50 tok/s, ≤231 M for 100 — from the *measured* 26.3 G-weights/s, not a byte-rate ÷ bits-per-weight | `SPEED_LEDGER.md` §11 |
| **P2** | is the expert path bandwidth- or compute-bound? | **MIXED — ~60% arithmetic.** A denser pack buys ≤1.32× on the FFN, not 2.5× | `probes/P2_EXPERT_PATH_DECOMPOSITION.md` |
| **P3** | what do donor SHAPES cost on the engine's kernels? | 0.5B 81 / 1.5B 30.7 / 3B 15.1 tok/s (matvec only). The ledger was 18% conservative because it priced dense FFNs at a *gather* rate | `probes/P3_DONOR_SHAPE_ON_ENGINE.md` |
| **R1** | what does a real runtime cost? | 48.3 tok/s at 0.5B (re-measured, §11). **The packing bought nothing, exactly as P2 predicted — and neither did the LUT kernel** | `probes/R1_DONOR_RUNTIME.md` |

**Terms nobody had attacked before the ledger, now priced:** the **output head** (was 40.4% of every
token in fp32 — fixed; and see §7, it is the floor a 10B cannot get under), the **KV cache** (still
fp32, untouched), the **attention projections**.

### 2.1 The LUT kernel — `donor_engine.c --lut`, the lever P2 and R1 named

| what | status |
|---|---|
| tile-major layout, `pshufb` tables, grouped scales | **built.** A `--lut` model and a `--quant packed` model hold **bit-identical weights** — the copy is a pure transpose |
| kernel correctness | **bit-exact** vs a scalar-integer reference; two planted controls fire (`--selftest-lut` cases B and C) |
| numeric cost of the int8 activations it requires | **measured.** rel l2 `1.40e-01` per-vector → **`3.10e-02` at G=32 channels per scale** |
| why it costs that | activation crest factor `amax/rms` is 8.3 avg / 69.6 max, so a 63-step grid leaves ~4–8 usable levels. `--lut-diag` measures it on `x` alone, so the kernel is not implicated |
| **rate** | ⚠ **measured: 1.05×.** 48.13 → 51.05 tok/s; 25.1 → 26.3 G-weights/s. **Not the 1.83× the ledger assumed** (`SPEED_LEDGER.md` §11) |
| BPB through the runtime | ❌ not run |

Two predictions were written before their sweeps and **both were wrong in magnitude**: clipping the
grid to `k·rms` (predicted an interior optimum; it is 3× worse at every `k`) and per-group scales
(predicted <0.02 at G=128; it is 0.052). Recorded in commits `95b7fd3` and `e02285c`.

## 3. The quality side — every structural result, and the conversion that undercuts them all

| probe | what was tried | outcome | where |
|---|---|---|---|
| **D0 / D0c** | carve the FFN into co-activation experts | **+1.09 BPB at E=32; +0.706 at the finest legal E.** Granularity helps, but the random null helps *more* → the mechanism does not strengthen (PARTIAL) | `probes/D0_COACTIVATION.md`, `probes/D0C_GRANULARITY.md` |
| **D1** | magnitude / structured pruning | see report | `results/d1_pruning.json` |
| **D2 / D3** | basis rotation, low-rank | see reports | `results/d2_basis.json` |
| **D4** | solve for thin replacement weights (Hessian) | recovery 0.483 honest vs 0.859 leaked → in-sample optimism; budget never swept | `probes/D4_RECONSTRUCTION.md` |
| **S1** | which bar predicts BPB under sparsity | \|h\| is the bar; A≡D at every digit | `benchmarks/donor_adaptation/s1/` |
| **T1** | **ternarize the donor — the engine's own rule** | **+4.738 BPB = 948 σ_seed. CONVERSION-FAILS** (measurement stands; verdict superseded in scope by T2) | `probes/T1_DONOR_TERNARIZATION.md` |
| **T2** | was that the FORMAT or one naive RULE? | **`RULE-HELPS`. It was the RULE.** FFN +3.309 → **+1.260**, 62% removed with no training. **BitLinear158 is statistically indistinguishable from RANDOM SIGNS** (−0.064 ± 0.126) | `probes/T2_TERNARIZATION_RULE.md` |
| **T2b** | does the winning rule survive outside the FFN? | **`UNIFORM`, by 1.0% of its bar.** The runnable model (197 tensors, R3) costs **+2.708**, `1.584×` the FFN alone vs a 1.60 bar — ci95 `[1.514, 1.649]`, the bar is INSIDE it. Head ternarization **not** withdrawn: the head costs `+0.339` alone and **`−0.009 ± 0.020` on top of a ternary FFN+attention**. Per weight **attention is 4.98× the FFN** | `probes/T2B_ORGAN_COVERAGE.md` |

**T2's decomposition, paired between arms** (`probes/T2_TERNARIZATION_RULE.md` §4):

| what | Δ BPB | significant |
|---|---|---|
| searching the per-row scale at all | −0.686 | yes |
| **weighting that search by activation RMS** | **−0.914** | yes |
| GPTQ error compensation **on a well-placed grid** | −0.449 | yes |
| GPTQ error compensation **on the naive grid** | +0.223 | **no** |
| TWN instead of BitLinear158 | −0.225 | **no** |

> ⚠ **Every structural result above was measured on fp32 weights the engine cannot consume**, and
> none was ever composed with the conversion. "Does carving a *ternarized* donor cost the same as
> carving an fp32 one?" has never been asked — and it is now a cheaper question than it was, since
> the conversion it would have to compose with costs +1.260 rather than +3.309.

## 4. Open, in priority order

1. **Export with the winning rule and measure BPB THROUGH `donor_engine.c`.** Every quality number
   this programme owns is a PyTorch number about a model the engine executes. `--bpb` exists and
   has never been run at scale. This closes the loop.
2. **D4b** — the calibration budget. Promoted from bookkeeping: T2's two best arms are both
   calibration-driven, so every one of their numbers is a **floor**.
3. **`bench_matrix.sh`** — `--fuse` × `OMP_WAIT_POLICY`, crossed, 3 reps, idle machine only.
   `--fuse` is built and verified bit-identical but **never timed**. This is the outstanding test of
   the 32 µs/call hypothesis and the 1.37× ceiling (`SPEED_LEDGER.md` §11.4).
4. **T3 — rotate the basis before ternarizing.** Pre-registered (`briefs/BRIEF_T3_ROTATION.md`),
   runner not written. D2's kurtosis table is already on disk and is its enabling measurement.
5. **Healing** (QAT / layer-wise distillation) — still on the critical path per T2 §7. It now
   starts from +1.260 (FFN, R5) / **+2.708 (whole runnable model, R3, T2b §3)** rather than +3.309,
   and T2b §6 says where to aim it: **attention, 4.98× the FFN's damage per weight at 10% of the
   weights**.
6. **S1's scale arm** — blocked on the fp16 NaN (`eager` attention overflows QK^T; diagnosed, §5).
   Every sparsity result this programme owns is measured at one size.
7. An already-MoE donor, and **a donor with a small vocabulary** (§7).

## 5. Bugs found in our own instruments (all fixed, all with controls added)

| bug | how it presented | fix |
|---|---|---|
| **The A1.2 gate reported PASS over an all-NaN run** | `max(0.0, nan) == 0.0` in Python swallowed all 51 comparisons | hard-fail on any non-finite, minimum comparison count, and **7 planted self-test cases run before any model loads** — the old code fails 3 of them |
| **`l1_keep_count` turned a NaN into a plausible measurement** | `clamp_(1, F)` mapped a NaN row to "keep 1 neuron" → achieved sparsity `8959/8960` | raises `FloatingPointError` on non-finite input |
| **fp16 NaN blamed on the GPU** | my own diagnostic did not pass `attn_implementation` and tested SDPA, not the `eager` path the probe uses | reproduced on CPU with `--attn eager`; cause is HF eager computing QK^T in fp16 (**274,672 vs the 65,504 limit**) before dividing by √head_dim |
| **T1's planted control was mis-specified** | required random signs ≫ ternarization; they are only 1.21× apart, so it returned VOID on sound numbers | identity-substitution control added (bit-exact). **T2 §4 then showed the premise itself was false**: random signs are not ≫ the treatment, they are indistinguishable from it |
| **`--calib-seqs` defaulted to 8 while T2 measures at 32** | exporting `--rule R3` would have built a model on a quarter of the calibration budget that produced the number, and the BPB gap would have read as the runtime disagreeing with PyTorch | default → 32, stderr warning otherwise, and the sidecar records the budget and the organ lists |
| **the LUT diagnostic reported whole-vector crest while groups were active** | it kept calling the derived figure "effective levels" when the grid was per-group, i.e. a plausible number describing the wrong thing | crest computed in-group, both labels corrected, header states which scale is in force |

## 6. Working rules this programme has paid for

- **A negative is only as strong as the sweep behind the setting it was measured at.** (D0c, D4b, T2)
- **A control must itself be shown to fire.** A gate never seen to trip is decoration. (§5, row 1)
- **Profile before optimising.** The head was 40% of every token and nobody had looked. (R1)
- **When reproducing a failure, reproduce the CONFIGURATION, not just the model.** A different
  library default invalidated a conclusion. (§5, row 3)
- **Byte-rate ÷ bytes-per-weight is only valid where the path is bandwidth-bound.** (P2 → R1)
- **A Δ against a baseline cannot support a claim about one arm versus another.** The arms are
  correlated across sequences; the contrast has to be bootstrapped paired. Doing it moved two of
  T2's apparent results — TWN and GPTQ-on-the-naive-grid — from "worse/better" to *not
  distinguishable*. (T2 §4)
- **When an instrument returns an impossible ordering, test the instrument before the finding.**
  GPTQ scoring below its own starting point is not physically possible; two controls showed the
  code was right and the objective was wrong. (T2 §5)
- **Quantization damage does not add across organs.** The head costs `+0.339` alone and
  `−0.009 ± 0.020` on top of a ternary FFN+attention; the three single-organ arms sum to `+3.184`
  where the combination measures `+2.708`. A per-organ cost is only a cost *in the company it was
  measured in*. (T2b §4)
- **A pre-registered threshold needs its own interval before the label is read as settled.** T2b
  passed its 1.60 bar at 1.584 — but the ratio's ci95 is `[1.514, 1.649]` and a third of the
  bootstrap lands on the other side. The label stands; the confidence in it does not. (T2b §5)

## 7. The head, the tokenizer, and the thing nobody priced

`donor_speed_budget.py` prices the output head against the measured 26.3 G-weights/s of the
built LUT kernel (`SPEED_LEDGER.md` §11). The head is a dense GEMV of `D × V` touched on **every** token, and **no** MoE, carve,
sparsity or reconstruction result in this programme touches it.

| donor | D | V | head | % of the 495 M budget for 10B @ 50 tok/s |
|---|---|---|---|---|
| Qwen3-8B | 4096 | 151,936 | 622 M | **126%** |
| openai/gpt-oss-20b | 2880 | 201,088 | 579 M | 117% |
| Qwen2.5-Coder-7B | 3584 | 152,064 | 545 M | 110% |
| nvidia/Nemotron-H-8B | 4096 | 131,072 | 537 M | 108% |
| mistralai/Mistral-7B-v0.3 | 4096 | 32,768 | 134 M | **27%** |
| microsoft/Phi-3-mini | 3072 | 32,064 | 98 M | 20% |

**Four of twelve donors on this disk have an output head larger than the entire 50 tok/s budget.**
Qwen3-8B cannot reach 50 tok/s on this machine even if every other weight in it were free.

**Qwen3-8B and Mistral-7B are the same width.** Their heads differ by 4.6× entirely because of
vocabulary size. Every donor-adaptation probe this programme owns was measured on Qwen, which has
the worst head-to-body ratio on the disk. **Vocabulary is a speed variable, it is chosen rather
than earned, and it has never been treated as one.**

Two measured facts sharpen it: the head was 40.4% of every token in fp32 before R1 ternarized it
(`probes/R1_DONOR_RUNTIME.md` §3), and it is the one organ that pays **nothing** for int8
activations on the LUT path — 0.13997 → 0.14084 when excluded — because its input is the final
RMSNorm output (`donor_engine.c --lut-no-head`, commit `95b7fd3`).
