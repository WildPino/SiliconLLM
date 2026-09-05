# BRIEF E1 — Does the model the ENGINE executes score the BPB that PyTorch says it does?

**Status: PRE-REGISTERED. Written and pushed before the run, per standing practice.**
**Author: the Adapter / Principal. Date: 2026-09-05.**
**Depends on: `probes/R1_DONOR_RUNTIME.md`, `probes/T2_TERNARIZATION_RULE.md`,
`probes/T2B_ORGAN_COVERAGE.md`, `probes/T3_ROTATION.md`, `INDEX.md` §4 items 1 and 2.**

---

## 1. The gap this closes, stated exactly

**Every quality number this programme owns is a PyTorch number.** T1's `+4.738`, T2's `+1.260`,
T2b's `+2.708`, T3's `−0.220`, D0–D4, S1 — all of them are `common.bpb(model, ids, byts)` on a
`transformers` module holding simulated-ternary fp32 tensors.

**Every speed number this programme owns is an engine number**, and the engine consumes a
different artifact: a flat binary produced by `qwen_export.py`, read by `donor_engine.c`, executed
by hand-written kernels with their own RMSNorm, their own RoPE, their own KV cache and their own
softmax.

**The bridge between them is one parity gate on one short prompt** (`parity_gate.py`: `rel l2
2.8e-06`, top-1 `1.0000`). That gate compares *logits at a handful of positions*. It has never
been shown that the engine, over a held-out slice, produces the **BPB** that the probe pipeline
attributes to it — and BPB is the unit every gate in this programme is written in.

`donor_engine.c --bpb` was built for this and **has never been run at scale**. It is item 1 of
`INDEX.md` §4 and it has been item 1 since the list was written.

**Two facts sharpen it into a question with a real chance of failing:**

1. **No R3 model has ever been exported.** `D:/_ktmp/qwen05b_packed.bin`'s sidecar has no `rule`
   field at all — it predates `--rule`, so it is R0/BitLinear158, mean zero fraction `0.327`.
   Every speed number in `SPEED_LEDGER.md` was measured on a model whose quality T2 showed to be
   the *worst* of the five rules. The format is identical, so the speed numbers stand; but the
   artifact that would demonstrate the programme's best quality claim **does not exist on disk**.
2. **The exporter's R3 path has never had its output checked against the rule that produced the
   number.** `qwen_export.py` imports `t2_rules` rather than copying it (good), but importing the
   rule is not the same as writing the rule's output correctly into a packed binary with a
   per-row scale, a nibble layout and a tile-major permutation.

> **The question:** does `BPB(engine)` equal `BPB(PyTorch)` on the same weights and the same
> bytes — at fp32, and then at ternary?

## 2. Why the fp32 arm must come first, and must be separate

A disagreement at ternary has two possible causes that a single measurement cannot separate:
the **runtime** computes something different from `transformers` (norm epsilon, RoPE, GQA
broadcast, KV, the final norm, the head), or the **export** writes something different from what
`t2_rules` quantized.

So the arms are ordered so that each adds exactly one thing:

| arm | what changes vs the arm above | what it can prove |
|---|---|---|
| **F32** | — (fp32 export, no quantization anywhere) | the runtime's arithmetic and the harness's byte accounting agree with PyTorch |
| **TQ** | R3 applied to FFN + attention (196/168 tensors), packed | the export writes the rule correctly, and the ternary kernel computes it |
| **TQH** | `--head-ternary` on top | the whole runnable model, i.e. T2b's arm FAH |

**This is the same ordering `donor_engine.c`'s own header argues for** ("Building the ternary path
first would leave a bug in this file indistinguishable from the cost of the conversion").

## 3. Fixed before the run

- **Slice:** the shared `heldout` slice, 24×512, seed 1234,
  `ids_sha256 = a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`,
  **51,870 scored bytes**. Written to `ids.bin` as little-endian `int32`, `--seqlen 512`.
- **Scoring convention, identical on both sides by construction:** tokens `1..511` predicted from
  `0..510`; `BPB = total_nats / (ln 2 × total_scored_bytes)`. `common.bpb` sums
  `-log_softmax(logits[:, :-1]).gather(ids[:, 1:])`; `donor_engine.c` sums the same term over
  `t+1 < SL`. **The engine reports nats, not BPB** — the division is done in the runner, from the
  slice metadata, not from a number typed by hand.
- **Calibration for R3:** `calib` 32×512 seed 42424, `--calib-seqs 32` — T2's operating point.
  Any other value is a different experiment (`--calib-seqs` already warns).
- **Donors, both, because the two halves of this programme live on different models:**
  - **Qwen2.5-0.5B** (`D=896 F=4864 L=24 heads 14/2 hd=64`) — what the runtime is benchmarked on.
    All three arms, full 24×512.
  - **Qwen2.5-1.5B** rev `8faed761…` (`D=1536 F=8960 L=28 heads 12/2 hd=128`) — where every
    quality number lives, and a different GQA ratio and head_dim, so it exercises shapes the
    0.5B does not. Arms TQ and TQH full; **arm F32 on the first 4 sequences only**, pre-registered
    as a subset because at 6.2 GB of weight traffic per token the full slice is ~98 minutes and
    the arm is an *agreement* test, not a number this programme will quote.
- **Speed is not measured here and no timing from this probe may be quoted.** `--bpb` runs a
  full prefill per position on a machine that may be doing anything else; a contended timing is
  not a timing.

## 4. Pre-registered decision rule — fixed before any result

Let `Δ_arm = BPB(engine) − BPB(PyTorch)` on the same weights, same slice. σ_seed = 0.005.

**Gate A — the weights must be the same weights, checked before any BPB is compared.**
For each ternary arm, the runner re-reads the exported binary, unpacks it, and compares
`codes × scale` against the tensor `t2_rules.r3_actsearch` produced in memory, **tensor by
tensor**. Required: **bit-identical** (`np.array_equal` on the dequantized fp32). A BPB agreement
reached with different weights is a coincidence; a BPB disagreement with identical weights is a
runtime bug. **Without this check neither reading is available.**

**Gate B — the fp32 arm.** `|Δ_F32| ≤ 0.002` (0.4 σ_seed). The parity gate already puts the
logits at `rel l2 2.8e-06`; anything larger than 0.002 BPB is a systematic difference, not
accumulated fp noise.

| outcome | condition | what it means |
|---|---|---|
| **LOOP-CLOSED** | Gate A passes, `\|Δ_F32\| ≤ 0.002`, and `\|Δ_TQ\| ≤ 0.01` **and** `\|Δ_TQH\| ≤ 0.01` | the engine scores what PyTorch scores. **Every quality number in this programme becomes a statement about the deliverable**, and T2b's `+2.708` is the runtime's number, not a simulation's |
| **RUNTIME-DIVERGES** | Gate A and Gate B pass, some ternary `\|Δ\| > 0.01` | the fp32 path agrees and the ternary path does not, with the weights proven identical → the divergence is in the ternary/packed **kernel**, and it is a bug with a bounded search space. **Every ternary quality number is then a claim about PyTorch only** |
| **EXPORT-DIVERGES** | Gate A fails | the binary does not hold the weights the rule produced. The BPB comparison is not run at all |
| **HARNESS-DIVERGES** | Gate B fails | the runtime and the reference are not scoring the same thing even at fp32. Nothing about quantization can be read, and `parity_gate.py` is shown to be too weak a bridge |

`0.01` is the ternary bar because it is 2 σ_seed and it is 0.37% of the `+2.708` that T2b
measured — small enough that agreement means something, large enough that it is not testing
fp32-vs-fp32 accumulation order in a kernel that legitimately sums in a different order.

## 5. What this brief does NOT do

- It does **not** change `donor_engine.c` or the export format.
- It does **not** measure speed, and no number from it may enter `SPEED_LEDGER.md`.
- It does **not** test the RMSNorm fold (INDEX §4 item 2). The fold changes the *weights*; this
  brief tests that the engine agrees with PyTorch about *whatever weights it is given*. **The fold
  is the next brief and it depends on this one**: until the loop is closed, measuring the fold
  through the engine could not distinguish the fold's effect from a runtime disagreement.
- It does **not** sweep the calibration budget (D4b remains unrun; every R3 number is a floor).
- It does **not** touch `--lut`. The LUT path is a separate numeric object (`rel l2 3.10e-02` at
  G=32) and mixing it in would confound a correctness question with an approximation.

## 6. Cost

Export: 0.5B fp32 ~2 GB, packed ~0.8 GB; 1.5B fp32 ~6.2 GB, packed ~2.3 GB. Scoring: 12,288
forwards per full arm — roughly 15 min (0.5B fp32), 4 min (0.5B packed), 11 min (1.5B packed),
16 min (1.5B fp32 on the 4-sequence subset). Plus one PyTorch reference per arm (~2 min at 0.5B,
~2 min at 1.5B — these are the numbers T2b already has for the 1.5B ternary arms and they will be
**re-derived, not copied**, so the comparison is against a number produced in the same process).
**CPU only, no GPU, no gradients, under two hours.**

## 7. Reporting

`probes/E1_BPB_THROUGH_ENGINE.md`. Report, per donor and per arm: `BPB(engine)`, `BPB(PyTorch)`,
`Δ`, the per-sequence maximum `|Δ|` (a mean can hide a single divergent sequence), Gate A's
tensor count and worst mismatch, the §4 label verbatim, and — for the 1.5B ternary arms — the
comparison against T2b's standing `+2.716656` (arm FA) and `+2.708111` (arm FAH), which is a
**replication constant that names its arm, its organ set and its file**, per T3 §6.1.
