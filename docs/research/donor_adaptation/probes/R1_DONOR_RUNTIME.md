# R1 — The donor runtime: a pretrained model executing on our own architecture

**Status: WORKING AND VERIFIED. Speed 40–46 tok/s at 0.5B, t6. Quality: see §5, it is bad.**
**Date: 2026-09-04.**
**Code: `benchmarks/donor_adaptation/engine/{qwen_export.py, donor_engine.c, parity_gate.py}`.**
**Commits: `f73e026` (first run + parity), `81dbb3f` (head, packing, profile).**

> **Not audited.** Written by the figure that built it.

---

## 1. What did not exist before this

`engine.c` is an **SSM** engine. It has no attention organ, no RoPE, and no KV cache, so **no donor
could ever execute on it** — every donor-adaptation result on record (D0, D0c, D1, D2, D3, D4, S1)
was measured in PyTorch, about a runtime that could not run the thing being measured.

`donor_engine.c` is a transformer runtime in the same style and with the same numeric conventions:
ternary weight codes with one fp32 scale per output row, fp32 norms/biases/embeddings, fp32
activations. It implements RMSNorm, q/k/v/o projections with Qwen2's biases, RoPE in HF's
`rotate_half` convention, grouped-query attention with a KV cache, SwiGLU, and a tied-or-untied
output head.

## 2. The parity gate, which runs before any claim

`parity_gate.py` feeds the same tokens to the runtime and to PyTorch **holding the same weights**,
and compares logits. A runtime that is fast and wrong is worth nothing.

| configuration | worst relative l2 | top-1 agreement |
|---|---|---|
| ternary body, tied fp32 head | **2.223e-06** | **1.0000** |
| packed body, ternary head | **2.825e-06** | **1.0000** |

**Two design decisions that made this possible, and both are load-bearing:**

- **The exporter emits fp32 as well as ternary.** The fp32 mode exists so the runtime can be proved
  correct *before* quantization is in play. Building the ternary path first would leave a bug in the
  C indistinguishable from the cost of the conversion.
- **The exporter imports `ternarize()` from the T1 probe** rather than re-deriving the rule. One
  definition of the conversion, shared by the C export and the Python reference, so they cannot
  drift.

The loader asserts it consumed **exactly** the file's byte count, so a layout mismatch cannot pass
silently.

## 3. Profile first

`--profile` reports per-organ wall clock. The first breakdown (ternary body, fp32 tied head, t6):

| organ | ms/token | share |
|---|---|---|
| **head** | **17.422** | **40.4%** |
| ffn | 16.902 | 39.2% |
| qkv_proj | 6.323 | 14.7% |
| attention | 1.268 | 2.9% |
| o_proj | 1.120 | 2.6% |
| norm+glue | 0.105 | 0.2% |

**40% of every token was the output head, in fp32** — 545 MB moved per token at 31.3 GB/s, i.e. at
the machine's streaming rate, so it was doing nothing wrong; there was simply far too much of it.
`SPEED_LEDGER.md` §2 had flagged the head as a term **no probe had ever touched**. This is its bill.

## 4. Two optimisations, and only one worked

**4.1 Untie and ternarize the head — worked.** `--head-ternary`. The embedding table stays fp32
because it is a row *lookup* (3.5 KB/token, free to stream); the head is a dense GEMV over 151,936
rows. **23.5 → 38.0 tok/s**; head 17.42 → 4.65 ms.

**4.2 Pack two trits per byte — did essentially nothing, and that was predicted.** `--quant packed`
is engine.c's own base-3 g=2 format, 4 bits per weight instead of 8, decoded with the same `pshufb`
trick. Verified lossless twice: the round-trip is bit-identical in Python, and the two kernels agree
to **5.4e-07** against each other on real logits.

**FFN went 15.16 → 15.77 ms. The whole model went 38.0 → 36.5–46.2 tok/s. Halving the bytes did not
halve the time.**

The reason is that this kernel is **compute-bound, not bandwidth-bound**: per 8 bytes the packed path
does the same two FMAs *plus* two shuffles and two converts. **`P2_EXPERT_PATH_DECOMPOSITION.md`
measured exactly this three hours earlier** on a synthetic kernel — ~60% of the expert path is
arithmetic, so a denser pack buys **at most 1.32×** on the FFN — and here that prediction is
confirmed end to end on a real pretrained model. A synthetic microbenchmark predicted a real
system's behaviour, which is the thing this programme's own history says usually fails.

**The remaining lever on this term is the LUT kernel** — table lookups and adds instead of
multiply-accumulate, with int8-quantized activations. That is what engine.c's pshufb-LUT path exists
for, and this runtime does not use it yet.

## 5. ⚠ Speed is not the binding constraint. Quality is.

**`T1_DONOR_TERNARIZATION.md` measured the conversion this runtime performs at `+4.738 BPB` on
Qwen2.5-1.5B — 948 σ_seed, on a 0.7676 baseline.** The model this runtime executes at 40–46 tok/s is
a model whose quality the conversion has destroyed.

**Nothing in this report should be read as progress toward a usable system until that is fixed.** It
is progress toward a *runtime*, which is a precondition and not the goal. Healing (QAT, layer-wise
distillation) is the critical path, and it is training.

## 6. Against the goal

The goal is **~10B at 50 tok/s** (good) or **100** (excellent).

| | measured |
|---|---|
| Qwen2.5-0.5B, t1 | 12.45 tok/s |
| **Qwen2.5-0.5B, t6** | **40–46 tok/s** (stable from context 20 to 200) |

**At 0.5B we are just short of the "good" bar, and 0.5B is 20× smaller than the target.** P3's
shape-only benchmark predicted 81 tok/s for 0.5B on the matvec term alone; the realised 40–46
includes everything P3 excluded (attention, RoPE, softmax, KV, norms, residuals), and the gap
between the two is the honest cost of being a real runtime rather than a weight-streaming model.

## 7. Owed

- **A Controller audit.**
- The **LUT kernel** (§4.2) — the only identified lever left on the dominant term.
- BPB **measured through this runtime** rather than through PyTorch, which closes the loop between
  what the engine computes and what the probes claim. `--bpb` exists and is untested at scale.
- Larger donors. Everything here is 0.5B; the exporter is size-agnostic but nothing above 0.5B has
  been exported or run.
- KV cache is fp32 and unquantized. Untouched, and it is the term that grows with context.

---

## 8. AMENDMENT (2026-09-04, same day) — the rates in §4 and §6 were contended

Everything above was timed while this machine was also running heavy probes. Re-measured on an
idle machine, three consecutive `--bench 300` repetitions, same binary, same weights:

| config | tok/s | reproducibility |
|---|---|---|
| packed + ternary head, t6 | **48.24 / 48.28 / 48.33** | ±0.05 |
| **`--lut`** (built after this report) | 51.05 / 50.22 / 50.47 | one outlier at 46.44 |

**§6's "40–46 tok/s" and `SPEED_LEDGER.md` §10's 36.1 tok/s are withdrawn as rates.** The honest
figure for this runtime at 0.5B is **48.3 tok/s**, and the delivered weight rate is **25.1
G-weights/s** (494.0 M active weights / 19.69 ms of weight-matvec), not the 18.6 that §10 recorded
and that had already been propagated into `donor_speed_budget.py` and `INDEX.md`. The full
re-measurement, with the per-organ profile, is `SPEED_LEDGER.md` §11.

**§4.2 and §7 named the LUT kernel "the remaining lever on this term". It has been built, and it
is worth 1.05×** — ffn ×1.07, head ×1.18, and `qkv_proj` actually got *slower*. Its correctness is
established (bit-exact vs a scalar-integer reference, two planted controls firing) and its cost is
not correctness but the int8 activations it requires, measured separately. So the sentence in §4.2
stands as a description of where the time is, and fails as a prediction of what could be done
about it: **§7's first owed item is now closed with a negative.**

The trajectory in §1 of `INDEX.md` should be read as 23.5 → 38.0 (ternary head) → 48.3 (packed)
→ 51.0 (LUT). **The only large win in that chain remains the one from profiling — the head.**
