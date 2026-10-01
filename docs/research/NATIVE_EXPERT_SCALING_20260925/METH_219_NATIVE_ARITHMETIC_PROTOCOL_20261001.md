# METH-219: saved-core native arithmetic development

Freeze before execution. Resolve a missing composition variable: METH-214
rejects BF16 framework reconstruction, while METH-127 native core computes
FP32 and METH-182 FFN uses signed group-64 weight AND activation int8.
Do not infer native quality from BF16, or from FP32 weight-only quality.
Reuse physical METH-211 core and METH-126 exact BF16 factor bank; preserve
METH-214 failure. No new source, reserve, training or routing calibration.

## Fixed apparatus and arms

Use METH-213's now-consumed 24 sources and anchors, exact bound hashes in
the runner. Regenerate official BF16 donor prompt choices; reuse its
hash-bound full-head NLL rows. Recompute BF16 E1280 and require every
document NLL and prompt match count to reconcile exactly with METH-214.
Matched third control: original BF16 pretrained core decoded to FP32,
METH-127 BF16-rounded expert input/factors/gates/intermediates/residual,
FP32 route. SDPA throughout, TF32 disabled, float32 precision highest.
METH-127 validated eager attention; this SDPA development is not full
C parity. Router top-k tie order and FP accumulation also remain native
parity requirements.

Construct candidate from config only, supply all 290 parameters by all
364 METH-211 tensors, verify every copy and tied-head pointer. Decode
FFN signed codes times FP16 group scales into FP32 without BF16 weight
rounding. Exact BF16 embedding/head and attention decode to FP32;
stored controls retain FP32. Proposal R8 codes times half row scales
decode to FP32, without BF16 rounding. Attach actual METH-126 bank via
METH-127 loader; do not reinterpret unrounded checkpoint factors.

Two fixed candidate arms, both with identical stored core and bank:

- W8A32: decoded Q8 weights, FP32 input and hidden activations.
- W8A8: first FFN input rounded BF16 as in METH-182; group-64 maximum
  absolute / 127, nearest-even signed int8 [-127,127]; zero group scale
  one. FP32 gate/up and SiLU, requantize hidden similarly before down.
  GPU emulator dequantizes activations and uses FP32 matmul; C computes
  integer group dots and FP32 scale sums. No bit-exact claim.

Before model scoring, export every METH-211 FFN code/scale byte to
METH-182 layout, audit each segment readback/hash, run existing six-thread
METH-182 executable on existing METH-125 states. Compare all 16 x 24
outputs to the W8A8 emulator: median relative L2 <=1e-4, worst <=5e-4.
Stop on failure. Component timings are not full-model or large-n rate.

## Frozen development decision

Each candidate must pass all original quality limits, with the added
matched FP32 control: full-head BPB delta <=0.01 pooled / <=0.02 each
category against donor, BF16 E1280 AND FP32 native reference. Official
donor-top1 agreement loss <=1 percentage point pooled / <=2 each
category against BOTH E1280 controls. No confidence-limit adjustment.
For every candidate prompt state, K64 proposal must include the full
exact-head winner and exact selected-row rerank must reproduce it;
lowest token ID resolves equal exact logits. This is finite-state evidence,
not exact probabilities from K64 or universal shortlist safety.

W8A8 pass licenses freezing a NEW independent source protocol, then
native full parity/rate on that same arithmetic/storage artifact.
W8A32-only pass licenses a float-input grouped-Q8 native component cost
experiment, because existing fast kernel changes activations; no fresh
quality until that cost path is credible. Both fail: stop unchanged
precision variants and specify a trained/activation-sensitive transfer
change. No generation, task or blind review during this development.

Budget: local RTX 3060, six host threads, <=30 minutes after imports,
20 GiB RSS, 10.5 GiB allocated GPU; <1 GB additional local output.
No T4. Frozen core, factor-bank and consumed-source hashes recorded.
Physical core storage remains 820.707 MB; the prior ideal selected-weight
ledger does not price activation traffic, native kernel compute or DRAM
cache-line overhead. Large-n route/function coupling remains a separate
required mechanism; failed METH-218 routing is not promoted.

Command:

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth219_native_arithmetic_development.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth219_native_arithmetic_development_result.json --ffn-binary results/native_expert_scaling/meth219_m211_group64_ffn.bin
```
