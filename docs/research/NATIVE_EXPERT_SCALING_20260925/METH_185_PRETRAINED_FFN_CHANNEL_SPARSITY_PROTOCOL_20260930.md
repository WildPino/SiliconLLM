# METH-185: pretrained FFN channel sparsity upper-bound screen

**Uncertainty and evidence.** The BF16 Qwen2.5-0.5B-Instruct donor plus
learned E1280 bank preserves quality, but its full native FP32 reference
is only 20.791 tok/s with the improved head. Grouped-R8 FFN still takes
12.829 ms/token in a C component and its quality-valid composition
fails blind semantics. Grouped-Q4 FFN sharply damages full-model
quality. Test a different pretrained-to-conditional transformation:
selectively read channels of the donor's actual dense SwiGLU FFN.
If even an activation-aware top-K channel oracle cannot closely
reproduce the donor FFN, a cheaper approximate router for that same
channel partition is unlikely to work without retraining.

**Bound sources and operation.** Bind the BF16 Qwen2.5-0.5B-Instruct
source revision `7ae557604adf67be50417f59c2c2f167def9a775`,
source SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
and METH-125's BF16 E1280 pre-MLP state file SHA-256
`f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`,
shaped 256 positions × 24 layers × 896. These are real states from
the quality-valid BF16 E1280 assembly, not a quantized-core trajectory
or fresh quality cohort. For each layer and state, calculate the
original BF16 gate/up, SiLU(gate)×up activation and down projection.
For each FFN channel `j`, score `abs(activation[j])` times the FP32
L2 norm of the corresponding BF16 down-projection column. Select the
highest K scores with stable lowest-channel-ID tie breaking, zero all
other activations, and apply the original BF16 down projection. Test
K = 128, 256, 512, 1024 and 2048 of 4864. Verify that K=4864
reproduces the unmasked BF16 output exactly.

**Frozen controls and decision.** At K=512, 1024 and 2048 also select
the K channels with largest down-column norm alone, fixed for a whole
layer, to distinguish conditional from static pruning. Across all
6,144 layer/state outputs, record relative output L2, pooled median,
95th percentile, maximum and per-layer medians for each arm. A K≤2048
licenses a learned cheap-router and full-model candidate only if
(1) dynamic median relative L2 ≤1%, (2) dynamic 95th percentile ≤5%,
and (3) its dynamic median is at most half the static-norm median.
Choose the smallest passing K; do not adjust these criteria after
results. The ideal BF16 weight stream for selected gate/up/down rows
is `24×3×896×K×2` bytes/token, excluding routing, core/head,
activations and real DRAM transactions. This oracle still computes
all gate/up channels and scores, so a pass is only a necessary
representation screen, never evidence of runtime savings, useful
larger expert counts or donor-relative model quality. A failure
rejects direct top-K channel micro-experts at these K and error bars,
not other learned decompositions.

**Cost and stop.** Local RTX 3060 only, six host threads, 15 minutes,
10.5 GiB peak allocated GPU memory, 12 GiB RSS and 1 GB new disk.
Stop on source/state hash failure, K=4864 parity failure, nonfinite
output or resource limit; retain failed stage and partial layer rows.
Do not use T4. The result record supplies the exact runner revision
and command.
