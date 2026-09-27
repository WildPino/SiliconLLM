# METH-76: trained E128/E1280 product-key component on CPU

**Question.** METH-54 shows a synthetic rank-64 product-key router has
sublinear CPU cost at E27k/E273k, and METH-58 checks a learned E128
component. The METH-71 E1280 checkpoint has learned independent factors
and broad routing but fails METH-72/74 promotion. Measure exact native
component behavior and CPU cost at E128 versus E1280 on the same
METH-71 update-64 recipe, without making a quality claim.

Bind the METH-71 E128 and E1280 checkpoints by their recorded SHA-256
and use the first code prompt from the frozen METH-72 manifest. Export
all 24 layers: FP32 rank-64 projection/product keys and BF16
forward-value A/B rank-8 factors, preserving 8×16 and 32×40 axis
dimensions. Store a versioned header with dimensions and exact byte
length. Re-read every exported router/factor byte against the
checkpoint. Require all expert factor pairs to be distinct by
content within each layer, while recording changed-slot and
selected-slot evidence separately; distinct bytes alone do not
prove a useful expert.

For each layer, capture first/last actual BF16 hidden states on the
same prompt and two fixed-seed synthetic BF16 states. Freeze 96
fixtures per arm with PyTorch top-four IDs, gates and residuals. A
single-thread C reader must check unordered top-four IDs exactly,
gates within 1e-5, and residual maximum absolute error ≤0.03 for
all 96 fixtures. A deliberate +1 residual mutation must make the
checker fail. Report the maximum observed errors.

After correctness, benchmark 24-layer route and selected-factor
residual loops separately on the saved first actual hidden state of
each layer. Use 1,024 repeated token-equivalent passes and five
independent repetitions, report medians and checksums to prevent
optimization. Require the E1280 route median ≤3 ms/token and the
E1280/E128 route latency ratio ≤2×. Record router and selected-factor
addressed bytes per token, physical bank bytes, CPU/RSS and compiler.
These bounds qualify only the component's CPU cost, not an actual
LUT-coded bank, full `engine.c` integration or accepted tokens/s.

Local RTX 3060 for export, ≤10.5 GiB peak allocated GPU, ≤20 GiB
RSS, ≤15 minutes per export and ≤2 minutes per native arm. No T4.
Stop with partial evidence on mismatch or budget failure. The
METH-72/74 quality failures remain binding regardless of CPU result.
