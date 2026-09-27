# METH-64: selected-factor CPU offload and gradient parity

**Question.** METH-63 allocates E1280 but dense FP32 Adam requires at
least 7.046 GB for factor parameters, gradients and moments before the
donor and activations. Establish whether CPU-resident factors can feed
only selected top-four rows to the GPU and receive exact selected-row
gradients, so larger expert count need not dictate GPU parameter state.
This is an apparatus gate before donor training, not a learned-quality
result.

**Fixed rule.** Keep rank-8 A/B FP32 factor banks on CPU, with distinct
storage for every E1280 pair in every layer. Keep rank-64 product-key
router tensors on GPU. For each forward, compute top-four IDs on GPU,
gather factor rows from CPU, move only gathered rows to GPU and attach
autograd leaves. During backward, aggregate duplicate row gradients
by expert ID into CPU sparse maps; no full E1280 gradient tensor may be
materialized. Use the same SiLU, gate softmax and selected-residual
arithmetic as METH-55.

**Numerical check.** First compare CPU-offloaded and dense GPU factor
implementations at a small 5×8 grid, width 16, rank 3, top four, with
identical nonzero A/B and fixed product-key weights. On a fixed batch
of seven inputs, require equal routes and forward outputs, and maximum
absolute A/B row-gradient difference ≤2e-6 after duplicate aggregation.
Require CPU sparse gradient rows to equal the dense nonzero row set.

**Scale check.** Instantiate 24 independent 32×40 factor banks at width
896/rank 8 and run one forward/backward over 32 fixed synthetic hidden
vectors per layer with a nonzero B initialization solely for gradient
verification. Check finite loss/gradients, exact product-key top-four
against exhaustive pair scoring, no GPU bank tensor, and a constant
selected-row transfer bound of 24×32×4×2×8×896×4 bytes per direction
before duplicate compression. Record GPU peak allocation, CPU RSS,
elapsed time and unique gradient rows/layer. Abort above 4.5 GiB GPU
allocated, 10 GiB process RSS or 8 minutes. This is a local RTX 3060
run; no T4.

**Decision.** A pass only validates the offload gradient path and its
memory/cost envelope on synthetic states. A separate frozen experiment
must perform joint donor training with sparse row optimizer state,
compare E128/E1280 on disjoint content, verify changed expert slots and
route load, and then evaluate generation/task/semantics. Synthetic
gradients and initialized distinct storage cannot count as learned
expert capacity.
