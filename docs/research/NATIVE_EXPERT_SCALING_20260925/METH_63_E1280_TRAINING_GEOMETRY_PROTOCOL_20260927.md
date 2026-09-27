# METH-63: 10× distinct-expert training geometry preflight

**Question.** METH-57 validates 128 jointly trained residual experts per
layer; METH-54 measures a synthetic product-key CPU router at much larger
counts. Before a 10× trained ladder, test whether the same per-expert
rank-8 A/B geometry can be allocated with 1,280 independent slots per
layer, routed with exact rank-64 product-key top four, and used without
increasing selected factor work. This is a technical preflight, not a
quality or learned-distinctness claim.

**Frozen geometry.** Keep 24 layers, width 896, rank 8, top four, FP32
training factors and FP32 rank-64 projection/keys. Compare E128=8×16
with E1280=32×40. Use the METH-55 random A initialization and zero B
initialization with deterministic CPU seeds, and independently allocated
parameters for every pair. No copied, aliased or replicated expert
rows. The base function in the preflight is identity so that zero B
must give bit-identical outputs before training.

**Checks.** Instantiate all 24 layers at both E values on CPU. Record
physical parameter bytes, process RSS, elapsed time, distinct storage
addresses, and expected BF16 inference payload. On 32 fixed random
896-vectors per layer, compare the 16-pair product-key selection with
exhaustive pair scoring, including top-four IDs and scores; require
zero mismatches. Check zero-B output identity exactly and finite
factor/router values. Record the count of selected expert slots and
maximum/mean route load for the frozen inputs as descriptive only.
Abort on >8 GiB process RSS or >8 minutes. No GPU or T4 is required.

**Decision.** Passing establishes only feasible storage and exact route
arithmetic at 10× on this host. It does not establish that the extra
experts can learn distinct useful roles, that a CPU LUT implementation
handles them, donor-relative quality, DRAM traffic or ≥50 accepted
tokens/s. The next experiment must train E1280 with an explicit update
budget and compare it against an E128 control on independent quality
and route-load endpoints; newly initialized but unchanged slots do not
count as distinct learned experts.
