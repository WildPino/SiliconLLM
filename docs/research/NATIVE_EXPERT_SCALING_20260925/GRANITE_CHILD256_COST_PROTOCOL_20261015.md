# Synthetic Q4_K child256 projection-bundle cost screen

**Status: source prepared for review; no compilation or run has been authorized in this preparation step.** This is an operator-bundle cost screen for one small-instruct geometry, not an inference engine, donor-quality test, or a claim about a larger family. All weights and inputs are synthetic and deterministic. The 24 layer inputs and attention-context vectors are independent fixtures. The fixture omits causal attention, RoPE, KV/cache, token sampling and generation. A pass is necessary cost evidence only; it cannot admit a 50-token/s model.

## Frozen scope

Geometry is 24 layers, width 1024, 32 resident experts, top 8 parents, parent FFN width 512, two 256-wide children, and 49,155 vocabulary rows. The 49,155 row head is explicitly handled as full four-row I8/A16 tiles plus a three-row tail. Q/K/V/O and the full head keep the M413 I8/A16 coefficient, per-row F32 scale and I64 integer-dot arithmetic. The routed expert gate/up/down alone uses Q4_K blocks with Q8_K activations. A seeded valid raw 144-byte Q4_K block producer gives variable finite positive d/dmin, valid scale/minimum fields, and nibble codes. It is **not** a measured F32→Q4 quantizer or a source-weight export. Parent blocks are copied into child-major packed buffers without requantization; both parent and child copies and all 32 experts per layer remain resident and are counted in stored payload.

The same eight fixture IDs are used in both modes. Parent scores are F32 dot products over 32 experts, followed by source-style top-8 selected-logit softmax and sorted-parent accumulation. Each selected parent also has two input-dependent F32 child logits and a stable two-way softmax. Both modes pay the parent router and selector cost. In `all_children`, both children execute and each has the original parent mass; their down outputs reconstruct the parent contribution. In `one_child`, the selected child executes with mass `2 × selector_probability × parent_mass`. This compensation is a declared synthetic cost operation, not a learned or quality-preserving rule. Norms, A16 and Q8 quantization, SiLU, residual mixing, BF16 embedding lookup and the entire I8 vocabulary head are inside the timed call.

Per call, I8 projections and head touch 125,832,192 coded coefficients and 491,532 scale bytes. The full parent Q4 expert bank is 679,477,248 bytes; the child-major copy is another 679,477,248 bytes. `all_children` evaluates 301,989,888 Q4 coefficients (169,869,312 block bytes); `one_child` evaluates 150,994,944 (84,934,656 block bytes). Both read 3,145,728 parent-router F32 bytes and 1,572,864 selected-child-logit F32 bytes. The descriptor is analytical executed payload, not observed DRAM traffic: CPU cache reuse, line fetches, prefetching, instruction cost and memory contention remain measured by elapsed time only. The 512 MiB eviction sweep, where requested, is outside each timer and is a cache-pressure intervention, not proof of a physical DRAM miss.

## Prospective controls and decision

An explicit `--check-only` stage must pass before timing. It checks every source→child Q4 block copy, runs both modes on one complete fixture, compares every computed Q4 output with an independently decoded scalar Q4×Q8 reference (full-vector relative L2 ≤ 1e-5), and verifies sample I8 integer/scaled Q/K/V/O/head rows with scalar I64. It independently checks parent IDs/masses, child logits/IDs/probabilities, SiLU, mixture and residual arithmetic. The receipt includes SHA-256 of all executed projection, expert, mixture, residual and full-head output vectors; skipped child arrays are excluded. Any nonfinite value or check failure rejects timing.

One fixed worker profile uses exactly three validated physical cores with logical masks 1, 4, 16 (logical CPUs 0, 2, 4). Each process performs exactly one untimed warmup and then three measured repetitions of all eight fixed fixtures. There are three independent processes per mode and per cache profile (resident, then explicitly cache-evicted), in balanced mode order. Each process/fixture median across its three repetitions must be ≤14 ms, and the maximum/minimum of all nine observations for that fixed mode/profile/fixture must be ≤1.10. No outlier or warmup is removed after inspection. Identical mode/fixture output SHA-256 across processes, repetitions and profiles is required. Any failed run, receipt or cap stops the sequence and retains the failure. Overall wall time is capped at 15 minutes, parent+child RSS at 6 GiB, each process at 120 seconds, and output at 512 MiB.

After review, the controller's three explicit steps are:

```powershell
python benchmarks/native_expert_scaling/granite_child256_cost.py --freeze results/native_expert_scaling/granite_child256_cost_20261015/freeze.json
python benchmarks/native_expert_scaling/granite_child256_cost.py --check-only --binding results/native_expert_scaling/granite_child256_cost_20261015/freeze.json --output results/native_expert_scaling/granite_child256_cost_20261015/check
python benchmarks/native_expert_scaling/granite_child256_cost.py --run --binding results/native_expert_scaling/granite_child256_cost_20261015/freeze.json --check-receipt results/native_expert_scaling/granite_child256_cost_20261015/check/result.json --output results/native_expert_scaling/granite_child256_cost_20261015/timing
```

The freeze binds exact source, protocol, helpers, physical-worker profile, Git HEAD, compiler and OpenMP runtime hashes. The controller compiles a fresh binary, keeps exclusive JSONL process logs and exit receipts, verifies the separate numeric control and full output hashes, and records every timing decision. Nothing in this protocol authorizes a model port, weight download, training run, or extrapolation to a larger-family core.

## First launch and deterministic-build repair

The initial numeric control passed. Timing stopped before launching any worker
because separately linked PE binaries differed in timestamps (two byte offsets,
128 and94724). The strict binary-identity check retained that first failure.
The repaired freeze adds `-Wl,--no-insert-timestamp`; it changes PE metadata,
not C arithmetic or timing gates. New exclusive check/timing directories retain
the original results. The repaired numeric stage must pass before timing.
