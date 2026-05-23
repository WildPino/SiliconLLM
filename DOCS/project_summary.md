# SUMMARY: Project SiliconLLM - Rethinking LLM starting from the CPU

## Context and Philosophy
We are tackling a grand engineering challenge: reinventing the concept of a Large Language Model (LLM) not by starting from existing architectures (e.g., Transformers), but from the native execution and throughput capabilities of modern CPUs. The core idea is that **the CPU should not chase GPU paradigms** (massive parallelism on huge matrices), but rather impose its own natural shape onto the model: small blocks, stream prediction, cache locality, branchless logic, parallel pipelines, and native `1-bit` / `2x2` operations.

## Phase 1: Hardware Archeology and Passive Benchmarking
Instead of immediately theorizing and writing ML models, we built a benchmarking lab in C (`benchmark.c`) compiled with GCC to map out the "cognitive geography" of the CPU (specifically tested on an AMD Ryzen 5 3600X).

We measured clock-cycle throughput using `__rdtsc()` and tested:
1. Sequential memory accesses (to simulate predictable streams)
2. Random memory accesses (to simulate sparse dependencies/attention)
3. Base speed of logic primitives (scalar XOR + Popcount, unrolled 2x2 matrix multiplications).

## The Crucial Discovery: Routing Entropy
The results highlighted a fundamental difference:
* The CPU processes sequential data at an impressive pace (2-3 cycles/element even for very large blocks), fully exploiting the hardware prefetcher.
* **Random accesses destroy the pipeline:** a cache miss ending up in L3 memory causes performance to plummet, going from ~3 cycles to a whopping **56 cycles** of stall (~14ns lost per single element).

This led us to a strong theoretical conclusion: the real enemy is not the limitation in raw computing power (FLOPs), but the **entropy of memory routing**. Building a model based on random lookups, sparse KV caches, or global attention paralyses the CPU.

## The Shift in Direction: "Wave-Like" Computation
A clear design direction emerges from this analysis: we must move away from the transformer. The new model will need to be:
* **Wave-like:** based on local propagation, nearby transformations, and contiguous state (similar to discrete PDEs or cellular automata).
* **Clock-Cell Based:** the minimum computing units will not be massive "layers", but tiny blocks (e.g., 2x2 matrices) capable of exactly saturating the pipeline without ever exceeding L1/L2 cache limits (512KB).

## Next Steps: Phase 1.5 (Active Computation)
Having measured "passive memory", the plan is now to test "active computation" to understand how to keep the CPU under optimal stress. The upcoming tests involve:
1. **Dependency Chains:** measuring pure Serial vs Out-of-Order (OoO) execution.
2. **SIMD Occupancy:** measuring the latency in packing/unpacking bits and the actual occupancy of AVX.
3. **Popcount Scalability:** testing XNOR+Popcount (the possible native "dot product") in scalar form vs AVX2/AVX512.
4. **Tiny Tiles and Branching:** testing 2x2 / 4x4 blocks, Hadamard logic, and measuring the performance gap between predicted logic (if/else) and branchless algebra (bit-masking).
