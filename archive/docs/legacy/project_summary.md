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

## Project Status
**Current Phase**: Phase 5 (Engine Construction Preparation)
**Status**: Exploration Complete. The core architecture is consolidated.

## Technical Foundation (The CPU-Native ESN)
The project has converged on a **Reservoir Computing (Echo State Network)** paradigm optimized for AVX2 execution.

1. **The Substrate**: A 256-cell 1D grid processed via AVX2 `__m256i` arithmetic (`adds_epu8`, `subs_epu8`, `avg_epu8`). Throughput is ~370 cycles per virtual engine per tick, operating entirely in L1 cache.
2. **Temporal Memory (M4)**: A perfect ring-buffer delay line that stores raw input tokens, providing exact historical routing.
3. **Echo State Property (ESP) Enforcement**: A global soft-damping mechanism that forces the chaotic wave to forget its initial conditions and slave its state entirely to the input sequence.
4. **The Readout**: A zero-cost spatial integration strategy (R2) that collapses the grid into 32 channels. These channels (plus the M4 buffer) are fed into a linear Least Mean Squares (LMS) weighting layer.

We have proven that training the non-linear rules of the wave is mathematically intractable (fractal fitness landscape), but that an LMS readout placed on top of a fixed, damped, arithmetic wave provides massive learning capacity. The final step is tuning the ESP damping (e.g., `(3*state + wave)>>2`) to allow the signal to trigger the arithmetic non-linear saturations (which are necessary to solve linearly-inseparable tasks like XOR-2).
