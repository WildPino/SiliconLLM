# Phase 1: Hardware Archaeology and CPU Benchmarking

This directory contains the documentation and source code for the first phase of the **SiliconLLM** project.

## Goal
The goal of this phase is to stop thinking at the "transformer layer" level and start thinking at the "CPU clock cycle" level. We are building a hardware measurement lab to understand exactly how the target CPU (Ryzen 5 3600X, Zen 2 architecture) behaves under different loads.

## Ryzen 5 3600X Architecture Basics
- **Cores/Threads:** 6 Cores / 12 Threads
- **L1 Cache:** 32 KB instruction + 32 KB data per core
- **L2 Cache:** 512 KB per core
- **L3 Cache:** 32 MB shared (2x16MB CCX structure in Zen 2)
- **SIMD Support:** AVX, AVX2 (256-bit width)
- **Compiler:** GCC

## Benchmarks
Since we don't have access to dedicated profiling tools like VTune or uProf, we will measure the CPU performance cycle-by-cycle using the `__rdtsc()` intrinsic, which reads the Time Stamp Counter directly from the processor.

We will run four fundamental microbenchmarks in pure C:

### 1. Sequential Memory Access
Measures the latency and throughput of reading/writing contiguous blocks of memory. We will test sizes that fit in L1, L2, L3, and main memory (RAM) to map the cache boundaries and understand the penalty of leaving the L1/L2 cache.

### 2. Random Memory Access
Measures the cost of unpredictable memory access patterns. This defeats the hardware prefetcher and causes frequent cache misses, simulating the cost of unpredictable routing or divergent flow control in a neural model.

### 3. Bitwise Operations (XOR + Popcount)
Measures the throughput of essential 1-bit model primitives: XNOR, popcount, and packed bit manipulation. We will test scalar and vectorized (SIMD) versions.

### 4. Tiny Matrix Kernels
Measures the execution pipeline occupancy when computing very small, localized blocks (e.g., 2x2 or 4x4 matrix multiplications). We will ensure these tiny kernels remain entirely within the L1 cache to test the limits of the Fused Multiply-Add (FMA) units.

## Next Steps
After analyzing the baseline metrics from these benchmarks, we will define the "sweet spot" for our clock cell's working set size and choose the ideal core operations to base our model upon.
