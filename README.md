# SiliconLLM

A research project exploring CPU-native sequence modeling — rethinking language model architecture from the ground up based on hardware thermodynamics rather than GPU-centric paradigms.

## Core Idea

Modern CPUs are bottlenecked not by raw compute (FLOPs) but by **memory routing entropy**. Every cache miss costs ~56 cycles vs ~3 cycles for sequential access. Transformer-style global attention is fundamentally at odds with this reality.

SiliconLLM builds a sequence model whose shape is dictated by the CPU itself: small AVX2 blocks, stream prediction, L1/L2 cache locality, branchless arithmetic, and wave-like local propagation instead of global attention.

## Architecture

The current implementation is a **CPU-native Reservoir Computing (Echo State Network)**:

- **Wave substrate**: 128-block AVX2 grid (4096 cells, ~128KB — fits in L1/L2). Uses saturating arithmetic (`adds_epu8`, `subs_epu8`, `avg_epu8`), ~370 cycles/tick.
- **Encoding**: Each input byte maps to a 32-byte random binary codebook vector, injected into a single contiguous grid block.
- **Temporal memory (T3)**: A 16-token ring-buffer re-injected spatially-shifted each tick, creating physical temporal discounting via wave diffusion.
- **Readout**: 32-dimensional lane-aware pooling over the codebook axes, fed into a linear LMS layer (the only trained component).

Training the wave dynamics is mathematically intractable (fractal fitness landscape). The fixed damped wave provides non-linear feature extraction; only the linear readout is learned.

## Repository Structure

```
SiliconLLM/
├── src/            Core library (silicon_v0, silicon_entropy, wave_engine)
├── benchmarks/     All benchmark, evaluation, and training harnesses
├── data/           Input corpora and test datasets
├── scripts/        Utility and dataset preparation scripts
├── docs/           Architecture decisions, phase notes, research
│   ├── phases/     Per-phase technical documentation
│   ├── research/   Low-level CPU architecture research
│   └── archive/    Historical plans and superseded walkthroughs
└── build.bat       Build entry point (GCC/MinGW, AVX2)
```

## Building

```bat
build.bat
```

Requires GCC with AVX2 support (`-mavx2`). Compiled binaries go into `bin/`.

## Key Results

| Metric | Value |
|---|---|
| Sequential throughput | ~3 cycles/element |
| Random (L3 miss) penalty | ~56 cycles/element |
| Wave tick (128-block grid) | ~370 cycles |
| Bigram prediction accuracy | Measured across phases 10–14 |

See `docs/architecture_decisions.md` for the full reasoning trail and `docs/phases/` for per-phase benchmarking results.
