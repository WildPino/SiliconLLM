# Archive — historical / superseded

This directory preserves earlier eras of the project, kept for provenance. **None of it is
part of the current CPU-native LLM design** — see the top-level [`README.md`](../README.md)
and [`docs/SCALEUP_ARCHITECTURE.md`](../docs/SCALEUP_ARCHITECTURE.md) for the live work.

- **`compressor/`** — the original *Silicon Entropy Engine* lossless compressor (V1.0.x,
  Phases 1–40): a streaming mixture of statistical experts blended by an online
  exponentiated-gradient MoE, shaped by CPU cache topology rather than GPU parallelism.
- **`benchmarks/`, `scripts/`** — the full experiment history, including the token-level and
  "mantra-pure" eras (Phases 42–54) that led to the current SSM design. Many are dead ends;
  they are kept here as the honest record of what was tried and rejected.
- **`docs/`** — reference documentation for the compressor era.

Comments in the oldest experiment scripts may still contain Italian (the working language of
early development); these dead-end files were intentionally not retranslated.
