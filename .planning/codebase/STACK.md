# SiliconLLM — technology and runtime map

**Snapshot:** 2026-09-14. This is a research/benchmark repository, not a packaged application. The current research corpus is much newer than the initial graph snapshot; always check `git log` and the research ledger before quoting a result.

## Purpose

The project co-designs a CPU-first large agentic LLM for a Zen2/no-VNNI machine. The central design combines a small cache-resident compute core with sparse/indexed capacity in DRAM. The active donor-adaptation programme tests whether pretrained Qwen donors can survive the project's formats and whether the measured engine budget can reach 50/100 tok/s.

## Languages and execution

- **C:** inference engines and microbenchmarks, primarily `benchmarks/phase60/engine.c` and `benchmarks/donor_adaptation/engine/donor_engine.c`.
- **Python:** exporters, calibration, BPB/quality probes, speed probes, result adjudication, corpus construction and Phase 64 training; examples are `benchmarks/donor_adaptation/engine/e1_bpb_through_engine.py`, `benchmarks/donor_adaptation/engine/synth_export.py`, and `benchmarks/phase64/mve/mve_train.py`.
- **Shell / PowerShell:** reproducible run wrappers such as `benchmarks/phase64/bench_64_0.sh` and the archived phase runners.
- **Markdown / JSON / JSONL / binary artifacts:** preregistrations, ledgers, reports, sidecars, token streams, weights and result tables.

## Core runtime

- Reference hardware: AMD Ryzen 5 3600X / Zen2, AVX2 + FMA, DDR4, 16 MB L3 per CCX, no VNNI. Hardware-dependent values are runtime parameters, not universal constants.
- C engine build policy is normally `-O3 -mavx2 -mfma -ffp-contract=on`, with optional Zen2 tuning; see `benchmarks/phase64/bench_64_0.sh`.
- CPU inference uses OpenMP threads, fp32 state, packed/ternary LUT kernels, int8 fallback paths and optional attention/exponential/matvec arms. The engine prints a `CONFIG` line; current experiments increasingly gate on that line rather than on an executable filename.
- Local Python execution uses `.venv` / `.venv_wsl` when available. GPU training and teacher scoring use Linux/Kaggle T4 or local NVIDIA tooling.

## Important formats

- fp32 dense weights: faithful reference, high traffic.
- ternary / packed LUT: about 0.5 B/weight, fast but post-hoc quality can collapse.
- int8 per-row scaled paths: 1 B/weight; E60/E62 establish the precision cliff and E63 adds carved-int8 loading.
- factorized low-rank matrices and carved FFN artifacts: structural traffic reductions, with separate sidecar metadata.
- Binary artifacts and their JSON sidecars are often in `D:\_ktmp\...` or Gitignored result trees; the repository documents provenance but does not necessarily version the large binaries.

## Main dependencies and configuration

- PyTorch, NumPy and Transformers support donor conversion, calibration, BPB checks and training.
- Hugging Face model snapshots are pinned by model revision in scripts/sidecars; donor examples include Qwen2.5-Coder 0.5B/1.5B/3B/7B.
- Kaggle, vLLM and CUDA/T4 are used by the Phase 64 MVE and H0/H1 training paths; local Windows is the primary CPU research environment.
- Configuration truth is spread across exporter arguments, sidecars and engine `CONFIG` output. Relevant files: `benchmarks/donor_adaptation/configs/_manifest.json`, `benchmarks/donor_adaptation/engine/qwen_export.py`, `benchmarks/donor_adaptation/engine/synth_export.py`.

## Build and run entry points

1. Build/run donor engine: `benchmarks/donor_adaptation/engine/donor_engine.c` plus the experiment-specific Python runner.
2. Export donors: `benchmarks/donor_adaptation/engine/qwen_export.py` and `benchmarks/donor_adaptation/engine/synth_export.py`.
3. Score quality: `benchmarks/donor_adaptation/engine/e1_bpb_through_engine.py` and the E-specific runners.
4. Run speed: E-specific engine runners or `benchmarks/phase60/engine.c`; results are summarized in `docs/research/donor_adaptation/SPEED_LEDGER.md`.
5. Run Phase 64 training/data pipeline: `benchmarks/phase64/mve/` and `benchmarks/phase64/data/`.

