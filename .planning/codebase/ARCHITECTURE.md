# SiliconLLM — architecture and data flow

## System thesis

The frozen architecture in `docs/SCALEUP_ARCHITECTURE.md` is CPU-first: keep reusable compute small and cache-resident, move capacity to sparse/indexed memory, stream sequentially, and use verification to increase tokens per stream when shared traffic dominates. The implementation is an experimental engine plus a large measurement harness, not yet a single production training/inference product.

## Runtime layers

```text
donor snapshot / synthetic shape
        ↓
calibration + exporter + sidecar
        ↓
binary artifact (fp32 / packed / ternary / int8 / low-rank / carved)
        ↓
donor_engine.c or donor_engine_e*.exe
        ↓
quality: BPB, logits, teacher-forced top-1, greedy generation
speed: tok/s, component timing, bandwidth, occupancy
        ↓
pre-registered gate + result JSON + probe + SPEED_LEDGER/INDEX
```

## Engine decomposition

`benchmarks/donor_adaptation/engine/donor_engine.c` owns file loading, metadata, attention/SSM recurrence, FFN/MoE routing, top-k selection, kernels, BPB/logit/timing modes and configuration reporting. `benchmarks/phase60/engine.c` is the earlier/general engine used by the phase-60/63/64 scale-up apparatus. They share concepts but are not interchangeable without parity gates.

The current donor engine has several orthogonal arms:

- attention reduction (`serial` vs `avx4` and related attention arms);
- packed/ternary LUT matvec;
- int8 matrix kinds (`MK_I8`, `MK_I8_T`) added for carved FFN paths;
- low-rank factored q/o/head paths;
- FFN carve and router selection;
- exponential and activation/matvec acceleration flags.

The engine prints `CONFIG` so a result can assert the actual arm. E64 showed why this is load-bearing: comparing binaries without comparing kernel configuration misclassified an attention-flag change as build noise.

## Research architecture

Each experiment is a small vertical slice:

1. A brief fixes question, scope, arms, predictions and gates.
2. An apparatus/exporter builds the exact artifact and writes provenance.
3. A runner executes planted controls before new cells.
4. A probe explains the result, including void/malformed/owed cases.
5. `docs/research/donor_adaptation/INDEX.md` is the narrative index and `SPEED_LEDGER.md` is the speed/phase ledger.

The programme deliberately separates arithmetic correctness, quality, ranking, speed and training. A positive speed shape with synthetic/noise weights is not a model-quality result; a post-hoc quality result does not imply trainability.

## Phase 64 training architecture

`benchmarks/phase64/mve/` is a separate end-to-end pilot: data/tokenizer → teacher logits → fp pretraining → QAT → MoE upcycle + recall → reverse-KL. It exercises the pipeline at 30.1M total / 11.2M active parameters on TinyStories, with fixed gates for cross-tokenizer KD, resume survival, recall stability, QAT transition and throughput. `benchmarks/phase64/data/` provides corpus/packing/audit controls.

## State and authority

- Canonical research narrative: `docs/research/donor_adaptation/INDEX.md`.
- Canonical speed history: `docs/research/donor_adaptation/SPEED_LEDGER.md`.
- Current owner actions: `docs/COMMUNICATION.md`.
- Design/spec rationale: `docs/SCALEUP_ARCHITECTURE.md` and `docs/research/donor_adaptation/decisions/`.
- Newer unpromoted working-tree evidence is explicitly lower authority until its probe/ledger is updated and committed.

