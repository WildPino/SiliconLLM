<div align="center">

# Silicon Entropy Engine

### A CPU-native LLM, co-designed for the memory wall

**Making a large, agentic-capable language model fast on a consumer CPU — by architecture, not by brute force.**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21128459-blue)](https://doi.org/10.5281/zenodo.21128459)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Latest release](https://img.shields.io/github/v/release/WildPino/SiliconLLM?color=2dd4bf&label=release)](https://github.com/WildPino/SiliconLLM/releases)
![Target](https://img.shields.io/badge/target-x86--64%20AVX2%20(Zen%202)-informational)

![Architecture](assets/architecture.png)

<sub>Research project · target hardware: Ryzen 5 3600X (Zen 2, AVX2, **no** AVX-512 / VNNI) · 128K-context goal</sub>

</div>

---

## What this is

Modern LLM inference is **bandwidth-bound**: on a CPU you spend most of every token streaming weights from memory, not computing. This project asks a different question than "how do we run a GPU model on a CPU" — it asks **what architecture would you design if the CPU cache hierarchy were the first constraint, not an afterthought.**

The answer is a **split design**:

- a small, **always-on "thinking" core** — an SSM (state-space) backbone with a ternary, sparse MLP, sized to live entirely inside the **L3 cache**;
- a large, **sparse "knowing" tier** — a retrieval index for long-context recall plus a granular mixture-of-experts, which scale *total* knowledge while keeping the *active* parameters per token inside the cache budget.

Everything is held together by a **block-decode chassis** that turns the CPU's worst case (random, per-token memory access) into its best case (bulk, sequential streaming).

> **Honest status.** This is a **research project at the validation stage**, not a finished product. The foundation has been measured end-to-end on real Zen 2 hardware (below). The unified C inference engine is the roadmap, not yet the deliverable. Every number on this page comes from a real experiment, and every claim is stated with its honest limitations.

---

## The measured findings

The project advances by **pre-registered probes**: each hypothesis gets a controlled experiment with a gate fixed *before* the results are seen. Here is what has survived.

### 1 · Ternary weights are the right call on Zen 2 — and cost almost nothing in quality

Without AVX-512/VNNI, the usual int8 quantization path barely helps (~1.2×). A **ternary (1.58-bit) weight** kernel using `pshufb` byte-LUTs goes the *other* way — fewer bits means faster — and every kernel is bit-exact against a scalar reference.

<div align="center">
<img src="assets/bench_ternary_speed.png" width="49%">
<img src="assets/bench_ternary_quality.png" width="49%">
</div>

Trained from scratch (QAT, not post-training), the ternary MLP costs only **+0.028 BPB** at 5M params on TinyStories — and that is an *upper bound*, since the ternary run was still improving when measured. We ternarize **only the MLP** (the byte-sink), keeping the recurrent backbone in fp32; the ternary gap is known to shrink with scale, so micro-scale is the pessimistic end.

### 2 · Activation sparsity and the L3 cliff

A gated **dReLU** MLP is naturally sparse — up to **92%** of hidden units are inactive per token — at essentially zero quality cost (+0.0006 BPB, matched training). Combined with ternary weights, this compounds along **independent axes** to roughly **21× fewer MLP bytes per token** versus fp32-dense, for about **+0.03 BPB** total.

But sparsity only pays if the *working set* fits the cache. A synthetic sweep on the real 3600X finds a sharp **step at 16 MB — the exact L3-per-CCX size** — below which the CPU is compute-bound at ~100 GB/s, and above which it falls off a cliff toward the ~28 GB/s DRAM floor. **This 16 MB is the keystone constraint** that sizes the entire architecture.

<div align="center">
<img src="assets/bench_cache_cliff.png" width="58%">
<img src="assets/bench_compound.png" width="40%">
</div>

### 3 · The active set is predictable — so you skip, you don't gather

Reducing bytes is worthless if the bytes you *do* need are scattered (a random gather defeats the cache). Two findings close this:

- the active set is **intrinsically predictable in-place** — a cheap linear probe on the current state forecasts 86–92% of the active units, no special training needed;
- a **temporal-coherence regularizer** makes that sparsity **block-structured** (contiguous, cache-friendly) at zero quality cost — recovering the sparsity headline as *real, skippable* blocks rather than a scatter.

<div align="center">
<img src="assets/bench_predict_sparsity.png" width="85%">
</div>

Together: you can skip ~half the MLP blocks, and cheaply predict *which* half — the control system that lets a mixture-of-experts and the cache actually pay off in bandwidth.

### 4 · Long-context recall that fits the budget

A two-stage IVF-PQ retrieval tier gives **128K-context recall in ~18 µs/query** on CPU. The load-bearing originality is the **learned InfoNCE representation** (not the partition, which can be data-independent and simpler). A predicted failure mode — query *drift* over long contexts — was investigated and found **absent** on this SSM: the state norm is bounded, so recall stays flat versus distance in-distribution and 21× out-of-distribution.

### 5 · Scaling capacity without scaling the active cost — granular MoE

A mixture-of-experts grows *total* parameters while keeping the *active* parameters per token inside the cache budget. At matched active cost (and matched total params), a **granular MoE** (many small experts, top-k routed) not only preserves quality — it improves on the dense baseline, and even edges out a dense model with 4× the active parameters. Fine-grained experts beat coarse ones, confirming the block-structure finding above.

<div align="center">
<img src="assets/bench_moe.png" width="70%">
</div>

Routing has **no temporal locality** — the active expert set is i.i.d.-like across tokens (measured twice, at neuron and expert granularity). So the experts do *not* form a hot pool that stays cached; instead they are **streamed from DRAM in contiguous, granularity-bounded chunks** (bulk and sequential — ρ-safe, no pointer-chasing). This splits the memory budget into **two pools**: a resident core (≤ 16 MB L3, reused every token) and a streamed expert tier (a few MB/token at the DRAM floor). That two-pool model is the honest answer to the random-latency concern: no hot pool (measured), but no latency trap either (by construction).

---

## Status

| Component | State | Evidence |
|---|---|---|
| Ternary 1.58-bit LUT kernel | ✅ validated on Zen 2 | 4.2–5.0× matvec, bit-exact, +0.028 BPB |
| Activation sparsity (gated dReLU) | ✅ validated | 92% sparse, 2.12× skip, +0.0006 BPB |
| Cache-residency budget (16 MB L3) | ✅ measured | bandwidth cliff at L3-per-CCX |
| Long-context recall tier | ✅ de-risked end-to-end | ~18 µs/query, drift-free |
| In-place predictability | ✅ validated | 86–92% recall, predictor-free |
| Block-structured sparsity | ✅ found (byproduct) | 18% → 50% skippable @ zero cost |
| Granular MoE (capacity tier) | ✅ validated | granular > dense at iso-active; fine > coarse |
| Two-pool memory model | ✅ characterized | resident core + streamed experts (no hot pool) |
| Block-decode / MTP execution | 📋 designed | roadmap (execution chassis) |
| Unified C inference engine | 🔧 in progress | reference core underway |

See [`docs/SCALEUP_ARCHITECTURE.md`](docs/SCALEUP_ARCHITECTURE.md) for the full buildable blueprint, and [`HANDOFF.md`](HANDOFF.md) for the complete technical narrative including every negative result.

---

## What this is *not* (scope discipline)

- **Not a deployed speedup.** The probes run in a sandbox (a ~5M model that fits in cache), so they measure the *property* — the quality cost, the cache behavior, the predictability — not the realized end-to-end bandwidth of a large model. That is what the C engine will measure.
- **Not a finished LLM.** The current model is trained on TinyStories at small scale to isolate architectural questions cleanly. Broad-distribution quality at scale is future work.
- **Honest about magnitude.** Individual levers are stated at their *predictor-free, measured* value (e.g. 2.12× sparsity, ~3× residency), never the optimistic ceiling. The compound win is one-to-two orders of magnitude, but no single headline number is load-bearing.

---

## Reproduce the findings

The benchmark charts on this page are generated from the measured numbers:

```sh
python scripts/make_readme_charts.py     # -> assets/*.png
```

The probes themselves live under `benchmarks/phase55-57/` (weight-streaming: ternary kernel, sparsity, cache sweep) and the retrieval work under `benchmarks/phase56/`. The C microbenchmarks target Zen 2:

```sh
clang -O3 -mavx2 -march=znver2 benchmarks/phase57/phase57_lutbench.c   -o lutbench   -lm
clang -O3 -mavx2 -march=znver2 benchmarks/phase57/phase57_cachesweep.c -o cachesweep -lm
```

---

## Repository layout

```
SiliconLLM/
├── assets/                     README charts (generated from measured data)
├── docs/
│   ├── SCALEUP_ARCHITECTURE.md   the buildable blueprint (the current design)
│   ├── EXTERNAL_REVIEW_01.md     external technical review + responses
│   └── research/                 background research reports
├── benchmarks/
│   ├── phase55/                  CPU SSM language model + C inference kernel
│   ├── phase56/                  long-context recall (IVF-PQ, drift, MQAR)
│   └── phase57/                  weight-streaming + predictor/MoE probes
├── scripts/                    chart generation
├── archive/                    historical / superseded (compressor + early eras)
└── HANDOFF.md                  full technical narrative
```

---

## Origin: the Silicon Entropy Engine compressor (archived)

This project began as a **CPU-native lossless compressor** — a streaming mixture of statistical experts blended by an online exponentiated-gradient MoE, shaped by cache topology rather than GPU parallelism. That compressor (V1.0.x, Phases 1–40) is stable and its limits are documented; the token-level and "mantra-pure" eras (Phases 42–54) that followed are the research path that led here. All of it is preserved under the labeled **`archive/`** area as historical / superseded work. The name — *Silicon Entropy Engine* — carried over. See [`CHANGELOG.md`](CHANGELOG.md) for the full phase history.

---

## Citing this work

If this project's findings or design inform your work, please cite it — see [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository" button). Each release is archived on Zenodo with a DOI:

> **DOI (all versions): [10.5281/zenodo.21128459](https://doi.org/10.5281/zenodo.21128459)** — this concept DOI always resolves to the latest release.

The dated commit history is the record of priority.

## License

Code is licensed under the **GNU Affero General Public License v3.0** — see [`LICENSE`](LICENSE). The documentation and figures are intended for reuse under **CC BY 4.0** (attribution required). If you build on this work, the AGPL's network-use clause applies; for a commercial license, contact the author.
