# MoE fidelity and native low-bit inference: evidence lane

Cutoff: 2026-10-06. This lane contains 13 primary-paper clusters and 16 claim records. Versions, conference publications and author repositories belonging to the same project count once. Only sources available by the cutoff support conclusions. Paper text, official repository documentation and selected QMoE source functions were inspected; no quantization, model, GPU, native timing or Kaggle experiment was executed.

## Main implications for the research decision

A negative outcome is scientifically useful when the conclusion names the method family, representation, model, task, evaluation distribution, storage budget and available compute. The literature does not establish that every pretrained floating model can become fully ternary while preserving its original function under the present stringent gates. Conversely, failure of the current symmetric group-64 screens does not exclude other alphabets, calibration conventions, router adaptation or mixed precision. These are separate hypotheses, not permission to redefine failed preregistrations.

The strongest directly relevant comparison is MN01, especially MN-E01–03. A controlled follow-up should isolate representation and calibration effects without changing the original teacher or routing semantics. Capacity changes belong in a separately named reproduction arm. Numeric reconstruction and masked-span task loss answer different questions; one cannot translate published loss tolerances into the present expert RMS or output agreement gates. The paper and exact implementation are linked in the registry.

There are independent MoE-specific reasons to distrust a calibration-only success: coverage, affinity, heterogeneous sensitivity and altered downstream routing are investigated in MN03–07 and MN13. Treat an expert's absence as missing evidence. Expert-balanced calibration may help fit quality, but evaluation must retain natural routing and independently selected documents. Log which experts lack adequate unique documents, not only token counts. Generated or oversampled calibration is an admissible future intervention only after its distribution and budget are frozen; it cannot retroactively supply a missing held-out sample.

MN13 motivates progressive router adaptation as a distinct method family. Its average integer-bit budget must not be relabelled a ternary alphabet. Candidate selection and model-wide task gradients also create a different optimization cost from local sequential expert reconstruction. If explored, preserve fixed-routing and adapted-routing arms and report both route disagreement and task loss. An unchanged router weight matrix alone does not imply unchanged routing when previous hidden states change.

Actual low-bit execution is a separate empirical question. MN08–09 motivate comparing packed decode, LUT setup, SIMD accumulation and numerical fidelity as a complete implementation. A compact checkpoint cannot supply a timing result. Benchmarks should match expert shapes and row counts, include group scales and any output residual, and measure warm and cold regimes under the owner's resource reservation. CPU measurements must report compiler, ISA, thread count, affinity, caches and all setup work. Keep kernels with altered activation arithmetic or lossy LUT operations separate from the qualified exact packed implementation.

MN12 is relevant to the deployment architecture but cannot certify the all-ternary objective. For any hybrid offloading study, count all high- and low-precision copies, model core, caches, staging buffers and transfers. A reduction in GPU residency can coexist with greater total host storage. Compare end-to-end execution under identical residency and memory envelopes, not an offloading baseline that repeatedly reads large uncompressed weights.

## Unresolved questions with concrete scientific value

1. Under original capacity and untouched natural held-out routing, does an asymmetric two-level-plus-zero expert representation improve the storage–quality frontier relative to the frozen symmetric implementation? Count both nonzero levels and all deployment metadata.
2. Does calibration-only special-token exclusion improve held-out ordinary-token reconstruction or masked-span loss without harming sentinel/EOS prediction? Freeze token categories before fitting and preserve the existing all-row result.
3. After a meaningful local expert improvement, does a full model intervention transfer across naturally routed modules and tasks? Teacher contexts alone do not resolve accumulated hidden-state or router changes.
4. Can a compact residual meet the total deployed storage gate and still reduce complete native latency? Its floating computation, memory loads and intermediate buffer belong in the measured path.
5. If a mixed-bit/router-adapted result works while all-ternary fails, what proportion of higher precision is essential? Such a result is an explicit relaxation, useful for native-expert-scaling but different from success on the original goal.

## Proposed scoped stopping criteria

Stop a candidate family after audited preregistered controls establish arithmetic correctness, sufficiently covered natural held-out comparisons fail absolute quality or full storage gates, and one literature-motivated bounded correction fails to change that conclusion. Report confidence intervals, sample coverage and cost. A collection failure or insufficient coverage is an inconclusive experiment, not a quality-negative result.

Stop native optimization of an implementation when exact arithmetic passes but complete uncontended timing, including scale/residual/setup work, fails the practical speed gate in the intended cache and row-count regimes. State the CPU and kernel configuration; do not infer an ISA-independent lower bound.

Close the overarching study with a negative result once the tested families and meaningful alternatives are explicitly bounded, completed results are independently audited, source gaps are catalogued, and the evidence explains why further candidates are not justified by the remaining resource budget. The conclusion should say that no admitted method satisfied the joint gates within the investigated scope, rather than asserting mathematical impossibility.

## Source map

- [MN01: QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795v1) — 2310.16795v1, 2023-10-25.
- [MN02: Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness](https://arxiv.org/abs/2310.02410v1) — 2310.02410v1, 2023-10-03.
- [MN03: QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts](https://arxiv.org/abs/2406.08155v2) — 2406.08155v2, 2025-02-25.
- [MN04: MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance](https://arxiv.org/abs/2505.03804v1) — 2505.03804v1, 2025-05-02.
- [MN05: Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270v2) — 2410.06270v2, 2025-02-22.
- [MN06: EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization](https://arxiv.org/abs/2506.13329v1) — 2506.13329v1, 2025-06-16.
- [MN07: AlphaQ: Calibration-Free Bit Allocation for Mixture-of-Experts Quantization](https://arxiv.org/abs/2606.04980v1) — 2606.04980v1, 2026-06-03.
- [MN08: T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge](https://arxiv.org/abs/2407.00088v2) — 2407.00088v2, 2025-03-25.
- [MN09: Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://aclanthology.org/2025.acl-long.457/) — ACL2025, 2025-07.
- [MN10: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961v3) — 2101.03961v3, 2022-06-16.
- [MN11: Mixtral of Experts](https://arxiv.org/abs/2401.04088v1) — 2401.04088v1, 2024-01-08.
- [MN12: HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/abs/2411.01433v2) — 2411.01433v2, 2024-11-06.
- [MN13: GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](https://arxiv.org/abs/2605.23078v1) — 2605.23078v1, 2026-05-21.

## Method and limitations

The searches covered pretrained MoE quantization, low-bit failure, expert calibration imbalance, mixed-bit allocation, routing sensitivity, progressive allocation, native LUT kernels and CPU–GPU offloading. Retrieval used primary paper HTML/PDF, author repositories and public repository commit metadata. Stable source IDs are SHA-256 of canonical URLs; that hash identifies a URL, not a retained source body. Selected official QMoE code and T-MAC README body hashes were independently computed from retrieved UTF-8 text; raw bodies are not saved in this lane.

Publication aggregates, BLEU/perplexity and benchmark accuracy are not original-function agreement measurements. Reported hardware and versions differ materially from the Ryzen 5 3600X, RTX3060 and individual T4 devices. Author measurements were not independently reproduced. EAQuant's author-linked repository retrieval failed, so no implementation-level claim is assigned to it. HOBBIT lacks an inspected author implementation in this lane. These limitations are preserved in the source ledger rather than filled with assumptions.
