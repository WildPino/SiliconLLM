# Evidence register — 10B roadmap

Snapshot: 2026-09-16, repository HEAD `f1f9cfd59d34071a004248bc04c9d56e6038c596`. Documentary research: no experiments were run. Local results are read from canonical reports, not reproduced. The following IDs are stable within this dossier. External sources are primary sources; their results are not SiliconLLM benchmarks.

| ID | Source and locator | Evidence used | Limitation / status |
|---|---|---|---|
| L01 | [engine.c](../../../benchmarks/phase60/engine.c), lines 52–75, 189–245, 921–934 | Fixed native dimensions; E1M1/E4M1; loader retains fp32 matrices and ternary representations; donor-shape is a synthetic harness | Code reading; not a universal donor runtime |
| L02 | [donor_engine.c](../../../benchmarks/donor_adaptation/engine/donor_engine.c), header and loader | Separate Transformer runtime, with RoPE/KV and donor formats | Do not conflate execution on this branch with SSM conversion |
| L03 | [PHASE64_BUDGET](../../PHASE64_BUDGET.md), §§1–4 | Aggregate DRAM 40–44 GB/s; resident projections up to about 185 GB/s; experts 4.2 GB/s integrated versus 17 kernel-pure; about 8.4 µs/expert overhead | Measurements on specific shapes/protocols, not 10B constants |
| L04 | [SCALEUP_ARCHITECTURE](../../SCALEUP_ARCHITECTURE.md), §§3.8, 5.1, 9 | Mamba1, SWA, MoE; near-i.i.d. routing unions; conditional block-verify; data ordering affects training | Spec v1 and small native measurements; historical parts superseded by donor probes |
| L05 | [ENGINE_PLAN](../../ENGINE_PLAN.md), principles and status | Parity-first, portability, no fast-math; ternary projections P61 +0.018–0.022 BPB | Not bit-exact relative to the teacher after lossy transformations |
| L06 | [README](../../../README.md), results §§1–8 | +0.00004 BPB is the cost of inference optimizations; joint ternary+dReLU recipe cost about +0.013±0.005 in the sandbox; MVE KD loses to CE off-domain | These three quantities are not interchangeable |
| L07 | [P1 nibble packing](../donor_adaptation/probes/P1_NIBBLE_PACKING.md), Stage 1 and Stage 2, line 270 | 2 bit/weight option already present, parity; 1.58–1.91× matvec/s t6 on donor shapes with aliasing control | Not 1.58-bit, not an end-to-end multiplier |
| L08 | [E36](../donor_adaptation/probes/E36_TEN_BILLION_AT_FIFTY.md), §§1–8 | 9,999,220,736-parameter artifact; 49.96 tok/s recorded, FFN 1.17% active | Synthetic weights, 40-token bench, no quality |
| L09 | [E40](../donor_adaptation/probes/E40_ATTENTION_LEVERS_EXHAUSTED.md), opening and R128 table | Synthetic R128 with active FFN: 112.73/106.44 tok/s; modified geometry | Noise, high variance, no pretrained 10B |
| L10 | [E63](../donor_adaptation/probes/E63_THE_TEN_BILLION_CELL_AT_ONE_BYTE.md), §16, line 440 | A10B mixed-format 49.37 tok/s, CI [49.18,50.49]; A10B closed, ancillary cells incomplete | Int8-carved FFN only; not an all-int8 model; do not repeat the gate |
| L11 | [Corrected screen](../donor_adaptation/audits/TARGET_DONOR_FOLLOWUP_2026-09-16.md), opening and table | E63: 517,210,112 byte charged/token; 106,168,320 of 928,251,904 weights charged int8; OLMoE and other screening | 45.828 Gweights/s does not transfer to full-int8 |
| L12 | [E66](../donor_adaptation/probes/E66_ONE_BYTE_AT_SEVEN_BILLION.md), A2 table and gate | Real 7B donor, int8 layer/ternary head: 0.674404586411 BPB, +0.000378031411; 137/160 agreement against gate 150 | Quality/rank, no throughput stated; rank gate failed |
| L13 | [E64](../donor_adaptation/probes/E64_CARVE_ON_INT8.md), run 2 | Post-hoc k3 int8 FFN carve 4.128510 BPB, above chance 4.069819 | Mixed baseline; not an all-int8 model; quality-only |
| L14 | [H4](../donor_adaptation/probes/H4_AGGRESSIVE_RANK_STEP_ZERO.md), §§10–11 | rank48 q/o training: BPB 2.8035765→1.1537375; free 5→6/160, TF 21→66/160, 2.8h on one T4 | Score recovery, generation at the floor; fp32 masters do not rescue |
| L15 | [H5](../donor_adaptation/probes/H5_CROSS_COMPOSITION_RESULT.md) | Frozen H4+H2I: BPB 1.1537375→1.3991680; free 6→1/160 | Closes post-hoc shortcut; not joint training |
| L16 | [E67](../donor_adaptation/probes/E67_OUTPUT_AWARE_SELECTION.md) | Output-greedy selector improves local SSE by only 4.23% | Not a global oracle, not BPB, not speed |
| L17 | [E68](../donor_adaptation/probes/E68_SHARED_RESIDUAL.md) | Shared rank64 residual: SSE ratio 0.46674069016; layer1 worsens 1.771×; layer27 dominates | PARTIAL_LOCAL_SIGNAL; selection uses dense-z oracle; not an executable router |
| L18 | [H2I](../donor_adaptation/probes/H2I_ONE_BYTE_CARVE_BASELINE.md) and [INDEX](../donor_adaptation/INDEX.md), initial 2026-09-16 update | Trained one-byte score improves, TF95/160 fails strict >99; terminal SCORE-ONLY | Does not authorize export or rerun of the same object |
| L19 | [Axis map](../donor_adaptation/audits/AXIS_COVERAGE_AND_NO_DUPLICATION_MAP.md) | Coverage, scope, and priority of canonical sources | Index, not independent evidence |
| L20 | [T3](../donor_adaptation/probes/T3_ROTATION.md), §§0–1.3 | Rotation experiment formally VOID due to preregistered control | Do not cite as conclusive proof that every rotation fails |
| L21 | [Training plan](../../PHASE64_TRAINING_PLAN.md), §13 | MVE 3860 tok/s T4×2, DDP1.80×; CE-primary; FP16 forward overflow can block training | Small shape; not throughput for 10B |
| L22 | [E65](../donor_adaptation/probes/E65_RANK_FRACTION_COST.md), r/D table | Q/O rank1/3: +0.052689 BPB; rank1/32: +1.705835 on the donor1.5B | No transfer law to target width |
| L23 | [H2I export gap](../donor_adaptation/audits/H2I_ENGINE_EXPORT_GAP_AUDIT.md), §§format/export | Tagged-v2 can represent operators that the current exporter does not emit exactly | H2I fails the composite gate: do not automatically implement export |
| L24 | [E59](../donor_adaptation/probes/E59_THE_FAST_KERNEL_DOES_NOT_SURVIVE.md), opening | LUTBLK accelerates the studied path but changes many tokens through activation quantization | Not a lossless kernel; speed and quality must be measured together |
| W01 | [SVD-LLM, Wang et al., 2024](https://arxiv.org/abs/2403.07378), abstract | Whitening and truncation-aware low rank | Does not demonstrate rank1/32 at near-zero loss |
| W02 | [QuIP#, Tseng et al., 2024](https://arxiv.org/abs/2402.04396), abstract | Hadamard for incoherence, lattice VQ, and fine-tuning | Not diagonalization, not free ternarization |
| W03 | [AQLM, Egiazarian et al., 2024](https://arxiv.org/abs/2401.06118), abstract | Input-adaptive additive codebooks and joint block optimization | CPU-specific layout/codebooks, not the current ternary kernel |
| W04 | [Speculative decoding, Leviathan et al., 2023](https://arxiv.org/abs/2211.17192), abstract | Verification with preserved distribution | Published speedup does not transfer to this CPU/MoE |
| W05 | [T-MAC, Wei et al., 2024/2025](https://arxiv.org/abs/2407.00088), abstract | LUT for low-bit weights on CPU | Different hardware, shapes, and activations |
| W06 | [SparseGPT, Frantar and Alistarh, 2023](https://arxiv.org/abs/2301.00774), abstract | Unstructured/semi-structured pruning with reconstruction | Does not demonstrate 99% sparsity or automatic AVX2 acceleration |
| W07 | [SmoothQuant, Xiao et al., 2023](https://arxiv.org/abs/2211.10438), abstract | Equivalent rescaling to manage W8A8 outliers | No VNNI: quality is possible, speedup must be measured |
| W08 | [Mamba, Gu and Dao, 2023](https://arxiv.org/abs/2312.07378), abstract | Selective state-space | Not an exact algebraic conversion from attention |
| W09 | [IBM Granite H Tiny Base](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base), model card | Donor7B/~1B active, hybrid Mamba2/attention/MoE, Apache2 | Official metadata, not a local measurement or target10B |
| W10 | [Liquid LFM2.5 8B A1B Base](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base), model card | 8.3B total/1.5B active, conv+GQA, specific license | Neither Mamba1 nor drop-in for the runtime |
| W11 | [MOHAWK](https://arxiv.org/abs/2408.10189), paper | Progressive Transformer→SSM distillation | Not joint ternarization+extreme sparsity |
| W12 | [Mamba in the Llama](https://arxiv.org/abs/2408.15237), paper | Weight reuse and progressive hybrid conversion | GPU results and benchmarks differ from the BPB gate |
| W13 | [BitNet2B4T](https://arxiv.org/html/2504.12285v2), technical report | Native ternary from scratch, 2B on 4T tokens | Not an economical conversion of pretrained10B |
| W14 | [Sparse Upcycling](https://arxiv.org/abs/2212.05055), paper | Dense checkpoints as MoE initialization | Replication does not freely create distinct capacity |
| W15 | [NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/), specifications | 16GB and 65TFLOPSFP16 peak | Theoretical peak, not training throughput |
| W16 | [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus), [CUDA types](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html) | T4/Turing CC7.5, no native hardware BF16 | Consulted during parallel research; FP16 requires numerical management |

Additional donor and conversion bibliography is listed in the roadmap with primary links. No external source jointly demonstrates pretrained 10B, this architecture, 50–100 tok/s, and ΔBPB≤0.02. Formulas and budgets proposed in the report are explicitly labeled derivations or design choices.
