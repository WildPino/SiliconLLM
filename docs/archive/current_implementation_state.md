# Current Implementation State

## 1. Overview
The **SiliconLLM** project is a novel CPU-native approach to sequential modeling, abandoning the traditional Attention-based Transformer architecture in favor of a **Persistent Wave Engine**. This engine aims to achieve competitive bits-per-byte (BPB) compression and language modeling using physical wave dynamics, random binary codebooks, and thermodynamic damping, heavily accelerated via AVX2 vector instructions.

## 2. Core Components

### 2.1 Silicon Sequence Compressor V0 (`silicon_v0.c`)
This is the foundational engine of the architecture:
- **Spatial State**: A 128-block AVX2 state (`__m256i state[128]`).
- **Codebook**: A random binary single-block codebook with 256 entries.
- **Historical M4 Buffer**: A circular history buffer storing recent tokens.
- **Wave Dynamics**: Employs discrete topology shift-window injection, thermodynamic damping (exponential decay), and path integration (wave dynamics) to process sequences efficiently.

### 2.2 Silicon Entropy Engine (`silicon_entropy.c`)
An evolution of the base engine designed specifically to predict the next byte by maintaining a multi-level state:
- **L0 State**: Represents immediate, fast-changing local context (32D features extracted from historical M4 and V0 state).
- **L1 State**: Represents slower, chunk-averaged context, updated via Exponential Moving Average (EMA).
- **Feature Extraction**: Extracts a 192-dimensional vector (SEE_FEATURE_DIM) that concatenates the L0 state and the L1 state.

## 3. Benchmark Progression
The project relies on empirical evaluation to iteratively refine the architecture. The current sequence of benchmarks (`benchmarks/` directory) includes:

- **Benchmark 10 (Audit)**: Auditing and validation.
- **Benchmark 11 (Generalization)**: Tests how the engine generalizes across data distributions.
- **Benchmark 12 (Hybrid)**: Hybrid modeling experiments.
- **Benchmark 13 (Entropy)**: Deep dive into optimal history size (T=4 was found optimal) and feature dimensionality.
- **Benchmark 14 (Hierarchy/Bottleneck)**: Focuses on extracting hierarchical representations and understanding the information bottlenecks of the engine.

## 4. Output and Data Management
- **`data/`**: Contains raw corpora such as `natural_text.txt`, C source samples, and shuffled datasets used for evaluation.
- **`logs/`**: Detailed run logs, training logs, and audit results are grouped here.
- **`results/`**: Organized by phases (`phase11`, `phase12`, `phase13`, `phase14`, etc.). Contains detailed text predictions, BPB scores, and generated markdown logs from each experimental run.

## 5. Architectural Paradigms
- **CPU-Native First**: Every mathematical operation is optimized for x86 CPUs using `_mm256_*` AVX2 instructions.
- **No Floating Point in the Core**: The core wave propagation and state updates use 8-bit or 16-bit integers (`__m256i`). Floating-point math is deferred entirely to the readout/prediction layer (Logistic Regression).
- **Residual Connections**: Readout models typically employ a baseline n-gram (unigram/bigram) model and use the Wave Engine's features as a residual signal, allowing the network to focus on long-range dependencies.
