## Learning Architectures Designed for Modern CPUs

**Towards 1-bit and 2x2 models built around the clock cycle, throughput, and the granularity of execution units**

### Abstract

This paper proposes a radical reformulation of neural model design—not starting from the architecture of an LLM, but from the operations that can actually be executed efficiently by modern CPUs. The starting point is not the maximum parallelization typical of GPUs, but the internal structure of the CPU: pipelines, decoders, micro-op caches, branch prediction, load/store, SIMD, FMA, bit-manipulation, vector multiplications, fused instructions, and in some architectures, specialized matrix extensions.

The central idea is simple but strong: if the model is designed to naturally "fall" onto the functional units of the CPU, the bottleneck is no longer backpropagation as an abstract concept, but the alignment between the model's algebra and the processor's per-cycle capabilities. From this perspective, a 1-bit or 2x2 block network is not a quantization curiosity, but a potential native form of computation.

The goal of this paper is to build a mathematical language for this approach: first, we describe the machine, then we derive the model's structure.

---

## 1. Introduction

The most common mistake when designing models for CPUs is treating the CPU as a slower GPU. This is not the case. The CPU is a different machine, optimized for:

* Low latency per single thread
* Fine-grained flow control
* Branch prediction
* Out-of-order execution
* Complex and fused instructions
* Memory access with a strong cache hierarchy
* Moderate but highly efficient SIMD
* Selective vectorization
* High-density bitwise and integer operations
* Specific specialized extensions for matrices or blocks

A GPU, on the contrary, benefits from a massive pool of threads and massive throughput on uniform operations. The CPU does not compete on the same level but can be leveraged much better if the problem is formulated in its natural space.

The question is therefore not: "How do I run an LLM on a CPU?"
The right question is: "What form must a model take so the CPU can execute it with high occupancy of its units per clock cycle?"

This changes everything.

---

## 2. The CPU as a System of Per-Cycle Transformations

### 2.1 Clock Cycle, Throughput, and Latency

A CPU does not execute "one operation at a time" in a naive sense. Each clock cycle goes through multiple stages: fetch, decode, rename, dispatch, issue, execute, retire.

For model design, two quantities matter:

* **Latency:** How many cycles are needed before the result of a dependency is available.
* **Throughput:** How many independent operations can be sustained per cycle.

The key point is that a modern CPU can have many units running in parallel, but only if the dependency graph allows it. Therefore, the model must not just be "small": it must be well decomposed.

We define the useful throughput of an operation as:


$$T_{\mathrm{eff}} = \frac{\text{completed useful operations}}{\text{clock cycles}}$$

The goal is not to maximize complexity per token, but to maximize the density of useful work per cycle with minimal dependencies.

### 2.2 The CPU as a Graph of Functional Units

We can model the CPU as a bipartite graph between:

* Issued instructions
* Available functional units

Each instruction is a requirement vector:


$$I_k = (a_k, b_k, m_k, s_k, v_k, f_k)$$

Where, for example:

* $a_k$: Integer arithmetic type
* $b_k$: Bitwise operation
* $m_k$: Memory access
* $s_k$: Shift/rotate
* $v_k$: SIMD vector operation
* $f_k$: Fused multiply-add or composite operation

An efficient core maps the highest possible number of instructions onto different units within the same time frame.

---

## 3. Operations Modern CPUs Do Well

This section does not list "the entire possible ISA," but rather the conceptual building blocks that truly matter for model design.

### 3.1 Scalar Integer Operations

Modern CPUs execute the following extremely well:

* Addition and subtraction
* Integer multiplication
* Comparisons and min/max
* Saturating arithmetic (on some ISAs)
* Logical and arithmetic shifts, rotations
* Bit test, mask, set, clear
* Popcount, leading/trailing zero count, bit scan
* Conditional branches and setcc-like operations

For a 1-bit or 2-bit model, this is crucial: bitwise is not a limitation, it is a primary resource.

### 3.2 Vector SIMD

Modern CPUs support wide register vectors, featuring various families of SIMD instructions. The principle is that a single operation acts on multiple elements in parallel.

Mathematically, instead of:


$$y_i = f(x_i)$$

The following is executed on blocks of components:


$$\mathbf{y} = f(\mathbf{x})$$

This naturally supports:

* Block matrices and grouped tensors
* Convolutions or blockwise products
* Activations applied to groups
* Vector accumulations of errors or scores

### 3.3 FMA and Fused Operations

Fused multiply-add operations are central:


$$z = a \cdot b + c$$


executed in a single conceptual instruction.

The fusion of multiplication and addition is highly significant. A well-built model should produce many expressions reducible to this form, as FMA reduces temporary traffic and better utilizes datapaths.

### 3.4 Load/Store and Cache Hierarchy

No CPU model can ignore memory. The cost is often not the computation, but data movement. Modern CPUs have:

* Registers, L1, and L2 caches
* Often a shared L3 cache
* Hardware prefetch and out-of-order load/store

An efficient model must minimize erratic accesses and prioritize:

* Linear streaming and local reuse
* Small, repeated blocks
* A working set compatible with the cache

### 3.5 Branch Prediction and Flow Control

The CPU executes branches well, but only if they are predictable. A model with too many divergent conditions or chaotic routing degrades throughput. Flow control must be:

* Regular, stable, and locally predictable
* Ideally replaced by masks and arithmetic selections

### 3.6 Specialized Matrix Extensions

Some modern CPUs feature extensions closer to block multiplications, including forms of matrix instructions or tile-based execution. This suggests a crucial direction: the model doesn't necessarily have to be vectorial in the classic sense; it can be tile-algebraic.

Instead of thinking of a weight as a single scalar, it can be viewed as a $2 \times 2$, $4 \times 4$, or small tile that aligns with the execution unit.

---

## 4. The Problem is Not Just the Model: It's the Computational Unit

Most neural architectures are designed around the wrong granularity. Modules are defined in terms of "layers", but the CPU reasons in terms of:

* Instructions and micro-operations
* Port mapping and dependencies
* Hidden latency and pipeline filling

Therefore, the true object of design is a clock-compatible computational unit. We define a computational block $B$ such that it:

* Can be executed with minimal dependencies
* Well-occupies one or more functional units
* Produces enough useful work per cycle
* Requires local and regular memory
* Repeats many times with a similar structure

This definition is far more useful than the classic layer concept.

---

## 5. 1-bit and 2x2 Models as Native CPU Forms

### 5.1 Why 1-bit?

If the system is designed for bitwise operations, the CPU is already in its natural domain. A binary weight is a discrete variable:


$$w_i \in \{-1,+1\}$$


or in boolean form:


$$w_i \in \{0,1\}$$

This enables:

* Masks, AND/OR/XOR, popcount, bit packing
* Fast comparisons
* Accumulation through summing discrete contributions

Multiplying by a bit can be reinterpreted as a selection or negation, which is often cheaper than floating-point multiplication.

### 5.2 Why 2x2?

A $2 \times 2$ matrix is the first non-trivial object where the following emerge:

* Composition and linear transformation
* Local rotation and two-dimensional mixing
* Accumulation of minimal correlations

The fact that many CPUs handle small vector blocks well suggests a hierarchical architecture made of micro-matrices, not massive dense tensors. Instead of a large linear projection:


$$Y = XW$$

A system of blocks is built:


$$Y_{ab} = \sum_{cd} X_{ac} W_{cd}$$


with $W$ decomposed into small, regular tiles.

---

## 6. Throughput Theory Per Model

### 6.1 Definition of Algorithmic Throughput

Let $C$ be the number of available clock cycles and $U$ the number of completed useful operations. The algorithmic throughput is:


$$\tau = \frac{U}{C}$$

The model is good if it maximizes $\tau$ while maintaining adequate accuracy or expressive capacity. However, this is not enough: we must also penalize the hidden costs of memory and control.

We therefore define:


$$\tau_{\mathrm{net}} = \frac{U}{C + \alpha M + \beta B + \gamma D}$$

Where:

* $M$ is the memory cost
* $B$ is the branch/divergence cost
* $D$ is the dependency and stall cost
* $\alpha, \beta, \gamma$ are architectural weights

The ideal architecture minimizes these penalizing terms.

### 6.2 Maximizing Work Per Cycle

A single cycle should execute the highest possible number of independent micro-operations. To achieve this, the model must be decomposable into a set of parallel instructions:


$$\mathcal{I}(t) = \{i_1, i_2, \dots, i_k\}$$


with a low degree of interference.

The structure must therefore be locally dense, globally regular, arithmetic-friendly, cache-friendly, and branch-light.

---

## 7. New Paradigm: From Layer to "Clock Cell"

I propose the concept of the **clock cell**: a computational cell designed to be executed in one clock window or a small number of cycles with high occupancy of available units.

A clock cell should:

* Receive small, contiguous inputs
* Produce compressible output
* Use CPU primitive operations
* Avoid long dependencies
* Be replicable in blocks

An entire network is thus a network of clock cells, not a stack of abstract layers. This aligns much more closely with the logic of a real CPU.

---

## 8. Compositional Operations Suitable for a CPU-Native Model

### 8.1 XOR-like and Mask-based Operations

For binary models, the most natural transformations are often akin to:

* XOR for state composition
* AND for gating
* OR for fusion
* NOT for inversion
* Popcount for measuring similarity

Semantic representation can be built as a combination of these primitives. For instance, a binary similarity between two vectors can be seen as:


$$s(x,y) = \mathrm{popcount}(\mathrm{XNOR}(x,y))$$


or an equivalent form.

### 8.2 2x2 Block Operations

A $2 \times 2$ transformation can represent a local mixing cell:


$$\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

If $a, b, c, d$ are binary or quasi-binary, the execution cost can be kept extremely low, and the network is composed of many local micro-mixers rather than a few massive multipliers.

### 8.3 Accumulative Operations

Many CPU-friendly structures are accumulative:


$$z_{t+1} = z_t + \Delta_t$$

This is perfect for progressive scores, voting, binary counting, evidence integration, and short-memory logic. A model using local accumulation instead of fully dense transformations can better exploit the scalar datapath.

---

## 9. Mathematical Structure of the Proposed Model

### 9.1 State

The internal state is not a giant float vector, but a set of discrete variables and small auxiliary continuous registers:


$$S = (b, u, a, m)$$

Where:

* $b$: Binary state
* $u$: Continuous support latents
* $a$: Accumulators
* $m$: Short-term memory or discrete momentum

### 9.2 Evolution

The evolution per layer/cell is:


$$S_{t+1} = \Phi(S_t, x_t)$$


with $\Phi$ designed to be implementable with CPU primitives. The function $\Phi$ does not necessarily need to be smooth; it just needs to execute well.

### 9.3 Update Decision

A cell can make a decision using a threshold rule:


$$b_i^{t+1} = \operatorname{sign}(\mu_i^t + \rho_i^t)$$

Where:

* $\mu_i^t$ is a signal accumulator
* $\rho_i^t$ is a local corrective term

This structure is fully compatible with integer arithmetic, bitwise logic, and small comparisons.

---

## 10. Adapting LLM Logic to this Machine

Herein lies the most important point: one must not take a standard Transformer and hope it "adapts" to the CPU. It must be rebuilt from the ground up.

Classic LLM structures use large weights, dense matrices, expensive softmaxes, global attention, and heavy floating-point operations. A CPU-native architecture should instead favor:

* Local competition instead of global softmax
* Block routing instead of dense attention
* Compressed memory instead of enormous states
* Discrete updates instead of continuous transformations everywhere
* 1-bit/2x2 compositions instead of massive GEMMs

In practice, the model's semantics must be built from many small operators, each perfectly aligned with the clock cycle.

---

## 11. A Structural Proposal: The Discrete Tile Network

Imagine a network composed of tiles:


$$T_{ij} \in \{-1,+1\}^{2 \times 2}$$

Each tile operates on local micro-states and produces a compressed output. The layer becomes a grid of tiles with local connectivity, summed contributions, binary gating, and context accumulation.

This enables:

* High regularity of data flow
* Low memory pressure
* Strong prefetchability
* Intensive use of SIMD and bit tricks
* Eventual mapping to future matrix extensions

---

## 12. Design Metrics of the CPU-Native Network

Such an architecture should be evaluated using different metrics than the standard ones.

### 12.1 IPC and Useful Work

$$\mathrm{IPC} = \frac{\text{retired instructions}}{\text{cycles}}$$


However, IPC alone is insufficient: we need *useful* instructions, not just *many* instructions.

### 12.2 Arithmetic Intensity vs. Memory Traffic

$$I = \frac{\text{arithmetic operations}}{\text{transferred bytes}}$$


The higher $I$ is, the more the model tends to be compute-bound rather than memory-bound.

### 12.3 Flow Regularity

A CPU-native network must have low structural entropy in its execution paths. In other words: fewer unpredictable branches, fewer sparse accesses, fewer long dependencies, and more structured repetition.

---

## 13. Strong Hypothesis: The CPU as a Discrete Model Accelerator

The most radical thesis is as follows:

The CPU is not merely a compromise for running large models; it can become a natural accelerator for models built on discrete algebra, 2x2 blocks, bitwise logic, accumulators, and regular micro-routines.

In this vision, a well-designed model could exploit the CPU far better than typically thought, leveraging:

* Internal parallelism per cycle
* Flow prediction
* The ability to execute multiple micro-operations per architectural instruction
* Cache locality
* Bit packing and small-block processing

The comparison with the GPU therefore changes: it is no longer about competing with massive throughput, but about using a model where the cost per token is drastically reduced because the computation is inherently compact and close to the hardware.

---

## 14. Conclusion

If we start from pure mathematics, the question is not how to "shrink" an LLM for a CPU. The question is: *What model theory emerges directly from a modern CPU?*

The answer proposed by this paper is:

* Use small discrete units, 1-bit or 2x2.
* Prioritize operations that the CPU can issue and sustain well per cycle.
* Maximize useful throughput, not just model size.
* Reduce dependencies, unpredictable branches, and memory traffic.
* Structure the network as a collection of clock cells and local tiles.
* Replace the classic transformer grammar with a grammar of discrete algebra, accumulation, and regular blocks.

The final perspective is clear: the CPU must not chase the GPU. It must impose its own shape on the model.

---

### Appendix A — Design Outline

A possible conceptual pipeline is:

* Discrete input or token
* Compact binary encoding
* Propagation via 2x2 tiles
* Local accumulation
* Threshold decision
* Block compression/expansion
* Discrete short-term memory
* Final output via voting or score

### Appendix B — Open Research Questions

* What is the optimal granularity of a tile to maximize effective IPC?
* Can a 1-bit model maintain enough expressiveness using only accumulators and 2x2 blocks?
* How much do branch prediction and cache locality matter compared to single-operator complexity?
* Is it possible to define a training theory directly on micro-operations rather than continuous gradients?
* What compromise between discretion and stability produces the best throughput per token?

### Appendix C — Central Thesis in One Sentence

The best form of a CPU model is not a miniaturized GPU-style transformer, but a discrete system built to optimally occupy the clock cycle, cache, and functional units of the CPU.