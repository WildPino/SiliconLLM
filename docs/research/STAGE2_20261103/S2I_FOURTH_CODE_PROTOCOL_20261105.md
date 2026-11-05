# Stage 2: the fourth code, step 1. A test of mechanism. PROTOCOL

2026-11-05. For stage 2, part 2. CPU, about 45 minutes beside the other work. No training. Step 2 waits for the go.

## Question

IQ2_BN spends 2 bits per weight and uses three values. At equal bytes, how much does the output error fall if the fourth is used?

## Thesis line (rule 4)

The bytes do not change, so neither does the speed on the desk. A pass does not bring the thesis closer by itself: it says the
form part 2 should train, at flash's width only by weights. Post-hoc error is not panel quality: test 3 showed a large raw error
the norms remove. So this is a test of mechanism with a mechanism budget.

## Literature (read in the papers)

| Source | What it gives here | Known ways to fail |
| --- | --- | --- |
| ParetoQ, arXiv 2502.02631, Table 1, §3.2.1 | LLaMA-3 8B, per channel, after quantization-aware training: 2 bits avg 71.2, Wiki2 8.0; 1.58 bits 69.0, 8.6. SEQ's 2-bit grid is (−1.5, −0.5, 0.5, 1.5), no zero, against a grid with one positive level | the numbers follow training; Figure 5 favours a zero at 3 bits and above; nothing here is post-hoc |
| SpinQuant, arXiv 2405.16406, §3.1, §4.2 | R2, a rotation per head on V and its transpose on the attention output, absorbed: no new parameters, no kernel support | the rotation is optimized for activations; weight error is left to GPTQ |

**The kernel, read:** `dequantize_row_iq2_bn` gives code − 1. `mul_mat_iq2bn_q8_K64` multiplies the unsigned code and subtracts
the activations' sum. So code 3 should read +2. `quantize_one_row_2bn` writes 0, 1, 2 only, from a threshold at half the row's
maximum: it has to be replaced. The row scale is a float, so it can carry a sign.

## Design (`s2q_code.py`, `s2q_parity.c`, `s2q_verdict.py`)

**Forms,** one float scale per row, 2 bits per weight:
- **tern (reference):** {−1, 0, 1}, the optimal threshold, by exact search on the sorted row;
- **asym:** {−1, 0, 1, 2} with a sign per row, Lloyd, the better sign kept;
- **bal:** {−1.5, −0.5, 0.5, 1.5}, Lloyd; in the kernel, code − 1 plus −0.5·s·sum(x) per row;
- **rot_\*, attention only:** the same after a Hadamard rotation with random signs per KV head, V's rows and the output
  projection's columns, in the weights.

**Where:** layers 5, 10, 15 of mini. Attention, shared expert, routed experts (all 256), one organ at a time. Input: the first
8,192 held-out positions of the training corpus's split, in test 2c's teacher stream. No panel uses them.

**Primary:** test 2c's block loss, mean of (nrm(o) − nrm(t))²·gw, in float. Reported: plain relative output error, weight error.

**Flash:** the BF16 GGUF is on disk, so the same layers are read: attention, shared expert, 16 experts. Weights only, reported.

**Kernel parity:** rows written by hand, IQ2_BN and IQ2_BN_R4, five shapes, batches 1, 4, 9 (decode and the wider paths), against
a double dot product.

## Found in a dry run (layer 10, 512 positions, before registration)

| Layer 10 | tern | asym | bal |
| --- | ---: | ---: | ---: |
| Attention: weight error / block | 0.199 / 0.504 | 0.152 / 0.416 | 0.127 / **0.680** |
| Shared: weight / block | 0.250 / 0.314 | 0.191 / 0.266 | 0.185 / 0.281 |
| Experts: weight / block | 0.194 / 0.260 | 0.147 / 0.193 | 0.122 / 0.159 |

On attention bal has the least weight error and the worst block. Its likely cause: with no zero, small weights on the inputs'
large channels become ±0.5·s. Rotation moved tern's block from 0.504 to 0.482. Parity, dry: code 3 reads +2 on every case
(relative error 0.006-0.009, the same as codes 0-2; the planted miss 0.51-0.63).

## Line (registered; `s2q_verdict.py`, selftest PASS, 7 cases)

Per organ, G = 1 − mean over layers of (block of the form / block of tern), for asym and bal. An organ passes if its better form
has G ≥ 0.25: two thirds of the 0.375 that theory gives a Gaussian at equal bytes.
- **THE FOURTH CODE PAYS:** all three organs pass, and parity holds.
- **THE FOURTH CODE PAYS ON SOME:** at least one passes, and parity holds.
- **NOT AT THIS ERROR:** none passes.
- **THE KERNEL DOES NOT READ IT:** an organ passes, and parity fails.

**Parity holds:** relative error of four, neg and bal at most 1.5× that of tern on every case, and the planted miss at least 10×.

## Controls

| Control | Rule | Failing means |
| --- | --- | --- |
| Trivial: Gaussian weights of equal row variance, on the same inputs, on three tensors per layer | weight gain of bal in 0.33-0.42; the output gain reported beside the real one | VOID; a real gain equal to the twin's is no more than any bell shape gives |
| Planted: weights on each form's grid plus noise of 5% of the step | the form's own quantizer recovers ≥ 99.9% of the weights, error ≤ 1.05× the noise | VOID |
| Identity: rotated weights, not quantized | block ≤ 1e-6 | VOID |
| Planted miss in the parity: code 3 read as +1 | at least 10× off | parity fails |

The block loss is test 2c's, already in use, so no new primary metric enters and rule 6 does not apply.

## Predictions

| | Predicted |
| --- | --- |
| G attention | best form asym, 0.12-0.22 |
| G shared | best form asym or bal, 0.10-0.25 |
| G experts | best form bal, 0.30-0.42 |
| Rotation, block gain on attention | 0.02-0.10 |
| Flash, weight gain of bal | 0.30-0.40 |
| Outcome | ON SOME 80%, PAYS 12%, NOT 8% |

These follow the dry run.

## What a result changes (rule 5)

- **PAYS:** step 2 trains four codes against ternary at equal budget, and part 2's format changes.
- **ON SOME:** step 2 trains the fourth code on those organs only; the others stay ternary. Experts are where it would matter:
  15.3 B of the weights.
- **NOT AT THIS ERROR:** part 2 stays ternary, and the kernel stays as it is.
- **KERNEL:** step 2 waits for a fix to the kernel, which is outside the stage.

## Parts specific to Ling

- **Specific:** the numbers; the large input channels behind attention's result; 4 KV heads for the rotation.
- **Not specific:** the grids, the kernel's reading of code 3, the method.

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| Training | none | step 1 as asked; ParetoQ's numbers come after training, so G is only a first reading |
| Flash in output error | weights only | no flash activations are on disk; a dump needs flash loaded for long, alone. Mini's weight and output columns give the bridge |
| An input-weighted quantizer | plain row error | same quantizer as test 2c; a weighted one might rescue bal on attention, and would be step 2's |
| All layers, stacked | three layers, one organ at a time | step 1; test 2c measures stacking |
| The quantizer in the engine's format | float | parity covers the format; the writer of the bytes comes with step 2 |

## Cost

About 45 minutes of CPU on 4 threads, beside the chains. Parity: under a minute.
