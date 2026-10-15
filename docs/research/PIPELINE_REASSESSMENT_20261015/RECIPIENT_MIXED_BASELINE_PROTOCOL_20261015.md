# Complete mixed-precision recipient: prospective physical comparison

15 October 2026. Freeze before conversion or model execution. This follows the
[complete I8 rejection](RECIPIENT_INT8_BASELINE_RESULT_20261015.md) and the
[core triage](RECIPIENT_CORE_TRIAGE_20261015.md). Months of T4 recovery are
acceptable under the owner's resource clarification; this bounded CPU test
decides whether the unchanged full-width recipient has enough physical room
before adding conditional donor memory. It does not select a small final donor.

## Fixed computation and encoding

Use the same original Qwen2.5-0.5B-Instruct snapshot and SHA256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
Preserve all 24 layers, D896, FFN4864, 14 query / 2 KV heads and all 151936
vocabulary IDs. Source embeddings stay BF16; biases and norms widen exactly
to F32. Explicit attention/head matrices use signed row-W8 with F32 scales;
FFN gate/up/down use symmetric W4 groups of 32 input columns with FP16 scales.
Nearest-even rounding uses the decoded FP16 scale in W4. The complete packet
has 291 named roles, independently checked against 290 original BF16 tensors.
Its catalog is compiled into `engine.c` with `SILICON_QWEN_RECIPIENT_ENTRY`.

Unlike the previous I8/F32-activation runtime, this profile quantizes each
W8 activation vector to signed I8 and each W4 activation group of 32 to signed
I8. Integer dots are exact before F32 output rounding. W4's AVX2 unsigned-nibble
multiply has pair sums bounded by 3810, safely below saturation, followed by
the exact signed-zero correction. BF16 group scales decode via F16C. Attention
uses complete causal F32 KV, double dot/value accumulation and six physical
workers; SIMD association may change rounding. RoPE, RMSNorm and SiLU remain.

The full W8/A8 head screens every ID, then refines the eight highest scores
against original BF16 embedding rows. Refined logits replace those eight
entries; all remaining approximate entries remain eligible in final argmax.
This definition is checked explicitly; it is not exact BF16 head equivalence
or a sub-millisecond readout claim. No fixed vocabulary shortlist, bank,
removed layer, skipped source operator, fitted transformation or GPU is used.

## Independent numerical gate

Bind the same retained canonical prompt IDs, context cost fixtures, source,
CPU implementation dependencies and compiler/runtime identities as the prior
baseline, plus every new source/protocol. No reserved quality cohort is opened.
Prepare a 32-token prefix request with output positions 0, 7, 15 and 31. The
native probe stores complete normalized states and all vocabulary logits at
those positions. Its forward must still process all preceding tokens.

The independent auditor verifies the entire packet, every source role and
scale/code derivation, padding, sizes and SHA256, without importing exporter
or native numerical routines. An explicit sequential CPU reference includes
every activation quantizer, all causal operators and head refinement.
Require finite outputs, exact top1, relative L2 <= 1e-3 for both normalized
hidden state and whole vocabulary logits at every selected position. This
checks decoded-wire execution, not BF16-donor quality. Shared NumPy/Torch
primitives are a declared dependency, not independently verified libraries.
Any numerical failure stops before timing and retains its first fault.

## Complete cost observations

Use six distinct physical cores, logical 0,2,4,6,8,10, ACTIVE/infinite wait,
fixed compile flags including FP contraction off. Native start/end affinity
readbacks must pass. Run three fresh-process repetitions of the same five
cases: three engineering chat prompts and repeated-prose contexts 512/2048,
each requesting up to 64 greedy own-history tokens, stopping on either EOS.
Each non-EOS output, including the final one, is actually forwarded and
charged. EOS scans are timed without inventing a forward. Prefill, process
wall, artifact verification/load and per-step decode are reported separately.
Prefill may omit unused intermediate heads, but its last-token head is full;
this difference from the I8 baseline must remain visible.

Keep all 15 observations even if cost fails early. The unchanged prospective
allowance is worst-process mean <=18 ms per charged native forward and
max/min rate spread <=1.10 in every case group. Repeat IDs must match within
each group. A PASS licenses a bounded memory-fusion study, not final quality
or useful-50 admission. A FAIL closes this fixed complete encoding/kernel
envelope; it does not establish a universal lower bound or forbid long
training of a different conditional geometry. No thread/wait/quantizer grid.

Matrix payload is analytically 357484032 bytes per forward including scales,
versus 495785472 for I8, a reduction of only 1.387x. BF16 embedding lookups,
norm/bias reads, activation/KV traffic, eight refinement rows and arithmetic
remain additional. Addressed payload is not PMU-measured physical DRAM.

## Resources and receipts

No downloads, T4 allocation or model training. Exclusive logs, requests,
binding, catalog, conversion manifest, owned PID/create-time and terminal
receipts are retained. Cap each child at 900 seconds and 8 GiB observed RSS,
the family at 2400 seconds and 2 GiB outputs. Terminate only an owned process
on cap/failure. No automatic numerical retry. All pinned inputs and artifact
hashes are verified again at closure. This bounded inquiry does not trigger
the owner's >=1-hour long-run pause rule.
