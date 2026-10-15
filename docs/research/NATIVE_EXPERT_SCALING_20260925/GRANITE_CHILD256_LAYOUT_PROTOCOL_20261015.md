# Q4_K parent512 to two child256 layout gate

This is an enabling packed-weight transfer check, not the requested projection-bundle cost screen. It does not measure latency, prove a 14 ms batch-1 target, evaluate language quality, or admit an end-to-end decoder. It uses seeded synthetic Q4_K blocks and synthetic activations. It downloads no donor weights and does not run training or GPU code.

## Exact operation

The scoped small-donor geometry is hidden width 1024 and parent expert width 512. Gate and up are 512 × 1024 each; down is 1024 × 512. Q4_K has a 256-input, 144-byte block, so split each parent into **two 256-wide children**. Child 0 receives parent rows 0–255; child 1 receives rows 256–511. For gate/up, copy the original four Q4_K blocks of each row into child-major contiguous rows. For down, copy the first or second original 256-input block of each output row into child-major contiguous 1024-row tensors. The operation neither decodes/requantizes blocks nor pads children. Four 128-wide children would not align with this Q4_K block format.

The fixture has 4096 gate/up blocks (589,824 bytes) and 2048 down blocks (294,912 bytes), or 6144 blocks (884,736 bytes) per parent expert. Child storage has the same total byte count. The child layout changes address order; it cannot itself save stored bytes. A future selector might load one of the two child groups, but selection and compensation are outside this gate.

## Controls and admission

The C fixture fills valid raw Q4_K blocks with seed-dependent positive finite half-precision powers-of-two d/dmin, all twelve packed scale/minimum bytes varied, and seeded nibble codes. It verifies byte-for-byte equality of **every** copied block. Q8_K quantization of a 512-element down activation must produce byte-identical 256-element segment blocks. An independent decoded scalar Q4×Q8 reference computes every gate/up and down output. The active AVX2 path is explicitly selected with `STRAT01_Q4K_Q8K_DIAGNOSTIC_ACTIVE_AVX2`; the wrapper's default dispatch is scalar. All gate/up copied-row AVX outputs must be bit-equal to their parent rows. Full gate/up and down AVX outputs, and the sum of both child down outputs, must be finite with relative L2 error at most 1e-5 against the scalar decoded reference. A complete synthetic SwiGLU control also computes the intermediate activation from copied gate/up outputs, checks its Q8 segments, and compares all reconstructed outputs against independently decoded down weights with parent contribution mass0.375 inherited by BOTH children; the same relative L2<=1e-5 gate applies. A failing block, segment, or full output is a failed gate; no fixture may be discarded after seeing its result.

Only after source review, freeze the four exact inputs (fixture C, controller Python, Q4_K header, this protocol) and compiler binary:

```powershell
python benchmarks/native_expert_scaling/granite_child256_layout.py --freeze results/native_expert_scaling/granite_child256_layout_20261015/freeze.json
```

The frozen controller offers one explicit launch; it verifies hashes before compiling a fresh binary and writes a new exclusive result directory with compile and run exit receipts. The process is single-threaded, has a 120-second compile timeout and 60-second fixture timeout, and only emits a small JSON receipt. No benchmark is launched by preparing this protocol.

```powershell
python benchmarks/native_expert_scaling/granite_child256_layout.py --run --binding results/native_expert_scaling/granite_child256_layout_20261015/freeze.json --output results/native_expert_scaling/granite_child256_layout_20261015/run01
```

Passing means only that synthetic parent blocks can be rearranged and numerically reconstructed at the Q4_K block boundary. It says nothing about source-weight export compatibility, child semantics after choosing only one child, or the necessary full bundle cost including parent routing, child selector, Q8 activation work, attention projections, residuals, and vocabulary head. Those are separate experiments and must be evaluated before an architecture decision.
