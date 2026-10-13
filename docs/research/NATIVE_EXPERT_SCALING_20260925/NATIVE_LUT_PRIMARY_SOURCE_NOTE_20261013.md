# Primary-source LUT comparison during native correctness run

13 October2026. Focused source lookup,not a reproduced benchmark or full-paper
review. Local selected-p6 artifact and measured CPU remain separate evidence.

[T-MAC paper,Wei et al.,arXiv2407.00088v2](https://arxiv.org/abs/2407.00088v2),
revised25 March2025,describes CPU mixed-precision matmul through bit-wise LUT
computation rather than weight dequantization. Authors report BitNet/Llama
throughputs on other devices;their figures are not our Falcon/CPU results.

[Official microsoft/T-MAC README](https://github.com/microsoft/T-MAC),checked
13 October2026,lists int1/2/3/4 with int8/fp16/fp32 activation interfaces. Its
techniques group one-bit weights,shift/accumulate planes,use tbl/pshuf and
reduce table size through sign symmetry. README-level interface support is
not proof of exact preserved activations/tables in every implementation.

Project inference:bit decomposition/table shape and hardware lookup primitive
deserve explicit comparison after current kernel cost attribution. Our p6
balanced ternary q lies[-364,364],requiring10 signed binary bits if represented
that way. Four-coordinate binary patterns have16 entries;two nibble codes/byte
would7,864,320 code bytes versus current7,557,120. Arithmetic counts only;
more planes/table precision/packing can change cost and error. No CPU rate,
native quality or useful conditional-n claim follows. Do not silently apply
int8 table/activation quantization to the qualified BF16 input boundary.
