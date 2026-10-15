# Exact expert child layout: result

15 October2026. **PASS for the synthetic packed transfer control only.**
Source/control frozen in commit2cb0b63; no donor weights, training or GPU.

All6144 Q4_K blocks of a512-wide expert were copied byte-for-byte into two
contiguous256-wide children. Stored expert payload remains884,736 bytes;
selecting one child would address442,368 expert-weight bytes, before selector,
activations, masses and other model organs. This is not measured DRAM traffic.
The existing explicitly enabled AVX2 path passed independently decoded scalar
controls. Relative L2 errors: gate/up6.3683e-8; down7.2513e-8; child down
reconstruction6.8947e-8; complete synthetic SwiGLU with inherited parent
mass0.375 on BOTH children5.5787e-8. Every output is finite. Q8 intermediate
segments and copied-row outputs are bit-identical. All packed scale/minimum
bytes vary; original Q4 blocks are not requantized.

Compiler exited0 in2.938s; fixture exited0 in0.109s. Passing establishes the
format boundary and reconstruction arithmetic on this deterministic fixture.
It does not establish source quantization quality, quality after selecting one
child, a cheap selector, full model cost, native generation or large-family speed.
No timing/rate admission is attached to a0.109s test duration.

[Binding](granite_child256_layout_binding_20261015.json),
[compile/numeric receipt](granite_child256_layout_result_20261015.json),
[protocol](GRANITE_CHILD256_LAYOUT_PROTOCOL_20261015.md).

Next discriminating step: charge the complete routed projection/FFN/head bundle
with actual packed experts, parent/child IDs and masses. Large-family protected
core/head/router feasibility is independently required before a substantial
conversion pilot. Do not tune selectors against DEV or assume every retained
coefficient contributes useful capacity after sparse selection.
