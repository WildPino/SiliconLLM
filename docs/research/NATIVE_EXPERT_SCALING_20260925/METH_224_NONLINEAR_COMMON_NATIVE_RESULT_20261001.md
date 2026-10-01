# METH-224: H768 BF16/FP32 common is numerically and natively feasible

The [frozen protocol](METH_224_NONLINEAR_COMMON_NATIVE_PROTOCOL_20261001.md)
and export/oracle/C kernel were committed at `093ff1f`. All72 actual
BF16 donor row/column subsets and24 zero FP32 biases read back exactly.
The99,176,472-byte fixture contains24 distinct donor-layer subsets,
not repeated small weights; original source tensor hashes are retained.
These arbitrary first768 units are a component fixture, not a quality
selection or revived hard-carve model.

| Fixed component gate | Measurement | Limit | Outcome |
| --- | ---: | ---: | --- |
| Median relative L2,384 existing token/layer states | 2.232303e-7 | <=1e-4 | Pass |
| Worst relative L2 | 4.342481e-7 | <=5e-4 | Pass |
| Median24-layer common ms/token,six CPU threads | **3.260796** | **<=10** | **Pass** |

[Raw result](meth224_common_native_feasibility_result.json), SHA256
`bf99a82180e1757479b2eeb6d68ce432bee8ae499b88083b738fabe0d89c16fa`,
binds every matrix/control segment, source/binary/C/executable/output
and numerical row. Runtime7.015 s after imports, ending RSS1.721 GB,
GPU peak18.2 MB, local RTX3060/six threads. Session71327 exited0;
no T4 or project inference remains active.

**Decision:** freeze actual output-function distillation of the nonlinear
common. This pass shows cost/precision of this fixture operator, not
preserved donor quality, trained compact knowledge, useful E16/E160,
accepted full rate or generality. A trained common must bind its stored
BF16 coefficients and FP32 bias to C parity; conditional recovery,
multi-layer causal composition and independent LLM quality remain required.
