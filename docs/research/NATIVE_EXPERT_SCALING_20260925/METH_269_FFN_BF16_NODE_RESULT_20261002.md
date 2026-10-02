# METH-269: BF16 rounding alone does not repair the input-function tail

Frozen ateb019f8,all6144 fixed consumed125 states,24layers,eight numerical
diagnostic arms,no fit or output artifact. Actual source BF16 equation
matches installed HF MLP bitwise; current equation matches259 stored FFN
bitwise; all values finite. Session73517 exits0,8.438s after imports,
RSS2,352,443,392 bytes,peakGPU200,323,072 bytes. No active job or repair.

| Arm | Energy-normalized squared error vs actual BF16 source | Mean per-state relative squared error |
| --- | --- | --- |
| Exact source,internal FP32 |.00000731114|.00001394705|
| Exact source,FP32/LUT |.00000731264|.00001423550|
| Exact source,BF16 nodes/LUT |.00000002352|.00000446156|
| Existing259 FP32-internal FFN |.00003673980|.00019728979|
| Same259 fields,BF16 nodes/LUT |.00003013608|.00020167504|
| Same259 fields,BF16 nodes/exact SiLU |.00003013004|.00020098384|
|259 readout,oracle source BF16 features |.00002173367|.00003866601|
|259 readout,oracle source FP32 features |.00002910366|.00004825245|

Candidate BF16 nodes lower energy-weighted error17.9743% but increase mean
state error2.2227%; maximum layer error also rises. Exact SiLU in that
rounded candidate yields nearly the same result. Do not promote a node-
rounding/LUT-only remedy from one favorable weighting. The source-only LUT
control is close in energy fidelity, but lower-energy/tail states still
have errors. Small LUT energy error is not a full quality certificate.

Oracle BF16 features with the unchanged readout lower mean state error
80.4014% and energy-weighted error40.8443%. They require full original
gate/up matrices and are not an accepted compact representation. This
identifies a substantial input-function/feature contribution on these
states, while residual readout error remains. It does not prove the
corresponding share of generated semantic errors or that more private
source rows will recover quality. Layers16/11/14 have the largest current
mean-state error; no layer-specific fit or candidate is selected here.

Raw [result](meth269_ffn_bf16_node_assay_result.json), SHA256
`95f905587e74de27b8d7661f9b4a0160af824cdc331c9cf34c8aef3df5b5cc1f`.
These states are reused component data,not the generated prefixes in268;
do not call their error reduction independent full-model preservation.
259 remains closed by267. Next prospectively price a fixed increase in
exact source input rows,keeping source selection/encoded readout/learned
bank/routing fixed,then actual native cost before new full-model quality.
Useful n/RAM/CPU route-LUT-DRAM,same-artifact>=50 acceptedtok/s and multiple
families/scales remain required.
