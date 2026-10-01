# METH-251: matched residual space does not yield useful count gain

**Decision: close this fixed matched residual-basis hierarchy.** Frozen
556148f, session84313 exits0. All source/route/reference,16 anchor/160 child
solver, mean, growth, snapshot and176 distinct-coefficient controls pass.
No apparatus repair or repeated fit/validation invocation was needed.

| SSE / target energy | Original reference | Matched E16 | E160 | Rotated E160 |
|---|---:|---:|---:|---:|
| Fit | .00004587007460894334 | .00004077703867252209 | .00003887617053338055 | Not scored |
| Consumed validation | .00011764126288971341 | .0001171985882763571 | .00011741261115189368 | .00011771319929170811 |

The newly adapted E16 reduces consumed original-reference error0.3763%.
E160 improves matched E16 fit4.6616% but **loses0.1826%** on consumed
validation. Its rotated advantage0.2554% is also below10%. Absolute1%
and both source-prior10% comparisons pass; matched count/rotated10% and
positive bootstrap gates fail. Paired10,000-window-bootstrap gain P05
-2.443680451960695e-7, P95 -1.832674096378143e-7, seed251252.

Both arms use the same extra rank32 output directions. E160 retains the
whole original and newly adapted parent, then adds only a private mapping
in that same space. Original per-window FP32 parent/prior controls replay
exactly. Only zero-support156 uses qualified smooth source sensitivity
against the complete matched parent; not a fitted slope/BF16 derivative.

Physical FP64 checkpoint313,456,588bytes SHA256
`6908e582dca2e5fcfa17c893ccbe2a5d0ba8ffa2e09ab4485f96148f9330d9bf`.
[Raw result](meth251_residual_basis_child_result.json) SHA256
`5419b7bdea27a4470688996a5e041d8e11b90e1dfeac3dc5c87dfd4567c7ff84`.
Runtime after imports93.047s, RSS3,442,245,632bytes, GPU peak3,547,291,648;
local RTX3060/six threads, no T4/download/native timing. Source windows
are consumed development data, not independent quality or full-model tasks.

The output-space bottleneck diagnosed by METH-250 is real at fit but does
not license useful count generalization. Close both inherited-left and
this matched residual-left recipe; no rank/tau/codec count retry. A changed
private nonlinear feature mechanism is proposed separately. No encoded
bank, large-RAM routing/DRAM, full quality/rate or donor-scale transfer.
