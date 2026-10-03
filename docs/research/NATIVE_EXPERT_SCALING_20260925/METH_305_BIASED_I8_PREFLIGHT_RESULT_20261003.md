# METH-305: exact integer kernel passes fidelity, fails native cost

**Stable cost FAIL before collection/training.** Freeze16894b0, terminalexit0,
5.312s controller/2.994s native. The new biased-byte/two-part activation
representation reproduces EVERY30 original303 output/route hash and every
old selftest field exactly, including real source Q6/F32 router/norms.

All65,025 signed weight/input combinations and5,929 mixed adjacent pairs
pass scalarI64/high/low pair controls. Direct unsplit saturation and row-bias
corruption faults detected. Original7,176 exact scalar/scaled rows,97,194
bank edges, I8 decoded relativeL2 3.630782203605679e-8, actual Q6 head64-row
1.742025356166587e-7,25-layer macro/child selection and packed-head/bad-bank
faults remain exact303. F32 quantization/whole fixture composition unchanged.

Complete weight descriptor447.358MB passes560MB; physically populated
9.437B routed coefficients/640 synthetic functions per layer remain. Native
medians **21.6527/22.0771/21.6325ms FAIL14ms**, repeatability1.0205524PASS.
PeakRSS9,955,139,584B includes extra split buffers; selected-child union37–40.
No donor weights trained/quality observed, no useful extra n or accepted rate.
Separate compiled runs do not establish a causal speedup versus303/304.

The exact signed-I8 byte-kernel path tested here stays closed before data
collection/training. Next cost candidate changes storage/precision to packed
I4 with row-tiled computation, reducing active coefficient bytes while retaining
the full source Q6 head and complete resident bank. This requires NEW numeric
controls and later source-quality evidence: exact303 hashes cannot be required
after a real precision change. Freeze its representation/scheduling/cost gates
before any run; no training until credible cost. Do not relax14ms or shrink n.

## Bindings and independent recount

- [raw result](meth305_biased_i8_preflight_result.json), SHA256
  `49b0b5b8d26abf11c914ab583262faf389544dceff8d674577aadb00adfd5a32`.
- C `c4f4152d91102e39cb849046b3ffea5521b4aa2f01f4d9d5165b976735c98e77`;
  controller `46e9b3985325fe382f28d94364b539ac6d90853da7a7f70190b787f5ec4c0597`;
  phase60 `71ed122ae16ee4b4e55dc2bdec417f46b8d8cd1fd78c391a8c41a6e2c4543c94`.
- Executable `ee9c8d47c9acb7d6fb1b50741935799fead68119fa56b4e403916295b6293f9c`;
  spec/source component bindings exact303. Intrinsic header
  `eb2a35ae03975e937db2be1c6f162c7172bf8e782cfe66296129585e7e619853`.
- Stdout `results/native_expert_scaling/meth305_biased_i8/meth303_native_run1.jsonl`,
  SHA `81d8838d689fb0f866ab49b373dcd31054ee7210478e1f68f4d8da02ddf60986`;
  stderr empty. Compile warning records the intentionally impossible comparison
  between saturatedI16 and64770; compilation succeeds without fast-math.

Independent recomputation of all30 prior hashes, old selftest fields, three
medians and spec/log bindings PASS. Frozen source/controller/engine bytes
matched filtered HEAD before execution. Preserve old sources/results.
Final donor-relative quality/useful RAM-scale n/SAMEartifact accepted50 and
multiple-family/scale requirements remain open.
