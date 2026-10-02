# METH-290: causal no-cache GPU reference is stable on the fixed controls

Frozen2c64a6b before observations. Session78446 terminal exit0; measured
PASS all10 guards. Same276 archive/725-field own loader/all76 helper hashes,
six consumed278 prompts/tail8/two arms, no cache BOTH sides.

- Original full-prefix/M8-head repeated96 hidden/logit rows BYTE-exact to285.
- Independent truncated full prefixes give all96 hidden rows BYTE-exact
  to corresponding original full-prefix rows.
- M1-head logit relativeL2 median/max: ablation .0000715562/.0000967954;
  complete .0000663343/.000106425.48/48 top1 in EACH arm; allfinite.
- The isolated head M1-versus-M8 arrays account for the ENTIRE observed
  truncated logit discrepancy; propagated hidden discrepancy is zero.

Thus the observed batch/head geometry cannot explain285's .0542238 native
ablation logit maximum. No-cache reference instability is not supported
on these controls. This is not universal stability, cached/full equivalence,
or native promotion. Original285/288/289 failures and5% limits remain intact.

74.812s total/endRSS2,802,102,272bytes/peakCUDA1,442,444,800bytes; allresource
stops pass. LocalRTX3060/six threads, no competing CPU rate/model job,
no download/T4. [Raw result](meth290_reference_prefix_equivalence_result.json)
SHA `797b8ef913cea4b8eab2754ab37e66c0c75026854eec05a3c0ca9f8bfa2abbcb`
contains all96 rows' error/top1 arrays and code/artifact/reference hashes.
NPZ retains repeated/truncated/head-control tensors. Command/bindings/gates
in [protocol](METH_290_REFERENCE_PREFIX_PROTOCOL_20261002.md).

Next causal localization: hybrid full-prefix execution on consumed source0,
original compact-core-only equation, with CPU operator groups substituted
and one group at a time restored to GPU on its ACTUAL intervened inputs.
Full CPU-group hybrid must first reproduce original285 CPU tail bytes;
otherwise the diagnostic apparatus stops. This oracle is nondeployable and
cannot establish quality/rate/useful new capacity. Native repair still needs
a new prospective equation and unchanged285 gates before complete quality.
