# STRAT-01 layer-1 FFN RMSNorm cross-input — apparatus qualification

**Status: `APPARATUS_READY_NO_DONOR_EXECUTION`.** This is a mechanical readiness
record only. It makes no scientific, model-quality, throughput, or causal claim.

## Canonical repair4

The sole qualifying raw record is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_ffn_rmsnorm_cross_input_apparatus_repair4_20260924`
(unversioned directory). It reports no errors and binds:

- adjudication SHA-256: `f4f6b4121d3a0cf300dc459821bc2e5893e796fbfa8585d38a9bad25df001337`
- binary SHA-256: `1015a0154b10834911a94556fbee58153b9443d998111ed998363300844f09f1`
- observed source HEAD: `7a616ab0006834aa74210e1e0b1f9eb8150c0acf`
- expected artifact: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
  `opened=false`, observed bytes and SHA-256 are null.

All artifact/payload-dependent counters were zero: diagnostic, RMSNorm,
up-projection, propagation, donor graph, and reference graph. Clang compilation,
the new C self-test, routed-up, SwiGLU, Q6, propagation, legacy, and Python
checks all returned rc 0. The new C self-test passed 6 checks and the Python
suite passed 6 tests.

Source bindings:

| Source | SHA-256 |
|---|---|
| runner `run_strat01_layer1_ffn_rmsnorm_cross_input.py` | `df87ae2d51b812d3b845b82b2edd141765021cbaee4450b23580251eb48be0e8` |
| tests `test_strat01_layer1_ffn_rmsnorm_cross_input.py` | `b38197adb5d9e0d3be88f193ed0d8cee63d798634fbaec5164401cbbf1dbe577` |
| frozen protocol | `868c23d1758ebe374790217822f7cdd7f8a89dd21029929af3afd98ad47c37fc` |
| `engine.c` | `120738e56d1ebfe43473c0ce15f8c16e865c03ded81e394b162f06f62cde0c09` |
| apparatus header | `3c1f11cf579a01a87fa53ef050cdf730216baad56d9e61d9ede83fceedb77ee6` |
| base runner | `3a170204ad5b08716bce29b9b8e5f7046bdc8b371d7e1f969d675fa75c5edafd` |

The apparatus binds predecessor adjudication
`2de61cdc02deeb9639044cb75cc614705a41eb3e133b00f3ad2746b80bc7c71d` and the immediate routed-up result tensor
`diagnostic/current_up_on_c_norm.downstream.f32le`, SHA-256
`81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`.
The predecessor result is
[`LAYER1_ROUTED_UPSTREAM_NORM_RESIDUAL_SUFFICIENT`](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_UP_PROJECTION_CROSS_INPUT_RESULT_20260924.md).

## Superseded mechanical records

The initial apparatus used an older byte-identical SwiGLU path as the C replay
anchor. Repair1 fixed that provenance but enforced aggregate gates only.
Repair2 added per-token gate enforcement and synthetic dilution rejection,
but its adjudicator consumed an immutable reference downstream target absent
from its evidence set, allowing a scientific `KeyError` after computation.
Repair3 was superseded because its unit test called `evidence()`, which opened
frozen scientific payloads during apparatus-only despite zero counters.
Repair4 replaces that call with static source-contract assertions, so
apparatus-only opens neither model nor scientific payload. It retains the
target-evidence binding, base-runner binding, and aggregate plus per-token
gates. No scientific operator, arm, or payload changed. Initial and repair1,
repair2, and repair3 records are superseded; only repair4 authorizes execution.

## Authorized next step

Repair4 is the sole apparatus authorization. After an exact-source commit of
the qualified implementation and protocol, perform one scientific,
zero-graph invocation. The frozen question remains whether the residual enters
through immutable `ffn_inp-1` or production RMSNorm; apparatus readiness itself
does not decide that question. Do not infer a scientific result from this
record.
