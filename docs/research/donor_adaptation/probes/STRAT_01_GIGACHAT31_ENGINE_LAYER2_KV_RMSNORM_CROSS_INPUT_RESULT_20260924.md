# STRAT-01 GigaChat 3.1 layer-2 KV RMSNorm cross-input result

**Canonical verdict:** `LAYER2_PROJECTED_KV_PREFIX_RESIDUAL_SUFFICIENT_RECOVERED`

The production layer-2 KV RMSNorm is byte-exact on both immutable reference
and C inputs. Exact reference projected input produces the exact reference
normalized prefix and downstream output; C projected input reproduces the
captured C normalized prefix and its failing downstream output. The
sufficient residual therefore already exists in the first 512 values of
`kv_cmpr_pe-2`, before KV RMSNorm.

## Scientific arms

| arm | normalized-prefix NRMSE | downstream NRMSE | result |
|---|---:|---:|---|
| captured reference | `0` | `0` | PASS; byte-exact |
| captured C | `0.0006316985540415843` | `0.002642086799405445` | downstream FAIL |
| RMSNorm(reference projected prefix) | `0` | `0` | **PASS; byte-exact** |
| RMSNorm(C projected prefix) | `0.0006316985540415843` | `0.002642086799405445` | **FAIL; byte-exact C replay** |

The downstream normalized-maximum values for reference and C are `0` and
`0.004687597394884091`. The C prefix itself remains inside the general gate;
the already accepted attention path amplifies it past the downstream NRMSE
limit. That descriptive prefix pass does not override the preregistered
downstream decision.

## Controls and provenance

- input-negation control: NRMSE `0.21760655152808162`;
- norm-weight-negation control: NRMSE `2.109615221511593`;
- every exact replay, frozen metric, schedule twin, input mutation, descriptor,
  and label-swap check passes;
- raw producer: one diagnostic invocation, zero donor/reference graphs;
- recovery: zero new invocations, model unopened, zero graphs;
- raw commit `a1ee4f61f8b10f763108f4e37830e642f7b859f6`;
- recovery commit `1447f6d135a185a21313b513c7b0e746ab780c0f`.

The raw record remains VOID at SHA-256
`2c628c3a9503cf882c2e6ce42a6c8211a5767b6269dc0694b148623da383aaaf`.
The canonical recovery record is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_kv_rmsnorm_cross_input_offline_recovery1_20260924/adjudication.json`

SHA-256:
`e4f4fb67e73e7089fee3f7900c768894b9c9bf0399b5c978169c06f7b07ca976`.

## Consequence and next coordinate

Close KV RMSNorm, normalized-prefix, positional-tail, query, attention/V-B,
schedule, and all predecessor axes. `kv_cmpr_pe-2` is the output of
`blk.2.attn_kv_a_mqa.weight` applied to `attn_norm-2`; immutable reference/C
payloads exist on both sides. The only non-duplicate successor is a zero-graph
cross-input test of exact reference versus C `attn_norm-2` through the
production layer-2 KV-A Q4_K projection, followed by the now-closed RMSNorm
and downstream path.

This result does not repair layer 2 or make claims about the KV-A projection,
earlier layer-2 input/RMSNorm, output projection, FFN/MoE, later layers,
tokenizer, generation, quality, RAM, or rate.
