# STRAT-01 GigaChat 3.1 layer-2 KV-A projection cross-input result

**Verdict:** `LAYER2_ATTN_NORM_INPUT_RESIDUAL_SUFFICIENT`

The production layer-2 KV-A Q4_K projection passes on exact reference
`attn_norm-2`. Feeding the captured C `attn_norm-2` reproduces the captured C
projection, normalized prefix, and downstream failure byte-for-byte. Close the
KV-A projection and every downstream operator; the sufficient residual already
exists at its input.

## Bound execution

- Git HEAD: `c61cd5396fba611d323876786811861208dcd381`;
- accepted GGUF SHA-256:
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- one diagnostic invocation;
- donor/reference graph executions: `0 / 0`;
- errors: none;
- raw directory:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_kv_a_projection_cross_input_repair1_20260924/`;
- `adjudication.json` SHA-256:
  `d16af478a412e22457f11e4d0946e8c7d9fec1c842dfee9657fb213ec8cc06d1`;
- C binary SHA-256:
  `0cf42a7ae5124fac96deb0f5c2806abcc9eca352274b34bba7ba9f1db7bd63c7`.

The earlier raw VOID bound in protocol Addendum A executed no diagnostic arm
and contributes no scientific value.

## Deciding arms

| arm | projection NRMSE | downstream NRMSE | downstream normalized max | gate |
|---|---:|---:|---:|---|
| captured reference projection | `0` | `0` | `0` | pass |
| production projection(reference `attn_norm-2`) | `0` | `0` | `0` | pass |
| captured C projection | `0.0001445203582` | `0.0026420867994` | `0.0046875973949` | fail |
| production projection(C `attn_norm-2`) | `0.0001445203582` | `0.0026420867994` | `0.0046875973949` | fail |

The direct projection difference remains below the descriptive `0.002` NRMSE
line, but the closed downstream path amplifies it beyond the adjudication line.
The downstream discrepancy is confined to tokens 5–7; token 6 reaches NRMSE
`0.0056163369954`.

Exact identities establish causality:

- computed reference projection / normalized prefix / downstream SHA-256:
  `81c439e40a9dfd06221a86b7cdc9b6019ea8742e6cf11371d1d2c50546346c61`,
  `0486a48dcd7289236760cc5238d31f78c6035fc5001942eb38302dd1ac4abb91`,
  `d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e`;
- computed C projection / normalized prefix / downstream SHA-256:
  `aae2f54a6391b84c156c3b61f3206be1170afb7e09995cc891f1fd9e7ebfc444`,
  `e35f17903938047d1f699a92a7e0a85c6abb691a834ffc1cfe363429620372d9`,
  `81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`.

## Controls and scope

Both planted controls reject: mutated token-7 input gives downstream NRMSE
`0.2176065515`, and negating the projected prefix gives `2.1096152215`.
Captured anchors, computed-C replay, schedule twins, seven payload mutations,
and label-swap rejection all pass.

Do not repeat KV-A, compressed-KV RMSNorm, partition, attention, V-B, query, or
any earlier layer-2 cell. The only new fidelity boundary is immutable
`l_out-1` through production `blk.2.attn_norm.weight`, followed by the now
closed KV-A/downstream path. This result does not repair layer 2 or establish
later-layer fidelity, tokenizer/generation quality, RAM, or rate.
