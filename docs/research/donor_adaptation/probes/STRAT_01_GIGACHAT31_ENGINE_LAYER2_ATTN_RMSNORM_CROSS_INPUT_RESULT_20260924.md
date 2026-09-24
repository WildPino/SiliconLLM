# STRAT-01 GigaChat 3.1 layer-2 attention RMSNorm cross-input result

**Canonical verdict:** `LAYER1_TERMINAL_RESIDUAL_SUFFICIENT`

Production layer-2 attention RMSNorm is exact on immutable reference
`l_out-1`. The accepted C `l_out-1` residual alone reproduces every downstream
byte and the registered failure. Close layer-2 attention RMSNorm, KV-A,
compressed-KV RMSNorm, partition, attention, V-B, and query.

## Deciding measurement

| arm | normalized output | KV-A projection | compressed prefix | downstream versus reference |
|---|---|---|---|---|
| captured reference norm | reference hash exact | reference hash exact | reference hash exact | exact; PASS |
| RMSNorm(reference `l_out-1`) | reference hash exact | reference hash exact | reference hash exact | exact; PASS |
| captured C norm | C hash exact | C hash exact | C hash exact | NRMSE `0.002642086799405445`; FAIL |
| RMSNorm(C `l_out-1`) | C hash exact | C hash exact | C hash exact | NRMSE `0.002642086799405445`; FAIL |

The normalized-maximum error of both C arms is
`0.004687597394884091`. Their per-token downstream NRMSE is zero for tokens
0--4, `0.002280978866229438` at token 5,
`0.005616336995401208` at token 6, and `0.0019956561137959386` at token 7.
The registered downstream gates were NRMSE `<=0.002` and normalized maximum
`<=0.01`.

RMSNorm itself is not the source: computed reference input is byte-identical
to captured reference `attn_norm-2`, and computed C input is byte-identical to
captured C `attn_norm-2`. Descriptively, the latter differs from reference by
only NRMSE `8.798445530899196e-06`, concentrated at token 5, before the closed
layer-2 path amplifies it.

## Controls and provenance

- all computed-C stages replay their captured C counterparts byte-exactly;
- captured reference/C anchors and both schedule twins are byte-exact;
- all seven mutated-input identity controls reject;
- the arm-label swap rejects;
- token-7 input negation rejects at downstream NRMSE `0.21760655152808162`;
- norm-weight negation rejects at downstream NRMSE `2.109615221511593`;
- one diagnostic invocation; zero donor and zero reference graphs;
- accepted artifact SHA-256:
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- committed source HEAD:
  `161da2983389c7ec4498f3d880c72be07251c29d`.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_attn_rmsnorm_cross_input_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `a42540afb96a70dfb411092d3f416cbbf840c83479302ac2eb1fd8bc9ad5d430` |
| scientific binary | `12512a3e9a69e8f2d884f0264fea25c93b7d82e72c193700955f2b53f7221ed7` |

## Consequence and next boundary

Do not repeat this cell or repair layer-2 RMSNorm/downstream operators. The
causal boundary moves upstream into the already captured terminal layer-1
sum `l_out-1 = ffn_inp-1 + ffn_out-1`.

Frozen-payload arithmetic is descriptive but strongly localizing: C versus
reference `ffn_inp-1` is NRMSE `5.400853707562925e-08`, whereas
`ffn_out-1` is `1.247051153528305e-05`. Reference/reference and C/C sums
reproduce their captured `l_out-1` bytes exactly. Reference FFN input plus C
FFN output has NRMSE `9.310916242406284e-06`; C FFN input plus reference FFN
output has only `3.595353521346349e-08`.

That arithmetic does not decide downstream sufficiency. The separately frozen
[terminal-component protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_TERMINAL_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md)
must propagate those immutable cross-sums through the closed layer-2 path with
zero new graphs. No tokenizer, later-layer, logits, generation, quality, RAM,
or rate claim follows.
