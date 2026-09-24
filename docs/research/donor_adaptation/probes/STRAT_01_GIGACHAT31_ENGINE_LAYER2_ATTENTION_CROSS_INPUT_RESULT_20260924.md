# STRAT-01 GigaChat 3.1 layer-2 attention cross-input result

**Verdict:** `LAYER2_KV_RESIDUAL_SUFFICIENT`

The exact-reference layer-2 causal-attention plus V-B operator is byte-exact.
The admitted query residual alone remains inside the frozen gate, while the
admitted KV residual alone is sufficient to reproduce the `kqv_out-2`
failure. Do not modify the attention/V-B operator or query side.

## Frozen-arm result

All arms are compared with the immutable reference `kqv_out-2` target under
the unchanged limits, NRMSE `<= 0.002` and normalized maximum `<= 0.01`.

| arm | NRMSE | normalized maximum | result |
|---|---:|---:|---|
| C query / C KV | `0.003014631769821054` | `0.004362653758957389` | FAIL; exact native replay |
| reference query / reference KV | `0` | `0` | **PASS; byte-exact** |
| C query / reference KV | `0.001491756896127374` | `0.002120913465841993` | **PASS** |
| reference query / C KV | `0.0027922348709553107` | `0.004362653758957389` | **FAIL** |

The KV-only arm is concentrated most strongly at token 6, NRMSE
`0.005812918127567518`, then token 5 at `0.002597218214464825` and token 7
at `0.0022262637340280545`. The query-only arm has one token-local excursion
at token 5 (`0.003403996177142727`) but its preregistered aggregate passes.

## Identity and controls

- one diagnostic invocation, zero donor graphs, and zero reference graphs;
- accepted artifact SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- `c_q__c_kv` is byte-exact to the immutable C target, SHA-256
  `47c8b7db2e7a7a01886033870c0806987f804a0ac5ec01e5e8b1db8a285a2ae2`;
- the exact-reference output is byte-exact to the immutable reference target,
  SHA-256
  `d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e`;
- all prefill/cached input and target twins are byte-exact;
- both V/K-prefix relations are byte-exact;
- all six one-byte input mutations and the mixed-label swap are rejected;
- query-negation and KV-row-swap controls reject at NRMSE
  `0.7645511253541801` and `1.624031225434044`;
- the consumed V-B matrix is `blk.2.attn_v_b.weight`, Q4_K
  `[512,192,32]`, file offset `595537024`, span `1769472` bytes;
- frozen C/reference metrics reproduce within `1e-12`.

## Canonical evidence

The scientific invocation ran from commit
`a2ffefa49c0a59d4ec31ce1f8802b5c7c40fb79e`.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_attention_cross_input_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `266eba9376ea338db89b6caf9cb86b956d88d25c48e517b9c1102874b61575c9` |
| C binary | `17c25794f94cb16d16e4ac6e20424d301e1247fd253e53e091ee2fa6c88defb3` |
| C-query/reference-KV output | `ff682f402cb3e74f816585861d89f4fe601784a80ebb0c4afd3ded296e05c87c` |
| reference-query/C-KV output | `a127b01e4c8f06a7e9a37dcc4dabe307a7a8733c6a6a84c8810d81125a884a92` |

## Consequence and next coordinate

The operator and query-side hypotheses are closed. The 576-value compact K
row is the remaining causal coordinate: its first 512 values are also the
latent V view, while its final 64 values are the positional/RoPE key tail.
Freeze a zero-graph cross-input diagnostic that holds reference query fixed
and independently substitutes those two KV partitions. This distinguishes
latent/value residual, positional-tail residual, and their interaction.

Do not rerun this cell, the layer-2 producer, or any predecessor. This result
does not repair layer 2 and makes no claim about output projection, FFN/MoE,
later layers, tokenizer, logits, generation, quality, RAM, or rate.
