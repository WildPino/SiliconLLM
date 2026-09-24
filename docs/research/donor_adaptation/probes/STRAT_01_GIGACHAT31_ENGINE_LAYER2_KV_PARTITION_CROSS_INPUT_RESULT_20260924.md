# STRAT-01 GigaChat 3.1 layer-2 KV-partition cross-input result

**Verdict:** `LAYER2_LATENT_VALUE_RESIDUAL_SUFFICIENT`

With reference query fixed, substituting only the C first-512 compact-KV
prefix is sufficient to fail the unchanged `kqv_out-2` gate. Substituting
only the C final-64 positional/RoPE tail passes. The sufficient residual is
therefore in the compressed latent values, which are also the attention value
view; the positional tail is not independently sufficient.

## Frozen-arm result

All arms use the immutable reference query and are compared with the immutable
reference `kqv_out-2` target under NRMSE `<= 0.002` and normalized maximum
`<= 0.01`.

| latent/value prefix 512 | positional tail 64 | NRMSE | normalized maximum | result |
|---|---|---:|---:|---|
| reference | reference | `0` | `0` | **PASS; byte-exact** |
| C | C | `0.0027922348709553107` | `0.004362653758957389` | FAIL; byte-exact predecessor replay |
| C | reference | `0.002642086799405445` | `0.004687597394884091` | **FAIL** |
| reference | C | `0.001504218399168285` | `0.0019335559278095562` | **PASS** |

The isolated latent/value failure is strongest at token 6, NRMSE
`0.005616336995401208`, followed by token 5 at `0.002280978866229438`.
Token 7 remains just inside its descriptive NRMSE boundary at
`0.0019956561137959386`; only the preregistered aggregate gate adjudicates.

## Identity and controls

- one boundary invocation, zero donor graphs, and zero reference graphs;
- accepted artifact SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- exact-reference output is byte-exact to the reference target, SHA-256
  `d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e`;
- complete-C-K output is byte-exact to the whole-KV predecessor, SHA-256
  `a127b01e4c8f06a7e9a37dcc4dabe307a7a8733c6a6a84c8810d81125a884a92`;
- all prefill/cached twins and both V/K-prefix identities are byte-exact;
- all six one-byte input mutations, the mixed-label swap, and the altered
  partition boundary are rejected;
- latent-negation and positional-row-swap controls reject at NRMSE
  `0.21760655152808162` and `1.2795178719501532`;
- the matrix is `blk.2.attn_v_b.weight`, Q4_K `[512,192,32]`, file offset
  `595537024`, span `1769472` bytes;
- frozen predecessor metrics reproduce within `1e-12`.

## Canonical evidence

The scientific invocation ran from commit
`887b833184d8e977b4c2424708c440eb909f2d87` and completed in 40.84 seconds.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_kv_partition_cross_input_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `c6f59af8cf83f413cee14ae38ed451270531badc1a71095875a5f39680eebe7f` |
| C binary | `e5b186e10d438868b40c8f3af054b6581b5135f415995308bcbdbd7bfc15b668` |
| C-prefix/reference-tail output | `81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254` |
| reference-prefix/C-tail output | `a0cc5b89f87c517c9d42bb8f8301dfeca5fe4a18acf76d982585a0cd164a9c6b` |

## Consequence and next coordinate

Close the positional tail, query, attention/V-B operator, schedule, and
complete-KV axes. The first 512 values are `kv_cmpr-2`, produced by applying
the layer-2 `attn_kv_a_norm` RMSNorm to the first 512 values of
`kv_cmpr_pe-2`. Both producer input and output already exist as immutable
reference/C payloads. The next non-duplicate cell must hold the downstream
accepted attention path fixed and distinguish upstream projected-prefix
residual from RMSNorm-operator residual using only those payloads and
`blk.2.attn_kv_a_norm.weight`; no new graph is justified.

This result does not repair layer 2 and makes no claim about the earlier
KV-A projection, output projection, FFN/MoE, later layers, tokenizer, logits,
generation, quality, RAM, or rate.
