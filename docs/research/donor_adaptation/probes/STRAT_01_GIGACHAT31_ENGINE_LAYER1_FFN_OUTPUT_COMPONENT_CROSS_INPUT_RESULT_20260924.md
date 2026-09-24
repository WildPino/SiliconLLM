# STRAT-01 GigaChat 3.1 layer-1 FFN-output component cross-input result

**Canonical verdict:** `LAYER1_ROUTED_MOE_OUTPUT_RESIDUAL_SUFFICIENT`

The sole scientific diagnostic authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_OUTPUT_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md)
completed from qualified producer commit
`c3c4d09d6197768b76fadec941881c9bb1031602`. It reports no error, exactly
one diagnostic invocation, and zero donor/reference graph executions.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_ffn_output_component_cross_input_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `4d03a081ab06cf695c4569c4d8ab615d2e2b88a2bfc520874da89e32685e984a` |
| C binary | `53a6af02ef819b436f5fbfca886bc884e4f5061bab65790a933983cb11fb8baa` |

The complete orchestration took `39.857` seconds. This is diagnostic cost,
not emitted-token throughput.

## Scientific outcome

Replacing only the routed MoE contribution with the captured C value exactly
reproduces the predecessor failing downstream hash. Replacing only the shared
expert contribution passes by more than four orders of magnitude at the NRMSE
gate.

| arm | downstream NRMSE | normalized maximum | result |
|---|---:|---:|---|
| captured reference FFN output | `0` | `0` | PASS |
| captured C FFN output | `0.002642086799405445` | `0.004687597394884091` | **FAIL** |
| computed reference routed + reference shared | `0` | `0` | PASS, byte-exact replay |
| computed C routed + C shared | `0.002642086799405445` | `0.004687597394884091` | **FAIL**, byte-exact replay |
| C routed + reference shared | `0.002642086799405445` | `0.004687597394884091` | **FAIL** |
| reference routed + C shared | `1.467939211440747e-7` | `4.900371526567658e-7` | PASS |

The decisive C-routed arm's downstream SHA-256 is
`81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`,
exactly the predecessor failure. Its candidate `ffn_out-1` SHA-256 is
`2d0b1e880f91423cfe59444ba89972e8943a3f1d61a8f50bdb89b2fe1a57ee1c`.
The inverse passing arm has downstream SHA-256
`b6d6d8b2e43ffa14637db5c767b08f9db250977bb6c1e6d4456e88a0f3483e96`.

Direct C-versus-reference component errors remain descriptive rather than
decision gates:

| component | NRMSE | normalized maximum |
|---|---:|---:|
| routed `ffn_moe_out-1` | `1.3547864451924633e-5` | `1.1138183142366114e-5` |
| shared `ffn_shexp-1` | `1.3587123993933162e-7` | `1.993116572604852e-7` |
| combined `ffn_out-1` | `1.247051153528305e-5` | `1.101912398768209e-5` |

The small routed residual is therefore causally sufficient only after the
already closed layer-2 amplifier; its local size did not license closing it.

## Replay and controls

- captured reference and C anchors are byte-exact;
- independently computed reference/reference and C/C arms replay all six
  downstream stages byte-exactly;
- prefill and cached-composition twins are byte-exact;
- one-byte mutations of all ten admitted payloads are refused;
- routed/shared arm-label swapping is rejected;
- negating the reference routed output at token 7 rejects at NRMSE
  `0.20962964201860176`;
- all apparatus self-tests and Python tests passed before the diagnostic.

## Interpretation and stop rule

Close the shared-expert output, the FFN terminal addition, and every layer-2
operator on exact inputs. Do not rerun either producer or this component split.
The only non-duplicate successor is the separately frozen
[routed-MoE component cross-input](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_MOE_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md),
which crosses immutable expert-down outputs with normalized router weights,
then uses the same ordered slot sum and closed downstream path. It executes no
donor/reference graph.

No repair, later-layer, tokenizer, logits, generation, task quality, RAM, or
rate claim follows. No `SPEED_LEDGER.md` update is due.
