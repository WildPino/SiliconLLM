# STRAT-01 FFN down-projection cross-input result

**Verdict:** `SWIGLU_INPUT_RESIDUAL_SUFFICIENT`

The single invocation authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_FFN_DOWN_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md)
completed at implementation commit
`68392bd3590aacf663c9dedc35b5701b199f6da8`. It executed zero donor graphs
and zero reference graphs. The raw record is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_ffn_down_cross_input_20260923/`

The canonical adjudication and run-manifest SHA-256 values are
`6d3ab2b6a1097502060380350238940d434d38b2b41bcdff8d7c586f5f8d05d0`
and `fbe5857f338acbd10e8ffa703746e02a2c084533ba9148fc2a5cfd4e0d02645f`.
The run completed in 43.291 seconds and reports no error.

## Direct and downstream outcome

The unchanged acceptance limits are NRMSE `<= 0.002` and normalized maximum
`<= 0.01`.

| arm | direct `ffn_out-0` NRMSE | direct max | `kqv_out-1` NRMSE | downstream max | first failure |
|---|---:|---:|---:|---:|---|
| captured reference `ffn_out-0` | `0` | `0` | `0.0003855998766858119` | `0.0011237641335101218` | none |
| reference `ffn_swiglu-0` through current Q6 | `8.348319997561636e-8` | `7.777310102196965e-8` | `0.0003855998766858119` | `0.0011237641335101218` | none |
| C `ffn_swiglu-0` through current Q6 | `0.0009913516476391816` | `0.0010415762554367286` | `0.0061366360889986695` | `0.0030491102772673647` | `kqv_out-1` |

Exact captured reference SwiGLU input makes the current production Q6_K×Q8_K
operator pass the direct gate by more than four orders of magnitude in NRMSE.
After reference residual addition, all eleven layer-1 checkpoints pass and the
terminal metrics equal the captured-reference control. The C SwiGLU arm
byte-replays frozen C `ffn_out-0`, the preceding hybrid terminal sum, and all
eleven propagated outputs; only `kqv_out-1` fails.

## Replay and controls

- Captured-reference control replays all eleven prior outputs exactly.
- C-input Q6 output is byte-exact to frozen C `ffn_out-0`.
- Its terminal sum and all eleven propagated outputs are byte-exact to the
  preceding reference-attention/C-FFN arm; all metrics reproduce within
  `1e-12`.
- One-byte mutations of all five admitted payloads are refused.
- Input-origin relabeling is rejected.
- Negated reference SwiGLU rejects directly at NRMSE `1.9999999992334967`
  and downstream at `1.8346344481325492`.
- Reference SwiGLU row swap rejects directly at NRMSE
  `0.5874851647841498` and downstream at `0.9675771907226235`.
- New C self-test: 7 checks PASS; terminal-component self-test: 9 checks
  PASS; new Python tests: 5 PASS; 73,024 legacy checks PASS with zero worst
  error; compiler stderr is empty.

## Interpretation and next boundary

The current production Q6_K down-projection path is sufficient on exact
captured `ffn_swiglu-0` input and must not be rewritten or remeasured on this
cell. The sufficient residual is already present in captured C
`ffn_swiglu-0`.

The next distinct zero-producer boundary is the SwiGLU construction
`SiLU(ffn_gate-0) * ffn_up-0`. A frozen successor should cross captured
C/reference gate and up payloads, include a current-operator
reference/reference arm to test the C `expf`/multiply semantics, require a C/C
byte replay, and propagate every generated SwiGLU through the now-closed Q6
helper plus exact reference residual and layer 1. This can distinguish an
operator residual, gate residual, up residual, independent sufficiency, or a
joint-only interaction without reopening Q6 or layer 1.

## Non-claims

This result attributes sufficiency to the captured SwiGLU input at the tested
boundary. It does not yet identify gate, up, SiLU/multiply semantics, or their
interaction. It makes no claim about Rung-2C repair, MoE, later layers,
tokenizer, logits, generation, quality, RAM, or rate. No `SPEED_LEDGER.md`
update is due.
