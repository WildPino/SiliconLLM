# STRAT-01 FFN SwiGLU cross-input diagnostic — VOID 1

**Status:** `VOID_FFN_SWIGLU_CROSS_INPUT`

**Implementation commit:** `2581da54a0363c3644d0b3459fd2b43fb7f27d4b`

The first invocation of the frozen
[FFN SwiGLU protocol](STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md)
is an apparatus VOID. It supplies no gate/up/operator verdict and none of its
arm metrics may be mined or cited as scientific evidence.

The immutable raw directory is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_ffn_swiglu_cross_input_20260923/`

Its adjudication and run-manifest SHA-256 values are
`57e7c3880c66f018ca1016e63c70740044b1d8f628e11e5c1c960d9490257753`
and `f42972fbd249d9fa226dd4e368f9c407d7bcb7c948d5e34343c80e9ad17f3589`.

## Failure

The diagnostic completed all five arms and both controls, then refused before
emitting an adjudicable report:

`SwiGLU downstream replay identity mismatch`

The refusal came from an extra C-side assertion requiring the terminal sum of
captured reference `ffn_swiglu-0` passed through **current Q6** to equal the
terminal sum made from captured reference `ffn_out-0`. That equality is not a
frozen protocol requirement and is known not to be byte-exact: the preceding
FFN-down result accepted current Q6 on exact SwiGLU by numerical gates, not by
identity with the captured reference output.

The generated `reference_captured_control_ffn_out-0.f32le` has SHA-256
`f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce`,
exactly matching the preceding diagnostic's
`reference_swiglu_current_q6_ffn_out-0.f32le`. Thus the arm replayed the
intended prior current-Q6 arm; the new assertion compared it with the wrong
object. The C/C SwiGLU and down output also retained their frozen identities,
including C SwiGLU SHA-256
`e4073a39ca0d6f904dab90b22f8fac9c9516c219b265611a102da8dd3597bc8d`.

No donor or reference graph was executed. The accepted GGUF was opened only
by the zero-producer diagnostic path. The failed report self-identifies as
`DIAGNOSTIC_FAILURE`; external adjudication did not run.

## Narrow repair authorization

One repaired invocation is owed after a fresh apparatus-only qualification.
The repair may only:

1. remove the foreign byte-identity assertion against the captured-reference
   terminal sum; and
2. add a regression that prevents its reintroduction.

The existing external adjudicator already requires the correct identity:
every captured-reference/current-Q6 checkpoint must match the preceding
`reference_swiglu_current_q6` arm. C/C byte replay, source/input hashes,
controls, thresholds, arm order, and the frozen decision rule must remain
unchanged. The VOID directory must not be overwritten or offline-adjudicated.
